"""Measure how much of the imitation error is caused by an insufficient state.

Equation (6) of the exposé puts the *one-step* forecasts ``mu_hat_{t+1|t}`` and
``vech(Sigma_hat_{t+1|t})`` into the state. The expert of Equation (5) plans over
``H = 21`` days and uses ``mu_hat_{tau+1|t}`` for every ``tau``. Whether that
matters depends entirely on the moment estimator:

- ``regime_mixture`` propagates filtered regime probabilities through the
  transition matrix, so its forecasts genuinely vary across the horizon. The
  state then omits roughly twenty forecast vectors the expert conditioned on,
  and behavioural cloning has an error floor no training removes.
- ``rolling`` uses one window of history, so its forecast is horizon-*constant*.
  The ``tau = 0`` entry in the state is then the whole of what the expert used,
  and the state is a sufficient statistic for its decision.

This script measures both: the horizon variation of each forecaster, and the
behavioural-cloning tracking error achieved against each expert. Everything else
— constraints, cost, risk aversion, horizon, paths, split, architecture, and
optimiser — is held fixed, so the difference is attributable.

Results are reported relative to each configuration's own never-trade baseline,
because the two experts trade at different rates and their absolute tracking
errors are not comparable.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset.demonstrations import DemonstrationDataset  # noqa: E402
from src.dataset.paths import (  # noqa: E402
    PathData,
    PathSpec,
    generate_path_from_spec,
)
from src.dataset.splits import split_paths  # noqa: E402
from src.dataset.state import StateSpec  # noqa: E402
from src.expert.constraints import PortfolioConstraints  # noqa: E402
from src.imitation.bc import (  # noqa: E402
    BCConfig,
    BehaviouralCloningTrainer,
    seed_everything,
)
from src.imitation.comparison import per_path_tracking_errors  # noqa: E402
from src.policy.builder import build_policy  # noqa: E402
from src.policy.reference import default_references  # noqa: E402
from src.simulation.presets import load_simulation_preset  # noqa: E402
from src.utils.observability import (  # noqa: E402
    MemoryTracker,
    configure_logging,
)
from src.utils.parallel import parallel_map  # noqa: E402

logger = logging.getLogger("check_state_sufficiency")

BASE_EXPERT: dict[str, Any] = {
    "horizon": 21,
    "risk_aversion": 5.0,
    "transaction_cost": 0.0025,
    "constraints": {
        "long_only": True,
        "min_weight": 0.0,
        "max_weight": 0.30,
        "leverage_limit": 1.0,
    },
}

MOMENTS = {
    "regime_mixture": {
        "estimator": "regime_mixture",
        "covariance": "ledoit_wolf",
        "window": 60,
        "blend_weight": 0.25,
        "n_regimes": 3,
        "burn_in": 300,
        "refit_every": 250,
    },
    "rolling": {"estimator": "rolling", "covariance": "ledoit_wolf", "window": 60},
}

POLICIES = {
    "mlp": {"kind": "mlp", "hidden_dimensions": [256, 128], "dropout": 0.1},
    "transformer": {
        "kind": "transformer",
        "encoder": {
            "d_model": 64,
            "n_heads": 4,
            "n_layers": 2,
            "feedforward_multiplier": 4,
            "dropout": 0.1,
        },
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/state_sufficiency"))
    parser.add_argument("--n-assets", type=int, default=8)
    parser.add_argument("--periods", type=int, default=1200)
    parser.add_argument("--n-paths", type=int, default=30)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Path-generation worker processes. 1 (default) is a serial loop, "
        "bit-identical to before; >1 fans cache misses across cores; 0 uses every core.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Emit DEBUG-level logging (per-epoch detail); default is INFO.",
    )
    return parser.parse_args()


def horizon_variation(path: PathData) -> dict[str, float]:
    """Measure how far the expert's forecasts move across its planning horizon."""
    expert = path.build_expert()
    midpoint = int(path.decision_times[len(path.decision_times) // 2])
    forecast = expert.forecaster.forecast(midpoint, expert.horizon)

    means, covariances = forecast.means, forecast.covariances
    mean_shift = float(np.abs(means[0] - means[-1]).max())
    covariance_shift = float(np.abs(covariances[0] - covariances[-1]).max())
    return {
        "horizon": int(expert.horizon),
        "mean_absolute_shift": mean_shift,
        "mean_relative_shift": mean_shift / float(np.abs(means[0]).max()),
        "covariance_absolute_shift": covariance_shift,
        "covariance_relative_shift": covariance_shift
        / float(np.abs(covariances[0]).max()),
        "state_is_sufficient": mean_shift == 0.0 and covariance_shift == 0.0,
    }


def run_configuration(
    name: str,
    moments: dict[str, Any],
    *,
    cache_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Generate paths for one expert, train each policy, and report the outcome."""
    tracker = MemoryTracker(logger, phase=name)
    expert_config = BASE_EXPERT | {"moments": moments}
    preset = load_simulation_preset("canonical_three_regime", args.n_assets)
    spec = StateSpec(lookback=20)
    constraints = PortfolioConstraints(**BASE_EXPERT["constraints"])

    cache_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    # Reuse any cached path (serial, cheap); generate only the misses, which may
    # be fanned across workers. Path seeds are base_seed + path_id with no global
    # RNG read, so a parallel-generated path is bit-identical to a cached one.
    paths: dict[int, PathData] = {}
    miss_specs: list[PathSpec] = []
    for path_id in range(args.n_paths):
        file_path = cache_dir / f"path_{path_id:04d}.npz"
        if file_path.exists():
            paths[path_id] = PathData.load(file_path)
            continue
        miss_specs.append(
            PathSpec(
                path_id=path_id,
                seed=args.base_seed + path_id,
                preset=preset,
                periods=args.periods,
                expert_config=expert_config,
            )
        )
    logger.info(
        "%s: %d cached paths, generating %d", name, len(paths), len(miss_specs)
    )

    def _tick(done: int, total: int) -> None:
        if done % 10 == 0 or done == total:
            logger.info("%s paths %d/%d generated", name, done, total)

    for generated in parallel_map(
        generate_path_from_spec,
        miss_specs,
        workers=args.workers,
        progress_callback=_tick,
    ):
        # Save happens in the parent, so no worker writes to the cache dir.
        generated.save(cache_dir / f"path_{generated.path_id:04d}.npz")
        paths[generated.path_id] = generated
    path_seconds = time.perf_counter() - started
    tracker.sample("paths-ready")

    split = split_paths(args.n_paths)
    datasets = {
        part: DemonstrationDataset(
            [paths[path_id] for path_id in split.of(part)], spec=spec
        )
        for part in ("train", "validation", "test")
    }

    references = default_references(
        spec,
        n_assets=args.n_assets,
        n_regimes=3,
        constraints=constraints,
    )
    baseline = float(
        np.mean(
            list(
                per_path_tracking_errors(
                    references["previous_weights"], datasets["test"]
                ).values()
            )
        )
    )

    results: dict[str, Any] = {}
    for policy_name, policy_config in POLICIES.items():
        seed_everything(0)
        policy = build_policy(
            policy_config,
            spec=spec,
            n_assets=args.n_assets,
            n_regimes=3,
            constraints=constraints,
        )
        trainer = BehaviouralCloningTrainer(
            policy,
            BCConfig(
                epochs=args.epochs,
                batch_size=256,
                learning_rate=1e-3,
                weight_decay=1e-2,
                patience=12,
                seed=0,
            ),
        )
        history = trainer.fit(datasets["train"], datasets["validation"], memory_tracker=tracker)
        test = trainer.evaluate(datasets["test"])
        train = trainer.evaluate(datasets["train"])
        results[policy_name] = {
            "test_tracking_error": test["tracking_error"],
            "ratio_to_never_trade": test["tracking_error"] / baseline,
            "train_tracking_error": train["tracking_error"],
            "best_epoch": history.best_epoch,
            "epochs_run": len(history.records),
            "stopped_early": history.stopped_early,
        }
        tracker.sample(f"{policy_name}-fitted")
        logger.info(
            "  %-12s ratio=%.3f testTE=%.4e best_epoch=%d/%d",
            policy_name,
            results[policy_name]["ratio_to_never_trade"],
            test["tracking_error"],
            history.best_epoch,
            len(history.records),
        )

    tracker.summary()
    return {
        "name": name,
        "moments": moments,
        "path_seconds": round(path_seconds, 1),
        "train_samples": len(datasets["train"]),
        "never_trade_tracking_error": baseline,
        "horizon_variation": horizon_variation(paths[0]),
        "policies": results,
    }


def _write_report(report_path: Path, report: dict[str, Any]) -> None:
    """Serialise the report atomically, so a mid-run kill can't corrupt it."""
    tmp = report_path.with_suffix(report_path.suffix + ".tmp")
    tmp.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    os.replace(tmp, report_path)


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "state_sufficiency.json"

    # Write an in-progress report before any work, then re-serialise after each
    # moment config completes. A kill mid-run then leaves valid JSON with the
    # configs that finished — the last run reached config 1 of 2 and produced no
    # file at all.
    report: dict[str, Any] = {"status": "in_progress", "configs": {}}
    _write_report(report_path, report)

    for name, moments in MOMENTS.items():
        logger.info("=== expert moments: %s", name)
        entry = run_configuration(
            name,
            moments,
            cache_dir=args.output_dir / "paths" / name,
            args=args,
        )
        variation = entry["horizon_variation"]
        logger.info(
            "  horizon variation: mu %.1f%%, Sigma %.1f%%, state sufficient: %s",
            variation["mean_relative_shift"] * 100,
            variation["covariance_relative_shift"] * 100,
            variation["state_is_sufficient"],
        )
        report["configs"][name] = entry
        _write_report(report_path, report)

    report["status"] = "complete"
    _write_report(report_path, report)

    logger.info("=== summary (tracking error relative to each expert's never-trade rule)")
    for name, entry in report["configs"].items():
        sufficient = entry["horizon_variation"]["state_is_sufficient"]
        ratios = " ".join(
            f"{policy}={values['ratio_to_never_trade']:.3f}"
            for policy, values in entry["policies"].items()
        )
        logger.info("  %-16s state_sufficient=%-5s  %s", name, str(sufficient), ratios)
    logger.info("report: %s", report_path)


if __name__ == "__main__":
    main()
