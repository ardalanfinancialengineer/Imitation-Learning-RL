"""Run and visualise the Weeks 17-21 policy models and behavioural cloning.

The script builds (or reloads) the Weeks 14-16 paths, trains the architecture
floor, the deterministic transformer policy, and the transformer-diffusion
policy on the same demonstrations under the same split protocol, ablates the
behavioural-cloning objective, and evaluates every acceptance check.

It exits with a non-zero status if any acceptance check fails.
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

import matplotlib
import numpy as np
import pandas as pd
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset.demonstrations import DemonstrationDataset  # noqa: E402
from src.dataset.paths import (  # noqa: E402
    PathData,
    PathSpec,
    generate_path_from_spec,
)
from src.dataset.splits import TemporalSplit, split_paths  # noqa: E402
from src.dataset.state import StateSpec  # noqa: E402
from src.expert.constraints import PortfolioConstraints  # noqa: E402
from src.imitation.bc import (  # noqa: E402
    BCConfig,
    BehaviouralCloningTrainer,
    seed_everything,
)
from src.imitation.comparison import (  # noqa: E402
    paired_bootstrap,
    per_path_tracking_errors,
    seed_summary,
)
from src.imitation.diagnostics import (  # noqa: E402
    PolicyDiagnostics,
    PolicyDiagnosticThresholds,
    evaluate_policy,
)
from src.policy.builder import build_policy  # noqa: E402
from src.policy.reference import default_references  # noqa: E402
from src.simulation.presets import load_simulation_preset  # noqa: E402
from src.utils.observability import MemoryTracker, configure_logging  # noqa: E402
from src.utils.parallel import parallel_map  # noqa: E402

POLICY_COLORS = {"mlp": "#9ecae1", "transformer": "#4c78a8", "diffusion": "#e45756"}

logger = logging.getLogger("run_bc_demo")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/bc_demo.yaml"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--n-paths", type=int, default=None)
    parser.add_argument("--periods", type=int, default=None)
    parser.add_argument("--n-assets", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument(
        "--skip-loss-ablation",
        action="store_true",
        help="Train only the three policies, not the objective ablation.",
    )
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
        help="Emit DEBUG-level logs (per-epoch detail, checkpoint writes).",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help="Write a mid-training checkpoint every K epochs (0 disables). "
        "Combine with --resume to continue an interrupted run.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume each policy from its checkpoint if one exists.",
    )
    return parser.parse_args()


def load_or_generate_paths(
    *,
    cache_dir: Path,
    n_paths: int,
    periods: int,
    base_seed: int,
    preset: Any,
    expert_config: dict[str, Any],
    workers: int = 1,
) -> tuple[dict[int, PathData], float, int]:
    """Reload cached paths that match this configuration, generating the rest.

    Expert trajectories are the expensive part of this demo and they do not
    depend on anything the policy stage varies, so regenerating them on every
    run would dominate the runtime for no benefit. A cached path is accepted
    only when its stored metadata matches the requested simulation *and* expert
    settings, so a configuration change can never be silently ignored.

    Runs in three phases so the cache stays correct under parallelism: (1) a
    serial scan that reuses every matching cached path and records the misses;
    (2) a possibly-parallel generation of only the misses (path seed is
    ``base_seed + path_id`` with no global-RNG read, so a parallel-generated
    path is bit-identical to a serial one); (3) a serial save+merge in the
    parent, so no worker ever writes to the cache directory concurrently.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[int, PathData] = {}
    reused = 0
    started = time.perf_counter()

    # Phase 1: serial cache scan. Each path is an independent Monte Carlo
    # realisation, so seeds must not collide; the split is then genuinely across
    # realisations.
    miss_specs: list[PathSpec] = []
    for path_id in range(n_paths):
        file_path = cache_dir / f"path_{path_id:04d}.npz"
        seed = base_seed + path_id
        if file_path.exists():
            try:
                candidate = PathData.load(file_path)
            except (ValueError, OSError, KeyError):
                candidate = None
            if candidate is not None and _matches(
                candidate, seed, periods, preset, expert_config
            ):
                paths[path_id] = candidate
                reused += 1
                continue
        miss_specs.append(
            PathSpec(
                path_id=path_id,
                seed=seed,
                preset=preset,
                periods=periods,
                expert_config=expert_config,
            )
        )

    # Phase 2 + 3: generate the misses (optionally across workers), then save
    # and merge them in the parent.
    total_misses = len(miss_specs)
    if total_misses:
        logger.info("generating %d path(s) (%d reused from cache)", total_misses, reused)

    def _tick(done: int, total: int) -> None:
        if done % 10 == 0 or done == total:
            logger.info("paths %d/%d generated", done, total)

    for generated in parallel_map(
        generate_path_from_spec,
        miss_specs,
        workers=workers,
        progress_callback=_tick,
    ):
        generated.save(cache_dir / f"path_{generated.path_id:04d}.npz")
        paths[generated.path_id] = generated

    return paths, time.perf_counter() - started, reused


def _write_report(report_path: Path, report: dict[str, Any]) -> None:
    """Serialise the report atomically so a kill mid-write never truncates it.

    The report is re-written after each policy completes, so an interrupted run
    leaves a valid JSON whose ``status`` is still ``"in_progress"`` and whose
    ``policies`` holds whatever finished — enough to diagnose where a long run
    died.
    """
    report_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = report_path.with_suffix(report_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    os.replace(tmp_path, report_path)


def _matches(
    path: PathData,
    seed: int,
    periods: int,
    preset: Any,
    expert_config: dict[str, Any],
) -> bool:
    metadata = path.metadata
    return (
        path.seed == seed
        and int(metadata.get("periods", -1)) == periods
        and metadata.get("preset") == preset.name
        and int(metadata.get("n_assets", -1)) == preset.n_assets
        and metadata.get("expert_config") == expert_config
    )


def train_one(
    name: str,
    policy_config: dict[str, Any],
    *,
    spec: StateSpec,
    datasets: dict[str, DemonstrationDataset],
    n_assets: int,
    n_regimes: int,
    constraints: PortfolioConstraints,
    bc_config: BCConfig,
    checkpoint_path: Path | None = None,
    checkpoint_every: int = 0,
    resume: bool = False,
    memory_tracker: MemoryTracker | None = None,
) -> dict[str, Any]:
    """Train one policy and return its history, metrics, and trainer."""
    # Seed before construction, not only before training: the weights are drawn
    # here, so seeding inside fit() would leave the initialisation to whatever
    # the previous policy in this loop left on the global RNG.
    seed_everything(bc_config.seed)
    policy = build_policy(
        policy_config,
        spec=spec,
        n_assets=n_assets,
        n_regimes=n_regimes,
        constraints=constraints,
    )
    trainer = BehaviouralCloningTrainer(policy, bc_config)
    history = trainer.fit(
        datasets["train"],
        datasets["validation"],
        checkpoint_path=checkpoint_path,
        checkpoint_every=checkpoint_every,
        resume=resume,
        memory_tracker=memory_tracker,
    )
    metrics = trainer.evaluate(datasets["test"])
    return {
        "name": name,
        "policy": policy,
        "trainer": trainer,
        "history": history,
        "test_metrics": metrics,
        "description": policy.describe(),
    }


def run_demo(
    cfg: dict[str, Any],
    output_dir: Path,
    *,
    n_paths_override: int | None = None,
    periods_override: int | None = None,
    n_assets_override: int | None = None,
    epochs_override: int | None = None,
    skip_loss_ablation: bool = False,
    workers: int = 1,
    checkpoint_every: int = 0,
    resume: bool = False,
) -> dict[str, PolicyDiagnostics]:
    """Train every policy, evaluate the checks, and persist a visual report."""
    simulation_cfg = _mapping(cfg.get("simulation"), "simulation")
    expert_cfg = _mapping(cfg.get("expert"), "expert")

    n_assets = int(n_assets_override or simulation_cfg["n_assets"])
    periods = int(periods_override or simulation_cfg["T"])
    n_paths = int(n_paths_override or simulation_cfg["n_paths"])
    base_seed = int(simulation_cfg["base_seed"])
    n_regimes = int(expert_cfg["moments"]["n_regimes"])

    preset = load_simulation_preset(str(simulation_cfg["preset"]), n_assets)
    spec = StateSpec(**(cfg.get("state") or {}))
    constraints = PortfolioConstraints(**expert_cfg["constraints"])

    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(cfg.get("paths", {}).get("cache_dir", output_dir / "paths"))
    paths, generation_seconds, reused = load_or_generate_paths(
        cache_dir=cache_dir,
        n_paths=n_paths,
        periods=periods,
        base_seed=base_seed,
        preset=preset,
        expert_config=expert_cfg,
        workers=workers,
    )

    split_cfg = dict(cfg.get("splits") or {})
    path_split = split_paths(
        n_paths,
        train_fraction=float(split_cfg.get("train_fraction", 0.7)),
        validation_fraction=float(split_cfg.get("validation_fraction", 0.15)),
    )
    temporal_split = TemporalSplit(**(split_cfg.get("temporal") or {}))
    datasets = {
        name: DemonstrationDataset(
            [paths[path_id] for path_id in path_split.of(name)], spec=spec
        )
        for name in ("train", "validation", "test")
    }

    training_cfg = dict(cfg.get("training") or {})
    if epochs_override is not None:
        training_cfg["epochs"] = epochs_override
    bc_config = BCConfig(**training_cfg)

    policy_configs = _mapping(cfg.get("policies"), "policies")
    seeds = [int(value) for value in (cfg.get("seeds") or [bc_config.seed])]

    # Every policy is trained once per seed, and every seed sees exactly the
    # same paths and the same split. That is the matched-run design Section 5.4
    # asks for: differences between policies are then not confounded with
    # differences between simulated markets.
    #
    # The loop below saves and frees as it goes rather than holding every
    # trainer in memory at once: a mid-scale run of several policies × several
    # seeds otherwise keeps every network *and its materialised state cache*
    # resident simultaneously, which is what exhausted memory on the last N=50
    # attempt. Only the first-seed policies (needed by diagnostics and the
    # dashboard) survive the loop; every other trainer is checkpointed to disk
    # and released.
    base_thresholds = dict(cfg.get("diagnostics") or {})
    overrides = dict(cfg.get("diagnostics_overrides") or {})
    look_ahead_path = paths[path_split.test[0]]
    checkpoint_dir = output_dir / "checkpoints"

    # Static header, written before any training so a mid-run kill still leaves a
    # valid, inspectable JSON. Policies and cross-seed stability fill in as they
    # complete; ``status`` flips to "complete" only at the very end.
    report: dict[str, Any] = {
        "status": "in_progress",
        "simulation": {
            "preset": preset.name,
            "n_assets": n_assets,
            "periods": periods,
            "n_paths": n_paths,
            "base_seed": base_seed,
            "paths_reused_from_cache": reused,
            "path_seconds": round(generation_seconds, 2),
        },
        "state": datasets["train"].assembler.describe(),
        "splits": {
            "in_distribution": path_split.to_dict(),
            "temporal": temporal_split.to_dict(),
        },
        "datasets": {name: len(value) for name, value in datasets.items()},
        "training": bc_config.to_dict(),
        "diagnostic_thresholds": {"base": base_thresholds, "overrides": overrides},
        "seeds": seeds,
        "policies": {},
    }
    report_path = output_dir / "bc_report.json"
    _write_report(report_path, report)

    results: dict[str, dict[str, Any]] = {}
    diagnostics: dict[str, PolicyDiagnostics] = {}
    # Only the cheap summaries needed by the cross-seed stability report are kept
    # for non-first seeds: the held-out tracking error per seed, plus the
    # first-seed per-path errors for the paired bootstrap.
    seed_tracking_errors: dict[str, list[float]] = {}
    first_seed_path_errors: dict[str, np.ndarray] = {}
    tracker = MemoryTracker(logger, "bc-training")

    for name, config in policy_configs.items():
        seed_tracking_errors[name] = []
        for seed in seeds:
            is_first_seed = seed == seeds[0]
            logger.info("training %s (seed=%d)", name, seed)
            result = train_one(
                name,
                config,
                spec=spec,
                datasets=datasets,
                n_assets=n_assets,
                n_regimes=n_regimes,
                constraints=constraints,
                bc_config=BCConfig(**{**bc_config.to_dict(), "seed": seed}),
                checkpoint_path=(
                    checkpoint_dir / f"{name}.fit.pt" if checkpoint_every else None
                ),
                checkpoint_every=checkpoint_every,
                resume=resume,
                memory_tracker=tracker,
            )
            seed_tracking_errors[name].append(result["test_metrics"]["tracking_error"])
            tracker.sample(f"{name}-seed{seed}")

            if is_first_seed:
                # The first seed carries the acceptance checks and the dashboard,
                # so it remains a statement about one reproducible run.
                first_seed_path_errors[name] = per_path_tracking_errors(
                    result["policy"], datasets["test"]
                )
                diagnostics[name] = evaluate_policy(
                    result["policy"],
                    result["trainer"],
                    train_dataset=datasets["train"],
                    validation_dataset=datasets["validation"],
                    holdout_dataset=datasets["test"],
                    look_ahead_path=look_ahead_path,
                    preset=preset,
                    thresholds=PolicyDiagnosticThresholds(
                        **(base_thresholds | dict(overrides.get(name) or {}))
                    ),
                )
                result["trainer"].save(
                    checkpoint_dir / f"{name}.pt",
                    extra={
                        "state_spec": spec.to_dict(),
                        "path_split": path_split.to_dict(),
                    },
                )
                report["policies"][name] = {
                    "description": result["description"],
                    "history": result["history"].to_dict(),
                    "test_metrics": result["test_metrics"],
                    "diagnostics": diagnostics[name].to_dict(),
                }
                _write_report(report_path, report)
                # Keep only what the dashboard needs; release the trainer (and
                # therefore its state cache) now that diagnostics are computed.
                results[name] = {
                    "name": name,
                    "policy": result["policy"],
                    "history": result["history"],
                    "test_metrics": result["test_metrics"],
                    "description": result["description"],
                }
                result["trainer"]._cache.clear()
            else:
                result["trainer"]._cache.clear()
            del result

    ablation = (
        {}
        if skip_loss_ablation
        else _loss_ablation(
            cfg,
            policy_configs,
            spec=spec,
            datasets=datasets,
            n_assets=n_assets,
            n_regimes=n_regimes,
            constraints=constraints,
            bc_config=bc_config,
        )
    )

    references = _reference_errors(datasets["test"], spec, n_assets, n_regimes, constraints)
    stability = _stability_report(
        seed_tracking_errors, first_seed_path_errors, seeds
    )
    tracker.summary()

    report["reference_tracking_errors"] = references
    report["stability"] = stability
    report["loss_ablation"] = ablation
    report["status"] = "complete"
    _write_report(report_path, report)
    _write_tables(output_dir, results, diagnostics, ablation, references)

    save_dashboard(
        results=results,
        stability=stability,
        diagnostics=diagnostics,
        references=references,
        ablation=ablation,
        datasets=datasets,
        spec=spec,
        output_path=output_dir / "bc_dashboard.png",
    )
    return diagnostics


def _loss_ablation(
    cfg: dict[str, Any],
    policy_configs: dict[str, Any],
    *,
    spec: StateSpec,
    datasets: dict[str, DemonstrationDataset],
    n_assets: int,
    n_regimes: int,
    constraints: PortfolioConstraints,
    bc_config: BCConfig,
) -> dict[str, Any]:
    """Retrain the deterministic transformer under each objective.

    Held on the transformer alone: the diffusion policy trains a denoising loss
    regardless of this setting, so including it would produce three identical
    rows and imply a comparison that was never run.
    """
    names = list(cfg.get("loss_ablation") or [])
    if not names or "transformer" not in policy_configs:
        return {}

    ablation: dict[str, Any] = {}
    for loss_name in names:
        if loss_name == bc_config.loss and loss_name in ablation:
            continue
        config = BCConfig(**{**bc_config.to_dict(), "loss": loss_name})
        result = train_one(
            f"transformer_{loss_name}",
            policy_configs["transformer"],
            spec=spec,
            datasets=datasets,
            n_assets=n_assets,
            n_regimes=n_regimes,
            constraints=constraints,
            bc_config=config,
        )
        ablation[loss_name] = {
            "test_metrics": result["test_metrics"],
            "best_epoch": result["history"].best_epoch,
            "seconds": round(result["history"].total_seconds, 2),
        }
    return ablation


def _stability_report(
    seed_tracking_errors: dict[str, list[float]],
    first_seed_path_errors: dict[str, np.ndarray],
    seeds: list[int],
) -> dict[str, Any]:
    """Summarise across seeds, and pair every policy against every other.

    Two distinct sources of variation are reported separately rather than
    pooled. ``per_policy`` spreads a metric over training seeds, which is
    optimisation noise. ``pairwise`` bootstraps the difference between two
    policies over held-out *paths* at a fixed seed, which is sampling noise.
    A gap that survives only one of the two is not a result.

    Takes only the cheap summaries — per-seed held-out tracking errors and the
    first-seed per-path errors — rather than the live trainers, so the caller can
    release each network as soon as it is trained (see ``run_demo``).
    """
    per_policy = {
        name: seed_summary(errors) for name, errors in seed_tracking_errors.items()
    }

    names = list(first_seed_path_errors)
    pairwise = {}
    for index, first in enumerate(names):
        for second in names[index + 1 :]:
            interval = paired_bootstrap(
                first_seed_path_errors[first], first_seed_path_errors[second]
            )
            pairwise[f"{first}_minus_{second}"] = interval.to_dict()

    return {
        "seeds": seeds,
        "per_policy": per_policy,
        "pairwise_paired_bootstrap": pairwise,
        "note": (
            "per_policy spreads the held-out tracking error over training seeds "
            "(optimisation noise). pairwise bootstraps the paired per-path "
            "difference at the first seed (sampling noise). A positive interval "
            "means the first policy is worse; one that straddles zero means the "
            "two are not distinguishable on this evidence."
        ),
    }


def _reference_errors(
    dataset: DemonstrationDataset,
    spec: StateSpec,
    n_assets: int,
    n_regimes: int,
    constraints: PortfolioConstraints,
) -> dict[str, float]:
    states = np.stack([dataset.sample(index)[0] for index in range(len(dataset))])
    actions = np.stack([dataset.sample(index)[1] for index in range(len(dataset))])
    references = default_references(
        spec, n_assets=n_assets, n_regimes=n_regimes, constraints=constraints
    )
    return {
        name: float(np.mean(np.sum((rule.act(states) - actions) ** 2, axis=1)))
        for name, rule in references.items()
    }


def _write_tables(
    output_dir: Path,
    results: dict[str, Any],
    diagnostics: dict[str, PolicyDiagnostics],
    ablation: dict[str, Any],
    references: dict[str, float],
) -> None:
    rows = []
    for name, result in results.items():
        metrics = result["test_metrics"]
        report = diagnostics[name]
        rows.append(
            {
                "policy": name,
                "parameters": result["policy"].n_parameters,
                "train_seconds": round(result["history"].total_seconds, 2),
                "best_epoch": result["history"].best_epoch,
                "test_tracking_error": metrics["tracking_error"],
                "test_absolute_error": metrics["absolute_error"],
                "teacher_forced_turnover": metrics["teacher_forced_turnover"],
                "expert_turnover": metrics["expert_turnover"],
                "feasibility_violation": metrics["feasibility_violation"],
                "inference_seconds": report.policy_seconds_per_decision,
                "expert_seconds": report.expert_seconds_per_decision,
                "latency_speedup": report.latency_speedup,
                "generalisation_ratio": report.generalisation_ratio,
                "passed": report.passed,
            }
        )
    for name, value in references.items():
        rows.append({"policy": f"reference:{name}", "test_tracking_error": value})
    pd.DataFrame(rows).to_csv(output_dir / "policy_comparison.csv", index=False)

    history_rows = [
        {"policy": name, **record.to_dict()}
        for name, result in results.items()
        for record in result["history"].records
    ]
    pd.DataFrame(history_rows).to_csv(output_dir / "training_history.csv", index=False)

    if ablation:
        pd.DataFrame(
            [
                {
                    "loss": loss_name,
                    "test_tracking_error": entry["test_metrics"]["tracking_error"],
                    "test_absolute_error": entry["test_metrics"]["absolute_error"],
                    "teacher_forced_turnover": entry["test_metrics"][
                        "teacher_forced_turnover"
                    ],
                    "best_epoch": entry["best_epoch"],
                    "seconds": entry["seconds"],
                }
                for loss_name, entry in ablation.items()
            ]
        ).to_csv(output_dir / "loss_ablation.csv", index=False)

    check_rows = [
        {"policy": name, "check": check, "passed": passed}
        for name, report in diagnostics.items()
        for check, passed in report.checks.items()
    ]
    pd.DataFrame(check_rows).to_csv(output_dir / "acceptance_checks.csv", index=False)


def save_dashboard(
    *,
    results: dict[str, Any],
    stability: dict[str, Any],
    diagnostics: dict[str, PolicyDiagnostics],
    references: dict[str, float],
    ablation: dict[str, Any],
    datasets: dict[str, DemonstrationDataset],
    spec: StateSpec,
    output_path: Path,
) -> None:
    """Save a six-panel visual audit of the policies and their training."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    names = list(results)
    fig.suptitle(
        "Weeks 17–21: Policy Models and Behavioural Cloning "
        f"(L={spec.lookback}, state dim {datasets['train'].state_dimension}, "
        f"{len(datasets['train'])} train samples)",
        fontsize=15,
        fontweight="bold",
    )

    # Panel 1: validation learning curves.
    for name in names:
        records = results[name]["history"].records
        axes[0, 0].plot(
            [record.epoch for record in records],
            [record.validation_tracking_error for record in records],
            label=name,
            color=POLICY_COLORS.get(name, "#666666"),
            linewidth=1.6,
        )
    axes[0, 0].axhline(
        references["previous_weights"],
        color="#222222",
        linestyle="--",
        linewidth=1.2,
        label="never trade",
    )
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_title("Validation tracking error (Equation 18)")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel(r"$\frac{1}{T}\sum\|w_t - w^*_t\|_2^2$")
    axes[0, 0].legend(fontsize=8)

    # Panel 2: held-out quality with seed spread, against the never-trade rule.
    # Equal weight is 36x worse and would flatten the scale; it is in the report.
    summaries = stability["per_policy"]
    means = [summaries[name]["mean"] for name in names]
    spreads = [summaries[name]["sd"] for name in names]
    axes[0, 1].bar(
        range(len(names)),
        means,
        yerr=spreads,
        capsize=4,
        color=[POLICY_COLORS.get(name, "#666666") for name in names],
    )
    axes[0, 1].axhline(
        references["previous_weights"],
        color="#222222",
        linestyle="--",
        linewidth=1.2,
        label="never trade",
    )
    axes[0, 1].set_xticks(range(len(names)))
    axes[0, 1].set_xticklabels(names, fontsize=8)
    axes[0, 1].set_ylabel("Held-out tracking error")
    n_seeds = summaries[names[0]]["n_seeds"]
    axes[0, 1].set_title(
        f"Held-out tracking error (mean ± sd over {n_seeds} seeds)\n"
        f"equal weight is {references['equal_weight']:.3f}, off scale",
        fontsize=10,
    )
    axes[0, 1].legend(fontsize=8)

    # Panel 3: research question 5, decision-time cost.
    width = 0.38
    positions = np.arange(len(names))
    policy_times = [diagnostics[name].policy_seconds_per_decision * 1e3 for name in names]
    expert_times = [diagnostics[name].expert_seconds_per_decision * 1e3 for name in names]
    axes[0, 2].bar(positions - width / 2, policy_times, width, label="policy", color="#4c78a8")
    axes[0, 2].bar(positions + width / 2, expert_times, width, label="expert QP", color="#e45756")
    axes[0, 2].set_xticks(positions)
    axes[0, 2].set_xticklabels(names, fontsize=8)
    axes[0, 2].set_yscale("log")
    axes[0, 2].set_ylabel("Milliseconds per decision (log)")
    axes[0, 2].set_title(
        "Inference cost per rebalancing date\n"
        + ", ".join(f"{name}: {diagnostics[name].latency_speedup:.0f}x" for name in names),
        fontsize=10,
    )
    axes[0, 2].legend(fontsize=8)

    # Panel 4: allocation agreement on held-out states.
    reference_name = min(names, key=lambda n: results[n]["test_metrics"]["tracking_error"])
    sample_count = min(400, len(datasets["test"]))
    states = np.stack([datasets["test"].sample(i)[0] for i in range(sample_count)])
    expert_actions = np.stack([datasets["test"].sample(i)[1] for i in range(sample_count)])
    predicted = np.asarray(results[reference_name]["policy"].act(states), dtype=float)
    axes[1, 0].scatter(
        expert_actions.reshape(-1), predicted.reshape(-1), s=4, alpha=0.25, color="#4c78a8"
    )
    limit = max(float(expert_actions.max()), float(predicted.max())) * 1.05
    axes[1, 0].plot([0, limit], [0, limit], color="#222222", linewidth=1.0, linestyle="--")
    axes[1, 0].set_xlim(0, limit)
    axes[1, 0].set_ylim(0, limit)
    axes[1, 0].set_title(f"Expert vs {reference_name} weights (held-out)")
    axes[1, 0].set_xlabel("Expert weight $w^*_{t,i}$")
    axes[1, 0].set_ylabel("Policy weight $w_{t,i}$")

    # Panel 5: does each policy actually beat never-trade, with an interval?
    # A bare inequality is cleared by noise: both heads start at w_{t-1}.
    pairs = [
        (name, diagnostics[name].baseline_intervals["previous_weights"])
        for name in names
    ]
    positions = np.arange(len(pairs))
    centres = np.array([interval.mean for _, interval in pairs])
    lows = np.array([interval.low for _, interval in pairs])
    highs = np.array([interval.high for _, interval in pairs])
    axes[1, 1].errorbar(
        centres,
        positions,
        xerr=[centres - lows, highs - centres],
        fmt="o",
        capsize=5,
        color="#4c78a8",
    )
    axes[1, 1].axvline(0.0, color="#e45756", linestyle="--", linewidth=1.2)
    axes[1, 1].set_yticks(positions)
    axes[1, 1].set_yticklabels([name for name, _ in pairs], fontsize=8)
    axes[1, 1].invert_yaxis()
    axes[1, 1].set_xlabel("Never-trade tracking error minus policy's")
    axes[1, 1].set_title(
        "Improvement over never trading\n"
        "(paired over test paths, 95% bootstrap; right of 0 is better)",
        fontsize=10,
    )

    # Panel 6: acceptance checks.
    check_names = list(next(iter(diagnostics.values())).checks)
    grid = np.array(
        [[1 if diagnostics[name].checks[check] else 0 for name in names] for check in check_names]
    )
    axes[1, 2].imshow(grid, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    axes[1, 2].set_xticks(range(len(names)))
    axes[1, 2].set_xticklabels(names, fontsize=8)
    axes[1, 2].set_yticks(range(len(check_names)))
    axes[1, 2].set_yticklabels([name.replace("_", " ") for name in check_names], fontsize=8)
    for row in range(grid.shape[0]):
        for column in range(grid.shape[1]):
            axes[1, 2].text(
                column,
                row,
                "PASS" if grid[row, column] else "FAIL",
                ha="center",
                va="center",
                fontsize=7,
                fontweight="bold",
            )
    everything_passed = all(report.passed for report in diagnostics.values())
    axes[1, 2].set_title(
        "Acceptance checks: " + ("PASS" if everything_passed else "FAIL"),
        color="#238b45" if everything_passed else "#cb181d",
        fontweight="bold",
    )

    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"configuration must contain a {name} mapping.")
    return value


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError(f"configuration file is empty or invalid: {args.config}")

    output_dir = args.output_dir or Path(
        cfg.get("paths", {}).get("output_dir", "data/bc_demo")
    )
    diagnostics = run_demo(
        cfg,
        output_dir,
        n_paths_override=args.n_paths,
        periods_override=args.periods,
        n_assets_override=args.n_assets,
        epochs_override=args.epochs,
        skip_loss_ablation=args.skip_loss_ablation,
        workers=args.workers,
        checkpoint_every=args.checkpoint_every,
        resume=args.resume,
    )

    logger.info("Wrote behavioural cloning demo to %s", output_dir)
    everything_passed = True
    for name, report in diagnostics.items():
        logger.info("%s: %s", name, "PASS" if report.passed else "FAIL")
        for check, passed in report.checks.items():
            logger.info("  %s: %s", check, "PASS" if passed else "FAIL")
        everything_passed = everything_passed and report.passed
    logger.info("dashboard: %s", output_dir / "bc_dashboard.png")
    logger.info("report: %s", output_dir / "bc_report.json")
    if not everything_passed:
        raise SystemExit("Policy acceptance checks failed; inspect bc_report.json.")


if __name__ == "__main__":
    main()
