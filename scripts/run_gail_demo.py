"""Run and visualise the completed Weeks 26-29 Wasserstein-GAIL stage.

The script behaviour-clones a policy, runs DAgger on a copy, then runs GAIL from
each of those two starting points, and repeats the GAIL run under both Lipschitz
enforcement methods the exposé cites. It reports the three-way comparison the
work package asks for, the stability signals it asks to monitor, and every
acceptance check.

It exits with a non-zero status if any acceptance check fails.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from copy import deepcopy
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
from src.dataset.paths import PathSpec, generate_path_from_spec  # noqa: E402
from src.dataset.splits import split_paths  # noqa: E402
from src.dataset.state import StateSpec  # noqa: E402
from src.expert.config import build_constraints  # noqa: E402
from src.imitation.bc import (  # noqa: E402
    BCConfig,
    BehaviouralCloningTrainer,
    seed_everything,
)
from src.imitation.comparison import paired_bootstrap  # noqa: E402
from src.imitation.critic import CriticConfig  # noqa: E402
from src.imitation.dagger import DAggerConfig, DAggerTrainer  # noqa: E402
from src.imitation.gail import GAILConfig, GAILTrainer  # noqa: E402
from src.imitation.gail_diagnostics import (  # noqa: E402
    GAILDiagnostics,
    GAILDiagnosticThresholds,
    compare_policies,
    evaluate_gail,
)
from src.imitation.rollout import (  # noqa: E402
    ExpertOracle,
    answer_rollouts,
    per_path_closed_loop_errors,
    rollout_metrics,
    rollout_policy,
)
from src.policy.builder import build_policy  # noqa: E402
from src.policy.reference import PreviousWeightsReference  # noqa: E402
from src.simulation.presets import load_simulation_preset  # noqa: E402
from src.utils.observability import MemoryTracker, configure_logging  # noqa: E402
from src.utils.parallel import parallel_map  # noqa: E402

logger = logging.getLogger("run_gail_demo")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/gail_demo.yaml"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--n-paths", type=int, default=None)
    parser.add_argument("--periods", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Path-generation worker processes. 1 (default) is a serial loop, "
        "bit-identical to before; >1 fans paths across cores; 0 uses every core.",
    )
    parser.add_argument(
        "--skip-ablation",
        action="store_true",
        help="Skip the weight-clipping stability comparison.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Emit DEBUG-level logging; default is INFO.",
    )
    return parser.parse_args()


def _log(message: str) -> None:
    logger.info("%s", message)


def run_demo(
    cfg: dict[str, Any],
    output_dir: Path,
    *,
    n_paths_override: int | None = None,
    periods_override: int | None = None,
    iterations_override: int | None = None,
    skip_ablation: bool = False,
    workers: int = 1,
) -> GAILDiagnostics:
    """Train BC, DAgger, and GAIL on shared paths, evaluate, and persist a report."""
    tracker = MemoryTracker(logger, phase="gail")
    simulation_cfg = _mapping(cfg.get("simulation"), "simulation")
    expert_cfg = _mapping(cfg.get("expert"), "expert")

    n_assets = int(simulation_cfg["n_assets"])
    periods = int(periods_override or simulation_cfg["T"])
    n_paths = int(n_paths_override or simulation_cfg["n_paths"])
    base_seed = int(simulation_cfg["base_seed"])

    spec = StateSpec(**(cfg.get("state") or {}))
    preset = load_simulation_preset(str(simulation_cfg["preset"]), n_assets)
    split_cfg = dict(cfg.get("splits") or {})
    split = split_paths(
        n_paths,
        train_fraction=float(split_cfg.get("train_fraction", 0.7)),
        validation_fraction=float(split_cfg.get("validation_fraction", 0.15)),
    )
    constraints = build_constraints(expert_cfg.get("constraints"))
    transaction_cost = float(expert_cfg["transaction_cost"])

    gail_settings = dict(cfg.get("gail") or {})
    if iterations_override is not None:
        gail_settings["iterations"] = iterations_override
    critic_settings = dict(cfg.get("critic") or {})
    critic_settings["hidden_dimensions"] = tuple(
        critic_settings.get("hidden_dimensions", (256, 128))
    )

    started = time.perf_counter()
    _log(f"generating {n_paths} paths of {periods} periods")
    specs = [
        PathSpec(
            path_id=path_id,
            seed=base_seed + path_id,
            preset=preset,
            periods=periods,
            expert_config=expert_cfg,
        )
        for path_id in range(n_paths)
    ]
    def _tick(done: int, total: int) -> None:
        if done % 10 == 0 or done == total:
            logger.info("paths %d/%d generated", done, total)

    paths = {
        result.path_id: result
        for result in parallel_map(
            generate_path_from_spec, specs, workers=workers, progress_callback=_tick
        )
    }
    tracker.sample("paths-ready")
    datasets = {
        name: DemonstrationDataset(
            [paths[path_id] for path_id in split.of(name)], spec=spec
        )
        for name in ("train", "validation", "test")
    }
    oracles = {name: ExpertOracle(datasets[name]) for name in ("train", "validation", "test")}

    # ---- Stage 1: behavioural cloning -------------------------------------
    bc_config = BCConfig(**(cfg.get("behavioural_cloning") or {}))
    seed_everything(bc_config.seed)
    cloned = build_policy(
        dict(cfg["policy"]),
        spec=spec,
        n_assets=n_assets,
        n_regimes=paths[0].n_regimes,
        constraints=constraints,
    )
    _log("stage 1: behavioural cloning")
    BehaviouralCloningTrainer(cloned, bc_config).fit(
        datasets["train"], datasets["validation"]
    )

    # ---- Stage 2: DAgger, for the comparison and as a warm start ----------
    _log("stage 2: DAgger")
    dagger_policy = deepcopy(cloned)
    dagger = DAggerTrainer(
        dagger_policy,
        bc_config=bc_config,
        config=DAggerConfig(**(cfg.get("dagger") or {})),
    )
    dagger_history, _ = dagger.fit(
        datasets["train"],
        datasets["validation"],
        oracle=oracles["train"],
        validation_oracle=oracles["validation"],
    )
    tracker.sample("dagger-fitted")

    # ---- Stage 3: GAIL from each starting point ---------------------------
    runs: dict[str, dict[str, Any]] = {}
    trainers: dict[str, GAILTrainer] = {}
    for label, start in (("gail_from_bc", cloned), ("gail_from_dagger", dagger_policy)):
        _log(f"stage 3: {label}")
        policy = deepcopy(start)
        trainer = GAILTrainer(
            policy,
            config=GAILConfig(**gail_settings),
            critic_config=CriticConfig(**critic_settings),
        )
        history = trainer.fit(
            datasets["train"],
            datasets["validation"],
            validation_oracle=oracles["validation"],
        )
        tracker.sample("gail-fitted")
        runs[label] = {"policy": policy, "history": history}
        trainers[label] = trainer
        _log(
            f"  {label}: best closed-loop {history.best_closed_loop_error:.4e} "
            f"at iteration {history.best_iteration} in {history.total_seconds:.0f}s"
        )

    # ---- Stability ablation ------------------------------------------------
    ablation: dict[str, Any] = {}
    if not skip_ablation:
        for method in (cfg.get("stability_ablation") or {}).get("lipschitz", []):
            if method == critic_settings.get("lipschitz"):
                ablation[method] = runs["gail_from_bc"]["history"].to_dict()
                continue
            _log(f"stability ablation: {method}")
            policy = deepcopy(cloned)
            trainer = GAILTrainer(
                policy,
                config=GAILConfig(**gail_settings),
                critic_config=CriticConfig(**(critic_settings | {"lipschitz": method})),
            )
            ablation[method] = trainer.fit(
                datasets["train"],
                datasets["validation"],
                validation_oracle=oracles["validation"],
            ).to_dict()

    # ---- Comparison --------------------------------------------------------
    _log("evaluating the three-way comparison on held-out paths")
    policies = {
        "behavioural_cloning": cloned,
        "dagger": dagger_policy,
        "gail_from_bc": runs["gail_from_bc"]["policy"],
        "gail_from_dagger": runs["gail_from_dagger"]["policy"],
        "never_trade": PreviousWeightsReference(
            spec, n_assets=n_assets, n_regimes=paths[0].n_regimes, constraints=constraints
        ),
    }
    comparison = _compare(policies, datasets["test"], oracles["test"], transaction_cost)
    # One critic scores every policy, because the Wasserstein estimate is only
    # comparable within a critic.
    learned = {name: policy for name, policy in policies.items() if name != "never_trade"}
    comparison["occupancy_distance"] = compare_policies(
        trainers["gail_from_bc"], learned, datasets["test"]
    )
    comparison["occupancy_distance_caveat"] = (
        "NOT a valid comparison between policies. One critic scores them all, "
        "which makes the numbers commensurable but not meaningful: the critic "
        "was trained to discriminate the expert from one particular policy and "
        "is near-optimal only for that pair. Applied elsewhere it is an "
        "arbitrary 1-Lipschitz function. On the shipped run it returns negative "
        "values - impossible for a distance lower bound - and ranks the best "
        "policy worst. A correct W1 per policy needs a fresh critic trained to "
        "convergence against each. See docs/gail.md."
    )

    diagnostics = evaluate_gail(
        trainers["gail_from_bc"],
        runs["gail_from_bc"]["history"],
        datasets["train"],
        datasets["test"],
        preset=preset,
        thresholds=GAILDiagnosticThresholds(**(cfg.get("diagnostics") or {})),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "simulation": {
            "preset": preset.name,
            "n_assets": n_assets,
            "periods": periods,
            "n_paths": n_paths,
            "wall_clock_seconds": round(time.perf_counter() - started, 1),
        },
        "gail": trainers["gail_from_bc"].describe(),
        "histories": {
            label: payload["history"].to_dict() for label, payload in runs.items()
        },
        "dagger_rounds": [record.to_dict() for record in dagger_history.records],
        "stability_ablation": {
            method: {"stability": payload["stability"], "diverged": payload["diverged"],
                     "best_closed_loop_tracking_error": payload[
                         "best_closed_loop_tracking_error"]}
            for method, payload in ablation.items()
        },
        "comparison": comparison,
        "diagnostics": diagnostics.to_dict(),
    }
    (output_dir / "gail_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    _write_tables(output_dir, runs, comparison, ablation)
    save_dashboard(
        runs=runs,
        ablation=ablation,
        comparison=comparison,
        diagnostics=diagnostics,
        output_path=output_dir / "gail_dashboard.png",
    )
    tracker.summary()
    return diagnostics


def _compare(
    policies: dict[str, Any],
    dataset: DemonstrationDataset,
    oracle: ExpertOracle,
    transaction_cost: float,
) -> dict[str, Any]:
    """Closed-loop metrics and paired intervals for every policy."""
    per_path: dict[str, dict[int, float]] = {}
    metrics: dict[str, dict[str, float]] = {}
    for name, policy in policies.items():
        rollouts = rollout_policy(policy, dataset)
        answers = answer_rollouts(oracle, rollouts)
        metrics[name] = rollout_metrics(rollouts, answers)
        per_path[name] = per_path_closed_loop_errors(rollouts, answers)

    intervals = {}
    for name in ("dagger", "gail_from_bc", "gail_from_dagger"):
        if name in per_path:
            intervals[f"behavioural_cloning_minus_{name}"] = paired_bootstrap(
                per_path["behavioural_cloning"], per_path[name]
            ).to_dict()
    if "dagger" in per_path and "gail_from_bc" in per_path:
        intervals["dagger_minus_gail_from_bc"] = paired_bootstrap(
            per_path["dagger"], per_path["gail_from_bc"]
        ).to_dict()

    del transaction_cost
    return {"metrics": metrics, "paired_intervals": intervals}


def _write_tables(
    output_dir: Path,
    runs: dict[str, dict[str, Any]],
    comparison: dict[str, Any],
    ablation: dict[str, Any],
) -> None:
    rows = []
    for label, payload in runs.items():
        for record in payload["history"].records:
            rows.append({"run": label} | record.to_dict())
    pd.DataFrame(rows).to_csv(output_dir / "gail_iterations.csv", index=False)

    summary = []
    for name, values in comparison["metrics"].items():
        summary.append(
            {
                "policy": name,
                "closed_loop_tracking_error": values["closed_loop_tracking_error"],
                "mean_anchor_drift": values["mean_anchor_drift"],
                "mean_turnover": values["mean_turnover"],
                "annualised_net_return": values["annualised_net_return"],
                "occupancy_distance": comparison["occupancy_distance"].get(name),
            }
        )
    pd.DataFrame(summary).to_csv(output_dir / "gail_comparison.csv", index=False)

    if ablation:
        pd.DataFrame(
            [{"lipschitz": method, **payload["stability"], "diverged": payload["diverged"]}
             for method, payload in ablation.items()]
        ).to_csv(output_dir / "gail_stability.csv", index=False)


def save_dashboard(
    *,
    runs: dict[str, dict[str, Any]],
    ablation: dict[str, Any],
    comparison: dict[str, Any],
    diagnostics: GAILDiagnostics,
    output_path: Path,
) -> None:
    """Save a six-panel visual audit of the GAIL stage."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    fig.suptitle(
        "Weeks 26–29: Wasserstein-regularised GAIL — matching occupancy, not actions",
        fontsize=15,
        fontweight="bold",
    )
    colours = {"gail_from_bc": "#4c78a8", "gail_from_dagger": "#e45756"}

    for label, payload in runs.items():
        history = payload["history"]
        iterations = history.series("iteration")
        axes[0, 0].plot(
            iterations,
            history.series("wasserstein_estimate"),
            color=colours.get(label, "#666666"),
            label=label.replace("_", " "),
        )
        finite = np.isfinite(history.series("closed_loop_tracking_error"))
        axes[0, 1].plot(
            iterations[finite],
            history.series("closed_loop_tracking_error")[finite],
            marker="o",
            markersize=4,
            color=colours.get(label, "#666666"),
            label=label.replace("_", " "),
        )
    axes[0, 0].set_title("Wasserstein estimate $E_{d_E}[D]-E_{d_\\pi}[D]$")
    axes[0, 0].set_xlabel("Adversarial iteration")
    axes[0, 0].set_ylabel("$W_1$ lower bound")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].set_title("Closed-loop tracking error during training")
    axes[0, 1].set_xlabel("Adversarial iteration")
    axes[0, 1].set_ylabel("$E\\,\\|w_t - w^*(s_t)\\|_2^2$")
    axes[0, 1].legend(fontsize=8)

    names = [n for n in comparison["metrics"] if n != "never_trade"]
    errors = [comparison["metrics"][n]["closed_loop_tracking_error"] for n in names]
    baseline = comparison["metrics"]["never_trade"]["closed_loop_tracking_error"]
    axes[0, 2].bar(
        [n.replace("_", "\n") for n in names],
        errors,
        color=["#bbbbbb", "#f2cf5b", "#4c78a8", "#e45756"][: len(names)],
    )
    axes[0, 2].axhline(baseline, color="#222222", linestyle="--", label="never trade")
    axes[0, 2].set_title("Closed-loop tracking error, held-out paths")
    axes[0, 2].set_ylabel("Tracking error")
    axes[0, 2].tick_params(axis="x", labelsize=7)
    axes[0, 2].legend(fontsize=8)

    if ablation:
        methods = list(ablation)
        volatility = [
            ablation[m]["stability"].get("wasserstein_volatility", 0.0) for m in methods
        ]
        lipschitz = [
            ablation[m]["stability"].get("max_measured_lipschitz", 0.0) for m in methods
        ]
        positions = np.arange(len(methods))
        axes[1, 0].bar(positions - 0.2, volatility, 0.4, label="$W_1$ volatility",
                       color="#4c78a8")
        axes[1, 0].bar(positions + 0.2, lipschitz, 0.4, label="max $\\|\\nabla D\\|$",
                       color="#e45756")
        axes[1, 0].axhline(1.0, color="#222222", linestyle=":", linewidth=1)
        axes[1, 0].set_xticks(positions, [m.replace("_", "\n") for m in methods],
                              fontsize=8)
        axes[1, 0].set_yscale("log")
        axes[1, 0].set_title("Stability: Lipschitz enforcement")
        axes[1, 0].legend(fontsize=8)
    else:
        axes[1, 0].set_axis_off()

    for label, payload in runs.items():
        history = payload["history"]
        axes[1, 1].plot(
            history.series("iteration"),
            history.series("measured_lipschitz"),
            color=colours.get(label, "#666666"),
            label=label.replace("_", " "),
        )
    axes[1, 1].axhline(1.0, color="#222222", linestyle="--", label="target = 1")
    axes[1, 1].set_title("Critic Lipschitz constant, measured not assumed")
    axes[1, 1].set_xlabel("Adversarial iteration")
    axes[1, 1].set_ylabel("$\\max\\|\\nabla_x D\\|$")
    axes[1, 1].legend(fontsize=8)

    check_names = [name.replace("_", " ").title() for name in diagnostics.checks]
    check_values = [1 if passed else 0 for passed in diagnostics.checks.values()]
    axes[1, 2].barh(
        check_names,
        np.ones(len(check_names)),
        color=["#31a354" if v else "#e45756" for v in check_values],
    )
    axes[1, 2].set_xlim(0.0, 1.0)
    axes[1, 2].set_xticks([])
    axes[1, 2].invert_yaxis()
    axes[1, 2].set_title(
        "Acceptance checks: " + ("PASS" if diagnostics.passed else "FAIL"),
        color="#238b45" if diagnostics.passed else "#cb181d",
        fontweight="bold",
    )
    for index, passed in enumerate(check_values):
        axes[1, 2].text(0.5, index, "PASS" if passed else "FAIL", ha="center",
                        va="center", color="white", fontweight="bold")
    axes[1, 2].set_xlabel(
        f"rollout gradient vs finite differences: {diagnostics.gradient_relative_error:.1e}",
        fontsize=8,
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
        cfg.get("paths", {}).get("output_dir", "data/gail_demo")
    )
    diagnostics = run_demo(
        cfg,
        output_dir,
        n_paths_override=args.n_paths,
        periods_override=args.periods,
        iterations_override=args.iterations,
        skip_ablation=args.skip_ablation,
        workers=args.workers,
    )

    logger.info("Wrote GAIL demo to %s", output_dir)
    logger.info("acceptance checks: %s", "PASS" if diagnostics.passed else "FAIL")
    for name, passed in diagnostics.checks.items():
        logger.info("  %s: %s", name, "PASS" if passed else "FAIL")
    logger.info("dashboard: %s", output_dir / "gail_dashboard.png")
    logger.info("report: %s", output_dir / "gail_report.json")
    if not diagnostics.passed:
        raise SystemExit("GAIL acceptance checks failed; inspect gail_report.json.")


if __name__ == "__main__":
    main()
