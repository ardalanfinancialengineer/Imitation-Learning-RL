"""Run and visualise the completed Weeks 22-25 DAgger stage.

The script behaviour-clones a policy, then runs the dataset-aggregation loop:
roll the learner out so it visits its own states, ask the expert what it would
have done there, add those pairs to the training set, retrain, repeat. It does
this for two policies under two expert configurations — the shipped one, where
Weeks 17-21 showed the state is information-limited, and a control where the
state is sufficient — and evaluates every acceptance check.

It exits with a non-zero status if any acceptance check fails.
"""

from __future__ import annotations

import argparse
import json
import logging
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
from src.dataset.paths import PathSpec, generate_path_from_spec  # noqa: E402
from src.dataset.splits import split_paths  # noqa: E402
from src.dataset.state import StateSpec  # noqa: E402
from src.expert.config import build_constraints  # noqa: E402
from src.imitation.bc import (  # noqa: E402
    BCConfig,
    BehaviouralCloningTrainer,
    seed_everything,
)
from src.imitation.dagger import DAggerConfig, DAggerTrainer  # noqa: E402
from src.imitation.dagger_diagnostics import (  # noqa: E402
    DAggerDiagnostics,
    DAggerDiagnosticThresholds,
    evaluate_dagger,
    improvement_interval,
)
from src.imitation.rollout import (  # noqa: E402
    ExpertOracle,
    answer_rollouts,
    per_path_closed_loop_errors,
    rollout_policy,
)
from src.policy.builder import build_policy  # noqa: E402
from src.policy.reference import PreviousWeightsReference  # noqa: E402
from src.simulation.presets import load_simulation_preset  # noqa: E402
from src.utils.observability import MemoryTracker, configure_logging  # noqa: E402
from src.utils.parallel import parallel_map  # noqa: E402

logger = logging.getLogger("run_dagger_demo")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/dagger_demo.yaml"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--n-paths", type=int, default=None)
    parser.add_argument("--periods", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument(
        "--skip-control",
        action="store_true",
        help="Run only the shipped expert, omitting the sufficient-state control.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Path-generation worker processes. 1 (default) is a serial loop, "
        "bit-identical to before; >1 fans paths across cores; 0 uses every core.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Emit DEBUG-level logging; default is INFO.",
    )
    return parser.parse_args()


def run_demo(
    cfg: dict[str, Any],
    output_dir: Path,
    *,
    n_paths_override: int | None = None,
    periods_override: int | None = None,
    iterations_override: int | None = None,
    skip_control: bool = False,
    workers: int = 1,
) -> DAggerDiagnostics:
    """Run DAgger for every policy and expert, evaluate checks, persist a report."""
    tracker = MemoryTracker(logger, phase="dagger")
    simulation_cfg = _mapping(cfg.get("simulation"), "simulation")
    experts_cfg = _mapping(cfg.get("experts"), "experts")
    policies_cfg = _mapping(cfg.get("policies"), "policies")

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

    bc_settings = dict(cfg.get("behavioural_cloning") or {})
    dagger_settings = dict(cfg.get("dagger") or {})
    if iterations_override is not None:
        dagger_settings["iterations"] = iterations_override

    names = ["shipped"] if skip_control else list(experts_cfg)
    results: dict[str, Any] = {}
    diagnostics: DAggerDiagnostics | None = None
    started = time.perf_counter()

    total_runs = len(names) * len(policies_cfg)
    completed = 0

    for expert_name in names:
        expert_cfg = experts_cfg[expert_name]
        # The loop below is minutes long per policy, so it reports as it goes.
        # A script that prints only at the end is indistinguishable from a hung
        # one, which is exactly what happened while this demo was being sized.
        _log(f"[{expert_name}] generating {n_paths} paths of {periods} periods")
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
        constraints = build_constraints(expert_cfg.get("constraints"))
        oracles = {
            name: ExpertOracle(datasets[name]) for name in ("train", "validation", "test")
        }

        baseline = _reference_closed_loop(spec, datasets["test"], oracles["test"],
                                          n_assets, constraints)
        expert_results: dict[str, Any] = {"never_trade_closed_loop": baseline}

        for policy_name, policy_cfg in policies_cfg.items():
            completed += 1
            _log(
                f"[{expert_name}] {policy_name}: behavioural cloning then "
                f"{dagger_settings.get('iterations', 5)} DAgger rounds "
                f"({completed}/{total_runs})"
            )
            run_started = time.perf_counter()
            expert_results[policy_name] = _run_one(
                policy_cfg=policy_cfg,
                spec=spec,
                n_assets=n_assets,
                n_regimes=paths[0].n_regimes,
                constraints=constraints,
                datasets=datasets,
                oracles=oracles,
                bc_settings=bc_settings,
                dagger_settings=dagger_settings,
                memory_tracker=tracker,
            )
            if diagnostics is None and expert_name == "shipped":
                diagnostics = evaluate_dagger(
                    expert_results[policy_name].pop("_policy"),
                    datasets["train"],
                    expert_results[policy_name].pop("_aggregate"),
                    oracles["train"],
                    preset=preset,
                    aggregate_sizes=expert_results[policy_name]["aggregate_sizes"],
                    thresholds=DAggerDiagnosticThresholds(
                        **(cfg.get("diagnostics") or {})
                    ),
                )
            payload = expert_results[policy_name]
            _log(
                f"[{expert_name}] {policy_name}: closed-loop "
                f"{payload['test_before']['closed_loop_tracking_error']:.4e} -> "
                f"{payload['test_after']['closed_loop_tracking_error']:.4e} "
                f"in {time.perf_counter() - run_started:.0f}s"
            )
            payload.pop("_policy", None)
            payload.pop("_aggregate", None)

        results[expert_name] = expert_results

    if diagnostics is None:  # pragma: no cover - guarded by names always non-empty
        raise RuntimeError("no diagnostics were produced.")
    _log("writing report and dashboard")

    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "simulation": {
            "preset": preset.name,
            "n_assets": n_assets,
            "periods": periods,
            "n_paths": n_paths,
            "base_seed": base_seed,
            "wall_clock_seconds": round(time.perf_counter() - started, 1),
        },
        "state": {"dimension": spec.dimension(n_assets, 3), **spec.to_dict()},
        "splits": split.to_dict(),
        "dagger_config": DAggerConfig(**dagger_settings).to_dict(),
        "results": results,
        "diagnostics": diagnostics.to_dict(),
    }
    (output_dir / "dagger_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    _write_tables(output_dir, results)
    save_dashboard(
        results=results,
        diagnostics=diagnostics,
        output_path=output_dir / "dagger_dashboard.png",
    )
    tracker.summary()
    return diagnostics


def _run_one(
    *,
    policy_cfg: dict[str, Any],
    spec: StateSpec,
    n_assets: int,
    n_regimes: int,
    constraints: Any,
    datasets: dict[str, DemonstrationDataset],
    oracles: dict[str, ExpertOracle],
    bc_settings: dict[str, Any],
    dagger_settings: dict[str, Any],
    memory_tracker: MemoryTracker | None = None,
) -> dict[str, Any]:
    """Behaviour-clone one policy, then run DAgger on it."""
    bc_config = BCConfig(**bc_settings)
    # Seeding must precede construction: weights are drawn at construction time.
    seed_everything(bc_config.seed)
    policy = build_policy(
        dict(policy_cfg),
        spec=spec,
        n_assets=n_assets,
        n_regimes=n_regimes,
        constraints=constraints,
    )

    trainer = BehaviouralCloningTrainer(policy, bc_config)
    bc_started = time.perf_counter()
    bc_history = trainer.fit(
        datasets["train"], datasets["validation"], memory_tracker=memory_tracker
    )
    bc_seconds = time.perf_counter() - bc_started

    before = _test_closed_loop(policy, datasets["test"], oracles["test"])

    dagger = DAggerTrainer(
        policy, bc_config=bc_config, config=DAggerConfig(**dagger_settings)
    )
    history, aggregate = dagger.fit(
        datasets["train"],
        datasets["validation"],
        oracle=oracles["train"],
        validation_oracle=oracles["validation"],
        memory_tracker=memory_tracker,
    )
    after = _test_closed_loop(policy, datasets["test"], oracles["test"])

    interval = improvement_interval(before["per_path"], after["per_path"])
    teacher_forced_after = trainer.evaluate(datasets["test"])["tracking_error"]

    return {
        "n_parameters": policy.n_parameters,
        "bc_seconds": round(bc_seconds, 1),
        "bc_best_validation_tracking_error": bc_history.best_tracking_error,
        "rounds": [record.to_dict() for record in history.records],
        "aggregate_sizes": [
            record.aggregate_samples for record in history.records
        ],
        "total_expert_queries": history.total_queries,
        "best_iteration": history.best_iteration,
        "dagger_seconds": round(history.total_seconds, 1),
        "test_before": before["metrics"],
        "test_after": after["metrics"],
        "test_teacher_forced_after": float(teacher_forced_after),
        "improvement": interval.to_dict(),
        "_policy": policy,
        "_aggregate": aggregate,
    }


def _test_closed_loop(
    policy: Any, dataset: DemonstrationDataset, oracle: ExpertOracle
) -> dict[str, Any]:
    from src.imitation.rollout import rollout_metrics

    rollouts = rollout_policy(policy, dataset)
    answers = answer_rollouts(oracle, rollouts)
    return {
        "metrics": rollout_metrics(rollouts, answers),
        "per_path": per_path_closed_loop_errors(rollouts, answers),
    }


def _reference_closed_loop(
    spec: StateSpec,
    dataset: DemonstrationDataset,
    oracle: ExpertOracle,
    n_assets: int,
    constraints: Any,
) -> dict[str, Any]:
    """Closed-loop error of never trading, the bar a rollout must clear."""
    reference = PreviousWeightsReference(
        spec, n_assets=n_assets, n_regimes=3, constraints=constraints
    )
    return _test_closed_loop(reference, dataset, oracle)["metrics"]


def _write_tables(output_dir: Path, results: dict[str, Any]) -> None:
    rows = []
    for expert_name, expert_results in results.items():
        for policy_name, payload in expert_results.items():
            if policy_name == "never_trade_closed_loop":
                continue
            for record in payload["rounds"]:
                rows.append({"expert": expert_name, "policy": policy_name} | record)
    pd.DataFrame(rows).to_csv(output_dir / "dagger_rounds.csv", index=False)

    summary = []
    for expert_name, expert_results in results.items():
        never_trade = expert_results["never_trade_closed_loop"][
            "closed_loop_tracking_error"
        ]
        for policy_name, payload in expert_results.items():
            if policy_name == "never_trade_closed_loop":
                continue
            summary.append(
                {
                    "expert": expert_name,
                    "policy": policy_name,
                    "never_trade_closed_loop": never_trade,
                    "closed_loop_before": payload["test_before"][
                        "closed_loop_tracking_error"
                    ],
                    "closed_loop_after": payload["test_after"][
                        "closed_loop_tracking_error"
                    ],
                    "improvement_mean": payload["improvement"]["mean"],
                    "improvement_low": payload["improvement"]["low"],
                    "improvement_high": payload["improvement"]["high"],
                    "excludes_zero": payload["improvement"]["excludes_zero"],
                    "drift_before": payload["test_before"]["mean_anchor_drift"],
                    "drift_after": payload["test_after"]["mean_anchor_drift"],
                    "expert_queries": payload["total_expert_queries"],
                }
            )
    pd.DataFrame(summary).to_csv(output_dir / "dagger_summary.csv", index=False)


def save_dashboard(
    *,
    results: dict[str, Any],
    diagnostics: DAggerDiagnostics,
    output_path: Path,
) -> None:
    """Save a six-panel visual audit of the DAgger stage."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    fig.suptitle(
        "Weeks 22–25: DAgger — training on the states the learner actually visits",
        fontsize=15,
        fontweight="bold",
    )

    colours = {"transformer": "#4c78a8", "mlp": "#e45756", "diffusion": "#31a354"}
    styles = {"shipped": "-", "control": "--"}
    policy_names = [
        name for name in next(iter(results.values())) if name != "never_trade_closed_loop"
    ]

    for expert_name, expert_results in results.items():
        for policy_name in policy_names:
            rounds = expert_results[policy_name]["rounds"]
            iterations = [record["iteration"] for record in rounds]
            errors = [record["closed_loop_tracking_error"] for record in rounds]
            drifts = [record["mean_anchor_drift"] for record in rounds]
            axes[0, 0].plot(
                iterations,
                errors,
                marker="o",
                markersize=4,
                color=colours.get(policy_name, "#666666"),
                linestyle=styles.get(expert_name, "-"),
                label=f"{policy_name} ({expert_name})",
            )
            axes[0, 1].plot(
                iterations,
                drifts,
                marker="o",
                markersize=4,
                color=colours.get(policy_name, "#666666"),
                linestyle=styles.get(expert_name, "-"),
                label=f"{policy_name} ({expert_name})",
            )

    axes[0, 0].set_title("Closed-loop tracking error falls with each round")
    axes[0, 0].set_xlabel("DAgger round (0 = behavioural cloning)")
    axes[0, 0].set_ylabel("$E\\,\\|w_t - w^*(s_t^{policy})\\|_2^2$")
    axes[0, 0].legend(fontsize=7)

    axes[0, 1].set_title("Covariate shift: drift of $w_{t-1}$ from the expert's")
    axes[0, 1].set_xlabel("DAgger round")
    axes[0, 1].set_ylabel("Mean $\\|w_{t-1}^{policy} - w_{t-1}^{expert}\\|_1$")
    axes[0, 1].legend(fontsize=7)

    labels, closed_before, closed_after, teacher = [], [], [], []
    for expert_name, expert_results in results.items():
        for policy_name in policy_names:
            payload = expert_results[policy_name]
            labels.append(f"{policy_name[:5]}\n{expert_name}")
            closed_before.append(payload["test_before"]["closed_loop_tracking_error"])
            closed_after.append(payload["test_after"]["closed_loop_tracking_error"])
            teacher.append(payload["test_teacher_forced_after"])
    positions = np.arange(len(labels))
    width = 0.28
    axes[0, 2].bar(positions - width, closed_before, width, label="closed-loop, BC",
                   color="#bbbbbb")
    axes[0, 2].bar(positions, closed_after, width, label="closed-loop, DAgger",
                   color="#4c78a8")
    axes[0, 2].bar(positions + width, teacher, width, label="teacher-forced",
                   color="#f2cf5b")
    axes[0, 2].set_yscale("log")
    axes[0, 2].set_xticks(positions, labels, fontsize=7)
    axes[0, 2].set_title("Teacher-forced error is blind to what DAgger fixes")
    axes[0, 2].set_ylabel("Tracking error (log)")
    axes[0, 2].legend(fontsize=7)

    means, lows, highs = [], [], []
    for expert_name, expert_results in results.items():
        for policy_name in policy_names:
            interval = expert_results[policy_name]["improvement"]
            means.append(interval["mean"])
            lows.append(interval["mean"] - interval["low"])
            highs.append(interval["high"] - interval["mean"])
    axes[1, 0].bar(
        positions,
        means,
        yerr=[lows, highs],
        capsize=4,
        color=["#31a354" if mean > 0 else "#e45756" for mean in means],
    )
    axes[1, 0].axhline(0.0, color="#222222", linewidth=1.0)
    axes[1, 0].set_xticks(positions, labels, fontsize=7)
    axes[1, 0].set_title("Improvement over BC, paired bootstrap over held-out paths")
    axes[1, 0].set_ylabel("Closed-loop error reduction")

    for expert_name, expert_results in results.items():
        for policy_name in policy_names:
            rounds = expert_results[policy_name]["rounds"]
            axes[1, 1].plot(
                [record["iteration"] for record in rounds],
                [record["aggregate_samples"] for record in rounds],
                marker="o",
                markersize=4,
                color=colours.get(policy_name, "#666666"),
                linestyle=styles.get(expert_name, "-"),
                label=f"{policy_name} ({expert_name})",
            )
    total_queries = sum(
        expert_results[policy_name]["total_expert_queries"]
        for expert_results in results.values()
        for policy_name in policy_names
    )
    axes[1, 1].set_title(f"Aggregated dataset grows ({total_queries} expert queries)")
    axes[1, 1].set_xlabel("DAgger round")
    axes[1, 1].set_ylabel("Training samples")
    axes[1, 1].legend(fontsize=7)

    check_names = [name.replace("_", " ").title() for name in diagnostics.checks]
    check_values = [1 if passed else 0 for passed in diagnostics.checks.values()]
    check_colors = ["#31a354" if passed else "#e45756" for passed in check_values]
    axes[1, 2].barh(check_names, np.ones(len(check_names)), color=check_colors)
    axes[1, 2].set_xlim(0.0, 1.0)
    axes[1, 2].set_xticks([])
    axes[1, 2].invert_yaxis()
    axes[1, 2].set_title(
        "Acceptance checks: " + ("PASS" if diagnostics.passed else "FAIL"),
        color="#238b45" if diagnostics.passed else "#cb181d",
        fontweight="bold",
    )
    for index, passed in enumerate(check_values):
        axes[1, 2].text(
            0.5,
            index,
            "PASS" if passed else "FAIL",
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
        )

    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def _log(message: str) -> None:
    logger.info("%s", message)


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
        cfg.get("paths", {}).get("output_dir", "data/dagger_demo")
    )
    diagnostics = run_demo(
        cfg,
        output_dir,
        n_paths_override=args.n_paths,
        periods_override=args.periods,
        iterations_override=args.iterations,
        skip_control=args.skip_control,
        workers=args.workers,
    )

    logger.info("Wrote DAgger demo to %s", output_dir)
    logger.info("acceptance checks: %s", "PASS" if diagnostics.passed else "FAIL")
    for name, passed in diagnostics.checks.items():
        logger.info("  %s: %s", name, "PASS" if passed else "FAIL")
    logger.info("dashboard: %s", output_dir / "dagger_dashboard.png")
    logger.info("report: %s", output_dir / "dagger_report.json")
    if not diagnostics.passed:
        raise SystemExit("DAgger acceptance checks failed; inspect dagger_report.json.")


if __name__ == "__main__":
    main()
