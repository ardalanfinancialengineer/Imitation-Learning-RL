"""Run and visualise the completed Weeks 30-33 PPO fine-tuning stage.

Behaviour-clones a policy, fine-tunes it with PPO on the Equation (11) reward,
trains a pure-PPO baseline from random initialisation for comparison, and runs
the exact-gradient baseline the differentiable rollout makes available. Reports
turnover, constraint satisfaction, and certainty-equivalent utility, and
evaluates every acceptance check.

Exits with a non-zero status if any acceptance check fails.
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
from src.imitation.differentiable import build_tape  # noqa: E402
from src.policy.builder import build_policy  # noqa: E402
from src.reinforcement.analytic import AnalyticConfig, AnalyticRewardTrainer  # noqa: E402
from src.reinforcement.diagnostics import (  # noqa: E402
    PPODiagnostics,
    PPODiagnosticThresholds,
    evaluate_ppo,
)
from src.reinforcement.ppo import PPOConfig, PPOTrainer  # noqa: E402
from src.reinforcement.reward import (  # noqa: E402
    RewardConfig,
    certainty_equivalent,
    net_returns,
)
from src.reinforcement.stochastic import ActorCriticConfig, GaussianActorCritic  # noqa: E402
from src.simulation.presets import load_simulation_preset  # noqa: E402
from src.utils.observability import MemoryTracker, configure_logging  # noqa: E402
from src.utils.parallel import parallel_map  # noqa: E402

logger = logging.getLogger("run_ppo_demo")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/ppo_demo.yaml"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--n-paths", type=int, default=None)
    parser.add_argument("--periods", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--skip-ablation", action="store_true")
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
) -> PPODiagnostics:
    """Train every arm, evaluate, and persist a visual report."""
    tracker = MemoryTracker(logger, phase="ppo")
    simulation_cfg = _mapping(cfg.get("simulation"), "simulation")
    expert_cfg = _mapping(cfg.get("expert"), "expert")

    n_assets = int(simulation_cfg["n_assets"])
    periods = int(periods_override or simulation_cfg["T"])
    n_paths = int(n_paths_override or simulation_cfg["n_paths"])
    base_seed = int(simulation_cfg["base_seed"])
    risk_aversion = float(expert_cfg["risk_aversion"])
    transaction_cost = float(expert_cfg["transaction_cost"])

    spec = StateSpec(**(cfg.get("state") or {}))
    preset = load_simulation_preset(str(simulation_cfg["preset"]), n_assets)
    split_cfg = dict(cfg.get("splits") or {})
    split = split_paths(
        n_paths,
        train_fraction=float(split_cfg.get("train_fraction", 0.7)),
        validation_fraction=float(split_cfg.get("validation_fraction", 0.15)),
    )
    constraints = build_constraints(expert_cfg.get("constraints"))

    ppo_settings = dict(cfg.get("ppo") or {})
    if iterations_override is not None:
        ppo_settings["iterations"] = iterations_override
    lambda_e = float((cfg.get("reward") or {}).get("expert_deviation_penalty", 0.0))

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
    test_tape = build_tape(datasets["test"], with_covariances=True)

    def make_policy(seed: int):
        seed_everything(seed)
        return build_policy(
            dict(cfg["policy"]),
            spec=spec,
            n_assets=n_assets,
            n_regimes=paths[0].n_regimes,
            constraints=constraints,
        )

    # ---- Stage 1: behavioural cloning -------------------------------------
    bc_config = BCConfig(**(cfg.get("behavioural_cloning") or {}))
    _log("stage 1: behavioural cloning")
    cloned = make_policy(bc_config.seed)
    BehaviouralCloningTrainer(cloned, bc_config).fit(
        datasets["train"], datasets["validation"]
    )
    reference = deepcopy(cloned)

    actor_config = ActorCriticConfig(
        **(
            (cfg.get("actor_critic") or {})
            | {
                "value_hidden_dimensions": tuple(
                    (cfg.get("actor_critic") or {}).get(
                        "value_hidden_dimensions", (128, 64)
                    )
                )
            }
        )
    )

    # ---- Stage 4a: hybrid, PPO from the imitation warm start --------------
    _log("stage 4a: hybrid PPO (imitation-initialised)")
    hybrid_reward = RewardConfig(
        risk_aversion=risk_aversion,
        transaction_cost=transaction_cost,
        expert_deviation_penalty=lambda_e,
    )
    hybrid = PPOTrainer(
        GaussianActorCritic(deepcopy(cloned), config=actor_config),
        hybrid_reward,
        config=PPOConfig(**ppo_settings),
    )
    hybrid_history = hybrid.fit(datasets["train"], datasets["validation"])
    tracker.sample("hybrid-fitted")
    _log(f"  hybrid: best CE {hybrid_history.best_certainty_equivalent:+.4e}")

    # ---- Stage 4b: pure PPO, no imitation and no expert in the reward -----
    _log("stage 4b: pure PPO (random initialisation, lambda_E = 0)")
    pure = PPOTrainer(
        GaussianActorCritic(make_policy(bc_config.seed + 1), config=actor_config),
        RewardConfig(
            risk_aversion=risk_aversion,
            transaction_cost=transaction_cost,
            expert_deviation_penalty=0.0,
        ),
        config=PPOConfig(**ppo_settings),
    )
    pure_history = pure.fit(datasets["train"], datasets["validation"])
    tracker.sample("pure-fitted")
    _log(f"  pure: best CE {pure_history.best_certainty_equivalent:+.4e}")

    # ---- Reported upper baseline on the training objective ----------------
    _log("exact-gradient baseline")
    analytic = AnalyticRewardTrainer(
        deepcopy(cloned), hybrid_reward, config=AnalyticConfig(**(cfg.get("analytic") or {}))
    )
    analytic_history = analytic.fit(datasets["train"], datasets["validation"])
    tracker.sample("analytic-fitted")
    train_tape = build_tape(datasets["train"], with_covariances=True)

    # ---- lambda_E ablation on the hybrid arm ------------------------------
    ablation: dict[str, Any] = {}
    if not skip_ablation:
        for value in (cfg.get("ablations") or {}).get("expert_deviation_penalty", []):
            if float(value) == lambda_e:
                ablation[f"{float(value):g}"] = {
                    "best_certainty_equivalent": hybrid_history.best_certainty_equivalent,
                    "records": [r.to_dict() for r in hybrid_history.records],
                }
                continue
            _log(f"ablation: lambda_E = {value}")
            trainer = PPOTrainer(
                GaussianActorCritic(deepcopy(cloned), config=actor_config),
                RewardConfig(
                    risk_aversion=risk_aversion,
                    transaction_cost=transaction_cost,
                    expert_deviation_penalty=float(value),
                ),
                config=PPOConfig(**ppo_settings),
            )
            history = trainer.fit(datasets["train"], datasets["validation"])
            ablation[f"{float(value):g}"] = {
                "best_certainty_equivalent": history.best_certainty_equivalent,
                "test": trainer.evaluate(test_tape),
            }

    # ---- Comparison on held-out paths -------------------------------------
    _log("evaluating on held-out paths")
    expert_net = net_returns(
        test_tape.expert_actions,
        test_tape.expert_anchors,
        test_tape.realised_returns,
        transaction_cost,
    )
    expert_ce = certainty_equivalent(expert_net, risk_aversion)
    comparison = {
        "expert": {
            "certainty_equivalent": expert_ce,
            "mean_net_return": float(expert_net.mean()),
            "annualised_net_return": float(expert_net.mean()) * 252.0,
            "mean_turnover": float(
                (test_tape.expert_actions - test_tape.expert_anchors)
                .abs()
                .sum(dim=-1)
                .mean()
            ),
            "feasibility_violation": 0.0,
        },
        "behavioural_cloning": PPOTrainer(
            GaussianActorCritic(reference, config=actor_config), hybrid_reward
        ).evaluate(test_tape),
        "hybrid_ppo": hybrid.evaluate(test_tape),
        "pure_ppo": pure.evaluate(test_tape),
        "analytic_upper_baseline": analytic.evaluate(test_tape),
    }
    for name, values in comparison.items():
        values["utility_loss"] = expert_ce - values["certainty_equivalent"]
        del name

    diagnostics = evaluate_ppo(
        hybrid,
        hybrid_history,
        datasets["test"],
        reference,
        thresholds=PPODiagnosticThresholds(**(cfg.get("diagnostics") or {})),
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
        "ppo": hybrid.describe(),
        "analytic": analytic.describe()
        | {
            "train_certainty_equivalent": analytic.evaluate(train_tape)[
                "certainty_equivalent"
            ],
            "caveat": (
                "Upper baseline on the TRAINING objective only. It "
                "differentiates through the realised return path, so it tunes "
                "to that sample path; measured here it improves training "
                "certainty equivalent while making held-out utility worse. "
                "PPO's sampling noise acts as a regulariser it does not have."
            ),
        },
        "histories": {
            "hybrid_ppo": hybrid_history.to_dict(),
            "pure_ppo": pure_history.to_dict(),
            "analytic": analytic_history.to_dict(),
        },
        "expert_deviation_ablation": ablation,
        "comparison": comparison,
        "diagnostics": diagnostics.to_dict(),
    }
    (output_dir / "ppo_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    _write_tables(output_dir, hybrid_history, pure_history, comparison)
    save_dashboard(
        hybrid=hybrid_history,
        pure=pure_history,
        comparison=comparison,
        diagnostics=diagnostics,
        output_path=output_dir / "ppo_dashboard.png",
    )
    tracker.summary()
    return diagnostics


def _write_tables(output_dir, hybrid, pure, comparison) -> None:
    rows = []
    for label, history in (("hybrid_ppo", hybrid), ("pure_ppo", pure)):
        for record in history.records:
            rows.append({"run": label} | record.to_dict())
    pd.DataFrame(rows).to_csv(output_dir / "ppo_iterations.csv", index=False)
    pd.DataFrame(
        [{"policy": name, **values} for name, values in comparison.items()]
    ).to_csv(output_dir / "ppo_comparison.csv", index=False)


def save_dashboard(*, hybrid, pure, comparison, diagnostics, output_path) -> None:
    """Save a six-panel visual audit of the PPO stage."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    fig.suptitle(
        "Weeks 30–33: PPO fine-tuning on the Equation (11) reward",
        fontsize=15,
        fontweight="bold",
    )
    colours = {"hybrid_ppo": "#4c78a8", "pure_ppo": "#e45756"}

    for label, history in (("hybrid_ppo", hybrid), ("pure_ppo", pure)):
        axes[0, 0].plot(
            history.series("iteration"),
            history.series("certainty_equivalent"),
            color=colours[label],
            label=label.replace("_", " "),
        )
        axes[0, 1].plot(
            history.series("iteration"),
            history.series("mean_turnover"),
            color=colours[label],
            label=label.replace("_", " "),
        )
    axes[0, 0].axhline(
        comparison["expert"]["certainty_equivalent"],
        color="#222222",
        linestyle="--",
        label="expert",
    )
    axes[0, 0].set_title("Certainty-equivalent utility (Equation 16)")
    axes[0, 0].set_xlabel("PPO iteration")
    axes[0, 0].set_ylabel("CE")
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].axhline(
        comparison["expert"]["mean_turnover"],
        color="#222222",
        linestyle="--",
        label="expert",
    )
    axes[0, 1].set_title("Turnover, reported not rewarded")
    axes[0, 1].set_xlabel("PPO iteration")
    axes[0, 1].set_ylabel("Mean $\\|w_t-w_{t-1}\\|_1$")
    axes[0, 1].legend(fontsize=8)

    names = list(comparison)
    values = [comparison[n]["certainty_equivalent"] for n in names]
    axes[0, 2].barh(
        [n.replace("_", "\n") for n in names],
        values,
        color=["#222222", "#bbbbbb", "#4c78a8", "#e45756", "#f2cf5b"][: len(names)],
    )
    axes[0, 2].invert_yaxis()
    axes[0, 2].tick_params(axis="y", labelsize=7)
    axes[0, 2].set_title("Held-out certainty equivalent")
    axes[0, 2].set_xlabel("CE (higher is better)")

    for label, history in (("hybrid_ppo", hybrid), ("pure_ppo", pure)):
        axes[1, 0].plot(
            history.series("iteration"),
            history.series("approximate_kl"),
            color=colours[label],
            label=label.replace("_", " "),
        )
    axes[1, 0].set_title("Approximate KL per update (PPO's trust region)")
    axes[1, 0].set_xlabel("PPO iteration")
    axes[1, 0].set_ylabel("KL")
    axes[1, 0].legend(fontsize=8)

    for label, history in (("hybrid_ppo", hybrid), ("pure_ppo", pure)):
        axes[1, 1].plot(
            history.series("iteration"),
            history.series("explained_variance"),
            color=colours[label],
            label=label.replace("_", " "),
        )
    axes[1, 1].set_title("Value-head explained variance")
    axes[1, 1].set_xlabel("PPO iteration")
    axes[1, 1].set_ylabel("Explained variance")
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
        f"max constraint violation across all arms: "
        f"{diagnostics.feasibility_violation:.1e}",
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
        cfg.get("paths", {}).get("output_dir", "data/ppo_demo")
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

    logger.info("Wrote PPO demo to %s", output_dir)
    logger.info("acceptance checks: %s", "PASS" if diagnostics.passed else "FAIL")
    for name, passed in diagnostics.checks.items():
        logger.info("  %s: %s", name, "PASS" if passed else "FAIL")
    logger.info("dashboard: %s", output_dir / "ppo_dashboard.png")
    logger.info("report: %s", output_dir / "ppo_report.json")
    if not diagnostics.passed:
        raise SystemExit("PPO acceptance checks failed; inspect ppo_report.json.")


if __name__ == "__main__":
    main()
