"""Measure expert decision-time cost at the exposé's final asset universes.

Research question 5 asks what inference-time benefit a learned policy offers
relative to re-solving the rolling-horizon quadratic programme at every
rebalancing date. That comparison is only meaningful against a measured
baseline, so this script records per-decision solve times for
``N in {50, 100, 200}`` under documented conditions, and checks that every
allocation produced at those sizes is feasible.

A short decision count is used per universe: the quantity of interest is the
per-decision cost, not a full 10,000-step path.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.expert.config import build_expert  # noqa: E402
from src.simulation.presets import load_simulation_preset  # noqa: E402

FINAL_UNIVERSES = (50, 100, 200)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/expert_demo.yaml"))
    parser.add_argument(
        "--output", type=Path, default=Path("data/expert_demo/expert_scale_check.json")
    )
    parser.add_argument("--decisions", type=int, default=25)
    parser.add_argument("--periods", type=int, default=400)
    return parser.parse_args()


def run_scale_check(
    cfg: dict, *, decisions: int = 25, periods: int = 400
) -> list[dict[str, object]]:
    """Run a few decisions at each final universe size and time them."""
    simulation_cfg = cfg["simulation"]
    expert_cfg = dict(cfg["expert"])
    # The HMM is not the object under test here and needs a long burn-in, so the
    # scale check isolates the optimiser with rolling shrinkage moments.
    expert_cfg["moments"] = {
        "estimator": "rolling",
        "covariance": "ledoit_wolf",
        "window": int((cfg["expert"].get("moments") or {}).get("window", 60)),
    }
    seed = int(simulation_cfg["seed"])
    preset_name = str(simulation_cfg["preset"])

    results: list[dict[str, object]] = []
    for n_assets in FINAL_UNIVERSES:
        # The per-asset cap must stay feasible for a fully invested portfolio.
        constraints = dict(expert_cfg.get("constraints") or {})
        constraints["max_weight"] = max(
            float(constraints.get("max_weight", 1.0)), 4.0 / n_assets
        )
        sized_cfg = expert_cfg | {"constraints": constraints}

        preset = load_simulation_preset(preset_name, n_assets)
        returns, _ = preset.build_simulator().simulate(T=periods, seed=seed)
        expert = build_expert(sized_cfg)

        started = perf_counter()
        trajectory = expert.run(returns.to_numpy(dtype=float), max_decisions=decisions)
        elapsed = perf_counter() - started

        results.append(
            {
                "n_assets": n_assets,
                "horizon": expert.horizon,
                "decision_variables": expert.horizon * n_assets,
                "decisions": trajectory.n_decisions,
                "median_solve_seconds": round(
                    float(np.median(trajectory.solve_seconds)), 5
                ),
                "mean_solve_seconds": round(float(trajectory.solve_seconds.mean()), 5),
                "max_solve_seconds": round(float(trajectory.solve_seconds.max()), 5),
                "total_wall_clock_seconds": round(elapsed, 3),
                "fallback_rate": trajectory.fallback_rate,
                "all_weights_feasible": bool(
                    expert.constraints.is_feasible(trajectory.weights)
                ),
                "max_weight": constraints["max_weight"],
            }
        )
    return results


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError(f"configuration file is empty or invalid: {args.config}")

    results = run_scale_check(cfg, decisions=args.decisions, periods=args.periods)
    report = {
        "environment": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    for result in results:
        print(
            f"N={result['n_assets']:>3} "
            f"({result['decision_variables']} decision variables): "
            f"median {float(result['median_solve_seconds']) * 1000:.1f} ms/decision, "
            f"feasible={result['all_weights_feasible']}"
        )
    print(f"Wrote expert scale report to {args.output}")
    if not all(result["all_weights_feasible"] for result in results):
        raise SystemExit("Infeasible allocations at final scale.")


if __name__ == "__main__":
    main()
