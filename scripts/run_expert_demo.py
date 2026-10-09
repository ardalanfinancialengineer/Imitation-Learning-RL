"""Run and visualise the completed Weeks 9-13 rolling-horizon Markowitz expert.

The script simulates one canonical three-regime path, rolls the friction-aware
H=21 expert across it, runs single-parameter ablations over transaction costs,
planning horizons, risk aversion, and covariance estimators, evaluates every
acceptance check, and writes a dashboard plus machine-readable evidence.

It exits with a non-zero status if any acceptance check fails.
"""

from __future__ import annotations

import argparse
import json
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

from src.expert.config import build_expert  # noqa: E402
from src.expert.diagnostics import (  # noqa: E402
    ExpertDiagnostics,
    ExpertDiagnosticThresholds,
    equal_weight_trajectory,
    evaluate_expert,
)
from src.expert.markowitz import ExpertTrajectory  # noqa: E402
from src.simulation.presets import load_simulation_preset  # noqa: E402

ANNUALISATION = 252.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/expert_demo.yaml"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--periods", type=int, default=None)
    parser.add_argument("--n-assets", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--skip-ablations",
        action="store_true",
        help="Run only the baseline and the cost grid required by the checks.",
    )
    return parser.parse_args()


def run_demo(
    cfg: dict[str, Any],
    output_dir: Path,
    *,
    periods_override: int | None = None,
    n_assets_override: int | None = None,
    seed_override: int | None = None,
    skip_ablations: bool = False,
) -> ExpertDiagnostics:
    """Simulate, run the expert, evaluate acceptance checks, and persist a report."""
    simulation_cfg = _mapping(cfg.get("simulation"), "simulation")
    expert_cfg = _mapping(cfg.get("expert"), "expert")
    ablation_cfg = dict(cfg.get("ablations") or {})

    n_assets = int(n_assets_override or simulation_cfg["n_assets"])
    periods = int(periods_override or simulation_cfg["T"])
    seed = int(simulation_cfg["seed"] if seed_override is None else seed_override)

    preset = load_simulation_preset(str(simulation_cfg["preset"]), n_assets)
    returns, regimes = preset.build_simulator().simulate(T=periods, seed=seed)
    values = returns.to_numpy(dtype=float)

    baseline_expert = build_expert(expert_cfg)
    started = time.perf_counter()
    baseline = baseline_expert.run(values)
    baseline_seconds = time.perf_counter() - started

    cost_grid = _as_float_list(
        ablation_cfg.get("transaction_costs", [0.0, expert_cfg["transaction_cost"]])
    )
    # Ablations that leave the moment estimator untouched reuse the prepared
    # forecaster, so the HMM is fitted once for the whole demo.
    shared = baseline_expert.forecaster
    cost_runs = _cost_ablation(expert_cfg, values, cost_grid, baseline_expert, baseline, shared)

    horizon_runs: dict[int, ExpertTrajectory] = {}
    gamma_runs: dict[float, ExpertTrajectory] = {}
    covariance_runs: dict[str, ExpertTrajectory] = {}
    if not skip_ablations:
        horizon_runs = {
            int(horizon): _run_variant(
                expert_cfg, values, forecaster=shared, horizon=int(horizon)
            )
            for horizon in ablation_cfg.get("horizons", [])
        }
        gamma_runs = {
            float(gamma): _run_variant(
                expert_cfg, values, forecaster=shared, risk_aversion=float(gamma)
            )
            for gamma in ablation_cfg.get("risk_aversions", [])
        }
        covariance_runs = {
            str(method): _run_variant(expert_cfg, values, covariance=str(method))
            for method in ablation_cfg.get("covariance_estimators", [])
        }

    diagnostics = evaluate_expert(
        baseline_expert,
        values,
        baseline,
        turnover_by_cost={
            cost: float(run.turnover.mean()) for cost, run in cost_runs.items()
        },
        horizon_turnover={
            horizon: float(run.turnover.mean()) for horizon, run in horizon_runs.items()
        },
        regimes=regimes,
        thresholds=ExpertDiagnosticThresholds(**(cfg.get("diagnostics") or {})),
    )

    benchmark = equal_weight_trajectory(
        values,
        baseline.decision_times,
        baseline_expert.transaction_cost,
        baseline_expert.constraints,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_tables(output_dir, returns, baseline, cost_runs, horizon_runs, gamma_runs)
    report = {
        "simulation": {
            "preset": preset.name,
            "periods": periods,
            "n_assets": n_assets,
            "seed": seed,
            "regime_names": list(preset.regime_names),
        },
        "expert": baseline_expert.describe(),
        "baseline": baseline.summary()
        | {"wall_clock_seconds": baseline_seconds}
        | _performance(baseline.net_returns),
        "equal_weight_benchmark": _performance(benchmark["net_returns"]),
        "ablations": {
            "transaction_costs": {
                f"{cost:.4f}": _ablation_row(run) for cost, run in cost_runs.items()
            },
            "horizons": {
                str(horizon): _ablation_row(run) for horizon, run in horizon_runs.items()
            },
            "risk_aversions": {
                f"{gamma:g}": _ablation_row(run) for gamma, run in gamma_runs.items()
            },
            "covariance_estimators": {
                name: _ablation_row(run) for name, run in covariance_runs.items()
            },
        },
        "diagnostics": diagnostics.to_dict(),
    }
    (output_dir / "expert_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    save_dashboard(
        returns=returns,
        regimes=regimes,
        regime_names=preset.regime_names,
        trajectory=baseline,
        benchmark=benchmark,
        expert=baseline_expert,
        cost_runs=cost_runs,
        horizon_runs=horizon_runs,
        diagnostics=diagnostics,
        output_path=output_dir / "expert_dashboard.png",
    )
    return diagnostics


def _cost_ablation(
    expert_cfg: dict[str, Any],
    values: np.ndarray,
    cost_grid: list[float],
    baseline_expert,
    baseline: ExpertTrajectory,
    forecaster,
) -> dict[float, ExpertTrajectory]:
    """Re-run the expert for each cost level, reusing the baseline where possible."""
    runs: dict[float, ExpertTrajectory] = {}
    for cost in sorted(cost_grid):
        if np.isclose(cost, baseline_expert.transaction_cost):
            runs[cost] = baseline
            continue
        runs[cost] = _run_variant(
            expert_cfg, values, forecaster=forecaster, transaction_cost=cost
        )
    return runs


def _run_variant(expert_cfg: dict[str, Any], values: np.ndarray, **overrides) -> ExpertTrajectory:
    """Run the expert with exactly one changed parameter."""
    expert = build_expert(expert_cfg, enforce_cost_range=False, **overrides)
    return expert.run(values)


def _ablation_row(trajectory: ExpertTrajectory) -> dict[str, Any]:
    return {
        "mean_turnover": float(trajectory.turnover.mean()),
        "median_solve_seconds": float(np.median(trajectory.solve_seconds)),
        "fallback_rate": trajectory.fallback_rate,
    } | _performance(trajectory.net_returns)


def _performance(net_returns: np.ndarray) -> dict[str, float]:
    returns = np.asarray(net_returns, dtype=float)
    volatility = float(returns.std(ddof=1)) if returns.size > 1 else 0.0
    equity = np.cumprod(1.0 + returns)
    drawdown = equity / np.maximum.accumulate(equity) - 1.0
    return {
        "annualised_return": float(returns.mean() * ANNUALISATION),
        "annualised_volatility": float(volatility * np.sqrt(ANNUALISATION)),
        "sharpe_ratio": float(
            returns.mean() / volatility * np.sqrt(ANNUALISATION)
        )
        if volatility > 0.0
        else 0.0,
        "max_drawdown": float(drawdown.min()),
        "total_return": float(equity[-1] - 1.0),
    }


def _write_tables(
    output_dir: Path,
    returns: pd.DataFrame,
    baseline: ExpertTrajectory,
    cost_runs: dict[float, ExpertTrajectory],
    horizon_runs: dict[int, ExpertTrajectory],
    gamma_runs: dict[float, ExpertTrajectory],
) -> None:
    weights = pd.DataFrame(
        baseline.weights, columns=[f"weight_{name}" for name in returns.columns]
    )
    weights.insert(0, "decision_t", baseline.decision_times)
    weights.insert(1, "turnover", baseline.turnover)
    weights.insert(2, "gross_return", baseline.gross_returns)
    weights.insert(3, "net_return", baseline.net_returns)
    weights.insert(4, "solve_seconds", baseline.solve_seconds)
    weights.insert(5, "status", list(baseline.statuses))
    weights.to_csv(output_dir / "expert_decisions.csv", index=False)

    rows = []
    for label, runs in (
        ("transaction_cost", cost_runs),
        ("horizon", horizon_runs),
        ("risk_aversion", gamma_runs),
    ):
        for setting, run in runs.items():
            rows.append(
                {"ablation": label, "setting": setting}
                | {
                    "mean_turnover": float(run.turnover.mean()),
                    "median_solve_seconds": float(np.median(run.solve_seconds)),
                }
                | _performance(run.net_returns)
            )
    if rows:
        pd.DataFrame(rows).to_csv(output_dir / "expert_ablations.csv", index=False)


def save_dashboard(
    *,
    returns: pd.DataFrame,
    regimes: np.ndarray,
    regime_names: tuple[str, ...],
    trajectory: ExpertTrajectory,
    benchmark: dict[str, np.ndarray],
    expert,
    cost_runs: dict[float, ExpertTrajectory],
    horizon_runs: dict[int, ExpertTrajectory],
    diagnostics: ExpertDiagnostics,
    output_path: Path,
) -> None:
    """Save a six-panel visual audit of one expert run."""
    regime_colors = ("#4c78a8", "#f2cf5b", "#e45756")
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    fig.suptitle(
        "Weeks 9–13: Rolling-Horizon Markowitz Expert "
        f"(H={expert.horizon}, γ={expert.risk_aversion:g}, "
        f"c={expert.transaction_cost:.4f})",
        fontsize=15,
        fontweight="bold",
    )

    times = trajectory.decision_times
    expert_equity = np.cumprod(1.0 + trajectory.net_returns) - 1.0
    benchmark_equity = np.cumprod(1.0 + benchmark["net_returns"]) - 1.0
    _shade_regimes(axes[0, 0], times, regimes[times], regime_colors)
    axes[0, 0].plot(times, expert_equity, color="#222222", linewidth=1.4, label="Expert (net)")
    axes[0, 0].plot(
        times, benchmark_equity, color="#e45756", linewidth=1.2, linestyle="--",
        label="Equal weight",
    )
    axes[0, 0].set_title("Cumulative net return")
    axes[0, 0].set_xlabel("Trading day")
    axes[0, 0].set_ylabel("Cumulative return")
    axes[0, 0].legend(loc="upper left", fontsize=8)

    axes[0, 1].stackplot(
        times,
        trajectory.weights.T,
        labels=list(returns.columns),
        colors=plt.cm.viridis(np.linspace(0.05, 0.95, trajectory.weights.shape[1])),
    )
    axes[0, 1].set_title(f"Executed allocation (cap u={expert.constraints.max_weight:g})")
    axes[0, 1].set_xlabel("Trading day")
    axes[0, 1].set_ylabel("Weight")
    axes[0, 1].set_ylim(0.0, 1.0)
    axes[0, 1].legend(loc="upper right", fontsize=6, ncol=2)

    costs = sorted(cost_runs)
    turnovers = [float(cost_runs[cost].turnover.mean()) for cost in costs]
    axes[0, 2].bar(
        [f"{cost:.4f}" for cost in costs], turnovers, color="#3182bd", width=0.55
    )
    axes[0, 2].set_title("Mean turnover falls as cost rises")
    axes[0, 2].set_xlabel("Transaction cost c")
    axes[0, 2].set_ylabel("Mean $\\|w_t - w_{t-1}\\|_1$")
    for index, value in enumerate(turnovers):
        axes[0, 2].text(index, value, f"{value:.3f}", ha="center", va="bottom", fontsize=8)

    accuracy = diagnostics.regime_filter_accuracy
    filter_path = getattr(expert.forecaster, "path", None)
    if filter_path is not None and accuracy is not None:
        # Filtered states are canonically ordered calm -> stressed by the
        # filter itself, so they line up with the preset's regime names.
        probabilities = filter_path.probabilities
        visible = slice(times[0], times[0] + min(600, times.size))
        axis = axes[1, 0]
        for regime in range(probabilities.shape[1]):
            axis.plot(
                np.arange(probabilities.shape[0])[visible],
                probabilities[visible, regime],
                linewidth=1.0,
                color=regime_colors[regime % len(regime_colors)],
                label=regime_names[regime],
            )
        _shade_regimes(
            axis, np.arange(probabilities.shape[0])[visible], regimes[visible], regime_colors
        )
        axis.set_title(f"HMM filtered regime probabilities (accuracy {accuracy:.2f})")
        axis.set_xlabel("Trading day")
        axis.set_ylabel("$\\hat{p}_t(k)$")
        axis.set_ylim(0.0, 1.0)
        axis.legend(loc="upper right", fontsize=7)
    else:
        axes[1, 0].text(
            0.5, 0.5, "No regime filter in this configuration", ha="center", va="center"
        )
        axes[1, 0].set_axis_off()

    milliseconds = trajectory.solve_seconds * 1000.0
    axes[1, 1].hist(milliseconds, bins=40, color="#31a354")
    axes[1, 1].axvline(
        np.median(milliseconds),
        color="#222222",
        linestyle="--",
        label=f"median {np.median(milliseconds):.1f} ms",
    )
    axes[1, 1].set_title("Decision-time cost of re-solving the QP")
    axes[1, 1].set_xlabel("Solve time (ms)")
    axes[1, 1].set_ylabel("Decisions")
    axes[1, 1].legend(fontsize=8)

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


def _shade_regimes(
    axis: plt.Axes,
    times: np.ndarray,
    regimes: np.ndarray,
    colors: tuple[str, ...],
) -> None:
    start = 0
    for end in range(1, regimes.size + 1):
        if end == regimes.size or regimes[end] != regimes[start]:
            axis.axvspan(
                times[start],
                times[end - 1],
                color=colors[int(regimes[start]) % len(colors)],
                alpha=0.16,
                linewidth=0,
            )
            start = end


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"configuration must contain a {name} mapping.")
    return value


def _as_float_list(values: object) -> list[float]:
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError("transaction cost grid must be a non-empty sequence.")
    return [float(value) for value in values]


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError(f"configuration file is empty or invalid: {args.config}")

    output_dir = args.output_dir or Path(
        cfg.get("paths", {}).get("output_dir", "data/expert_demo")
    )
    diagnostics = run_demo(
        cfg,
        output_dir,
        periods_override=args.periods,
        n_assets_override=args.n_assets,
        seed_override=args.seed,
        skip_ablations=args.skip_ablations,
    )

    print(f"Wrote expert demo to {output_dir}")
    print(f"acceptance checks: {'PASS' if diagnostics.passed else 'FAIL'}")
    for name, passed in diagnostics.checks.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    print(f"dashboard: {output_dir / 'expert_dashboard.png'}")
    print(f"report: {output_dir / 'expert_report.json'}")
    if not diagnostics.passed:
        raise SystemExit("Expert acceptance checks failed; inspect expert_report.json.")


if __name__ == "__main__":
    main()
