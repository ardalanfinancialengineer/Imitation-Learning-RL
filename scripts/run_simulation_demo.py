"""Run and visualise the completed Weeks 5-8 simulation environment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.simulation.costs import (  # noqa: E402
    ProportionalTransactionCost,
    validate_transaction_cost_grid,
)
from src.simulation.diagnostics import (  # noqa: E402
    DiagnosticThresholds,
    SimulationDiagnostics,
    evaluate_simulation,
)
from src.simulation.presets import (  # noqa: E402
    RegimeSimulationPreset,
    load_simulation_preset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/simulation_demo.yaml"),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--periods", type=int, default=None)
    parser.add_argument("--n-assets", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def run_demo(
    cfg: dict,
    output_dir: Path,
    *,
    periods_override: int | None = None,
    n_assets_override: int | None = None,
    seed_override: int | None = None,
) -> SimulationDiagnostics:
    """Generate returns, calculate diagnostics, and persist a visual report."""
    if not isinstance(cfg, dict):
        raise ValueError("configuration must be a mapping.")
    simulation_cfg = cfg.get("simulation")
    if not isinstance(simulation_cfg, dict):
        raise ValueError("configuration must contain a simulation mapping.")

    periods = _positive_int(
        periods_override if periods_override is not None else simulation_cfg.get("T"),
        "simulation.T",
    )
    n_assets = _positive_int(
        n_assets_override
        if n_assets_override is not None
        else simulation_cfg.get("n_assets"),
        "simulation.n_assets",
    )
    seed = _integer(
        seed_override if seed_override is not None else simulation_cfg.get("seed"),
        "simulation.seed",
    )
    preset = load_simulation_preset(str(simulation_cfg.get("preset")), n_assets)
    transaction_costs = validate_transaction_cost_grid(
        simulation_cfg.get("transaction_costs", [])
    )
    thresholds = DiagnosticThresholds(**cfg.get("diagnostics", {}))
    _validate_final_scale(cfg.get("final_scale"))

    simulator = preset.build_simulator()
    returns, regimes = simulator.simulate(T=periods, seed=seed)
    diagnostics = evaluate_simulation(
        returns,
        regimes,
        regime_names=preset.regime_names,
        transition_matrix=preset.transition_matrix,
        initial_distribution=preset.initial_distribution,
        target_means=preset.regime_means,
        target_covariances=preset.regime_covariances,
        thresholds=thresholds,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    returns.to_csv(output_dir / "returns.csv", index=False)
    pd.DataFrame(
        {
            "t": np.arange(periods),
            "regime": regimes,
            "regime_name": [preset.regime_names[index] for index in regimes],
        }
    ).to_csv(output_dir / "regimes.csv", index=False)

    transaction_cost_frame = _transaction_cost_examples(transaction_costs)
    transaction_cost_frame.to_csv(
        output_dir / "transaction_cost_examples.csv",
        index=False,
    )
    report = {
        "simulation": {
            "preset": preset.name,
            "periods": periods,
            "n_assets": n_assets,
            "seed": seed,
            "regime_names": list(preset.regime_names),
            "transaction_costs": list(transaction_costs),
            "transaction_cost_formula": "c * ||w_t - w_(t-1)||_1",
        },
        "final_scale": cfg["final_scale"],
        "diagnostics": diagnostics.to_dict(),
    }
    (output_dir / "diagnostics.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    save_dashboard(
        returns=returns,
        regimes=regimes,
        preset=preset,
        diagnostics=diagnostics,
        transaction_costs=transaction_costs,
        output_path=output_dir / "simulation_dashboard.png",
    )
    return diagnostics


def save_dashboard(
    *,
    returns: pd.DataFrame,
    regimes: np.ndarray,
    preset: RegimeSimulationPreset,
    diagnostics: SimulationDiagnostics,
    transaction_costs: tuple[float, ...],
    output_path: Path,
) -> None:
    """Save a six-panel visual audit of one simulation run."""
    colors = ("#4c78a8", "#f2cf5b", "#e45756")
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    fig.suptitle(
        "Weeks 5–8: Three-Regime Market Simulation",
        fontsize=16,
        fontweight="bold",
    )

    visible_periods = min(500, len(returns))
    equal_weight_returns = returns.iloc[:visible_periods].mean(axis=1)
    cumulative = (1.0 + equal_weight_returns).cumprod() - 1.0
    axes[0, 0].plot(cumulative.index, cumulative.values, color="#222222", linewidth=1.4)
    _shade_regimes(
        axes[0, 0],
        regimes[:visible_periods],
        colors,
    )
    axes[0, 0].set_title(f"Equal-weight cumulative return (first {visible_periods} days)")
    axes[0, 0].set_xlabel("Trading day")
    axes[0, 0].set_ylabel("Cumulative return")

    positions = np.arange(preset.n_regimes)
    width = 0.36
    axes[0, 1].bar(
        positions - width / 2,
        diagnostics.expected_frequencies,
        width,
        label="Expected",
        color="#9ecae9",
    )
    axes[0, 1].bar(
        positions + width / 2,
        diagnostics.empirical_frequencies,
        width,
        label="Empirical",
        color="#3182bd",
    )
    axes[0, 1].set_title("Regime occupancy")
    axes[0, 1].set_ylabel("Fraction of observations")
    axes[0, 1].set_xticks(positions, preset.regime_names)
    axes[0, 1].legend()

    target_volatility = np.sqrt(
        np.diagonal(preset.regime_covariances, axis1=1, axis2=2)
    ).mean(axis=1) * np.sqrt(252.0)
    empirical_volatility = np.sqrt(
        np.diagonal(diagnostics.empirical_covariances, axis1=1, axis2=2)
    ).mean(axis=1) * np.sqrt(252.0)
    axes[0, 2].bar(
        positions - width / 2,
        target_volatility,
        width,
        label="Target",
        color="#a1d99b",
    )
    axes[0, 2].bar(
        positions + width / 2,
        empirical_volatility,
        width,
        label="Empirical",
        color="#31a354",
    )
    axes[0, 2].set_title("Average annualised asset volatility")
    axes[0, 2].set_ylabel("Volatility")
    axes[0, 2].set_xticks(positions, preset.regime_names)
    axes[0, 2].legend()

    axes[1, 0].bar(
        positions - width / 2,
        np.diag(preset.transition_matrix),
        width,
        label="Target",
        color="#fdae6b",
    )
    axes[1, 0].bar(
        positions + width / 2,
        np.diag(diagnostics.empirical_transition_matrix),
        width,
        label="Empirical",
        color="#e6550d",
    )
    axes[1, 0].set_title("Regime persistence")
    axes[1, 0].set_ylabel("Self-transition probability")
    axes[1, 0].set_xticks(positions, preset.regime_names)
    axes[1, 0].set_ylim(0.0, 1.0)
    axes[1, 0].legend()

    turnovers = np.linspace(0.0, 2.0, 50)
    for rate in transaction_costs:
        model = ProportionalTransactionCost(rate)
        costs = model.rate * turnovers * 10_000.0
        axes[1, 1].plot(turnovers, costs, label=f"c = {rate:.4f}")
    axes[1, 1].set_title("Proportional transaction costs")
    axes[1, 1].set_xlabel("Full L1 turnover")
    axes[1, 1].set_ylabel("Cost (basis points)")
    axes[1, 1].legend()

    check_names = [name.replace("_", " ").title() for name in diagnostics.checks]
    check_values = [1 if passed else 0 for passed in diagnostics.checks.values()]
    check_colors = ["#31a354" if passed else "#e45756" for passed in check_values]
    axes[1, 2].barh(check_names, np.ones(len(check_names)), color=check_colors)
    axes[1, 2].set_xlim(0.0, 1.0)
    axes[1, 2].set_xticks([])
    axes[1, 2].set_title(
        "Diagnostics: " + ("PASS" if diagnostics.passed else "FAIL"),
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


def _transaction_cost_examples(
    transaction_costs: tuple[float, ...],
) -> pd.DataFrame:
    rows = []
    for turnover in (0.0, 0.25, 0.50, 1.0, 2.0):
        for rate in transaction_costs:
            rows.append(
                {
                    "transaction_cost_rate": rate,
                    "l1_turnover": turnover,
                    "cost": rate * turnover,
                    "cost_basis_points": rate * turnover * 10_000.0,
                }
            )
    return pd.DataFrame(rows)


def _shade_regimes(
    axis: plt.Axes,
    regimes: np.ndarray,
    colors: tuple[str, ...],
) -> None:
    start = 0
    for end in range(1, regimes.size + 1):
        if end == regimes.size or regimes[end] != regimes[start]:
            axis.axvspan(
                start,
                end - 1,
                color=colors[int(regimes[start])],
                alpha=0.18,
                linewidth=0,
            )
            start = end


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer.")
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return parsed


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer.")
    return int(value)


def _validate_final_scale(final_scale: object) -> None:
    if not isinstance(final_scale, dict):
        raise ValueError("configuration must contain a final_scale mapping.")
    n_assets = final_scale.get("n_assets")
    if n_assets != [50, 100, 200]:
        raise ValueError("final_scale.n_assets must be [50, 100, 200].")
    if _positive_int(final_scale.get("T"), "final_scale.T") != 10_000:
        raise ValueError("final_scale.T must be 10000.")


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError(f"configuration file is empty or invalid: {args.config}")
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = Path(cfg.get("paths", {}).get("output_dir", "data/simulation_demo"))

    diagnostics = run_demo(
        cfg,
        output_dir,
        periods_override=args.periods,
        n_assets_override=args.n_assets,
        seed_override=args.seed,
    )
    print(f"Wrote simulation demo to {output_dir}")
    print(f"diagnostics: {'PASS' if diagnostics.passed else 'FAIL'}")
    for name, passed in diagnostics.checks.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    print(f"dashboard: {output_dir / 'simulation_dashboard.png'}")
    print(f"report: {output_dir / 'diagnostics.json'}")
    if not diagnostics.passed:
        raise SystemExit("Simulation diagnostics failed; inspect diagnostics.json.")


if __name__ == "__main__":
    main()
