"""Generate a small synthetic expert-demonstration dataset for smoke testing.

The script stays within the foundational simulator/expert layer: it creates
regime-switching returns, rolls the Markowitz expert, saves arrays for later
inspection, and writes a compact diagnostic plot.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.expert.markowitz import RollingMarkowitzExpert  # noqa: E402
from src.simulation.presets import canonical_three_regime_preset  # noqa: E402
from src.simulation.returns import ReturnSimulator  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/test_dataset"))
    parser.add_argument("--periods", type=int, default=120)
    parser.add_argument("--n-assets", type=int, default=4)
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--risk-aversion", type=float, default=1.0)
    parser.add_argument("--max-weight", type=float, default=0.6)
    return parser.parse_args()


def make_simulator(n_assets: int) -> ReturnSimulator:
    return canonical_three_regime_preset(n_assets).build_simulator()


def shade_regimes(ax: plt.Axes, regimes: np.ndarray) -> None:
    colors = ["#dbeafe", "#fef3c7", "#fee2e2"]
    start = 0
    for t in range(1, len(regimes) + 1):
        if t == len(regimes) or regimes[t] != regimes[start]:
            ax.axvspan(start, t - 1, color=colors[int(regimes[start]) % len(colors)], alpha=0.35)
            start = t


def save_diagnostic_plot(
    returns: pd.DataFrame,
    regimes: np.ndarray,
    actions: np.ndarray,
    window: int,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=False, constrained_layout=True)

    cumulative_returns = (1.0 + returns).cumprod() - 1.0
    cumulative_returns.plot(ax=axes[0], linewidth=1.4)
    shade_regimes(axes[0], regimes)
    axes[0].set_title("Simulated cumulative asset returns")
    axes[0].set_ylabel("Cumulative return")
    axes[0].legend(loc="upper left", ncols=2, fontsize=8)

    axes[1].step(np.arange(regimes.size), regimes, where="post", color="#111827")
    axes[1].set_title("Latent regime path")
    axes[1].set_ylabel("Regime")
    axes[1].set_yticks(sorted(set(regimes.tolist())))

    decision_times = np.arange(window, window + actions.shape[0])
    axes[2].stackplot(decision_times, actions.T, labels=returns.columns)
    axes[2].set_title("Rolling Markowitz expert weights")
    axes[2].set_xlabel("Time")
    axes[2].set_ylabel("Weight")
    axes[2].set_ylim(0.0, 1.0)
    axes[2].legend(loc="upper left", ncols=2, fontsize=8)

    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.periods <= args.window:
        raise ValueError("--periods must be greater than --window.")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    simulator = make_simulator(args.n_assets)
    returns, regimes = simulator.simulate(T=args.periods, seed=args.seed)

    expert = RollingMarkowitzExpert(
        window=args.window,
        risk_aversion=args.risk_aversion,
        long_only=True,
        max_weight=args.max_weight,
    )
    observations, actions = expert.generate_demonstrations(returns)

    returns.to_csv(output_dir / "returns.csv", index=False)
    pd.DataFrame({"t": np.arange(regimes.size), "regime": regimes}).to_csv(
        output_dir / "regimes.csv",
        index=False,
    )
    pd.DataFrame(
        actions,
        columns=[f"weight_{name}" for name in returns.columns],
    ).assign(decision_t=np.arange(args.window, args.periods)).to_csv(
        output_dir / "expert_actions.csv",
        index=False,
    )
    np.savez_compressed(
        output_dir / "demonstrations.npz",
        observations=observations,
        actions=actions,
        regimes=regimes,
        returns=returns.to_numpy(),
        decision_times=np.arange(args.window, args.periods),
    )

    metadata = {
        "seed": args.seed,
        "periods": args.periods,
        "n_assets": args.n_assets,
        "window": args.window,
        "risk_aversion": args.risk_aversion,
        "max_weight": args.max_weight,
        "returns_shape": list(returns.shape),
        "regimes_shape": list(regimes.shape),
        "observations_shape": list(observations.shape),
        "actions_shape": list(actions.shape),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    plot_path = output_dir / "diagnostics.png"
    save_diagnostic_plot(returns, regimes, actions, args.window, plot_path)

    print(f"Wrote test dataset to {output_dir}")
    print(f"returns: {returns.shape}")
    print(f"regimes: {regimes.shape}")
    print(f"observations: {observations.shape}")
    print(f"actions: {actions.shape}")
    print(f"plot: {plot_path}")


if __name__ == "__main__":
    main()
