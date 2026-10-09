"""Generate synthetic returns and roll the Markowitz expert to collect demonstrations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.expert.markowitz import RollingMarkowitzExpert  # noqa: E402
from src.simulation.costs import validate_transaction_cost_grid  # noqa: E402
from src.simulation.presets import (  # noqa: E402
    constant_correlation_covariances,
    load_simulation_preset,
)
from src.simulation.regimes import RegimeSwitchingModel  # noqa: E402
from src.simulation.returns import ReturnSimulator  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def _load_regime_covariances(simulation_cfg: dict) -> np.ndarray:
    if "regime_covs" in simulation_cfg:
        return np.asarray(simulation_cfg["regime_covs"], dtype=float)
    if "regime_covariances" in simulation_cfg:
        return np.asarray(simulation_cfg["regime_covariances"], dtype=float)

    covariance_cfg = simulation_cfg.get("covariance")
    if not isinstance(covariance_cfg, dict):
        raise ValueError(
            "simulation must define either regime_covs/regime_covariances "
            "or a covariance construction block."
        )

    method = covariance_cfg.get("method", "constant_correlation")
    if method != "constant_correlation":
        raise ValueError(f"unsupported covariance construction method: {method!r}.")

    return constant_correlation_covariances(
        volatilities=np.asarray(covariance_cfg["volatilities"], dtype=float),
        correlations=np.asarray(covariance_cfg["correlations"], dtype=float),
    )


def build_simulator(simulation_cfg: dict) -> ReturnSimulator:
    n_assets = int(simulation_cfg["n_assets"])
    preset_name = simulation_cfg.get("preset")
    if preset_name is not None:
        explicit_keys = {
            "n_regimes",
            "transition_matrix",
            "initial_distribution",
            "regime_means",
            "regime_covs",
            "regime_covariances",
            "covariance",
        }
        conflicts = sorted(explicit_keys.intersection(simulation_cfg))
        if conflicts:
            raise ValueError(
                "simulation preset cannot be combined with explicit regime parameters: "
                f"{conflicts}."
            )
        return load_simulation_preset(str(preset_name), n_assets).build_simulator()

    n_regimes = int(simulation_cfg["n_regimes"])

    regime_model = RegimeSwitchingModel(
        n_regimes=n_regimes,
        transition_matrix=np.asarray(simulation_cfg["transition_matrix"], dtype=float),
        initial_distribution=np.asarray(simulation_cfg["initial_distribution"], dtype=float),
    )

    return ReturnSimulator(
        n_assets=n_assets,
        regime_model=regime_model,
        regime_means=np.asarray(simulation_cfg["regime_means"], dtype=float),
        regime_covs=_load_regime_covariances(simulation_cfg),
    )


def build_expert(expert_cfg: dict) -> RollingMarkowitzExpert:
    return RollingMarkowitzExpert(
        window=int(expert_cfg["window"]),
        risk_aversion=float(expert_cfg.get("risk_aversion", 1.0)),
        long_only=bool(expert_cfg.get("long_only", True)),
        max_weight=float(expert_cfg.get("max_weight", 1.0)),
    )


def save_outputs(
    output_dir: Path,
    returns: pd.DataFrame,
    regimes: np.ndarray,
    observations: np.ndarray,
    actions: np.ndarray,
    decision_times: np.ndarray,
    cfg: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    simulation_cfg = cfg["simulation"]
    preset_name = simulation_cfg.get("preset")
    if preset_name is not None:
        n_regimes = load_simulation_preset(
            str(preset_name),
            int(simulation_cfg["n_assets"]),
        ).n_regimes
    else:
        n_regimes = int(simulation_cfg["n_regimes"])
    transaction_costs = validate_transaction_cost_grid(
        simulation_cfg.get("transaction_costs", [0.001])
    )

    returns.to_csv(output_dir / "returns.csv", index=False)
    pd.DataFrame({"t": np.arange(regimes.size), "regime": regimes}).to_csv(
        output_dir / "regimes.csv",
        index=False,
    )

    action_frame = pd.DataFrame(
        actions,
        columns=[f"weight_{name}" for name in returns.columns],
    )
    action_frame.insert(0, "decision_t", decision_times)
    action_frame.to_csv(output_dir / "expert_actions.csv", index=False)

    np.savez_compressed(
        output_dir / "demonstrations.npz",
        observations=observations,
        actions=actions,
        returns=returns.to_numpy(dtype=float),
        regimes=regimes,
        decision_times=decision_times,
    )

    metadata = {
        "seed": int(simulation_cfg["seed"]),
        "periods": int(simulation_cfg["T"]),
        "n_assets": int(simulation_cfg["n_assets"]),
        "n_regimes": n_regimes,
        "transaction_costs": list(transaction_costs),
        "window": int(cfg["expert"]["window"]),
        "risk_aversion": float(cfg["expert"].get("risk_aversion", 1.0)),
        "long_only": bool(cfg["expert"].get("long_only", True)),
        "max_weight": float(cfg["expert"].get("max_weight", 1.0)),
        "asset_columns": list(returns.columns),
        "returns_shape": list(returns.shape),
        "regimes_shape": list(regimes.shape),
        "observations_shape": list(observations.shape),
        "actions_shape": list(actions.shape),
        "decision_times_shape": list(decision_times.shape),
        "policy_inputs": ["observations"],
        "diagnostic_only": ["regimes"],
        "config": cfg,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError(f"configuration file is empty or invalid: {args.config}")

    simulation_cfg = cfg["simulation"]
    expert_cfg = cfg["expert"]
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = Path(
            cfg.get("paths", {}).get(
                "expert_output_dir",
                "data/expert_demonstrations",
            )
        )

    periods = int(simulation_cfg["T"])
    seed = int(simulation_cfg["seed"])
    window = int(expert_cfg["window"])
    if periods <= window:
        raise ValueError("simulation.T must be greater than expert.window.")

    simulator = build_simulator(simulation_cfg)
    expert = build_expert(expert_cfg)

    returns, regimes = simulator.simulate(T=periods, seed=seed)
    observations, actions = expert.generate_demonstrations(returns)
    decision_times = np.arange(window, periods)

    save_outputs(
        output_dir=output_dir,
        returns=returns,
        regimes=regimes,
        observations=observations,
        actions=actions,
        decision_times=decision_times,
        cfg=cfg,
    )

    print(f"Wrote expert demonstrations to {output_dir}")
    print(f"returns: {returns.shape}")
    print(f"regimes: {regimes.shape}")
    print(f"observations: {observations.shape}")
    print(f"actions: {actions.shape}")


if __name__ == "__main__":
    main()
