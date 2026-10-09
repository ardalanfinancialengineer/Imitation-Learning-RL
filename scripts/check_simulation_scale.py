"""Smoke-test the exposé's final simulation dimensions and report runtimes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.simulation.presets import load_simulation_preset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/simulation_demo.yaml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/simulation_demo/scale_check.json"),
    )
    return parser.parse_args()


def run_scale_check(cfg: dict) -> list[dict[str, int | float | str]]:
    """Run each configured final dimension once and return timing metadata."""
    simulation_cfg = cfg.get("simulation")
    final_scale = cfg.get("final_scale")
    if not isinstance(simulation_cfg, dict) or not isinstance(final_scale, dict):
        raise ValueError("configuration must contain simulation and final_scale mappings.")

    preset_name = str(simulation_cfg.get("preset"))
    seed = int(simulation_cfg["seed"])
    periods = int(final_scale["T"])
    asset_universes = final_scale.get("n_assets")
    if asset_universes != [50, 100, 200] or periods != 10_000:
        raise ValueError("final scale must specify N=[50, 100, 200] and T=10000.")

    results: list[dict[str, int | float | str]] = []
    for n_assets in asset_universes:
        start = perf_counter()
        returns, regimes = load_simulation_preset(
            preset_name,
            n_assets,
        ).build_simulator().simulate(T=periods, seed=seed)
        elapsed_seconds = perf_counter() - start
        results.append(
            {
                "preset": preset_name,
                "n_assets": n_assets,
                "periods": periods,
                "returns_rows": returns.shape[0],
                "returns_columns": returns.shape[1],
                "regime_observations": regimes.shape[0],
                "elapsed_seconds": round(elapsed_seconds, 4),
            }
        )
    return results


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError(f"configuration file is empty or invalid: {args.config}")

    results = run_scale_check(cfg)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    for result in results:
        print(
            f"N={result['n_assets']}, T={result['periods']}: "
            f"{result['elapsed_seconds']:.4f}s"
        )
    print(f"Wrote scale report to {args.output}")


if __name__ == "__main__":
    main()
