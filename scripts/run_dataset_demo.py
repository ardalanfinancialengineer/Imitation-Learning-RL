"""Run and visualise the completed Weeks 14-16 dataset and environment.

The script generates a small set of independent paths, rolls the validated
Weeks 9-13 expert across each, assembles Equation (6) states lazily, applies the
Section 5.1 separation protocol, replays the expert through the Gymnasium
environment, and evaluates every acceptance check.

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
from src.dataset.diagnostics import (  # noqa: E402
    DatasetDiagnostics,
    DatasetDiagnosticThresholds,
    evaluate_dataset,
)
from src.dataset.paths import PathData, PathSpec, generate_path_from_spec  # noqa: E402
from src.dataset.splits import (  # noqa: E402
    SplitProtocol,
    TemporalSplit,
    default_zero_shot_shifts,
    split_paths,
)
from src.dataset.state import StateSpec  # noqa: E402
from src.reinforcement.environment import PortfolioEnv  # noqa: E402
from src.simulation.presets import load_simulation_preset  # noqa: E402
from src.utils.observability import MemoryTracker, configure_logging  # noqa: E402
from src.utils.parallel import parallel_map  # noqa: E402

logger = logging.getLogger("run_dataset_demo")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/dataset_demo.yaml"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--n-paths", type=int, default=None)
    parser.add_argument("--periods", type=int, default=None)
    parser.add_argument("--n-assets", type=int, default=None)
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
    n_assets_override: int | None = None,
    workers: int = 1,
) -> DatasetDiagnostics:
    """Build the dataset, evaluate the checks, and persist a visual report."""
    tracker = MemoryTracker(logger, phase="dataset")
    simulation_cfg = _mapping(cfg.get("simulation"), "simulation")
    expert_cfg = _mapping(cfg.get("expert"), "expert")

    n_assets = int(n_assets_override or simulation_cfg["n_assets"])
    periods = int(periods_override or simulation_cfg["T"])
    n_paths = int(n_paths_override or simulation_cfg["n_paths"])
    base_seed = int(simulation_cfg["base_seed"])

    preset = load_simulation_preset(str(simulation_cfg["preset"]), n_assets)
    spec = StateSpec(**(cfg.get("state") or {}))

    started = time.perf_counter()
    # Each path is an independent Monte Carlo realisation, so seeds must not
    # collide; the split is then genuinely across realisations. Path seeds are
    # base_seed + path_id with no global-RNG read, so parallel generation is
    # bit-identical to the serial loop (--workers 1).
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
    logger.info("generating %d paths (N=%d, T=%d)", n_paths, n_assets, periods)

    def _tick(done: int, total: int) -> None:
        if done % 10 == 0 or done == total:
            logger.info("paths %d/%d generated", done, total)

    paths: dict[int, PathData] = {
        result.path_id: result
        for result in parallel_map(
            generate_path_from_spec, specs, workers=workers, progress_callback=_tick
        )
    }
    generation_seconds = time.perf_counter() - started
    tracker.sample("paths-ready")

    split_cfg = dict(cfg.get("splits") or {})
    protocol = SplitProtocol(
        path_split=split_paths(
            n_paths,
            train_fraction=float(split_cfg.get("train_fraction", 0.7)),
            validation_fraction=float(split_cfg.get("validation_fraction", 0.15)),
        ),
        temporal_split=TemporalSplit(**(split_cfg.get("temporal") or {})),
        zero_shot_shifts=default_zero_shot_shifts(),
    )

    datasets = {
        name: DemonstrationDataset(
            [paths[path_id] for path_id in protocol.path_split.of(name)], spec=spec
        )
        for name in ("train", "validation", "test")
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    store_dir = output_dir / "paths"
    for path in paths.values():
        path.save(store_dir / f"path_{path.path_id:04d}.npz")

    thresholds = DatasetDiagnosticThresholds(**(cfg.get("diagnostics") or {}))
    diagnostics = evaluate_dataset(
        datasets["train"],
        paths,
        protocol,
        tmp_dir=output_dir / "tmp",
        n_assets=n_assets,
        preset=preset,
        thresholds=thresholds,
    )

    environment_cfg = dict(cfg.get("environment") or {})
    rollout = _expert_rollout(
        paths[protocol.path_split.test[0]],
        spec,
        float(environment_cfg.get("expert_deviation_penalty", 0.0)),
    )

    report = {
        "simulation": {
            "preset": preset.name,
            "n_assets": n_assets,
            "periods": periods,
            "n_paths": n_paths,
            "base_seed": base_seed,
            "generation_seconds": round(generation_seconds, 2),
        },
        "state": datasets["train"].assembler.describe(),
        "storage": _storage_report(store_dir, datasets, n_assets, spec, paths),
        "datasets": {name: value.describe() for name, value in datasets.items()},
        "splits": protocol.to_dict(),
        "environment": rollout["description"],
        "expert_rollout": rollout["summary"],
        "diagnostics": diagnostics.to_dict(),
    }
    (output_dir / "dataset_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    _write_tables(output_dir, datasets, protocol, paths)

    save_dashboard(
        datasets=datasets,
        paths=paths,
        protocol=protocol,
        spec=spec,
        rollout=rollout,
        diagnostics=diagnostics,
        storage=report["storage"],
        output_path=output_dir / "dataset_dashboard.png",
    )
    tracker.summary()
    return diagnostics


def _expert_rollout(
    path: PathData, spec: StateSpec, expert_deviation_penalty: float
) -> dict[str, Any]:
    """Replay the expert's own actions through the environment."""
    environment = PortfolioEnv(
        path, spec=spec, expert_deviation_penalty=expert_deviation_penalty
    )
    observation, _ = environment.reset()

    rewards, net_returns, turnovers, observations = [], [], [], [observation.copy()]
    for _ in range(environment.n_steps):
        _, reward, terminated, _, info = environment.step(environment.expert_action())
        rewards.append(reward)
        net_returns.append(info["net_return"])
        turnovers.append(info["turnover"])
        if terminated:
            break

    net = np.asarray(net_returns)
    return {
        "description": environment.describe(),
        "rewards": np.asarray(rewards),
        "net_returns": net,
        "turnovers": np.asarray(turnovers),
        "observation_dimension": observations[0].size,
        "summary": {
            "steps": len(rewards),
            "mean_reward": float(np.mean(rewards)),
            "total_net_return": float(np.prod(1.0 + net) - 1.0),
            "mean_turnover": float(np.mean(turnovers)),
        },
    }


def _storage_report(
    store_dir: Path,
    datasets: dict[str, DemonstrationDataset],
    n_assets: int,
    spec: StateSpec,
    paths: dict[int, PathData],
) -> dict[str, Any]:
    """Compare the stored footprint with a materialised-state alternative."""
    stored_bytes = sum(item.stat().st_size for item in store_dir.glob("*.npz"))
    n_samples = sum(len(value) for value in datasets.values())
    state_bytes = n_samples * datasets["train"].state_dimension * 4

    reference = paths[next(iter(paths))]
    final_dimension = spec.dimension(50, reference.n_regimes)
    final_materialised = 500 * 10_000 * final_dimension * 4
    return {
        "stored_megabytes": round(stored_bytes / 1e6, 2),
        "materialised_state_megabytes": round(state_bytes / 1e6, 2),
        "reduction_factor": round(state_bytes / max(stored_bytes, 1), 2),
        "final_scale_materialised_gigabytes": round(final_materialised / 1e9, 1),
        "final_scale_state_dimension": final_dimension,
    }


def _write_tables(
    output_dir: Path,
    datasets: dict[str, DemonstrationDataset],
    protocol: SplitProtocol,
    paths: dict[int, PathData],
) -> None:
    rows = []
    for name, dataset in datasets.items():
        for path_id in dataset.path_ids:
            entry = next(item for item in dataset.entries if item.data.path_id == path_id)
            rows.append(
                {
                    "split": name,
                    "path_id": path_id,
                    "seed": paths[path_id].seed,
                    "samples": int(entry.indices.size),
                    "decisions": paths[path_id].n_decisions,
                    "mean_turnover": float(paths[path_id].turnover.mean()),
                    "mean_net_return": float(paths[path_id].net_returns.mean()),
                }
            )
    pd.DataFrame(rows).to_csv(output_dir / "dataset_splits.csv", index=False)

    blocks = datasets["train"].assembler.spec.block_sizes(
        datasets["train"].assembler.n_assets, datasets["train"].assembler.n_regimes
    )
    pd.DataFrame(
        [{"block": name, "size": size} for name, size in blocks.items()]
    ).to_csv(output_dir / "state_blocks.csv", index=False)

    pd.DataFrame(
        [shift.to_dict() for shift in protocol.zero_shot_shifts]
    ).to_csv(output_dir / "zero_shot_shifts.csv", index=False)


def save_dashboard(
    *,
    datasets: dict[str, DemonstrationDataset],
    paths: dict[int, PathData],
    protocol: SplitProtocol,
    spec: StateSpec,
    rollout: dict[str, Any],
    diagnostics: DatasetDiagnostics,
    storage: dict[str, Any],
    output_path: Path,
) -> None:
    """Save a six-panel visual audit of the dataset and environment."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    train = datasets["train"]
    fig.suptitle(
        "Weeks 14–16: State, Dataset, Splits, and Environment "
        f"(L={spec.lookback}, state dim {train.state_dimension})",
        fontsize=15,
        fontweight="bold",
    )

    blocks = train.assembler.spec.block_sizes(
        train.assembler.n_assets, train.assembler.n_regimes
    )
    names = [name.replace("_", "\n") for name in blocks]
    axes[0, 0].bar(names, list(blocks.values()), color="#4c78a8")
    axes[0, 0].set_title("State composition, Equation (6)")
    axes[0, 0].set_ylabel("Dimensions")
    axes[0, 0].tick_params(axis="x", labelsize=7)
    for index, value in enumerate(blocks.values()):
        axes[0, 0].text(index, value, str(value), ha="center", va="bottom", fontsize=8)

    split_colors = {"train": "#4c78a8", "validation": "#f2cf5b", "test": "#e45756"}
    for offset, (name, dataset) in enumerate(datasets.items()):
        ids = list(dataset.path_ids)
        counts = [
            int(next(e for e in dataset.entries if e.data.path_id == i).indices.size)
            for i in ids
        ]
        axes[0, 1].bar(ids, counts, color=split_colors[name], label=name)
    axes[0, 1].set_title("Path-level split (disjoint realisations)")
    axes[0, 1].set_xlabel("Path id")
    axes[0, 1].set_ylabel("Samples")
    axes[0, 1].legend(fontsize=8)

    reference = paths[next(iter(paths))]
    segments = protocol.temporal_split.segments(reference.n_decisions)
    for index, (name, segment) in enumerate(segments.items()):
        axes[0, 2].barh(
            0,
            segment.stop - segment.start,
            left=segment.start,
            color=split_colors[name],
            label=name,
            height=0.5,
        )
    axes[0, 2].set_title("Temporal holdout within a path (ordered, contiguous)")
    axes[0, 2].set_xlabel("Decision index")
    axes[0, 2].set_yticks([])
    axes[0, 2].legend(fontsize=8, loc="upper center", ncol=3)

    equity = np.cumprod(1.0 + rollout["net_returns"]) - 1.0
    axes[1, 0].plot(equity, color="#222222", linewidth=1.4, label="Env. net return")
    axes[1, 0].set_title(
        "Expert replayed through the environment\n"
        f"(reproduces stored returns to {diagnostics.environment_difference:.1e})"
    )
    axes[1, 0].set_xlabel("Environment step")
    axes[1, 0].set_ylabel("Cumulative net return")
    axes[1, 0].legend(fontsize=8)

    # Budget sums are exactly one, so a histogram of them has no range at all.
    # The individual weights are the informative view: they show the cap binding.
    actions = train.actions()
    cap = float(paths[next(iter(paths))].metadata["expert"]["constraints"]["max_weight"])
    axes[1, 1].hist(actions.reshape(-1), bins=40, color="#31a354")
    axes[1, 1].axvline(cap, color="#e45756", linestyle="--", label=f"cap u = {cap:g}")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_title(
        f"Action validity: {actions.shape[0]} expert allocations\n"
        f"budget and non-negativity hold to {diagnostics.simplex_violation:.1e}"
    )
    axes[1, 1].set_xlabel("Individual weight")
    axes[1, 1].set_ylabel("Count (log)")
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
    axes[1, 2].set_xlabel(
        f"Stored {storage['stored_megabytes']} MB vs "
        f"{storage['materialised_state_megabytes']} MB materialised "
        f"({storage['reduction_factor']}x)",
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
        cfg.get("paths", {}).get("output_dir", "data/dataset_demo")
    )
    diagnostics = run_demo(
        cfg,
        output_dir,
        n_paths_override=args.n_paths,
        periods_override=args.periods,
        n_assets_override=args.n_assets,
        workers=args.workers,
    )

    logger.info("Wrote dataset demo to %s", output_dir)
    logger.info("acceptance checks: %s", "PASS" if diagnostics.passed else "FAIL")
    for name, passed in diagnostics.checks.items():
        logger.info("  %s: %s", name, "PASS" if passed else "FAIL")
    logger.info("dashboard: %s", output_dir / "dataset_dashboard.png")
    logger.info("report: %s", output_dir / "dataset_report.json")
    if not diagnostics.passed:
        raise SystemExit("Dataset acceptance checks failed; inspect dataset_report.json.")


if __name__ == "__main__":
    main()
