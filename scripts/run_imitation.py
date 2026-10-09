"""Train one behavioural-cloning policy from stored expert paths.

This is the pipeline entry point, the counterpart to ``scripts/run_expert.py``.
It trains a single policy on paths that already exist on disk and writes a
checkpoint plus a metrics file. It runs no acceptance checks and draws no
figures; ``scripts/run_bc_demo.py`` is the milestone report that does both.

Use this when the paths were generated once and several policies or several
objectives are to be trained against them without regenerating anything.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset.demonstrations import DemonstrationDataset  # noqa: E402
from src.dataset.paths import PathData  # noqa: E402
from src.dataset.splits import split_paths  # noqa: E402
from src.dataset.state import StateSpec  # noqa: E402
from src.expert.constraints import PortfolioConstraints  # noqa: E402
from src.imitation.bc import (  # noqa: E402
    BCConfig,
    BehaviouralCloningTrainer,
    seed_everything,
)
from src.policy.builder import build_policy  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/bc_demo.yaml"))
    parser.add_argument(
        "--paths-dir",
        type=Path,
        default=None,
        help="Directory of path_XXXX.npz files. Defaults to the config's cache_dir.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/imitation"))
    parser.add_argument(
        "--policy",
        type=str,
        default="transformer",
        help="Which entry of the config's `policies` block to train.",
    )
    parser.add_argument("--loss", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    return parser.parse_args()


def load_paths(paths_dir: Path) -> list[PathData]:
    files = sorted(paths_dir.glob("*.npz"))
    if not files:
        raise SystemExit(
            f"no stored paths in {paths_dir}. Generate them first, for example with "
            f"`python scripts/run_dataset_demo.py`."
        )
    return [PathData.load(file_path) for file_path in files]


def main() -> None:
    args = parse_args()
    cfg: dict[str, Any] = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError(f"configuration file is empty or invalid: {args.config}")

    paths_dir = args.paths_dir or Path(
        cfg.get("paths", {}).get("cache_dir", "data/bc_demo/paths")
    )
    paths = load_paths(paths_dir)

    spec = StateSpec(**(cfg.get("state") or {}))
    expert_cfg = cfg["expert"]
    constraints = PortfolioConstraints(**expert_cfg["constraints"])
    n_assets = paths[0].n_assets
    n_regimes = int(expert_cfg["moments"]["n_regimes"])

    split_cfg = dict(cfg.get("splits") or {})
    path_split = split_paths(
        len(paths),
        train_fraction=float(split_cfg.get("train_fraction", 0.7)),
        validation_fraction=float(split_cfg.get("validation_fraction", 0.15)),
    )
    by_id = {path.path_id: path for path in paths}
    datasets = {
        name: DemonstrationDataset(
            [by_id[path_id] for path_id in path_split.of(name)], spec=spec
        )
        for name in ("train", "validation", "test")
    }

    policies = cfg.get("policies") or {}
    if args.policy not in policies:
        raise SystemExit(
            f"policy {args.policy!r} is not in the config; available: {sorted(policies)}."
        )

    training_cfg = dict(cfg.get("training") or {})
    if args.loss is not None:
        training_cfg["loss"] = args.loss
    if args.epochs is not None:
        training_cfg["epochs"] = args.epochs
    bc_config = BCConfig(**training_cfg)

    # Seed before construction: the weights are drawn there, not in fit().
    seed_everything(bc_config.seed)
    policy = build_policy(
        policies[args.policy],
        spec=spec,
        n_assets=n_assets,
        n_regimes=n_regimes,
        constraints=constraints,
    )
    trainer = BehaviouralCloningTrainer(policy, bc_config)
    history = trainer.fit(datasets["train"], datasets["validation"])
    metrics = {name: trainer.evaluate(dataset) for name, dataset in datasets.items()}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save(
        args.output_dir / f"{args.policy}.pt",
        extra={"state_spec": spec.to_dict(), "path_split": path_split.to_dict()},
    )
    (args.output_dir / f"{args.policy}_metrics.json").write_text(
        json.dumps(
            {
                "policy": args.policy,
                "description": policy.describe(),
                "training": bc_config.to_dict(),
                "history": history.to_dict(),
                "metrics": metrics,
                "paths_dir": str(paths_dir),
                "n_paths": len(paths),
            },
            indent=2,
            default=float,
        ),
        encoding="utf-8",
    )

    print(f"Trained {args.policy} on {len(datasets['train'])} samples")
    print(f"  best epoch: {history.best_epoch} of {len(history.records)}")
    print(f"  test tracking error: {metrics['test']['tracking_error']:.4e}")
    print(f"  checkpoint: {args.output_dir / f'{args.policy}.pt'}")


if __name__ == "__main__":
    main()
