"""End-to-end test of the Weeks 14-16 dataset demonstration script."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "run_dataset_demo", PROJECT_ROOT / "scripts" / "run_dataset_demo.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_dataset_demo"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def demo_output(tmp_path_factory):
    """Run the shipped configuration at reduced size, keeping every check."""
    module = _load_script()
    cfg = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "dataset_demo.yaml").read_text(encoding="utf-8")
    )
    cfg["simulation"]["n_assets"] = 4
    cfg["simulation"]["T"] = 600
    cfg["simulation"]["n_paths"] = 4
    cfg["expert"]["horizon"] = 5
    cfg["expert"]["moments"]["burn_in"] = 250

    output_dir = tmp_path_factory.mktemp("dataset_demo")
    diagnostics = module.run_demo(cfg, output_dir)
    return diagnostics, output_dir


def test_every_acceptance_check_passes(demo_output):
    diagnostics, _ = demo_output
    assert diagnostics.passed, diagnostics.checks


def test_expected_artefacts_are_written(demo_output):
    _, output_dir = demo_output
    for name in (
        "dataset_dashboard.png",
        "dataset_report.json",
        "dataset_splits.csv",
        "state_blocks.csv",
        "zero_shot_shifts.csv",
    ):
        assert (output_dir / name).is_file(), f"{name} was not written"
    assert (output_dir / "dataset_dashboard.png").stat().st_size > 10_000
    assert len(list((output_dir / "paths").glob("*.npz"))) == 4


def test_report_records_the_state_and_split_protocol(demo_output):
    _, output_dir = demo_output
    report = json.loads((output_dir / "dataset_report.json").read_text(encoding="utf-8"))

    state = report["state"]
    assert state["spec"]["lookback"] == 20
    assert state["dimension"] == sum(state["block_sizes"].values())
    assert state["block_sizes"]["covariance"] == 4 * 5 // 2

    splits = report["splits"]
    train = set(splits["in_distribution"]["train"])
    test = set(splits["in_distribution"]["test"])
    assert not train & test
    assert len(splits["zero_shot"]) >= 4
    assert report["diagnostics"]["passed"] is True


def test_lazy_storage_is_far_smaller_than_materialised_states(demo_output):
    _, output_dir = demo_output
    storage = json.loads(
        (output_dir / "dataset_report.json").read_text(encoding="utf-8")
    )["storage"]
    assert storage["reduction_factor"] > 1.0
    assert storage["final_scale_state_dimension"] == 2380


def test_environment_rollout_is_recorded(demo_output):
    _, output_dir = demo_output
    report = json.loads((output_dir / "dataset_report.json").read_text(encoding="utf-8"))
    rollout = report["expert_rollout"]
    assert rollout["steps"] > 0
    assert report["environment"]["expert_deviation_penalty"] == 0.0
