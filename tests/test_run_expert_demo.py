"""End-to-end test of the Weeks 9-13 expert demonstration script."""

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
        "run_expert_demo", PROJECT_ROOT / "scripts" / "run_expert_demo.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_expert_demo"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def demo_output(tmp_path_factory):
    """Run the shipped configuration at a reduced size that keeps every check."""
    module = _load_script()
    cfg = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "expert_demo.yaml").read_text(encoding="utf-8")
    )
    cfg["simulation"]["n_assets"] = 4
    cfg["simulation"]["T"] = 700
    cfg["expert"]["horizon"] = 5
    cfg["expert"]["moments"]["burn_in"] = 300
    cfg["ablations"] = {"transaction_costs": [0.0, 0.005], "horizons": [1, 5]}

    output_dir = tmp_path_factory.mktemp("expert_demo")
    diagnostics = module.run_demo(cfg, output_dir)
    return diagnostics, output_dir


def test_every_acceptance_check_passes(demo_output):
    diagnostics, _ = demo_output
    assert diagnostics.passed, diagnostics.checks


def test_expected_artefacts_are_written(demo_output):
    _, output_dir = demo_output
    for name in (
        "expert_dashboard.png",
        "expert_report.json",
        "expert_decisions.csv",
        "expert_ablations.csv",
    ):
        assert (output_dir / name).is_file(), f"{name} was not written"
    assert (output_dir / "expert_dashboard.png").stat().st_size > 10_000


def test_report_records_the_exposé_settings(demo_output):
    _, output_dir = demo_output
    report = json.loads((output_dir / "expert_report.json").read_text(encoding="utf-8"))

    expert = report["expert"]
    assert expert["horizon"] == 5
    assert 0.001 <= expert["transaction_cost"] <= 0.005
    assert expert["constraints"]["long_only"] is True
    assert expert["forecaster"]["estimator"] == "regime_mixture"
    assert report["diagnostics"]["passed"] is True
    assert "equal_weight_benchmark" in report


def test_turnover_falls_across_the_cost_grid(demo_output):
    _, output_dir = demo_output
    report = json.loads((output_dir / "expert_report.json").read_text(encoding="utf-8"))
    turnovers = report["ablations"]["transaction_costs"]
    assert turnovers["0.0000"]["mean_turnover"] > turnovers["0.0050"]["mean_turnover"]
