"""End-to-end test of the Weeks 22-25 DAgger demonstration script."""

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
        "run_dagger_demo", PROJECT_ROOT / "scripts" / "run_dagger_demo.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_dagger_demo"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def demo_output(tmp_path_factory):
    """Run the shipped configuration at reduced size, keeping every check."""
    module = _load_script()
    cfg = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "dagger_demo.yaml").read_text(encoding="utf-8")
    )
    cfg["simulation"]["n_assets"] = 4
    cfg["simulation"]["T"] = 450
    cfg["simulation"]["n_paths"] = 5
    for expert in cfg["experts"].values():
        expert["horizon"] = 5
        expert["constraints"]["max_weight"] = 0.4
        if expert["moments"]["estimator"] == "regime_mixture":
            expert["moments"]["burn_in"] = 250
    cfg["policies"] = {"mlp": {"kind": "mlp", "hidden_dimensions": [32, 16]}}
    cfg["behavioural_cloning"]["epochs"] = 3
    cfg["dagger"] = cfg["dagger"] | {
        "iterations": 2,
        "epochs_per_round": 2,
        "query_fraction": 0.15,
    }

    output_dir = tmp_path_factory.mktemp("dagger_demo")
    diagnostics = module.run_demo(cfg, output_dir)
    return diagnostics, output_dir


def test_every_acceptance_check_passes(demo_output):
    diagnostics, _ = demo_output
    assert diagnostics.passed, diagnostics.checks


def test_expected_artefacts_are_written(demo_output):
    _, output_dir = demo_output
    for name in ("dagger_dashboard.png", "dagger_report.json", "dagger_rounds.csv",
                 "dagger_summary.csv"):
        assert (output_dir / name).is_file(), f"{name} was not written"
    assert (output_dir / "dagger_dashboard.png").stat().st_size > 10_000


def test_report_covers_both_expert_configurations(demo_output):
    _, output_dir = demo_output
    report = json.loads((output_dir / "dagger_report.json").read_text(encoding="utf-8"))
    assert set(report["results"]) == {"shipped", "control"}
    for expert_results in report["results"].values():
        assert "never_trade_closed_loop" in expert_results
        assert "mlp" in expert_results


def test_rounds_start_at_the_cloned_policy_and_grow(demo_output):
    _, output_dir = demo_output
    report = json.loads((output_dir / "dagger_report.json").read_text(encoding="utf-8"))
    rounds = report["results"]["shipped"]["mlp"]["rounds"]

    assert rounds[0]["iteration"] == 0
    assert rounds[0]["queries"] == 0
    sizes = [record["aggregate_samples"] for record in rounds]
    assert all(later > earlier for earlier, later in zip(sizes, sizes[1:]))


def test_improvement_is_reported_with_a_paired_interval(demo_output):
    _, output_dir = demo_output
    report = json.loads((output_dir / "dagger_report.json").read_text(encoding="utf-8"))
    improvement = report["results"]["shipped"]["mlp"]["improvement"]

    assert {"mean", "low", "high", "excludes_zero", "n_paths"} <= set(improvement)
    assert improvement["n_paths"] >= 1


def test_closed_loop_and_teacher_forced_are_both_recorded(demo_output):
    """The contrast between them is the milestone's central measurement."""
    _, output_dir = demo_output
    report = json.loads((output_dir / "dagger_report.json").read_text(encoding="utf-8"))
    payload = report["results"]["shipped"]["mlp"]

    assert "closed_loop_tracking_error" in payload["test_before"]
    assert "closed_loop_tracking_error" in payload["test_after"]
    assert payload["test_teacher_forced_after"] > 0.0
    assert (
        payload["test_after"]["closed_loop_tracking_error"]
        > payload["test_teacher_forced_after"]
    ), "closed-loop error should exceed teacher-forced error"
