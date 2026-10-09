"""End-to-end test of the Weeks 26-29 GAIL demonstration script."""

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
        "run_gail_demo", PROJECT_ROOT / "scripts" / "run_gail_demo.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_gail_demo"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def demo_output(tmp_path_factory):
    """Run the shipped configuration at reduced size, keeping every check."""
    module = _load_script()
    cfg = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "gail_demo.yaml").read_text(encoding="utf-8")
    )
    cfg["simulation"] |= {"n_assets": 4, "T": 420, "n_paths": 5}
    cfg["expert"]["horizon"] = 5
    cfg["expert"]["constraints"]["max_weight"] = 0.4
    cfg["expert"]["moments"] = {
        "estimator": "rolling",
        "covariance": "ledoit_wolf",
        "window": 60,
    }
    cfg["state"]["lookback"] = 10
    cfg["policy"] = {"kind": "mlp", "hidden_dimensions": [32, 16]}
    cfg["behavioural_cloning"]["epochs"] = 3
    cfg["dagger"] |= {"iterations": 1, "epochs_per_round": 2, "query_fraction": 0.1}
    cfg["gail"] |= {"iterations": 4, "segment_length": 8, "evaluate_every": 2}
    cfg["critic"] |= {"steps_per_round": 2, "hidden_dimensions": [32, 16]}

    output_dir = tmp_path_factory.mktemp("gail_demo")
    diagnostics = module.run_demo(cfg, output_dir)
    return diagnostics, output_dir


def test_every_acceptance_check_passes(demo_output):
    diagnostics, _ = demo_output
    assert diagnostics.passed, diagnostics.checks


def test_expected_artefacts_are_written(demo_output):
    _, output_dir = demo_output
    for name in (
        "gail_dashboard.png",
        "gail_report.json",
        "gail_iterations.csv",
        "gail_comparison.csv",
        "gail_stability.csv",
    ):
        assert (output_dir / name).is_file(), f"{name} was not written"
    assert (output_dir / "gail_dashboard.png").stat().st_size > 10_000


def test_the_report_compares_all_three_imitation_stages(demo_output):
    """The work package asks for GAIL against behavioural cloning and DAgger."""
    _, output_dir = demo_output
    report = json.loads((output_dir / "gail_report.json").read_text(encoding="utf-8"))
    metrics = report["comparison"]["metrics"]

    assert {"behavioural_cloning", "dagger", "gail_from_bc", "gail_from_dagger",
            "never_trade"} <= set(metrics)
    for values in metrics.values():
        assert values["closed_loop_tracking_error"] > 0.0


def test_paired_intervals_are_reported_for_each_comparison(demo_output):
    _, output_dir = demo_output
    report = json.loads((output_dir / "gail_report.json").read_text(encoding="utf-8"))
    intervals = report["comparison"]["paired_intervals"]

    assert "behavioural_cloning_minus_gail_from_bc" in intervals
    assert "dagger_minus_gail_from_bc" in intervals
    for interval in intervals.values():
        assert {"mean", "low", "high", "excludes_zero"} <= set(interval)


def test_every_policy_is_scored_by_the_same_critic(demo_output):
    """Wasserstein estimates from different critics are not comparable."""
    _, output_dir = demo_output
    report = json.loads((output_dir / "gail_report.json").read_text(encoding="utf-8"))
    distances = report["comparison"]["occupancy_distance"]

    assert {"behavioural_cloning", "dagger", "gail_from_bc"} <= set(distances)
    assert "never_trade" not in distances


def test_the_stability_ablation_covers_both_lipschitz_methods(demo_output):
    _, output_dir = demo_output
    report = json.loads((output_dir / "gail_report.json").read_text(encoding="utf-8"))
    ablation = report["stability_ablation"]

    assert {"gradient_penalty", "weight_clipping"} <= set(ablation)
    for payload in ablation.values():
        assert "wasserstein_volatility" in payload["stability"]


def test_the_gradient_check_is_recorded_in_the_report(demo_output):
    _, output_dir = demo_output
    report = json.loads((output_dir / "gail_report.json").read_text(encoding="utf-8"))
    check = report["diagnostics"]["rollout_gradient_correctness"]

    assert check["max_relative_error"] < 1e-6
    assert check["parameters_checked"] > 0
