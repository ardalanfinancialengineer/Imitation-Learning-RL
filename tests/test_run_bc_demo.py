"""End-to-end test of the Weeks 17-21 behavioural-cloning demonstration script."""

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
        "run_bc_demo", PROJECT_ROOT / "scripts" / "run_bc_demo.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_bc_demo"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def demo_output(tmp_path_factory):
    """Run the shipped configuration at reduced size.

    Every policy, every artefact, and every acceptance check still runs; only
    the simulation size, the model widths, and the epoch count shrink. The
    latency threshold is relaxed because a five-day-horizon expert over four
    assets solves in milliseconds, which is a fact about the fixture rather than
    about the policy — see ``tests/test_policy_diagnostics.py``.
    """
    module = _load_script()
    cfg = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "bc_demo.yaml").read_text(encoding="utf-8")
    )
    cfg["simulation"] |= {"n_assets": 4, "T": 500, "n_paths": 8}
    cfg["expert"]["horizon"] = 5
    cfg["expert"]["moments"] = {
        "estimator": "rolling",
        "covariance": "ledoit_wolf",
        "window": 60,
        "n_regimes": 3,
    }
    cfg["policies"]["mlp"]["hidden_dimensions"] = [32]
    for name in ("transformer", "diffusion"):
        cfg["policies"][name]["encoder"] |= {"d_model": 32, "n_layers": 1}
    cfg["policies"]["diffusion"]["diffusion"] |= {
        "n_steps": 20,
        "hidden_dimension": 64,
        "n_blocks": 2,
        "sampling_steps": 4,
    }
    cfg["training"] |= {"epochs": 6, "batch_size": 64, "patience": 6}
    # Two seeds, so the multi-seed path is exercised without tripling the cost.
    cfg["seeds"] = [0, 1]
    cfg["diagnostics"]["min_latency_speedup"] = 1.0
    cfg["loss_ablation"] = ["mse", "l1"]

    output_dir = tmp_path_factory.mktemp("bc_demo")
    cfg["paths"]["cache_dir"] = str(output_dir / "paths")
    diagnostics = module.run_demo(cfg, output_dir)
    return diagnostics, output_dir


def test_every_policy_is_trained_and_reported(demo_output):
    diagnostics, _ = demo_output
    assert set(diagnostics) == {"mlp", "transformer", "diffusion"}


def test_the_structural_checks_pass_for_every_policy(demo_output):
    """Feasibility, determinism, and causality must hold regardless of fit quality.

    Imitation *quality* checks are deliberately excluded here: six epochs on a
    four-asset fixture is not enough training for a policy to beat the
    never-trade rule, and asserting otherwise would make the test a measure of
    the fixture. Quality is gated in the shipped configuration instead.
    """
    diagnostics, _ = demo_output
    for name, report in diagnostics.items():
        assert report.checks["action_feasibility"], name
        assert report.checks["determinism"], name
        assert report.checks["policy_no_look_ahead"], name
        assert report.checks["optimisation_sanity"], name


def test_expected_artefacts_are_written(demo_output):
    _, output_dir = demo_output
    for name in (
        "bc_dashboard.png",
        "bc_report.json",
        "policy_comparison.csv",
        "training_history.csv",
        "loss_ablation.csv",
        "acceptance_checks.csv",
    ):
        assert (output_dir / name).is_file(), f"{name} was not written"
    assert (output_dir / "bc_dashboard.png").stat().st_size > 10_000
    for name in ("mlp", "transformer", "diffusion"):
        assert (output_dir / "checkpoints" / f"{name}.pt").is_file()


def test_report_records_the_architectures_and_the_evidence(demo_output):
    _, output_dir = demo_output
    report = json.loads((output_dir / "bc_report.json").read_text(encoding="utf-8"))

    assert report["policies"]["transformer"]["description"]["encoder"]["kind"] == (
        "transformer"
    )
    assert report["policies"]["diffusion"]["description"]["kind"] == (
        "transformer_diffusion"
    )
    # Calibration replaced the default adjustment scale with a measured one.
    assert report["policies"]["diffusion"]["description"]["action_scale"] > 0.0

    references = report["reference_tracking_errors"]
    assert set(references) == {"equal_weight", "previous_weights"}
    assert references["equal_weight"] > references["previous_weights"]

    for entry in report["policies"].values():
        latency = entry["diagnostics"]["inference_latency"]
        assert latency["policy_seconds_per_decision"] > 0.0
        assert latency["expert_seconds_per_decision"] > 0.0
        assert entry["diagnostics"]["memorisation_gap"]["test_over_train"] > 0.0


def test_the_splits_stay_disjoint(demo_output):
    _, output_dir = demo_output
    splits = json.loads(
        (output_dir / "bc_report.json").read_text(encoding="utf-8")
    )["splits"]["in_distribution"]
    train, validation, test = (
        set(splits["train"]),
        set(splits["validation"]),
        set(splits["test"]),
    )
    assert not (train & validation or train & test or validation & test)


def test_the_loss_ablation_covers_every_requested_objective(demo_output):
    _, output_dir = demo_output
    ablation = json.loads(
        (output_dir / "bc_report.json").read_text(encoding="utf-8")
    )["loss_ablation"]
    assert set(ablation) == {"mse", "l1"}
    for entry in ablation.values():
        assert entry["test_metrics"]["feasibility_violation"] <= 1e-6


def test_paths_are_reused_on_a_second_run(demo_output, tmp_path):
    """The cache must be a pure speed-up, keyed on the configuration."""
    module = sys.modules["run_bc_demo"]
    _, output_dir = demo_output
    from src.simulation.presets import canonical_three_regime_preset

    cfg = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "bc_demo.yaml").read_text(encoding="utf-8")
    )
    expert_cfg = cfg["expert"] | {"horizon": 5}
    expert_cfg["moments"] = {
        "estimator": "rolling",
        "covariance": "ledoit_wolf",
        "window": 60,
        "n_regimes": 3,
    }
    _, _, reused = module.load_or_generate_paths(
        cache_dir=output_dir / "paths",
        n_paths=8,
        periods=500,
        base_seed=42,
        preset=canonical_three_regime_preset(4),
        expert_config=expert_cfg,
    )
    assert reused == 8


def test_a_changed_expert_configuration_invalidates_the_cache(demo_output):
    module = sys.modules["run_bc_demo"]
    _, output_dir = demo_output
    from src.simulation.presets import canonical_three_regime_preset

    cfg = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "bc_demo.yaml").read_text(encoding="utf-8")
    )
    # A different risk aversion is a different expert, so nothing may be reused.
    expert_cfg = cfg["expert"] | {"horizon": 5, "risk_aversion": 9.0}
    expert_cfg["moments"] = {
        "estimator": "rolling",
        "covariance": "ledoit_wolf",
        "window": 60,
        "n_regimes": 3,
    }
    _, _, reused = module.load_or_generate_paths(
        cache_dir=output_dir / "paths",
        n_paths=2,
        periods=500,
        base_seed=42,
        preset=canonical_three_regime_preset(4),
        expert_config=expert_cfg,
    )
    assert reused == 0
