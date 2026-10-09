"""Tests for the dataset and environment acceptance checks."""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.dataset.demonstrations import DemonstrationDataset
from src.dataset.diagnostics import DatasetDiagnosticThresholds, evaluate_dataset
from src.dataset.paths import generate_path
from src.dataset.splits import (
    PathSplit,
    SplitProtocol,
    TemporalSplit,
    ZeroShotShift,
    default_zero_shot_shifts,
    split_paths,
)
from src.dataset.state import StateSpec
from src.simulation.presets import canonical_three_regime_preset

N_ASSETS = 4
SPEC = StateSpec(lookback=20)
EXPERT_CONFIG = {
    "horizon": 5,
    "risk_aversion": 5.0,
    "transaction_cost": 0.0025,
    "constraints": {"long_only": True, "max_weight": 0.4},
    "moments": {
        "estimator": "regime_mixture",
        "covariance": "ledoit_wolf",
        "window": 60,
        "blend_weight": 0.25,
        "n_regimes": 3,
        "burn_in": 250,
        "refit_every": 250,
    },
}


@pytest.fixture(scope="module")
def fixture():
    preset = canonical_three_regime_preset(N_ASSETS)
    paths = {
        path_id: generate_path(
            path_id=path_id,
            seed=42 + path_id,
            preset=preset,
            periods=400,
            expert_config=EXPERT_CONFIG,
        )
        for path_id in range(3)
    }
    protocol = SplitProtocol(
        path_split=split_paths(3),
        temporal_split=TemporalSplit(),
        zero_shot_shifts=default_zero_shot_shifts(),
    )
    dataset = DemonstrationDataset([paths[0]], spec=SPEC)
    return preset, paths, protocol, dataset


def _evaluate(fixture, tmp_path, **kwargs):
    preset, paths, protocol, dataset = fixture
    return evaluate_dataset(
        kwargs.pop("dataset", dataset),
        kwargs.pop("paths", paths),
        kwargs.pop("protocol", protocol),
        tmp_dir=tmp_path,
        n_assets=N_ASSETS,
        preset=preset,
        samples=2,
        **kwargs,
    )


def test_a_correct_dataset_passes_every_check(fixture, tmp_path):
    diagnostics = _evaluate(fixture, tmp_path)
    assert diagnostics.passed, diagnostics.checks
    assert set(diagnostics.checks) == {
        "state_no_look_ahead",
        "state_integrity",
        "action_validity",
        "split_disjointness",
        "environment_consistency",
        "store_roundtrip",
    }


def test_states_are_bit_identical_when_the_future_is_rewritten(fixture, tmp_path):
    """The milestone's headline requirement, at zero tolerance."""
    diagnostics = _evaluate(fixture, tmp_path)
    assert diagnostics.state_difference == 0.0
    assert diagnostics.checked_decision_times


def test_anchor_chain_sensitivity_is_reported_but_not_asserted(fixture, tmp_path):
    diagnostics = _evaluate(fixture, tmp_path)
    assert diagnostics.anchor_chain_sensitivity >= 0.0
    assert "anchor_chain_sensitivity" not in diagnostics.checks


def test_environment_replay_matches_the_stored_trajectory(fixture, tmp_path):
    diagnostics = _evaluate(fixture, tmp_path)
    assert diagnostics.environment_difference <= 1e-12


def test_report_is_json_serialisable(fixture, tmp_path):
    report = _evaluate(fixture, tmp_path).to_dict()
    assert json.loads(json.dumps(report))["passed"] is True


def test_a_split_referencing_unknown_paths_fails_the_split_check(fixture, tmp_path):
    """PathSplit rejects overlaps itself, so the check guards the other failure:
    a split that names paths the dataset does not actually contain."""
    _, _, protocol, _ = fixture
    broken = SplitProtocol(
        path_split=PathSplit(train=(0,), validation=(1,), test=(99,)),
        temporal_split=protocol.temporal_split,
        zero_shot_shifts=protocol.zero_shot_shifts,
    )
    diagnostics = _evaluate(fixture, tmp_path, protocol=broken)
    assert not diagnostics.checks["split_disjointness"]
    assert not diagnostics.passed


def test_a_shift_that_changes_nothing_fails_the_split_check(fixture, tmp_path):
    _, _, protocol, _ = fixture
    inert = ZeroShotShift(
        name="inert",
        description="Scales every factor by one, so nothing actually changes.",
        preset={"persistence_scale": 1.0},
    )
    broken = SplitProtocol(
        path_split=protocol.path_split,
        temporal_split=protocol.temporal_split,
        zero_shot_shifts=(inert,),
    )
    diagnostics = _evaluate(fixture, tmp_path, protocol=broken)
    assert not diagnostics.checks["split_disjointness"]
    assert not diagnostics.passed


def test_an_unreachable_simplex_tolerance_fails_the_action_check(fixture, tmp_path):
    diagnostics = _evaluate(
        fixture,
        tmp_path,
        thresholds=DatasetDiagnosticThresholds(max_simplex_violation=0.0),
    )
    assert not diagnostics.checks["action_validity"] or diagnostics.simplex_violation == 0.0


def test_corrupted_actions_fail_the_action_check(fixture, tmp_path):
    from dataclasses import replace

    _, paths, _, _ = fixture
    broken = replace(paths[0], expert_weights=paths[0].expert_weights * 2.0)
    dataset = DemonstrationDataset([broken], spec=SPEC)
    diagnostics = _evaluate(
        fixture, tmp_path, dataset=dataset, paths={0: broken}
    )
    assert not diagnostics.checks["action_validity"]


def test_requires_at_least_one_path(fixture, tmp_path):
    with pytest.raises(ValueError):
        _evaluate(fixture, tmp_path, paths={})


def test_state_dimension_matches_the_specification(fixture, tmp_path):
    diagnostics = _evaluate(fixture, tmp_path)
    assert diagnostics.state_dimension == diagnostics.expected_dimension
    assert sum(diagnostics.block_sizes.values()) == diagnostics.state_dimension
    assert diagnostics.block_sizes["covariance"] == N_ASSETS * (N_ASSETS + 1) // 2
    assert diagnostics.block_sizes["return_window"] == SPEC.lookback * N_ASSETS
    assert np.isfinite(diagnostics.simplex_violation)
