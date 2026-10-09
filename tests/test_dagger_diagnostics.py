"""Tests for the DAgger acceptance checks."""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.dataset.demonstrations import DemonstrationDataset
from src.dataset.paths import generate_path
from src.dataset.state import StateSpec
from src.expert.constraints import PortfolioConstraints
from src.imitation.bc import BCConfig, BehaviouralCloningTrainer, seed_everything
from src.imitation.dagger import AggregatedDataset, AggregatedRound
from src.imitation.dagger_diagnostics import (
    DAggerDiagnosticThresholds,
    evaluate_dagger,
    improvement_interval,
)
from src.imitation.rollout import ExpertOracle
from src.policy.builder import build_policy
from src.simulation.presets import canonical_three_regime_preset

N_ASSETS = 4
SPEC = StateSpec(lookback=20)
EXPERT_CONFIG = {
    "horizon": 5,
    "risk_aversion": 5.0,
    "transaction_cost": 0.0025,
    "constraints": {"long_only": True, "min_weight": 0.0, "max_weight": 0.4},
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
CONSTRAINTS = PortfolioConstraints(long_only=True, min_weight=0.0, max_weight=0.4)


@pytest.fixture(scope="module")
def fixture():
    preset = canonical_three_regime_preset(N_ASSETS)
    paths = [
        generate_path(
            path_id=index,
            seed=42 + index,
            preset=preset,
            periods=420,
            expert_config=EXPERT_CONFIG,
        )
        for index in range(3)
    ]
    train = DemonstrationDataset(paths[:2], spec=SPEC)
    validation = DemonstrationDataset(paths[2:], spec=SPEC)

    seed_everything(0)
    policy = build_policy(
        {"kind": "mlp", "hidden_dimensions": [32, 16]},
        spec=SPEC,
        n_assets=N_ASSETS,
        n_regimes=3,
        constraints=CONSTRAINTS,
    )
    BehaviouralCloningTrainer(
        policy, BCConfig(epochs=3, batch_size=128, seed=0)
    ).fit(train, validation)

    aggregate = AggregatedDataset(train)
    entry = train.entries[0]
    rows = entry.indices[:6]
    aggregate.add_round(
        [
            AggregatedRound(
                path_id=entry.data.path_id,
                sample_indices=rows,
                anchors=np.full((rows.size, N_ASSETS), 1.0 / N_ASSETS),
                expert_actions=np.full((rows.size, N_ASSETS), 1.0 / N_ASSETS),
            )
        ]
    )
    return preset, policy, train, aggregate, ExpertOracle(train)


def _evaluate(fixture, **kwargs):
    preset, policy, train, aggregate, oracle = fixture
    return evaluate_dagger(
        policy,
        train,
        aggregate,
        oracle,
        preset=preset,
        aggregate_sizes=kwargs.pop("aggregate_sizes", (len(train), len(aggregate))),
        samples=kwargs.pop("samples", 2),
        **kwargs,
    )


def test_a_correct_dagger_run_passes_every_check(fixture):
    diagnostics = _evaluate(fixture)
    assert diagnostics.passed, diagnostics.checks
    assert set(diagnostics.checks) == {
        "expert_query_consistency",
        "state_patch_exactness",
        "aggregation_growth",
        "closed_loop_feasibility",
        "rollout_no_look_ahead",
        "rollout_determinism",
    }


def test_the_query_and_patch_checks_hold_exactly(fixture):
    """Both are gated at zero tolerance, not merely 'close'."""
    diagnostics = _evaluate(fixture)
    assert diagnostics.query_difference == 0.0
    assert diagnostics.state_patch_difference == 0.0


def test_the_rollout_is_causal_at_zero_tolerance(fixture):
    diagnostics = _evaluate(fixture)
    assert diagnostics.look_ahead_difference == 0.0
    assert diagnostics.look_ahead_decision_times


def test_a_shrinking_aggregate_fails_the_growth_check(fixture):
    diagnostics = _evaluate(fixture, aggregate_sizes=(900, 800))
    assert not diagnostics.checks["aggregation_growth"]
    assert not diagnostics.passed


def test_a_single_size_fails_the_growth_check(fixture):
    assert not _evaluate(fixture, aggregate_sizes=(900,)).checks["aggregation_growth"]


def test_an_unreachable_feasibility_tolerance_fails_that_check(fixture):
    diagnostics = _evaluate(
        fixture,
        thresholds=DAggerDiagnosticThresholds(max_feasibility_violation=0.0),
    )
    assert diagnostics.checks["closed_loop_feasibility"] == (
        diagnostics.feasibility_violation == 0.0
    )


def test_report_is_json_serialisable(fixture):
    report = _evaluate(fixture).to_dict()
    assert json.loads(json.dumps(report))["passed"] is True


def test_rejects_negative_thresholds():
    with pytest.raises(ValueError):
        DAggerDiagnosticThresholds(max_query_difference=-1.0)


def test_improvement_interval_is_paired_over_paths():
    before = {0: 0.10, 1: 0.08, 2: 0.12}
    after = {0: 0.06, 1: 0.05, 2: 0.09}
    interval = improvement_interval(before, after)

    assert interval.mean == pytest.approx(np.mean([0.04, 0.03, 0.03]))
    assert interval.excludes_zero
    assert interval.n_units == 3


def test_improvement_interval_reports_no_gain_when_there_is_none():
    before = {0: 0.10, 1: 0.08}
    after = {0: 0.10, 1: 0.08}
    interval = improvement_interval(before, after)
    assert interval.mean == pytest.approx(0.0)
    assert not interval.excludes_zero
