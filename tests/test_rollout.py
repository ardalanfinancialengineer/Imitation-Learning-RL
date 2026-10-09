"""Tests for closed-loop rollouts and expert queries at learner-visited states."""

from __future__ import annotations

import numpy as np
import pytest

from src.dataset.demonstrations import DemonstrationDataset
from src.dataset.paths import PathData, generate_path
from src.dataset.state import StateSpec
from src.expert.constraints import PortfolioConstraints
from src.imitation.rollout import (
    ExpertOracle,
    answer_rollouts,
    per_path_closed_loop_errors,
    rollout_metrics,
    rollout_policy,
)
from src.policy.reference import EqualWeightReference, PreviousWeightsReference
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
def path() -> PathData:
    return generate_path(
        path_id=0,
        seed=42,
        preset=canonical_three_regime_preset(N_ASSETS),
        periods=450,
        expert_config=EXPERT_CONFIG,
    )


@pytest.fixture(scope="module")
def dataset(path) -> DemonstrationDataset:
    return DemonstrationDataset([path], spec=SPEC)


@pytest.fixture(scope="module")
def oracle(dataset) -> ExpertOracle:
    return ExpertOracle(dataset)


def _reference(kind: str = "previous"):
    factory = PreviousWeightsReference if kind == "previous" else EqualWeightReference
    return factory(SPEC, n_assets=N_ASSETS, n_regimes=3, constraints=CONSTRAINTS)


def test_substituting_the_experts_own_anchor_reproduces_the_stored_state(
    dataset, path
):
    """The state patch must be a no-op when the anchor is the expert's own."""
    entry = dataset.entries[0]
    for position in (0, 10, entry.indices.size - 1):
        sample_index = int(entry.indices[position])
        stored = dataset.assemble_state(entry, sample_index)
        patched = dataset.assemble_state(
            entry, sample_index, previous_weights=path.previous_weights[sample_index]
        )
        np.testing.assert_array_equal(patched, stored)


def test_substituting_a_different_anchor_changes_only_that_block(dataset):
    entry = dataset.entries[0]
    sample_index = int(entry.indices[5])
    stored = dataset.assemble_state(entry, sample_index)
    patched = dataset.assemble_state(
        entry, sample_index, previous_weights=np.full(N_ASSETS, 1.0 / N_ASSETS)
    )

    slices = dataset.assembler.block_slices()
    for name, block in slices.items():
        if name == "previous_weights":
            continue
        np.testing.assert_array_equal(
            patched[block], stored[block]
        ), f"block {name} changed"


def test_oracle_at_the_experts_own_anchor_reproduces_the_stored_action(
    oracle, path, dataset
):
    """The DAgger query must be the same programme the demonstrations came from."""
    entry = dataset.entries[0]
    for position in (0, 7, entry.indices.size - 1):
        sample_index = int(entry.indices[position])
        answer = oracle.query(
            path.path_id, sample_index, path.previous_weights[sample_index]
        )
        np.testing.assert_array_equal(answer, path.expert_weights[sample_index])


def test_oracle_answers_differ_at_a_different_anchor(oracle, path, dataset):
    entry = dataset.entries[0]
    sample_index = int(entry.indices[10])
    far = np.zeros(N_ASSETS)
    far[0] = 0.4
    far[1] = 0.4
    far[2] = 0.2
    answer = oracle.query(path.path_id, sample_index, far)
    assert not np.allclose(answer, path.expert_weights[sample_index])
    assert CONSTRAINTS.is_feasible(answer)


def test_oracle_rejects_paths_it_does_not_cover(oracle):
    with pytest.raises(KeyError, match="not covered"):
        oracle.query(999, 0, np.full(N_ASSETS, 0.25))


def test_rollout_chains_the_policys_own_previous_weights(dataset):
    rollout = rollout_policy(_reference("equal"), dataset)[0]
    np.testing.assert_allclose(
        rollout.policy_previous_weights[1:], rollout.policy_weights[:-1]
    )


def test_never_trade_rollout_holds_its_initial_portfolio(dataset):
    rollout = rollout_policy(_reference("previous"), dataset)[0]
    assert np.allclose(rollout.turnover, 0.0, atol=1e-9)
    assert np.allclose(rollout.policy_weights, rollout.policy_weights[0], atol=1e-9)


def test_rollout_realises_returns_after_the_decision(dataset, path):
    rollout = rollout_policy(_reference("equal"), dataset)[0]
    step = 5
    t = int(rollout.decision_times[step])
    expected = float(rollout.policy_weights[step] @ path.returns[t])
    assert rollout.gross_returns[step] == pytest.approx(expected)
    cost = float(path.metadata["expert"]["transaction_cost"])
    assert rollout.net_returns[step] == pytest.approx(
        expected - cost * rollout.turnover[step]
    )


def test_rollout_is_deterministic(dataset):
    first = rollout_policy(_reference("equal"), dataset)[0]
    second = rollout_policy(_reference("equal"), dataset)[0]
    np.testing.assert_array_equal(first.policy_weights, second.policy_weights)


def test_anchor_drift_measures_distance_from_the_experts_anchor(dataset):
    rollout = rollout_policy(_reference("equal"), dataset)[0]
    expected = np.abs(
        rollout.policy_previous_weights - rollout.expert_previous_weights
    ).sum(axis=1)
    np.testing.assert_allclose(rollout.anchor_drift, expected)


def test_beta_one_reproduces_the_experts_trajectory(dataset, path):
    """With beta = 1 every step defers to the expert, so the rollout is its path."""
    rollouts = rollout_policy(
        _reference("equal"),
        dataset,
        beta=1.0,
        generator=np.random.default_rng(0),
    )
    entry = dataset.entries[0]
    np.testing.assert_array_equal(
        rollouts[0].policy_weights, path.expert_weights[entry.indices]
    )


def test_beta_zero_never_defers(dataset):
    with_generator = rollout_policy(
        _reference("equal"), dataset, beta=0.0, generator=np.random.default_rng(0)
    )[0]
    without = rollout_policy(_reference("equal"), dataset)[0]
    np.testing.assert_array_equal(
        with_generator.policy_weights, without.policy_weights
    )


def test_beta_requires_a_generator(dataset):
    with pytest.raises(ValueError, match="generator"):
        rollout_policy(_reference("equal"), dataset, beta=0.5)


def test_rejects_a_beta_outside_the_unit_interval(dataset):
    with pytest.raises(ValueError):
        rollout_policy(_reference("equal"), dataset, beta=1.5)


def test_closed_loop_error_uses_the_experts_answer_at_the_visited_state(
    dataset, oracle
):
    """Closed-loop and trajectory tracking error are different quantities."""
    rollouts = rollout_policy(_reference("previous"), dataset)
    answers = answer_rollouts(oracle, rollouts)
    metrics = rollout_metrics(rollouts, answers)

    assert "closed_loop_tracking_error" in metrics
    assert metrics["closed_loop_tracking_error"] != pytest.approx(
        metrics["trajectory_tracking_error"]
    )


def test_metrics_omit_closed_loop_error_without_expert_answers(dataset):
    metrics = rollout_metrics(rollout_policy(_reference("equal"), dataset))
    assert "closed_loop_tracking_error" not in metrics
    assert metrics["n_paths"] == 1


def test_per_path_errors_are_keyed_by_path(dataset, oracle):
    rollouts = rollout_policy(_reference("equal"), dataset)
    answers = answer_rollouts(oracle, rollouts)
    errors = per_path_closed_loop_errors(rollouts, answers)
    assert set(errors) == {0}
    assert errors[0] > 0.0


def test_rollout_metrics_require_at_least_one_rollout():
    with pytest.raises(ValueError):
        rollout_metrics([])
