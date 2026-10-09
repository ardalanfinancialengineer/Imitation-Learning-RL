"""Tests for the DAgger loop: aggregation, beta schedule, and retraining."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.dataset.demonstrations import DemonstrationDataset
from src.dataset.paths import generate_path
from src.dataset.state import StateSpec
from src.expert.constraints import PortfolioConstraints
from src.imitation.bc import BCConfig, BehaviouralCloningTrainer, seed_everything
from src.imitation.dagger import (
    AggregatedDataset,
    AggregatedRound,
    DAggerConfig,
    DAggerTrainer,
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
def datasets():
    preset = canonical_three_regime_preset(N_ASSETS)
    paths = [
        generate_path(
            path_id=index,
            seed=42 + index,
            preset=preset,
            periods=430,
            expert_config=EXPERT_CONFIG,
        )
        for index in range(4)
    ]
    return (
        DemonstrationDataset(paths[:3], spec=SPEC),
        DemonstrationDataset(paths[3:], spec=SPEC),
    )


def _policy():
    seed_everything(0)
    return build_policy(
        {"kind": "mlp", "hidden_dimensions": [32, 16]},
        spec=SPEC,
        n_assets=N_ASSETS,
        n_regimes=3,
        constraints=CONSTRAINTS,
    )


def _cloned(datasets, epochs: int = 4):
    train, validation = datasets
    policy = _policy()
    trainer = BehaviouralCloningTrainer(
        policy, BCConfig(epochs=epochs, batch_size=128, seed=0)
    )
    trainer.fit(train, validation)
    return policy


# ----------------------------------------------------------------------
# Beta schedule
# ----------------------------------------------------------------------
def test_exponential_beta_decays_towards_learner_only_rollouts():
    config = DAggerConfig(beta_schedule="exponential", beta_decay=0.5, iterations=4)
    betas = [config.beta(index) for index in range(1, 5)]
    assert betas == [0.5, 0.25, 0.125, 0.0625]
    assert all(later < earlier for earlier, later in zip(betas, betas[1:]))


def test_zero_and_constant_beta_schedules():
    assert DAggerConfig(beta_schedule="zero").beta(1) == 0.0
    config = DAggerConfig(beta_schedule="constant", beta_constant=0.3)
    assert config.beta(1) == config.beta(9) == 0.3


def test_rejects_invalid_dagger_configuration():
    with pytest.raises(ValueError):
        DAggerConfig(iterations=0)
    with pytest.raises(ValueError):
        DAggerConfig(beta_schedule="linear")
    with pytest.raises(ValueError):
        DAggerConfig(query_fraction=0.0)
    with pytest.raises(ValueError):
        DAggerConfig(beta_decay=1.5)


# ----------------------------------------------------------------------
# Aggregated dataset
# ----------------------------------------------------------------------
def test_aggregate_starts_as_the_base_dataset(datasets):
    train, _ = datasets
    aggregate = AggregatedDataset(train)
    assert len(aggregate) == len(train)
    assert aggregate.n_aggregated == 0
    assert aggregate.state_dimension == train.state_dimension


def test_aggregate_preserves_the_base_samples_after_a_round(datasets):
    train, _ = datasets
    aggregate = AggregatedDataset(train)
    entry = train.entries[0]
    rows = entry.indices[:5]
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
    assert len(aggregate) == len(train) + rows.size
    for index in (0, len(train) // 2, len(train) - 1):
        state, action = aggregate.sample(index)
        base_state, base_action = train.sample(index)
        np.testing.assert_array_equal(state, base_state)
        np.testing.assert_array_equal(action, base_action)


def test_aggregated_samples_use_the_recorded_anchor(datasets):
    train, _ = datasets
    aggregate = AggregatedDataset(train)
    entry = train.entries[0]
    rows = entry.indices[:3]
    anchors = np.tile(np.array([0.4, 0.3, 0.2, 0.1]), (rows.size, 1))
    actions = np.tile(np.array([0.25, 0.25, 0.25, 0.25]), (rows.size, 1))
    aggregate.add_round(
        [
            AggregatedRound(
                path_id=entry.data.path_id,
                sample_indices=rows,
                anchors=anchors,
                expert_actions=actions,
            )
        ]
    )

    state, action = aggregate.sample(len(train))
    block = train.assembler.block_slices()["previous_weights"]
    np.testing.assert_allclose(state[block], anchors[0])
    np.testing.assert_array_equal(action, actions[0])


def test_aggregate_rejects_paths_outside_the_base_dataset(datasets):
    train, _ = datasets
    aggregate = AggregatedDataset(train)
    with pytest.raises(KeyError):
        aggregate.add_round(
            [
                AggregatedRound(
                    path_id=999,
                    sample_indices=np.array([0]),
                    anchors=np.full((1, N_ASSETS), 0.25),
                    expert_actions=np.full((1, N_ASSETS), 0.25),
                )
            ]
        )


def test_aggregate_yields_tensors_for_a_dataloader(datasets):
    train, _ = datasets
    aggregate = AggregatedDataset(train)
    state, action = aggregate[0]
    assert isinstance(state, torch.Tensor)
    assert state.shape == (train.state_dimension,)
    assert action.shape == (N_ASSETS,)


# ----------------------------------------------------------------------
# The loop
# ----------------------------------------------------------------------
def test_fit_records_round_zero_as_the_cloned_policy(datasets):
    train, validation = datasets
    policy = _cloned(datasets)
    trainer = DAggerTrainer(
        policy,
        bc_config=BCConfig(epochs=3, batch_size=128, seed=0),
        config=DAggerConfig(iterations=1, epochs_per_round=2, query_fraction=0.1),
    )
    history, _ = trainer.fit(train, validation)

    assert history.records[0].iteration == 0
    assert history.records[0].queries == 0
    assert history.records[0].aggregate_samples == len(train)
    assert len(history.records) == 2


def test_each_round_queries_the_expert_and_grows_the_aggregate(datasets):
    train, validation = datasets
    policy = _cloned(datasets)
    trainer = DAggerTrainer(
        policy,
        bc_config=BCConfig(epochs=3, batch_size=128, seed=0),
        config=DAggerConfig(iterations=2, epochs_per_round=2, query_fraction=0.1),
    )
    history, aggregate = trainer.fit(train, validation)

    sizes = [record.aggregate_samples for record in history.records]
    assert all(later > earlier for earlier, later in zip(sizes, sizes[1:]))
    assert history.total_queries > 0
    assert aggregate.n_aggregated == history.total_queries


def test_query_fraction_controls_how_many_states_are_labelled(datasets):
    train, validation = datasets
    policy = _cloned(datasets)
    small = DAggerTrainer(
        policy,
        bc_config=BCConfig(epochs=2, batch_size=128, seed=0),
        config=DAggerConfig(iterations=1, epochs_per_round=1, query_fraction=0.1),
    ).fit(train, validation)[0]
    large = DAggerTrainer(
        _cloned(datasets),
        bc_config=BCConfig(epochs=2, batch_size=128, seed=0),
        config=DAggerConfig(iterations=1, epochs_per_round=1, query_fraction=0.4),
    ).fit(train, validation)[0]
    assert large.total_queries > small.total_queries


def test_query_budget_is_respected(datasets):
    train, validation = datasets
    policy = _cloned(datasets)
    history, _ = DAggerTrainer(
        policy,
        bc_config=BCConfig(epochs=2, batch_size=128, seed=0),
        config=DAggerConfig(
            iterations=1,
            epochs_per_round=1,
            query_fraction=1.0,
            max_queries_per_round=17,
        ),
    ).fit(train, validation)
    assert history.records[1].queries <= 17


def test_the_returned_policy_is_the_best_round_not_the_last(datasets):
    """Round-to-round variation is real, so the loop selects on closed-loop error."""
    train, validation = datasets
    policy = _cloned(datasets)
    trainer = DAggerTrainer(
        policy,
        bc_config=BCConfig(epochs=3, batch_size=128, seed=0),
        config=DAggerConfig(iterations=3, epochs_per_round=2, query_fraction=0.1),
    )
    history, _ = trainer.fit(train, validation)

    errors = [record.closed_loop_tracking_error for record in history.records]
    assert history.best_closed_loop_error == pytest.approx(min(errors))
    assert history.best_iteration == int(np.argmin(errors))

    final = trainer.evaluate_closed_loop(validation)["metrics"]
    assert final["closed_loop_tracking_error"] == pytest.approx(
        history.best_closed_loop_error, rel=1e-9
    )


def test_the_standardiser_is_not_refitted_during_the_loop(datasets):
    train, validation = datasets
    policy = _cloned(datasets)
    before = policy.standardiser.describe()

    DAggerTrainer(
        policy,
        bc_config=BCConfig(epochs=2, batch_size=128, seed=0),
        config=DAggerConfig(iterations=1, epochs_per_round=1, query_fraction=0.1),
    ).fit(train, validation)

    assert policy.standardiser.describe() == before


def test_separate_oracles_are_used_for_training_and_validation(datasets):
    """Training and validation hold disjoint paths, so one oracle cannot serve both."""
    train, validation = datasets
    train_oracle = ExpertOracle(train)
    assert set(train_oracle.path_ids).isdisjoint(
        ExpertOracle(validation).path_ids
    )

    policy = _cloned(datasets)
    history, _ = DAggerTrainer(
        policy,
        bc_config=BCConfig(epochs=2, batch_size=128, seed=0),
        config=DAggerConfig(iterations=1, epochs_per_round=1, query_fraction=0.1),
    ).fit(train, validation, oracle=train_oracle)
    assert len(history.records) == 2
