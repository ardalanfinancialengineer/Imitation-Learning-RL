"""Tests for the behavioural-cloning objectives, training loop, and checkpoints."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.dataset.demonstrations import DemonstrationDataset
from src.dataset.paths import PathData, generate_path
from src.dataset.state import StateSpec
from src.expert.constraints import PortfolioConstraints
from src.imitation.bc import (
    BCConfig,
    BehaviouralCloningTrainer,
    TensorPairDataset,
    estimated_megabytes,
    materialise,
    seed_everything,
)
from src.imitation.losses import (
    allocation_l1,
    allocation_mse,
    get_loss,
    weight_divergence,
)
from src.policy.builder import build_policy
from src.simulation.presets import canonical_three_regime_preset

N_ASSETS = 4
N_REGIMES = 3
SPEC = StateSpec(lookback=10)
CONSTRAINTS = PortfolioConstraints(
    long_only=True, min_weight=0.0, max_weight=0.5, leverage_limit=1.0
)

# A rolling forecaster keeps the fixture cheap: the regime-mixture estimator
# needs several hundred periods of burn-in before it produces any decision.
EXPERT_CONFIG = {
    "horizon": 5,
    "risk_aversion": 5.0,
    "transaction_cost": 0.0025,
    "constraints": {
        "long_only": True,
        "min_weight": 0.0,
        "max_weight": 0.5,
        "leverage_limit": 1.0,
    },
    "moments": {"estimator": "rolling", "covariance": "ledoit_wolf", "window": 60},
}

TINY_POLICY = {"kind": "mlp", "hidden_dimensions": [32], "dropout": 0.0}


@pytest.fixture(scope="module")
def paths() -> list[PathData]:
    preset = canonical_three_regime_preset(N_ASSETS)
    return [
        generate_path(
            path_id=index,
            seed=100 + index,
            preset=preset,
            periods=260,
            expert_config=EXPERT_CONFIG,
        )
        for index in range(3)
    ]


@pytest.fixture(scope="module")
def datasets(paths) -> dict[str, DemonstrationDataset]:
    return {
        "train": DemonstrationDataset(paths[:2], spec=SPEC),
        "validation": DemonstrationDataset(paths[2:], spec=SPEC),
    }


def make_policy():
    return build_policy(
        dict(TINY_POLICY),
        spec=SPEC,
        n_assets=N_ASSETS,
        n_regimes=N_REGIMES,
        constraints=CONSTRAINTS,
    )


# ----------------------------------------------------------------------
# Objectives
# ----------------------------------------------------------------------
def test_mse_is_the_mean_squared_l2_norm_not_the_elementwise_mean():
    """``L_BC`` sums over assets; averaging would shrink it by a factor of N."""
    predicted = torch.zeros(2, 4)
    target = torch.full((2, 4), 0.5)
    assert float(allocation_mse(predicted, target)) == pytest.approx(1.0)
    assert float(allocation_l1(predicted, target)) == pytest.approx(2.0)


def test_weight_divergence_is_zero_for_identical_allocations():
    weights = torch.as_tensor([[0.5, 0.5, 0.0, 0.0], [0.25, 0.25, 0.25, 0.25]])
    assert float(weight_divergence(weights, weights)) == pytest.approx(0.0, abs=1e-6)


def test_weight_divergence_tolerates_the_exact_zeros_experts_produce():
    """Long-only expert allocations are mostly exact zeros; an unguarded log is undefined."""
    target = torch.as_tensor([[0.6, 0.4, 0.0, 0.0]])
    predicted = torch.as_tensor([[0.3, 0.3, 0.2, 0.2]])
    value = weight_divergence(predicted, target)
    assert torch.isfinite(value) and float(value) > 0.0


def test_weight_divergence_punishes_missing_an_expert_position():
    target = torch.as_tensor([[1.0, 0.0, 0.0, 0.0]])
    close = torch.as_tensor([[0.9, 0.1, 0.0, 0.0]])
    far = torch.as_tensor([[0.1, 0.3, 0.3, 0.3]])
    assert float(weight_divergence(far, target)) > float(weight_divergence(close, target))


def test_losses_validate_their_shapes():
    with pytest.raises(ValueError, match="same shape"):
        allocation_mse(torch.zeros(2, 4), torch.zeros(2, 5))
    with pytest.raises(ValueError, match="unknown loss"):
        get_loss("hinge")


# ----------------------------------------------------------------------
# Data plumbing
# ----------------------------------------------------------------------
def test_materialise_reproduces_the_lazy_dataset(datasets):
    states, actions = materialise(datasets["train"])
    assert states.shape == (len(datasets["train"]), datasets["train"].state_dimension)

    for index in (0, len(datasets["train"]) // 2, len(datasets["train"]) - 1):
        state, action = datasets["train"].sample(index)
        assert np.allclose(states[index].numpy(), state, atol=1e-6)
        assert np.allclose(actions[index].numpy(), action, atol=1e-6)


def test_cache_is_skipped_when_the_dataset_exceeds_the_budget(datasets):
    policy = make_policy()
    trainer = BehaviouralCloningTrainer(
        policy, BCConfig(epochs=1, cache_budget_megabytes=1e-9)
    )
    source, cached = trainer._source(datasets["train"])
    assert not cached
    assert source is datasets["train"]

    generous = BehaviouralCloningTrainer(
        make_policy(), BCConfig(epochs=1, cache_budget_megabytes=1e6)
    )
    source, cached = generous._source(datasets["train"])
    assert cached and isinstance(source, TensorPairDataset)


def test_estimated_footprint_matches_the_materialised_tensors(datasets):
    states, actions = materialise(datasets["train"])
    actual = (states.numel() + actions.numel()) * 4 / 1e6
    assert estimated_megabytes(datasets["train"]) == pytest.approx(actual)


def test_caching_does_not_change_the_evaluation_result(datasets):
    """Whether states are cached is a performance choice, never a numerical one."""
    seed_everything(0)
    cached_trainer = BehaviouralCloningTrainer(
        make_policy(), BCConfig(epochs=2, batch_size=64, seed=0)
    )
    cached_trainer.fit(datasets["train"], datasets["validation"])
    cached_metrics = cached_trainer.evaluate(datasets["validation"])

    seed_everything(0)
    lazy_trainer = BehaviouralCloningTrainer(
        make_policy(),
        BCConfig(epochs=2, batch_size=64, seed=0, cache_budget_megabytes=1e-9),
    )
    lazy_trainer.fit(datasets["train"], datasets["validation"])
    lazy_metrics = lazy_trainer.evaluate(datasets["validation"])

    assert cached_metrics["tracking_error"] == pytest.approx(
        lazy_metrics["tracking_error"], rel=1e-9
    )


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------
def test_training_reduces_the_objective(datasets):
    trainer = BehaviouralCloningTrainer(
        make_policy(), BCConfig(epochs=8, batch_size=64, seed=0, patience=8)
    )
    history = trainer.fit(datasets["train"], datasets["validation"])

    assert len(history.records) >= 1
    assert history.records[-1].train_loss < history.records[0].train_loss
    assert history.best_epoch >= 1
    assert np.isfinite(history.best_tracking_error)


def test_training_is_reproducible_under_a_fixed_seed(datasets):
    def run() -> np.ndarray:
        # Seeding must precede construction: the weights are drawn there, and
        # fit() seeding alone would leave the initialisation to the caller.
        seed_everything(7)
        trainer = BehaviouralCloningTrainer(
            make_policy(), BCConfig(epochs=3, batch_size=64, seed=7)
        )
        trainer.fit(datasets["train"], datasets["validation"])
        return trainer.policy.act(
            np.stack([datasets["validation"].sample(i)[0] for i in range(8)])
        )

    assert np.array_equal(run(), run())


def test_standardiser_is_fitted_on_training_data_only(datasets):
    """Validation statistics must not reach the policy's inputs."""
    trainer = BehaviouralCloningTrainer(
        make_policy(), BCConfig(epochs=1, batch_size=64, seed=0)
    )
    trainer.fit(datasets["train"], datasets["validation"])
    fitted = trainer.policy.standardiser.mean.clone()

    reference = make_policy()
    states, _ = materialise(datasets["train"])
    reference.standardiser.fit(
        states.numpy(), spec=SPEC, n_assets=N_ASSETS, n_regimes=N_REGIMES
    )
    assert torch.allclose(fitted, reference.standardiser.mean, atol=1e-6)


def test_early_stopping_halts_a_run_that_stops_improving(datasets):
    trainer = BehaviouralCloningTrainer(
        make_policy(), BCConfig(epochs=50, batch_size=64, seed=0, patience=1)
    )
    history = trainer.fit(datasets["train"], datasets["validation"])
    assert history.stopped_early
    assert len(history.records) < 50


def test_evaluation_reports_feasible_actions_and_a_matching_sample_count(datasets):
    trainer = BehaviouralCloningTrainer(
        make_policy(), BCConfig(epochs=2, batch_size=64, seed=0)
    )
    trainer.fit(datasets["train"], datasets["validation"])
    metrics = trainer.evaluate(datasets["validation"])

    assert metrics["n_samples"] == len(datasets["validation"])
    assert metrics["feasibility_violation"] <= 1e-6
    assert metrics["tracking_error"] >= 0.0
    assert metrics["expert_turnover"] > 0.0


@pytest.mark.parametrize("loss", ["mse", "l1", "weight_divergence"])
def test_every_objective_trains_and_yields_feasible_actions(datasets, loss):
    trainer = BehaviouralCloningTrainer(
        make_policy(), BCConfig(epochs=3, batch_size=64, seed=0, loss=loss)
    )
    history = trainer.fit(datasets["train"], datasets["validation"])
    metrics = trainer.evaluate(datasets["validation"])

    assert np.isfinite(history.records[-1].train_loss)
    assert metrics["feasibility_violation"] <= 1e-6


def test_diffusion_policy_trains_its_denoising_objective(datasets):
    """The configured allocation loss is irrelevant to a diffusion training step."""
    policy = build_policy(
        {
            "kind": "diffusion",
            "encoder": {"d_model": 32, "n_heads": 4, "n_layers": 1, "dropout": 0.0},
            "diffusion": {"n_steps": 20, "hidden_dimension": 64, "n_blocks": 2,
                          "sampling_steps": 4},
        },
        spec=SPEC,
        n_assets=N_ASSETS,
        n_regimes=N_REGIMES,
        constraints=CONSTRAINTS,
    )
    trainer = BehaviouralCloningTrainer(
        policy, BCConfig(epochs=3, batch_size=64, seed=0, patience=3)
    )
    history = trainer.fit(datasets["train"], datasets["validation"])

    # Calibration replaced the default scale with a measured one.
    assert float(policy.action_scale) != pytest.approx(0.05)
    assert history.records[-1].train_loss < history.records[0].train_loss
    assert trainer.evaluate(datasets["validation"])["feasibility_violation"] <= 1e-6


def test_config_rejects_invalid_settings():
    with pytest.raises(ValueError, match="unknown loss"):
        BCConfig(loss="hinge")
    with pytest.raises(ValueError, match="epochs"):
        BCConfig(epochs=0)
    with pytest.raises(ValueError, match="learning_rate"):
        BCConfig(learning_rate=0.0)


# ----------------------------------------------------------------------
# Persistence
# ----------------------------------------------------------------------
def test_checkpoint_round_trips_the_policy_and_its_standardiser(datasets, tmp_path):
    trainer = BehaviouralCloningTrainer(
        make_policy(), BCConfig(epochs=2, batch_size=64, seed=0)
    )
    trainer.fit(datasets["train"], datasets["validation"])
    file_path = tmp_path / "policy.pt"
    trainer.save(file_path, extra={"state_spec": SPEC.to_dict()})

    checkpoint = torch.load(file_path, weights_only=False)
    restored = make_policy()
    restored.load_state_dict(checkpoint["policy_state_dict"])

    states = np.stack([datasets["validation"].sample(i)[0] for i in range(8)])
    assert np.array_equal(trainer.policy.act(states), restored.act(states))
    assert checkpoint["extra"]["state_spec"]["lookback"] == SPEC.lookback
    assert checkpoint["bc_config"]["seed"] == 0
