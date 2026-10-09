"""Tests for the Equation (10) and (11) reward and the stochastic actor-critic."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from src.dataset.demonstrations import DemonstrationDataset
from src.dataset.paths import generate_path
from src.dataset.state import StateSpec
from src.expert.constraints import PortfolioConstraints
from src.imitation.bc import seed_everything
from src.imitation.differentiable import build_tape
from src.policy.builder import build_policy
from src.reinforcement.reward import (
    RewardConfig,
    certainty_equivalent,
    net_returns,
    rewards,
    tape_rewards,
    utility_loss,
)
from src.reinforcement.stochastic import (
    LOG_STD_MAX,
    LOG_STD_MIN,
    ActorCriticConfig,
    GaussianActorCritic,
)
from src.simulation.presets import canonical_three_regime_preset

N_ASSETS = 4
SPEC = StateSpec(lookback=10)
CONSTRAINTS = PortfolioConstraints(long_only=True, min_weight=0.0, max_weight=0.4)
EXPERT_CONFIG = {
    "horizon": 5,
    "risk_aversion": 5.0,
    "transaction_cost": 0.0025,
    "constraints": {"long_only": True, "min_weight": 0.0, "max_weight": 0.4},
    "moments": {"estimator": "rolling", "covariance": "ledoit_wolf", "window": 60},
}


@pytest.fixture(scope="module")
def dataset() -> DemonstrationDataset:
    preset = canonical_three_regime_preset(N_ASSETS)
    paths = [
        generate_path(
            path_id=index,
            seed=42 + index,
            preset=preset,
            periods=260,
            expert_config=EXPERT_CONFIG,
        )
        for index in range(2)
    ]
    return DemonstrationDataset(paths, spec=SPEC)


# ----------------------------------------------------------------------
# Reward
# ----------------------------------------------------------------------
def test_net_return_charges_the_cost_once():
    actions = torch.tensor([[0.5, 0.5, 0.0, 0.0]])
    anchors = torch.tensor([[0.25, 0.25, 0.25, 0.25]])
    realised = torch.tensor([[0.01, -0.01, 0.02, 0.0]])

    net = net_returns(actions, anchors, realised, 0.01)
    gross = float((actions * realised).sum())
    turnover = float((actions - anchors).abs().sum())
    assert float(net) == pytest.approx(gross - 0.01 * turnover)


def test_reward_subtracts_risk_and_expert_deviation():
    actions = torch.tensor([[0.5, 0.5]])
    anchors = torch.tensor([[0.5, 0.5]])
    realised = torch.tensor([[0.01, 0.0]])
    covariance = torch.eye(2).unsqueeze(0) * 1e-4
    expert = torch.tensor([[0.4, 0.6]])
    config = RewardConfig(
        risk_aversion=5.0, transaction_cost=0.0, expert_deviation_penalty=2.0
    )

    value = rewards(actions, anchors, realised, covariance, expert, config)
    expected = (
        0.005
        - 0.5 * 5.0 * float(actions @ covariance[0] @ actions.T)
        - 2.0 * float(((actions - expert) ** 2).sum())
    )
    assert float(value) == pytest.approx(expected)


def test_turnover_is_not_penalised_twice():
    """The exposé is explicit that the cost term already prices turnover."""
    actions = torch.tensor([[1.0, 0.0]])
    anchors = torch.tensor([[0.0, 1.0]])
    realised = torch.zeros(1, 2)
    covariance = torch.zeros(1, 2, 2)
    config = RewardConfig(risk_aversion=0.0, transaction_cost=0.01)

    value = float(rewards(actions, anchors, realised, covariance, actions, config))
    turnover = float((actions - anchors).abs().sum())
    assert value == pytest.approx(-0.01 * turnover)


def test_zero_coefficients_reduce_the_reward_to_the_net_return():
    actions = torch.rand(4, N_ASSETS)
    actions = actions / actions.sum(dim=1, keepdim=True)
    anchors = torch.full((4, N_ASSETS), 1.0 / N_ASSETS)
    realised = torch.randn(4, N_ASSETS) * 0.01
    covariance = torch.eye(N_ASSETS).repeat(4, 1, 1) * 1e-4
    config = RewardConfig(risk_aversion=0.0, transaction_cost=0.0)

    value = rewards(actions, anchors, realised, covariance, actions, config)
    assert torch.allclose(value, (actions * realised).sum(dim=-1))


def test_certainty_equivalent_matches_equation_sixteen():
    values = torch.tensor([0.01, -0.02, 0.03, 0.0])
    expected = float(values.mean() - 0.5 * 5.0 * values.var(unbiased=True))
    assert certainty_equivalent(values, 5.0) == pytest.approx(expected)


def test_utility_loss_is_positive_when_the_policy_is_worse():
    expert = torch.tensor([0.01, 0.01, 0.01])
    policy = torch.tensor([0.0, 0.0, 0.0])
    assert utility_loss(expert, policy, 5.0) > 0.0


def test_tape_rewards_require_covariances(dataset):
    tape = build_tape(dataset, with_covariances=False, max_steps=4)
    with pytest.raises(ValueError, match="forecast covariances"):
        tape_rewards(
            tape,
            tape.expert_actions[:4],
            tape.expert_anchors[:4],
            RewardConfig(risk_aversion=5.0, transaction_cost=0.0025),
        )


def test_reward_is_differentiable_in_the_action(dataset):
    tape = build_tape(dataset, with_covariances=True, max_steps=4)
    actions = tape.expert_actions[:4].clone().requires_grad_(True)
    value = tape_rewards(
        tape,
        actions,
        tape.expert_anchors[:4],
        RewardConfig(risk_aversion=5.0, transaction_cost=0.0025),
    )
    value.sum().backward()
    assert actions.grad is not None
    assert float(actions.grad.abs().sum()) > 0.0


def test_rejects_negative_coefficients():
    with pytest.raises(ValueError):
        RewardConfig(risk_aversion=-1.0, transaction_cost=0.0)
    with pytest.raises(ValueError):
        RewardConfig(risk_aversion=1.0, transaction_cost=-0.1)


# ----------------------------------------------------------------------
# Actor-critic
# ----------------------------------------------------------------------
def _actor_critic(**kwargs) -> GaussianActorCritic:
    seed_everything(0)
    policy = build_policy(
        {"kind": "mlp", "hidden_dimensions": [32, 16], "dropout": 0.0},
        spec=SPEC,
        n_assets=N_ASSETS,
        n_regimes=3,
        constraints=CONSTRAINTS,
    )
    return GaussianActorCritic(policy, config=ActorCriticConfig(**kwargs))


def test_wrapping_does_not_change_the_deterministic_allocation(dataset):
    """This is what makes the hybrid arm start where imitation finished."""
    actor_critic = _actor_critic()
    tape = build_tape(dataset, max_steps=4)
    states = tape.expert_state(0)

    with torch.no_grad():
        wrapped = actor_critic.project(actor_critic.mean(states))
        direct = actor_critic.policy.allocate(states)
    assert torch.allclose(wrapped, direct, atol=1e-7)


def test_log_probability_matches_torch_distributions(dataset):
    actor_critic = _actor_critic()
    tape = build_tape(dataset, max_steps=4)
    states = tape.expert_state(0)

    with torch.no_grad():
        mean = actor_critic.mean(states)
        scale = actor_critic.standard_deviation
        latent = mean + 0.37 * scale
        ours = actor_critic.log_probability(states, latent)
        theirs = torch.distributions.Normal(mean, scale).log_prob(latent).sum(dim=-1)
    assert torch.allclose(ours, theirs, atol=1e-5)


def test_entropy_matches_the_gaussian_formula():
    actor_critic = _actor_critic(log_std_init=-2.0)
    scale = actor_critic.standard_deviation
    expected = float(
        (0.5 * math.log(2.0 * math.pi * math.e) + scale.log()).sum()
    )
    assert float(actor_critic.entropy()) == pytest.approx(expected)


def test_sampled_actions_are_always_feasible(dataset):
    actor_critic = _actor_critic(log_std_init=-1.0)
    tape = build_tape(dataset, max_steps=6)
    generator = torch.Generator().manual_seed(0)

    for step in range(tape.n_steps):
        drawn = actor_critic.sample(
            tape.expert_state(step), generator=generator
        )
        weights = drawn["action"].detach().numpy().astype(np.float64)
        assert CONSTRAINTS.is_feasible(weights)


def test_deterministic_sampling_returns_the_mean(dataset):
    actor_critic = _actor_critic()
    states = build_tape(dataset, max_steps=2).expert_state(0)
    drawn = actor_critic.sample(states, deterministic=True)
    assert torch.allclose(drawn["latent"], actor_critic.mean(states))


def test_the_standard_deviation_is_clamped():
    actor_critic = _actor_critic()
    with torch.no_grad():
        actor_critic.log_std.fill_(50.0)
    assert float(actor_critic.standard_deviation.max()) == pytest.approx(
        math.exp(LOG_STD_MAX)
    )
    with torch.no_grad():
        actor_critic.log_std.fill_(-50.0)
    assert float(actor_critic.standard_deviation.min()) == pytest.approx(
        math.exp(LOG_STD_MIN)
    )


def test_the_value_head_starts_at_zero(dataset):
    """A zero-initialised value head keeps the first advantage from being noise."""
    actor_critic = _actor_critic()
    states = build_tape(dataset, max_steps=2).expert_state(0)
    with torch.no_grad():
        assert torch.allclose(
            actor_critic.value(states), torch.zeros(states.shape[0]), atol=1e-7
        )


def test_the_policy_parameters_are_registered():
    """The wrapper must own the policy's parameters or PPO cannot update them."""
    actor_critic = _actor_critic()
    wrapper_parameters = {id(p) for p in actor_critic.parameters()}
    assert all(id(p) in wrapper_parameters for p in actor_critic.policy.parameters())


def test_rejects_a_policy_without_scores():
    from src.policy.reference import EqualWeightReference

    reference = EqualWeightReference(
        SPEC, n_assets=N_ASSETS, n_regimes=3, constraints=CONSTRAINTS
    )
    with pytest.raises(TypeError, match="scores"):
        GaussianActorCritic(reference)


def test_rejects_invalid_configuration():
    with pytest.raises(ValueError):
        ActorCriticConfig(log_std_init=10.0)
    with pytest.raises(ValueError):
        ActorCriticConfig(value_hidden_dimensions=())
