"""Tests for the Wasserstein critic and its Lipschitz enforcement."""

from __future__ import annotations

import pytest
import torch

from src.dataset.state import StateSpec
from src.imitation.critic import CriticConfig, WassersteinCritic
from src.policy.standardise import StateStandardiser

N_ASSETS = 4
SPEC = StateSpec(lookback=5)
STATE_DIM = SPEC.dimension(N_ASSETS, 3)


def _critic(**kwargs) -> WassersteinCritic:
    standardiser = StateStandardiser(STATE_DIM)
    torch.manual_seed(0)
    return WassersteinCritic(
        state_dimension=STATE_DIM,
        n_assets=N_ASSETS,
        standardiser=standardiser,
        config=CriticConfig(**kwargs),
    )


def _batch(n: int = 16, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    states = torch.randn(n, STATE_DIM, generator=generator)
    actions = torch.rand(n, N_ASSETS, generator=generator)
    return states, actions / actions.sum(dim=1, keepdim=True)


def test_score_returns_one_number_per_pair():
    critic = _critic()
    states, actions = _batch()
    assert critic.score(states, actions).shape == (16,)


def test_rejects_mismatched_dimensions():
    critic = _critic()
    states, actions = _batch()
    with pytest.raises(ValueError):
        critic.score(states[:, :-1], actions)
    with pytest.raises(ValueError):
        critic.score(states, actions[:, :-1])


def test_wasserstein_estimate_is_the_dual_gap():
    """Equation (9) in its dual form: the mean score difference."""
    critic = _critic()
    expert_states, expert_actions = _batch(seed=0)
    policy_states, policy_actions = _batch(seed=1)

    estimate = critic.wasserstein_estimate(
        expert_states, expert_actions, policy_states, policy_actions
    )
    expected = critic.score(expert_states, expert_actions).mean() - critic.score(
        policy_states, policy_actions
    ).mean()
    assert torch.allclose(estimate, expected)


def test_the_estimate_is_zero_between_identical_distributions():
    critic = _critic()
    states, actions = _batch()
    estimate = critic.wasserstein_estimate(states, actions, states, actions)
    assert float(estimate.detach()) == pytest.approx(0.0, abs=1e-6)


def test_the_estimate_flips_sign_when_the_arguments_swap():
    critic = _critic()
    first = _batch(seed=0)
    second = _batch(seed=1)
    forward = float(critic.wasserstein_estimate(*first, *second).detach())
    backward = float(critic.wasserstein_estimate(*second, *first).detach())
    assert forward == pytest.approx(-backward, abs=1e-6)


def test_the_penalty_shrinks_as_the_critic_approaches_unit_gradient():
    """The penalty is ``(||grad D|| - 1)^2``, so it is minimised at a norm of 1.

    The network is piecewise linear, so scaling the output layer scales every
    input gradient by the same factor. Rescaling to put the measured norm at 1
    must therefore reduce the penalty sharply.
    """
    critic = _critic()
    expert = _batch(seed=0)
    policy = _batch(seed=1)

    with torch.no_grad():
        critic.network[-1].weight.mul_(200.0)
    steep = float(critic.gradient_penalty(*expert, *policy).detach())

    measured = critic.measured_lipschitz_constant(*expert)
    with torch.no_grad():
        critic.network[-1].weight.div_(measured)
        critic.network[-1].bias.div_(measured)
    normalised = float(critic.gradient_penalty(*expert, *policy).detach())

    assert steep > 50.0
    assert normalised < steep / 20.0
    assert critic.measured_lipschitz_constant(*expert) == pytest.approx(1.0, abs=0.5)


def test_gradient_penalty_is_positive_for_a_steep_critic():
    critic = _critic()
    with torch.no_grad():
        for layer in critic.network:
            if isinstance(layer, torch.nn.Linear):
                layer.weight.mul_(50.0)
    expert = _batch(seed=0)
    policy = _batch(seed=1)
    assert float(critic.gradient_penalty(*expert, *policy).detach()) > 1.0


def test_gradient_penalty_enters_the_critic_loss():
    expert = _batch(seed=0)
    policy = _batch(seed=1)
    with_penalty = _critic(lipschitz="gradient_penalty")
    without = _critic(lipschitz="weight_clipping")

    _, first = with_penalty.critic_loss(*expert, *policy)
    _, second = without.critic_loss(*expert, *policy)
    assert first["gradient_penalty"] > 0.0
    assert second["gradient_penalty"] == 0.0
    assert second["critic_loss"] == pytest.approx(-second["wasserstein_estimate"])


def test_weight_clipping_bounds_every_parameter():
    critic = _critic(lipschitz="weight_clipping", clip_value=0.01)
    with torch.no_grad():
        for parameter in critic.network.parameters():
            parameter.add_(5.0)
    critic.enforce_constraint()
    for parameter in critic.network.parameters():
        assert float(parameter.abs().max()) <= 0.01 + 1e-9


def test_gradient_penalty_mode_does_not_clip():
    critic = _critic(lipschitz="gradient_penalty")
    with torch.no_grad():
        for parameter in critic.network.parameters():
            parameter.add_(5.0)
    critic.enforce_constraint()
    assert float(next(critic.network.parameters()).abs().max()) > 1.0


def test_measured_lipschitz_constant_is_reported_not_assumed():
    critic = _critic()
    states, actions = _batch()
    assert critic.measured_lipschitz_constant(states, actions) > 0.0


def test_rejects_invalid_configuration():
    with pytest.raises(ValueError):
        CriticConfig(lipschitz="spectral_norm")
    with pytest.raises(ValueError):
        CriticConfig(hidden_dimensions=())
    with pytest.raises(ValueError):
        CriticConfig(steps_per_round=0)
    with pytest.raises(ValueError):
        CriticConfig(learning_rate=0.0)


def test_the_critic_has_no_batch_normalisation():
    """Batch norm would make a score depend on the batch a sample landed in."""
    critic = _critic()
    assert not any(
        isinstance(layer, (torch.nn.BatchNorm1d, torch.nn.SyncBatchNorm))
        for layer in critic.network
    )


def test_scores_do_not_depend_on_the_batch():
    critic = _critic()
    states, actions = _batch(n=8)
    together = critic.score(states, actions)
    apart = torch.cat([critic.score(states[i : i + 1], actions[i : i + 1]) for i in range(8)])
    assert torch.allclose(together, apart, atol=1e-6)
