"""Tests for the Gymnasium portfolio environment."""

from __future__ import annotations

import numpy as np
import pytest

from src.dataset.paths import PathData, generate_path
from src.dataset.state import StateSpec
from src.reinforcement.environment import PortfolioEnv
from src.simulation.presets import canonical_three_regime_preset

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

SPEC = StateSpec(lookback=20)


@pytest.fixture(scope="module")
def path() -> PathData:
    return generate_path(
        path_id=0,
        seed=42,
        preset=canonical_three_regime_preset(4),
        periods=400,
        expert_config=EXPERT_CONFIG,
    )


def test_spaces_match_the_state_and_action_definitions(path):
    environment = PortfolioEnv(path, spec=SPEC)
    assert environment.observation_space.shape == (SPEC.dimension(4, 3),)
    assert environment.action_space.shape == (4,)
    assert environment.action_space.low.min() == 0.0, "long-only lower bound"


def test_reset_returns_a_valid_observation(path):
    environment = PortfolioEnv(path, spec=SPEC)
    observation, info = environment.reset()
    assert environment.observation_space.contains(observation)
    assert info["step"] == 0
    assert np.isclose(info["previous_weights"].sum(), 1.0)


def test_replaying_the_expert_reproduces_the_stored_trajectory(path):
    """The strongest end-to-end check: env and expert must agree exactly."""
    environment = PortfolioEnv(path, spec=SPEC)
    environment.reset()

    for _ in range(environment.n_steps):
        _, _, terminated, _, info = environment.step(environment.expert_action())
        sample_index = info["sample_index"]
        assert info["net_return"] == pytest.approx(
            float(path.net_returns[sample_index]), abs=1e-12
        )
        assert info["turnover"] == pytest.approx(
            float(path.turnover[sample_index]), abs=1e-12
        )
        assert info["gross_return"] == pytest.approx(
            float(path.gross_returns[sample_index]), abs=1e-12
        )
        if terminated:
            break


def test_reward_matches_equation_eleven(path):
    environment = PortfolioEnv(path, spec=SPEC, expert_deviation_penalty=0.5)
    environment.reset()
    action = np.full(4, 0.25)

    _, reward, _, _, info = environment.step(action)
    expected = (
        info["net_return"] - info["risk_penalty"] - 0.5 * info["expert_deviation"]
    )
    assert reward == pytest.approx(expected)


def test_transaction_cost_is_charged_exactly_once(path):
    environment = PortfolioEnv(path, spec=SPEC)
    environment.reset()
    _, _, _, _, info = environment.step(np.array([1.0, 0.0, 0.0, 0.0]))
    assert info["net_return"] == pytest.approx(
        info["gross_return"] - environment.transaction_cost * info["turnover"]
    )


def test_expert_deviation_penalty_defaults_to_zero(path):
    environment = PortfolioEnv(path, spec=SPEC)
    environment.reset()
    _, reward, _, _, info = environment.step(np.full(4, 0.25))
    assert reward == pytest.approx(info["net_return"] - info["risk_penalty"])


def test_actions_are_projected_onto_the_feasible_set(path):
    environment = PortfolioEnv(path, spec=SPEC)
    environment.reset()
    _, _, _, _, info = environment.step(np.array([5.0, -3.0, 0.1, 0.2]))

    weights = info["previous_weights"]
    assert np.isclose(weights.sum(), 1.0)
    assert weights.min() >= -1e-9
    assert weights.max() <= environment.constraints.max_weight + 1e-6


def test_episode_terminates_after_the_final_decision(path):
    environment = PortfolioEnv(path, spec=SPEC)
    environment.reset()
    for step in range(environment.n_steps):
        _, _, terminated, truncated, _ = environment.step(np.full(4, 0.25))
        assert truncated is False
        assert terminated == (step == environment.n_steps - 1)

    with pytest.raises(RuntimeError):
        environment.step(np.full(4, 0.25))


def test_observations_never_depend_on_the_realised_return(path):
    """Stepping must not change the observation for a date already passed."""
    environment = PortfolioEnv(path, spec=SPEC)
    first, _ = environment.reset()
    environment.step(np.full(4, 0.25))
    replay, _ = environment.reset()
    np.testing.assert_array_equal(first, replay)


def test_reset_accepts_an_initial_portfolio(path):
    environment = PortfolioEnv(path, spec=SPEC)
    _, info = environment.reset(options={"initial_weights": np.array([0.4, 0.3, 0.2, 0.1])})
    np.testing.assert_allclose(info["previous_weights"], [0.4, 0.3, 0.2, 0.1], atol=1e-9)


def test_decision_slice_restricts_the_episode(path):
    full = PortfolioEnv(path, spec=SPEC)
    partial = PortfolioEnv(path, spec=SPEC, decision_slice=slice(0, 30))
    assert partial.n_steps < full.n_steps
    assert partial.n_steps <= 30


def test_rejects_a_negative_expert_deviation_penalty(path):
    with pytest.raises(ValueError):
        PortfolioEnv(path, spec=SPEC, expert_deviation_penalty=-1.0)
