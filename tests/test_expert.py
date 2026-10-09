"""Unit tests for the Markowitz expert policy."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.expert.markowitz import RollingMarkowitzExpert


class TestRollingMarkowitzExpert:
    def test_compute_weights_returns_valid_constrained_portfolio(self):
        rng = np.random.default_rng(123)
        returns_window = rng.normal(loc=0.001, scale=0.02, size=(30, 4))
        expert = RollingMarkowitzExpert(window=30, risk_aversion=1.0, max_weight=0.45)

        weights = expert.compute_weights(returns_window)

        assert weights.shape == (4,)
        assert np.all(np.isfinite(weights))
        assert np.all(weights >= -1e-8)
        assert np.isclose(weights.sum(), 1.0, atol=1e-6)
        assert np.all(weights <= 0.45 + 1e-6)

    def test_solver_failure_falls_back_to_feasible_equal_weights(self, monkeypatch):
        rng = np.random.default_rng(321)
        returns_window = rng.normal(loc=0.0, scale=0.01, size=(20, 5))
        expert = RollingMarkowitzExpert(window=20, max_weight=0.4)

        expert.compute_weights(returns_window)  # compile the programme
        programme = expert._expert._programme(5)
        monkeypatch.setattr(programme, "plan", lambda: None)

        weights = expert.compute_weights(returns_window)

        np.testing.assert_allclose(weights, np.full(5, 0.2))
        assert np.isclose(weights.sum(), 1.0)
        assert np.all(weights <= 0.4)

    def test_generate_demonstrations_uses_only_past_windows(self):
        values = np.arange(24, dtype=float).reshape(8, 3) / 100.0
        returns = pd.DataFrame(values, columns=["a", "b", "c"])
        expert = RollingMarkowitzExpert(window=3, max_weight=1.0)

        observations, actions = expert.generate_demonstrations(returns)

        assert observations.shape == (5, 3, 3)
        assert actions.shape == (5, 3)
        np.testing.assert_array_equal(observations[0], values[0:3])
        np.testing.assert_array_equal(observations[1], values[1:4])
        np.testing.assert_array_equal(observations[-1], values[4:7])
        assert not np.any(observations[-1] == values[-1])

    def test_demonstration_actions_are_reproducible_from_their_window(self):
        rng = np.random.default_rng(7)
        values = rng.normal(loc=0.0005, scale=0.01, size=(40, 4))
        returns = pd.DataFrame(values, columns=list("abcd"))
        expert = RollingMarkowitzExpert(window=10, max_weight=0.5)

        observations, actions = expert.generate_demonstrations(returns)

        for index in (0, 5, len(actions) - 1):
            replayed = expert.compute_weights(observations[index])
            np.testing.assert_allclose(replayed, actions[index], atol=1e-9)
