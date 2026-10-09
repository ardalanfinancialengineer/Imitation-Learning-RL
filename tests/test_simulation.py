"""Unit tests for the simulation module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.simulation.regimes import RegimeSwitchingModel
from src.simulation.returns import ReturnSimulator


def _valid_regime_model() -> RegimeSwitchingModel:
    return RegimeSwitchingModel(
        n_regimes=2,
        transition_matrix=np.array(
            [
                [0.9, 0.1],
                [0.2, 0.8],
            ]
        ),
        initial_distribution=np.array([0.6, 0.4]),
    )


class TestRegimeSwitchingModel:
    def test_sample_regime_sequence_is_reproducible_and_valid(self):
        model = _valid_regime_model()

        regimes_a = model.sample_regime_sequence(T=100, seed=123)
        regimes_b = model.sample_regime_sequence(T=100, seed=123)

        np.testing.assert_array_equal(regimes_a, regimes_b)
        assert regimes_a.shape == (100,)
        assert np.issubdtype(regimes_a.dtype, np.integer)
        assert np.all((regimes_a >= 0) & (regimes_a < model.n_regimes))

    @pytest.mark.parametrize(
        "transition_matrix",
        [
            np.array([[1.0, 0.0, 0.0], [0.2, 0.8, 0.0]]),
            np.array([[0.8, 0.4], [0.2, 0.8]]),
            np.array([[0.8, -0.2], [0.2, 0.8]]),
            np.array([[1.0, np.nan], [0.2, 0.8]]),
        ],
    )
    def test_invalid_transition_matrices_raise(self, transition_matrix):
        with pytest.raises(ValueError):
            RegimeSwitchingModel(n_regimes=2, transition_matrix=transition_matrix)

    def test_invalid_initial_distribution_raises(self):
        with pytest.raises(ValueError):
            RegimeSwitchingModel(
                n_regimes=2,
                transition_matrix=np.eye(2),
                initial_distribution=np.array([0.8, 0.5]),
            )


class TestReturnSimulator:
    def test_simulate_is_reproducible_and_has_expected_shapes(self):
        simulator = ReturnSimulator(
            n_assets=3,
            regime_model=_valid_regime_model(),
            regime_means=[
                np.array([0.001, 0.002, 0.0005]),
                np.array([-0.001, 0.0003, 0.0015]),
            ],
            regime_covs=[
                np.diag([0.01, 0.02, 0.015]),
                np.array(
                    [
                        [0.02, 0.003, 0.001],
                        [0.003, 0.03, 0.002],
                        [0.001, 0.002, 0.025],
                    ]
                ),
            ],
        )

        returns_a, regimes_a = simulator.simulate(T=50, seed=7)
        returns_b, regimes_b = simulator.simulate(T=50, seed=7)

        assert isinstance(returns_a, pd.DataFrame)
        assert returns_a.shape == (50, 3)
        assert regimes_a.shape == (50,)
        pd.testing.assert_frame_equal(returns_a, returns_b)
        np.testing.assert_array_equal(regimes_a, regimes_b)
        assert np.all((regimes_a >= 0) & (regimes_a < 2))

    @pytest.mark.parametrize(
        "regime_covs",
        [
            [
                np.array([[0.01, 0.002], [0.003, 0.02]]),
                np.eye(2),
            ],
            [
                np.array([[1.0, 2.0], [2.0, 1.0]]),
                np.eye(2),
            ],
            [
                np.array([[0.01, np.inf], [np.inf, 0.02]]),
                np.eye(2),
            ],
        ],
    )
    def test_invalid_covariance_inputs_raise(self, regime_covs):
        with pytest.raises(ValueError):
            ReturnSimulator(
                n_assets=2,
                regime_model=_valid_regime_model(),
                regime_means=[np.zeros(2), np.zeros(2)],
                regime_covs=regime_covs,
            )
