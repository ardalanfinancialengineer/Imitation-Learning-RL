"""Tests for demonstration data helpers and return normalisation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.utils.data import (
    Demonstrations,
    build_observation,
    fit_return_normalizer,
    load_demonstrations,
    normalise_returns,
    train_validation_split,
)


def test_load_demonstrations_validates_current_npz_format(tmp_path):
    observations = np.arange(24, dtype=float).reshape(4, 3, 2)
    actions = np.full((4, 2), 0.5)
    returns = np.arange(12, dtype=float).reshape(6, 2)
    regimes = np.array([0, 0, 1, 1, 0, 0])
    decision_times = np.arange(3, 7)
    path = tmp_path / "demonstrations.npz"
    np.savez_compressed(
        path,
        observations=observations,
        actions=actions,
        returns=returns,
        regimes=regimes,
        decision_times=decision_times,
    )

    demos = load_demonstrations(path)

    assert demos.n_samples == 4
    assert demos.window == 3
    assert demos.n_assets == 2
    np.testing.assert_array_equal(demos.observations, observations)
    np.testing.assert_array_equal(demos.actions, actions)
    np.testing.assert_array_equal(demos.regimes, regimes)
    np.testing.assert_array_equal(demos.decision_times, decision_times)


def test_train_validation_split_is_chronological():
    demos = Demonstrations(
        observations=np.arange(30, dtype=float).reshape(5, 3, 2),
        actions=np.full((5, 2), 0.5),
        decision_times=np.arange(3, 8),
    )

    train, validation = train_validation_split(demos, train_fraction=0.6)

    assert train.observations.shape == (3, 3, 2)
    assert validation.observations.shape == (2, 3, 2)
    np.testing.assert_array_equal(train.decision_times, np.array([3, 4, 5]))
    np.testing.assert_array_equal(validation.decision_times, np.array([6, 7]))


def test_return_normalizer_uses_training_statistics_without_validation_leakage():
    train_returns = pd.DataFrame({"asset_0": [1.0, 3.0], "asset_1": [2.0, 6.0]})
    validation_returns = pd.DataFrame({"asset_0": [5.0], "asset_1": [10.0]})

    normalizer = fit_return_normalizer(train_returns, method="zscore")
    transformed_validation = normalise_returns(validation_returns, normalizer=normalizer)

    expected = pd.DataFrame({"asset_0": [3.0], "asset_1": [3.0]})
    pd.testing.assert_frame_equal(transformed_validation, expected)


def test_minmax_normalization_handles_observation_arrays():
    train_observations = np.array(
        [
            [[1.0, 10.0], [3.0, 20.0]],
            [[5.0, 30.0], [7.0, 40.0]],
        ]
    )
    normalizer = fit_return_normalizer(train_observations, method="minmax")

    transformed = normalise_returns(np.array([[9.0, 50.0]]), normalizer=normalizer)

    np.testing.assert_allclose(transformed, np.array([[8.0 / 6.0, 40.0 / 30.0]]))


def test_build_observation_contains_window_statistics_only():
    returns_window = np.array(
        [
            [0.01, 0.02],
            [0.03, 0.04],
            [0.05, 0.06],
        ]
    )

    observation = build_observation(returns_window, include_cov=True)

    assert observation.shape == (5,)
    np.testing.assert_allclose(observation[:2], np.array([0.03, 0.04]))
    np.testing.assert_allclose(observation[2:4], returns_window.var(axis=0, ddof=1))
    assert observation[-1] == pytest.approx(1.0)
