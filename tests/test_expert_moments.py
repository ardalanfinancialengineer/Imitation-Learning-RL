"""Tests for the expert's causal moment forecasters."""

from __future__ import annotations

import numpy as np
import pytest

from src.expert.moments import (
    MomentForecast,
    RegimeMixtureForecaster,
    RollingMomentForecaster,
    estimate_covariance,
)
from src.expert.regime_filter import CausalRegimeFilter
from src.simulation.presets import canonical_three_regime_preset


def _returns(n_assets: int = 5, periods: int = 900, seed: int = 3) -> np.ndarray:
    preset = canonical_three_regime_preset(n_assets)
    simulated, _ = preset.build_simulator().simulate(T=periods, seed=seed)
    return simulated.to_numpy(dtype=float)


def test_sample_and_shrinkage_estimators_are_positive_definite():
    window = _returns(n_assets=30, periods=60)
    for method in ("sample", "ledoit_wolf"):
        covariance = estimate_covariance(window, method)
        assert covariance.shape == (30, 30)
        assert np.allclose(covariance, covariance.T)
        assert np.linalg.eigvalsh(covariance).min() > 0.0


def test_shrinkage_is_better_conditioned_than_the_sample_estimator():
    """Ledoit-Wolf exists precisely for the n_assets ~ window regime."""
    window = _returns(n_assets=40, periods=50)
    sample = np.linalg.cond(estimate_covariance(window, "sample"))
    shrunk = np.linalg.cond(estimate_covariance(window, "ledoit_wolf"))
    assert shrunk < sample


def test_estimate_covariance_rejects_unknown_method():
    with pytest.raises(ValueError):
        estimate_covariance(_returns(periods=30), "oracle")


def test_rolling_forecast_matches_the_window_moments():
    returns = _returns(periods=400)
    forecaster = RollingMomentForecaster(60)
    forecaster.prepare(returns)

    forecast = forecaster.forecast(200, 5)
    window = returns[140:200]
    assert forecast.means.shape == (5, returns.shape[1])
    assert np.allclose(forecast.means[0], window.mean(axis=0))
    assert np.allclose(forecast.covariances[0], estimate_covariance(window, "sample"))
    assert np.allclose(forecast.means[0], forecast.means[-1]), "held constant over H"


def test_rolling_forecast_ignores_returns_after_the_decision_date():
    returns = _returns(periods=400)
    corrupted = returns.copy()
    corrupted[200:] = 9.0

    baseline = RollingMomentForecaster(60)
    baseline.prepare(returns)
    perturbed = RollingMomentForecaster(60)
    perturbed.prepare(corrupted)

    assert np.allclose(
        baseline.forecast(200, 5).means, perturbed.forecast(200, 5).means
    )
    assert np.allclose(
        baseline.forecast(200, 5).covariances, perturbed.forecast(200, 5).covariances
    )


def test_regime_mixture_forecast_ignores_returns_after_the_decision_date():
    returns = _returns(periods=800)
    corrupted = returns.copy()
    corrupted[600:] = 9.0

    def build() -> RegimeMixtureForecaster:
        return RegimeMixtureForecaster(
            CausalRegimeFilter(3, burn_in=250, refit_every=250),
            blend_window=60,
            blend_weight=0.25,
        )

    baseline = build()
    baseline.prepare(returns)
    perturbed = build()
    perturbed.prepare(corrupted)

    assert np.allclose(baseline.forecast(600, 21).means, perturbed.forecast(600, 21).means)
    assert np.allclose(
        baseline.forecast(600, 21).covariances, perturbed.forecast(600, 21).covariances
    )


def test_regime_mixture_covariances_are_valid_and_horizon_dependent():
    returns = _returns(n_assets=4, periods=800)
    forecaster = RegimeMixtureForecaster(CausalRegimeFilter(3, burn_in=250, refit_every=250))
    forecaster.prepare(returns)

    forecast = forecaster.forecast(600, 21)
    assert forecast.covariances.shape == (21, 4, 4)
    for step in forecast.covariances:
        assert np.allclose(step, step.T)
        assert np.linalg.eigvalsh(step).min() > 0.0
    assert not np.allclose(
        forecast.covariances[0], forecast.covariances[-1]
    ), "regime probabilities must drift towards the stationary distribution"


def test_regime_mixture_covariance_includes_between_regime_variance():
    """The law-of-total-variance term must widen the mixture covariance."""
    returns = _returns(n_assets=3, periods=800)
    forecaster = RegimeMixtureForecaster(CausalRegimeFilter(3, burn_in=250, refit_every=250))
    forecaster.prepare(returns)

    forecast = forecaster.forecast(600, 1)
    parameters = forecaster.path.parameters_at(600)
    probabilities = forecaster.path.predict(600, 1)[0]
    within = np.einsum("k,kij->ij", probabilities, parameters.covariances)
    assert np.all(np.diag(forecast.covariances[0]) >= np.diag(within) - 1e-15)


def test_forecast_rejects_dates_without_enough_history():
    returns = _returns(periods=400)
    forecaster = RollingMomentForecaster(60)
    forecaster.prepare(returns)
    with pytest.raises(ValueError):
        forecaster.forecast(30, 5)
    with pytest.raises(ValueError):
        forecaster.forecast(500, 5)


def test_forecast_requires_prepare():
    with pytest.raises(RuntimeError):
        RollingMomentForecaster(60).forecast(100, 5)


def test_moment_forecast_validates_shapes():
    with pytest.raises(ValueError):
        MomentForecast(means=np.zeros((3, 2)), covariances=np.zeros((3, 3, 3)))
    with pytest.raises(ValueError):
        MomentForecast(means=np.full((2, 2), np.nan), covariances=np.zeros((2, 2, 2)))
