"""Tests for causal Gaussian-HMM regime filtering."""

from __future__ import annotations

import numpy as np
import pytest

from src.expert.regime_filter import CausalRegimeFilter, GaussianHMM
from src.simulation.presets import canonical_three_regime_preset


def _simulate(n_assets: int = 4, periods: int = 900, seed: int = 7):
    preset = canonical_three_regime_preset(n_assets)
    returns, regimes = preset.build_simulator().simulate(T=periods, seed=seed)
    return returns.to_numpy(dtype=float), regimes


def test_fit_recovers_regime_volatility_ordering():
    returns, _ = _simulate()
    parameters = GaussianHMM(3).fit(returns)

    volatilities = np.sqrt(np.trace(parameters.covariances, axis1=1, axis2=2))
    assert np.all(np.diff(volatilities) > 0.0), "states must be ordered calm to stressed"
    assert np.allclose(parameters.transition_matrix.sum(axis=1), 1.0)
    assert np.isclose(parameters.initial_distribution.sum(), 1.0)
    assert np.all(np.linalg.eigvalsh(parameters.covariances) > 0.0)


def test_fit_is_deterministic():
    returns, _ = _simulate(periods=600)
    first = GaussianHMM(3).fit(returns)
    second = GaussianHMM(3).fit(returns)
    assert np.allclose(first.means, second.means)
    assert np.allclose(first.transition_matrix, second.transition_matrix)
    assert first.log_likelihood == pytest.approx(second.log_likelihood)


def test_filtered_probabilities_are_valid_distributions():
    returns, _ = _simulate(periods=600)
    model = GaussianHMM(3)
    probabilities = model.filter(returns, model.fit(returns))
    assert probabilities.shape == (returns.shape[0], 3)
    assert np.all(probabilities >= 0.0)
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_filtering_at_t_ignores_later_returns():
    """The forward recursion at t must not react to observations after t."""
    returns, _ = _simulate(periods=500)
    model = GaussianHMM(3)
    parameters = model.fit(returns)

    corrupted = returns.copy()
    corrupted[300:] = 5.0 * corrupted[300:] + 0.01
    original = model.filter(returns, parameters)
    perturbed = model.filter(corrupted, parameters)

    assert np.allclose(original[:300], perturbed[:300])
    assert not np.allclose(original[300:], perturbed[300:])


def test_causal_filter_uses_only_past_returns():
    """Filtered probabilities up to t are unchanged if the future is replaced."""
    returns, _ = _simulate(periods=800)
    regime_filter = CausalRegimeFilter(3, burn_in=250, refit_every=250)

    full = regime_filter.run(returns)
    truncated = regime_filter.run(returns[:600])

    assert np.allclose(full.probabilities[:600], truncated.probabilities)


def test_parameters_at_returns_the_fit_in_force():
    returns, _ = _simulate(periods=800)
    path = CausalRegimeFilter(3, burn_in=250, refit_every=250).run(returns)

    assert path.fit_times[0] == 250
    assert np.allclose(
        path.parameters_at(400).means, path.fits[0].means
    ), "a decision at t=400 must use the fit estimated at t=250"
    assert np.allclose(path.parameters_at(600).means, path.fits[1].means)
    with pytest.raises(ValueError):
        path.parameters_at(100)


def test_predict_propagates_through_the_transition_matrix():
    returns, _ = _simulate(periods=600)
    path = CausalRegimeFilter(3, burn_in=250, refit_every=250).run(returns)

    predicted = path.predict(400, 3)
    expected = path.probabilities[399] @ path.parameters_at(400).transition_matrix
    assert np.allclose(predicted[0], expected)
    assert np.allclose(predicted.sum(axis=1), 1.0)


def test_filter_beats_chance_on_simulated_regimes():
    returns, regimes = _simulate(n_assets=6, periods=1200, seed=42)
    path = CausalRegimeFilter(3, burn_in=250, refit_every=250).run(returns)

    predicted = path.probabilities.argmax(axis=1)[250:]
    accuracy = float((predicted == regimes[250:]).mean())
    assert accuracy > 0.55, f"filter accuracy {accuracy:.3f} is no better than chance"


def test_rejects_paths_shorter_than_burn_in():
    returns, _ = _simulate(periods=300)
    with pytest.raises(ValueError):
        CausalRegimeFilter(3, burn_in=400).run(returns)


def test_rejects_invalid_configuration():
    with pytest.raises(ValueError):
        GaussianHMM(0)
    with pytest.raises(ValueError):
        CausalRegimeFilter(3, burn_in=1)
    with pytest.raises(ValueError):
        CausalRegimeFilter(3, refit_every=0)


def test_rejects_non_finite_returns():
    returns, _ = _simulate(periods=300)
    corrupted = returns.copy()
    corrupted[10, 0] = np.nan
    with pytest.raises(ValueError):
        GaussianHMM(3).fit(corrupted)
