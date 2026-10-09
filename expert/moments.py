"""Causal moment forecasts used by the rolling-horizon Markowitz expert.

The expert needs ``mu_hat_{tau+1|t}`` and ``Sigma_hat_{tau+1|t}`` for every
planning step ``tau = t, ..., t+H-1``. All forecasters implemented here obey
one contract:

    ``forecast(t, horizon)`` may use ``returns[:t]`` and nothing else.

Index convention: ``returns[t]`` is the return realised *after* the decision
taken at index ``t``. Decision ``t`` therefore conditions on ``returns[:t]``
and earns ``returns[t]``, which matches ``F_t = sigma(r_1, ..., r_t)`` in the
exposé with ``returns[t] = r_{t+1}``.

Two covariance options are offered, as required by the Weeks 9-13 work
package: the plain rolling sample covariance and the Ledoit-Wolf shrinkage
estimator, which is better conditioned when ``n_assets`` approaches the window
length.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from sklearn.covariance import ledoit_wolf

COVARIANCE_ESTIMATORS = ("sample", "ledoit_wolf")


@dataclass(frozen=True)
class MomentForecast:
    """Per-horizon-step mean and covariance forecasts formed at one date."""

    means: np.ndarray
    covariances: np.ndarray

    def __post_init__(self) -> None:
        means = np.asarray(self.means, dtype=float)
        covariances = np.asarray(self.covariances, dtype=float)
        if means.ndim != 2:
            raise ValueError("means must have shape (horizon, n_assets).")
        if covariances.shape != (means.shape[0], means.shape[1], means.shape[1]):
            raise ValueError("covariances must have shape (horizon, n_assets, n_assets).")
        if not np.all(np.isfinite(means)) or not np.all(np.isfinite(covariances)):
            raise ValueError("moment forecasts must contain only finite values.")
        object.__setattr__(self, "means", means)
        object.__setattr__(self, "covariances", covariances)

    @property
    def horizon(self) -> int:
        return int(self.means.shape[0])

    @property
    def n_assets(self) -> int:
        return int(self.means.shape[1])


class MomentForecaster(Protocol):
    """Interface every expert moment estimator must satisfy."""

    #: First decision index at which the forecaster can produce a forecast.
    min_history: int

    def prepare(self, returns: np.ndarray) -> None:
        """Precompute path-level quantities.

        Implementations may only precompute quantities that a *causal* filter
        could also compute online. ``tests/test_expert_moments.py`` enforces
        this by perturbing future returns and asserting forecasts are unchanged.
        """

    def forecast(self, t: int, horizon: int) -> MomentForecast:
        """Return forecasts for planning steps ``t, ..., t + horizon - 1``."""

    def describe(self) -> dict[str, object]:
        """Return a JSON-serialisable description for experiment metadata."""


def estimate_covariance(window: np.ndarray, method: str = "sample") -> np.ndarray:
    """Estimate a covariance matrix from a return window.

    Args:
        window: Array of shape ``(n_observations, n_assets)``.
        method: ``"sample"`` or ``"ledoit_wolf"``.
    """
    observations = np.asarray(window, dtype=float)
    if observations.ndim != 2 or observations.shape[0] < 2 or observations.shape[1] == 0:
        raise ValueError("window must have shape (n_observations >= 2, n_assets >= 1).")
    if not np.all(np.isfinite(observations)):
        raise ValueError("window must contain only finite values.")
    if method not in COVARIANCE_ESTIMATORS:
        raise ValueError(
            f"unknown covariance estimator {method!r}; expected one of {COVARIANCE_ESTIMATORS}."
        )

    if method == "sample":
        covariance = np.cov(observations, rowvar=False, ddof=1)
    else:
        covariance, _ = ledoit_wolf(observations, assume_centered=False)
    return make_positive_definite(np.atleast_2d(np.asarray(covariance, dtype=float)))


def make_positive_definite(covariance: np.ndarray, ridge: float = 1e-12) -> np.ndarray:
    """Symmetrise a covariance matrix and lift it to positive definiteness."""
    symmetric = 0.5 * (covariance + covariance.T)
    if not np.all(np.isfinite(symmetric)):
        raise ValueError("covariance contains non-finite values.")
    eigenvalues = np.linalg.eigvalsh(symmetric)
    floor = max(ridge, float(eigenvalues.max()) * 1e-12)
    if eigenvalues.min() < floor:
        symmetric = symmetric + np.eye(symmetric.shape[0]) * (floor - eigenvalues.min())
    return symmetric


class RollingMomentForecaster:
    """Rolling-window moments held constant across the planning horizon.

    This is the classical benchmark estimator: the mean and covariance of the
    most recent ``window`` observations are used as the forecast for every step
    of the horizon.
    """

    def __init__(self, window: int, *, covariance: str = "sample") -> None:
        if not isinstance(window, int) or isinstance(window, bool) or window < 2:
            raise ValueError("window must be an integer of at least 2.")
        if covariance not in COVARIANCE_ESTIMATORS:
            raise ValueError(
                f"unknown covariance estimator {covariance!r}; "
                f"expected one of {COVARIANCE_ESTIMATORS}."
            )
        self.window = window
        self.covariance = covariance
        self.min_history = window
        self._returns: np.ndarray | None = None

    def prepare(self, returns: np.ndarray) -> None:
        self._returns = _validate_returns(returns)

    def forecast(self, t: int, horizon: int) -> MomentForecast:
        returns = _require_prepared(self._returns)
        _validate_decision_index(t, horizon, self.min_history, returns.shape[0])

        window = returns[t - self.window : t]
        mean = window.mean(axis=0)
        covariance = estimate_covariance(window, self.covariance)
        return MomentForecast(
            means=np.tile(mean, (horizon, 1)),
            covariances=np.tile(covariance, (horizon, 1, 1)),
        )

    def describe(self) -> dict[str, object]:
        return {
            "estimator": "rolling",
            "window": self.window,
            "covariance": self.covariance,
            "min_history": self.min_history,
        }


class RegimeMixtureForecaster:
    """Regime-aware moments from HMM-filtered probabilities.

    Following Section 4.1 of the exposé, filtered probabilities are propagated
    through the estimated transition matrix and used to mix the estimated
    regime-conditional Gaussians:

        q_h      = p_hat_t P^h,
        mu_h     = sum_k q_h(k) mu_k,
        Sigma_h  = sum_k q_h(k) (Sigma_k + mu_k mu_k^T) - mu_h mu_h^T.

    The covariance follows the law of total variance, so it also captures the
    dispersion *between* regimes, not only the within-regime risk. Regime
    parameters come from the HMM, never from the simulator.
    """

    def __init__(
        self,
        regime_filter,
        *,
        blend_window: int | None = None,
        blend_weight: float = 0.0,
        covariance: str = "ledoit_wolf",
    ) -> None:
        if blend_weight < 0.0 or blend_weight > 1.0 or not np.isfinite(blend_weight):
            raise ValueError("blend_weight must lie in [0, 1].")
        if blend_weight > 0.0 and blend_window is None:
            raise ValueError("blend_window is required when blend_weight is positive.")
        if blend_window is not None and (
            not isinstance(blend_window, int) or blend_window < 2
        ):
            raise ValueError("blend_window must be an integer of at least 2.")
        if covariance not in COVARIANCE_ESTIMATORS:
            raise ValueError(f"unknown covariance estimator {covariance!r}.")

        self.regime_filter = regime_filter
        self.blend_window = blend_window
        self.blend_weight = float(blend_weight)
        self.covariance = covariance
        self.min_history = max(regime_filter.burn_in, blend_window or 0)
        self._returns: np.ndarray | None = None
        self._path = None
        self._fingerprint: tuple[tuple[int, ...], bytes] | None = None

    def prepare(self, returns: np.ndarray) -> None:
        values = _validate_returns(returns)
        # Fitting the HMM is the dominant cost of an expert run. Ablations that
        # vary only the cost, horizon, or risk aversion re-prepare the same
        # path, so an identical array reuses the previous fit.
        fingerprint = (values.shape, values.tobytes())
        if self._path is not None and fingerprint == self._fingerprint:
            self._returns = values
            return
        self._returns = values
        self._path = self.regime_filter.run(values)
        self._fingerprint = fingerprint

    def restore(self, returns: np.ndarray, path) -> None:
        """Reattach a previously computed filter path without refitting.

        Baum-Welch is deterministic, so a stored :class:`RegimeFilterPath` is
        exactly what ``prepare`` would recompute. Persisting it lets a dataset
        reassemble states without paying for the HMM again on every load.
        """
        values = _validate_returns(returns)
        if path.probabilities.shape[0] != values.shape[0]:
            raise ValueError("stored filter path length does not match the returns.")
        if path.regime_means.shape[1] != values.shape[1]:
            raise ValueError("stored filter path does not match the asset count.")
        self._returns = values
        self._path = path
        self._fingerprint = (values.shape, values.tobytes())

    @property
    def path(self):
        """Return the fitted :class:`RegimeFilterPath`, for diagnostics."""
        if self._path is None:
            raise RuntimeError("prepare() must be called before accessing the filter path.")
        return self._path

    def forecast(self, t: int, horizon: int) -> MomentForecast:
        returns = _require_prepared(self._returns)
        _validate_decision_index(t, horizon, self.min_history, returns.shape[0])
        path = self.path

        # Both the propagated probabilities and the regime-conditional moments
        # must come from the fit in force at `t`, never from a later refit.
        parameters = path.parameters_at(t)
        predicted = path.predict(t, horizon)
        regime_means = parameters.means
        regime_covariances = parameters.covariances
        second_moments = regime_covariances + np.einsum(
            "ki,kj->kij", regime_means, regime_means
        )

        means = predicted @ regime_means
        covariances = np.einsum("hk,kij->hij", predicted, second_moments)
        covariances -= np.einsum("hi,hj->hij", means, means)

        if self.blend_weight > 0.0 and self.blend_window is not None:
            window = returns[t - self.blend_window : t]
            sample_mean = window.mean(axis=0)
            sample_covariance = estimate_covariance(window, self.covariance)
            means = (1.0 - self.blend_weight) * means + self.blend_weight * sample_mean
            covariances = (
                1.0 - self.blend_weight
            ) * covariances + self.blend_weight * sample_covariance

        covariances = np.stack([make_positive_definite(step) for step in covariances])
        return MomentForecast(means=means, covariances=covariances)

    def describe(self) -> dict[str, object]:
        return {
            "estimator": "regime_mixture",
            "n_regimes": int(self.regime_filter.n_regimes),
            "burn_in": int(self.regime_filter.burn_in),
            "refit_every": int(self.regime_filter.refit_every),
            "blend_window": self.blend_window,
            "blend_weight": self.blend_weight,
            "covariance": self.covariance,
            "min_history": self.min_history,
        }


def _validate_returns(returns: np.ndarray) -> np.ndarray:
    values = np.asarray(returns, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("returns must have shape (T, n_assets).")
    if not np.all(np.isfinite(values)):
        raise ValueError("returns must contain only finite values.")
    return values


def _require_prepared(returns: np.ndarray | None) -> np.ndarray:
    if returns is None:
        raise RuntimeError("prepare() must be called before forecast().")
    return returns


def _validate_decision_index(
    t: int, horizon: int, min_history: int, n_observations: int
) -> None:
    if not isinstance(t, (int, np.integer)) or isinstance(t, bool):
        raise ValueError("t must be an integer decision index.")
    if not isinstance(horizon, (int, np.integer)) or horizon < 1:
        raise ValueError("horizon must be a positive integer.")
    if t < min_history:
        raise ValueError(
            f"decision index {t} has insufficient history; need at least {min_history}."
        )
    if t > n_observations:
        raise ValueError("decision index exceeds the prepared return path.")
