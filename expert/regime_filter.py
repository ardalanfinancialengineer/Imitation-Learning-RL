"""Causal Gaussian hidden-Markov filtering of latent market regimes.

The exposé (Section 4.1) requires the expert to work with *estimated* regime
probabilities

    p_hat_t(k) = Pr(z_t = k | r_1, ..., r_t),

obtained from a hidden Markov model that only sees information available at
time ``t``. The simulator's true regime labels are never used here; they are
reserved for diagnostics and evaluation.

Two pieces are implemented:

``GaussianHMM``
    Baum-Welch (EM) estimation of transition matrix, initial distribution,
    regime means, and regime covariances, plus a scaled forward recursion.

``CausalRegimeFilter``
    A refit schedule around ``GaussianHMM``. Parameters used to filter time
    ``t`` are estimated on ``returns[:t]`` only, so no future observation can
    influence the filtered probabilities at ``t``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_triangular

_LOG_TWO_PI = float(np.log(2.0 * np.pi))


@dataclass(frozen=True)
class GaussianHMMParameters:
    """Fitted parameters of a Gaussian hidden Markov model."""

    transition_matrix: np.ndarray
    initial_distribution: np.ndarray
    means: np.ndarray
    covariances: np.ndarray
    log_likelihood: float
    n_iterations: int
    converged: bool

    @property
    def n_regimes(self) -> int:
        return int(self.means.shape[0])

    @property
    def n_assets(self) -> int:
        return int(self.means.shape[1])


class GaussianHMM:
    """Gaussian hidden Markov model fitted by Baum-Welch.

    Covariances are shrunk towards a diagonal target and ridged, because the
    per-regime effective sample size is small relative to ``n_assets`` in the
    high-dimensional configurations of the project.
    """

    def __init__(
        self,
        n_regimes: int = 3,
        *,
        max_iterations: int = 60,
        tolerance: float = 1e-4,
        covariance_shrinkage: float = 0.10,
        covariance_ridge: float = 1e-10,
    ) -> None:
        if not isinstance(n_regimes, int) or isinstance(n_regimes, bool) or n_regimes < 1:
            raise ValueError("n_regimes must be a positive integer.")
        if not isinstance(max_iterations, int) or max_iterations < 1:
            raise ValueError("max_iterations must be a positive integer.")
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("tolerance must be positive and finite.")
        if not 0.0 <= covariance_shrinkage < 1.0:
            raise ValueError("covariance_shrinkage must lie in [0, 1).")
        if not np.isfinite(covariance_ridge) or covariance_ridge <= 0.0:
            raise ValueError("covariance_ridge must be positive and finite.")

        self.n_regimes = n_regimes
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.covariance_shrinkage = covariance_shrinkage
        self.covariance_ridge = covariance_ridge

    def fit(self, returns: np.ndarray) -> GaussianHMMParameters:
        """Estimate model parameters from a return history.

        Args:
            returns: Array of shape ``(n_observations, n_assets)``.
        """
        observations = _validate_observations(returns)
        n_observations = observations.shape[0]
        if n_observations < 2 * self.n_regimes:
            raise ValueError(
                "returns must contain at least two observations per regime to fit an HMM."
            )

        transition_matrix, initial_distribution, means, covariances = self._initialise(
            observations
        )

        previous_log_likelihood = -np.inf
        log_likelihood = -np.inf
        converged = False
        iteration = 0
        for iteration in range(1, self.max_iterations + 1):
            densities = self._emission_densities(observations, means, covariances)
            alpha, scale, beta = self._forward_backward(
                densities, transition_matrix, initial_distribution
            )
            log_likelihood = float(np.log(scale).sum())

            gamma = alpha * beta
            gamma /= np.maximum(gamma.sum(axis=1, keepdims=True), 1e-300)

            # Expected transition counts, summed over time.
            weighted = alpha[:-1, :, None] * transition_matrix[None, :, :]
            weighted = weighted * (densities[1:] * beta[1:] / scale[1:, None])[:, None, :]
            xi = weighted.sum(axis=0)

            initial_distribution = _normalise(gamma[0])
            transition_matrix = xi / np.maximum(xi.sum(axis=1, keepdims=True), 1e-300)
            means, covariances = self._maximise_emissions(observations, gamma)

            if abs(log_likelihood - previous_log_likelihood) < self.tolerance * max(
                1.0, abs(previous_log_likelihood)
            ):
                converged = True
                break
            previous_log_likelihood = log_likelihood

        return _canonical_order(
            GaussianHMMParameters(
                transition_matrix=transition_matrix,
                initial_distribution=initial_distribution,
                means=means,
                covariances=covariances,
                log_likelihood=log_likelihood,
                n_iterations=iteration,
                converged=converged,
            )
        )

    def filter(
        self,
        returns: np.ndarray,
        parameters: GaussianHMMParameters,
    ) -> np.ndarray:
        """Return causal filtered probabilities with shape ``(n_observations, K)``.

        Row ``t`` is ``Pr(z_t = k | r_1, ..., r_t)`` and depends only on
        observations up to and including ``t``.
        """
        observations = _validate_observations(returns)
        if observations.shape[1] != parameters.n_assets:
            raise ValueError("returns and fitted parameters disagree on n_assets.")

        densities = self._emission_densities(
            observations, parameters.means, parameters.covariances
        )
        alpha, _, _ = self._forward_backward(
            densities,
            parameters.transition_matrix,
            parameters.initial_distribution,
            backward=False,
        )
        return alpha

    def _initialise(
        self, observations: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Deterministically seed EM by sorting blocks on cross-sectional volatility.

        Calm, volatile, and stressed states differ mainly in dispersion, so
        ranking short blocks by their average absolute return separates the
        states well enough for EM to refine without any random restarts.
        """
        n_observations = observations.shape[0]
        dispersion = np.abs(observations).mean(axis=1)
        block = max(1, n_observations // (10 * self.n_regimes))
        smoothed = _moving_average(dispersion, block)

        quantiles = np.quantile(smoothed, np.linspace(0.0, 1.0, self.n_regimes + 1)[1:-1])
        labels = np.searchsorted(quantiles, smoothed, side="right")

        responsibilities = np.zeros((n_observations, self.n_regimes))
        responsibilities[np.arange(n_observations), labels] = 1.0
        # Blend towards uniform so no regime starts with zero responsibility.
        responsibilities = 0.9 * responsibilities + 0.1 / self.n_regimes
        means, covariances = self._maximise_emissions(observations, responsibilities)

        transition_matrix = np.full(
            (self.n_regimes, self.n_regimes), 0.1 / max(1, self.n_regimes - 1)
        )
        np.fill_diagonal(transition_matrix, 0.9)
        if self.n_regimes == 1:
            transition_matrix = np.ones((1, 1))
        initial_distribution = np.full(self.n_regimes, 1.0 / self.n_regimes)
        return transition_matrix, initial_distribution, means, covariances

    def _maximise_emissions(
        self,
        observations: np.ndarray,
        responsibilities: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        n_assets = observations.shape[1]
        weights = np.maximum(responsibilities.sum(axis=0), 1e-10)
        means = (responsibilities.T @ observations) / weights[:, None]

        covariances = np.empty((self.n_regimes, n_assets, n_assets))
        pooled = np.cov(observations, rowvar=False, ddof=0)
        pooled = np.atleast_2d(pooled)
        for regime in range(self.n_regimes):
            centred = observations - means[regime]
            scaled = centred * responsibilities[:, regime, None]
            covariance = (scaled.T @ centred) / weights[regime]
            covariance = 0.5 * (covariance + covariance.T)
            target = np.diag(np.maximum(np.diag(covariance), np.diag(pooled) * 1e-6))
            covariance = (
                1.0 - self.covariance_shrinkage
            ) * covariance + self.covariance_shrinkage * target
            covariances[regime] = _make_positive_definite(covariance, self.covariance_ridge)
        return means, covariances

    @staticmethod
    def _emission_densities(
        observations: np.ndarray,
        means: np.ndarray,
        covariances: np.ndarray,
    ) -> np.ndarray:
        """Return scaled emission likelihoods with shape ``(T, K)``.

        Densities are exponentiated after subtracting the per-observation
        maximum log density. A common positive factor per row cancels in every
        forward/backward normalisation, so this only prevents underflow.
        """
        n_regimes = means.shape[0]
        log_densities = np.empty((observations.shape[0], n_regimes))
        for regime in range(n_regimes):
            log_densities[:, regime] = _gaussian_log_pdf(
                observations, means[regime], covariances[regime]
            )
        log_densities -= log_densities.max(axis=1, keepdims=True)
        return np.exp(log_densities)

    @staticmethod
    def _forward_backward(
        densities: np.ndarray,
        transition_matrix: np.ndarray,
        initial_distribution: np.ndarray,
        *,
        backward: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_observations, n_regimes = densities.shape
        alpha = np.empty((n_observations, n_regimes))
        scale = np.empty(n_observations)

        current = initial_distribution * densities[0]
        scale[0] = max(current.sum(), 1e-300)
        alpha[0] = current / scale[0]
        for t in range(1, n_observations):
            current = (alpha[t - 1] @ transition_matrix) * densities[t]
            scale[t] = max(current.sum(), 1e-300)
            alpha[t] = current / scale[t]

        if not backward:
            return alpha, scale, np.ones_like(alpha)

        beta = np.ones((n_observations, n_regimes))
        for t in range(n_observations - 2, -1, -1):
            beta[t] = transition_matrix @ (densities[t + 1] * beta[t + 1]) / scale[t + 1]
        return alpha, scale, beta


@dataclass(frozen=True)
class RegimeFilterPath:
    """Causal filtered regime probabilities and per-refit model parameters.

    Every refit is retained. A query at decision index ``t`` is answered with
    the most recent fit estimated at or before ``t``; keeping only the final fit
    would let parameters estimated on later data influence earlier decisions.

    Hidden states are canonically ordered by increasing average variance, so
    state ``0`` is the calmest across all refits and the probability vector
    keeps a stable meaning over time.
    """

    probabilities: np.ndarray
    fits: tuple[GaussianHMMParameters, ...]
    fit_times: tuple[int, ...]
    burn_in: int

    def parameters_at(self, t: int) -> GaussianHMMParameters:
        """Return the parameters in force at decision index ``t``."""
        index = int(np.searchsorted(np.asarray(self.fit_times), int(t), side="right")) - 1
        if index < 0:
            raise ValueError(
                f"decision index {t} precedes the first fit at {self.fit_times[0]}."
            )
        return self.fits[index]

    @property
    def transition_matrix(self) -> np.ndarray:
        """Transition matrix of the most recent fit, for reporting."""
        return self.fits[-1].transition_matrix

    @property
    def regime_means(self) -> np.ndarray:
        """Regime means of the most recent fit, for reporting."""
        return self.fits[-1].means

    @property
    def regime_covariances(self) -> np.ndarray:
        """Regime covariances of the most recent fit, for reporting."""
        return self.fits[-1].covariances

    def predict(self, t: int, horizon: int) -> np.ndarray:
        """Propagate the filtered distribution ``horizon`` steps ahead.

        Returns an array of shape ``(horizon, K)`` whose row ``h`` is
        ``Pr(z_{t+h+1} = k | F_t) = (p_hat_t P^{h+1})_k``, using only the fit
        available at ``t``.
        """
        if not isinstance(t, (int, np.integer)) or t < 1:
            raise ValueError("t must be a positive integer decision index.")
        if t > self.probabilities.shape[0]:
            raise ValueError("t exceeds the filtered path length.")
        if not isinstance(horizon, (int, np.integer)) or horizon < 1:
            raise ValueError("horizon must be a positive integer.")

        transition_matrix = self.parameters_at(t).transition_matrix
        predicted = np.empty((int(horizon), self.probabilities.shape[1]))
        current = self.probabilities[t - 1]
        for step in range(int(horizon)):
            current = current @ transition_matrix
            predicted[step] = current
        return predicted


class CausalRegimeFilter:
    """Run a Gaussian HMM over a path while respecting the information set.

    Parameters are re-estimated on a schedule. A refit at time ``t`` uses
    ``returns[:t]``, and filtered probabilities for ``s >= t`` use that fit
    until the next refit. Before ``burn_in`` there is not enough history to
    fit, so probabilities fall back to the uniform distribution and the expert
    is expected not to trade there.
    """

    def __init__(
        self,
        n_regimes: int = 3,
        *,
        burn_in: int = 250,
        refit_every: int = 250,
        model: GaussianHMM | None = None,
    ) -> None:
        if not isinstance(burn_in, int) or isinstance(burn_in, bool) or burn_in < 2:
            raise ValueError("burn_in must be an integer of at least 2.")
        if not isinstance(refit_every, int) or isinstance(refit_every, bool):
            raise ValueError("refit_every must be an integer.")
        if refit_every < 1:
            raise ValueError("refit_every must be a positive integer.")

        self.model = model if model is not None else GaussianHMM(n_regimes)
        self.n_regimes = self.model.n_regimes
        self.burn_in = burn_in
        self.refit_every = refit_every

    def run(self, returns: np.ndarray) -> RegimeFilterPath:
        """Filter a full return path under the refit schedule."""
        observations = _validate_observations(returns)
        n_observations = observations.shape[0]
        if n_observations < self.burn_in:
            raise ValueError(
                f"returns must contain at least burn_in={self.burn_in} observations."
            )

        probabilities = np.full(
            (n_observations, self.n_regimes), 1.0 / self.n_regimes, dtype=float
        )
        fit_times: list[int] = []
        fits: list[GaussianHMMParameters] = []

        fit_at = self.burn_in
        while fit_at <= n_observations:
            parameters = self.model.fit(observations[:fit_at])
            fit_times.append(fit_at)
            fits.append(parameters)
            # Probabilities from this fit apply from `fit_at` until the next refit.
            # The recursion is restarted from the beginning of the path so that
            # the filtered state at `fit_at` reflects the whole known history.
            next_fit = min(fit_at + self.refit_every, n_observations + 1)
            segment_end = min(next_fit - 1, n_observations)
            filtered = self.model.filter(observations[:segment_end], parameters)
            probabilities[fit_at - 1 : segment_end] = filtered[fit_at - 1 : segment_end]
            fit_at = next_fit

        if not fits:  # pragma: no cover - guarded by the length check above
            raise RuntimeError("regime filter produced no fit.")

        return RegimeFilterPath(
            probabilities=probabilities,
            fits=tuple(fits),
            fit_times=tuple(fit_times),
            burn_in=self.burn_in,
        )


def _validate_observations(returns: np.ndarray) -> np.ndarray:
    observations = np.asarray(returns, dtype=float)
    if observations.ndim != 2 or observations.shape[0] == 0 or observations.shape[1] == 0:
        raise ValueError("returns must have shape (n_observations, n_assets).")
    if not np.all(np.isfinite(observations)):
        raise ValueError("returns must contain only finite values.")
    return observations


def _gaussian_log_pdf(
    observations: np.ndarray,
    mean: np.ndarray,
    covariance: np.ndarray,
) -> np.ndarray:
    n_assets = observations.shape[1]
    cholesky = np.linalg.cholesky(covariance)
    centred = observations - mean
    solved = solve_triangular(cholesky, centred.T, lower=True, check_finite=False)
    log_determinant = 2.0 * np.log(np.diag(cholesky)).sum()
    return -0.5 * (
        n_assets * _LOG_TWO_PI + log_determinant + np.square(solved).sum(axis=0)
    )


def _make_positive_definite(covariance: np.ndarray, ridge: float) -> np.ndarray:
    symmetric = 0.5 * (covariance + covariance.T)
    eigenvalues = np.linalg.eigvalsh(symmetric)
    floor = max(ridge, float(np.max(eigenvalues)) * 1e-10 if eigenvalues.size else ridge)
    if eigenvalues.min() < floor:
        symmetric = symmetric + np.eye(symmetric.shape[0]) * (floor - eigenvalues.min())
    return symmetric


def _canonical_order(parameters: GaussianHMMParameters) -> GaussianHMMParameters:
    """Relabel hidden states by increasing average variance.

    EM leaves the state labels arbitrary. Successive refits could therefore
    permute them, which would make the filtered probability vector mean
    different things at different dates. Sorting by average variance matches
    the calm/volatile/stressed ordering of the simulation preset and keeps the
    labelling stable across refits.
    """
    ordering = np.argsort(np.trace(parameters.covariances, axis1=1, axis2=2))
    if np.array_equal(ordering, np.arange(ordering.size)):
        return parameters
    return GaussianHMMParameters(
        transition_matrix=parameters.transition_matrix[np.ix_(ordering, ordering)],
        initial_distribution=parameters.initial_distribution[ordering],
        means=parameters.means[ordering],
        covariances=parameters.covariances[ordering],
        log_likelihood=parameters.log_likelihood,
        n_iterations=parameters.n_iterations,
        converged=parameters.converged,
    )


def _normalise(vector: np.ndarray) -> np.ndarray:
    total = vector.sum()
    if total <= 0.0:
        return np.full(vector.size, 1.0 / vector.size)
    return vector / total


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values
    kernel = np.ones(window) / window
    padded = np.concatenate(
        [np.full(window - 1, values[0]), values]
    )
    return np.convolve(padded, kernel, mode="valid")
