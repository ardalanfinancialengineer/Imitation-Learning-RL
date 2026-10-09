"""Quantitative diagnostics for regime-switching return simulations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DiagnosticThresholds:
    """Acceptance thresholds for one deterministic simulation run."""

    max_frequency_error: float = 0.03
    max_transition_error: float = 0.05
    max_mean_z_score: float = 5.0
    max_variance_standardized_error: float = 5.0
    max_covariance_error_ratio: float = 3.0

    def __post_init__(self) -> None:
        for field_name, value in asdict(self).items():
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{field_name} must be positive and finite.")


@dataclass(frozen=True)
class SimulationDiagnostics:
    """Observed-vs-target diagnostics for a simulated return path."""

    regime_names: tuple[str, ...]
    counts: np.ndarray
    expected_frequencies: np.ndarray
    empirical_frequencies: np.ndarray
    target_transition_matrix: np.ndarray
    empirical_transition_matrix: np.ndarray
    empirical_means: np.ndarray
    empirical_covariances: np.ndarray
    frequency_errors: np.ndarray
    transition_errors: np.ndarray
    mean_z_scores: np.ndarray
    variance_standardized_errors: np.ndarray
    covariance_relative_errors: np.ndarray
    covariance_error_ratios: np.ndarray
    checks: dict[str, bool]
    thresholds: DiagnosticThresholds

    @property
    def passed(self) -> bool:
        return all(self.checks.values())

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable report with summary and per-regime data."""
        per_regime = []
        for regime, name in enumerate(self.regime_names):
            per_regime.append(
                {
                    "index": regime,
                    "name": name,
                    "count": int(self.counts[regime]),
                    "expected_frequency": float(self.expected_frequencies[regime]),
                    "empirical_frequency": float(self.empirical_frequencies[regime]),
                    "frequency_error": float(self.frequency_errors[regime]),
                    "target_persistence": float(
                        self.target_transition_matrix[regime, regime]
                    ),
                    "empirical_persistence": float(
                        self.empirical_transition_matrix[regime, regime]
                    ),
                    "max_abs_mean_z_score": float(
                        np.max(np.abs(self.mean_z_scores[regime]))
                    ),
                    "max_abs_variance_standardized_error": float(
                        np.max(np.abs(self.variance_standardized_errors[regime]))
                    ),
                    "covariance_relative_error": float(
                        self.covariance_relative_errors[regime]
                    ),
                    "covariance_error_ratio": float(
                        self.covariance_error_ratios[regime]
                    ),
                }
            )

        return {
            "passed": self.passed,
            "checks": self.checks,
            "thresholds": asdict(self.thresholds),
            "per_regime": per_regime,
            "target_transition_matrix": self.target_transition_matrix.tolist(),
            "empirical_transition_matrix": self.empirical_transition_matrix.tolist(),
            "transition_error_matrix": self.transition_errors.tolist(),
        }


def expected_regime_frequencies(
    transition_matrix: np.ndarray,
    initial_distribution: np.ndarray,
    periods: int,
) -> np.ndarray:
    """Compute finite-horizon expected occupancy from the configured initial law."""
    transition_matrix, initial_distribution = _validate_markov_inputs(
        transition_matrix,
        initial_distribution,
    )
    if not isinstance(periods, int) or isinstance(periods, bool) or periods <= 0:
        raise ValueError("periods must be a positive integer.")

    distribution = initial_distribution.copy()
    expected_counts = np.zeros_like(distribution)
    for _ in range(periods):
        expected_counts += distribution
        distribution = distribution @ transition_matrix
    return expected_counts / periods


def empirical_transition_matrix(
    regimes: np.ndarray,
    n_regimes: int,
) -> np.ndarray:
    """Estimate row-conditional transition probabilities from regime labels."""
    regimes = _validate_regimes(regimes, n_regimes)
    if regimes.size < 2:
        raise ValueError("at least two regime observations are required.")

    counts = np.zeros((n_regimes, n_regimes), dtype=int)
    np.add.at(counts, (regimes[:-1], regimes[1:]), 1)
    row_counts = counts.sum(axis=1)
    if np.any(row_counts == 0):
        missing = np.flatnonzero(row_counts == 0).tolist()
        raise ValueError(f"cannot estimate transitions for unobserved source regimes: {missing}.")
    return counts / row_counts[:, None]


def evaluate_simulation(
    returns: pd.DataFrame | np.ndarray,
    regimes: np.ndarray,
    *,
    regime_names: tuple[str, ...],
    transition_matrix: np.ndarray,
    initial_distribution: np.ndarray,
    target_means: np.ndarray,
    target_covariances: np.ndarray,
    thresholds: DiagnosticThresholds | None = None,
) -> SimulationDiagnostics:
    """Check frequencies, persistence, means, variances, and covariances.

    Mean and variance discrepancies are standardised by their Gaussian sampling
    errors. Covariance error is divided by the root-mean-square Frobenius error
    implied by a Wishart sample covariance. These scale-aware diagnostics remain
    meaningful for both small development universes and the final 200-asset runs.
    """
    values = (
        returns.to_numpy(dtype=float)
        if isinstance(returns, pd.DataFrame)
        else np.asarray(returns, dtype=float)
    )
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] == 0:
        raise ValueError("returns must have shape (T, n_assets) with T >= 2.")
    if not np.all(np.isfinite(values)):
        raise ValueError("returns must contain only finite values.")

    n_regimes = len(regime_names)
    if n_regimes == 0 or len(set(regime_names)) != n_regimes:
        raise ValueError("regime_names must be non-empty and unique.")
    regimes = _validate_regimes(regimes, n_regimes)
    if regimes.shape[0] != values.shape[0]:
        raise ValueError("returns and regimes must have the same number of periods.")

    transition_matrix, initial_distribution = _validate_markov_inputs(
        transition_matrix,
        initial_distribution,
    )
    if transition_matrix.shape[0] != n_regimes:
        raise ValueError("regime_names must align with the transition matrix.")

    target_means = np.asarray(target_means, dtype=float)
    target_covariances = np.asarray(target_covariances, dtype=float)
    expected_mean_shape = (n_regimes, values.shape[1])
    expected_covariance_shape = (n_regimes, values.shape[1], values.shape[1])
    if target_means.shape != expected_mean_shape:
        raise ValueError(f"target_means must have shape {expected_mean_shape}.")
    if target_covariances.shape != expected_covariance_shape:
        raise ValueError(f"target_covariances must have shape {expected_covariance_shape}.")
    if not np.all(np.isfinite(target_means)) or not np.all(
        np.isfinite(target_covariances)
    ):
        raise ValueError("target moments must contain only finite values.")
    for regime, covariance in enumerate(target_covariances):
        if not np.allclose(covariance, covariance.T, atol=1e-10):
            raise ValueError(f"target_covariances[{regime}] must be symmetric.")
        minimum_eigenvalue = np.linalg.eigvalsh(covariance).min()
        if minimum_eigenvalue < -1e-10:
            raise ValueError(
                f"target_covariances[{regime}] must be positive semidefinite."
            )

    thresholds = thresholds or DiagnosticThresholds()
    counts = np.bincount(regimes, minlength=n_regimes)
    if np.any(counts < 2):
        sparse = np.flatnonzero(counts < 2).tolist()
        raise ValueError(f"at least two returns are required in every regime: {sparse}.")

    empirical_frequencies = counts / regimes.size
    expected_frequencies = expected_regime_frequencies(
        transition_matrix,
        initial_distribution,
        regimes.size,
    )
    frequency_errors = empirical_frequencies - expected_frequencies
    observed_transition_matrix = empirical_transition_matrix(regimes, n_regimes)
    transition_errors = observed_transition_matrix - transition_matrix

    empirical_means = np.empty_like(target_means)
    empirical_covariances = np.empty_like(target_covariances)
    mean_z_scores = np.empty_like(target_means)
    variance_standardized_errors = np.empty_like(target_means)
    covariance_relative_errors = np.empty(n_regimes, dtype=float)
    covariance_error_ratios = np.empty(n_regimes, dtype=float)

    for regime in range(n_regimes):
        samples = values[regimes == regime]
        sample_count = samples.shape[0]
        empirical_means[regime] = samples.mean(axis=0)
        empirical_covariances[regime] = np.cov(samples, rowvar=False, ddof=1)

        target_variances = np.diag(target_covariances[regime])
        if np.any(target_variances <= 0.0):
            raise ValueError("target covariance diagonals must be strictly positive.")
        mean_standard_errors = np.sqrt(target_variances / sample_count)
        mean_z_scores[regime] = (
            empirical_means[regime] - target_means[regime]
        ) / mean_standard_errors

        empirical_variances = np.diag(empirical_covariances[regime])
        variance_standard_errors = target_variances * np.sqrt(
            2.0 / (sample_count - 1)
        )
        variance_standardized_errors[regime] = (
            empirical_variances - target_variances
        ) / variance_standard_errors

        covariance_difference = (
            empirical_covariances[regime] - target_covariances[regime]
        )
        covariance_norm = np.linalg.norm(target_covariances[regime], ord="fro")
        observed_error = np.linalg.norm(covariance_difference, ord="fro")
        covariance_relative_errors[regime] = observed_error / covariance_norm

        expected_squared_error = (
            covariance_norm**2 + np.trace(target_covariances[regime]) ** 2
        ) / (sample_count - 1)
        covariance_error_ratios[regime] = observed_error / np.sqrt(
            expected_squared_error
        )

    checks = {
        "regime_frequencies": bool(
            np.max(np.abs(frequency_errors)) <= thresholds.max_frequency_error
        ),
        "regime_transitions_and_persistence": bool(
            np.max(np.abs(transition_errors)) <= thresholds.max_transition_error
        ),
        "regime_means": bool(
            np.max(np.abs(mean_z_scores)) <= thresholds.max_mean_z_score
        ),
        "regime_variances": bool(
            np.max(np.abs(variance_standardized_errors))
            <= thresholds.max_variance_standardized_error
        ),
        "regime_covariances": bool(
            np.max(covariance_error_ratios)
            <= thresholds.max_covariance_error_ratio
        ),
    }

    return SimulationDiagnostics(
        regime_names=tuple(regime_names),
        counts=counts,
        expected_frequencies=expected_frequencies,
        empirical_frequencies=empirical_frequencies,
        target_transition_matrix=transition_matrix.copy(),
        empirical_transition_matrix=observed_transition_matrix,
        empirical_means=empirical_means,
        empirical_covariances=empirical_covariances,
        frequency_errors=frequency_errors,
        transition_errors=transition_errors,
        mean_z_scores=mean_z_scores,
        variance_standardized_errors=variance_standardized_errors,
        covariance_relative_errors=covariance_relative_errors,
        covariance_error_ratios=covariance_error_ratios,
        checks=checks,
        thresholds=thresholds,
    )


def _validate_markov_inputs(
    transition_matrix: np.ndarray,
    initial_distribution: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    transition_matrix = np.asarray(transition_matrix, dtype=float)
    initial_distribution = np.asarray(initial_distribution, dtype=float)
    if transition_matrix.ndim != 2 or transition_matrix.shape[0] != transition_matrix.shape[1]:
        raise ValueError("transition_matrix must be square.")
    if initial_distribution.shape != (transition_matrix.shape[0],):
        raise ValueError("initial_distribution must align with transition_matrix.")
    if not np.all(np.isfinite(transition_matrix)) or np.any(transition_matrix < 0.0):
        raise ValueError("transition_matrix must contain finite nonnegative values.")
    if not np.allclose(transition_matrix.sum(axis=1), 1.0, atol=1e-10):
        raise ValueError("transition_matrix rows must sum to one.")
    if not np.all(np.isfinite(initial_distribution)) or np.any(
        initial_distribution < 0.0
    ):
        raise ValueError("initial_distribution must contain finite nonnegative values.")
    if not np.isclose(initial_distribution.sum(), 1.0, atol=1e-10):
        raise ValueError("initial_distribution must sum to one.")
    return transition_matrix, initial_distribution


def _validate_regimes(regimes: np.ndarray, n_regimes: int) -> np.ndarray:
    if not isinstance(n_regimes, int) or isinstance(n_regimes, bool) or n_regimes <= 0:
        raise ValueError("n_regimes must be a positive integer.")
    regimes = np.asarray(regimes)
    if regimes.ndim != 1 or regimes.size == 0:
        raise ValueError("regimes must be a non-empty 1-D array.")
    if not np.issubdtype(regimes.dtype, np.integer):
        raise ValueError("regimes must contain integer labels.")
    regimes = regimes.astype(int, copy=False)
    if np.any(regimes < 0) or np.any(regimes >= n_regimes):
        raise ValueError("regime labels must lie in [0, n_regimes).")
    return regimes
