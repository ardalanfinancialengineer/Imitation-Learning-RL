"""Reusable, documented parameterisations for regime-switching simulations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .regimes import RegimeSwitchingModel
from .returns import ReturnSimulator

CANONICAL_PRESET_NAME = "canonical_three_regime"
CANONICAL_REGIME_NAMES = ("calm", "volatile", "stressed")


def constant_correlation_covariance(
    volatilities: np.ndarray,
    correlation: float,
) -> np.ndarray:
    """Construct a covariance matrix from volatilities and one correlation.

    The admissible lower bound for an ``n``-dimensional constant-correlation
    matrix is ``-1 / (n - 1)``. Equality is accepted because it produces a
    positive-semidefinite (singular) matrix.
    """
    volatilities = np.asarray(volatilities, dtype=float)
    if volatilities.ndim != 1 or volatilities.size == 0:
        raise ValueError("volatilities must be a non-empty 1-D array.")
    if not np.all(np.isfinite(volatilities)) or np.any(volatilities <= 0.0):
        raise ValueError("volatilities must contain positive finite values.")
    if isinstance(correlation, bool) or not isinstance(
        correlation,
        (int, float, np.integer, np.floating),
    ):
        raise ValueError("correlation must be a finite scalar.")
    correlation_value = float(correlation)
    if not np.isfinite(correlation_value):
        raise ValueError("correlation must be a finite scalar.")
    if not -1.0 <= correlation_value <= 1.0:
        raise ValueError("correlation must lie in [-1, 1].")
    if volatilities.size > 1:
        lower_bound = -1.0 / (volatilities.size - 1)
        if correlation_value < lower_bound:
            raise ValueError(
                "correlation is too negative for a positive-semidefinite "
                "constant-correlation matrix."
            )

    correlation_matrix = np.full(
        (volatilities.size, volatilities.size),
        correlation_value,
        dtype=float,
    )
    np.fill_diagonal(correlation_matrix, 1.0)
    return np.outer(volatilities, volatilities) * correlation_matrix


def constant_correlation_covariances(
    volatilities: np.ndarray,
    correlations: np.ndarray | float,
) -> np.ndarray:
    """Construct one constant-correlation covariance matrix per regime."""
    volatilities = np.asarray(volatilities, dtype=float)
    if volatilities.ndim != 2:
        raise ValueError("volatilities must have shape (n_regimes, n_assets).")
    if volatilities.shape[0] == 0:
        raise ValueError("volatilities must contain at least one regime.")

    correlation_values = np.asarray(correlations, dtype=float)
    if correlation_values.ndim == 0:
        correlation_values = np.full(volatilities.shape[0], float(correlation_values))
    if correlation_values.shape != (volatilities.shape[0],):
        raise ValueError("correlations must be scalar or have shape (n_regimes,).")

    return np.stack(
        [
            constant_correlation_covariance(regime_volatilities, correlation)
            for regime_volatilities, correlation in zip(
                volatilities,
                correlation_values,
                strict=True,
            )
        ]
    )


@dataclass(frozen=True)
class RegimeSimulationPreset:
    """Complete return-generating parameters for a named regime model."""

    name: str
    regime_names: tuple[str, ...]
    transition_matrix: np.ndarray
    initial_distribution: np.ndarray
    regime_means: np.ndarray
    regime_covariances: np.ndarray

    @property
    def n_assets(self) -> int:
        return int(self.regime_means.shape[1])

    @property
    def n_regimes(self) -> int:
        return len(self.regime_names)

    def build_simulator(self) -> ReturnSimulator:
        """Build an independent simulator from this preset."""
        regime_model = RegimeSwitchingModel(
            n_regimes=self.n_regimes,
            transition_matrix=self.transition_matrix.copy(),
            initial_distribution=self.initial_distribution.copy(),
        )
        return ReturnSimulator(
            n_assets=self.n_assets,
            regime_model=regime_model,
            regime_means=self.regime_means.copy(),
            regime_covs=self.regime_covariances.copy(),
        )


def canonical_three_regime_preset(n_assets: int) -> RegimeSimulationPreset:
    """Return the project's scalable calm/volatile/stressed market preset.

    Means, volatilities, and correlations worsen monotonically from calm to
    stressed. The same construction supports development universes and the
    final ``N in {50, 100, 200}`` experiment sizes without storing large
    covariance matrices in YAML.
    """
    if not isinstance(n_assets, int) or isinstance(n_assets, bool) or n_assets <= 0:
        raise ValueError("n_assets must be a positive integer.")

    transition_matrix = np.array(
        [
            [0.92, 0.07, 0.01],
            [0.08, 0.84, 0.08],
            [0.03, 0.17, 0.80],
        ],
        dtype=float,
    )
    initial_distribution = np.array([0.70, 0.20, 0.10], dtype=float)
    regime_means = np.vstack(
        [
            np.linspace(0.0002, 0.0006, n_assets),
            np.linspace(-0.0002, 0.0003, n_assets),
            np.linspace(-0.0012, -0.0004, n_assets),
        ]
    )
    regime_volatilities = np.vstack(
        [
            np.linspace(0.008, 0.014, n_assets),
            np.linspace(0.016, 0.030, n_assets),
            np.linspace(0.025, 0.045, n_assets),
        ]
    )
    regime_covariances = constant_correlation_covariances(
        regime_volatilities,
        correlations=np.array([0.20, 0.40, 0.55]),
    )

    return RegimeSimulationPreset(
        name=CANONICAL_PRESET_NAME,
        regime_names=CANONICAL_REGIME_NAMES,
        transition_matrix=transition_matrix,
        initial_distribution=initial_distribution,
        regime_means=regime_means,
        regime_covariances=regime_covariances,
    )


UNMATCHED_PRESET_NAME = "unmatched_three_regime"


def unmatched_three_regime_preset(n_assets: int) -> RegimeSimulationPreset:
    """Return a calm/volatile/stressed preset with a genuine cross-sectional edge.

    The canonical preset builds regime means and volatilities with *matched*
    ``linspace`` ranges (means and vols both ascend across assets), so
    ``mu_i / sigma_i^2`` is nearly constant, the mean-variance frontier is close
    to degenerate, and equal weight sits almost on it. That makes the Markowitz
    expert a weak benchmark (open decision 1).

    This variant keeps everything about the canonical preset — the three-regime
    structure, the transition dynamics, the initial distribution, the
    correlations, and the *set* of per-regime means and volatilities — but
    **reverses the volatility ordering relative to the means** within each
    regime. High-mean assets are therefore paired with low volatility rather than
    high, so ``mu_i / sigma_i^2`` varies substantially across the cross-section
    and the optimiser has a real risk-adjusted tilt to exploit that equal weight
    cannot capture. It exists to test empirically whether the expert beats equal
    weight once the frontier is non-degenerate.
    """
    if not isinstance(n_assets, int) or isinstance(n_assets, bool) or n_assets <= 0:
        raise ValueError("n_assets must be a positive integer.")

    base = canonical_three_regime_preset(n_assets)
    # Reverse the asset ordering of the volatilities within every regime, so the
    # highest-mean asset carries the lowest volatility. Means and correlations
    # are left exactly as the canonical preset defines them.
    reversed_volatilities = np.sqrt(
        np.diagonal(base.regime_covariances, axis1=1, axis2=2)
    )[:, ::-1]
    regime_covariances = constant_correlation_covariances(
        reversed_volatilities,
        correlations=np.array([0.20, 0.40, 0.55]),
    )

    return RegimeSimulationPreset(
        name=UNMATCHED_PRESET_NAME,
        regime_names=CANONICAL_REGIME_NAMES,
        transition_matrix=base.transition_matrix.copy(),
        initial_distribution=base.initial_distribution.copy(),
        regime_means=base.regime_means.copy(),
        regime_covariances=regime_covariances,
    )


def shifted_three_regime_preset(
    n_assets: int,
    *,
    name: str = "shifted_three_regime",
    persistence_scale: float = 1.0,
    correlation_scale: float = 1.0,
    volatility_scale: float = 1.0,
    mean_scale: float = 1.0,
) -> RegimeSimulationPreset:
    """Return a perturbed variant of the canonical preset.

    Used to build the exposé's distribution-shift / zero-shot evaluation
    setting: unseen transition matrices, covariance configurations, and regime
    persistence patterns. Each factor perturbs exactly one aspect so a shift
    result can be attributed.

    ``persistence_scale`` multiplies the self-transition probabilities and
    renormalises the off-diagonal mass, so values below one make regimes switch
    more often. The other factors scale correlations, volatilities, and means.
    """
    base = canonical_three_regime_preset(n_assets)
    for value, label in (
        (persistence_scale, "persistence_scale"),
        (correlation_scale, "correlation_scale"),
        (volatility_scale, "volatility_scale"),
        (mean_scale, "mean_scale"),
    ):
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{label} must be positive and finite.")

    transition_matrix = _rescale_persistence(base.transition_matrix, persistence_scale)

    base_correlations = np.array([0.20, 0.40, 0.55], dtype=float)
    correlations = np.clip(base_correlations * correlation_scale, -0.99, 0.99)
    volatilities = (
        np.sqrt(np.diagonal(base.regime_covariances, axis1=1, axis2=2)) * volatility_scale
    )

    return RegimeSimulationPreset(
        name=name,
        regime_names=base.regime_names,
        transition_matrix=transition_matrix,
        initial_distribution=base.initial_distribution.copy(),
        regime_means=base.regime_means * mean_scale,
        regime_covariances=constant_correlation_covariances(volatilities, correlations),
    )


def _rescale_persistence(transition_matrix: np.ndarray, scale: float) -> np.ndarray:
    """Scale self-transition probabilities and renormalise each row."""
    rescaled = transition_matrix.copy()
    n_regimes = rescaled.shape[0]
    for regime in range(n_regimes):
        persistence = float(np.clip(rescaled[regime, regime] * scale, 1e-6, 1.0 - 1e-6))
        off_diagonal = rescaled[regime].copy()
        off_diagonal[regime] = 0.0
        remaining = off_diagonal.sum()
        if remaining <= 0.0:
            off_diagonal[:] = (1.0 - persistence) / (n_regimes - 1)
        else:
            off_diagonal *= (1.0 - persistence) / remaining
        off_diagonal[regime] = persistence
        rescaled[regime] = off_diagonal
    return rescaled


def load_simulation_preset(name: str, n_assets: int) -> RegimeSimulationPreset:
    """Load a simulation preset by its stable configuration name."""
    if name == CANONICAL_PRESET_NAME:
        return canonical_three_regime_preset(n_assets)
    if name == UNMATCHED_PRESET_NAME:
        return unmatched_three_regime_preset(n_assets)
    raise ValueError(
        f"unknown simulation preset {name!r}; expected one of "
        f"{CANONICAL_PRESET_NAME!r} or {UNMATCHED_PRESET_NAME!r}."
    )
