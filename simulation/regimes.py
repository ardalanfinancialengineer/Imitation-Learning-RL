"""Hidden Markov Model regime generator for stochastic market states."""

from __future__ import annotations

import numpy as np


class RegimeSwitchingModel:
    """Generates latent market regime sequences via a Markov chain.

    Regimes capture distinct return-generating environments (e.g. bull, bear,
    high-volatility crash), characterised by per-regime means and covariances.
    """

    def __init__(
        self,
        n_regimes: int,
        transition_matrix: np.ndarray,
        initial_distribution: np.ndarray | None = None,
    ) -> None:
        """
        Args:
            n_regimes: Number of latent regimes.
            transition_matrix: (n_regimes, n_regimes) row-stochastic transition matrix.
            initial_distribution: Stationary or custom initial state probabilities.
        """
        if not isinstance(n_regimes, int) or n_regimes <= 0:
            raise ValueError("n_regimes must be a positive integer.")

        transition_matrix = np.asarray(transition_matrix, dtype=float)
        if transition_matrix.shape != (n_regimes, n_regimes):
            raise ValueError(
                "transition_matrix must have shape "
                f"({n_regimes}, {n_regimes}); got {transition_matrix.shape}."
            )
        if not np.all(np.isfinite(transition_matrix)):
            raise ValueError("transition_matrix must contain only finite values.")
        if np.any(transition_matrix < 0.0):
            raise ValueError("transition_matrix probabilities must be nonnegative.")
        if not np.allclose(transition_matrix.sum(axis=1), 1.0, atol=1e-10):
            raise ValueError("transition_matrix rows must sum to one.")

        if initial_distribution is None:
            initial_distribution = np.full(n_regimes, 1.0 / n_regimes)
        else:
            initial_distribution = np.asarray(initial_distribution, dtype=float)
            if initial_distribution.shape != (n_regimes,):
                raise ValueError(
                    "initial_distribution must have shape "
                    f"({n_regimes},); got {initial_distribution.shape}."
                )
            if not np.all(np.isfinite(initial_distribution)):
                raise ValueError("initial_distribution must contain only finite values.")
            if np.any(initial_distribution < 0.0):
                raise ValueError("initial_distribution probabilities must be nonnegative.")
            if not np.isclose(initial_distribution.sum(), 1.0, atol=1e-10):
                raise ValueError("initial_distribution must sum to one.")

        self.n_regimes = n_regimes
        self.transition_matrix = transition_matrix
        self.initial_distribution = initial_distribution

    def sample_regime_sequence(self, T: int, seed: int | None = None) -> np.ndarray:
        """Sample a regime sequence of length T.

        Args:
            T: Number of time steps.
            seed: Random seed for reproducibility.

        Returns:
            Integer array of shape (T,) with values in [0, n_regimes).
        """
        if not isinstance(T, int) or T < 0:
            raise ValueError("T must be a nonnegative integer.")

        regimes = np.empty(T, dtype=int)
        if T == 0:
            return regimes

        rng = np.random.default_rng(seed)
        labels = np.arange(self.n_regimes)

        regimes[0] = rng.choice(labels, p=self.initial_distribution)
        for t in range(1, T):
            regimes[t] = rng.choice(labels, p=self.transition_matrix[regimes[t - 1]])

        return regimes
