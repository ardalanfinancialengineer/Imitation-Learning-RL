"""Synthetic multi-asset return simulation conditioned on latent regimes."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .regimes import RegimeSwitchingModel


class ReturnSimulator:
    """Simulates multivariate stock returns driven by a regime-switching process.

    Each regime has an associated mean vector and covariance matrix. Returns are
    drawn from the corresponding multivariate normal distribution at each step.
    """

    def __init__(
        self,
        n_assets: int,
        regime_model: RegimeSwitchingModel,
        regime_means: Sequence[np.ndarray] | np.ndarray,
        regime_covs: Sequence[np.ndarray] | np.ndarray,
    ) -> None:
        """
        Args:
            n_assets: Number of risky assets.
            regime_model: Fitted or configured RegimeSwitchingModel.
            regime_means: List of mean vectors, one per regime.
            regime_covs: List of covariance matrices, one per regime.
        """
        if not isinstance(n_assets, int) or n_assets <= 0:
            raise ValueError("n_assets must be a positive integer.")
        if not isinstance(regime_model, RegimeSwitchingModel):
            raise TypeError("regime_model must be a RegimeSwitchingModel.")

        n_regimes = regime_model.n_regimes
        means = np.asarray(regime_means, dtype=float)
        covariances = np.asarray(regime_covs, dtype=float)

        if means.shape != (n_regimes, n_assets):
            raise ValueError(
                "regime_means must have shape "
                f"({n_regimes}, {n_assets}); got {means.shape}."
            )
        if covariances.shape != (n_regimes, n_assets, n_assets):
            raise ValueError(
                "regime_covs must have shape "
                f"({n_regimes}, {n_assets}, {n_assets}); got {covariances.shape}."
            )
        if not np.all(np.isfinite(means)):
            raise ValueError("regime_means must contain only finite values.")
        if not np.all(np.isfinite(covariances)):
            raise ValueError("regime_covs must contain only finite values.")

        symmetry_tol = 1e-10
        psd_tol = 1e-10
        for regime, cov in enumerate(covariances):
            if not np.allclose(cov, cov.T, atol=symmetry_tol):
                raise ValueError(f"regime_covs[{regime}] must be symmetric.")
            min_eigenvalue = np.linalg.eigvalsh(cov).min()
            if min_eigenvalue < -psd_tol:
                raise ValueError(
                    f"regime_covs[{regime}] must be positive semidefinite; "
                    f"minimum eigenvalue is {min_eigenvalue:.3g}."
                )

        self.n_assets = n_assets
        self.regime_model = regime_model
        self.regime_means = means
        self.regime_covs = 0.5 * (
            covariances + np.swapaxes(covariances, 1, 2)
        )

    def simulate(
        self, T: int, seed: int | None = None
    ) -> tuple[pd.DataFrame, np.ndarray]:
        """Simulate T periods of asset returns.

        Args:
            T: Number of time steps.
            seed: Random seed for reproducibility.

        Returns:
            Tuple of (returns DataFrame of shape (T, n_assets),
                      regime sequence array of shape (T,)).
        """
        if not isinstance(T, int) or T < 0:
            raise ValueError("T must be a nonnegative integer.")

        seed_sequence = np.random.SeedSequence(seed)
        regime_seed_sequence, returns_seed_sequence = seed_sequence.spawn(2)
        regime_seed = int(regime_seed_sequence.generate_state(1)[0])
        rng = np.random.default_rng(returns_seed_sequence)

        regimes = self.regime_model.sample_regime_sequence(T, seed=regime_seed)
        simulated = np.empty((T, self.n_assets), dtype=float)

        # Draw each regime's observations in one batch. This avoids repeating
        # covariance factorisation at every time step and makes the final
        # 200-asset, 10,000-period configuration practical.
        for regime in range(self.regime_model.n_regimes):
            positions = np.flatnonzero(regimes == regime)
            if positions.size == 0:
                continue
            simulated[positions] = rng.multivariate_normal(
                mean=self.regime_means[regime],
                cov=self.regime_covs[regime],
                size=positions.size,
                check_valid="raise",
                tol=1e-10,
            )

        columns = [f"asset_{i}" for i in range(self.n_assets)]
        returns = pd.DataFrame(simulated, columns=columns)
        return returns, regimes
