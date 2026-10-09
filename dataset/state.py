"""State and action representation for the learned policy.

Equation (6) of the exposé defines the state at time ``t`` as

    s_t = [ r_{t-L+1:t}, mu_hat_{t+1|t}, vech(Sigma_hat_{t+1|t}),
            p_hat_t, w_{t-1}, c, gamma ]

and the action as ``a_t = w_t`` on the long-only simplex. This module builds
that vector, reports its block structure, and offers the compact covariance
encoding the exposé mentions as a scalability option.

Every component is available at ``t`` under ``F_t``: the return window is
strictly past, the moments come from a causal forecaster, the regime
probabilities are filtered, and ``w_{t-1}`` is the previously implemented
portfolio. No component may depend on ``returns[t]`` or later.

Dimensions grow quickly. With ``L = 20`` and ``K = 3``:

| ``N`` | vech | full state | compact state |
|------:|-----:|-----------:|--------------:|
| 8     |   36 |        217 |           195 |
| 50    | 1275 |       2380 |          1161 |
| 200   |20100 |      24505 |          4611 |

The full encoding follows Equation (6) exactly and is the default. The compact
encoding replaces ``vech`` with per-asset volatilities and six correlation
summary statistics, which keeps the state linear in ``N``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

COVARIANCE_ENCODINGS = ("vech", "compact")

#: Number of scalars produced by the compact correlation summary.
_CORRELATION_SUMMARY_SIZE = 6


@dataclass(frozen=True)
class StateSpec:
    """Which blocks make up the state vector, and how they are encoded."""

    lookback: int = 20
    covariance_encoding: str = "vech"
    include_regime_probabilities: bool = True
    include_previous_weights: bool = True
    include_economic_parameters: bool = True

    def __post_init__(self) -> None:
        if (
            not isinstance(self.lookback, int)
            or isinstance(self.lookback, bool)
            or self.lookback < 1
        ):
            raise ValueError("lookback must be a positive integer.")
        if self.covariance_encoding not in COVARIANCE_ENCODINGS:
            raise ValueError(
                f"unknown covariance_encoding {self.covariance_encoding!r}; "
                f"expected one of {COVARIANCE_ENCODINGS}."
            )
        for field_name in (
            "include_regime_probabilities",
            "include_previous_weights",
            "include_economic_parameters",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise TypeError(f"{field_name} must be a bool.")

    def covariance_size(self, n_assets: int) -> int:
        """Return the number of scalars used to encode the covariance."""
        if self.covariance_encoding == "vech":
            return n_assets * (n_assets + 1) // 2
        return n_assets + _CORRELATION_SUMMARY_SIZE

    def block_sizes(self, n_assets: int, n_regimes: int) -> dict[str, int]:
        """Return the size of each state block, in assembly order."""
        _validate_dimensions(n_assets, n_regimes)
        sizes = {
            "return_window": self.lookback * n_assets,
            "expected_returns": n_assets,
            "covariance": self.covariance_size(n_assets),
        }
        if self.include_regime_probabilities:
            sizes["regime_probabilities"] = n_regimes
        if self.include_previous_weights:
            sizes["previous_weights"] = n_assets
        if self.include_economic_parameters:
            sizes["economic_parameters"] = 2
        return sizes

    def block_slices(self, n_assets: int, n_regimes: int) -> dict[str, slice]:
        """Return the slice occupied by each block, for interpretation and plots."""
        slices: dict[str, slice] = {}
        start = 0
        for name, size in self.block_sizes(n_assets, n_regimes).items():
            slices[name] = slice(start, start + size)
            start += size
        return slices

    def dimension(self, n_assets: int, n_regimes: int) -> int:
        """Return the total state dimension."""
        return int(sum(self.block_sizes(n_assets, n_regimes).values()))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable description."""
        return {
            "lookback": self.lookback,
            "covariance_encoding": self.covariance_encoding,
            "include_regime_probabilities": self.include_regime_probabilities,
            "include_previous_weights": self.include_previous_weights,
            "include_economic_parameters": self.include_economic_parameters,
        }


class StateAssembler:
    """Assemble Equation (6) state vectors from causally available components."""

    def __init__(
        self,
        spec: StateSpec,
        *,
        n_assets: int,
        n_regimes: int,
        transaction_cost: float,
        risk_aversion: float,
    ) -> None:
        _validate_dimensions(n_assets, n_regimes)
        if not np.isfinite(transaction_cost) or transaction_cost < 0.0:
            raise ValueError("transaction_cost must be finite and nonnegative.")
        if not np.isfinite(risk_aversion) or risk_aversion < 0.0:
            raise ValueError("risk_aversion must be finite and nonnegative.")

        self.spec = spec
        self.n_assets = n_assets
        self.n_regimes = n_regimes
        self.transaction_cost = float(transaction_cost)
        self.risk_aversion = float(risk_aversion)
        self._lower_indices = np.tril_indices(n_assets)
        self._offdiagonal = ~np.eye(n_assets, dtype=bool)

    @property
    def dimension(self) -> int:
        return self.spec.dimension(self.n_assets, self.n_regimes)

    def block_slices(self) -> dict[str, slice]:
        return self.spec.block_slices(self.n_assets, self.n_regimes)

    def assemble(
        self,
        *,
        return_window: np.ndarray,
        expected_returns: np.ndarray,
        covariance: np.ndarray,
        regime_probabilities: np.ndarray | None = None,
        previous_weights: np.ndarray | None = None,
    ) -> np.ndarray:
        """Build one state vector.

        Args:
            return_window: Shape ``(lookback, n_assets)``, strictly past returns.
            expected_returns: ``mu_hat_{t+1|t}``, shape ``(n_assets,)``.
            covariance: ``Sigma_hat_{t+1|t}``, shape ``(n_assets, n_assets)``.
            regime_probabilities: Filtered ``p_hat_t``, shape ``(n_regimes,)``.
            previous_weights: ``w_{t-1}``, shape ``(n_assets,)``.
        """
        window = self._check(
            return_window, (self.spec.lookback, self.n_assets), "return_window"
        )
        mean = self._check(expected_returns, (self.n_assets,), "expected_returns")
        covariance_matrix = self._check(
            covariance, (self.n_assets, self.n_assets), "covariance"
        )

        blocks = [window.reshape(-1), mean, self._encode_covariance(covariance_matrix)]

        if self.spec.include_regime_probabilities:
            if regime_probabilities is None:
                raise ValueError("regime_probabilities are required by this StateSpec.")
            probabilities = self._check(
                regime_probabilities, (self.n_regimes,), "regime_probabilities"
            )
            if abs(float(probabilities.sum()) - 1.0) > 1e-6:
                raise ValueError("regime_probabilities must sum to one.")
            blocks.append(probabilities)

        if self.spec.include_previous_weights:
            if previous_weights is None:
                raise ValueError("previous_weights are required by this StateSpec.")
            blocks.append(self._check(previous_weights, (self.n_assets,), "previous_weights"))

        if self.spec.include_economic_parameters:
            blocks.append(np.array([self.transaction_cost, self.risk_aversion]))

        state = np.concatenate(blocks)
        if state.size != self.dimension:  # pragma: no cover - guarded by block sizes
            raise RuntimeError(
                f"assembled state has {state.size} entries; expected {self.dimension}."
            )
        return state

    def _encode_covariance(self, covariance: np.ndarray) -> np.ndarray:
        if self.spec.covariance_encoding == "vech":
            return covariance[self._lower_indices]
        return self._compact_covariance(covariance)

    def _compact_covariance(self, covariance: np.ndarray) -> np.ndarray:
        """Encode a covariance as volatilities plus correlation summary statistics.

        The six summaries are the mean, standard deviation, minimum, and maximum
        off-diagonal correlation, the share of total variance explained by the
        leading eigenvalue, and the effective rank. Together they capture the
        level and concentration of co-movement, which is what the allocation
        actually responds to, at a cost linear rather than quadratic in
        ``n_assets``.
        """
        volatilities = np.sqrt(np.maximum(np.diag(covariance), 0.0))
        safe = np.where(volatilities > 0.0, volatilities, 1.0)
        correlation = covariance / np.outer(safe, safe)
        off_diagonal = correlation[self._offdiagonal]

        eigenvalues = np.clip(np.linalg.eigvalsh(covariance), 0.0, None)
        total = float(eigenvalues.sum())
        if total > 0.0:
            shares = eigenvalues / total
            leading_share = float(shares.max())
            positive = shares[shares > 0.0]
            effective_rank = float(np.exp(-np.sum(positive * np.log(positive))))
        else:  # pragma: no cover - a zero covariance is rejected upstream
            leading_share = 0.0
            effective_rank = 0.0

        summary = np.array(
            [
                float(off_diagonal.mean()) if off_diagonal.size else 0.0,
                float(off_diagonal.std()) if off_diagonal.size else 0.0,
                float(off_diagonal.min()) if off_diagonal.size else 0.0,
                float(off_diagonal.max()) if off_diagonal.size else 0.0,
                leading_share,
                effective_rank,
            ]
        )
        return np.concatenate([volatilities, summary])

    @staticmethod
    def _check(values: np.ndarray, shape: tuple[int, ...], name: str) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        if array.shape != shape:
            raise ValueError(f"{name} must have shape {shape}; got {array.shape}.")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain only finite values.")
        return array

    def describe(self) -> dict[str, Any]:
        """Return a JSON-serialisable description for experiment metadata."""
        return {
            "spec": self.spec.to_dict(),
            "n_assets": self.n_assets,
            "n_regimes": self.n_regimes,
            "transaction_cost": self.transaction_cost,
            "risk_aversion": self.risk_aversion,
            "dimension": self.dimension,
            "block_sizes": self.spec.block_sizes(self.n_assets, self.n_regimes),
        }


def _validate_dimensions(n_assets: int, n_regimes: int) -> None:
    for value, name in ((n_assets, "n_assets"), (n_regimes, "n_regimes")):
        if not isinstance(value, (int, np.integer)) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
