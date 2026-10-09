"""Portfolio constraints for the rolling-horizon Markowitz expert.

The exposé constrains every planned allocation by

    1^T w = 1,
    l_i <= w_i <= u_i,
    ||w||_1 <= L.

The long-only baseline uses ``l_i = 0``, which makes the leverage limit
redundant because ``||w||_1 = 1``. The limit is retained so later extensions
with limited short positions do not need a different constraint object.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FEASIBILITY_TOLERANCE = 1e-6


@dataclass(frozen=True)
class PortfolioConstraints:
    """Budget, box, and leverage constraints shared by expert and policies."""

    long_only: bool = True
    min_weight: float = 0.0
    max_weight: float = 1.0
    leverage_limit: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.long_only, bool):
            raise TypeError("long_only must be a bool.")
        minimum = _finite_scalar(self.min_weight, "min_weight")
        maximum = _finite_scalar(self.max_weight, "max_weight")
        if minimum > maximum:
            raise ValueError("min_weight must not exceed max_weight.")
        if self.long_only and minimum < 0.0:
            raise ValueError("min_weight must be nonnegative in a long-only portfolio.")
        if maximum <= 0.0:
            raise ValueError("max_weight must be positive.")

        object.__setattr__(self, "min_weight", minimum)
        object.__setattr__(self, "max_weight", maximum)

        if self.leverage_limit is not None:
            leverage = _finite_scalar(self.leverage_limit, "leverage_limit")
            if leverage < 1.0:
                raise ValueError(
                    "leverage_limit must be at least 1 for a fully invested portfolio."
                )
            object.__setattr__(self, "leverage_limit", leverage)

    @property
    def effective_min_weight(self) -> float:
        """Lower bound after applying the long-only restriction."""
        return max(self.min_weight, 0.0) if self.long_only else self.min_weight

    def validate_for(self, n_assets: int) -> None:
        """Raise if a fully invested portfolio cannot satisfy the box bounds."""
        if not isinstance(n_assets, int) or isinstance(n_assets, bool) or n_assets <= 0:
            raise ValueError("n_assets must be a positive integer.")
        if self.max_weight * n_assets < 1.0 - FEASIBILITY_TOLERANCE:
            raise ValueError(
                f"max_weight={self.max_weight} cannot fund a fully invested portfolio "
                f"with {n_assets} assets."
            )
        if self.effective_min_weight * n_assets > 1.0 + FEASIBILITY_TOLERANCE:
            raise ValueError(
                f"min_weight={self.effective_min_weight} over-funds a portfolio "
                f"with {n_assets} assets."
            )

    def violations(
        self,
        weights: np.ndarray,
        *,
        tolerance: float = FEASIBILITY_TOLERANCE,
    ) -> dict[str, float]:
        """Return the size of every constraint violation, zero where satisfied.

        Accepts one weight vector ``(n_assets,)`` or a batch
        ``(n_samples, n_assets)`` and reports the worst violation in the batch.
        """
        values = np.asarray(weights, dtype=float)
        if values.ndim == 1:
            values = values[np.newaxis, :]
        if values.ndim != 2 or values.shape[1] == 0:
            raise ValueError("weights must have shape (n_assets,) or (n_samples, n_assets).")
        if not np.all(np.isfinite(values)):
            return {"non_finite": np.inf}

        report = {
            "budget": float(np.max(np.abs(values.sum(axis=1) - 1.0))),
            "upper_bound": float(max(0.0, np.max(values - self.max_weight))),
            "lower_bound": float(max(0.0, np.max(self.effective_min_weight - values))),
        }
        if self.leverage_limit is not None:
            excess = np.abs(values).sum(axis=1) - self.leverage_limit
            report["leverage"] = float(max(0.0, np.max(excess)))
        return {name: (0.0 if size <= tolerance else size) for name, size in report.items()}

    def is_feasible(
        self,
        weights: np.ndarray,
        *,
        tolerance: float = FEASIBILITY_TOLERANCE,
    ) -> bool:
        """Return whether every constraint holds within ``tolerance``."""
        return all(
            size == 0.0 for size in self.violations(weights, tolerance=tolerance).values()
        )

    def project(self, weights: np.ndarray) -> np.ndarray:
        """Project one weight vector onto the budget and box constraints.

        Used as a numerical safety net when a solver returns a slightly
        infeasible point. The projection is Euclidean onto
        ``{w : 1^T w = 1, l <= w <= u}`` and is solved by bisection on the
        single dual variable of the budget constraint.
        """
        values = np.asarray(weights, dtype=float)
        if values.ndim != 1 or values.size == 0:
            raise ValueError("weights must be a non-empty 1-D array.")
        if not np.all(np.isfinite(values)):
            raise ValueError("weights must contain only finite values.")

        n_assets = values.size
        self.validate_for(n_assets)
        lower = self.effective_min_weight
        upper = self.max_weight

        low = float(values.min() - upper)
        high = float(values.max() - lower)
        for _ in range(200):
            midpoint = 0.5 * (low + high)
            if np.clip(values - midpoint, lower, upper).sum() > 1.0:
                low = midpoint
            else:
                high = midpoint

        projected = np.clip(values - high, lower, upper)
        residual = 1.0 - projected.sum()
        if abs(residual) > 1e-12:
            free = np.flatnonzero(
                (projected > lower + 1e-12) & (projected < upper - 1e-12)
            )
            if free.size:
                projected[free] += residual / free.size
                projected = np.clip(projected, lower, upper)
        return projected

    def default_weights(self, n_assets: int) -> np.ndarray:
        """Return a feasible fallback allocation, the equal-weight portfolio."""
        self.validate_for(n_assets)
        return self.project(np.full(n_assets, 1.0 / n_assets))

    def to_dict(self) -> dict[str, float | bool | None]:
        """Return a JSON-serialisable description of the constraint set."""
        return {
            "long_only": self.long_only,
            "min_weight": self.min_weight,
            "max_weight": self.max_weight,
            "leverage_limit": self.leverage_limit,
        }


def _finite_scalar(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        raise ValueError(f"{name} must be a finite scalar.")
    parsed = float(value)
    if not np.isfinite(parsed):
        raise ValueError(f"{name} must be a finite scalar.")
    return parsed
