"""Proportional transaction costs used by the simulation environment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EXPERIMENT_COST_MIN = 0.001
EXPERIMENT_COST_MAX = 0.005
DEFAULT_TRANSACTION_COSTS = (0.001, 0.0025, 0.005)


def validate_transaction_cost_grid(
    rates: list[float] | tuple[float, ...] | np.ndarray,
    *,
    minimum: float = EXPERIMENT_COST_MIN,
    maximum: float = EXPERIMENT_COST_MAX,
) -> tuple[float, ...]:
    """Validate and canonicalise transaction costs for an experiment.

    The exposé prescribes ``c in [0.001, 0.005]``. A grid must be non-empty,
    finite, unique, increasing, and contained in that interval so experiment
    metadata remains unambiguous and reproducible.
    """
    values = np.asarray(rates, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("transaction cost rates must be a non-empty 1-D sequence.")
    if not np.all(np.isfinite(values)):
        raise ValueError("transaction cost rates must contain only finite values.")
    if not np.isfinite(minimum) or not np.isfinite(maximum) or minimum > maximum:
        raise ValueError("transaction cost bounds must be finite and ordered.")
    if np.any(values < minimum) or np.any(values > maximum):
        raise ValueError(
            f"transaction cost rates must lie in [{minimum:.4f}, {maximum:.4f}]."
        )
    if np.any(np.diff(values) <= 0.0):
        raise ValueError("transaction cost rates must be strictly increasing and unique.")
    return tuple(float(value) for value in values)


@dataclass(frozen=True)
class ProportionalTransactionCost:
    """Apply ``c * ||w_t - w_(t-1)||_1`` exactly as defined in the exposé.

    The model accepts zero for no-cost ablations. Use
    :func:`validate_transaction_cost_grid` to enforce the narrower experiment
    range prescribed by the project plan.
    """

    rate: float

    def __post_init__(self) -> None:
        if isinstance(self.rate, bool) or not isinstance(
            self.rate,
            (int, float, np.integer, np.floating),
        ):
            raise ValueError("transaction cost rate must be a finite scalar.")
        rate = float(self.rate)
        if not np.isfinite(rate):
            raise ValueError("transaction cost rate must be a finite scalar.")
        if not 0.0 <= rate <= 1.0:
            raise ValueError("transaction cost rate must lie in [0, 1].")
        object.__setattr__(self, "rate", rate)

    def turnover(
        self,
        weights: np.ndarray,
        previous_weights: np.ndarray,
    ) -> np.ndarray | float:
        """Return full L1 allocation change along the last array axis."""
        current, previous = _validate_weight_arrays(weights, previous_weights)
        result = np.abs(current - previous).sum(axis=-1)
        return float(result) if result.ndim == 0 else result

    def cost(
        self,
        weights: np.ndarray,
        previous_weights: np.ndarray,
    ) -> np.ndarray | float:
        """Return proportional transaction cost for one or many portfolios."""
        result = np.asarray(self.turnover(weights, previous_weights)) * self.rate
        return float(result) if result.ndim == 0 else result

    def net_return(
        self,
        gross_return: float | np.ndarray,
        weights: np.ndarray,
        previous_weights: np.ndarray,
    ) -> np.ndarray | float:
        """Deduct transaction cost once from realised gross portfolio return."""
        gross = np.asarray(gross_return, dtype=float)
        if not np.all(np.isfinite(gross)):
            raise ValueError("gross_return must contain only finite values.")
        costs = np.asarray(self.cost(weights, previous_weights))
        if gross.ndim > 0 and gross.shape != costs.shape:
            raise ValueError(
                "gross_return must be scalar or exactly match the portfolio batch shape."
            )
        result = gross - costs
        return float(result) if result.ndim == 0 else result


def _validate_weight_arrays(
    weights: np.ndarray,
    previous_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    current = np.asarray(weights, dtype=float)
    previous = np.asarray(previous_weights, dtype=float)
    if current.ndim not in {1, 2}:
        raise ValueError("weights must have shape (n_assets,) or (n_samples, n_assets).")
    if previous.shape != current.shape:
        raise ValueError("weights and previous_weights must have identical shapes.")
    if current.shape[-1] == 0:
        raise ValueError("weight arrays must contain at least one asset.")
    if not np.all(np.isfinite(current)) or not np.all(np.isfinite(previous)):
        raise ValueError("weight arrays must contain only finite values.")
    return current, previous
