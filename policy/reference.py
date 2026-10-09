"""Non-learned reference allocations, used as the bar behavioural cloning must clear.

A behavioural-cloning tracking error is not interpretable on its own — it has
units of squared portfolio weight and no natural scale. It becomes interpretable
against rules that require no learning at all, and there are two that matter:

``EqualWeightReference``
    The ``1/N`` portfolio of strategy 1 in Section 5.2. It says how much of the
    expert's allocation is explained by "hold a diversified portfolio".

``PreviousWeightsReference``
    Return ``w_{t-1}`` unchanged, i.e. never trade. This is the strong one. The
    expert pays a proportional cost on every adjustment and therefore rebalances
    gently, so persistence explains a large share of its behaviour — and
    ``w_{t-1}`` is a block of the state, meaning a trained policy could reproduce
    this rule by reading one slice and ignoring everything else. Beating it is
    the minimum evidence that a policy learned the expert's *decision* rather
    than its inertia.

Both read from raw, unstandardised states through ``StateSpec.block_slices``, so
they consume exactly the information the learned policies do.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..dataset.state import StateSpec
from ..expert.constraints import PortfolioConstraints


class ReferencePolicy:
    """A deterministic allocation rule with no trainable parameters."""

    name = "reference"

    def __init__(
        self,
        spec: StateSpec,
        *,
        n_assets: int,
        n_regimes: int,
        constraints: PortfolioConstraints | None = None,
    ) -> None:
        self.spec = spec
        self.n_assets = int(n_assets)
        self.n_regimes = int(n_regimes)
        self.constraints = (
            constraints if constraints is not None else PortfolioConstraints()
        )
        self.constraints.validate_for(self.n_assets)
        self.state_dimension = spec.dimension(n_assets, n_regimes)
        self._slices = spec.block_slices(n_assets, n_regimes)

    def act(self, states: np.ndarray) -> np.ndarray:
        values = np.asarray(states, dtype=float)
        single = values.ndim == 1
        if single:
            values = values[np.newaxis, :]
        if values.ndim != 2 or values.shape[1] != self.state_dimension:
            raise ValueError(
                f"states must have shape (batch, {self.state_dimension}) or "
                f"({self.state_dimension},); got {np.shape(states)}."
            )
        weights = self._allocate(values)
        return weights[0] if single else weights

    def _allocate(self, states: np.ndarray) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def describe(self) -> dict[str, Any]:
        return {"kind": self.name, "n_assets": self.n_assets}


class EqualWeightReference(ReferencePolicy):
    """Always allocate ``1/N``, projected onto the feasible set."""

    name = "equal_weight"

    def _allocate(self, states: np.ndarray) -> np.ndarray:
        weights = self.constraints.default_weights(self.n_assets)
        return np.tile(weights, (states.shape[0], 1))


class PreviousWeightsReference(ReferencePolicy):
    """Return ``w_{t-1}`` from the state: the never-trade rule."""

    name = "previous_weights"

    def __init__(self, spec: StateSpec, **kwargs: Any) -> None:
        super().__init__(spec, **kwargs)
        if "previous_weights" not in self._slices:
            raise ValueError(
                "PreviousWeightsReference requires a StateSpec that includes w_{t-1}."
            )

    def _allocate(self, states: np.ndarray) -> np.ndarray:
        previous = states[:, self._slices["previous_weights"]]
        return np.stack([self.constraints.project(row) for row in previous])


def default_references(
    spec: StateSpec,
    *,
    n_assets: int,
    n_regimes: int,
    constraints: PortfolioConstraints | None = None,
) -> dict[str, ReferencePolicy]:
    """Return the reference rules a trained policy is required to beat."""
    return {
        EqualWeightReference.name: EqualWeightReference(
            spec, n_assets=n_assets, n_regimes=n_regimes, constraints=constraints
        ),
        PreviousWeightsReference.name: PreviousWeightsReference(
            spec, n_assets=n_assets, n_regimes=n_regimes, constraints=constraints
        ),
    }
