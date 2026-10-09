"""Deterministic allocation policy: state encoder plus a projected head.

This is strategy 3 of Section 5.2 — the "deterministic transformer imitation
policy" the diffusion model is measured against. It is deliberately the simplest
thing that can satisfy the action space: encode the state, read out ``N``
scores, and project them onto the feasible set.

**Why a projection and not a softmax.** A softmax head is the reflexive choice
for simplex-valued outputs and it is the wrong one here. It cannot represent an
exact zero, whereas the long-only expert with a ``0.30`` position cap produces
allocations that are mostly exact zeros; and it cannot respect the cap at all,
so a softmax policy would routinely propose positions the expert is forbidden
from taking. The Euclidean projection of Equation (8) handles both: it is onto
the exact constraint set of Equation (5), it produces genuine sparsity, and it
is differentiable almost everywhere. See :mod:`src.policy.simplex`.

**Residual around the previous portfolio.** The head predicts a *deviation* from
``w_{t-1}`` rather than an allocation from scratch. Under transaction costs the
expert is sticky — the optimal move is usually a small adjustment to what is
already held — so predicting the adjustment puts the model's capacity where the
variation actually is, and makes "hold the current portfolio" the zero-output
default rather than something that has to be learned. The previous weights are
read from the state's own ``previous_weights`` block, so this adds no input the
policy did not already have under ``F_t``. It can be switched off to ablate the
choice.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from ..dataset.state import StateSpec
from ..expert.constraints import PortfolioConstraints
from .base import AllocationLoss, AllocationPolicy
from .encoder import EncoderConfig, MLPStateEncoder, TransformerStateEncoder
from .simplex import project_box_simplex

ENCODER_KINDS = ("transformer", "mlp")


class DeterministicPolicy(AllocationPolicy):
    """Encoder plus a linear read-out, projected onto the feasible set."""

    def __init__(
        self,
        spec: StateSpec,
        *,
        n_assets: int,
        n_regimes: int,
        constraints: PortfolioConstraints | None = None,
        encoder: str = "transformer",
        encoder_config: EncoderConfig | None = None,
        hidden_dimensions: tuple[int, ...] = (256, 128),
        head_dimension: int = 128,
        dropout: float = 0.1,
        residual_from_previous: bool = True,
    ) -> None:
        if encoder not in ENCODER_KINDS:
            raise ValueError(f"unknown encoder {encoder!r}; expected one of {ENCODER_KINDS}.")
        super().__init__(
            state_dimension=spec.dimension(n_assets, n_regimes),
            n_assets=n_assets,
            constraints=constraints if constraints is not None else PortfolioConstraints(),
        )
        self.spec = spec
        self.n_regimes = int(n_regimes)
        self.encoder_kind = encoder
        self.residual_from_previous = bool(residual_from_previous)

        self.encoder: TransformerStateEncoder | MLPStateEncoder
        if encoder == "transformer":
            self.encoder = TransformerStateEncoder(
                spec, n_assets=n_assets, n_regimes=n_regimes, config=encoder_config
            )
        else:
            self.encoder = MLPStateEncoder(
                spec,
                n_assets=n_assets,
                n_regimes=n_regimes,
                hidden_dimensions=hidden_dimensions,
                dropout=dropout,
            )

        # Start at the identity map "hold w_{t-1}" (or at equal weight when the
        # residual is off), so the first epoch improves on a sane portfolio
        # instead of on noise.
        read_out = nn.Linear(head_dimension, n_assets)
        nn.init.zeros_(read_out.weight)
        nn.init.zeros_(read_out.bias)
        self.head = nn.Sequential(
            nn.Linear(self.encoder.output_dimension, head_dimension),
            nn.GELU(),
            read_out,
        )

        slices = spec.block_slices(n_assets, n_regimes)
        if self.residual_from_previous and "previous_weights" not in slices:
            raise ValueError(
                "residual_from_previous requires a StateSpec that includes w_{t-1}."
            )
        self._previous_slice = slices.get("previous_weights")

    def scores(self, states: torch.Tensor) -> torch.Tensor:
        """Return unprojected allocation scores, shape ``(batch, n_assets)``."""
        encoded = self.encoder(self.standardiser(states))
        adjustment = self.head(encoded)
        if not self.residual_from_previous:
            return adjustment
        assert self._previous_slice is not None
        return states[:, self._previous_slice] + adjustment

    def allocate(self, states: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        if kwargs:
            raise TypeError(f"unexpected arguments: {sorted(kwargs)}")
        lower, upper = self.bounds
        return project_box_simplex(self.scores(states), lower=lower, upper=upper)

    def bc_loss(
        self, states: torch.Tensor, actions: torch.Tensor, loss: AllocationLoss
    ) -> torch.Tensor:
        return loss(self.allocate(states), actions)

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "deterministic",
            "encoder": self.encoder.describe(),
            "residual_from_previous": self.residual_from_previous,
            "head": "projected linear read-out (Equation 8)",
            **self.base_description(),
        }
