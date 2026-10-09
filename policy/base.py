"""Shared interface for allocation policies.

Every policy maps an Equation (6) state to a feasible allocation. Training,
evaluation, the environment, and the acceptance checks all talk to policies
through this interface, so the deterministic transformer, the
transformer-diffusion model, and the plain MLP floor are interchangeable.

Two methods carry the design decisions.

``bc_loss`` is what behavioural cloning minimises. It is a policy method rather
than a free function because the two model families do not share an objective:
a deterministic policy regresses its output onto ``w*_t``, whereas a diffusion
policy trains a denoiser and never produces an allocation during a training
step. Putting the objective behind one name keeps the trainer polymorphic
instead of branching on model type.

``act`` is the deployment path. It runs under ``torch.no_grad`` in eval mode and
returns NumPy, which is what the Gymnasium environment and the latency
comparison of research question 5 need.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from ..expert.constraints import PortfolioConstraints
from .standardise import StateStandardiser

#: Signature of a behavioural-cloning allocation loss.
AllocationLoss = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class AllocationPolicy(nn.Module, ABC):
    """A state-to-allocation map that always returns feasible weights."""

    def __init__(
        self,
        *,
        state_dimension: int,
        n_assets: int,
        constraints: PortfolioConstraints,
    ) -> None:
        super().__init__()
        constraints.validate_for(n_assets)
        self.state_dimension = int(state_dimension)
        self.n_assets = int(n_assets)
        self.constraints = constraints
        self.standardiser = StateStandardiser(self.state_dimension)

    # ------------------------------------------------------------------
    # Required by subclasses
    # ------------------------------------------------------------------
    @abstractmethod
    def allocate(self, states: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Return feasible allocations for a batch of raw states."""

    @abstractmethod
    def bc_loss(self, states: torch.Tensor, actions: torch.Tensor, loss: AllocationLoss):
        """Return the scalar behavioural-cloning objective for one batch."""

    @abstractmethod
    def describe(self) -> dict[str, Any]:
        """Return a JSON-serialisable description for experiment metadata."""

    # ------------------------------------------------------------------
    # Optional training-data calibration
    # ------------------------------------------------------------------
    def calibrate(self, states: torch.Tensor, actions: torch.Tensor) -> None:
        """Set any non-learned scale from training data. Default: nothing to do.

        Called by the trainer immediately after the standardiser is fitted, with
        training samples only. It exists for the diffusion policy, whose noise
        schedule assumes a target of roughly unit scale and which therefore
        needs to know how large the expert's adjustments actually are.
        """
        del states, actions

    # ------------------------------------------------------------------
    # Shared behaviour
    # ------------------------------------------------------------------
    def forward(self, states: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return self.allocate(states, **kwargs)

    @property
    def bounds(self) -> tuple[float, float]:
        """Box bounds ``(l, u)`` the projection must respect."""
        return self.constraints.effective_min_weight, self.constraints.max_weight

    @property
    def n_parameters(self) -> int:
        return int(sum(parameter.numel() for parameter in self.parameters()))

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @torch.no_grad()
    def act(self, states: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Return allocations for raw NumPy states, one row per state.

        A single ``(state_dimension,)`` vector is accepted and answered with a
        single ``(n_assets,)`` allocation.
        """
        values = np.asarray(states, dtype=np.float64)
        single = values.ndim == 1
        if single:
            values = values[np.newaxis, :]
        if values.ndim != 2 or values.shape[1] != self.state_dimension:
            raise ValueError(
                f"states must have shape (batch, {self.state_dimension}) or "
                f"({self.state_dimension},); got {np.shape(states)}."
            )

        was_training = self.training
        self.eval()
        try:
            batch = torch.as_tensor(values, dtype=torch.float32, device=self.device)
            weights = self.allocate(batch, **kwargs).cpu().numpy().astype(np.float64)
        finally:
            self.train(was_training)
        return weights[0] if single else weights

    def base_description(self) -> dict[str, Any]:
        return {
            "state_dimension": self.state_dimension,
            "n_assets": self.n_assets,
            "n_parameters": self.n_parameters,
            "constraints": self.constraints.to_dict(),
            "standardiser": self.standardiser.describe(),
        }
