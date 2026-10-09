"""Wasserstein critic over occupancy measures.

Stage 3 of the exposé's imitation pipeline matches the expert's *occupancy
distribution* rather than its individual actions. Equation (9) writes the
Wasserstein-1 distance between the expert occupancy ``d_E`` and the policy
occupancy ``d_pi`` in its Kantorovich-Rubinstein dual form,

    W1(d_E, d_pi) = sup_{||D||_Lip <= 1} ( E_{(s,a)~d_E}[D(s,a)]
                                           - E_{(s,a)~d_pi}[D(s,a)] ),

so the critic ``D`` is a scalar function of a state-action pair, constrained to
be 1-Lipschitz. Maximising that difference estimates ``W1``; the policy then
minimises it.

Two Lipschitz enforcement methods are implemented, both cited in the exposé:

``gradient_penalty``
    Gulrajani et al. (2017): penalise ``(||grad_x D(x)|| - 1)^2`` at points
    interpolated between the two distributions. The default.
``weight_clipping``
    Arjovsky et al. (2017): clamp every critic weight into ``[-c, c]`` after
    each update. Kept because comparing the two is the canonical stability
    experiment in this literature, and "monitor training stability" is what the
    work package asks for.

One feature of this problem makes the critic's job unusually well posed. At a
given date the expert and the policy see the *same* market blocks — the return
window, the moment forecasts, and the filtered regime probabilities are
exogenous. So the two occupancy measures differ only in ``w_{t-1}`` and in the
action taken, and ``W1`` is effectively a distance over trajectories of
allocations conditional on identical market conditions, rather than over the
full high-dimensional state. That is a far better-conditioned problem than
generic GAIL faces, and it is why interpolating between an expert pair and a
policy pair at the *same* date is meaningful here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn

from ..policy.standardise import StateStandardiser

LIPSCHITZ_METHODS = ("gradient_penalty", "weight_clipping")


@dataclass(frozen=True)
class CriticConfig:
    """Architecture and Lipschitz settings for the Wasserstein critic."""

    hidden_dimensions: tuple[int, ...] = (256, 128)
    lipschitz: str = "gradient_penalty"
    gradient_penalty_weight: float = 10.0
    clip_value: float = 0.01
    learning_rate: float = 1e-4
    steps_per_round: int = 5
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.lipschitz not in LIPSCHITZ_METHODS:
            raise ValueError(
                f"unknown lipschitz method {self.lipschitz!r}; "
                f"expected one of {LIPSCHITZ_METHODS}."
            )
        if not self.hidden_dimensions:
            raise ValueError("hidden_dimensions must be non-empty.")
        for name in ("gradient_penalty_weight", "clip_value", "learning_rate"):
            value = getattr(self, name)
            if value <= 0.0:
                raise ValueError(f"{name} must be positive.")
        if not isinstance(self.steps_per_round, int) or self.steps_per_round < 1:
            raise ValueError("steps_per_round must be a positive integer.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1).")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {"hidden_dimensions": list(self.hidden_dimensions)}


class WassersteinCritic(nn.Module):
    """A 1-Lipschitz scalar score on state-action pairs.

    The state is standardised with the *policy's* fitted statistics, frozen and
    training-only, so the critic and the policy see identical inputs and the
    critic cannot key on a scaling the policy never sees.

    There is no batch normalisation anywhere: it makes a sample's score depend
    on the batch it landed in, which breaks both the Lipschitz argument and the
    reproducibility the rest of this project maintains.
    """

    def __init__(
        self,
        *,
        state_dimension: int,
        n_assets: int,
        standardiser: StateStandardiser,
        config: CriticConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config if config is not None else CriticConfig()
        self.state_dimension = int(state_dimension)
        self.n_assets = int(n_assets)
        self.standardiser = standardiser

        layers: list[nn.Module] = []
        width = self.state_dimension + self.n_assets
        for hidden in self.config.hidden_dimensions:
            layers.append(nn.Linear(width, hidden))
            layers.append(nn.LeakyReLU(0.2))
            if self.config.dropout > 0.0:
                layers.append(nn.Dropout(self.config.dropout))
            width = hidden
        layers.append(nn.Linear(width, 1))
        self.network = nn.Sequential(*layers)

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------
    def inputs(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Return the standardised ``(s, a)`` pair the critic actually scores."""
        if states.shape[-1] != self.state_dimension:
            raise ValueError(
                f"states must have last dimension {self.state_dimension}; "
                f"got {states.shape[-1]}."
            )
        if actions.shape[-1] != self.n_assets:
            raise ValueError(
                f"actions must have last dimension {self.n_assets}; "
                f"got {actions.shape[-1]}."
            )
        return torch.cat([self.standardiser(states), actions], dim=-1)

    def score(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Return ``D(s, a)`` with shape ``(batch,)``."""
        return self.network(self.inputs(states, actions)).squeeze(-1)

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.score(states, actions)

    def score_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
        """Score an already-assembled critic input, used by the penalty."""
        return self.network(inputs).squeeze(-1)

    # ------------------------------------------------------------------
    # Objective
    # ------------------------------------------------------------------
    def wasserstein_estimate(
        self,
        expert_states: torch.Tensor,
        expert_actions: torch.Tensor,
        policy_states: torch.Tensor,
        policy_actions: torch.Tensor,
    ) -> torch.Tensor:
        """Return the dual estimate ``E_dE[D] - E_dpi[D]``.

        This is a *lower bound* on ``W1`` for any 1-Lipschitz critic, and
        approaches it as the critic approaches the supremum in Equation (9). It
        is reported every round as the milestone's primary stability signal.
        """
        return self.score(expert_states, expert_actions).mean() - self.score(
            policy_states, policy_actions
        ).mean()

    def gradient_penalty(
        self,
        expert_states: torch.Tensor,
        expert_actions: torch.Tensor,
        policy_states: torch.Tensor,
        policy_actions: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Return Gulrajani's ``(||grad D|| - 1)^2`` penalty.

        Interpolation is between an expert pair and a policy pair drawn at the
        *same* decision date, which is what makes the interpolated point a
        plausible state-action pair rather than a blend of unrelated market
        conditions.
        """
        expert = self.inputs(expert_states, expert_actions)
        policy = self.inputs(policy_states, policy_actions)
        if expert.shape != policy.shape:
            raise ValueError("expert and policy batches must have the same shape.")

        weights = torch.rand(
            expert.shape[0], 1, device=expert.device, generator=generator
        )
        interpolated = (weights * expert + (1.0 - weights) * policy).requires_grad_(True)
        scores = self.score_inputs(interpolated)
        gradients = torch.autograd.grad(
            outputs=scores.sum(),
            inputs=interpolated,
            create_graph=True,
        )[0]
        norms = gradients.norm(2, dim=-1)
        return ((norms - 1.0) ** 2).mean()

    def critic_loss(
        self,
        expert_states: torch.Tensor,
        expert_actions: torch.Tensor,
        policy_states: torch.Tensor,
        policy_actions: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Return the critic's minimisation objective and its components.

        The critic maximises the dual gap, so it minimises its negative, plus
        the penalty when gradient penalty is the enforcement method.
        """
        estimate = self.wasserstein_estimate(
            expert_states, expert_actions, policy_states, policy_actions
        )
        loss = -estimate
        penalty_value = 0.0
        if self.config.lipschitz == "gradient_penalty":
            penalty = self.gradient_penalty(
                expert_states,
                expert_actions,
                policy_states,
                policy_actions,
                generator=generator,
            )
            loss = loss + self.config.gradient_penalty_weight * penalty
            penalty_value = float(penalty.detach())

        return loss, {
            "wasserstein_estimate": float(estimate.detach()),
            "gradient_penalty": penalty_value,
            "critic_loss": float(loss.detach()),
        }

    def enforce_constraint(self) -> None:
        """Apply weight clipping, if that is the configured method."""
        if self.config.lipschitz != "weight_clipping":
            return
        limit = self.config.clip_value
        with torch.no_grad():
            for parameter in self.network.parameters():
                parameter.clamp_(-limit, limit)

    @torch.no_grad()
    def measured_lipschitz_constant(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> float:
        """Return the largest input-gradient norm seen on a batch.

        Reported rather than assumed: a gradient penalty is a soft constraint,
        so the realised constant is an empirical quantity and is exactly what
        the stability panel should show.
        """
        with torch.enable_grad():
            inputs = self.inputs(states, actions).detach().requires_grad_(True)
            scores = self.score_inputs(inputs)
            gradients = torch.autograd.grad(scores.sum(), inputs)[0]
        return float(gradients.norm(2, dim=-1).max())

    @property
    def n_parameters(self) -> int:
        return int(sum(parameter.numel() for parameter in self.parameters()))

    def describe(self) -> dict[str, Any]:
        """Return a JSON-serialisable description for experiment metadata."""
        return {
            "state_dimension": self.state_dimension,
            "n_assets": self.n_assets,
            "n_parameters": self.n_parameters,
            "config": self.config.to_dict(),
            "objective": "W1(d_E, d_pi) = sup_{||D||_Lip<=1} E_dE[D] - E_dpi[D]",
        }
