"""A stochastic actor-critic wrapper around a deterministic allocation policy.

PPO needs a density: its clipped surrogate is built on the likelihood ratio
``pi_theta(a|s) / pi_theta_old(a|s)``, and every policy built so far is
deterministic.

The distribution is placed in the **pre-projection** space. The policy's
``scores(s)`` — the raw output before Equation (8) projects it onto the feasible
set — becomes the mean of a diagonal Gaussian, an action is sampled there, and
the projection is treated as part of the environment:

    z ~ N(scores(s), diag(sigma^2)),   w = Pi(z)

This placement is the whole design, and the alternatives are worse for concrete
reasons. Sampling on the simplex and then clipping gives a density that is wrong
wherever the clip binds, which under a 0.30 cap is most of the time and
precisely where the expert is most distinctive. A Dirichlet has a clean density
on the simplex but cannot represent the expert's exact zeros or the position
cap, and cannot be warm-started from the existing heads. Putting the Gaussian
before the projection keeps ``log pi(z|s)`` exact, keeps every executed
allocation feasible by construction, and warm-starts from behavioural cloning,
DAgger, or GAIL weights with no adapter at all.

The value head is a separate MLP on the standardised state rather than a second
head on the shared encoder. Sharing would couple the value loss into the
encoder, and the actor here starts from a carefully trained imitation solution
that a randomly initialised critic's gradients would disturb.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any, cast

import torch
import torch.nn as nn

from ..policy.base import AllocationPolicy
from ..policy.simplex import project_box_simplex

#: Bounds on the learned log standard deviation. The lower bound stops the
#: policy collapsing to determinism, which makes the ratio degenerate; the upper
#: bound stops it exploring so widely that every sample projects to the same
#: corner and the gradient vanishes.
LOG_STD_MIN = -6.0
LOG_STD_MAX = 0.0


@dataclass(frozen=True)
class ActorCriticConfig:
    """Exploration and value-head settings."""

    log_std_init: float = -3.0
    value_hidden_dimensions: tuple[int, ...] = (128, 64)
    learn_log_std: bool = True

    def __post_init__(self) -> None:
        if not LOG_STD_MIN <= self.log_std_init <= LOG_STD_MAX:
            raise ValueError(
                f"log_std_init must lie in [{LOG_STD_MIN}, {LOG_STD_MAX}]."
            )
        if not self.value_hidden_dimensions:
            raise ValueError("value_hidden_dimensions must be non-empty.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {
            "value_hidden_dimensions": list(self.value_hidden_dimensions)
        }


class GaussianActorCritic(nn.Module):
    """Wraps an allocation policy with Gaussian exploration and a value head."""

    def __init__(
        self,
        policy: AllocationPolicy,
        *,
        config: ActorCriticConfig | None = None,
    ) -> None:
        super().__init__()
        if not hasattr(policy, "scores"):
            raise TypeError(
                "the policy must expose scores(), the pre-projection output the "
                "Gaussian is placed on."
            )
        self.policy = policy
        self.config = config if config is not None else ActorCriticConfig()

        self.log_std = nn.Parameter(
            torch.full((policy.n_assets,), float(self.config.log_std_init)),
            requires_grad=self.config.learn_log_std,
        )

        layers: list[nn.Module] = []
        width = policy.state_dimension
        for hidden in self.config.value_hidden_dimensions:
            layers.append(nn.Linear(width, hidden))
            layers.append(nn.Tanh())
            width = hidden
        read_out = nn.Linear(width, 1)
        nn.init.zeros_(read_out.weight)
        nn.init.zeros_(read_out.bias)
        layers.append(read_out)
        self.value_network = nn.Sequential(*layers)

    # ------------------------------------------------------------------
    # Distribution
    # ------------------------------------------------------------------
    @property
    def standard_deviation(self) -> torch.Tensor:
        return self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX).exp()

    def mean(self, states: torch.Tensor) -> torch.Tensor:
        """Return the pre-projection mean, i.e. the deterministic policy output.

        ``scores`` is not on the :class:`AllocationPolicy` base class — only the
        deterministic heads expose it — so the lookup is cast. The constructor
        rejects a policy without it, so the cast cannot fail at runtime.
        """
        scores = cast(Callable[[torch.Tensor], torch.Tensor], self.policy.scores)
        return scores(states)

    def value(self, states: torch.Tensor) -> torch.Tensor:
        standardised = self.policy.standardiser.forward(states)
        return self.value_network(standardised).squeeze(-1)

    def log_probability(
        self, states: torch.Tensor, latent: torch.Tensor
    ) -> torch.Tensor:
        """Return ``log pi(z|s)``, summed over assets.

        Exact, because ``z`` is the quantity actually sampled. The projection
        that turns ``z`` into an allocation belongs to the environment.
        """
        mean = self.mean(states)
        scale = self.standard_deviation
        variance = scale**2
        elementwise = (
            -0.5 * ((latent - mean) ** 2) / variance
            - scale.log()
            - 0.5 * math.log(2.0 * math.pi)
        )
        return elementwise.sum(dim=-1)

    def entropy(self) -> torch.Tensor:
        """Differential entropy of the diagonal Gaussian, summed over assets."""
        scale = self.standard_deviation
        return (0.5 * math.log(2.0 * math.pi * math.e) + scale.log()).sum()

    def project(self, latent: torch.Tensor) -> torch.Tensor:
        """Apply Equation (8) to a latent sample."""
        lower, upper = self.policy.bounds
        return project_box_simplex(latent, lower=lower, upper=upper)

    # ------------------------------------------------------------------
    # Acting
    # ------------------------------------------------------------------
    def sample(
        self,
        states: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
        deterministic: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Draw one latent action and return everything a PPO buffer needs."""
        mean = self.mean(states)
        if deterministic:
            latent = mean
        else:
            noise = torch.randn(
                mean.shape, generator=generator, device=mean.device, dtype=mean.dtype
            )
            latent = mean + noise * self.standard_deviation
        return {
            "latent": latent,
            "action": self.project(latent),
            "log_probability": self.log_probability(states, latent),
            "value": self.value(states),
        }

    def evaluate(
        self, states: torch.Tensor, latent: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(log_prob, entropy, value)`` for stored latents."""
        return (
            self.log_probability(states, latent),
            self.entropy(),
            self.value(states),
        )

    def describe(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "standard_deviation": [float(v) for v in self.standard_deviation.detach()],
            "value_parameters": int(
                sum(p.numel() for p in self.value_network.parameters())
            ),
            "policy_parameters": self.policy.n_parameters,
            "distribution": "diagonal Gaussian on the pre-projection scores",
        }
