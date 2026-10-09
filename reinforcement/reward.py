"""The exposé's reinforcement-learning reward, Equations (10) and (11).

The realised net portfolio return charges the transaction cost exactly once,

    R_tilde_{t+1}(w_t) = w_t' r_{t+1} - c ||w_t - w_{t-1}||_1,        (10)

and the PPO reward adds an estimated-risk term and an expert-deviation penalty,

    rho_t = R_tilde_{t+1}(w_t)
            - (gamma/2) w_t' Sigma_hat_{t+1|t} w_t
            - lambda_E ||w_t - w*_t||_2^2.                            (11)

Turnover is **not** a separate term. The exposé is explicit that "transaction
costs already capture the direct cost of changing portfolio weights", so adding
turnover again would price the same friction twice. It is reported as an
evaluation measure instead.

This module computes the reward over the batched rollout tape rather than by
stepping :class:`~src.reinforcement.environment.PortfolioEnv`, because PPO needs
many rollouts and the tape steps every path at once. That is a duplicated
implementation of the same equation, which is a risk, so an acceptance check
asserts the two agree exactly on a replayed trajectory.

Every term is differentiable in ``w``, which is what lets the exact-gradient
baseline optimise the identical objective by backpropagation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch

from ..imitation.differentiable import RolloutTape


@dataclass(frozen=True)
class RewardConfig:
    """Coefficients of Equation (11)."""

    risk_aversion: float
    transaction_cost: float
    expert_deviation_penalty: float = 0.0

    def __post_init__(self) -> None:
        for name in ("risk_aversion", "transaction_cost", "expert_deviation_penalty"):
            value = getattr(self, name)
            if value < 0.0:
                raise ValueError(f"{name} must be nonnegative.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def net_returns(
    actions: torch.Tensor,
    anchors: torch.Tensor,
    realised: torch.Tensor,
    transaction_cost: float,
) -> torch.Tensor:
    """Equation (10): realised return with the cost charged once."""
    gross = (actions * realised).sum(dim=-1)
    turnover = (actions - anchors).abs().sum(dim=-1)
    return gross - transaction_cost * turnover


def rewards(
    actions: torch.Tensor,
    anchors: torch.Tensor,
    realised: torch.Tensor,
    covariances: torch.Tensor,
    expert_actions: torch.Tensor,
    config: RewardConfig,
) -> torch.Tensor:
    """Equation (11), elementwise over a batch of decisions."""
    net = net_returns(actions, anchors, realised, config.transaction_cost)
    risk = torch.einsum("...i,...ij,...j->...", actions, covariances, actions)
    deviation = ((actions - expert_actions) ** 2).sum(dim=-1)
    return (
        net
        - 0.5 * config.risk_aversion * risk
        - config.expert_deviation_penalty * deviation
    )


def tape_rewards(
    tape: RolloutTape,
    actions: torch.Tensor,
    anchors: torch.Tensor,
    config: RewardConfig,
    *,
    start: int = 0,
) -> torch.Tensor:
    """Equation (11) for a rolled segment of a tape."""
    if tape.forecast_covariances is None:
        raise ValueError(
            "the tape carries no forecast covariances; build it with "
            "with_covariances=True so the Equation (11) risk term can be computed."
        )
    stop = start + actions.shape[0]
    return rewards(
        actions,
        anchors,
        tape.realised_returns[start:stop],
        tape.forecast_covariances[start:stop],
        tape.expert_actions[start:stop],
        config,
    )


def certainty_equivalent(net: torch.Tensor, risk_aversion: float) -> float:
    """Equation (16): ``CE = E[R] - (gamma/2) Var(R)``.

    Computed on *realised net* returns, which is what Section 5.3 reports, and
    is a different quantity from the reward's estimated-risk term: this one uses
    the realised variance of the strategy rather than the expert's forecast.
    """
    values = net.reshape(-1)
    if values.numel() < 2:
        return float(values.mean())
    return float(values.mean() - 0.5 * risk_aversion * values.var(unbiased=True))


def utility_loss(
    expert_net: torch.Tensor, policy_net: torch.Tensor, risk_aversion: float
) -> float:
    """Equation (17): ``CE(pi_E) - CE(pi)``. Positive means the policy is worse."""
    return certainty_equivalent(expert_net, risk_aversion) - certainty_equivalent(
        policy_net, risk_aversion
    )
