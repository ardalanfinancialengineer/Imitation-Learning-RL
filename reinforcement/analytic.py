"""Exact-gradient optimisation of the Equation (11) reward.

PPO's clipped surrogate is a variance-reduced *estimator* of a gradient that,
on this problem, can be computed exactly. Weeks 26-29 established why: only
``w_{t-1}`` carries action dependence, so the closed-loop rollout is a
differentiable recurrence, and the Equation (11) reward is differentiable in the
allocation. Backpropagating the summed reward through a truncated rollout
therefore gives ``d(sum rho_t)/dtheta`` without sampling.

This is **not** a replacement for PPO — it is a reported upper baseline. Its
purpose is diagnostic: it bounds what the reward itself can deliver, so a weak
PPO result can be attributed to the estimator rather than to the objective. Very
few reinforcement-learning settings admit that comparison, and this one does for
free.

Two honest caveats. The exact gradient optimises the deterministic policy with
no exploration, so it can settle in a local optimum a stochastic search would
escape. And it uses the simulator's realised returns directly, which is exactly
the differentiability PPO deliberately does not assume — so it is an upper
baseline for *this* simulator rather than a method that transfers to a market.
"""

from __future__ import annotations

import time
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import torch

from ..dataset.demonstrations import DemonstrationDataset
from ..imitation.bc import seed_everything
from ..imitation.differentiable import RolloutTape, build_tape, rollout_segment
from ..policy.base import AllocationPolicy
from ..utils.device import resolve_device
from .reward import RewardConfig, certainty_equivalent, net_returns, tape_rewards


@dataclass(frozen=True)
class AnalyticConfig:
    """Settings for the exact-gradient baseline."""

    iterations: int = 60
    segment_length: int = 32
    learning_rate: float = 1e-4
    gradient_clip: float = 1.0
    evaluate_every: int = 5
    seed: int = 0
    device: str = "cpu"

    def __post_init__(self) -> None:
        for name in ("iterations", "segment_length", "evaluate_every"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")
        if self.gradient_clip < 0.0:
            raise ValueError("gradient_clip must be nonnegative.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AnalyticHistory:
    """Outcome of one exact-gradient run."""

    records: list[dict[str, float]] = field(default_factory=list)
    total_seconds: float = 0.0
    best_iteration: int = 0
    best_certainty_equivalent: float = -float("inf")

    def series(self, name: str) -> np.ndarray:
        return np.array([record[name] for record in self.records], dtype=float)

    def to_dict(self) -> dict[str, Any]:
        return {
            "iterations_run": len(self.records),
            "total_seconds": round(self.total_seconds, 2),
            "best_iteration": self.best_iteration,
            "best_certainty_equivalent": self.best_certainty_equivalent,
            "records": list(self.records),
        }


class AnalyticRewardTrainer:
    """Maximise the Equation (11) reward by backpropagation through the rollout."""

    def __init__(
        self,
        policy: AllocationPolicy,
        reward: RewardConfig,
        *,
        config: AnalyticConfig | None = None,
    ) -> None:
        self.policy = policy
        self.reward = reward
        self.config = config if config is not None else AnalyticConfig()
        self.device = resolve_device(self.config.device)
        self.policy.to(self.device)

    def fit(
        self,
        train_dataset: DemonstrationDataset,
        validation_dataset: DemonstrationDataset,
    ) -> AnalyticHistory:
        seed_everything(self.config.seed)
        tape = build_tape(train_dataset, device=self.device, with_covariances=True)
        validation_tape = build_tape(
            validation_dataset, device=self.device, with_covariances=True
        )

        optimiser = torch.optim.AdamW(
            self.policy.parameters(), lr=self.config.learning_rate
        )
        rng = np.random.default_rng(self.config.seed)
        history = AnalyticHistory()
        best_state = deepcopy(self.policy.state_dict())
        started = time.perf_counter()

        for iteration in range(1, self.config.iterations + 1):
            length = min(self.config.segment_length, tape.n_steps)
            start = int(rng.integers(0, max(1, tape.n_steps - length + 1)))
            anchor = tape.expert_anchors[start]

            optimiser.zero_grad(set_to_none=True)
            segment = rollout_segment(self.policy, tape, anchor, start, length)
            rewards = tape_rewards(
                tape, segment.actions, segment.anchors, self.reward, start=start
            )
            # Maximise the reward, so minimise its negative mean.
            loss = -rewards.mean()
            loss.backward()
            norm = float(
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.config.gradient_clip
                )
            )
            optimiser.step()

            evaluate = (
                iteration % self.config.evaluate_every == 0
                or iteration == self.config.iterations
            )
            metrics = self.evaluate(validation_tape) if evaluate else {}
            history.records.append(
                {
                    "iteration": float(iteration),
                    "reward": float(-loss.detach()),
                    "gradient_norm": norm,
                    "certainty_equivalent": metrics.get(
                        "certainty_equivalent", float("nan")
                    ),
                    "mean_turnover": metrics.get("mean_turnover", float("nan")),
                }
            )
            if evaluate and metrics["certainty_equivalent"] > (
                history.best_certainty_equivalent
            ):
                history.best_certainty_equivalent = metrics["certainty_equivalent"]
                history.best_iteration = iteration
                best_state = deepcopy(self.policy.state_dict())

        self.policy.load_state_dict(best_state)
        history.total_seconds = time.perf_counter() - started
        return history

    @torch.no_grad()
    def evaluate(self, tape: RolloutTape) -> dict[str, float]:
        """Score the policy on a tape with the same measures PPO reports."""
        segment = rollout_segment(
            self.policy, tape, tape.initial_anchor, 0, tape.n_steps
        )
        net = net_returns(
            segment.actions,
            segment.anchors,
            tape.realised_returns,
            self.reward.transaction_cost,
        )
        violations = self.policy.constraints.violations(
            segment.actions.reshape(-1, tape.n_assets).cpu().numpy().astype(np.float64)
        )
        return {
            "mean_net_return": float(net.mean()),
            "annualised_net_return": float(net.mean()) * 252.0,
            "certainty_equivalent": certainty_equivalent(net, self.reward.risk_aversion),
            "mean_turnover": float(segment.turnover.mean()),
            "feasibility_violation": max(violations.values(), default=0.0),
        }

    def describe(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "reward": self.reward.to_dict(),
            "estimator": (
                "exact gradient by backpropagation through the differentiable "
                "rollout; reported as an upper baseline for PPO, not as a "
                "competing method"
            ),
        }
