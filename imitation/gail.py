"""Stage 3 of the imitation pipeline: Wasserstein-regularised GAIL.

Behavioural cloning matches the expert's *action* at states the expert visited.
DAgger fixes the states. GAIL changes the target: it matches the expert's
**occupancy measure** — the joint distribution of states and actions a policy
induces — rather than its per-decision answers. Ho and Ermon (2016) show that is
the right object when the goal is to reproduce behaviour rather than to predict
labels, and the exposé extends it with a Wasserstein critic (Equation 9)
following Arjovsky et al. (2017) and Gulrajani et al. (2017).

The training loop is the usual two-player alternation:

1. roll the current policy out to collect the policy occupancy ``d_pi``;
2. update the critic to maximise ``E_dE[D] - E_dpi[D]``, which estimates
   ``W1(d_E, d_pi)`` from below;
3. update the policy to minimise that same quantity.

Step 3 is where this implementation departs from the textbook, and the
departure is in the *estimator* rather than the objective. Standard GAIL cannot
differentiate through the environment, so it treats ``D`` as a reward and uses a
policy-gradient method. Here the environment is differentiable — see
:mod:`src.imitation.differentiable` — so the policy gradient is obtained by
backpropagation through a truncated rollout. It is exact rather than sampled,
which removes the dominant source of GAIL's instability and is why this
milestone can study stability without first implementing PPO.

The acceptance suite verifies the claim rather than asserting it: the rollout
gradient is checked against central finite differences in float64 and agrees to
about ``1e-9``.

Model selection is on validation **closed-loop tracking error**, not on the
Wasserstein estimate. That is a deliberate compromise. ``W1`` is GAIL's own
objective but it is measured against a moving critic and is not comparable
across BC, DAgger, and GAIL, whereas tracking error is defined identically for
all three and is what the work package asks to compare. The estimate is
recorded every iteration so the tension stays visible.
"""

from __future__ import annotations

import time
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import torch

from ..dataset.demonstrations import DemonstrationDataset
from ..policy.base import AllocationPolicy
from ..utils.device import resolve_device
from .bc import seed_everything
from .critic import CriticConfig, WassersteinCritic
from .differentiable import (
    RolloutTape,
    build_tape,
    realised_net_returns,
    rollout_segment,
    rollout_tape,
)
from .rollout import ExpertOracle, answer_rollouts, rollout_metrics, rollout_policy


@dataclass(frozen=True)
class GAILConfig:
    """Settings for the adversarial occupancy-matching loop."""

    iterations: int = 40
    segment_length: int = 32
    policy_steps: int = 1
    policy_learning_rate: float = 1e-4
    batch_size: int = 256
    bc_anchor_weight: float = 0.0
    gradient_clip: float = 1.0
    evaluate_every: int = 5
    seed: int = 0
    device: str = "cpu"

    def __post_init__(self) -> None:
        for name in ("iterations", "segment_length", "policy_steps", "batch_size",
                     "evaluate_every"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if self.policy_learning_rate <= 0.0:
            raise ValueError("policy_learning_rate must be positive.")
        for name in ("bc_anchor_weight", "gradient_clip"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be nonnegative.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GAILRecord:
    """One adversarial iteration, as reported in the stability table."""

    iteration: int
    wasserstein_estimate: float
    critic_loss: float
    gradient_penalty: float
    policy_loss: float
    critic_gradient_norm: float
    policy_gradient_norm: float
    measured_lipschitz: float
    closed_loop_tracking_error: float
    mean_turnover: float
    seconds: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GAILHistory:
    """Outcome of one GAIL run, with the stability signals the milestone asks for."""

    records: list[GAILRecord] = field(default_factory=list)
    total_seconds: float = 0.0
    best_iteration: int = 0
    best_closed_loop_error: float = float("inf")
    diverged: bool = False

    def series(self, name: str) -> np.ndarray:
        """Return one recorded quantity across iterations, as an array."""
        return np.array([getattr(record, name) for record in self.records], dtype=float)

    def stability(self) -> dict[str, float]:
        """Summarise how well behaved the adversarial game was.

        ``wasserstein_volatility`` is the standard deviation of the estimate
        over the second half of training, relative to its mean level. A stable
        game settles; an unstable one oscillates or runs away, and this
        distinguishes the two without needing to eyeball the curve.
        """
        estimates = self.series("wasserstein_estimate")
        if estimates.size < 4:
            return {"wasserstein_volatility": 0.0, "wasserstein_drift": 0.0}
        tail = estimates[estimates.size // 2 :]
        level = float(np.mean(np.abs(tail))) or 1.0
        first_half = float(np.mean(estimates[: estimates.size // 2]))
        return {
            "wasserstein_volatility": float(np.std(tail) / level),
            "wasserstein_drift": float(np.mean(tail) - first_half),
            "wasserstein_final": float(estimates[-1]),
            "wasserstein_min": float(estimates.min()),
            "max_measured_lipschitz": float(self.series("measured_lipschitz").max()),
            "max_policy_gradient_norm": float(self.series("policy_gradient_norm").max()),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "iterations_run": len(self.records),
            "total_seconds": round(self.total_seconds, 2),
            "best_iteration": self.best_iteration,
            "best_closed_loop_tracking_error": self.best_closed_loop_error,
            "diverged": self.diverged,
            "stability": self.stability(),
            "records": [record.to_dict() for record in self.records],
        }


class GAILTrainer:
    """Match the expert's occupancy measure with a Wasserstein critic."""

    def __init__(
        self,
        policy: AllocationPolicy,
        *,
        config: GAILConfig | None = None,
        critic_config: CriticConfig | None = None,
    ) -> None:
        self.policy = policy
        self.config = config if config is not None else GAILConfig()
        self.critic_config = critic_config if critic_config is not None else CriticConfig()
        self.device = resolve_device(self.config.device)
        self.policy.to(self.device)
        self.critic = WassersteinCritic(
            state_dimension=policy.state_dimension,
            n_assets=policy.n_assets,
            standardiser=policy.standardiser,
            config=self.critic_config,
        ).to(self.device)

    def fit(
        self,
        train_dataset: DemonstrationDataset,
        validation_dataset: DemonstrationDataset,
        *,
        validation_oracle: ExpertOracle | None = None,
        max_steps: int | None = None,
    ) -> GAILHistory:
        """Run the adversarial loop.

        The policy is expected to be already behaviour-cloned. Starting the
        critic against an untrained policy would have it separating two
        distributions that differ for uninteresting reasons.
        """
        seed_everything(self.config.seed)
        tape = build_tape(train_dataset, device=self.device, max_steps=max_steps)
        oracle = (
            validation_oracle
            if validation_oracle is not None
            else ExpertOracle(validation_dataset)
        )

        policy_optimiser = torch.optim.AdamW(
            self.policy.parameters(), lr=self.config.policy_learning_rate
        )
        critic_optimiser = torch.optim.AdamW(
            self.critic.parameters(), lr=self.critic_config.learning_rate
        )
        generator = torch.Generator(device="cpu").manual_seed(self.config.seed)
        rng = np.random.default_rng(self.config.seed)

        history = GAILHistory()
        best_state = deepcopy(self.policy.state_dict())
        started = time.perf_counter()

        baseline = self._evaluate(validation_dataset, oracle)
        history.best_closed_loop_error = baseline["closed_loop_tracking_error"]

        for iteration in range(1, self.config.iterations + 1):
            iteration_started = time.perf_counter()

            occupancy = rollout_tape(self.policy, tape)
            critic_metrics, critic_norm = self._update_critic(
                tape, occupancy, critic_optimiser, generator, rng
            )
            policy_loss, policy_norm = self._update_policy(
                tape, policy_optimiser, rng
            )

            evaluate = (
                iteration % self.config.evaluate_every == 0
                or iteration == self.config.iterations
            )
            metrics = (
                self._evaluate(validation_dataset, oracle)
                if evaluate
                else {"closed_loop_tracking_error": float("nan"), "mean_turnover": float("nan")}
            )

            sample_states, sample_actions = self._sample(tape, occupancy, rng)
            record = GAILRecord(
                iteration=iteration,
                wasserstein_estimate=critic_metrics["wasserstein_estimate"],
                critic_loss=critic_metrics["critic_loss"],
                gradient_penalty=critic_metrics["gradient_penalty"],
                policy_loss=policy_loss,
                critic_gradient_norm=critic_norm,
                policy_gradient_norm=policy_norm,
                measured_lipschitz=self.critic.measured_lipschitz_constant(
                    sample_states, sample_actions
                ),
                closed_loop_tracking_error=metrics["closed_loop_tracking_error"],
                mean_turnover=metrics["mean_turnover"],
                seconds=time.perf_counter() - iteration_started,
            )
            history.records.append(record)

            if not np.isfinite(record.wasserstein_estimate) or not np.isfinite(
                record.policy_loss
            ):
                history.diverged = True
                break

            if evaluate and record.closed_loop_tracking_error < (
                history.best_closed_loop_error
            ):
                history.best_closed_loop_error = record.closed_loop_tracking_error
                history.best_iteration = iteration
                best_state = deepcopy(self.policy.state_dict())

        # Selection on the comparable metric, for the same reason DAgger selects
        # on the round rather than returning the last one: the adversarial game
        # does not converge monotonically and the final iterate is a draw from
        # its variation.
        self.policy.load_state_dict(best_state)
        history.total_seconds = time.perf_counter() - started
        return history

    # ------------------------------------------------------------------
    # Updates
    # ------------------------------------------------------------------
    def _update_critic(
        self,
        tape: RolloutTape,
        occupancy: Any,
        optimiser: torch.optim.Optimizer,
        generator: torch.Generator,
        rng: np.random.Generator,
    ) -> tuple[dict[str, float], float]:
        """Take the configured number of critic steps against a fixed policy."""
        metrics: dict[str, float] = {}
        norm = 0.0
        for _ in range(self.critic_config.steps_per_round):
            steps, columns = self._indices(tape, rng)
            expert_states = tape.gather_expert(steps, columns)
            expert_actions = tape.expert_actions[steps, columns]
            policy_states = occupancy.states[steps, columns]
            policy_actions = occupancy.actions[steps, columns]

            optimiser.zero_grad(set_to_none=True)
            loss, metrics = self.critic.critic_loss(
                expert_states,
                expert_actions,
                policy_states.detach(),
                policy_actions.detach(),
                generator=generator,
            )
            loss.backward()
            norm = float(
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), float("inf"))
            )
            optimiser.step()
            self.critic.enforce_constraint()
        return metrics, norm

    def _update_policy(
        self,
        tape: RolloutTape,
        optimiser: torch.optim.Optimizer,
        rng: np.random.Generator,
    ) -> tuple[float, float]:
        """Minimise the critic's dual gap through a truncated differentiable rollout."""
        total_loss = 0.0
        norm = 0.0
        for _ in range(self.config.policy_steps):
            length = min(self.config.segment_length, tape.n_steps)
            start = int(rng.integers(0, max(1, tape.n_steps - length + 1)))
            # The anchor at the segment start is the expert's, which keeps
            # segments comparable across iterations and stops an early bad
            # policy from being scored only on states it corrupted itself.
            anchor = tape.expert_anchors[start]

            optimiser.zero_grad(set_to_none=True)
            segment = rollout_segment(self.policy, tape, anchor, start, length)
            states, actions = segment.flat()
            loss = -self.critic.score(states, actions).mean()

            if self.config.bc_anchor_weight > 0.0:
                expert = tape.expert_actions[start : start + length].reshape(
                    -1, tape.n_assets
                )
                loss = loss + self.config.bc_anchor_weight * (
                    (actions - expert) ** 2
                ).sum(dim=-1).mean()

            loss.backward()
            norm = float(
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.config.gradient_clip
                )
            )
            optimiser.step()
            total_loss += float(loss.detach())
        return total_loss / self.config.policy_steps, norm

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _indices(
        self, tape: RolloutTape, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample matched ``(step, path)`` positions for the critic minibatch.

        Expert and policy pairs are drawn at the *same* positions. Because the
        market blocks are shared, that makes the two samples differ only in the
        anchor and the action, which is exactly the difference the critic should
        be measuring.
        """
        count = min(self.config.batch_size, tape.n_steps * tape.n_paths)
        steps = rng.integers(0, tape.n_steps, size=count)
        columns = rng.integers(0, tape.n_paths, size=count)
        return steps, columns

    def _sample(
        self, tape: RolloutTape, occupancy: Any, rng: np.random.Generator
    ) -> tuple[torch.Tensor, torch.Tensor]:
        steps, columns = self._indices(tape, rng)
        return (
            occupancy.states[steps, columns].detach(),
            occupancy.actions[steps, columns].detach(),
        )

    def _evaluate(
        self, dataset: DemonstrationDataset, oracle: ExpertOracle
    ) -> dict[str, float]:
        rollouts = rollout_policy(self.policy, dataset)
        answers = answer_rollouts(oracle, rollouts)
        return rollout_metrics(rollouts, answers)

    def occupancy_distance(
        self, dataset: DemonstrationDataset, *, max_steps: int | None = None
    ) -> float:
        """Return the critic's ``W1`` estimate on a held-out set of paths.

        Reported as a comparison quantity across BC, DAgger, and GAIL. It is a
        *lower bound* under this critic rather than the true ``W1``, and it is
        only comparable between policies scored by the **same** critic, which is
        why the demo scores every policy with the critic from one run.
        """
        tape = build_tape(dataset, device=self.device, max_steps=max_steps)
        occupancy = rollout_tape(self.policy, tape)
        with torch.no_grad():
            expert_states = torch.cat(
                [tape.expert_state(step) for step in range(tape.n_steps)]
            )
            expert_actions = tape.expert_actions.reshape(-1, tape.n_assets)
            policy_states, policy_actions = occupancy.flat()
            return float(
                self.critic.wasserstein_estimate(
                    expert_states, expert_actions, policy_states, policy_actions
                )
            )

    def realised_performance(
        self, dataset: DemonstrationDataset, transaction_cost: float
    ) -> dict[str, float]:
        """Return realised net return and turnover for a closed-loop rollout."""
        tape = build_tape(dataset, device=self.device)
        occupancy = rollout_tape(self.policy, tape)
        net = realised_net_returns(tape, occupancy, transaction_cost)
        return {
            "mean_net_return": float(net.mean()),
            "annualised_net_return": float(net.mean()) * 252.0,
            "mean_turnover": float(occupancy.turnover.mean()),
        }

    def describe(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "critic": self.critic.describe(),
            "policy_parameters": self.policy.n_parameters,
            "estimator": (
                "truncated backpropagation through the differentiable rollout; "
                "same Equation (9) objective as policy-gradient GAIL"
            ),
        }
