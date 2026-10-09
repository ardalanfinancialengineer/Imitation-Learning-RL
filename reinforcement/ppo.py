"""PPO fine-tuning against the Equation (11) reward.

Stage 4 of the exposé's pipeline. The policy is warm-started from imitation
learning and then optimised on realised outcomes, which is what research
question 4 asks about: whether fine-tuning improves realised certainty-equivalent
utility "without unacceptable turnover, drawdown, or deviation from the expert
benchmark".

PPO is Schulman et al.'s clipped surrogate,

    L(theta) = E[ min( r_t A_t, clip(r_t, 1-eps, 1+eps) A_t ) ],
    r_t = pi_theta(z_t|s_t) / pi_theta_old(z_t|s_t),

with generalised advantage estimation, a clipped value loss, and an entropy
bonus. The exposé chooses it because it limits how far one update can move the
policy, which matters more than usual here: the starting point is a carefully
trained imitation solution that is easy to destroy and expensive to recover.

Three details are specific to this problem.

**The stored action is the latent, not the allocation.** The Gaussian lives
before Equation (8)'s projection, so the ratio is computed on ``z`` while the
environment executes ``Pi(z)``. See :mod:`src.reinforcement.stochastic`.

**Episodes are the full path, and there is no bootstrapping across paths.** Each
simulated path is one episode with a known end, so the advantage of the last
step uses a terminal value of zero rather than a bootstrap.

**Rollouts are batched across paths on the tape.** The environment is a fixed
return series, so every path can be stepped simultaneously, and the whole
rollout is one batched forward pass per step rather than one per path.
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
from ..imitation.differentiable import RolloutTape, build_tape
from ..utils.device import resolve_device
from .reward import RewardConfig, certainty_equivalent, net_returns, tape_rewards
from .stochastic import GaussianActorCritic


@dataclass(frozen=True)
class PPOConfig:
    """Optimisation settings for the clipped surrogate."""

    iterations: int = 40
    epochs_per_iteration: int = 4
    minibatch_size: int = 256
    learning_rate: float = 3e-5
    clip_range: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 1e-4
    gae_lambda: float = 0.95
    discount: float = 0.99
    gradient_clip: float = 0.5
    target_kl: float = 0.02
    seed: int = 0
    device: str = "cpu"

    def __post_init__(self) -> None:
        for name in ("iterations", "epochs_per_iteration", "minibatch_size"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")
        if not 0.0 < self.clip_range < 1.0:
            raise ValueError("clip_range must lie strictly in (0, 1).")
        for name in ("gae_lambda", "discount"):
            if not 0.0 < getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must lie in (0, 1].")
        for name in ("value_coefficient", "entropy_coefficient", "gradient_clip",
                     "target_kl"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be nonnegative.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PPORecord:
    """One PPO iteration."""

    iteration: int
    mean_reward: float
    mean_net_return: float
    certainty_equivalent: float
    mean_turnover: float
    policy_loss: float
    value_loss: float
    entropy: float
    approximate_kl: float
    clip_fraction: float
    explained_variance: float
    standard_deviation: float
    feasibility_violation: float
    seconds: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PPOHistory:
    """Outcome of one PPO run."""

    records: list[PPORecord] = field(default_factory=list)
    total_seconds: float = 0.0
    best_iteration: int = 0
    best_certainty_equivalent: float = -float("inf")

    def series(self, name: str) -> np.ndarray:
        """Return one recorded quantity across iterations."""
        return np.array([getattr(record, name) for record in self.records], dtype=float)

    def to_dict(self) -> dict[str, Any]:
        return {
            "iterations_run": len(self.records),
            "total_seconds": round(self.total_seconds, 2),
            "best_iteration": self.best_iteration,
            "best_certainty_equivalent": self.best_certainty_equivalent,
            "records": [record.to_dict() for record in self.records],
        }


class PPOTrainer:
    """Fine-tune an allocation policy on the Equation (11) reward."""

    def __init__(
        self,
        actor_critic: GaussianActorCritic,
        reward: RewardConfig,
        *,
        config: PPOConfig | None = None,
    ) -> None:
        self.actor_critic = actor_critic
        self.reward = reward
        self.config = config if config is not None else PPOConfig()
        self.device = resolve_device(self.config.device)
        self.actor_critic.to(self.device)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit(
        self,
        train_dataset: DemonstrationDataset,
        validation_dataset: DemonstrationDataset,
    ) -> PPOHistory:
        """Run the configured number of PPO iterations."""
        seed_everything(self.config.seed)
        tape = build_tape(train_dataset, device=self.device, with_covariances=True)
        validation_tape = build_tape(
            validation_dataset, device=self.device, with_covariances=True
        )

        optimiser = torch.optim.AdamW(
            self.actor_critic.parameters(), lr=self.config.learning_rate
        )
        generator = torch.Generator(device="cpu").manual_seed(self.config.seed)
        rng = np.random.default_rng(self.config.seed)

        history = PPOHistory()
        best_state = deepcopy(self.actor_critic.state_dict())
        started = time.perf_counter()

        for iteration in range(1, self.config.iterations + 1):
            iteration_started = time.perf_counter()
            batch = self._collect(tape, generator)
            losses = self._update(batch, optimiser, rng)
            evaluation = self.evaluate(validation_tape)

            record = PPORecord(
                iteration=iteration,
                mean_reward=float(batch["rewards"].mean()),
                mean_net_return=evaluation["mean_net_return"],
                certainty_equivalent=evaluation["certainty_equivalent"],
                mean_turnover=evaluation["mean_turnover"],
                seconds=time.perf_counter() - iteration_started,
                standard_deviation=float(
                    self.actor_critic.standard_deviation.mean().detach()
                ),
                feasibility_violation=evaluation["feasibility_violation"],
                **losses,
            )
            history.records.append(record)

            if record.certainty_equivalent > history.best_certainty_equivalent:
                history.best_certainty_equivalent = record.certainty_equivalent
                history.best_iteration = iteration
                best_state = deepcopy(self.actor_critic.state_dict())

        # Selection on validation certainty-equivalent utility, the quantity
        # research question 4 is about, for the same reason DAgger and GAIL
        # select on a round rather than returning the last one.
        self.actor_critic.load_state_dict(best_state)
        history.total_seconds = time.perf_counter() - started
        return history

    # ------------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _collect(
        self, tape: RolloutTape, generator: torch.Generator
    ) -> dict[str, torch.Tensor]:
        """Roll every path simultaneously and compute GAE advantages."""
        states, latents, log_probabilities, values = [], [], [], []
        actions, anchors = [], []

        anchor = tape.initial_anchor
        for step in range(tape.n_steps):
            state = tape.state(step, anchor)
            drawn = self.actor_critic.sample(state, generator=generator)
            states.append(state)
            latents.append(drawn["latent"])
            log_probabilities.append(drawn["log_probability"])
            values.append(drawn["value"])
            actions.append(drawn["action"])
            anchors.append(anchor)
            anchor = drawn["action"]

        stacked = {
            "states": torch.stack(states),
            "latents": torch.stack(latents),
            "log_probabilities": torch.stack(log_probabilities),
            "values": torch.stack(values),
            "actions": torch.stack(actions),
            "anchors": torch.stack(anchors),
        }
        stacked["rewards"] = tape_rewards(
            tape, stacked["actions"], stacked["anchors"], self.reward
        )
        stacked["advantages"], stacked["returns"] = self._advantages(
            stacked["rewards"], stacked["values"]
        )
        return stacked

    def _advantages(
        self, rewards: torch.Tensor, values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generalised advantage estimation with a terminal value of zero.

        Each path is a complete episode with a known end, so the final step
        bootstraps from zero rather than from a value estimate.
        """
        advantages = torch.zeros_like(rewards)
        running = torch.zeros_like(rewards[0])
        for step in reversed(range(rewards.shape[0])):
            next_value = (
                torch.zeros_like(values[0])
                if step == rewards.shape[0] - 1
                else values[step + 1]
            )
            delta = rewards[step] + self.config.discount * next_value - values[step]
            running = delta + self.config.discount * self.config.gae_lambda * running
            advantages[step] = running
        return advantages, advantages + values

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------
    def _update(
        self,
        batch: dict[str, torch.Tensor],
        optimiser: torch.optim.Optimizer,
        rng: np.random.Generator,
    ) -> dict[str, float]:
        """Run the clipped surrogate over the collected batch."""
        states = batch["states"].reshape(-1, batch["states"].shape[-1])
        latents = batch["latents"].reshape(-1, batch["latents"].shape[-1])
        old_log_probabilities = batch["log_probabilities"].reshape(-1)
        advantages = batch["advantages"].reshape(-1)
        returns = batch["returns"].reshape(-1)
        old_values = batch["values"].reshape(-1)

        # Advantage normalisation. The Equation (11) reward is of order 1e-3, so
        # without it the surrogate gradient is swamped by the value loss.
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        totals = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approximate_kl": 0.0,
            "clip_fraction": 0.0,
        }
        updates = 0
        count = states.shape[0]

        for _ in range(self.config.epochs_per_iteration):
            order = rng.permutation(count)
            for start in range(0, count, self.config.minibatch_size):
                index = torch.as_tensor(
                    order[start : start + self.config.minibatch_size],
                    device=states.device,
                )
                log_probability, entropy, value = self.actor_critic.evaluate(
                    states[index], latents[index]
                )
                ratio = (log_probability - old_log_probabilities[index]).exp()
                clipped = ratio.clamp(
                    1.0 - self.config.clip_range, 1.0 + self.config.clip_range
                )
                surrogate = -torch.min(
                    ratio * advantages[index], clipped * advantages[index]
                ).mean()

                value_clipped = old_values[index] + (
                    value - old_values[index]
                ).clamp(-self.config.clip_range, self.config.clip_range)
                value_loss = torch.max(
                    (value - returns[index]) ** 2,
                    (value_clipped - returns[index]) ** 2,
                ).mean()

                loss = (
                    surrogate
                    + self.config.value_coefficient * value_loss
                    - self.config.entropy_coefficient * entropy
                )

                optimiser.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.actor_critic.parameters(), self.config.gradient_clip
                )
                optimiser.step()

                with torch.no_grad():
                    approximate_kl = float(
                        (old_log_probabilities[index] - log_probability).mean()
                    )
                    totals["policy_loss"] += float(surrogate.detach())
                    totals["value_loss"] += float(value_loss.detach())
                    totals["entropy"] += float(entropy.detach())
                    totals["approximate_kl"] += approximate_kl
                    totals["clip_fraction"] += float(
                        ((ratio - 1.0).abs() > self.config.clip_range).float().mean()
                    )
                updates += 1

            # Early stop on the KL budget, which is the practical form of PPO's
            # "limit very large changes between training steps".
            if self.config.target_kl > 0.0 and abs(
                totals["approximate_kl"] / max(updates, 1)
            ) > self.config.target_kl:
                break

        divisor = max(updates, 1)
        metrics = {name: value / divisor for name, value in totals.items()}
        metrics["explained_variance"] = _explained_variance(returns, old_values)
        return metrics

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    @torch.no_grad()
    def evaluate(self, tape: RolloutTape) -> dict[str, float]:
        """Score the *deterministic* policy on a tape.

        Exploration noise is switched off: the deployed policy is the mean, and
        reporting the noisy one would understate it.
        """
        anchor = tape.initial_anchor
        actions, anchors = [], []
        for step in range(tape.n_steps):
            drawn = self.actor_critic.sample(
                tape.state(step, anchor), deterministic=True
            )
            actions.append(drawn["action"])
            anchors.append(anchor)
            anchor = drawn["action"]

        stacked_actions = torch.stack(actions)
        stacked_anchors = torch.stack(anchors)
        net = net_returns(
            stacked_actions,
            stacked_anchors,
            tape.realised_returns,
            self.reward.transaction_cost,
        )
        turnover = (stacked_actions - stacked_anchors).abs().sum(dim=-1)
        violations = self.actor_critic.policy.constraints.violations(
            stacked_actions.reshape(-1, tape.n_assets).cpu().numpy().astype(np.float64)
        )
        return {
            "mean_net_return": float(net.mean()),
            "annualised_net_return": float(net.mean()) * 252.0,
            "certainty_equivalent": certainty_equivalent(net, self.reward.risk_aversion),
            "mean_turnover": float(turnover.mean()),
            "feasibility_violation": max(violations.values(), default=0.0),
        }

    def describe(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "reward": self.reward.to_dict(),
            "actor_critic": self.actor_critic.describe(),
            "objective": (
                "clipped surrogate on rho_t = R_tilde - (gamma/2) w'Sigma w "
                "- lambda_E ||w - w*||^2 (Equation 11)"
            ),
        }


def _explained_variance(returns: torch.Tensor, values: torch.Tensor) -> float:
    """Fraction of the return variance the value head accounts for."""
    variance = float(returns.var())
    if variance <= 0.0:
        return 0.0
    return float(1.0 - (returns - values).var() / variance)
