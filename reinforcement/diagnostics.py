"""Acceptance checks for the PPO fine-tuning milestone.

Two checks guard things this milestone newly risks.

``reward_matches_environment`` is the important one. Equation (11) is now
implemented twice — once in :class:`~src.reinforcement.environment.PortfolioEnv`
for the Gymnasium interface, and once over the batched tape because PPO needs
many rollouts and stepping one path at a time is too slow. Two implementations
of one equation is exactly the kind of duplication that drifts silently, so the
check replays a trajectory through both and requires them to agree.

``log_probability_correctness`` guards the density in PPO's likelihood ratio.
The Gaussian is written out by hand rather than taken from
``torch.distributions`` so that it stays differentiable in the mean without a
distribution object being rebuilt each step; the check compares it against
``torch.distributions.Normal`` to make sure the hand-rolled version is right.

``warm_start_fidelity`` is what makes "hybrid" mean anything: wrapping an
imitation-trained policy in the actor-critic must not change the allocation it
produces, or the hybrid arm would not actually start where imitation finished.

The comparison between hybrid and pure PPO is **reported**, not gated, as in the
two preceding milestones.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch

from ..dataset.demonstrations import DemonstrationDataset
from ..imitation.differentiable import build_tape
from ..policy.base import AllocationPolicy
from .environment import PortfolioEnv
from .ppo import PPOHistory, PPOTrainer
from .reward import RewardConfig, rewards
from .stochastic import GaussianActorCritic


@dataclass(frozen=True)
class PPODiagnosticThresholds:
    """Deterministic acceptance thresholds for one PPO run."""

    max_reward_difference: float = 1e-9
    max_feasibility_violation: float = 1e-6
    max_log_probability_difference: float = 1e-5
    max_warm_start_difference: float = 1e-6
    max_approximate_kl: float = 0.5
    max_determinism_difference: float = 0.0

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be nonnegative and finite.")


@dataclass(frozen=True)
class PPODiagnostics:
    """Observed-versus-required behaviour of one PPO run."""

    checks: dict[str, bool]
    reward_difference: float
    feasibility_violation: float
    log_probability_difference: float
    warm_start_difference: float
    max_approximate_kl: float
    determinism_difference: float
    thresholds: PPODiagnosticThresholds

    @property
    def passed(self) -> bool:
        return all(self.checks.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "checks": dict(self.checks),
            "reward_matches_environment": {
                "max_difference": float(self.reward_difference),
                "note": (
                    "Equation (11) is implemented twice — in PortfolioEnv and "
                    "over the batched tape. This check is what stops the two "
                    "drifting apart."
                ),
            },
            "action_feasibility": {"max_violation": float(self.feasibility_violation)},
            "log_probability_correctness": {
                "max_difference": float(self.log_probability_difference)
            },
            "warm_start_fidelity": {"max_difference": float(self.warm_start_difference)},
            "training_stability": {"max_approximate_kl": float(self.max_approximate_kl)},
            "rollout_determinism": {
                "max_difference": float(self.determinism_difference)
            },
            "thresholds": asdict(self.thresholds),
        }


def evaluate_ppo(
    trainer: PPOTrainer,
    history: PPOHistory,
    dataset: DemonstrationDataset,
    reference_policy: AllocationPolicy,
    *,
    thresholds: PPODiagnosticThresholds | None = None,
) -> PPODiagnostics:
    """Run every acceptance check for one PPO run."""
    limits = thresholds if thresholds is not None else PPODiagnosticThresholds()

    reward_difference = _reward_agreement(dataset, trainer.reward)
    feasibility, determinism = _feasibility_and_determinism(trainer, dataset)
    log_probability = _log_probability_difference(trainer.actor_critic, dataset)
    warm_start = _warm_start_difference(trainer.actor_critic, reference_policy, dataset)
    kl = float(np.max(np.abs(history.series("approximate_kl")))) if history.records else 0.0

    checks = {
        "reward_matches_environment": reward_difference <= limits.max_reward_difference,
        "action_feasibility": feasibility <= limits.max_feasibility_violation,
        "log_probability_correctness": log_probability
        <= limits.max_log_probability_difference,
        "warm_start_fidelity": warm_start <= limits.max_warm_start_difference,
        "training_stability": np.isfinite(kl) and kl <= limits.max_approximate_kl,
        "rollout_determinism": determinism <= limits.max_determinism_difference,
    }

    return PPODiagnostics(
        checks=checks,
        reward_difference=reward_difference,
        feasibility_violation=feasibility,
        log_probability_difference=log_probability,
        warm_start_difference=warm_start,
        max_approximate_kl=kl,
        determinism_difference=determinism,
        thresholds=limits,
    )


def _reward_agreement(dataset: DemonstrationDataset, config: RewardConfig) -> float:
    """Replay the expert through the environment and through the tape reward.

    The environment applies Equation (11) one step at a time; the tape applies it
    to a batch. Any disagreement means the duplicate implementations have drifted.
    """
    entry = dataset.entries[0]
    path = entry.data
    environment = PortfolioEnv(
        path,
        spec=dataset.spec,
        expert_deviation_penalty=config.expert_deviation_penalty,
    )
    environment.reset()

    # float64 throughout: the environment works in NumPy double precision, so a
    # float32 tape would disagree at about 1e-9 for representation reasons alone
    # and the check could not be set tight enough to catch a real drift.
    tape = build_tape(
        DemonstrationDataset([path], spec=dataset.spec),
        dtype=torch.float64,
        with_covariances=True,
        max_steps=min(32, entry.indices.size),
    )
    actions, anchors = [], []
    environment_rewards = []
    anchor = tape.initial_anchor[0]
    for step in range(tape.n_steps):
        action = torch.as_tensor(environment.expert_action(), dtype=torch.float64)
        _, reward, terminated, _, _ = environment.step(action.numpy())
        environment_rewards.append(reward)
        actions.append(action)
        anchors.append(anchor)
        anchor = action
        if terminated:
            break

    count = len(environment_rewards)
    covariances = tape.forecast_covariances
    assert covariances is not None  # built with with_covariances=True
    tape_reward = rewards(
        torch.stack(actions),
        torch.stack(anchors),
        tape.realised_returns[:count, 0],
        covariances[:count, 0],
        tape.expert_actions[:count, 0],
        config,
    )
    return float(
        np.abs(tape_reward.numpy() - np.array(environment_rewards)).max()
    )


def _feasibility_and_determinism(
    trainer: PPOTrainer, dataset: DemonstrationDataset
) -> tuple[float, float]:
    tape = build_tape(dataset, device=trainer.device, with_covariances=True)
    first = _deterministic_actions(trainer.actor_critic, tape)
    second = _deterministic_actions(trainer.actor_critic, tape)

    violations = trainer.actor_critic.policy.constraints.violations(
        first.reshape(-1, tape.n_assets).cpu().numpy().astype(np.float64)
    )
    return (
        max(violations.values(), default=0.0),
        float((first - second).abs().max()),
    )


@torch.no_grad()
def _deterministic_actions(actor_critic: GaussianActorCritic, tape: Any) -> torch.Tensor:
    anchor = tape.initial_anchor
    actions = []
    for step in range(tape.n_steps):
        drawn = actor_critic.sample(tape.state(step, anchor), deterministic=True)
        actions.append(drawn["action"])
        anchor = drawn["action"]
    return torch.stack(actions)


def _log_probability_difference(
    actor_critic: GaussianActorCritic, dataset: DemonstrationDataset
) -> float:
    """Check the hand-written Gaussian density against torch.distributions."""
    tape = build_tape(dataset, with_covariances=False, max_steps=8)
    states = tape.expert_state(0)
    with torch.no_grad():
        mean = actor_critic.mean(states)
        scale = actor_critic.standard_deviation
        latent = mean + 0.5 * scale
        theirs = (
            torch.distributions.Normal(mean, scale).log_prob(latent).sum(dim=-1)
        )
        ours = actor_critic.log_probability(states, latent)
    return float((ours - theirs).abs().max())


def _warm_start_difference(
    actor_critic: GaussianActorCritic,
    reference: AllocationPolicy,
    dataset: DemonstrationDataset,
) -> float:
    """Wrapping a policy in the actor-critic must not change its allocation.

    A **fresh** wrapper is built around the reference policy for the comparison.
    Using the trainer's own wrapper instead would measure how much PPO moved the
    policy — which is the point of running PPO — rather than whether the hybrid
    arm starts where imitation finished. That mistake was made first, and the
    check duly reported a difference of 3.3e-3.
    """
    tape = build_tape(dataset, with_covariances=False, max_steps=8)
    states = tape.expert_state(0)
    wrapper = GaussianActorCritic(reference, config=actor_critic.config)
    with torch.no_grad():
        wrapped = wrapper.project(wrapper.mean(states))
        direct = reference.allocate(states)
    return float((wrapped - direct).abs().max())
