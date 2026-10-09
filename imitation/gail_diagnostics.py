"""Acceptance checks for the GAIL milestone.

Two of these matter more than the rest, because this implementation makes a
claim the other milestones did not.

``rollout_gradient_correctness`` is the load-bearing one. The whole reason
GAIL can be trained here without a policy-gradient method is that the rollout
is differentiable and ``dW1/dtheta`` is exact. That is a claim about
correctness, not a modelling choice, so it is verified against central finite
differences rather than argued for. The check runs in float64: in float32 the
finite-difference estimate itself carries about ``1e-2`` relative error, which
would hide a genuinely wrong gradient.

``state_splice_exactness`` guards the optimisation that makes the rollout
affordable. States are assembled by splicing a differentiable anchor between
two precomputed halves rather than by calling the NumPy assembler each step; if
that splice ever disagreed with the assembler, the policy would be trained on
states the rest of the project never produces.

The remaining checks are the usual: feasible actions, a critic whose Lipschitz
constant is close to the one Equation (9) assumes, a run that did not diverge,
a rollout that ignores the future, and reproducibility.

The comparison against behavioural cloning and DAgger is **reported**, not
gated, for the same reason as in Weeks 22-25: the work package says
"investigate" and "compare", and a gate on the outcome would invite tuning
until it passed.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch

from ..dataset.demonstrations import DemonstrationDataset
from ..policy.base import AllocationPolicy
from .differentiable import build_tape, rollout_segment, rollout_tape
from .gail import GAILHistory, GAILTrainer


@dataclass(frozen=True)
class GAILDiagnosticThresholds:
    """Deterministic acceptance thresholds for one GAIL run."""

    max_gradient_relative_error: float = 1e-6
    max_splice_difference: float = 1e-6
    max_lipschitz_excess: float = 1.0
    max_feasibility_violation: float = 1e-6
    max_wasserstein_volatility: float = 1.0
    max_look_ahead_difference: float = 0.0
    max_determinism_difference: float = 0.0

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be nonnegative and finite.")


@dataclass(frozen=True)
class GAILDiagnostics:
    """Observed-versus-required behaviour of one GAIL run."""

    checks: dict[str, bool]
    gradient_relative_error: float
    checked_parameters: int
    splice_difference: float
    measured_lipschitz: float
    feasibility_violation: float
    wasserstein_volatility: float
    diverged: bool
    look_ahead_difference: float
    determinism_difference: float
    thresholds: GAILDiagnosticThresholds

    @property
    def passed(self) -> bool:
        return all(self.checks.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "checks": dict(self.checks),
            "rollout_gradient_correctness": {
                "max_relative_error": float(self.gradient_relative_error),
                "parameters_checked": int(self.checked_parameters),
                "note": (
                    "Backpropagation through the differentiable rollout against "
                    "central finite differences, in float64. This is what "
                    "licenses replacing GAIL's policy gradient with an exact one."
                ),
            },
            "state_splice_exactness": {"max_difference": float(self.splice_difference)},
            "critic_lipschitz": {"measured_constant": float(self.measured_lipschitz)},
            "action_feasibility": {"max_violation": float(self.feasibility_violation)},
            "training_stability": {
                "wasserstein_volatility": float(self.wasserstein_volatility),
                "diverged": self.diverged,
            },
            "rollout_no_look_ahead": {
                "max_action_difference": float(self.look_ahead_difference)
            },
            "rollout_determinism": {
                "max_difference": float(self.determinism_difference)
            },
            "thresholds": asdict(self.thresholds),
        }


def evaluate_gail(
    trainer: GAILTrainer,
    history: GAILHistory,
    train_dataset: DemonstrationDataset,
    holdout_dataset: DemonstrationDataset,
    *,
    preset: Any,
    gradient_parameters: int = 5,
    thresholds: GAILDiagnosticThresholds | None = None,
) -> GAILDiagnostics:
    """Run every acceptance check for one GAIL run."""
    limits = thresholds if thresholds is not None else GAILDiagnosticThresholds()
    policy = trainer.policy

    gradient_error, checked = _gradient_error(policy, train_dataset, gradient_parameters)
    splice = _splice_difference(train_dataset)
    lipschitz, feasibility = _critic_and_feasibility(trainer, holdout_dataset)
    determinism = _determinism_difference(policy, holdout_dataset)
    look_ahead = _look_ahead_difference(policy, holdout_dataset, preset)

    stability = history.stability()
    volatility = float(stability.get("wasserstein_volatility", 0.0))

    checks = {
        "rollout_gradient_correctness": gradient_error
        <= limits.max_gradient_relative_error,
        "state_splice_exactness": splice <= limits.max_splice_difference,
        "critic_lipschitz": abs(lipschitz - 1.0) <= limits.max_lipschitz_excess,
        "action_feasibility": feasibility <= limits.max_feasibility_violation,
        "training_stability": (not history.diverged)
        and volatility <= limits.max_wasserstein_volatility,
        "rollout_no_look_ahead": look_ahead <= limits.max_look_ahead_difference,
        "rollout_determinism": determinism <= limits.max_determinism_difference,
    }

    return GAILDiagnostics(
        checks=checks,
        gradient_relative_error=gradient_error,
        checked_parameters=checked,
        splice_difference=splice,
        measured_lipschitz=lipschitz,
        feasibility_violation=feasibility,
        wasserstein_volatility=volatility,
        diverged=history.diverged,
        look_ahead_difference=look_ahead,
        determinism_difference=determinism,
        thresholds=limits,
    )


def _gradient_error(
    policy: AllocationPolicy,
    dataset: DemonstrationDataset,
    count: int,
    *,
    segment: int = 8,
    epsilon: float = 1e-6,
) -> tuple[float, int]:
    """Compare backpropagated rollout gradients with central finite differences.

    Run in float64 throughout. The objective is a plain smooth functional of the
    rolled actions rather than the critic score, so the check tests the rollout
    and the projection rather than the critic's own graph.
    """
    probe = deepcopy(policy).double()
    tape = build_tape(dataset, dtype=torch.float64, max_steps=segment + 1)

    def objective() -> torch.Tensor:
        rolled = rollout_segment(probe, tape, tape.initial_anchor, 0, segment)
        return (rolled.actions**2).sum() + rolled.turnover.sum()

    probe.zero_grad(set_to_none=True)
    objective().backward()

    parameters = [p for p in probe.parameters() if p.grad is not None and p.numel() > 1]
    if not parameters:
        return 0.0, 0
    def largest_gradient(parameter: torch.Tensor) -> float:
        gradient = parameter.grad
        return 0.0 if gradient is None else float(gradient.abs().max())

    target = max(parameters, key=largest_gradient)
    target_gradient = target.grad
    assert target_gradient is not None  # filtered above
    gradients = target_gradient.detach().reshape(-1).clone()
    flat = target.data.reshape(-1)

    order = torch.argsort(gradients.abs(), descending=True)[:count]
    worst = 0.0
    for index in order:
        original = flat[index].item()
        flat[index] = original + epsilon
        plus = float(objective().detach())
        flat[index] = original - epsilon
        minus = float(objective().detach())
        flat[index] = original

        estimate = (plus - minus) / (2.0 * epsilon)
        analytic = gradients[index].item()
        scale = max(abs(estimate), abs(analytic), 1e-12)
        worst = max(worst, abs(estimate - analytic) / scale)
    return worst, int(order.numel())


def _splice_difference(dataset: DemonstrationDataset) -> float:
    """The spliced state must equal the assembler's, at the expert's own anchor."""
    tape = build_tape(dataset, dtype=torch.float64, max_steps=8)
    entry = dataset.entries[0]
    worst = 0.0
    for step in range(tape.n_steps):
        spliced = tape.expert_state(step)[0].numpy()
        reference = dataset.assemble_state(entry, int(entry.indices[step]))
        worst = max(worst, float(np.abs(spliced - reference).max()))
    return worst


def _critic_and_feasibility(
    trainer: GAILTrainer, dataset: DemonstrationDataset
) -> tuple[float, float]:
    """Return the critic's measured Lipschitz constant and the action violation."""
    tape = build_tape(dataset, device=trainer.device)
    occupancy = rollout_tape(trainer.policy, tape)
    states, actions = occupancy.flat()

    lipschitz = trainer.critic.measured_lipschitz_constant(
        states[:512].detach(), actions[:512].detach()
    )
    violations = trainer.policy.constraints.violations(
        actions.detach().cpu().numpy().astype(np.float64)
    )
    return lipschitz, max(violations.values(), default=0.0)


def _determinism_difference(
    policy: AllocationPolicy, dataset: DemonstrationDataset
) -> float:
    from .rollout import rollout_policy

    first = rollout_policy(policy, dataset)
    second = rollout_policy(policy, dataset)
    return max(
        float(np.abs(a.policy_weights - b.policy_weights).max())
        for a, b in zip(first, second)
    )


def _look_ahead_difference(
    policy: AllocationPolicy, dataset: DemonstrationDataset, preset: Any
) -> float:
    """Rewrite the future and re-roll, with the Weeks 14-16 control discipline."""
    from ..dataset.diagnostics import rebuild_truncated_path
    from .rollout import rollout_policy

    entry = dataset.entries[0]
    path = entry.data
    sample_index = int(entry.indices[min(24, entry.indices.size - 1)])
    t = int(path.decision_times[sample_index])

    altered = path.returns.copy()
    n_after = altered.shape[0] - t
    altered[t:] = path.returns[:n_after][::-1]

    def final_action(returns: np.ndarray) -> np.ndarray:
        rebuilt = rebuild_truncated_path(path, returns, preset, sample_index)
        rebuilt_dataset = DemonstrationDataset([rebuilt], spec=dataset.spec)
        return rollout_policy(policy, rebuilt_dataset)[0].policy_weights[-1]

    control = final_action(path.returns.copy())
    treatment = final_action(altered)
    return float(np.abs(treatment - control).max())


def compare_policies(
    trainer: GAILTrainer,
    named_policies: dict[str, AllocationPolicy],
    dataset: DemonstrationDataset,
) -> dict[str, float]:
    """Score several policies with the *same* critic.

    **The result is not a valid comparison between policies**, and the shipped
    run demonstrates why: it returns negative values for two of them, which is
    impossible for a lower bound on a distance, and ranks the policy with the
    best closed-loop error worst.

    Using one critic makes the numbers commensurable but not meaningful. A
    critic is trained to discriminate the expert from **one particular** policy
    and approaches the supremum of Equation (9) only for that pair; applied to a
    policy it never saw it is an arbitrary 1-Lipschitz function. Estimating
    ``W1(d_E, d_pi)`` per policy properly requires a fresh critic trained to
    convergence against each one.

    Kept because the per-policy score is still a useful diagnostic of what the
    critic learned during its own run, and because the failure is instructive.
    """
    original = trainer.policy
    distances: dict[str, float] = {}
    try:
        for name, policy in named_policies.items():
            trainer.policy = policy
            distances[name] = trainer.occupancy_distance(dataset)
    finally:
        trainer.policy = original
    return distances
