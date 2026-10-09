"""Acceptance checks for the DAgger milestone.

The exposé frames Weeks 22-25 as an investigation — whether "collecting
examples from the learner's own trajectories improves its ability to follow the
expert over time" — not as a result to be guaranteed. The checks here therefore
gate the **mechanics** of the loop, and the improvement itself is reported with
a paired bootstrap interval rather than asserted. A milestone that gated on
"DAgger must help" would either be vacuous or would invite tuning until it
passed.

Six checks:

``expert_query_consistency``
    Querying the oracle at the expert's *own* anchor reproduces the stored
    expert action bit-identically. This is the check that the DAgger query is
    the same programme the demonstrations came from, rather than a re-derived
    approximation of it;
``state_patch_exactness``
    Assembling a state with the expert's own anchor substituted reproduces the
    stored state bit-identically, so the only thing a learner-visited state
    changes is what it is supposed to change;
``aggregation_growth``
    Every round adds samples, and the base demonstrations survive unchanged at
    the front of the aggregate;
``closed_loop_feasibility``
    Every allocation the policy takes during a rollout, and every expert answer
    collected, satisfies Equation (5)'s constraints;
``rollout_no_look_ahead``
    Rewriting the returns after a decision date leaves the closed-loop action
    at that date unchanged. Weeks 14-16 established this for the assembled
    state and Weeks 17-21 for the single-step policy output; here it is checked
    through a *sequential* rollout, where a leak could also enter through the
    anchor chain;
``rollout_determinism``
    Repeating a rollout reproduces the trajectory exactly.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from ..dataset.demonstrations import DemonstrationDataset
from ..dataset.diagnostics import rebuild_truncated_path
from ..dataset.paths import PathData
from ..policy.base import AllocationPolicy
from .comparison import BootstrapInterval, paired_bootstrap
from .dagger import AggregatedDataset
from .rollout import ExpertOracle, PathRollout, rollout_policy


@dataclass(frozen=True)
class DAggerDiagnosticThresholds:
    """Deterministic acceptance thresholds for one DAgger run."""

    max_query_difference: float = 0.0
    max_state_patch_difference: float = 0.0
    max_feasibility_violation: float = 1e-6
    max_look_ahead_difference: float = 0.0
    max_determinism_difference: float = 0.0

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be nonnegative and finite.")


@dataclass(frozen=True)
class DAggerDiagnostics:
    """Observed-versus-required behaviour of one DAgger run."""

    checks: dict[str, bool]
    query_difference: float
    state_patch_difference: float
    feasibility_violation: float
    look_ahead_difference: float
    look_ahead_decision_times: tuple[int, ...]
    determinism_difference: float
    aggregate_sizes: tuple[int, ...]
    thresholds: DAggerDiagnosticThresholds

    @property
    def passed(self) -> bool:
        return all(self.checks.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "checks": dict(self.checks),
            "expert_query_consistency": {"max_difference": float(self.query_difference)},
            "state_patch_exactness": {
                "max_difference": float(self.state_patch_difference)
            },
            "closed_loop_feasibility": {
                "max_violation": float(self.feasibility_violation)
            },
            "rollout_no_look_ahead": {
                "max_action_difference": float(self.look_ahead_difference),
                "checked_decision_times": list(self.look_ahead_decision_times),
            },
            "rollout_determinism": {"max_difference": float(self.determinism_difference)},
            "aggregation_growth": {"samples_after_each_round": list(self.aggregate_sizes)},
            "thresholds": asdict(self.thresholds),
        }


def evaluate_dagger(
    policy: AllocationPolicy,
    dataset: DemonstrationDataset,
    aggregate: AggregatedDataset,
    oracle: ExpertOracle,
    *,
    preset: Any,
    aggregate_sizes: tuple[int, ...],
    samples: int = 3,
    thresholds: DAggerDiagnosticThresholds | None = None,
) -> DAggerDiagnostics:
    """Run every acceptance check for one DAgger run."""
    limits = thresholds if thresholds is not None else DAggerDiagnosticThresholds()
    entry = dataset.entries[0]
    path = entry.data

    positions = np.unique(
        np.linspace(0, entry.indices.size - 1, min(samples, entry.indices.size))
        .round()
        .astype(int)
    )

    query_difference = 0.0
    patch_difference = 0.0
    for position in positions:
        sample_index = int(entry.indices[position])
        anchor = path.previous_weights[sample_index]
        answer = oracle.query(path.path_id, sample_index, anchor)
        query_difference = max(
            query_difference,
            float(np.abs(answer - path.expert_weights[sample_index]).max()),
        )
        patched = dataset.assemble_state(entry, sample_index, previous_weights=anchor)
        stored = dataset.assemble_state(entry, sample_index)
        patch_difference = max(patch_difference, float(np.abs(patched - stored).max()))

    rollouts = rollout_policy(policy, dataset)
    feasibility = _feasibility_violation(policy, rollouts, oracle)
    determinism = _determinism_difference(policy, dataset, rollouts)
    look_ahead, checked = _look_ahead_difference(policy, dataset, preset, positions)

    # At least two entries are required: a run always records round 0 plus one
    # aggregated round, so a single size means nothing was ever collected and
    # an empty pairwise comparison would vacuously pass.
    growth_ok = len(aggregate_sizes) >= 2 and all(
        later > earlier
        for earlier, later in zip(aggregate_sizes, aggregate_sizes[1:])
    )
    growth_ok = growth_ok and _base_preserved(aggregate, dataset)

    checks = {
        "expert_query_consistency": query_difference <= limits.max_query_difference,
        "state_patch_exactness": patch_difference <= limits.max_state_patch_difference,
        "aggregation_growth": growth_ok,
        "closed_loop_feasibility": feasibility <= limits.max_feasibility_violation,
        "rollout_no_look_ahead": look_ahead <= limits.max_look_ahead_difference,
        "rollout_determinism": determinism <= limits.max_determinism_difference,
    }

    return DAggerDiagnostics(
        checks=checks,
        query_difference=query_difference,
        state_patch_difference=patch_difference,
        feasibility_violation=feasibility,
        look_ahead_difference=look_ahead,
        look_ahead_decision_times=checked,
        determinism_difference=determinism,
        aggregate_sizes=tuple(int(size) for size in aggregate_sizes),
        thresholds=limits,
    )


def _feasibility_violation(
    policy: AllocationPolicy, rollouts: list[PathRollout], oracle: ExpertOracle
) -> float:
    worst = 0.0
    for rollout in rollouts:
        for weights in (rollout.policy_weights,):
            violations = policy.constraints.violations(weights)
            worst = max(worst, max(violations.values(), default=0.0))
        answers = oracle.query_many(
            rollout.path_id, rollout.sample_indices, rollout.policy_previous_weights
        )
        violations = policy.constraints.violations(answers)
        worst = max(worst, max(violations.values(), default=0.0))
    return worst


def _determinism_difference(
    policy: AllocationPolicy,
    dataset: DemonstrationDataset,
    rollouts: list[PathRollout],
) -> float:
    repeated = rollout_policy(policy, dataset)
    worst = 0.0
    for first, second in zip(rollouts, repeated):
        worst = max(
            worst, float(np.abs(first.policy_weights - second.policy_weights).max())
        )
    return worst


def _look_ahead_difference(
    policy: AllocationPolicy,
    dataset: DemonstrationDataset,
    preset: Any,
    positions: np.ndarray,
) -> tuple[float, tuple[int, ...]]:
    """Rewrite the future and re-run the rollout up to the decision date.

    Same control-versus-treatment discipline as Weeks 14-16: both sides are
    rebuilt the same way and differ only in the returns after ``t``, so the
    last-bit solver noise that the anchor chain amplifies cancels and the
    comparison can be made at zero tolerance.
    """
    entry = dataset.entries[0]
    path = entry.data
    worst = 0.0
    checked: list[int] = []

    for position in positions:
        sample_index = int(entry.indices[position])
        t = int(path.decision_times[sample_index])

        altered = path.returns.copy()
        n_after = altered.shape[0] - t
        altered[t:] = path.returns[:n_after][::-1]

        control = _final_action(policy, dataset, path, path.returns.copy(), preset, sample_index)
        treatment = _final_action(policy, dataset, path, altered, preset, sample_index)
        worst = max(worst, float(np.abs(treatment - control).max()))
        checked.append(t)
    return worst, tuple(checked)


def _final_action(
    policy: AllocationPolicy,
    dataset: DemonstrationDataset,
    path: PathData,
    returns: np.ndarray,
    preset: Any,
    sample_index: int,
) -> np.ndarray:
    rebuilt = rebuild_truncated_path(path, returns, preset, sample_index)
    rebuilt_dataset = DemonstrationDataset([rebuilt], spec=dataset.spec)
    rollouts = rollout_policy(policy, rebuilt_dataset)
    return rollouts[0].policy_weights[-1]


def _base_preserved(aggregate: AggregatedDataset, dataset: DemonstrationDataset) -> bool:
    """The base demonstrations must survive aggregation unchanged."""
    if aggregate.n_base != len(dataset):
        return False
    for index in np.unique(np.linspace(0, len(dataset) - 1, 3).round().astype(int)):
        state, action = aggregate.sample(int(index))
        base_state, base_action = dataset.sample(int(index))
        if not (
            np.array_equal(state, base_state) and np.array_equal(action, base_action)
        ):
            return False
    return True


def improvement_interval(
    before: dict[int, float], after: dict[int, float], *, seed: int = 0
) -> BootstrapInterval:
    """Paired bootstrap of ``before - after`` over held-out paths.

    Positive means DAgger reduced closed-loop tracking error. Section 5.4
    prescribes exactly this pairing; the units are paths rather than decisions
    because decisions within a path are chained through ``w_{t-1}``.
    """
    return paired_bootstrap(before, after, seed=seed)
