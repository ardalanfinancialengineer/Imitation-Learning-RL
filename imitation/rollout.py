"""Closed-loop rollouts and expert queries at learner-visited states.

Behavioural cloning is evaluated *teacher-forced*: each state carries the
expert's own ``w_{t-1}``, so the policy is scored one decision at a time on the
expert's trajectory. That metric is structurally blind to the failure DAgger
exists to fix. Once the learner acts, its own small errors move it to states the
expert never visited, and the teacher-forced number cannot see that because it
never lets the learner's mistakes accumulate.

This module supplies the closed-loop view.

One property of this problem makes both the rollout and the expert query exact
and cheap. At decision ``t`` the market blocks of Equation (6) — the return
window, ``mu_hat_{t+1|t}``, ``vech(Sigma_hat_{t+1|t})``, and ``p_hat_t`` —
depend only on the simulated path, not on who is trading. **Only ``w_{t-1}``
differs** under the learner's trajectory. So:

- a learner-visited state is the stored state with one block substituted, and
- the expert query is the Weeks 9-13 programme re-solved at the same date with
  the learner's anchor.

No re-simulation is needed, and the covariate shift DAgger addresses is
localised to exactly one block, which makes it directly measurable rather than
merely asserted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..dataset.demonstrations import DemonstrationDataset, PathEntry
from ..expert.markowitz import RollingHorizonMarkowitzExpert
from ..policy.base import AllocationPolicy
from ..policy.reference import ReferencePolicy


@dataclass(frozen=True)
class PathRollout:
    """One closed-loop pass of a policy over a single path."""

    path_id: int
    sample_indices: np.ndarray
    decision_times: np.ndarray
    policy_weights: np.ndarray
    policy_previous_weights: np.ndarray
    expert_weights: np.ndarray
    expert_previous_weights: np.ndarray
    gross_returns: np.ndarray
    net_returns: np.ndarray
    turnover: np.ndarray

    @property
    def n_steps(self) -> int:
        return int(self.sample_indices.size)

    @property
    def anchor_drift(self) -> np.ndarray:
        """Per-step ``||w_{t-1}^policy - w_{t-1}^expert||_1``.

        This *is* the covariate shift, because the anchor block is the only
        part of the state that the learner's own trajectory changes.
        """
        return np.abs(self.policy_previous_weights - self.expert_previous_weights).sum(
            axis=1
        )


def rollout_policy(
    policy: AllocationPolicy | ReferencePolicy,
    dataset: DemonstrationDataset,
    *,
    initial_weights: np.ndarray | None = None,
    beta: float = 0.0,
    generator: np.random.Generator | None = None,
) -> list[PathRollout]:
    """Run ``policy`` closed-loop over every path of ``dataset``.

    The policy starts from the same initial portfolio the expert did and then
    acts on its *own* previous allocation at each step, so it visits its own
    state distribution rather than the expert's.

    ``beta`` is Ross et al.'s mixture weight: with that probability the step
    defers to the expert, which keeps early DAgger rounds near the demonstrated
    distribution. The expert branch replays the **stored** action ``w*_t``,
    which was computed at the expert's own anchor rather than at the mixture's
    current one. Re-solving instead would cost one quadratic programme per
    deferred step — the dominant cost of a round — for a branch whose weight
    decays to nothing after the first rounds. The approximation affects only
    which states get visited; the expert answers subsequently collected at those
    states are always exact, taken at the anchor actually reached.
    """
    if not 0.0 <= beta <= 1.0 or not np.isfinite(beta):
        raise ValueError("beta must lie in [0, 1].")
    if beta > 0.0 and generator is None:
        raise ValueError("a generator is required when beta is positive.")
    return [
        _rollout_entry(policy, dataset, entry, initial_weights, beta, generator)
        for entry in dataset.entries
    ]


def _rollout_entry(
    policy: AllocationPolicy | ReferencePolicy,
    dataset: DemonstrationDataset,
    entry: PathEntry,
    initial_weights: np.ndarray | None,
    beta: float = 0.0,
    generator: np.random.Generator | None = None,
) -> PathRollout:
    path = entry.data
    indices = entry.indices
    n_assets = path.n_assets
    constraints = policy.constraints

    anchor = (
        path.previous_weights[int(indices[0])].copy()
        if initial_weights is None
        else constraints.project(np.asarray(initial_weights, dtype=float))
    )

    weights = np.empty((indices.size, n_assets))
    anchors = np.empty((indices.size, n_assets))
    for step, sample_index in enumerate(indices):
        state = dataset.assemble_state(
            entry, int(sample_index), previous_weights=anchor
        )
        action = np.asarray(policy.act(state), dtype=np.float64)
        if beta > 0.0 and generator is not None and generator.random() < beta:
            action = path.expert_weights[int(sample_index)].astype(np.float64)
        anchors[step] = anchor
        weights[step] = action
        anchor = action

    decision_times = path.decision_times[indices]
    realised = path.returns[decision_times]
    gross = np.einsum("ij,ij->i", weights, realised)
    turnover = np.abs(weights - anchors).sum(axis=1)
    cost = float(path.metadata["expert"]["transaction_cost"])

    return PathRollout(
        path_id=path.path_id,
        sample_indices=indices.copy(),
        decision_times=decision_times,
        policy_weights=weights,
        policy_previous_weights=anchors,
        expert_weights=path.expert_weights[indices],
        expert_previous_weights=path.previous_weights[indices],
        gross_returns=gross,
        net_returns=gross - cost * turnover,
        turnover=turnover,
    )


class ExpertOracle:
    """Answers "what would the expert do here?" at arbitrary anchors.

    This is the DAgger query. The expert's forecasts at a date are independent
    of the anchor, so they are computed once per date and reused across rounds;
    only the quadratic programme is re-solved. That matters because the solve is
    the entire cost — about 8 ms per query at ``N = 8``, and it is paid once per
    collected state per round.
    """

    def __init__(self, dataset: DemonstrationDataset) -> None:
        self._experts: dict[int, RollingHorizonMarkowitzExpert] = {}
        self._entries: dict[int, PathEntry] = {}
        self._forecasts: dict[tuple[int, int], Any] = {}
        for entry in dataset.entries:
            path_id = entry.data.path_id
            self._entries[path_id] = entry
            self._experts[path_id] = entry.data.build_expert()
        self.n_queries = 0

    @property
    def path_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self._experts))

    def forecast(self, path_id: int, sample_index: int) -> Any:
        """Return the expert's moment forecast at one decision date."""
        key = (path_id, int(sample_index))
        if key not in self._forecasts:
            entry = self._entries[path_id]
            expert = self._experts[path_id]
            t = int(entry.data.decision_times[sample_index])
            self._forecasts[key] = expert.forecaster.forecast(t, expert.horizon)
        return self._forecasts[key]

    def query(
        self, path_id: int, sample_index: int, previous_weights: np.ndarray
    ) -> np.ndarray:
        """Return the expert's allocation at ``sample_index`` given an anchor."""
        if path_id not in self._experts:
            raise KeyError(
                f"path {path_id} is not covered by this oracle; it knows "
                f"{sorted(self._experts)}. Build a separate oracle per dataset — "
                "the training and validation splits hold different paths."
            )
        expert = self._experts[path_id]
        forecast = self.forecast(path_id, sample_index)
        solution = expert.solve(forecast, np.asarray(previous_weights, dtype=float))
        self.n_queries += 1
        return solution.weights

    def query_many(
        self,
        path_id: int,
        sample_indices: np.ndarray,
        previous_weights: np.ndarray,
    ) -> np.ndarray:
        """Answer a batch of queries on one path."""
        answers = np.empty((len(sample_indices), previous_weights.shape[1]))
        for row, sample_index in enumerate(sample_indices):
            answers[row] = self.query(path_id, int(sample_index), previous_weights[row])
        return answers


def rollout_metrics(
    rollouts: list[PathRollout],
    expert_answers: dict[int, np.ndarray] | None = None,
) -> dict[str, float]:
    """Summarise a set of closed-loop rollouts.

    ``closed_loop_tracking_error`` compares the policy's allocation with what
    the expert *would have done at the state the policy actually reached*, which
    needs ``expert_answers``. Without them only the trajectory statistics and
    the comparison against the expert's own trajectory are available; the latter
    is reported as ``trajectory_tracking_error`` and is a different, weaker
    quantity because the two allocations are answers to different states.
    """
    if not rollouts:
        raise ValueError("at least one rollout is required.")

    trajectory_errors, closed_loop_errors, drifts = [], [], []
    net_returns, turnovers = [], []
    for rollout in rollouts:
        trajectory_errors.append(
            np.sum((rollout.policy_weights - rollout.expert_weights) ** 2, axis=1)
        )
        drifts.append(rollout.anchor_drift)
        net_returns.append(rollout.net_returns)
        turnovers.append(rollout.turnover)
        if expert_answers is not None:
            answers = expert_answers[rollout.path_id]
            closed_loop_errors.append(
                np.sum((rollout.policy_weights - answers) ** 2, axis=1)
            )

    metrics = {
        "trajectory_tracking_error": float(np.mean(np.concatenate(trajectory_errors))),
        "mean_anchor_drift": float(np.mean(np.concatenate(drifts))),
        "final_anchor_drift": float(np.mean([drift[-1] for drift in drifts])),
        "mean_turnover": float(np.mean(np.concatenate(turnovers))),
        "mean_net_return": float(np.mean(np.concatenate(net_returns))),
        "annualised_net_return": float(np.mean(np.concatenate(net_returns)) * 252.0),
        "n_steps": int(sum(rollout.n_steps for rollout in rollouts)),
        "n_paths": len(rollouts),
    }
    if closed_loop_errors:
        metrics["closed_loop_tracking_error"] = float(
            np.mean(np.concatenate(closed_loop_errors))
        )
    return metrics


def per_path_closed_loop_errors(
    rollouts: list[PathRollout], expert_answers: dict[int, np.ndarray]
) -> dict[int, float]:
    """Return closed-loop tracking error per path, for paired comparison.

    Paired bootstrapping is done over paths for the reason given in
    :mod:`src.imitation.comparison`: decisions inside a path are chained through
    ``w_{t-1}`` and are anything but independent.
    """
    return {
        rollout.path_id: float(
            np.mean(
                np.sum((rollout.policy_weights - expert_answers[rollout.path_id]) ** 2, axis=1)
            )
        )
        for rollout in rollouts
    }


def answer_rollouts(
    oracle: ExpertOracle, rollouts: list[PathRollout]
) -> dict[int, np.ndarray]:
    """Query the expert at every state the policy actually visited."""
    return {
        rollout.path_id: oracle.query_many(
            rollout.path_id, rollout.sample_indices, rollout.policy_previous_weights
        )
        for rollout in rollouts
    }
