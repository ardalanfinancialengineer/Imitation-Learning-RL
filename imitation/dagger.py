"""Stage 2 of the imitation pipeline: Dataset Aggregation (DAgger).

Behavioural cloning learns only from states the *expert* visited. Once the
learned policy acts, its own small errors move it to states the demonstrations
never covered, and the errors compound. Ross et al. [2011] address this by
letting the current policy interact with the environment, asking the expert what
it would have done at the visited states, adding those pairs to the training
set, and retraining on the aggregate.

The exposé notes why this matters especially here: "the current allocation
influences future transaction costs and therefore future decisions". That is
exact in this implementation. The anchor ``w_{t-1}`` is both a block of the
state and the point the expert's cost term is anchored on, so a learner that
drifts changes *both* what it sees and what the right answer is.

Three implementation decisions carry weight.

**Aggregated states are stored as anchors, not as states.** A round records, per
path, which decisions were queried, the learner's anchor at each, and the
expert's answer — three small arrays. States are reassembled on demand by
substituting the anchor block, exactly as Weeks 14-16 established for the base
dataset. Materialising them instead would defeat that design at the first round.

**The standardiser is frozen after round 0.** Input statistics stay those of the
round-0 training split. Refitting them as the aggregate grows would move the
input scale under a warm-started network and make rounds incomparable.

**Only the anchor is resampled, never the market.** The return path is fixed, so
every round revisits the same dates with different anchors. This is what makes
the expert query exact rather than an approximation on a re-simulated world.
"""

from __future__ import annotations

import logging
import time
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from ..dataset.demonstrations import DemonstrationDataset, PathEntry
from ..dataset.state import StateSpec
from ..policy.base import AllocationPolicy
from ..utils.observability import MemoryTracker
from .bc import BCConfig, BehaviouralCloningTrainer, seed_everything
from .rollout import (
    ExpertOracle,
    PathRollout,
    answer_rollouts,
    per_path_closed_loop_errors,
    rollout_metrics,
    rollout_policy,
)

BETA_SCHEDULES = ("exponential", "zero", "constant")

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DAggerConfig:
    """Settings for the dataset-aggregation loop."""

    iterations: int = 4
    beta_schedule: str = "exponential"
    beta_decay: float = 0.5
    beta_constant: float = 0.0
    query_fraction: float = 0.25
    epochs_per_round: int = 8
    retrain_from_scratch: bool = False
    seed: int = 0
    max_queries_per_round: int = 20_000

    def __post_init__(self) -> None:
        if not isinstance(self.iterations, int) or self.iterations < 1:
            raise ValueError("iterations must be a positive integer.")
        if self.beta_schedule not in BETA_SCHEDULES:
            raise ValueError(
                f"unknown beta_schedule {self.beta_schedule!r}; "
                f"expected one of {BETA_SCHEDULES}."
            )
        for name in ("beta_decay", "beta_constant", "query_fraction"):
            value = getattr(self, name)
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1].")
        if self.query_fraction <= 0.0:
            raise ValueError("query_fraction must be positive.")
        if not isinstance(self.epochs_per_round, int) or self.epochs_per_round < 1:
            raise ValueError("epochs_per_round must be a positive integer.")
        if not isinstance(self.max_queries_per_round, int) or (
            self.max_queries_per_round < 1
        ):
            raise ValueError("max_queries_per_round must be a positive integer.")

    def beta(self, iteration: int) -> float:
        """Probability of deferring to the expert during round ``iteration``.

        Ross et al. mix the expert into the rollout policy so early rounds stay
        near the demonstrated distribution. ``exponential`` uses
        ``beta_i = beta_decay^i``, which is their suggested schedule and reaches
        a learner-only rollout quickly; ``zero`` is pure DAgger from the first
        round; ``constant`` holds a fixed mixture throughout.
        """
        if self.beta_schedule == "zero":
            return 0.0
        if self.beta_schedule == "constant":
            return float(self.beta_constant)
        return float(self.beta_decay**iteration)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {
            "betas": [self.beta(index) for index in range(1, self.iterations + 1)]
        }


@dataclass(frozen=True)
class RoundRecord:
    """One DAgger iteration, as reported in the history table."""

    iteration: int
    beta: float
    queries: int
    aggregate_samples: int
    mean_anchor_drift: float
    closed_loop_tracking_error: float
    teacher_forced_tracking_error: float
    mean_turnover: float
    annualised_net_return: float
    seconds: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DAggerHistory:
    """Outcome of one DAgger run."""

    records: list[RoundRecord] = field(default_factory=list)
    total_seconds: float = 0.0
    total_queries: int = 0
    best_iteration: int = 0
    best_closed_loop_error: float = float("inf")

    def to_dict(self) -> dict[str, Any]:
        return {
            "rounds_run": len(self.records),
            "total_seconds": round(self.total_seconds, 2),
            "total_expert_queries": self.total_queries,
            "best_iteration": self.best_iteration,
            "best_closed_loop_tracking_error": self.best_closed_loop_error,
            "records": [record.to_dict() for record in self.records],
        }


@dataclass(frozen=True)
class AggregatedRound:
    """The states one round collected, stored as anchors rather than states."""

    path_id: int
    sample_indices: np.ndarray
    anchors: np.ndarray
    expert_actions: np.ndarray


class AggregatedDataset(Dataset):
    """The base demonstrations plus every DAgger round, assembled on access.

    Presents the same surface as :class:`DemonstrationDataset` so the
    behavioural-cloning trainer, the evaluation metrics, and the paired
    comparison all work unchanged.
    """

    def __init__(self, base: DemonstrationDataset) -> None:
        self.base = base
        self.spec: StateSpec = base.spec
        self.assembler = base.assembler
        self.dtype = base.dtype
        self._rounds: list[AggregatedRound] = []
        self._entry_by_path: dict[int, PathEntry] = {
            entry.data.path_id: entry for entry in base.entries
        }
        self._extra_index: list[tuple[int, int]] = []

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------
    def add_round(self, rounds: list[AggregatedRound]) -> int:
        """Append one round's collected pairs. Returns the number added."""
        added = 0
        for collected in rounds:
            if collected.path_id not in self._entry_by_path:
                raise KeyError(f"path {collected.path_id} is not in the base dataset.")
            round_index = len(self._rounds)
            self._rounds.append(collected)
            self._extra_index.extend(
                (round_index, row) for row in range(collected.sample_indices.size)
            )
            added += int(collected.sample_indices.size)
        return added

    @property
    def n_aggregated(self) -> int:
        return len(self._extra_index)

    @property
    def n_base(self) -> int:
        return len(self.base)

    # ------------------------------------------------------------------
    # Dataset surface
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.base) + len(self._extra_index)

    def sample(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Return one ``(state, expert action)`` pair as NumPy arrays."""
        if index < len(self.base):
            return self.base.sample(index)
        round_index, row = self._extra_index[index - len(self.base)]
        collected = self._rounds[round_index]
        entry = self._entry_by_path[collected.path_id]
        state = self.base.assemble_state(
            entry,
            int(collected.sample_indices[row]),
            previous_weights=collected.anchors[row],
        )
        return state, collected.expert_actions[row].copy()

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        state, action = self.sample(index)
        return (
            torch.as_tensor(state, dtype=self.dtype),
            torch.as_tensor(action, dtype=self.dtype),
        )

    @property
    def state_dimension(self) -> int:
        return self.base.state_dimension

    @property
    def n_assets(self) -> int:
        return self.base.n_assets

    @property
    def entries(self) -> tuple[PathEntry, ...]:
        return self.base.entries

    @property
    def path_ids(self) -> tuple[int, ...]:
        return self.base.path_ids

    def describe(self) -> dict[str, Any]:
        return {
            "n_samples": len(self),
            "n_base_samples": self.n_base,
            "n_aggregated_samples": self.n_aggregated,
            "n_rounds": len({id(item) for item in self._rounds}),
            "state": self.assembler.describe(),
        }


class DAggerTrainer:
    """Run the dataset-aggregation loop around a behavioural-cloning trainer."""

    def __init__(
        self,
        policy: AllocationPolicy,
        *,
        bc_config: BCConfig | None = None,
        config: DAggerConfig | None = None,
    ) -> None:
        self.policy = policy
        self.bc_config = bc_config if bc_config is not None else BCConfig()
        self.config = config if config is not None else DAggerConfig()

    def fit(
        self,
        train_dataset: DemonstrationDataset,
        validation_dataset: DemonstrationDataset,
        *,
        oracle: ExpertOracle | None = None,
        validation_oracle: ExpertOracle | None = None,
        memory_tracker: MemoryTracker | None = None,
    ) -> tuple[DAggerHistory, AggregatedDataset]:
        """Aggregate and retrain for the configured number of rounds.

        The policy is expected to be **already behaviour-cloned**: DAgger
        extends Stage 1 rather than replacing it, and rolling out an untrained
        network would collect states from a random walk.
        """
        aggregate = AggregatedDataset(train_dataset)
        query_oracle = oracle if oracle is not None else ExpertOracle(train_dataset)
        # Validation paths are disjoint from training paths by construction, so
        # scoring the closed-loop rollout needs its own oracle.
        scoring_oracle = (
            validation_oracle
            if validation_oracle is not None
            else ExpertOracle(validation_dataset)
        )
        history = DAggerHistory()
        generator = np.random.default_rng(self.config.seed)
        started = time.perf_counter()

        # Round 0 is the behaviour-cloned policy itself. Recording it is what
        # makes the improvement measurable: without it the first aggregated
        # round has nothing to be compared against.
        baseline = self._record(
            iteration=0,
            beta=1.0,
            queries=0,
            aggregate_samples=len(aggregate),
            validation_dataset=validation_dataset,
            scoring_oracle=scoring_oracle,
            seconds=0.0,
        )
        history.records.append(baseline)
        history.best_closed_loop_error = baseline.closed_loop_tracking_error
        history.best_iteration = 0
        best_state = deepcopy(self.policy.state_dict())
        logger.info(
            "dagger round 0/%d (baseline) closed_loop_te=%.4e",
            self.config.iterations,
            baseline.closed_loop_tracking_error,
        )
        if memory_tracker is not None:
            memory_tracker.sample("dagger-round-0")

        for iteration in range(1, self.config.iterations + 1):
            round_started = time.perf_counter()
            beta = self.config.beta(iteration)

            rollouts = self._collect(train_dataset, beta, generator)
            collected, queries = self._query(rollouts, query_oracle, generator)
            aggregate.add_round(collected)

            bc_config = BCConfig(
                **(self.bc_config.to_dict() | {"epochs": self.config.epochs_per_round})
            )
            trainer = BehaviouralCloningTrainer(self.policy, bc_config)
            if self.config.retrain_from_scratch:
                seed_everything(self.config.seed + iteration)
            trainer.fit(
                aggregate,
                validation_dataset,
                fit_standardiser=False,
                seed=self.config.seed + iteration,
            )

            record = self._record(
                iteration=iteration,
                beta=beta,
                queries=queries,
                aggregate_samples=len(aggregate),
                validation_dataset=validation_dataset,
                scoring_oracle=scoring_oracle,
                seconds=time.perf_counter() - round_started,
                trainer=trainer,
            )
            history.records.append(record)
            history.total_queries += queries
            if record.closed_loop_tracking_error < history.best_closed_loop_error:
                history.best_closed_loop_error = record.closed_loop_tracking_error
                history.best_iteration = iteration
                best_state = deepcopy(self.policy.state_dict())
            logger.info(
                "dagger round %d/%d beta=%.3f queries=%d closed_loop_te=%.4e (%.1fs)",
                iteration,
                self.config.iterations,
                beta,
                queries,
                record.closed_loop_tracking_error,
                record.seconds,
            )
            if memory_tracker is not None:
                memory_tracker.sample(f"dagger-round-{iteration}")

        # Select the round on closed-loop error, which is what DAgger targets.
        # Round-to-round variation is real — the aggregate changes and the
        # network is retrained each round — so returning the final round would
        # report a draw from that variation rather than the method's result.
        self.policy.load_state_dict(best_state)
        history.total_seconds = time.perf_counter() - started
        return history, aggregate

    def _record(
        self,
        *,
        iteration: int,
        beta: float,
        queries: int,
        aggregate_samples: int,
        validation_dataset: DemonstrationDataset,
        scoring_oracle: ExpertOracle,
        seconds: float,
        trainer: BehaviouralCloningTrainer | None = None,
    ) -> RoundRecord:
        """Score the current policy both closed-loop and teacher-forced."""
        evaluation = self.evaluate_closed_loop(validation_dataset, scoring_oracle)
        scorer = (
            trainer
            if trainer is not None
            else BehaviouralCloningTrainer(self.policy, self.bc_config)
        )
        teacher_forced = scorer.evaluate(validation_dataset)["tracking_error"]
        return RoundRecord(
            iteration=iteration,
            beta=beta,
            queries=queries,
            aggregate_samples=aggregate_samples,
            mean_anchor_drift=evaluation["metrics"]["mean_anchor_drift"],
            closed_loop_tracking_error=evaluation["metrics"][
                "closed_loop_tracking_error"
            ],
            teacher_forced_tracking_error=float(teacher_forced),
            mean_turnover=evaluation["metrics"]["mean_turnover"],
            annualised_net_return=evaluation["metrics"]["annualised_net_return"],
            seconds=seconds,
        )

    # ------------------------------------------------------------------
    # Round steps
    # ------------------------------------------------------------------
    def _collect(
        self,
        dataset: DemonstrationDataset,
        beta: float,
        generator: np.random.Generator,
    ) -> list[PathRollout]:
        """Roll out the beta-mixed policy to collect visited states."""
        return rollout_policy(
            self.policy, dataset, beta=beta, generator=generator
        )

    def _query(
        self,
        rollouts: list[PathRollout],
        oracle: ExpertOracle,
        generator: np.random.Generator,
    ) -> tuple[list[AggregatedRound], int]:
        """Ask the expert what it would do at a sample of the visited states.

        Only a fraction of the visited states is queried. Consecutive states on
        a path are highly redundant — they share all but one day of the return
        window and differ in the anchor by one small trade — so querying every
        one buys little and the quadratic programme is the entire cost of a
        round.
        """
        budget = self.config.max_queries_per_round
        collected: list[AggregatedRound] = []
        total = 0
        for rollout in rollouts:
            if total >= budget:
                break
            count = max(1, int(round(rollout.n_steps * self.config.query_fraction)))
            count = min(count, rollout.n_steps, budget - total)
            rows = np.sort(
                generator.choice(rollout.n_steps, size=count, replace=False)
            )
            anchors = rollout.policy_previous_weights[rows]
            indices = rollout.sample_indices[rows]
            answers = oracle.query_many(rollout.path_id, indices, anchors)
            collected.append(
                AggregatedRound(
                    path_id=rollout.path_id,
                    sample_indices=indices,
                    anchors=anchors,
                    expert_actions=answers,
                )
            )
            total += count
        return collected, total

    def evaluate_closed_loop(
        self, dataset: DemonstrationDataset, oracle: ExpertOracle | None = None
    ) -> dict[str, Any]:
        """Roll the policy out and score it against the expert at its own states."""
        query_oracle = oracle if oracle is not None else ExpertOracle(dataset)
        rollouts = rollout_policy(self.policy, dataset)
        answers = answer_rollouts(query_oracle, rollouts)
        return {
            "rollouts": rollouts,
            "expert_answers": answers,
            "metrics": rollout_metrics(rollouts, answers),
            "per_path": per_path_closed_loop_errors(rollouts, answers),
        }


