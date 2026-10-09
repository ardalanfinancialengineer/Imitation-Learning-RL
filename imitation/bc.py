"""Behavioural cloning: supervised imitation of the rolling-horizon expert.

Stage 1 of the exposé's imitation pipeline treats expert imitation as a
supervised problem over ``D_E = {(s_t, w*_t)}``. This module is the training
loop, the evaluation metrics, and the checkpoint format for that stage.

Three parts encode decisions that are easy to get wrong.

**Optional state cache.** Weeks 14-16 established that states are assembled on
access rather than persisted, because materialising the final-scale dataset
would need about 48 GB. Training re-reads the whole dataset every epoch, so
paying that assembly cost repeatedly is wasteful *at development scale* — where
the states do fit in memory. The trainer therefore materialises them once when
the estimated footprint is under an explicit budget, and falls back to lazy
assembly when it is not. The budget is a configuration value rather than a
heuristic, so the same code trains a 10-path demo and a 500-path final run and
the choice is recorded in the report.

**Training-only standardisation.** The input statistics are fitted on the
training split alone and then frozen, including for validation. Fitting them on
the concatenation would leak the validation distribution into the policy's
inputs and quietly inflate every held-out number.

**Model selection on tracking error, not on loss.** Early stopping watches the
validation tracking error of Equation (18), which is the quantity the exposé
actually reports, and which is defined identically for every policy. Watching
the training objective instead would be meaningless for the diffusion policy,
whose objective is a denoising loss on a different scale entirely and is not
comparable with the deterministic policy's.
"""

from __future__ import annotations

import logging
import os
import random
import time
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..dataset.state import StateAssembler, StateSpec
from ..policy.base import AllocationPolicy
from ..utils.device import resolve_device
from ..utils.observability import MemoryTracker
from .losses import LOSS_NAMES, get_loss

#: Bytes per float32 entry, used to size the cache before allocating it.
_BYTES_PER_ENTRY = 4

logger = logging.getLogger(__name__)

#: Tag identifying the mid-training checkpoint format written by ``fit``.
_CHECKPOINT_FORMAT = "bc-fit-checkpoint-v1"


class TrainingDataset(Protocol):
    """What the trainer needs of a dataset.

    :class:`~src.dataset.demonstrations.DemonstrationDataset` satisfies this,
    and so does the DAgger :class:`~src.imitation.dagger.AggregatedDataset`,
    which is not a subclass — it wraps a base dataset and adds the rounds. The
    protocol states the contract explicitly so the two stay interchangeable
    without one inheriting from the other.
    """

    spec: StateSpec
    assembler: StateAssembler

    def __len__(self) -> int: ...

    def sample(self, index: int) -> tuple[np.ndarray, np.ndarray]: ...

    @property
    def state_dimension(self) -> int: ...

    @property
    def n_assets(self) -> int: ...


@dataclass(frozen=True)
class BCConfig:
    """Optimisation, regularisation, and reproducibility settings."""

    epochs: int = 30
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    loss: str = "mse"
    gradient_clip: float = 1.0
    seed: int = 0
    patience: int = 8
    min_delta: float = 0.0
    device: str = "cpu"
    cache_budget_megabytes: float = 512.0
    standardiser_samples: int = 20_000

    def __post_init__(self) -> None:
        for name in ("epochs", "batch_size", "patience"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if self.loss not in LOSS_NAMES:
            raise ValueError(f"unknown loss {self.loss!r}; expected one of {LOSS_NAMES}.")
        for name in ("learning_rate", "cache_budget_megabytes"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive and finite.")
        for name in ("weight_decay", "gradient_clip", "min_delta"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be nonnegative and finite.")
        if (
            not isinstance(self.standardiser_samples, int)
            or self.standardiser_samples < 2
        ):
            raise ValueError("standardiser_samples must be an integer of at least 2.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EpochRecord:
    """One epoch of training, as reported in the history table."""

    epoch: int
    train_loss: float
    validation_loss: float
    validation_tracking_error: float
    validation_absolute_error: float
    seconds: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BCHistory:
    """Outcome of one behavioural-cloning run."""

    records: list[EpochRecord] = field(default_factory=list)
    best_epoch: int = 0
    best_tracking_error: float = float("inf")
    stopped_early: bool = False
    total_seconds: float = 0.0
    cached_states: bool = False
    samples_per_epoch: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "epochs_run": len(self.records),
            "best_epoch": self.best_epoch,
            "best_validation_tracking_error": self.best_tracking_error,
            "stopped_early": self.stopped_early,
            "total_seconds": round(self.total_seconds, 2),
            "cached_states": self.cached_states,
            "samples_per_epoch": self.samples_per_epoch,
            "records": [record.to_dict() for record in self.records],
        }


class TensorPairDataset(Dataset):
    """In-memory ``(state, action)`` pairs, used when the cache fits the budget."""

    def __init__(self, states: torch.Tensor, actions: torch.Tensor) -> None:
        if states.shape[0] != actions.shape[0]:
            raise ValueError("states and actions must have the same number of rows.")
        self.states = states
        self.actions = actions

    def __len__(self) -> int:
        return int(self.states.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.states[index], self.actions[index]


def materialise(
    dataset: TrainingDataset, *, dtype: torch.dtype = torch.float32
) -> tuple[torch.Tensor, torch.Tensor]:
    """Assemble every ``(state, action)`` pair of a dataset into two tensors."""
    states = torch.empty((len(dataset), dataset.state_dimension), dtype=dtype)
    actions = torch.empty((len(dataset), dataset.n_assets), dtype=dtype)
    for index in range(len(dataset)):
        state, action = dataset.sample(index)
        states[index] = torch.as_tensor(state, dtype=dtype)
        actions[index] = torch.as_tensor(action, dtype=dtype)
    return states, actions


def estimated_megabytes(dataset: TrainingDataset) -> float:
    """Return the memory a fully materialised copy of ``dataset`` would need."""
    entries = len(dataset) * (dataset.state_dimension + dataset.n_assets)
    return entries * _BYTES_PER_ENTRY / 1e6


class BehaviouralCloningTrainer:
    """Train an :class:`AllocationPolicy` to reproduce expert allocations."""

    def __init__(self, policy: AllocationPolicy, config: BCConfig | None = None) -> None:
        self.policy = policy
        self.config = config if config is not None else BCConfig()
        self.loss_fn = get_loss(self.config.loss)
        self.device = resolve_device(self.config.device)
        self.policy.to(self.device)
        self._cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit(
        self,
        train_dataset: TrainingDataset,
        validation_dataset: TrainingDataset,
        *,
        fit_standardiser: bool = True,
        seed: int | None = None,
        checkpoint_path: Path | None = None,
        checkpoint_every: int = 0,
        resume: bool = False,
        memory_tracker: MemoryTracker | None = None,
    ) -> BCHistory:
        """Train on ``train_dataset`` and select the model on ``validation_dataset``.

        ``fit_standardiser=False`` keeps the input statistics already in place.
        DAgger needs this: it retrains every round on a dataset that grows with
        the learner's own visited states, and refitting the standardiser each
        round would move the input scale under a warm-started network and make
        rounds incomparable. The statistics stay those of the round-0 training
        split, which is still training-only data.

        Long single trainings can be made survivable. With ``checkpoint_path`` and
        ``checkpoint_every=K`` the trainer writes a checkpoint every K epochs
        (atomically, so an OOM kill mid-write cannot corrupt it). Passing
        ``resume=True`` alongside an existing ``checkpoint_path`` continues from
        the last checkpoint — restoring the optimiser, scheduler, and every RNG
        state so the resumed run is bit-identical to the uninterrupted one from
        that epoch onward. ``memory_tracker`` samples process memory each epoch.
        """
        seed_everything(self.config.seed if seed is None else seed)

        train_source, cached = self._source(train_dataset)
        validation_source, _ = self._source(validation_dataset)

        resuming = bool(resume and checkpoint_path and checkpoint_path.exists())
        # On resume the standardiser statistics are already inside the restored
        # policy buffers; re-fitting them would shift the frozen training-only
        # scale and silently change results.
        if fit_standardiser and not resuming:
            self._fit_standardiser(train_dataset, train_source)

        optimiser = torch.optim.AdamW(
            self.policy.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser, T_max=self.config.epochs
        )
        generator = torch.Generator().manual_seed(self.config.seed)
        loader = DataLoader(
            train_source,
            batch_size=self.config.batch_size,
            shuffle=True,
            generator=generator,
            drop_last=False,
        )

        history = BCHistory(
            cached_states=cached, samples_per_epoch=len(train_dataset)
        )
        best_state = deepcopy(self.policy.state_dict())
        since_improvement = 0
        start_epoch = 1

        if resuming:
            # Restore *after* the fresh seed and after the optimiser, scheduler,
            # and generator are built, so the restored state overwrites — rather
            # than is overwritten by — the fresh construction. This is what makes
            # a resumed run bit-identical from here on.
            start_epoch, since_improvement, best_state = self._restore_checkpoint(
                cast(Path, checkpoint_path),
                optimiser=optimiser,
                scheduler=scheduler,
                generator=generator,
                history=history,
            )
            logger.info(
                "resuming %s from epoch %d/%d (best_te=%.4e)",
                checkpoint_path,
                start_epoch,
                self.config.epochs,
                history.best_tracking_error,
            )

        started = time.perf_counter()

        for epoch in range(start_epoch, self.config.epochs + 1):
            epoch_started = time.perf_counter()
            train_loss = self._train_epoch(loader, optimiser)
            scheduler.step()
            metrics = self.evaluate(validation_dataset, source=validation_source)

            record = EpochRecord(
                epoch=epoch,
                train_loss=train_loss,
                validation_loss=metrics["loss"],
                validation_tracking_error=metrics["tracking_error"],
                validation_absolute_error=metrics["absolute_error"],
                seconds=time.perf_counter() - epoch_started,
            )
            history.records.append(record)
            logger.info(
                "epoch %d/%d train_loss=%.4e val_te=%.4e (%.1fs)",
                epoch,
                self.config.epochs,
                record.train_loss,
                record.validation_tracking_error,
                record.seconds,
            )
            logger.debug(
                "epoch %d val_loss=%.4e val_abs=%.4e",
                epoch,
                record.validation_loss,
                record.validation_absolute_error,
            )
            if memory_tracker is not None:
                memory_tracker.sample(f"bc-epoch-{epoch}")

            if record.validation_tracking_error < history.best_tracking_error - (
                self.config.min_delta
            ):
                history.best_tracking_error = record.validation_tracking_error
                history.best_epoch = epoch
                best_state = deepcopy(self.policy.state_dict())
                since_improvement = 0
            else:
                since_improvement += 1
                if since_improvement >= self.config.patience:
                    history.stopped_early = True
                    break

            if checkpoint_path and checkpoint_every and epoch % checkpoint_every == 0:
                self._write_checkpoint(
                    checkpoint_path,
                    epoch=epoch,
                    optimiser=optimiser,
                    scheduler=scheduler,
                    generator=generator,
                    best_state=best_state,
                    since_improvement=since_improvement,
                    history=history,
                )
                logger.debug("checkpoint written at epoch %d -> %s", epoch, checkpoint_path)

        history.total_seconds = time.perf_counter() - started
        self.policy.load_state_dict(best_state)
        return history

    def _train_epoch(self, loader: DataLoader, optimiser: torch.optim.Optimizer) -> float:
        self.policy.train()
        total = 0.0
        seen = 0
        for states, actions in loader:
            states = states.to(self.device)
            actions = actions.to(self.device)
            optimiser.zero_grad(set_to_none=True)
            loss = self.policy.bc_loss(states, actions, self.loss_fn)
            loss.backward()
            if self.config.gradient_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.config.gradient_clip
                )
            optimiser.step()
            total += float(loss.detach()) * states.shape[0]
            seen += states.shape[0]
        return total / max(seen, 1)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    @torch.no_grad()
    def evaluate(
        self,
        dataset: TrainingDataset,
        *,
        source: Dataset | None = None,
        batch_size: int | None = None,
    ) -> dict[str, float]:
        """Return imitation metrics for ``dataset`` under the current policy.

        ``tracking_error`` is Equation (18) of the exposé,
        ``(1/T) sum_t ||w_t - w*_t||_2^2``. ``teacher_forced_turnover`` is the
        mean L1 distance from the *expert's* ``w_{t-1}``, not from the policy's
        own previous action: behavioural cloning is evaluated one decision at a
        time, so it measures how large a trade the policy proposes, not the
        turnover of a closed-loop rollout.
        """
        if source is None:
            source, _ = self._source(dataset)
        self.policy.eval()
        loader = DataLoader(
            source, batch_size=batch_size or self.config.batch_size, shuffle=False
        )

        previous_slice = dataset.spec.block_slices(
            dataset.n_assets, dataset.assembler.n_regimes
        ).get("previous_weights")

        totals = {
            "loss": 0.0,
            "tracking_error": 0.0,
            "absolute_error": 0.0,
            "teacher_forced_turnover": 0.0,
            "expert_turnover": 0.0,
        }
        worst_violation = 0.0
        seen = 0

        for states, actions in loader:
            states = states.to(self.device)
            actions = actions.to(self.device)
            # No sampling generator: every policy is asked for its deterministic
            # answer. A per-row generator would make the diffusion policy's
            # allocation depend on which batch a state landed in, because
            # `torch.randn` is not invariant to the shape it is asked for — so
            # the final short batch of an epoch would be scored differently from
            # the rest.
            predicted = self.policy.allocate(states)

            count = states.shape[0]
            totals["loss"] += float(self.loss_fn(predicted, actions)) * count
            totals["tracking_error"] += float(
                torch.sum((predicted - actions) ** 2)
            )
            totals["absolute_error"] += float(torch.sum((predicted - actions).abs()))
            if previous_slice is not None:
                previous = states[:, previous_slice]
                totals["teacher_forced_turnover"] += float(
                    torch.sum((predicted - previous).abs())
                )
                totals["expert_turnover"] += float(torch.sum((actions - previous).abs()))
            seen += count

            violations = self.policy.constraints.violations(
                predicted.cpu().numpy().astype(np.float64)
            )
            worst_violation = max(worst_violation, max(violations.values(), default=0.0))

        divisor = max(seen, 1)
        metrics = {name: value / divisor for name, value in totals.items()}
        metrics["feasibility_violation"] = worst_violation
        metrics["n_samples"] = float(seen)
        return metrics

    # ------------------------------------------------------------------
    # Data plumbing
    # ------------------------------------------------------------------
    def _source(self, dataset: TrainingDataset) -> tuple[Dataset, bool]:
        """Return a cached copy of ``dataset`` if it fits the budget, else itself."""
        key = id(dataset)
        if key in self._cache:
            states, actions = self._cache[key]
            return TensorPairDataset(states, actions), True
        if estimated_megabytes(dataset) > self.config.cache_budget_megabytes:
            # Both concrete implementations are torch Datasets; the protocol
            # deliberately states only what the trainer reads.
            return cast(Dataset, dataset), False
        states, actions = materialise(dataset)
        self._cache[key] = (states, actions)
        return TensorPairDataset(states, actions), True

    def _fit_standardiser(
        self, dataset: TrainingDataset, source: Dataset
    ) -> None:
        """Estimate input statistics and any action scale from training samples only."""
        n_available = len(dataset)
        count = min(self.config.standardiser_samples, n_available)
        indices = np.unique(np.linspace(0, n_available - 1, count).round().astype(int))

        if isinstance(source, TensorPairDataset):
            states = source.states[indices]
            actions = source.actions[indices]
        else:
            pairs = [dataset.sample(int(index)) for index in indices]
            states = torch.as_tensor(
                np.stack([pair[0] for pair in pairs]), dtype=torch.float32
            )
            actions = torch.as_tensor(
                np.stack([pair[1] for pair in pairs]), dtype=torch.float32
            )

        self.policy.standardiser.fit(
            states.numpy(),
            spec=dataset.spec,
            n_assets=dataset.n_assets,
            n_regimes=dataset.assembler.n_regimes,
        )
        self.policy.calibrate(states, actions)
        self.policy.to(self.device)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def _write_checkpoint(
        self,
        file_path: Path,
        *,
        epoch: int,
        optimiser: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        generator: torch.Generator,
        best_state: dict[str, Any],
        since_improvement: int,
        history: BCHistory,
    ) -> None:
        """Persist enough to resume ``fit`` from after ``epoch``, written atomically.

        Everything needed for a *bit-identical* continuation is captured: the live
        policy weights, the AdamW moments, the cosine-schedule position, the best
        weights and early-stopping bookkeeping so far, the full history, and all
        four RNG streams (the shuffle generator included). ``bc_config`` is stored
        so a resume against a different configuration can be refused rather than
        silently corrupting the history table.
        """
        file_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": _CHECKPOINT_FORMAT,
            "epoch": epoch,
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": optimiser.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_state": best_state,
            "best_epoch": history.best_epoch,
            "best_tracking_error": history.best_tracking_error,
            "stopped_early": history.stopped_early,
            "since_improvement": since_improvement,
            "records": [record.to_dict() for record in history.records],
            "cached_states": history.cached_states,
            "samples_per_epoch": history.samples_per_epoch,
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "generator": generator.get_state(),
            },
            "bc_config": self.config.to_dict(),
        }
        # Atomic: a kill mid-write leaves the previous checkpoint intact rather
        # than a truncated file the next resume would choke on.
        tmp_path = file_path.with_suffix(file_path.suffix + ".tmp")
        torch.save(payload, tmp_path)
        os.replace(tmp_path, file_path)

    def _restore_checkpoint(
        self,
        file_path: Path,
        *,
        optimiser: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        generator: torch.Generator,
        history: BCHistory,
    ) -> tuple[int, int, dict[str, Any]]:
        """Load a checkpoint into the live objects and report where to continue.

        Returns ``(start_epoch, since_improvement, best_state)``.

        Raises if the checkpoint was written by a different format version or for
        a different :class:`BCConfig`, because continuing either would produce a
        history that is not comparable with the interrupted run.
        """
        checkpoint = torch.load(file_path, weights_only=False)
        if checkpoint.get("format") != _CHECKPOINT_FORMAT:
            raise ValueError(
                f"checkpoint {file_path} has format {checkpoint.get('format')!r}, "
                f"expected {_CHECKPOINT_FORMAT!r}"
            )
        if checkpoint.get("bc_config") != self.config.to_dict():
            raise ValueError(
                f"checkpoint {file_path} was written for a different BCConfig; "
                "resuming would corrupt the history table."
            )

        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.policy.to(self.device)
        optimiser.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        rng = checkpoint["rng"]
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(_as_byte_tensor(rng["torch"]))
        generator.set_state(_as_byte_tensor(rng["generator"]))

        history.records = [EpochRecord(**record) for record in checkpoint["records"]]
        history.best_epoch = checkpoint["best_epoch"]
        history.best_tracking_error = checkpoint["best_tracking_error"]
        history.stopped_early = checkpoint["stopped_early"]
        history.cached_states = checkpoint["cached_states"]
        history.samples_per_epoch = checkpoint["samples_per_epoch"]

        start_epoch = int(checkpoint["epoch"]) + 1
        return start_epoch, int(checkpoint["since_improvement"]), checkpoint["best_state"]

    def save(self, file_path: Path, *, extra: dict[str, Any] | None = None) -> None:
        """Write policy weights, standardiser, and configuration to disk."""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "policy_description": self.policy.describe(),
                "bc_config": self.config.to_dict(),
                "extra": extra or {},
            },
            file_path,
        )


def _as_byte_tensor(state: Any) -> torch.Tensor:
    """Coerce a saved RNG state back to the ``uint8`` CPU tensor torch expects.

    ``torch.save``/``torch.load`` round-trips the byte tensor faithfully, but a
    state that arrives as a numpy array or plain buffer would be rejected by
    ``set_rng_state``; normalising here keeps resume robust to how it was stored.
    """
    tensor = torch.as_tensor(state, dtype=torch.uint8)
    return tensor.cpu()


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and torch so a run is reproducible end to end.

    Call this **before constructing the policy**, not only before training. A
    network's weights are drawn at construction time, so a trainer that seeds
    inside :meth:`BehaviouralCloningTrainer.fit` — as this one does — still
    starts from whatever initialisation the caller happened to produce. Seeding
    twice is harmless; seeding too late silently costs reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - no CUDA in the test environment
        torch.cuda.manual_seed_all(seed)
