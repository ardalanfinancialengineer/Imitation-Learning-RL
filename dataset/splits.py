"""Data separation protocol implementing Section 5.1 of the exposé.

Three evaluation settings are distinguished, and all three are defined here
*before* any policy exists. Fixing the protocol up front is what stops it being
adjusted once results are visible.

**In-distribution holdout** — separation is at the *path* level. Training,
validation, and test paths are disjoint sets of independent Monte Carlo
realisations drawn from the same simulation parameters, so no two splits ever
share a stochastic realisation.

**Temporal holdout** — within each held-out path, later segments are evaluated
using decisions constructed solely from earlier information. Segments are
contiguous and ordered, never interleaved, because shuffling decision dates
across a split would leak information through the return window and through
``w_{t-1}``.

**Distribution shift / zero-shot** — evaluation paths generated from presets the
policy never trained on: altered regime persistence, altered covariance
structure, altered means, and transaction costs at the edges of the prescribed
range. Each shift perturbs one aspect so a result can be attributed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

SPLIT_NAMES = ("train", "validation", "test")


@dataclass(frozen=True)
class PathSplit:
    """Disjoint path-level assignment for the in-distribution setting."""

    train: tuple[int, ...]
    validation: tuple[int, ...]
    test: tuple[int, ...]

    def __post_init__(self) -> None:
        for name in SPLIT_NAMES:
            ids = getattr(self, name)
            if len(set(ids)) != len(ids):
                raise ValueError(f"{name} split contains duplicate path ids.")
        pairs = (
            ("train", "validation"),
            ("train", "test"),
            ("validation", "test"),
        )
        for first, second in pairs:
            overlap = set(getattr(self, first)) & set(getattr(self, second))
            if overlap:
                raise ValueError(
                    f"{first} and {second} splits share path ids {sorted(overlap)}."
                )

    @property
    def all_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self.train + self.validation + self.test))

    def of(self, name: str) -> tuple[int, ...]:
        if name not in SPLIT_NAMES:
            raise ValueError(f"unknown split {name!r}; expected one of {SPLIT_NAMES}.")
        return tuple(getattr(self, name))

    def to_dict(self) -> dict[str, list[int]]:
        return {name: list(self.of(name)) for name in SPLIT_NAMES}


def split_paths(
    n_paths: int,
    *,
    train_fraction: float = 0.7,
    validation_fraction: float = 0.15,
    seed: int | None = None,
) -> PathSplit:
    """Assign independent paths to disjoint train/validation/test sets.

    Assignment is by path index when ``seed`` is ``None``, which keeps the split
    reproducible and inspectable. Paths are independent realisations, so a
    deterministic assignment carries no ordering bias; a seed is offered only
    for robustness checks that want to vary the partition.
    """
    if not isinstance(n_paths, int) or isinstance(n_paths, bool) or n_paths < 3:
        raise ValueError("n_paths must be an integer of at least 3.")
    for value, name in (
        (train_fraction, "train_fraction"),
        (validation_fraction, "validation_fraction"),
    ):
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be positive and finite.")
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError("train and validation fractions must leave a non-empty test set.")

    order = np.arange(n_paths)
    if seed is not None:
        np.random.default_rng(seed).shuffle(order)

    n_train = max(1, int(round(n_paths * train_fraction)))
    n_validation = max(1, int(round(n_paths * validation_fraction)))
    if n_train + n_validation >= n_paths:
        n_train = max(1, n_paths - 2)
        n_validation = 1

    return PathSplit(
        train=tuple(int(value) for value in order[:n_train]),
        validation=tuple(int(value) for value in order[n_train : n_train + n_validation]),
        test=tuple(int(value) for value in order[n_train + n_validation :]),
    )


@dataclass(frozen=True)
class TemporalSplit:
    """Contiguous, ordered segmentation of the decisions within one path."""

    train_fraction: float = 0.7
    validation_fraction: float = 0.15

    def __post_init__(self) -> None:
        for value, name in (
            (self.train_fraction, "train_fraction"),
            (self.validation_fraction, "validation_fraction"),
        ):
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite.")
        if self.train_fraction + self.validation_fraction >= 1.0:
            raise ValueError(
                "train and validation fractions must leave a non-empty test segment."
            )

    def segments(self, n_decisions: int) -> dict[str, slice]:
        """Return the contiguous decision slice for each split."""
        if not isinstance(n_decisions, (int, np.integer)) or n_decisions < 3:
            raise ValueError("n_decisions must be an integer of at least 3.")
        n_train = max(1, int(n_decisions * self.train_fraction))
        n_validation = max(1, int(n_decisions * self.validation_fraction))
        if n_train + n_validation >= n_decisions:
            n_train = max(1, int(n_decisions) - 2)
            n_validation = 1
        return {
            "train": slice(0, n_train),
            "validation": slice(n_train, n_train + n_validation),
            "test": slice(n_train + n_validation, int(n_decisions)),
        }

    def to_dict(self) -> dict[str, float]:
        return {
            "train_fraction": self.train_fraction,
            "validation_fraction": self.validation_fraction,
        }


@dataclass(frozen=True)
class ZeroShotShift:
    """One named distribution-shift evaluation setting."""

    name: str
    description: str
    preset: dict[str, Any] = field(default_factory=dict)
    transaction_cost: float | None = None
    n_assets: int | None = None

    def preset_config(self) -> dict[str, Any]:
        """Return the preset configuration understood by ``build_preset``."""
        return {"name": self.name, **self.preset}

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "preset": dict(self.preset),
            "transaction_cost": self.transaction_cost,
            "n_assets": self.n_assets,
        }


def default_zero_shot_shifts() -> tuple[ZeroShotShift, ...]:
    """Return the shift settings named in Section 5.1 of the exposé.

    Each entry changes exactly one aspect of the data-generating process or the
    experiment configuration, so a drop in performance can be attributed rather
    than merely observed.
    """
    return (
        ZeroShotShift(
            name="low_persistence",
            description="Regimes switch far more often than in training.",
            preset={"persistence_scale": 0.75},
        ),
        ZeroShotShift(
            name="high_correlation",
            description="Cross-asset correlation is markedly higher in every regime.",
            preset={"correlation_scale": 1.6},
        ),
        ZeroShotShift(
            name="high_volatility",
            description="All regime volatilities are scaled up.",
            preset={"volatility_scale": 1.5},
        ),
        ZeroShotShift(
            name="weak_drift",
            description="Regime means are compressed towards zero.",
            preset={"mean_scale": 0.5},
        ),
        ZeroShotShift(
            name="high_cost",
            description="Transaction cost at the top of the prescribed range.",
            transaction_cost=0.005,
        ),
    )


@dataclass(frozen=True)
class SplitProtocol:
    """The complete data separation protocol for one experiment."""

    path_split: PathSplit
    temporal_split: TemporalSplit
    zero_shot_shifts: tuple[ZeroShotShift, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "in_distribution": self.path_split.to_dict(),
            "temporal": self.temporal_split.to_dict(),
            "zero_shot": [shift.to_dict() for shift in self.zero_shot_shifts],
        }
