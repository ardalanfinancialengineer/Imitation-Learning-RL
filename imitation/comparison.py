"""Paired comparison of policies with bootstrap confidence intervals.

Section 5.4 of the exposé is explicit about how strategies are to be compared:
"all strategies are evaluated on the same simulated return paths, and their
performance is compared path by path", with bootstrap confidence intervals
[Efron and Tibshirani, 1993] assessing how stable the differences are.

A bare inequality between two mean tracking errors does not do that. Two
policies differing by a few per cent on one seed and one draw of test paths are
not distinguishable, and reporting the winner without an interval invites a
conclusion the data does not support. This module supplies the missing piece:

``per_path_tracking_errors``
    Equation (18) evaluated separately on each held-out path, which is the
    unit the pairing is done over.
``paired_bootstrap``
    A percentile bootstrap over paths for the *difference* between two policies
    evaluated on those same paths. Resampling paths rather than individual
    decisions is deliberate: decisions within a path are strongly dependent —
    consecutive states share a return window and are chained through
    ``w_{t-1}`` — so treating them as independent would understate the interval
    by roughly the square root of the decisions per path.

Two sources of variation are kept separate. The bootstrap covers sampling
variation across evaluation paths at a fixed seed. Repeating training under
several seeds covers optimisation variation. Both are reported, because a
difference that survives one and not the other is not a finding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..dataset.demonstrations import DemonstrationDataset
from ..policy.base import AllocationPolicy
from ..policy.reference import ReferencePolicy

#: Bootstrap resamples. Enough that the reported percentile bounds are stable
#: to the third decimal at the sizes used here.
DEFAULT_RESAMPLES = 2000


@dataclass(frozen=True)
class BootstrapInterval:
    """A percentile bootstrap interval for a paired mean difference."""

    mean: float
    low: float
    high: float
    level: float
    n_units: int
    n_resamples: int

    @property
    def excludes_zero(self) -> bool:
        """Whether the interval lies entirely on one side of zero."""
        return self.low > 0.0 or self.high < 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean": float(self.mean),
            "low": float(self.low),
            "high": float(self.high),
            "level": float(self.level),
            "n_paths": int(self.n_units),
            "n_resamples": int(self.n_resamples),
            "excludes_zero": self.excludes_zero,
        }


def per_path_tracking_errors(
    policy: AllocationPolicy | ReferencePolicy,
    dataset: DemonstrationDataset,
) -> dict[int, float]:
    """Return Equation (18) evaluated separately on each path of ``dataset``.

    Accepts both learned policies and the non-learned reference rules, so the
    same pairing machinery compares a policy against another policy or against
    a baseline.
    """
    errors: dict[int, float] = {}
    for entry in dataset.entries:
        states = np.stack(
            [dataset.assemble_state(entry, int(index)) for index in entry.indices]
        )
        expert = entry.data.expert_weights[entry.indices]
        predicted = np.asarray(policy.act(states), dtype=np.float64)
        errors[entry.data.path_id] = float(
            np.mean(np.sum((predicted - expert) ** 2, axis=1))
        )
    return errors


def paired_bootstrap(
    first: dict[int, float],
    second: dict[int, float],
    *,
    level: float = 0.95,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
) -> BootstrapInterval:
    """Bootstrap the mean of ``first - second`` over the paths they share.

    Both arguments map path id to that path's tracking error, so the difference
    is taken path by path before any averaging — the pairing Section 5.4 asks
    for. A positive interval means ``first`` is worse.
    """
    shared = sorted(set(first) & set(second))
    if not shared:
        raise ValueError("the two results share no paths.")
    if not 0.0 < level < 1.0:
        raise ValueError("level must lie strictly between 0 and 1.")
    if not isinstance(resamples, int) or resamples < 2:
        raise ValueError("resamples must be an integer of at least 2.")

    differences = np.array([first[path_id] - second[path_id] for path_id in shared])
    generator = np.random.default_rng(seed)

    if differences.size == 1:
        # A single path carries no information about sampling variation; report
        # the point estimate with an undefined interval rather than a fake one.
        value = float(differences[0])
        return BootstrapInterval(
            mean=value,
            low=float("nan"),
            high=float("nan"),
            level=level,
            n_units=1,
            n_resamples=0,
        )

    indices = generator.integers(0, differences.size, size=(resamples, differences.size))
    means = differences[indices].mean(axis=1)
    tail = 0.5 * (1.0 - level)
    return BootstrapInterval(
        mean=float(differences.mean()),
        low=float(np.quantile(means, tail)),
        high=float(np.quantile(means, 1.0 - tail)),
        level=level,
        n_units=differences.size,
        n_resamples=resamples,
    )


def seed_summary(values: list[float]) -> dict[str, float]:
    """Summarise a metric across training seeds.

    The spread here is optimisation variation, which is a different thing from
    the bootstrap's sampling variation and is reported alongside it rather than
    pooled into it.
    """
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        raise ValueError("at least one value is required.")
    return {
        "mean": float(array.mean()),
        "sd": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "min": float(array.min()),
        "max": float(array.max()),
        "n_seeds": int(array.size),
    }
