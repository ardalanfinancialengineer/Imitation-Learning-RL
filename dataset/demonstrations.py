"""Lazily assembled demonstration dataset for imitation learning.

States are built on access rather than stored. The dataset holds one
:class:`PathData` per path plus the rebuilt expert forecaster, and assembles
Equation (6) for a sample only when that sample is requested. See
``src/dataset/paths.py`` for why: materialising the final-scale dataset would
need roughly 48 GB.

Sample ``i`` of a path corresponds to decision index ``t = decision_times[i]``.
The state uses ``returns[:t]`` and the target action is the expert's
implemented ``w*_t``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from ..expert.moments import MomentForecaster
from .paths import PathData
from .state import StateAssembler, StateSpec


@dataclass(frozen=True)
class PathEntry:
    """One path with everything needed to assemble its states."""

    data: PathData
    forecaster: MomentForecaster
    horizon: int
    indices: np.ndarray


class DemonstrationDataset(Dataset):
    """(state, expert action) pairs assembled on demand from stored paths."""

    def __init__(
        self,
        paths: Sequence[PathData],
        *,
        spec: StateSpec | None = None,
        decision_slice: slice | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if not paths:
            raise ValueError("at least one path is required.")

        self.spec = spec if spec is not None else StateSpec()
        self.dtype = dtype
        self._entries: list[PathEntry] = []
        self._index: list[tuple[int, int]] = []

        first = paths[0]
        expert_settings = first.metadata["expert"]
        self.assembler = StateAssembler(
            self.spec,
            n_assets=first.n_assets,
            n_regimes=first.n_regimes,
            transaction_cost=float(expert_settings["transaction_cost"]),
            risk_aversion=float(expert_settings["risk_aversion"]),
        )

        for entry_index, path in enumerate(paths):
            if path.n_assets != first.n_assets:
                raise ValueError("all paths must share the same asset count.")
            expert = path.build_expert()
            usable = self._usable_indices(path, decision_slice)
            if usable.size == 0:
                continue
            self._entries.append(
                PathEntry(
                    data=path,
                    forecaster=expert.forecaster,
                    horizon=expert.horizon,
                    indices=usable,
                )
            )
            self._index.extend((entry_index, int(i)) for i in usable)

        if not self._index:
            raise ValueError(
                "no usable samples: every decision falls before the state lookback."
            )

    def _usable_indices(self, path: PathData, decision_slice: slice | None) -> np.ndarray:
        """Return sample indices whose full return window is available.

        A decision at ``t`` needs ``returns[t - lookback : t]``, so decisions
        earlier than the lookback are dropped rather than zero-padded. Padding
        would inject a constant the policy could learn to key on.
        """
        candidates = np.arange(path.n_decisions)
        if decision_slice is not None:
            candidates = candidates[decision_slice]
        times = path.decision_times[candidates]
        return candidates[times >= self.spec.lookback]

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        state, action = self.sample(index)
        return (
            torch.as_tensor(state, dtype=self.dtype),
            torch.as_tensor(action, dtype=self.dtype),
        )

    def sample(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Return one ``(state, expert action)`` pair as NumPy arrays."""
        entry_index, sample_index = self._index[index]
        entry = self._entries[entry_index]
        return (
            self.assemble_state(entry, sample_index),
            entry.data.expert_weights[sample_index].copy(),
        )

    def assemble_state(
        self,
        entry: PathEntry,
        sample_index: int,
        *,
        previous_weights: np.ndarray | None = None,
    ) -> np.ndarray:
        """Assemble the state for one decision of one path.

        ``previous_weights`` substitutes ``w_{t-1}`` instead of reading the
        expert's own anchor. DAgger needs this: at decision ``t`` the market
        blocks — return window, ``mu_hat``, ``vech(Sigma_hat)``, ``p_hat_t`` —
        depend only on the path and are identical whoever is trading, so a
        state the *learner* visited is this state with one block swapped.
        Leaving it ``None`` reproduces the expert's own state exactly.
        """
        path = entry.data
        t = int(path.decision_times[sample_index])
        forecast = entry.forecaster.forecast(t, entry.horizon)

        probabilities = None
        if self.spec.include_regime_probabilities:
            if path.filter_probabilities is None:
                # A non-filtering expert still needs a regime block of the right
                # width; the uninformative prior is the honest stand-in.
                probabilities = np.full(path.n_regimes, 1.0 / path.n_regimes)
            else:
                probabilities = path.filter_probabilities[t - 1]

        anchor = (
            path.previous_weights[sample_index]
            if previous_weights is None
            else np.asarray(previous_weights, dtype=float)
        )
        return self.assembler.assemble(
            return_window=path.returns[t - self.spec.lookback : t],
            expected_returns=forecast.means[0],
            covariance=forecast.covariances[0],
            regime_probabilities=probabilities,
            previous_weights=anchor,
        )

    def state_at(self, path_id: int, sample_index: int) -> np.ndarray:
        """Assemble a state by path id, for diagnostics and no-look-ahead checks."""
        for entry in self._entries:
            if entry.data.path_id == path_id:
                return self.assemble_state(entry, sample_index)
        raise KeyError(f"path {path_id} is not part of this dataset.")

    @property
    def entries(self) -> tuple[PathEntry, ...]:
        """Return the per-path entries, for callers that drive assembly directly."""
        return tuple(self._entries)

    @property
    def path_ids(self) -> tuple[int, ...]:
        return tuple(entry.data.path_id for entry in self._entries)

    @property
    def state_dimension(self) -> int:
        return self.assembler.dimension

    @property
    def n_assets(self) -> int:
        return self.assembler.n_assets

    def actions(self) -> np.ndarray:
        """Return every expert action in this dataset, stacked."""
        return np.concatenate(
            [entry.data.expert_weights[entry.indices] for entry in self._entries]
        )

    def describe(self) -> dict[str, Any]:
        """Return a JSON-serialisable description for experiment metadata."""
        return {
            "n_samples": len(self),
            "n_paths": len(self._entries),
            "path_ids": list(self.path_ids),
            "state": self.assembler.describe(),
            "samples_per_path": {
                str(entry.data.path_id): int(entry.indices.size) for entry in self._entries
            },
        }
