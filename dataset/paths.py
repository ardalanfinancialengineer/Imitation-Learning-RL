"""Generation, storage, and reloading of expert trajectory paths.

A path bundles one simulated market realisation with the expert decisions taken
along it. Only the *raw* ingredients are persisted:

- simulated returns and latent regimes;
- the expert's implemented weights, anchors, and realised performance;
- the fitted HMM parameters and filtered probabilities, when the expert uses a
  regime filter.

States are **not** persisted. Assembling Equation (6) for 500 paths of 10,000
steps at ``N = 50`` would need about 48 GB, and the ``vech`` block alone about
26 GB. Storing the ingredients instead costs roughly 9 MB per path, and the
state assembler rebuilds any state on demand. This is exact rather than
approximate: Baum-Welch is deterministic here, so a restored filter path is
precisely what a refit would produce.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..expert.config import build_expert
from ..expert.markowitz import RollingHorizonMarkowitzExpert
from ..expert.moments import RegimeMixtureForecaster
from ..expert.regime_filter import GaussianHMMParameters, RegimeFilterPath
from ..simulation.presets import (
    RegimeSimulationPreset,
    canonical_three_regime_preset,
    shifted_three_regime_preset,
)

PATH_FORMAT_VERSION = 2


@dataclass(frozen=True)
class PathData:
    """One simulated path with its expert trajectory."""

    path_id: int
    seed: int
    returns: np.ndarray
    regimes: np.ndarray
    decision_times: np.ndarray
    expert_weights: np.ndarray
    previous_weights: np.ndarray
    gross_returns: np.ndarray
    net_returns: np.ndarray
    turnover: np.ndarray
    filter_probabilities: np.ndarray | None
    filter_fit_times: np.ndarray | None
    filter_transition_matrices: np.ndarray | None
    filter_means: np.ndarray | None
    filter_covariances: np.ndarray | None
    metadata: dict[str, Any]

    @property
    def n_periods(self) -> int:
        return int(self.returns.shape[0])

    @property
    def n_assets(self) -> int:
        return int(self.returns.shape[1])

    @property
    def n_decisions(self) -> int:
        return int(self.decision_times.shape[0])

    @property
    def n_regimes(self) -> int:
        if self.filter_probabilities is not None:
            return int(self.filter_probabilities.shape[1])
        return int(self.metadata.get("n_regimes", 3))

    def save(self, file_path: Path) -> None:
        """Write the path to a compressed ``.npz`` file."""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {
            "returns": self.returns,
            "regimes": self.regimes,
            "decision_times": self.decision_times,
            "expert_weights": self.expert_weights,
            "previous_weights": self.previous_weights,
            "gross_returns": self.gross_returns,
            "net_returns": self.net_returns,
            "turnover": self.turnover,
        }
        for name in (
            "filter_probabilities",
            "filter_fit_times",
            "filter_transition_matrices",
            "filter_means",
            "filter_covariances",
        ):
            value = getattr(self, name)
            if value is not None:
                arrays[name] = value
        arrays["path_id"] = np.array(self.path_id)
        arrays["seed"] = np.array(self.seed)
        arrays["metadata"] = np.array(json.dumps(self.metadata))
        # numpy's stub declares the second parameter as `allow_pickle: bool`, so
        # it rejects an arbitrary **kwargs of arrays even though that is the
        # documented calling convention.
        np.savez_compressed(file_path, **arrays)  # type: ignore[arg-type]

    @classmethod
    def load(cls, file_path: Path) -> PathData:
        """Read a path written by :meth:`save`."""
        with np.load(file_path, allow_pickle=False) as stored:
            metadata = json.loads(str(stored["metadata"]))
            version = int(metadata.get("format_version", 0))
            if version != PATH_FORMAT_VERSION:
                raise ValueError(
                    f"path file {file_path.name} has format version {version}; "
                    f"expected {PATH_FORMAT_VERSION}."
                )
            return cls(
                path_id=int(stored["path_id"]),
                seed=int(stored["seed"]),
                returns=stored["returns"],
                regimes=stored["regimes"],
                decision_times=stored["decision_times"],
                expert_weights=stored["expert_weights"],
                previous_weights=stored["previous_weights"],
                gross_returns=stored["gross_returns"],
                net_returns=stored["net_returns"],
                turnover=stored["turnover"],
                filter_probabilities=_optional(stored, "filter_probabilities"),
                filter_fit_times=_optional(stored, "filter_fit_times"),
                filter_transition_matrices=_optional(stored, "filter_transition_matrices"),
                filter_means=_optional(stored, "filter_means"),
                filter_covariances=_optional(stored, "filter_covariances"),
                metadata=metadata,
            )

    def filter_path(self) -> RegimeFilterPath | None:
        """Rebuild the stored :class:`RegimeFilterPath`, if the expert used one."""
        if (
            self.filter_probabilities is None
            or self.filter_fit_times is None
            or self.filter_transition_matrices is None
            or self.filter_means is None
            or self.filter_covariances is None
        ):
            return None
        fits = tuple(
            GaussianHMMParameters(
                transition_matrix=self.filter_transition_matrices[index],
                # The initial distribution is not needed to answer forecasts and
                # is not persisted; the filtered probabilities already encode it.
                initial_distribution=np.full(
                    self.filter_means.shape[1], 1.0 / self.filter_means.shape[1]
                ),
                means=self.filter_means[index],
                covariances=self.filter_covariances[index],
                log_likelihood=float("nan"),
                n_iterations=0,
                converged=True,
            )
            for index in range(self.filter_fit_times.shape[0])
        )
        return RegimeFilterPath(
            probabilities=self.filter_probabilities,
            fits=fits,
            fit_times=tuple(int(value) for value in self.filter_fit_times),
            burn_in=int(self.metadata["expert"]["forecaster"].get("burn_in", 0)),
        )

    def build_expert(self) -> RollingHorizonMarkowitzExpert:
        """Rebuild the expert that produced this path, with moments prepared.

        A stored filter path is reattached instead of refitted, which is what
        makes reloading a dataset cheap.
        """
        expert = build_expert(self.metadata["expert_config"], enforce_cost_range=False)
        stored_path = self.filter_path()
        if isinstance(expert.forecaster, RegimeMixtureForecaster) and stored_path is not None:
            expert.forecaster.restore(self.returns, stored_path)
        else:
            expert.forecaster.prepare(self.returns)
        return expert


def generate_path(
    *,
    path_id: int,
    seed: int,
    preset: RegimeSimulationPreset,
    periods: int,
    expert_config: dict[str, Any],
    warmup: int | None = None,
    max_decisions: int | None = None,
) -> PathData:
    """Simulate one market path and roll the expert across it."""
    returns_frame, regimes = preset.build_simulator().simulate(T=periods, seed=seed)
    return build_path_from_returns(
        path_id=path_id,
        seed=seed,
        returns=returns_frame.to_numpy(dtype=float),
        regimes=regimes,
        preset=preset,
        expert_config=expert_config,
        warmup=warmup,
        max_decisions=max_decisions,
    )


@dataclass(frozen=True)
class PathSpec:
    """A self-contained, picklable description of one path to generate.

    Bundles every argument of :func:`generate_path` into one object so a path can
    be handed to a worker process as a single picklable value. It exists for the
    parallel harness (:func:`src.utils.parallel.parallel_map`): a frozen dataclass
    of numpy arrays (via ``preset``) and a plain ``expert_config`` dict survives
    the ``spawn`` pickling round-trip cleanly, which a keyword-only function call
    or a closure would not.
    """

    path_id: int
    seed: int
    preset: RegimeSimulationPreset
    periods: int
    expert_config: dict[str, Any]
    warmup: int | None = None
    max_decisions: int | None = None


def generate_path_from_spec(spec: PathSpec) -> PathData:
    """Generate one path from a :class:`PathSpec`.

    The single-positional-argument shape is what :func:`parallel_map` maps over.
    It forwards to :func:`generate_path` unchanged, so the result is identical to
    calling that function directly with the same fields.
    """
    return generate_path(
        path_id=spec.path_id,
        seed=spec.seed,
        preset=spec.preset,
        periods=spec.periods,
        expert_config=spec.expert_config,
        warmup=spec.warmup,
        max_decisions=spec.max_decisions,
    )


def build_path_from_returns(
    *,
    path_id: int,
    seed: int,
    returns: np.ndarray,
    regimes: np.ndarray,
    preset: RegimeSimulationPreset,
    expert_config: dict[str, Any],
    warmup: int | None = None,
    max_decisions: int | None = None,
) -> PathData:
    """Roll the expert across a supplied return array and package the result.

    Split out from :func:`generate_path` so the no-look-ahead check can rebuild
    a path from modified returns and re-estimate everything from scratch,
    rather than reusing stored fits that could not have changed by construction.
    """
    expert = build_expert(expert_config, enforce_cost_range=False)
    trajectory = expert.run(returns, warmup=warmup, max_decisions=max_decisions)

    filter_path = None
    forecaster = expert.forecaster
    if isinstance(forecaster, RegimeMixtureForecaster):
        filter_path = forecaster.path

    metadata = {
        "format_version": PATH_FORMAT_VERSION,
        "preset": preset.name,
        "regime_names": list(preset.regime_names),
        "n_regimes": preset.n_regimes,
        "n_assets": preset.n_assets,
        "periods": int(returns.shape[0]),
        "expert": expert.describe(),
        "expert_config": expert_config,
    }

    return PathData(
        path_id=path_id,
        seed=seed,
        returns=returns,
        regimes=regimes,
        decision_times=trajectory.decision_times,
        expert_weights=trajectory.weights,
        previous_weights=trajectory.previous_weights,
        gross_returns=trajectory.gross_returns,
        net_returns=trajectory.net_returns,
        turnover=trajectory.turnover,
        filter_probabilities=(
            None if filter_path is None else filter_path.probabilities
        ),
        filter_fit_times=(
            None if filter_path is None else np.asarray(filter_path.fit_times, dtype=int)
        ),
        filter_transition_matrices=(
            None
            if filter_path is None
            else np.stack([fit.transition_matrix for fit in filter_path.fits])
        ),
        filter_means=(
            None if filter_path is None else np.stack([fit.means for fit in filter_path.fits])
        ),
        filter_covariances=(
            None
            if filter_path is None
            else np.stack([fit.covariances for fit in filter_path.fits])
        ),
        metadata=metadata,
    )


def build_preset(preset_config: dict[str, Any], n_assets: int) -> RegimeSimulationPreset:
    """Build a canonical or perturbed preset from configuration."""
    config = dict(preset_config or {})
    name = str(config.pop("name", "canonical_three_regime"))
    if not config:
        if name != "canonical_three_regime":
            raise ValueError(f"unknown preset {name!r} without shift parameters.")
        return canonical_three_regime_preset(n_assets)
    unknown = set(config) - {
        "persistence_scale",
        "correlation_scale",
        "volatility_scale",
        "mean_scale",
    }
    if unknown:
        raise ValueError(f"unknown preset shift parameters: {sorted(unknown)}.")
    return shifted_three_regime_preset(
        n_assets, name=name, **{key: float(value) for key, value in config.items()}
    )


def _optional(stored: Any, name: str) -> np.ndarray | None:
    return stored[name] if name in stored.files else None
