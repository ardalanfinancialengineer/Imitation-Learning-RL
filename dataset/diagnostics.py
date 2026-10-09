"""Acceptance checks for the dataset and environment milestone.

The Weeks 14-16 work package requires a test "that no future information is
used in the decision process". That is turned into six deterministic checks:

``state_no_look_ahead``
    rewriting every return *after* a decision date leaves the assembled state
    at that date bit-identical, against a control rebuild that differs only in
    the future. Any leak through the return window, the moment forecast, the
    filtered regime probabilities, or the previous weights fails this;
``state_integrity``
    assembled states match the analytic dimension, are finite, and place each
    block where ``block_slices`` says it does;
``action_validity``
    every expert action lies on the long-only simplex within tolerance;
``split_disjointness``
    train, validation, and test paths are disjoint, temporal segments are
    contiguous and ordered, and each zero-shot shift really differs from the
    training preset;
``environment_consistency``
    replaying the expert's own actions through the environment reproduces the
    stored net returns and turnover exactly;
``store_roundtrip``
    saving and reloading a path reproduces its states exactly.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..reinforcement.environment import PortfolioEnv
from .demonstrations import DemonstrationDataset
from .paths import PathData, build_path_from_returns, build_preset
from .splits import SplitProtocol
from .state import StateSpec


@dataclass(frozen=True)
class DatasetDiagnosticThresholds:
    """Deterministic acceptance thresholds for one dataset build."""

    max_state_difference: float = 0.0
    max_simplex_violation: float = 1e-6
    max_environment_difference: float = 1e-12
    max_roundtrip_difference: float = 0.0

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be nonnegative and finite.")


@dataclass(frozen=True)
class DatasetDiagnostics:
    """Observed-versus-required behaviour of one dataset build."""

    checks: dict[str, bool]
    state_difference: float
    checked_decision_times: tuple[int, ...]
    state_dimension: int
    expected_dimension: int
    block_sizes: dict[str, int]
    simplex_violation: float
    environment_difference: float
    roundtrip_difference: float
    anchor_chain_sensitivity: float
    split_report: dict[str, Any]
    thresholds: DatasetDiagnosticThresholds

    @property
    def passed(self) -> bool:
        return all(self.checks.values())

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable report."""
        return {
            "passed": self.passed,
            "checks": dict(self.checks),
            "state_no_look_ahead": {
                "max_state_difference": float(self.state_difference),
                "checked_decision_times": list(self.checked_decision_times),
            },
            "state_integrity": {
                "dimension": self.state_dimension,
                "expected_dimension": self.expected_dimension,
                "block_sizes": dict(self.block_sizes),
            },
            "action_validity": {"max_simplex_violation": float(self.simplex_violation)},
            "environment_consistency": {
                "max_difference": float(self.environment_difference)
            },
            "store_roundtrip": {"max_state_difference": float(self.roundtrip_difference)},
            "anchor_chain_sensitivity": {
                "max_weight_drift": float(self.anchor_chain_sensitivity),
                "note": (
                    "Reported, not asserted. Two arithmetically equivalent runs of the "
                    "same expert on the same returns drift by this much, because the "
                    "transaction-cost term chains w_t to w_{t-1} and amplifies last-bit "
                    "solver noise. It is the reproducibility limit for stored "
                    "trajectories."
                ),
            },
            "splits": self.split_report,
            "thresholds": asdict(self.thresholds),
        }


def evaluate_dataset(
    dataset: DemonstrationDataset,
    paths: dict[int, PathData],
    protocol: SplitProtocol,
    *,
    tmp_dir: Path,
    n_assets: int,
    preset: Any,
    samples: int = 4,
    thresholds: DatasetDiagnosticThresholds | None = None,
) -> DatasetDiagnostics:
    """Run every acceptance check for one dataset build."""
    limits = thresholds if thresholds is not None else DatasetDiagnosticThresholds()
    if not paths:
        raise ValueError("at least one path is required.")

    reference = paths[next(iter(paths))]
    state_difference, checked = _look_ahead_difference(
        dataset, reference, preset, samples
    )
    dimension_ok, block_sizes = _state_integrity(dataset)
    simplex_violation = _simplex_violation(dataset)
    environment_difference = _environment_difference(reference, dataset.spec)
    roundtrip_difference = _roundtrip_difference(reference, dataset.spec, tmp_dir)
    sensitivity = _anchor_chain_sensitivity(dataset, reference, preset)
    split_report, splits_ok = _split_report(protocol, paths, n_assets)

    checks = {
        "state_no_look_ahead": state_difference <= limits.max_state_difference,
        "state_integrity": dimension_ok,
        "action_validity": simplex_violation <= limits.max_simplex_violation,
        "split_disjointness": splits_ok,
        "environment_consistency": (
            environment_difference <= limits.max_environment_difference
        ),
        "store_roundtrip": roundtrip_difference <= limits.max_roundtrip_difference,
    }

    return DatasetDiagnostics(
        checks=checks,
        state_difference=state_difference,
        checked_decision_times=checked,
        state_dimension=dataset.state_dimension,
        expected_dimension=dataset.spec.dimension(n_assets, reference.n_regimes),
        block_sizes=block_sizes,
        simplex_violation=simplex_violation,
        environment_difference=environment_difference,
        roundtrip_difference=roundtrip_difference,
        anchor_chain_sensitivity=sensitivity,
        split_report=split_report,
        thresholds=limits,
    )


def _look_ahead_difference(
    dataset: DemonstrationDataset,
    path: PathData,
    preset: Any,
    samples: int,
) -> tuple[float, tuple[int, ...]]:
    """Rewrite the future and measure how much the state at ``t`` moves.

    Everything downstream of the returns is rebuilt from scratch on the altered
    array: the HMM is refitted, the expert re-solved, and the state
    reassembled. Reusing the stored fits would make the check vacuous, because
    stored arrays cannot change when the future does.

    The comparison is against a **control** rebuild with the same run length
    and the unaltered future, not against the originally stored path. The
    reason is measured, not defensive: the QP solve is reproducible only to
    about ``1e-14`` (BLAS reductions are not order-deterministic), and the
    transaction-cost term chains ``w_t`` to ``w_{t-1}``, so that last-bit noise
    is amplified to roughly ``1e-5`` after a few hundred decisions. Comparing
    two rebuilds that differ *only* in the future cancels this exactly and
    leaves a bit-identical test of causality. Comparing against the stored path
    instead would force a tolerance loose enough to hide a small real leak.

    The replacement future is the time-reversed opening segment of the same
    path rather than a constant, so it is genuinely different data at a
    realistic scale and keeps the HMM's covariance estimates well conditioned.
    """
    entry = next(item for item in dataset.entries if item.data.path_id == path.path_id)
    if entry.indices.size == 0:
        return 0.0, ()

    positions = np.unique(
        np.linspace(0, entry.indices.size - 1, min(samples, entry.indices.size))
        .round()
        .astype(int)
    )
    worst = 0.0
    checked: list[int] = []
    for position in positions:
        sample_index = int(entry.indices[position])
        t = int(path.decision_times[sample_index])

        altered_returns = path.returns.copy()
        n_after = altered_returns.shape[0] - t
        altered_returns[t:] = path.returns[:n_after][::-1]

        control = rebuild_truncated_path(path, path.returns.copy(), preset, sample_index)
        treatment = rebuild_truncated_path(path, altered_returns, preset, sample_index)

        worst = max(
            worst,
            float(
                np.abs(
                    _state_of(treatment, dataset, sample_index)
                    - _state_of(control, dataset, sample_index)
                ).max()
            ),
        )
        checked.append(t)
    return worst, tuple(checked)


def rebuild_truncated_path(
    path: PathData, returns: np.ndarray, preset: Any, sample_index: int
) -> PathData:
    """Re-derive a path from returns, with run length fixed for comparability.

    Public because the behavioural-cloning milestone runs the same
    control-versus-treatment comparison one layer further out, on the *policy's*
    action rather than on the assembled state. Reimplementing it there would
    risk the two checks silently diverging on the discipline described in
    :func:`_look_ahead_difference`.
    """
    return build_path_from_returns(
        path_id=path.path_id,
        seed=path.seed,
        returns=returns,
        regimes=path.regimes,
        preset=preset,
        expert_config=path.metadata["expert_config"],
        warmup=int(path.decision_times[0]),
        max_decisions=sample_index + 1,
    )


def _state_of(
    path: PathData, dataset: DemonstrationDataset, sample_index: int
) -> np.ndarray:
    rebuilt = DemonstrationDataset([path], spec=dataset.spec)
    return rebuilt.assemble_state(rebuilt.entries[0], sample_index)


def _anchor_chain_sensitivity(
    dataset: DemonstrationDataset, path: PathData, preset: Any
) -> float:
    """Report how far last-bit solver noise propagates along the anchor chain.

    Not a pass/fail check. It quantifies how much two arithmetically equivalent
    runs of the *same* expert on the *same* returns can drift apart, which is
    the practical reproducibility limit for stored expert trajectories.
    """
    entry = next(item for item in dataset.entries if item.data.path_id == path.path_id)
    if entry.indices.size == 0:
        return 0.0
    sample_index = int(entry.indices[-1])
    control = rebuild_truncated_path(path, path.returns.copy(), preset, sample_index)
    return float(
        np.abs(
            control.expert_weights[: sample_index + 1]
            - path.expert_weights[: sample_index + 1]
        ).max()
    )


def _state_integrity(dataset: DemonstrationDataset) -> tuple[bool, dict[str, int]]:
    """Check dimension, finiteness, and block placement of assembled states."""
    block_sizes = dataset.assembler.spec.block_sizes(
        dataset.assembler.n_assets, dataset.assembler.n_regimes
    )
    slices = dataset.assembler.block_slices()
    state, _ = dataset.sample(0)

    if state.shape != (dataset.state_dimension,) or not np.all(np.isfinite(state)):
        return False, block_sizes
    if sum(block_sizes.values()) != dataset.state_dimension:
        return False, block_sizes

    if "regime_probabilities" in slices:
        probabilities = state[slices["regime_probabilities"]]
        if abs(float(probabilities.sum()) - 1.0) > 1e-6 or np.any(probabilities < -1e-12):
            return False, block_sizes
    if "economic_parameters" in slices:
        parameters = state[slices["economic_parameters"]]
        expected = np.array(
            [dataset.assembler.transaction_cost, dataset.assembler.risk_aversion]
        )
        if not np.allclose(parameters, expected):
            return False, block_sizes
    return True, block_sizes


def _simplex_violation(dataset: DemonstrationDataset) -> float:
    actions = dataset.actions()
    budget = float(np.abs(actions.sum(axis=1) - 1.0).max())
    negativity = float(max(0.0, -actions.min()))
    return max(budget, negativity)


def _environment_difference(path: PathData, spec: StateSpec) -> float:
    """Replay the expert through the environment and compare realised outcomes."""
    environment = PortfolioEnv(path, spec=spec)
    environment.reset()

    worst = 0.0
    for _ in range(environment.n_steps):
        action = environment.expert_action()
        _, _, terminated, _, info = environment.step(action)
        sample_index = int(info["sample_index"])
        worst = max(
            worst,
            abs(info["net_return"] - float(path.net_returns[sample_index])),
            abs(info["turnover"] - float(path.turnover[sample_index])),
        )
        if terminated:
            break
    return worst


def _roundtrip_difference(path: PathData, spec: StateSpec, tmp_dir: Path) -> float:
    """Save, reload, and compare assembled states."""
    tmp_dir.mkdir(parents=True, exist_ok=True)
    file_path = tmp_dir / f"roundtrip_path_{path.path_id}.npz"
    path.save(file_path)
    reloaded = PathData.load(file_path)

    original = DemonstrationDataset([path], spec=spec)
    restored = DemonstrationDataset([reloaded], spec=spec)
    if len(original) != len(restored):
        return float("inf")

    worst = 0.0
    for index in np.unique(np.linspace(0, len(original) - 1, 5).round().astype(int)):
        first, first_action = original.sample(int(index))
        second, second_action = restored.sample(int(index))
        worst = max(
            worst,
            float(np.abs(first - second).max()),
            float(np.abs(first_action - second_action).max()),
        )
    return worst


def _split_report(
    protocol: SplitProtocol,
    paths: dict[int, PathData],
    n_assets: int,
) -> tuple[dict[str, Any], bool]:
    """Verify path disjointness, temporal ordering, and shift distinctness."""
    report: dict[str, Any] = protocol.to_dict()
    split = protocol.path_split

    train, validation, test = (
        set(split.train),
        set(split.validation),
        set(split.test),
    )
    disjoint = not (train & validation or train & test or validation & test)
    non_empty = bool(train and validation and test)
    known = set(paths)
    covered = train | validation | test
    consistent = covered <= known

    reference = paths[next(iter(paths))]
    segments = protocol.temporal_split.segments(reference.n_decisions)
    ordered = (
        segments["train"].stop == segments["validation"].start
        and segments["validation"].stop == segments["test"].start
        and segments["test"].stop == reference.n_decisions
        and segments["train"].start == 0
    )
    report["temporal_segments"] = {
        name: [int(value.start), int(value.stop)] for name, value in segments.items()
    }

    baseline = build_preset({"name": "canonical_three_regime"}, n_assets)
    shifts_differ = True
    shift_details = []
    for shift in protocol.zero_shot_shifts:
        if not shift.preset:
            shift_details.append({"name": shift.name, "differs": True, "aspect": "cost"})
            continue
        shifted = build_preset(shift.preset_config(), n_assets)
        differs = not (
            np.allclose(shifted.transition_matrix, baseline.transition_matrix)
            and np.allclose(shifted.regime_covariances, baseline.regime_covariances)
            and np.allclose(shifted.regime_means, baseline.regime_means)
        )
        shifts_differ = shifts_differ and differs
        shift_details.append(
            {"name": shift.name, "differs": differs, "aspect": "data_generating_process"}
        )
    report["zero_shot_checks"] = shift_details
    report["path_split_disjoint"] = disjoint

    return report, bool(disjoint and non_empty and consistent and ordered and shifts_differ)
