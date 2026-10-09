"""Tests for :func:`src.utils.parallel.parallel_map` and its path harness.

The behaviour that matters and must not regress:

* ``workers <= 1`` is a plain in-process loop — no pool, no pickling — so
  existing serial runs are untouched by the harness.
* Output is in input order regardless of which worker finishes first.
* Parallel path generation is **bit-identical** to serial. The acceptance checks
  depend on exact reproducibility, so this is a correctness property, not a
  nicety — it holds because each path seed is ``base_seed + path_id`` with no
  global-RNG read, and because workers pin BLAS to one thread (multi-threaded
  BLAS would reorder float64 reductions and perturb the last bits).
* Workers actually pin their BLAS/torch threads, and the pool uses ``spawn``.
"""

from __future__ import annotations

import multiprocessing
import os
import pickle

import numpy as np
import pytest

from src.dataset.paths import PathData, PathSpec, generate_path_from_spec
from src.simulation.presets import load_simulation_preset
from src.utils.parallel import parallel_map, resolve_workers

# Tiny, fast expert configuration — just enough structure to exercise a real
# path without the cost of the demo-sized runs.
_EXPERT_CONFIG = {
    "horizon": 5,
    "risk_aversion": 5.0,
    "transaction_cost": 0.0025,
    "constraints": {
        "long_only": True,
        "min_weight": 0.0,
        "max_weight": 0.5,
        "leverage_limit": 1.0,
    },
    "moments": {"estimator": "rolling", "covariance": "sample", "window": 30},
}


def _square(value: int) -> int:
    return value * value


def _report_thread_pins(_ignored: int) -> tuple[str, str, int]:
    """Return the thread caps a worker sees, so the initializer can be checked."""
    import torch

    return (
        os.environ.get("OMP_NUM_THREADS", ""),
        os.environ.get("VECLIB_MAXIMUM_THREADS", ""),
        torch.get_num_threads(),
    )


def _report_start_method(_ignored: int) -> str:
    return multiprocessing.get_start_method()


def _make_specs(preset_name: str, n_paths: int) -> list[PathSpec]:
    preset = load_simulation_preset(preset_name, n_assets=3)
    return [
        PathSpec(
            path_id=path_id,
            seed=42 + path_id,
            preset=preset,
            periods=60,
            expert_config=_EXPERT_CONFIG,
        )
        for path_id in range(n_paths)
    ]


def _assert_paths_identical(left: PathData, right: PathData) -> None:
    assert left.path_id == right.path_id
    assert left.seed == right.seed
    for field in (
        "returns",
        "regimes",
        "decision_times",
        "expert_weights",
        "previous_weights",
        "gross_returns",
        "net_returns",
        "turnover",
    ):
        np.testing.assert_array_equal(getattr(left, field), getattr(right, field))


def test_resolve_workers_passes_positive_through():
    assert resolve_workers(1) == 1
    assert resolve_workers(8) == 8


def test_resolve_workers_zero_and_negative_mean_all_cores():
    expected = os.cpu_count() or 1
    assert resolve_workers(0) == expected
    assert resolve_workers(-1) == expected


def test_workers_one_is_a_plain_serial_map():
    values = [1, 2, 3, 4]
    assert parallel_map(_square, values, workers=1) == [v * v for v in values]


def test_empty_input_returns_empty():
    assert parallel_map(_square, [], workers=4) == []


def test_output_preserves_input_order_across_workers():
    values = list(range(10))
    assert parallel_map(_square, values, workers=4) == [v * v for v in values]


def test_progress_callback_fires_once_per_item_serial():
    values = [1, 2, 3, 4]
    seen: list[tuple[int, int]] = []
    parallel_map(_square, values, workers=1, progress_callback=lambda d, t: seen.append((d, t)))
    assert seen == [(1, 4), (2, 4), (3, 4), (4, 4)]


def test_progress_callback_fires_once_per_item_parallel():
    values = list(range(6))
    dones: list[int] = []
    totals: set[int] = set()

    def record(done: int, total: int) -> None:
        dones.append(done)
        totals.add(total)

    parallel_map(_square, values, workers=4, progress_callback=record)
    # Arrival order is nondeterministic, but the callback must fire exactly once
    # per item, and ``done`` must climb from 1 to len(values).
    assert sorted(dones) == [1, 2, 3, 4, 5, 6]
    assert totals == {6}


def test_progress_callback_does_not_disturb_order():
    values = list(range(10))
    result = parallel_map(_square, values, workers=4, progress_callback=lambda d, t: None)
    assert result == [v * v for v in values]


@pytest.mark.parametrize(
    "preset_name", ["canonical_three_regime", "unmatched_three_regime"]
)
def test_parallel_generation_is_bit_identical_to_serial(preset_name):
    specs = _make_specs(preset_name, n_paths=3)
    serial = [generate_path_from_spec(spec) for spec in specs]
    parallel = parallel_map(generate_path_from_spec, specs, workers=3)
    assert [p.path_id for p in parallel] == [p.path_id for p in serial]
    for expected, actual in zip(serial, parallel, strict=True):
        _assert_paths_identical(expected, actual)


def test_results_identical_across_worker_counts():
    specs = _make_specs("canonical_three_regime", n_paths=4)
    baseline = parallel_map(generate_path_from_spec, specs, workers=1)
    for workers in (2, 4):
        other = parallel_map(generate_path_from_spec, specs, workers=workers)
        for expected, actual in zip(baseline, other, strict=True):
            _assert_paths_identical(expected, actual)


def test_worker_pins_blas_and_torch_threads():
    # More than one item so the pool path (not the serial short-circuit) runs.
    pins = parallel_map(_report_thread_pins, [0, 1], workers=2)
    for omp, veclib, torch_threads in pins:
        assert omp == "1"
        assert veclib == "1"
        assert torch_threads == 1


def test_pool_uses_spawn_context():
    methods = parallel_map(_report_start_method, [0, 1], workers=2)
    assert all(method == "spawn" for method in methods)


def test_path_spec_pickle_round_trips():
    (spec,) = _make_specs("canonical_three_regime", n_paths=1)
    restored = pickle.loads(pickle.dumps(spec))
    assert restored.path_id == spec.path_id
    assert restored.seed == spec.seed
    assert restored.periods == spec.periods
    assert restored.expert_config == spec.expert_config
    np.testing.assert_array_equal(
        restored.preset.regime_means, spec.preset.regime_means
    )
