"""Tests for the Section 5.1 data separation protocol."""

from __future__ import annotations

import numpy as np
import pytest

from src.dataset.paths import build_preset
from src.dataset.splits import (
    PathSplit,
    TemporalSplit,
    default_zero_shot_shifts,
    split_paths,
)
from src.simulation.presets import canonical_three_regime_preset


def test_path_split_is_disjoint_and_covers_every_path():
    split = split_paths(20)
    assert set(split.train) | set(split.validation) | set(split.test) == set(range(20))
    assert not set(split.train) & set(split.validation)
    assert not set(split.train) & set(split.test)
    assert not set(split.validation) & set(split.test)


def test_path_split_rejects_overlapping_assignments():
    with pytest.raises(ValueError):
        PathSplit(train=(0, 1), validation=(1, 2), test=(3,))
    with pytest.raises(ValueError):
        PathSplit(train=(0, 0), validation=(1,), test=(2,))


def test_path_split_is_deterministic_without_a_seed():
    assert split_paths(30).to_dict() == split_paths(30).to_dict()


def test_seeded_path_split_still_partitions_every_path():
    split = split_paths(30, seed=7)
    assert set(split.all_ids) == set(range(30))
    assert len(split.train) + len(split.validation) + len(split.test) == 30


def test_every_split_is_non_empty_even_for_the_smallest_dataset():
    split = split_paths(3)
    for name in ("train", "validation", "test"):
        assert split.of(name), f"{name} split is empty"


def test_temporal_segments_are_contiguous_and_ordered():
    segments = TemporalSplit().segments(1000)
    assert segments["train"].start == 0
    assert segments["train"].stop == segments["validation"].start
    assert segments["validation"].stop == segments["test"].start
    assert segments["test"].stop == 1000


def test_temporal_segments_never_interleave_decisions():
    """Later evaluation segments must come strictly after earlier training ones."""
    segments = TemporalSplit().segments(500)
    train = np.arange(500)[segments["train"]]
    test = np.arange(500)[segments["test"]]
    assert train.max() < test.min()


def test_temporal_split_rejects_fractions_leaving_no_test_segment():
    with pytest.raises(ValueError):
        TemporalSplit(train_fraction=0.9, validation_fraction=0.2)


def test_every_zero_shot_shift_changes_the_data_generating_process():
    baseline = canonical_three_regime_preset(6)
    for shift in default_zero_shot_shifts():
        if not shift.preset:
            assert shift.transaction_cost is not None, shift.name
            continue
        shifted = build_preset(shift.preset_config(), 6)
        identical = (
            np.allclose(shifted.transition_matrix, baseline.transition_matrix)
            and np.allclose(shifted.regime_covariances, baseline.regime_covariances)
            and np.allclose(shifted.regime_means, baseline.regime_means)
        )
        assert not identical, f"shift {shift.name} does not change anything"


def test_shifted_presets_remain_valid_simulation_parameters():
    for shift in default_zero_shot_shifts():
        if not shift.preset:
            continue
        preset = build_preset(shift.preset_config(), 5)
        assert np.allclose(preset.transition_matrix.sum(axis=1), 1.0)
        assert np.all(preset.transition_matrix >= 0.0)
        eigenvalues = np.linalg.eigvalsh(preset.regime_covariances)
        assert np.all(eigenvalues > -1e-12), f"{shift.name} covariance is not PSD"


def test_lower_persistence_shift_really_lowers_persistence():
    baseline = canonical_three_regime_preset(4)
    shifted = build_preset({"name": "low", "persistence_scale": 0.75}, 4)
    assert np.all(np.diag(shifted.transition_matrix) < np.diag(baseline.transition_matrix))


def test_build_preset_rejects_unknown_parameters():
    with pytest.raises(ValueError):
        build_preset({"name": "x", "unknown_scale": 2.0}, 4)
    with pytest.raises(ValueError):
        build_preset({"name": "not_a_preset"}, 4)
