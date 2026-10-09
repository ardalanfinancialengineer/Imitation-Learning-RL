"""Tests for path generation, storage, and the lazy demonstration dataset."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from src.dataset.demonstrations import DemonstrationDataset
from src.dataset.paths import PathData, generate_path
from src.dataset.state import StateSpec
from src.simulation.presets import canonical_three_regime_preset

EXPERT_CONFIG = {
    "horizon": 5,
    "risk_aversion": 5.0,
    "transaction_cost": 0.0025,
    "constraints": {"long_only": True, "max_weight": 0.4},
    "moments": {
        "estimator": "regime_mixture",
        "covariance": "ledoit_wolf",
        "window": 60,
        "blend_weight": 0.25,
        "n_regimes": 3,
        "burn_in": 250,
        "refit_every": 250,
    },
}

ROLLING_CONFIG = EXPERT_CONFIG | {
    "moments": {"estimator": "rolling", "covariance": "sample", "window": 60}
}


@pytest.fixture(scope="module")
def path() -> PathData:
    return generate_path(
        path_id=0,
        seed=42,
        preset=canonical_three_regime_preset(4),
        periods=500,
        expert_config=EXPERT_CONFIG,
    )


def test_generated_path_has_consistent_shapes(path):
    assert path.returns.shape == (500, 4)
    assert path.regimes.shape == (500,)
    assert path.expert_weights.shape == (path.n_decisions, 4)
    assert path.previous_weights.shape == path.expert_weights.shape
    assert path.decision_times.shape == (path.n_decisions,)
    assert path.filter_probabilities is not None
    assert path.filter_probabilities.shape == (500, 3)


def test_expert_actions_lie_on_the_simplex(path):
    assert np.allclose(path.expert_weights.sum(axis=1), 1.0, atol=1e-6)
    assert path.expert_weights.min() >= -1e-9
    assert path.expert_weights.max() <= 0.4 + 1e-6


def test_save_and_load_roundtrips_every_array(path, tmp_path):
    file_path = tmp_path / "path.npz"
    path.save(file_path)
    reloaded = PathData.load(file_path)

    assert reloaded.path_id == path.path_id
    assert reloaded.seed == path.seed
    for name in (
        "returns",
        "regimes",
        "decision_times",
        "expert_weights",
        "previous_weights",
        "net_returns",
        "turnover",
        "filter_probabilities",
        "filter_fit_times",
        "filter_means",
    ):
        np.testing.assert_array_equal(getattr(reloaded, name), getattr(path, name))
    assert reloaded.metadata["expert"]["horizon"] == 5


def test_load_rejects_an_unknown_format_version(path, tmp_path):
    file_path = tmp_path / "path.npz"
    path.save(file_path)
    stale = PathData.load(file_path)
    stale.metadata["format_version"] = 0
    stale.save(file_path)
    with pytest.raises(ValueError, match="format version"):
        PathData.load(file_path)


def test_restored_expert_reproduces_forecasts_without_refitting(path, tmp_path):
    """A reloaded path must give the same moments as the original."""
    file_path = tmp_path / "path.npz"
    path.save(file_path)

    original = path.build_expert()
    restored = PathData.load(file_path).build_expert()
    t = int(path.decision_times[len(path.decision_times) // 2])

    first = original.forecaster.forecast(t, original.horizon)
    second = restored.forecaster.forecast(t, restored.horizon)
    np.testing.assert_allclose(first.means, second.means, rtol=0, atol=0)
    np.testing.assert_allclose(first.covariances, second.covariances, rtol=0, atol=0)


def test_dataset_assembles_states_of_the_expected_dimension(path):
    spec = StateSpec(lookback=20)
    dataset = DemonstrationDataset([path], spec=spec)
    assert dataset.state_dimension == spec.dimension(4, 3)

    state, action = dataset.sample(0)
    assert state.shape == (dataset.state_dimension,)
    assert action.shape == (4,)
    assert np.all(np.isfinite(state))


def test_dataset_state_window_is_strictly_past(path):
    """The return-window block must equal returns[t - L : t], never touching t."""
    spec = StateSpec(lookback=20)
    dataset = DemonstrationDataset([path], spec=spec)
    entry = dataset.entries[0]
    sample_index = int(entry.indices[10])
    t = int(path.decision_times[sample_index])

    state = dataset.assemble_state(entry, sample_index)
    window = state[dataset.assembler.block_slices()["return_window"]]
    np.testing.assert_array_equal(window, path.returns[t - 20 : t].reshape(-1))


def test_dataset_target_is_the_expert_action_for_the_same_decision(path):
    dataset = DemonstrationDataset([path], spec=StateSpec(lookback=20))
    entry = dataset.entries[0]
    for position in (0, 5, entry.indices.size - 1):
        sample_index = int(entry.indices[position])
        _, action = dataset.sample(position)
        np.testing.assert_array_equal(action, path.expert_weights[sample_index])


def test_dataset_drops_decisions_without_a_full_lookback(path):
    """Short windows are dropped, not zero-padded."""
    dataset = DemonstrationDataset([path], spec=StateSpec(lookback=20))
    entry = dataset.entries[0]
    assert np.all(path.decision_times[entry.indices] >= 20)


def test_dataset_works_with_a_pytorch_dataloader(path):
    dataset = DemonstrationDataset([path], spec=StateSpec(lookback=10))
    loader = DataLoader(dataset, batch_size=8, shuffle=False)
    states, actions = next(iter(loader))

    assert states.shape == (8, dataset.state_dimension)
    assert actions.shape == (8, 4)
    assert states.dtype == torch.float32
    assert torch.allclose(actions.sum(dim=1), torch.ones(8), atol=1e-5)


def test_dataset_honours_a_temporal_decision_slice(path):
    full = DemonstrationDataset([path], spec=StateSpec(lookback=20))
    head = DemonstrationDataset(
        [path], spec=StateSpec(lookback=20), decision_slice=slice(0, 50)
    )
    assert len(head) < len(full)
    assert head.entries[0].indices.max() < 50


def test_dataset_supports_a_non_filtering_expert():
    """A rolling-moment expert has no filter, so the regime block is the prior."""
    rolling_path = generate_path(
        path_id=1,
        seed=7,
        preset=canonical_three_regime_preset(4),
        periods=200,
        expert_config=ROLLING_CONFIG,
    )
    assert rolling_path.filter_probabilities is None

    dataset = DemonstrationDataset([rolling_path], spec=StateSpec(lookback=20))
    state, _ = dataset.sample(0)
    block = state[dataset.assembler.block_slices()["regime_probabilities"]]
    np.testing.assert_allclose(block, np.full(3, 1 / 3))


def test_dataset_rejects_paths_with_mismatched_asset_counts(path):
    other = generate_path(
        path_id=2,
        seed=9,
        preset=canonical_three_regime_preset(5),
        periods=200,
        expert_config=ROLLING_CONFIG,
    )
    with pytest.raises(ValueError, match="asset count"):
        DemonstrationDataset([path, other])


def test_dataset_rejects_an_empty_path_list():
    with pytest.raises(ValueError):
        DemonstrationDataset([])
