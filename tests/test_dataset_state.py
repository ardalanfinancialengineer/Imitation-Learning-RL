"""Tests for the Equation (6) state representation."""

from __future__ import annotations

import numpy as np
import pytest

from src.dataset.state import StateAssembler, StateSpec


def _assembler(n_assets: int = 4, n_regimes: int = 3, **spec_kwargs) -> StateAssembler:
    return StateAssembler(
        StateSpec(**spec_kwargs),
        n_assets=n_assets,
        n_regimes=n_regimes,
        transaction_cost=0.0025,
        risk_aversion=5.0,
    )


def _components(n_assets: int = 4, n_regimes: int = 3, lookback: int = 20):
    rng = np.random.default_rng(0)
    covariance = np.eye(n_assets) * 1e-4 + 2e-5
    return {
        "return_window": rng.normal(0.0, 0.01, size=(lookback, n_assets)),
        "expected_returns": rng.normal(0.0005, 0.0001, size=n_assets),
        "covariance": covariance,
        "regime_probabilities": np.full(n_regimes, 1.0 / n_regimes),
        "previous_weights": np.full(n_assets, 1.0 / n_assets),
    }


def test_dimension_matches_the_expose_formula():
    """N=50, L=20, K=3 must reproduce the 2380 figure stated in the exposé."""
    spec = StateSpec(lookback=20)
    assert spec.dimension(50, 3) == 20 * 50 + 50 + 50 * 51 // 2 + 3 + 50 + 2 == 2380


def test_block_sizes_sum_to_the_dimension():
    spec = StateSpec(lookback=12)
    sizes = spec.block_sizes(7, 3)
    assert sum(sizes.values()) == spec.dimension(7, 3)
    assert list(sizes) == [
        "return_window",
        "expected_returns",
        "covariance",
        "regime_probabilities",
        "previous_weights",
        "economic_parameters",
    ]


def test_block_slices_are_contiguous_and_ordered():
    spec = StateSpec(lookback=5)
    slices = spec.block_slices(4, 3)
    previous_stop = 0
    for block in slices.values():
        assert block.start == previous_stop
        previous_stop = block.stop
    assert previous_stop == spec.dimension(4, 3)


def test_assembled_state_places_each_block_where_the_slices_say():
    assembler = _assembler()
    components = _components()
    state = assembler.assemble(**components)
    slices = assembler.block_slices()

    assert state.shape == (assembler.dimension,)
    assert np.allclose(
        state[slices["return_window"]], components["return_window"].reshape(-1)
    )
    assert np.allclose(state[slices["expected_returns"]], components["expected_returns"])
    assert np.allclose(
        state[slices["regime_probabilities"]], components["regime_probabilities"]
    )
    assert np.allclose(state[slices["previous_weights"]], components["previous_weights"])
    assert np.allclose(state[slices["economic_parameters"]], [0.0025, 5.0])


def test_vech_block_recovers_the_lower_triangle():
    assembler = _assembler(n_assets=4)
    components = _components(n_assets=4)
    covariance = np.arange(16, dtype=float).reshape(4, 4)
    covariance = covariance + covariance.T
    components["covariance"] = covariance

    state = assembler.assemble(**components)
    encoded = state[assembler.block_slices()["covariance"]]
    assert np.allclose(encoded, covariance[np.tril_indices(4)])
    assert encoded.size == 4 * 5 // 2


def test_compact_encoding_is_linear_in_the_asset_count():
    full = StateSpec(covariance_encoding="vech")
    compact = StateSpec(covariance_encoding="compact")
    assert compact.covariance_size(50) == 56
    assert compact.covariance_size(200) == 206
    assert full.covariance_size(200) == 20100
    assert compact.dimension(200, 3) < full.dimension(200, 3)


def test_compact_encoding_reports_volatilities_and_correlation_summary():
    assembler = _assembler(n_assets=4, covariance_encoding="compact")
    components = _components(n_assets=4)
    volatilities = np.array([0.01, 0.02, 0.03, 0.04])
    correlation = np.full((4, 4), 0.5)
    np.fill_diagonal(correlation, 1.0)
    components["covariance"] = np.outer(volatilities, volatilities) * correlation

    state = assembler.assemble(**components)
    encoded = state[assembler.block_slices()["covariance"]]
    assert np.allclose(encoded[:4], volatilities)
    # Mean, standard deviation, minimum, and maximum off-diagonal correlation.
    assert encoded[4] == pytest.approx(0.5)
    assert encoded[5] == pytest.approx(0.0, abs=1e-12)
    assert encoded[6] == pytest.approx(0.5)
    assert encoded[7] == pytest.approx(0.5)
    assert 0.0 < encoded[8] <= 1.0, "leading eigenvalue share"
    assert 1.0 <= encoded[9] <= 4.0, "effective rank"


def test_optional_blocks_can_be_switched_off():
    assembler = _assembler(
        include_regime_probabilities=False,
        include_previous_weights=False,
        include_economic_parameters=False,
    )
    components = _components()
    state = assembler.assemble(
        return_window=components["return_window"],
        expected_returns=components["expected_returns"],
        covariance=components["covariance"],
    )
    assert state.size == 20 * 4 + 4 + 10
    assert set(assembler.block_slices()) == {
        "return_window",
        "expected_returns",
        "covariance",
    }


def test_rejects_missing_required_blocks():
    assembler = _assembler()
    components = _components()
    components.pop("regime_probabilities")
    with pytest.raises(ValueError):
        assembler.assemble(**components)


def test_rejects_regime_probabilities_that_do_not_sum_to_one():
    assembler = _assembler()
    components = _components()
    components["regime_probabilities"] = np.array([0.5, 0.2, 0.1])
    with pytest.raises(ValueError):
        assembler.assemble(**components)


def test_rejects_wrong_shapes_and_non_finite_values():
    assembler = _assembler()
    components = _components()
    components["expected_returns"] = np.zeros(3)
    with pytest.raises(ValueError):
        assembler.assemble(**components)

    components = _components()
    components["return_window"][0, 0] = np.nan
    with pytest.raises(ValueError):
        assembler.assemble(**components)


def test_rejects_invalid_specifications():
    with pytest.raises(ValueError):
        StateSpec(lookback=0)
    with pytest.raises(ValueError):
        StateSpec(covariance_encoding="pca")
    with pytest.raises(ValueError):
        _assembler(n_assets=0)
