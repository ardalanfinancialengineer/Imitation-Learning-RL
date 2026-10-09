"""Tests for the Section 5.4 paired comparison and bootstrap intervals."""

from __future__ import annotations

import numpy as np
import pytest

from src.imitation.comparison import (
    BootstrapInterval,
    paired_bootstrap,
    per_path_tracking_errors,
    seed_summary,
)


def test_a_clear_difference_yields_an_interval_that_excludes_zero():
    first = {path_id: 1.0 + 0.01 * path_id for path_id in range(10)}
    second = {path_id: 0.5 + 0.01 * path_id for path_id in range(10)}

    interval = paired_bootstrap(first, second)
    assert interval.mean == pytest.approx(0.5)
    assert interval.excludes_zero
    assert interval.low > 0.0


def test_noise_around_zero_yields_an_interval_that_straddles_it():
    """The point of the interval: a coin-flip difference must not read as real."""
    generator = np.random.default_rng(0)
    first = {index: float(value) for index, value in enumerate(generator.normal(size=40))}
    second = {index: float(value) for index, value in enumerate(generator.normal(size=40))}

    interval = paired_bootstrap(first, second)
    assert not interval.excludes_zero
    assert interval.low < 0.0 < interval.high


def test_the_difference_is_taken_path_by_path_before_averaging():
    """Pairing is the whole point; averaging first would discard it."""
    # Both have the same mean, but the per-path difference is a constant +1.
    first = {0: 10.0, 1: 2.0}
    second = {0: 9.0, 1: 1.0}
    assert paired_bootstrap(first, second).mean == pytest.approx(1.0)


def test_only_shared_paths_are_compared():
    first = {0: 1.0, 1: 2.0, 2: 3.0}
    second = {1: 1.0, 2: 1.0, 9: 5.0}
    interval = paired_bootstrap(first, second)
    assert interval.n_units == 2
    assert interval.mean == pytest.approx(1.5)


def test_a_single_path_reports_no_interval_rather_than_a_fabricated_one():
    """One path carries no information about sampling variation."""
    interval = paired_bootstrap({0: 2.0}, {0: 1.0})
    assert interval.mean == pytest.approx(1.0)
    assert np.isnan(interval.low) and np.isnan(interval.high)
    assert not interval.excludes_zero


def test_the_interval_is_reproducible_and_widens_at_a_higher_level():
    generator = np.random.default_rng(1)
    first = {index: float(value) for index, value in enumerate(generator.normal(size=30))}
    second = {index: 0.0 for index in range(30)}

    narrow = paired_bootstrap(first, second, level=0.80, seed=3)
    wide = paired_bootstrap(first, second, level=0.99, seed=3)
    repeat = paired_bootstrap(first, second, level=0.80, seed=3)

    assert (narrow.low, narrow.high) == (repeat.low, repeat.high)
    assert wide.low < narrow.low and wide.high > narrow.high


def test_paired_bootstrap_validates_its_inputs():
    with pytest.raises(ValueError, match="share no paths"):
        paired_bootstrap({0: 1.0}, {1: 1.0})
    with pytest.raises(ValueError, match="level"):
        paired_bootstrap({0: 1.0, 1: 2.0}, {0: 0.0, 1: 0.0}, level=1.0)
    with pytest.raises(ValueError, match="resamples"):
        paired_bootstrap({0: 1.0, 1: 2.0}, {0: 0.0, 1: 0.0}, resamples=1)


def test_interval_serialises_for_the_report():
    interval = BootstrapInterval(
        mean=1.0, low=0.5, high=1.5, level=0.95, n_units=7, n_resamples=100
    )
    payload = interval.to_dict()
    assert payload["n_paths"] == 7
    assert payload["excludes_zero"] is True


def test_seed_summary_reports_spread_and_handles_a_single_seed():
    summary = seed_summary([1.0, 2.0, 3.0])
    assert summary["mean"] == pytest.approx(2.0)
    assert summary["sd"] == pytest.approx(1.0)
    assert summary["n_seeds"] == 3

    single = seed_summary([4.0])
    assert single["sd"] == 0.0 and single["n_seeds"] == 1

    with pytest.raises(ValueError, match="at least one value"):
        seed_summary([])


def test_per_path_errors_are_reported_separately_for_each_path():
    """The pairing unit must be the path, not the pooled sample set."""
    from src.dataset.demonstrations import DemonstrationDataset
    from src.dataset.paths import generate_path
    from src.dataset.state import StateSpec
    from src.expert.constraints import PortfolioConstraints
    from src.policy.reference import PreviousWeightsReference
    from src.simulation.presets import canonical_three_regime_preset

    spec = StateSpec(lookback=10)
    expert_config = {
        "horizon": 5,
        "risk_aversion": 5.0,
        "transaction_cost": 0.0025,
        "constraints": {"long_only": True, "min_weight": 0.0, "max_weight": 0.5},
        "moments": {"estimator": "rolling", "covariance": "ledoit_wolf", "window": 60},
    }
    preset = canonical_three_regime_preset(4)
    paths = [
        generate_path(
            path_id=index,
            seed=500 + index,
            preset=preset,
            periods=240,
            expert_config=expert_config,
        )
        for index in range(3)
    ]
    dataset = DemonstrationDataset(paths, spec=spec)
    reference = PreviousWeightsReference(
        spec,
        n_assets=4,
        n_regimes=3,
        constraints=PortfolioConstraints(long_only=True, max_weight=0.5),
    )

    errors = per_path_tracking_errors(reference, dataset)
    assert set(errors) == {0, 1, 2}
    assert all(value > 0.0 for value in errors.values())
    # Independent realisations, so the per-path values must not be identical.
    assert len(set(round(value, 12) for value in errors.values())) == 3
