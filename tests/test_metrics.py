"""Unit tests for portfolio evaluation metrics."""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation import metrics
from src.evaluation.metrics import (
    calmar_ratio,
    max_drawdown,
    sharpe_ratio,
    tracking_error,
    turnover,
)


class TestMetrics:
    def test_sharpe_ratio_matches_known_sample_value(self):
        returns = np.array([0.01, 0.02, 0.03])

        result = sharpe_ratio(returns, risk_free=0.0, annualisation=1)

        assert result == pytest.approx(2.0)

    def test_max_drawdown_uses_peak_to_trough_wealth_loss(self):
        cumulative_returns = np.array([0.0, 0.10, -0.10, 0.20])

        result = max_drawdown(cumulative_returns)

        assert result == pytest.approx(1.0 - 0.90 / 1.10)

    def test_calmar_ratio_uses_geometric_annualised_return(self):
        returns = np.array([0.10, -0.10, 0.10])

        result = calmar_ratio(returns, annualisation=3)

        assert result == pytest.approx(0.89)

    def test_turnover_is_mean_one_way_allocation_change(self):
        weights = np.array(
            [
                [0.5, 0.5],
                [0.7, 0.3],
                [0.4, 0.6],
            ]
        )

        result = turnover(weights)

        assert result == pytest.approx(0.25)

    def test_tracking_error_is_mean_l1_allocation_distance(self):
        weights_agent = np.array([[0.5, 0.5], [0.8, 0.2]])
        weights_expert = np.array([[0.4, 0.6], [0.6, 0.4]])

        result = tracking_error(weights_agent, weights_expert)

        assert result == pytest.approx(0.3)

    @pytest.mark.parametrize(
        "metric, values",
        [
            (sharpe_ratio, np.array([[0.01, 0.02]])),
            (calmar_ratio, np.array([])),
            (max_drawdown, np.array([0.0, np.nan])),
        ],
    )
    def test_return_metrics_validate_inputs(self, metric, values):
        with pytest.raises(ValueError):
            metric(values)

    def test_tracking_error_requires_matching_shapes(self):
        with pytest.raises(ValueError):
            tracking_error(np.ones((2, 2)), np.ones((3, 2)))


# ----------------------------------------------------------------------
# Section 5.3 measures added in the Weeks 1-33 audit
# ----------------------------------------------------------------------
def test_annualised_return_scales_the_mean():
    returns = np.array([0.001, 0.002, -0.001, 0.002])
    assert metrics.annualised_return(returns) == pytest.approx(
        float(returns.mean()) * 252
    )


def test_annualised_volatility_scales_by_the_square_root():
    returns = np.array([0.01, -0.02, 0.03, -0.01, 0.02])
    assert metrics.annualised_volatility(returns) == pytest.approx(
        float(returns.std(ddof=1)) * np.sqrt(252)
    )


def test_annualised_volatility_of_a_single_observation_is_zero():
    assert metrics.annualised_volatility(np.array([0.01])) == 0.0


def test_sortino_ignores_upside_variation():
    """Two series with the same downside but different upside must differ."""
    calm_upside = np.array([-0.01, 0.01, 0.01, 0.01])
    wild_upside = np.array([-0.01, 0.05, 0.05, 0.05])
    assert metrics.sortino_ratio(wild_upside) > metrics.sortino_ratio(calm_upside)


def test_sortino_exceeds_sharpe_when_losses_are_rare():
    """The point of the measure: symmetric volatility over-penalises this series."""
    returns = np.array([0.02, 0.03, 0.02, -0.005, 0.03, 0.02])
    assert metrics.sortino_ratio(returns) > metrics.sharpe_ratio(returns)


def test_sortino_matches_its_definition():
    returns = np.array([0.01, -0.02, 0.03, -0.01])
    shortfall = np.minimum(returns, 0.0)
    downside = float(np.sqrt(np.mean(shortfall**2)))
    expected = float(returns.mean()) / downside * np.sqrt(252)
    assert metrics.sortino_ratio(returns) == pytest.approx(expected)


def test_sortino_is_infinite_without_downside():
    assert metrics.sortino_ratio(np.array([0.01, 0.02, 0.03])) == float("inf")


def test_sortino_is_zero_for_a_flat_series():
    assert metrics.sortino_ratio(np.zeros(5)) == 0.0


def test_the_new_measures_reject_bad_annualisation():
    returns = np.array([0.01, 0.02])
    for function in (
        metrics.annualised_return,
        metrics.annualised_volatility,
        metrics.sortino_ratio,
    ):
        with pytest.raises(ValueError):
            function(returns, annualisation=0)


def test_evaluation_package_exposes_the_section_five_three_measures():
    """One import site for Section 5.3, as the package docstring promises."""
    import src.evaluation as evaluation

    for name in (
        "annualised_return",
        "annualised_volatility",
        "sharpe_ratio",
        "sortino_ratio",
        "max_drawdown",
        "turnover",
        "tracking_error",
    ):
        assert hasattr(evaluation, name), name
