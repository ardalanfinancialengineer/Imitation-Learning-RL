"""Tests for the portfolio constraint set used by expert and policies."""

from __future__ import annotations

import numpy as np
import pytest

from src.expert.constraints import PortfolioConstraints


def test_rejects_inconsistent_bounds():
    with pytest.raises(ValueError):
        PortfolioConstraints(min_weight=0.5, max_weight=0.2)


def test_rejects_negative_lower_bound_when_long_only():
    with pytest.raises(ValueError):
        PortfolioConstraints(long_only=True, min_weight=-0.1)


def test_rejects_leverage_below_full_investment():
    with pytest.raises(ValueError):
        PortfolioConstraints(leverage_limit=0.5)


def test_validate_for_rejects_infeasible_cap():
    constraints = PortfolioConstraints(max_weight=0.2)
    constraints.validate_for(5)
    with pytest.raises(ValueError):
        constraints.validate_for(4)


def test_validate_for_rejects_overfunding_floor():
    constraints = PortfolioConstraints(min_weight=0.3, max_weight=0.5)
    with pytest.raises(ValueError):
        constraints.validate_for(4)


def test_projection_respects_budget_and_box():
    constraints = PortfolioConstraints(max_weight=0.4)
    projected = constraints.project(np.array([2.0, -1.0, 0.3, 0.1, 0.5]))
    assert np.isclose(projected.sum(), 1.0)
    assert projected.min() >= -1e-12
    assert projected.max() <= 0.4 + 1e-9
    assert constraints.is_feasible(projected)


def test_projection_is_identity_on_feasible_points():
    constraints = PortfolioConstraints(max_weight=0.5)
    weights = np.array([0.25, 0.25, 0.5])
    assert np.allclose(constraints.project(weights), weights, atol=1e-9)


def test_violations_report_each_broken_constraint():
    constraints = PortfolioConstraints(max_weight=0.4)
    violations = constraints.violations(np.array([0.9, 0.9, -0.9]))
    assert violations["budget"] > 0.0
    assert violations["upper_bound"] > 0.0
    assert violations["lower_bound"] > 0.0


def test_violations_accept_batches():
    constraints = PortfolioConstraints(max_weight=0.6)
    batch = np.array([[0.5, 0.5], [0.6, 0.4]])
    assert constraints.is_feasible(batch)


def test_default_weights_are_feasible():
    constraints = PortfolioConstraints(max_weight=0.3)
    weights = constraints.default_weights(10)
    assert np.isclose(weights.sum(), 1.0)
    assert constraints.is_feasible(weights)


def test_leverage_violation_detected_when_shorting_allowed():
    constraints = PortfolioConstraints(
        long_only=False, min_weight=-1.0, max_weight=1.0, leverage_limit=1.5
    )
    assert constraints.is_feasible(np.array([0.6, 0.6, -0.2]))
    assert not constraints.is_feasible(np.array([1.5, 0.5, -1.0]))
