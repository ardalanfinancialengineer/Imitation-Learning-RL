"""Tests for the expert acceptance checks."""

from __future__ import annotations

import numpy as np
import pytest

from src.expert.config import build_expert
from src.expert.constraints import PortfolioConstraints
from src.expert.diagnostics import (
    ExpertDiagnosticThresholds,
    equal_weight_trajectory,
    evaluate_expert,
)
from src.simulation.presets import canonical_three_regime_preset

EXPERT_CONFIG = {
    "horizon": 5,
    "risk_aversion": 20.0,
    "transaction_cost": 0.0025,
    "constraints": {"long_only": True, "max_weight": 0.4},
    "moments": {"estimator": "rolling", "covariance": "ledoit_wolf", "window": 60},
}


def _simulate(n_assets: int = 5, periods: int = 300, seed: int = 5):
    preset = canonical_three_regime_preset(n_assets)
    returns, regimes = preset.build_simulator().simulate(T=periods, seed=seed)
    return returns.to_numpy(dtype=float), regimes


def _evaluate(**kwargs):
    values, regimes = _simulate()
    expert = build_expert(EXPERT_CONFIG)
    trajectory = expert.run(values)
    turnover_by_cost = kwargs.pop(
        "turnover_by_cost",
        {
            cost: float(
                build_expert(EXPERT_CONFIG, transaction_cost=cost, enforce_cost_range=False)
                .run(values)
                .turnover.mean()
            )
            for cost in (0.0, 0.005)
        },
    )
    return evaluate_expert(
        expert, values, trajectory, turnover_by_cost=turnover_by_cost, **kwargs
    )


def test_a_correct_expert_passes_every_check():
    diagnostics = _evaluate()
    assert diagnostics.passed, diagnostics.checks
    assert set(diagnostics.checks) == {
        "feasibility",
        "solver_health",
        "no_look_ahead",
        "determinism",
        "transaction_cost_response",
        "runtime",
    }
    assert diagnostics.replication_error <= 1e-9
    assert diagnostics.turnover_reduction > 0.0


def test_report_is_json_serialisable():
    import json

    report = _evaluate().to_dict()
    assert json.loads(json.dumps(report))["passed"] is True


def test_regime_check_is_added_only_when_a_filter_is_used():
    _, regimes = _simulate()
    diagnostics = _evaluate(regimes=regimes)
    assert "regime_filter_accuracy" not in diagnostics.checks
    assert diagnostics.regime_filter_accuracy is None


def test_regime_filter_accuracy_is_scored_for_a_filtering_expert():
    values, regimes = _simulate(n_assets=4, periods=800, seed=42)
    config = EXPERT_CONFIG | {
        "moments": {
            "estimator": "regime_mixture",
            "covariance": "ledoit_wolf",
            "window": 60,
            "burn_in": 250,
            "refit_every": 250,
            "blend_weight": 0.25,
        }
    }
    expert = build_expert(config)
    trajectory = expert.run(values)
    diagnostics = evaluate_expert(
        expert,
        values,
        trajectory,
        turnover_by_cost={0.0: 0.5, 0.005: 0.1},
        regimes=regimes,
        replication_samples=2,
    )
    assert diagnostics.regime_filter_accuracy is not None
    assert diagnostics.checks["regime_filter_accuracy"]


def test_non_monotone_turnover_fails_the_cost_check():
    diagnostics = _evaluate(turnover_by_cost={0.0: 0.05, 0.005: 0.20})
    assert not diagnostics.checks["transaction_cost_response"]
    assert not diagnostics.passed


def test_unreachable_runtime_budget_fails_the_runtime_check():
    diagnostics = _evaluate(
        thresholds=ExpertDiagnosticThresholds(max_median_solve_seconds=1e-9)
    )
    assert not diagnostics.checks["runtime"]


def test_requires_at_least_two_cost_levels():
    with pytest.raises(ValueError):
        _evaluate(turnover_by_cost={0.001: 0.1})


def test_equal_weight_benchmark_matches_a_manual_calculation():
    values, _ = _simulate()
    times = np.arange(60, 100)
    benchmark = equal_weight_trajectory(
        values, times, 0.0025, PortfolioConstraints(max_weight=0.4)
    )
    assert np.allclose(benchmark["weights"], 1.0 / values.shape[1])
    assert np.allclose(benchmark["gross_returns"], values[times].mean(axis=1))
    assert np.allclose(benchmark["turnover"], 0.0)
