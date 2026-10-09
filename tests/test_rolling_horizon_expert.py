"""Tests for the rolling-horizon, friction-aware Markowitz expert."""

from __future__ import annotations

import numpy as np
import pytest

from src.expert.constraints import PortfolioConstraints
from src.expert.markowitz import (
    MomentForecast,
    RollingHorizonMarkowitzExpert,
)
from src.expert.moments import RollingMomentForecaster
from src.simulation.presets import canonical_three_regime_preset

HORIZON = 21


def _returns(n_assets: int = 5, periods: int = 500, seed: int = 11) -> np.ndarray:
    preset = canonical_three_regime_preset(n_assets)
    simulated, _ = preset.build_simulator().simulate(T=periods, seed=seed)
    return simulated.to_numpy(dtype=float)


def _expert(**overrides) -> RollingHorizonMarkowitzExpert:
    settings = {
        "horizon": HORIZON,
        "risk_aversion": 20.0,
        "transaction_cost": 0.001,
        "constraints": PortfolioConstraints(max_weight=0.4),
    }
    settings.update(overrides)
    forecaster = settings.pop("forecaster", None) or RollingMomentForecaster(60)
    return RollingHorizonMarkowitzExpert(forecaster, **settings)


def _forecast(n_assets: int, horizon: int = HORIZON, mean_scale: float = 1.0) -> MomentForecast:
    means = np.tile(np.linspace(0.0002, 0.0010, n_assets) * mean_scale, (horizon, 1))
    covariance = np.eye(n_assets) * 1e-4 + 2e-5
    return MomentForecast(means=means, covariances=np.tile(covariance, (horizon, 1, 1)))


def test_solution_is_feasible_and_plans_the_full_horizon():
    expert = _expert()
    solution = expert.solve(_forecast(5), np.full(5, 0.2))

    assert solution.plan.shape == (HORIZON, 5)
    assert solution.status == "optimal"
    assert not solution.used_fallback
    assert expert.constraints.is_feasible(solution.weights)
    assert np.isclose(solution.weights.sum(), 1.0)
    assert solution.weights.max() <= 0.4 + 1e-6
    assert solution.weights.min() >= -1e-9


def test_only_the_first_planned_allocation_is_returned():
    expert = _expert()
    solution = expert.solve(_forecast(5), np.full(5, 0.2))
    assert np.allclose(solution.weights, expert.constraints.project(solution.plan[0]))


def test_all_planned_steps_satisfy_the_constraints():
    expert = _expert()
    plan = expert.solve(_forecast(6), np.full(6, 1 / 6)).plan
    assert expert.constraints.is_feasible(plan, tolerance=1e-5)


def test_transaction_cost_anchors_on_the_previous_portfolio():
    """A large cost must keep the expert close to the portfolio it already holds."""
    forecast = _forecast(5)
    anchor = np.array([0.4, 0.4, 0.2, 0.0, 0.0])

    free = _expert(transaction_cost=0.0).solve(forecast, anchor).weights
    sticky = _expert(transaction_cost=0.05).solve(forecast, anchor).weights

    assert np.abs(sticky - anchor).sum() < np.abs(free - anchor).sum()


def test_higher_cost_reduces_realised_turnover():
    returns = _returns(periods=400)
    turnovers = [
        _expert(transaction_cost=cost).run(returns).turnover.mean()
        for cost in (0.0, 0.001, 0.005)
    ]
    assert turnovers[0] > turnovers[-1]
    assert all(
        earlier >= later - 1e-9 for earlier, later in zip(turnovers, turnovers[1:])
    )


def test_higher_risk_aversion_reduces_forecast_portfolio_variance():
    forecast = MomentForecast(
        means=np.tile(np.array([0.002, 0.0005, 0.0005, 0.0005]), (HORIZON, 1)),
        covariances=np.tile(np.diag([4e-4, 1e-4, 1e-4, 1e-4]), (HORIZON, 1, 1)),
    )
    anchor = np.full(4, 0.25)
    covariance = forecast.covariances[0]

    timid = _expert(risk_aversion=200.0).solve(forecast, anchor).weights
    bold = _expert(risk_aversion=1.0).solve(forecast, anchor).weights
    assert timid @ covariance @ timid < bold @ covariance @ bold


def test_horizon_one_matches_the_single_period_solution():
    """With zero cost the multi-period programme separates across steps."""
    forecast_long = _forecast(4, horizon=HORIZON)
    forecast_short = _forecast(4, horizon=1)
    anchor = np.full(4, 0.25)

    long_run = _expert(transaction_cost=0.0).solve(forecast_long, anchor).weights
    short_run = _expert(horizon=1, transaction_cost=0.0).solve(forecast_short, anchor).weights
    assert np.allclose(long_run, short_run, atol=1e-6)


def test_multi_period_planning_still_trades_where_myopic_planning_freezes():
    """The horizon amortises a one-off cost against H periods of benefit.

    A myopic optimiser compares a full transaction cost against a single day of
    expected improvement, so at the top of the prescribed cost range it stops
    trading almost entirely. The H=21 programme pays the same cost once but
    collects the benefit over the whole horizon, and keeps rebalancing. This is
    the economic reason the exposé specifies a rolling *horizon* expert.
    """
    returns = _returns(periods=400)
    myopic = _expert(horizon=1, transaction_cost=0.005).run(returns)
    planned = _expert(horizon=HORIZON, transaction_cost=0.005).run(returns)

    assert myopic.turnover.mean() < 1e-2, "myopic expert should be nearly frozen"
    assert planned.turnover.mean() > 10.0 * myopic.turnover.mean()


def test_run_reports_no_look_ahead_decisions():
    returns = _returns(periods=300)
    expert = _expert()
    trajectory = expert.run(returns, warmup=60)

    index = trajectory.n_decisions // 2
    t = int(trajectory.decision_times[index])
    replica = expert.clone()
    replica.forecaster.prepare(returns[:t])
    replayed = replica.decide(t, trajectory.previous_weights[index]).weights
    assert np.allclose(replayed, trajectory.weights[index], atol=1e-9)


def test_run_realises_returns_after_the_decision():
    returns = _returns(periods=200)
    expert = _expert(transaction_cost=0.002)
    trajectory = expert.run(returns, warmup=60)

    expected_gross = trajectory.weights[3] @ returns[trajectory.decision_times[3]]
    assert trajectory.gross_returns[3] == pytest.approx(expected_gross)
    expected_net = expected_gross - 0.002 * np.abs(
        trajectory.weights[3] - trajectory.previous_weights[3]
    ).sum()
    assert trajectory.net_returns[3] == pytest.approx(expected_net)


def test_run_chains_the_previous_weights():
    returns = _returns(periods=200)
    trajectory = _expert().run(returns, warmup=60)
    assert np.allclose(trajectory.previous_weights[1:], trajectory.weights[:-1])


def test_run_is_reproducible():
    returns = _returns(periods=250)
    first = _expert().run(returns, warmup=60)
    second = _expert().run(returns, warmup=60)
    assert np.allclose(first.weights, second.weights)


def test_run_records_solver_health():
    returns = _returns(periods=250)
    trajectory = _expert().run(returns, warmup=60)
    assert set(trajectory.statuses) <= {"optimal", "optimal_inaccurate"}
    assert trajectory.fallback_rate == 0.0
    assert np.all(trajectory.solve_seconds >= 0.0)


def test_leverage_constraint_binds_when_shorting_is_allowed():
    constraints = PortfolioConstraints(
        long_only=False, min_weight=-0.5, max_weight=1.0, leverage_limit=1.2
    )
    forecast = _forecast(4, mean_scale=40.0)
    weights = _expert(constraints=constraints, risk_aversion=1.0).solve(
        forecast, np.full(4, 0.25)
    ).weights
    assert np.abs(weights).sum() <= 1.2 + 1e-5
    assert np.isclose(weights.sum(), 1.0, atol=1e-6)


def test_scales_to_a_fifty_asset_universe():
    returns = _returns(n_assets=50, periods=200)
    expert = _expert(
        constraints=PortfolioConstraints(max_weight=0.1),
        forecaster=RollingMomentForecaster(60, covariance="ledoit_wolf"),
    )
    trajectory = expert.run(returns, warmup=60, max_decisions=20)
    assert trajectory.n_decisions == 20
    assert expert.constraints.is_feasible(trajectory.weights)
    assert trajectory.fallback_rate == 0.0


def test_solver_preference_switches_on_problem_size():
    """CLARABEL leads on small problems; OSQP leads once the KKT solve dominates."""
    small = _expert().solve(_forecast(8), np.full(8, 0.125))
    assert small.status == "optimal"

    small_order = _expert()._programme(8)._solver_order
    large_order = _expert()._programme(200)._solver_order
    assert small_order[0] == "CLARABEL"
    assert large_order[0] == "OSQP"
    assert small_order[-1] is None, "the CVXPY default must remain the last resort"


def test_rejects_invalid_configuration():
    with pytest.raises(ValueError):
        _expert(horizon=0)
    with pytest.raises(ValueError):
        _expert(risk_aversion=-1.0)
    with pytest.raises(ValueError):
        _expert(transaction_cost=-0.001)


def test_rejects_mismatched_forecast_horizon():
    with pytest.raises(ValueError):
        _expert().solve(_forecast(4, horizon=5), np.full(4, 0.25))


def test_rejects_warmup_below_the_forecaster_requirement():
    with pytest.raises(ValueError):
        _expert().run(_returns(periods=200), warmup=10)


def test_rejects_paths_shorter_than_the_warmup():
    with pytest.raises(ValueError):
        _expert().run(_returns(periods=50), warmup=60)


def test_falls_back_to_a_feasible_portfolio_when_the_solver_fails(monkeypatch):
    expert = _expert()
    anchor = np.full(5, 0.2)
    programme = expert._programme(5)
    monkeypatch.setattr(programme, "plan", lambda: None)

    solution = expert.solve(_forecast(5), anchor)
    assert solution.used_fallback
    assert expert.constraints.is_feasible(solution.weights)
    assert np.allclose(solution.weights, anchor), "holding costs no turnover"
