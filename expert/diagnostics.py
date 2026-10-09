"""Acceptance checks for the rolling-horizon Markowitz expert.

The Weeks 9-13 work package requires evidence that "the optimisation runs
correctly and produces feasible portfolios". This module turns that into six
deterministic, machine-checkable properties of one expert configuration:

``feasibility``
    every executed allocation satisfies budget, box, and leverage constraints;
``solver_health``
    the solver reports an optimal status and the feasible fallback is never
    needed more often than the configured tolerance;
``no_look_ahead``
    re-deciding at a sampled date with the history *truncated at that date*
    reproduces the executed weights exactly;
``determinism``
    solving the same programme twice returns the same allocation;
``transaction_cost_response``
    average turnover falls as the proportional cost rises, which is the
    economic signature of the cost term actually entering the objective;
``runtime``
    the median solve time stays inside the configured budget.

A regime-filter accuracy check is added when simulator regime labels are
supplied. Those labels are used for evaluation only and never enter the
expert's information set.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .constraints import PortfolioConstraints
from .markowitz import ExpertTrajectory, RollingHorizonMarkowitzExpert


@dataclass(frozen=True)
class ExpertDiagnosticThresholds:
    """Deterministic acceptance thresholds for one expert configuration."""

    max_constraint_violation: float = 1e-6
    max_fallback_rate: float = 0.0
    max_replication_error: float = 1e-9
    max_determinism_error: float = 1e-9
    min_turnover_reduction: float = 1e-4
    max_median_solve_seconds: float = 0.5
    min_regime_filter_accuracy: float = 0.45

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be nonnegative and finite.")


@dataclass(frozen=True)
class ExpertDiagnostics:
    """Observed-versus-required behaviour of one expert configuration."""

    checks: dict[str, bool]
    constraint_violations: dict[str, float]
    fallback_rate: float
    replication_error: float
    replication_times: tuple[int, ...]
    determinism_error: float
    turnover_by_cost: dict[float, float]
    turnover_reduction: float
    median_solve_seconds: float
    max_solve_seconds: float
    statuses: tuple[str, ...]
    regime_filter_accuracy: float | None
    horizon_turnover: dict[int, float]
    thresholds: ExpertDiagnosticThresholds

    @property
    def passed(self) -> bool:
        return all(self.checks.values())

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable report."""
        return {
            "passed": self.passed,
            "checks": dict(self.checks),
            "constraint_violations": {
                name: float(size) for name, size in self.constraint_violations.items()
            },
            "solver": {
                "fallback_rate": float(self.fallback_rate),
                "median_solve_seconds": float(self.median_solve_seconds),
                "max_solve_seconds": float(self.max_solve_seconds),
                "statuses": list(self.statuses),
            },
            "no_look_ahead": {
                "max_weight_difference": float(self.replication_error),
                "checked_decision_times": list(self.replication_times),
            },
            "determinism": {"max_weight_difference": float(self.determinism_error)},
            "transaction_cost_response": {
                "mean_turnover_by_cost": {
                    f"{cost:.4f}": float(turnover)
                    for cost, turnover in sorted(self.turnover_by_cost.items())
                },
                "turnover_reduction": float(self.turnover_reduction),
            },
            "horizon_response": {
                "mean_turnover_by_horizon": {
                    str(horizon): float(turnover)
                    for horizon, turnover in sorted(self.horizon_turnover.items())
                }
            },
            "regime_filter_accuracy": (
                None
                if self.regime_filter_accuracy is None
                else float(self.regime_filter_accuracy)
            ),
            "thresholds": asdict(self.thresholds),
        }


def evaluate_expert(
    expert: RollingHorizonMarkowitzExpert,
    returns: np.ndarray,
    trajectory: ExpertTrajectory,
    *,
    turnover_by_cost: dict[float, float],
    horizon_turnover: dict[int, float] | None = None,
    regimes: np.ndarray | None = None,
    replication_samples: int = 4,
    thresholds: ExpertDiagnosticThresholds | None = None,
) -> ExpertDiagnostics:
    """Run every acceptance check for one expert run.

    Args:
        expert: The configuration that produced ``trajectory``.
        returns: The full return path of shape ``(T, n_assets)``.
        trajectory: Executed decisions from :meth:`RollingHorizonMarkowitzExpert.run`.
        turnover_by_cost: Mean turnover per transaction-cost level, from runs
            that differ only in ``transaction_cost``.
        horizon_turnover: Optional mean turnover per planning horizon, reported
            but not asserted.
        regimes: Optional simulator regime labels, used only to score the
            filter.
        replication_samples: Number of decision dates re-solved from truncated
            history.
        thresholds: Acceptance thresholds.
    """
    limits = thresholds if thresholds is not None else ExpertDiagnosticThresholds()
    values = np.asarray(returns, dtype=float)
    if values.ndim != 2:
        raise ValueError("returns must have shape (T, n_assets).")
    if trajectory.n_decisions == 0:
        raise ValueError("trajectory must contain at least one decision.")

    violations = expert.constraints.violations(
        trajectory.weights, tolerance=limits.max_constraint_violation
    )
    feasible = all(size == 0.0 for size in violations.values())

    statuses = tuple(sorted(set(trajectory.statuses)))
    healthy_statuses = set(statuses) <= {"optimal", "optimal_inaccurate"}
    solver_healthy = healthy_statuses and (
        trajectory.fallback_rate <= limits.max_fallback_rate
    )

    replication_error, replication_times = _replication_error(
        expert, values, trajectory, replication_samples
    )
    determinism_error = _determinism_error(expert, values, trajectory)

    costs = sorted(turnover_by_cost)
    if len(costs) < 2:
        raise ValueError("turnover_by_cost must contain at least two cost levels.")
    turnover_reduction = float(turnover_by_cost[costs[0]] - turnover_by_cost[costs[-1]])
    monotone = all(
        turnover_by_cost[low] >= turnover_by_cost[high] - limits.max_determinism_error
        for low, high in zip(costs, costs[1:])
    )

    median_solve = float(np.median(trajectory.solve_seconds))
    max_solve = float(trajectory.solve_seconds.max())

    accuracy = None if regimes is None else _regime_filter_accuracy(expert, values, regimes)

    checks = {
        "feasibility": feasible,
        "solver_health": solver_healthy,
        "no_look_ahead": replication_error <= limits.max_replication_error,
        "determinism": determinism_error <= limits.max_determinism_error,
        "transaction_cost_response": monotone
        and turnover_reduction >= limits.min_turnover_reduction,
        "runtime": median_solve <= limits.max_median_solve_seconds,
    }
    if accuracy is not None:
        checks["regime_filter_accuracy"] = accuracy >= limits.min_regime_filter_accuracy

    return ExpertDiagnostics(
        checks=checks,
        constraint_violations=violations,
        fallback_rate=trajectory.fallback_rate,
        replication_error=replication_error,
        replication_times=replication_times,
        determinism_error=determinism_error,
        turnover_by_cost=dict(turnover_by_cost),
        turnover_reduction=turnover_reduction,
        median_solve_seconds=median_solve,
        max_solve_seconds=max_solve,
        statuses=statuses,
        regime_filter_accuracy=accuracy,
        horizon_turnover=dict(horizon_turnover or {}),
        thresholds=limits,
    )


def _replication_error(
    expert: RollingHorizonMarkowitzExpert,
    returns: np.ndarray,
    trajectory: ExpertTrajectory,
    samples: int,
) -> tuple[float, tuple[int, ...]]:
    """Re-decide from truncated history and measure the weight difference.

    A decision that used any information after ``t`` cannot survive replacing
    ``returns`` with ``returns[:t]``, so this is a direct test of the
    no-look-ahead principle rather than a proxy for it.
    """
    if samples < 1 or trajectory.n_decisions == 0:
        return 0.0, ()

    indices = np.unique(
        np.linspace(0, trajectory.n_decisions - 1, min(samples, trajectory.n_decisions))
        .round()
        .astype(int)
    )
    replica = expert.clone()
    worst = 0.0
    checked: list[int] = []
    for index in indices:
        t = int(trajectory.decision_times[index])
        replica.forecaster.prepare(returns[:t])
        solution = replica.decide(t, trajectory.previous_weights[index])
        worst = max(worst, float(np.abs(solution.weights - trajectory.weights[index]).max()))
        checked.append(t)
    return worst, tuple(checked)


def _determinism_error(
    expert: RollingHorizonMarkowitzExpert,
    returns: np.ndarray,
    trajectory: ExpertTrajectory,
) -> float:
    """Solve the first and last decisions twice and compare the allocations."""
    replica = expert.clone()
    replica.forecaster.prepare(returns)
    worst = 0.0
    for index in {0, trajectory.n_decisions - 1}:
        t = int(trajectory.decision_times[index])
        first = replica.decide(t, trajectory.previous_weights[index]).weights
        second = replica.decide(t, trajectory.previous_weights[index]).weights
        worst = max(worst, float(np.abs(first - second).max()))
    return worst


def _regime_filter_accuracy(
    expert: RollingHorizonMarkowitzExpert,
    returns: np.ndarray,
    regimes: np.ndarray,
) -> float | None:
    """Score filtered regimes against simulator truth, if a filter is in use.

    Both the filter and the canonical simulation preset order states by
    increasing variance, so filtered state ``k`` is directly comparable with
    simulator regime ``k`` without any post-hoc matching.
    """
    forecaster = expert.forecaster
    if not hasattr(forecaster, "path"):
        return None
    try:
        path = forecaster.path
    except RuntimeError:
        return None

    labels = np.asarray(regimes, dtype=int)
    if labels.shape[0] != returns.shape[0]:
        raise ValueError("regimes must have one label per return observation.")

    predicted = path.probabilities.argmax(axis=1)
    evaluated = slice(path.burn_in, labels.size)
    return float((predicted[evaluated] == labels[evaluated]).mean())


def equal_weight_trajectory(
    returns: np.ndarray,
    decision_times: np.ndarray,
    transaction_cost: float,
    constraints: PortfolioConstraints | None = None,
) -> dict[str, np.ndarray]:
    """Return the equal-weight benchmark over the same decision dates.

    Weights are held constant and drift-induced rebalancing turnover is
    ignored, so the benchmark pays no transaction costs. That is deliberately
    favourable to the benchmark: any outperformance by the expert is then not
    an artefact of charging the comparison more than the expert.
    """
    values = np.asarray(returns, dtype=float)
    times = np.asarray(decision_times, dtype=int)
    bounds = constraints if constraints is not None else PortfolioConstraints()
    weights = bounds.default_weights(values.shape[1])

    gross = values[times] @ weights
    turnover = np.zeros(times.size)
    net = gross - transaction_cost * turnover
    return {"weights": weights, "gross_returns": gross, "net_returns": net, "turnover": turnover}
