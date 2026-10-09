"""Rolling-horizon, friction-aware Markowitz expert.

At every rebalancing date ``t`` the expert solves the multi-period programme of
the exposé (Equation 5),

    max_{w_t|t, ..., w_{t+H-1|t}}
        sum_{tau=t}^{t+H-1} [ w_tau' mu_hat_{tau+1|t}
                              - (gamma/2) w_tau' Sigma_hat_{tau+1|t} w_tau
                              - c ||w_tau - w_{tau-1}||_1 ]
    s.t. 1' w_tau = 1,  l_i <= w_{i,tau} <= u_i,  ||w_tau||_1 <= L,

with ``w_{t-1|t}`` anchored on the previously *implemented* portfolio. Only the
first planned allocation ``w*_t = w*_{t|t}`` is executed; the programme is
re-solved at the next date with newly available information. Every forecast
``mu_hat_{tau+1|t}``, ``Sigma_hat_{tau+1|t}`` is produced by a
:class:`~src.expert.moments.MomentForecaster` from ``returns[:t]`` only, so the
expert carries no look-ahead.

Note on the risk-aversion convention: ``gamma`` multiplies the *variance* term,
as in the exposé. An earlier single-period implementation scaled the mean by
``gamma`` instead; the two agree only up to a reparameterisation, and this
module follows the exposé.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Any

import cvxpy as cp
import numpy as np
import pandas as pd

from .constraints import PortfolioConstraints
from .moments import MomentForecast, MomentForecaster, RollingMomentForecaster

DEFAULT_HORIZON = 21

# Solver preference depends on problem size, measured on this problem shape
# (H * N decision variables, one L1 term per step, box and budget constraints):
#
#   decision vars | CLARABEL | OSQP    | SCS
#   168  (N=8)    |   4.5 ms |  5.2 ms |  5.9 ms
#   1050 (N=50)   |  74.8 ms | 75.9 ms | 76.2 ms
#   2100 (N=100)  |  3457 ms |  517 ms |  343 ms
#   4200 (N=200)  | 10596 ms | 2141 ms | 1309 ms
#
# CLARABEL is an interior-point method: most accurate, and fastest until its
# dense KKT factorisation starts to dominate. Above the threshold OSQP is
# several times faster and stays within ~2e-4 of CLARABEL's weights, whereas
# SCS is faster still but an order of magnitude looser (~1e-3 at small N).
_SOLVER_PREFERENCE_SMALL = ("CLARABEL", "OSQP", "SCS")
_SOLVER_PREFERENCE_LARGE = ("OSQP", "CLARABEL", "SCS")
_LARGE_PROBLEM_VARIABLES = 1500


@dataclass(frozen=True)
class ExpertSolution:
    """Outcome of one rolling-horizon optimisation."""

    weights: np.ndarray
    plan: np.ndarray
    status: str
    solve_seconds: float
    objective_value: float
    used_fallback: bool


@dataclass(frozen=True)
class ExpertTrajectory:
    """Executed expert decisions and realised performance for one path."""

    decision_times: np.ndarray
    weights: np.ndarray
    previous_weights: np.ndarray
    gross_returns: np.ndarray
    net_returns: np.ndarray
    turnover: np.ndarray
    solve_seconds: np.ndarray
    statuses: tuple[str, ...]
    fallbacks: np.ndarray
    forecast_means: np.ndarray
    forecast_covariance_diagonals: np.ndarray
    settings: dict[str, Any] = field(default_factory=dict)

    @property
    def n_decisions(self) -> int:
        return int(self.weights.shape[0])

    @property
    def fallback_rate(self) -> float:
        return float(self.fallbacks.mean()) if self.fallbacks.size else 0.0

    def summary(self) -> dict[str, Any]:
        """Return JSON-serialisable headline statistics for reports."""
        return {
            "n_decisions": self.n_decisions,
            "mean_turnover": float(self.turnover.mean()),
            "mean_net_return": float(self.net_returns.mean()),
            "total_net_return": float(np.prod(1.0 + self.net_returns) - 1.0),
            "annualised_net_return": float(self.net_returns.mean() * 252.0),
            "annualised_volatility": float(self.net_returns.std(ddof=1) * np.sqrt(252.0))
            if self.net_returns.size > 1
            else 0.0,
            "mean_solve_seconds": float(self.solve_seconds.mean()),
            "median_solve_seconds": float(np.median(self.solve_seconds)),
            "max_solve_seconds": float(self.solve_seconds.max()),
            "fallback_rate": self.fallback_rate,
            "unique_statuses": sorted(set(self.statuses)),
            "settings": self.settings,
        }


class RollingHorizonMarkowitzExpert:
    """Multi-period mean-variance expert with transaction costs and constraints."""

    def __init__(
        self,
        forecaster: MomentForecaster,
        *,
        horizon: int = DEFAULT_HORIZON,
        risk_aversion: float = 5.0,
        transaction_cost: float = 0.001,
        constraints: PortfolioConstraints | None = None,
        solver: str | None = None,
    ) -> None:
        if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 1:
            raise ValueError("horizon must be a positive integer.")
        if not np.isfinite(risk_aversion) or risk_aversion < 0.0:
            raise ValueError("risk_aversion must be finite and nonnegative.")
        if not np.isfinite(transaction_cost) or transaction_cost < 0.0:
            raise ValueError("transaction_cost must be finite and nonnegative.")
        if not hasattr(forecaster, "forecast") or not hasattr(forecaster, "prepare"):
            raise TypeError("forecaster must implement prepare() and forecast().")

        self.forecaster = forecaster
        self.horizon = horizon
        self.risk_aversion = float(risk_aversion)
        self.transaction_cost = float(transaction_cost)
        self.constraints = constraints if constraints is not None else PortfolioConstraints()
        self.solver = solver
        self._compiled: _CompiledProgramme | None = None

    # ------------------------------------------------------------------
    # Single decisions
    # ------------------------------------------------------------------
    def solve(
        self,
        forecast: MomentForecast,
        previous_weights: np.ndarray,
    ) -> ExpertSolution:
        """Solve one rolling-horizon programme for given forecasts."""
        anchor = np.asarray(previous_weights, dtype=float)
        if anchor.ndim != 1 or anchor.size != forecast.n_assets:
            raise ValueError("previous_weights must have shape (n_assets,).")
        if not np.all(np.isfinite(anchor)):
            raise ValueError("previous_weights must contain only finite values.")
        if forecast.horizon != self.horizon:
            raise ValueError(
                f"forecast horizon {forecast.horizon} does not match expert horizon "
                f"{self.horizon}."
            )
        self.constraints.validate_for(forecast.n_assets)

        programme = self._programme(forecast.n_assets)
        programme.load(forecast, anchor)

        started = time.perf_counter()
        status, objective_value = programme.solve(self.solver)
        elapsed = time.perf_counter() - started

        plan = programme.plan()
        if plan is not None:
            candidate = self.constraints.project(plan[0])
            if self.constraints.is_feasible(candidate):
                return ExpertSolution(
                    weights=candidate,
                    plan=plan,
                    status=status,
                    solve_seconds=elapsed,
                    objective_value=objective_value,
                    used_fallback=False,
                )

        fallback = self._fallback_weights(anchor, forecast.n_assets)
        return ExpertSolution(
            weights=fallback,
            plan=np.tile(fallback, (self.horizon, 1)),
            status=status,
            solve_seconds=elapsed,
            objective_value=float("nan"),
            used_fallback=True,
        )

    def decide(self, t: int, previous_weights: np.ndarray) -> ExpertSolution:
        """Solve the programme at decision index ``t`` of a prepared path."""
        forecast = self.forecaster.forecast(t, self.horizon)
        return self.solve(forecast, previous_weights)

    def compute_weights(
        self,
        history: np.ndarray,
        previous_weights: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return the implemented allocation given a raw return history.

        Convenience entry point for one-off decisions and for later DAgger
        queries. It re-prepares the forecaster on ``history``, so prefer
        :meth:`run` when rolling over a whole path.
        """
        observations = np.asarray(history, dtype=float)
        if observations.ndim != 2 or observations.shape[0] == 0:
            raise ValueError("history must have shape (n_observations, n_assets).")
        anchor = (
            self.constraints.default_weights(observations.shape[1])
            if previous_weights is None
            else np.asarray(previous_weights, dtype=float)
        )
        self.forecaster.prepare(observations)
        return self.decide(observations.shape[0], anchor).weights

    # ------------------------------------------------------------------
    # Full paths
    # ------------------------------------------------------------------
    def run(
        self,
        returns: pd.DataFrame | np.ndarray,
        *,
        warmup: int | None = None,
        initial_weights: np.ndarray | None = None,
        max_decisions: int | None = None,
    ) -> ExpertTrajectory:
        """Roll the expert over a return path and record realised performance.

        The decision at index ``t`` sees ``returns[:t]`` and earns
        ``returns[t]`` net of the cost of trading into it.
        """
        values = _as_return_array(returns)
        n_periods, n_assets = values.shape
        self.constraints.validate_for(n_assets)

        min_history = int(getattr(self.forecaster, "min_history", 1))
        start = min_history if warmup is None else int(warmup)
        if start < min_history:
            raise ValueError(
                f"warmup={start} is shorter than the forecaster's min_history={min_history}."
            )
        if start >= n_periods:
            raise ValueError("returns must contain more periods than the warmup.")

        decision_times = np.arange(start, n_periods)
        if max_decisions is not None:
            if not isinstance(max_decisions, int) or max_decisions < 1:
                raise ValueError("max_decisions must be a positive integer.")
            decision_times = decision_times[:max_decisions]
        n_decisions = decision_times.size

        anchor = (
            self.constraints.default_weights(n_assets)
            if initial_weights is None
            else self.constraints.project(np.asarray(initial_weights, dtype=float))
        )

        self.forecaster.prepare(values)

        weights = np.empty((n_decisions, n_assets))
        previous = np.empty((n_decisions, n_assets))
        solve_seconds = np.empty(n_decisions)
        fallbacks = np.zeros(n_decisions, dtype=bool)
        forecast_means = np.empty((n_decisions, n_assets))
        forecast_variances = np.empty((n_decisions, n_assets))
        statuses: list[str] = []

        for index, t in enumerate(decision_times):
            forecast = self.forecaster.forecast(int(t), self.horizon)
            solution = self.solve(forecast, anchor)

            previous[index] = anchor
            weights[index] = solution.weights
            solve_seconds[index] = solution.solve_seconds
            fallbacks[index] = solution.used_fallback
            forecast_means[index] = forecast.means[0]
            forecast_variances[index] = np.diag(forecast.covariances[0])
            statuses.append(solution.status)
            anchor = solution.weights

        realised = values[decision_times]
        gross_returns = np.einsum("ij,ij->i", weights, realised)
        turnover = np.abs(weights - previous).sum(axis=1)
        net_returns = gross_returns - self.transaction_cost * turnover

        return ExpertTrajectory(
            decision_times=decision_times,
            weights=weights,
            previous_weights=previous,
            gross_returns=gross_returns,
            net_returns=net_returns,
            turnover=turnover,
            solve_seconds=solve_seconds,
            statuses=tuple(statuses),
            fallbacks=fallbacks,
            forecast_means=forecast_means,
            forecast_covariance_diagonals=forecast_variances,
            settings=self.describe(),
        )

    def clone(
        self, *, forecaster: MomentForecaster | None = None
    ) -> RollingHorizonMarkowitzExpert:
        """Return an independent expert with the same settings.

        The compiled CVXPY programme holds solver handles that cannot be
        copied, so it is rebuilt lazily in the clone rather than duplicated.
        Diagnostics use this to re-solve decisions without disturbing the
        forecaster state of the run under test.
        """
        return RollingHorizonMarkowitzExpert(
            forecaster if forecaster is not None else copy.deepcopy(self.forecaster),
            horizon=self.horizon,
            risk_aversion=self.risk_aversion,
            transaction_cost=self.transaction_cost,
            constraints=self.constraints,
            solver=self.solver,
        )

    def describe(self) -> dict[str, Any]:
        """Return a JSON-serialisable description of the expert configuration."""
        return {
            "horizon": self.horizon,
            "risk_aversion": self.risk_aversion,
            "transaction_cost": self.transaction_cost,
            "constraints": self.constraints.to_dict(),
            "forecaster": self.forecaster.describe(),
            "solver": self.solver or "auto",
            "objective": (
                "max sum_tau [w'mu - (gamma/2) w'Sigma w - c ||w_tau - w_{tau-1}||_1]"
            ),
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _programme(self, n_assets: int) -> _CompiledProgramme:
        if self._compiled is None or self._compiled.n_assets != n_assets:
            self._compiled = _CompiledProgramme(
                horizon=self.horizon,
                n_assets=n_assets,
                risk_aversion=self.risk_aversion,
                transaction_cost=self.transaction_cost,
                constraints=self.constraints,
            )
        return self._compiled

    def _fallback_weights(self, anchor: np.ndarray, n_assets: int) -> np.ndarray:
        """Return a feasible allocation when the solver fails.

        Holding the previous portfolio is preferred because it incurs no
        turnover; the equal-weight portfolio is the last resort.
        """
        if self.constraints.is_feasible(anchor):
            return anchor.copy()
        return self.constraints.default_weights(n_assets)


class _CompiledProgramme:
    """A CVXPY programme compiled once and re-solved through parameters.

    Rebuilding the problem at every rebalancing date dominates the runtime of a
    10,000-step path. Forecasts therefore enter as parameters. The risk term is
    written as ``||L' w||^2`` with ``Sigma = L L'`` because a parameterised
    ``quad_form`` is not DPP-compliant, whereas a parameter matrix multiplying
    the variable is.
    """

    def __init__(
        self,
        *,
        horizon: int,
        n_assets: int,
        risk_aversion: float,
        transaction_cost: float,
        constraints: PortfolioConstraints,
    ) -> None:
        self.horizon = horizon
        self.n_assets = n_assets
        self.constraints = constraints

        self.weights = cp.Variable((horizon, n_assets), name="weights")
        self.mean_parameter = cp.Parameter((horizon, n_assets), name="means")
        self.cholesky_parameters = [
            cp.Parameter((n_assets, n_assets), name=f"chol_{step}")
            for step in range(horizon)
        ]
        self.anchor_parameter = cp.Parameter(n_assets, name="previous_weights")

        expected_return = cp.sum(cp.multiply(self.mean_parameter, self.weights))
        risk = cp.sum(
            [
                cp.sum_squares(self.cholesky_parameters[step] @ self.weights[step, :])
                for step in range(horizon)
            ]
        )
        objective = -expected_return + 0.5 * risk_aversion * risk

        if transaction_cost > 0.0:
            trades = [cp.norm1(self.weights[0, :] - self.anchor_parameter)]
            trades += [
                cp.norm1(self.weights[step, :] - self.weights[step - 1, :])
                for step in range(1, horizon)
            ]
            objective = objective + transaction_cost * cp.sum(cp.hstack(trades))

        problem_constraints = [
            cp.sum(self.weights, axis=1) == 1.0,
            self.weights <= constraints.max_weight,
            self.weights >= constraints.effective_min_weight,
        ]
        # With long-only, fully invested weights the L1 norm equals one, so an
        # explicit leverage constraint would be redundant and only slow the
        # solver down. It is imposed once short positions are allowed.
        if constraints.leverage_limit is not None and not constraints.long_only:
            problem_constraints += [
                cp.norm1(self.weights[step, :]) <= constraints.leverage_limit
                for step in range(horizon)
            ]

        self.problem = cp.Problem(cp.Minimize(objective), problem_constraints)
        self._solver_order = self._resolve_solver_order(horizon * n_assets)

    def load(self, forecast: MomentForecast, anchor: np.ndarray) -> None:
        """Bind one date's forecasts and previously implemented weights."""
        self.mean_parameter.value = forecast.means
        for step in range(self.horizon):
            self.cholesky_parameters[step].value = _upper_cholesky(
                forecast.covariances[step]
            )
        self.anchor_parameter.value = anchor

    def solve(self, solver: str | None) -> tuple[str, float]:
        # Warm starting is deliberately off. It would make each allocation
        # depend on the solves that preceded it, so replaying a single decision
        # in isolation -- which DAgger, the no-look-ahead check, and any audit
        # of an individual date all do -- would only reproduce it to solver
        # tolerance. The programme is compiled once, so the saving is small.
        order = [solver] if solver is not None else self._solver_order
        status = "not_attempted"
        for candidate in order:
            try:
                if candidate is None:
                    self.problem.solve(warm_start=False)
                else:
                    self.problem.solve(solver=candidate, warm_start=False)
            except (cp.SolverError, ValueError):
                status = "solver_error"
                continue
            status = str(self.problem.status)
            if status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
                value = self.problem.value
                return status, float(value) if value is not None else float("nan")
        return status, float("nan")

    def plan(self) -> np.ndarray | None:
        """Return the planned weight path, or ``None`` if unusable."""
        value = self.weights.value
        if value is None:
            return None
        planned = np.asarray(value, dtype=float).reshape(self.horizon, self.n_assets)
        if not np.all(np.isfinite(planned)):
            return None
        return planned

    @staticmethod
    def _resolve_solver_order(n_variables: int) -> list[str | None]:
        preference = (
            _SOLVER_PREFERENCE_LARGE
            if n_variables > _LARGE_PROBLEM_VARIABLES
            else _SOLVER_PREFERENCE_SMALL
        )
        installed = set(cp.installed_solvers())
        order: list[str | None] = [name for name in preference if name in installed]
        order.append(None)
        return order


class RollingMarkowitzExpert:
    """Single-period rolling-window Markowitz expert.

    Retained so existing scripts and demonstration datasets keep working. It is
    a thin wrapper over :class:`RollingHorizonMarkowitzExpert` configured with
    ``horizon=1`` and zero transaction cost. Note that ``risk_aversion`` now
    scales the variance term, matching Equation (5) of the exposé.
    """

    def __init__(
        self,
        window: int,
        risk_aversion: float = 1.0,
        long_only: bool = True,
        max_weight: float = 1.0,
    ) -> None:
        if not isinstance(window, int) or isinstance(window, bool) or window < 2:
            raise ValueError("window must be an integer of at least 2.")
        if not isinstance(long_only, bool):
            raise TypeError("long_only must be a bool.")

        self.window = window
        self.risk_aversion = float(risk_aversion)
        self.long_only = long_only
        self.max_weight = float(max_weight)
        self.constraints = PortfolioConstraints(
            long_only=long_only,
            min_weight=0.0 if long_only else -float(max_weight),
            max_weight=float(max_weight),
        )
        self._expert = RollingHorizonMarkowitzExpert(
            RollingMomentForecaster(window),
            horizon=1,
            risk_aversion=float(risk_aversion),
            transaction_cost=0.0,
            constraints=self.constraints,
        )

    def compute_weights(self, returns_window: np.ndarray) -> np.ndarray:
        """Solve the single-period problem for exactly ``window`` observations."""
        values = np.asarray(returns_window, dtype=float)
        if values.ndim != 2:
            raise ValueError("returns_window must be a 2-D array.")
        if values.shape[0] != self.window:
            raise ValueError(
                f"returns_window must have exactly {self.window} rows; "
                f"got {values.shape[0]}."
            )
        if values.shape[1] == 0:
            raise ValueError("returns_window must contain at least one asset.")
        if not np.all(np.isfinite(values)):
            raise ValueError("returns_window must contain only finite values.")
        self.constraints.validate_for(values.shape[1])
        return self._expert.compute_weights(values)

    def generate_demonstrations(
        self, returns: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        """Roll the expert over a full return series to collect (obs, action) pairs."""
        if not isinstance(returns, pd.DataFrame):
            raise TypeError("returns must be a pandas DataFrame.")

        values = returns.to_numpy(dtype=float)
        if values.ndim != 2 or values.shape[1] == 0:
            raise ValueError(
                "returns must have shape (T, n_assets) with at least one asset."
            )
        if values.shape[0] <= self.window:
            raise ValueError("returns must contain more rows than the rolling window.")
        if not np.all(np.isfinite(values)):
            raise ValueError("returns must contain only finite values.")

        trajectory = self._expert.run(values, warmup=self.window)
        n_samples = values.shape[0] - self.window
        observations = np.empty((n_samples, self.window, values.shape[1]))
        for index, t in enumerate(trajectory.decision_times):
            observations[index] = values[t - self.window : t]
        return observations, trajectory.weights


def _as_return_array(returns: pd.DataFrame | np.ndarray) -> np.ndarray:
    values = (
        returns.to_numpy(dtype=float)
        if isinstance(returns, pd.DataFrame)
        else np.asarray(returns, dtype=float)
    )
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("returns must have shape (T, n_assets).")
    if not np.all(np.isfinite(values)):
        raise ValueError("returns must contain only finite values.")
    return values


def _upper_cholesky(covariance: np.ndarray) -> np.ndarray:
    """Return ``L'`` for ``Sigma = L L'``, ridging if the matrix is borderline."""
    matrix = 0.5 * (covariance + covariance.T)
    for attempt in range(6):
        try:
            return np.linalg.cholesky(matrix).T
        except np.linalg.LinAlgError:
            scale = max(float(np.trace(matrix)) / matrix.shape[0], 1e-16)
            matrix = matrix + np.eye(matrix.shape[0]) * scale * (10.0**attempt) * 1e-10
    raise np.linalg.LinAlgError("covariance forecast is not positive definite.")
