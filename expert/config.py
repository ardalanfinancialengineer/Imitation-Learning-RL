"""Build expert components from configuration mappings.

Keeping construction in one place means the demo script, the demonstration
pipeline, and the ablation runs all instantiate the expert identically, and
that ablations differ from the baseline in exactly one documented parameter.
"""

from __future__ import annotations

from typing import Any

from ..simulation.costs import EXPERIMENT_COST_MAX, EXPERIMENT_COST_MIN
from .constraints import PortfolioConstraints
from .markowitz import DEFAULT_HORIZON, RollingHorizonMarkowitzExpert
from .moments import (
    COVARIANCE_ESTIMATORS,
    MomentForecaster,
    RegimeMixtureForecaster,
    RollingMomentForecaster,
)
from .regime_filter import CausalRegimeFilter

MOMENT_ESTIMATORS = ("rolling", "regime_mixture")


def build_constraints(constraints_cfg: dict[str, Any] | None) -> PortfolioConstraints:
    """Build the portfolio constraint set from its configuration block."""
    cfg = dict(constraints_cfg or {})
    unknown = set(cfg) - {"long_only", "min_weight", "max_weight", "leverage_limit"}
    if unknown:
        raise ValueError(f"unknown constraint keys: {sorted(unknown)}.")
    return PortfolioConstraints(
        long_only=bool(cfg.get("long_only", True)),
        min_weight=float(cfg.get("min_weight", 0.0)),
        max_weight=float(cfg.get("max_weight", 1.0)),
        leverage_limit=(
            None if cfg.get("leverage_limit") is None else float(cfg["leverage_limit"])
        ),
    )


def build_forecaster(
    moments_cfg: dict[str, Any] | None,
    *,
    estimator: str | None = None,
    covariance: str | None = None,
) -> MomentForecaster:
    """Build a causal moment forecaster, with optional ablation overrides."""
    cfg = dict(moments_cfg or {})
    chosen = str(estimator if estimator is not None else cfg.get("estimator", "rolling"))
    if chosen not in MOMENT_ESTIMATORS:
        raise ValueError(
            f"unknown moment estimator {chosen!r}; expected one of {MOMENT_ESTIMATORS}."
        )
    covariance_method = str(
        covariance if covariance is not None else cfg.get("covariance", "sample")
    )
    if covariance_method not in COVARIANCE_ESTIMATORS:
        raise ValueError(f"unknown covariance estimator {covariance_method!r}.")

    window = int(cfg.get("window", 60))
    if chosen == "rolling":
        return RollingMomentForecaster(window, covariance=covariance_method)

    regime_filter = CausalRegimeFilter(
        n_regimes=int(cfg.get("n_regimes", 3)),
        burn_in=int(cfg.get("burn_in", 500)),
        refit_every=int(cfg.get("refit_every", 250)),
    )
    blend_weight = float(cfg.get("blend_weight", 0.0))
    return RegimeMixtureForecaster(
        regime_filter,
        blend_window=window if blend_weight > 0.0 else None,
        blend_weight=blend_weight,
        covariance=covariance_method,
    )


def build_expert(
    expert_cfg: dict[str, Any],
    *,
    horizon: int | None = None,
    risk_aversion: float | None = None,
    transaction_cost: float | None = None,
    estimator: str | None = None,
    covariance: str | None = None,
    forecaster: MomentForecaster | None = None,
    enforce_cost_range: bool = True,
) -> RollingHorizonMarkowitzExpert:
    """Build a rolling-horizon expert, overriding at most a few parameters.

    ``enforce_cost_range`` applies the exposé's experiment range
    ``c in [0.001, 0.005]``. It is relaxed for the ``c = 0`` ablation, which
    exists to demonstrate that the cost term changes behaviour at all.

    Passing ``forecaster`` shares an already-prepared moment estimator across
    ablations that do not change it. Forecasts are read-only after
    ``prepare()``, and re-estimating the HMM for every cost or horizon setting
    would dominate the runtime for no change in output.
    """
    if forecaster is not None and (estimator is not None or covariance is not None):
        raise ValueError("forecaster cannot be combined with estimator/covariance overrides.")
    if not isinstance(expert_cfg, dict):
        raise ValueError("expert configuration must be a mapping.")

    cost = float(
        transaction_cost
        if transaction_cost is not None
        else expert_cfg.get("transaction_cost", EXPERIMENT_COST_MIN)
    )
    if enforce_cost_range and not EXPERIMENT_COST_MIN <= cost <= EXPERIMENT_COST_MAX:
        raise ValueError(
            f"transaction_cost must lie in [{EXPERIMENT_COST_MIN}, {EXPERIMENT_COST_MAX}]; "
            f"got {cost}."
        )

    return RollingHorizonMarkowitzExpert(
        forecaster
        if forecaster is not None
        else build_forecaster(
            expert_cfg.get("moments"), estimator=estimator, covariance=covariance
        ),
        horizon=int(
            horizon if horizon is not None else expert_cfg.get("horizon", DEFAULT_HORIZON)
        ),
        risk_aversion=float(
            risk_aversion
            if risk_aversion is not None
            else expert_cfg.get("risk_aversion", 5.0)
        ),
        transaction_cost=cost,
        constraints=build_constraints(expert_cfg.get("constraints")),
        solver=expert_cfg.get("solver"),
    )
