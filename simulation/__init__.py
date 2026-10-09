"""Synthetic stock return simulation with stochastic regime switching."""

from .costs import (
    DEFAULT_TRANSACTION_COSTS,
    EXPERIMENT_COST_MAX,
    EXPERIMENT_COST_MIN,
    ProportionalTransactionCost,
    validate_transaction_cost_grid,
)
from .diagnostics import (
    DiagnosticThresholds,
    SimulationDiagnostics,
    evaluate_simulation,
)
from .presets import (
    CANONICAL_PRESET_NAME,
    CANONICAL_REGIME_NAMES,
    UNMATCHED_PRESET_NAME,
    RegimeSimulationPreset,
    canonical_three_regime_preset,
    load_simulation_preset,
    unmatched_three_regime_preset,
)
from .regimes import RegimeSwitchingModel
from .returns import ReturnSimulator

__all__ = [
    "CANONICAL_PRESET_NAME",
    "CANONICAL_REGIME_NAMES",
    "UNMATCHED_PRESET_NAME",
    "DEFAULT_TRANSACTION_COSTS",
    "EXPERIMENT_COST_MAX",
    "EXPERIMENT_COST_MIN",
    "DiagnosticThresholds",
    "ProportionalTransactionCost",
    "RegimeSimulationPreset",
    "RegimeSwitchingModel",
    "ReturnSimulator",
    "SimulationDiagnostics",
    "canonical_three_regime_preset",
    "evaluate_simulation",
    "load_simulation_preset",
    "unmatched_three_regime_preset",
    "validate_transaction_cost_grid",
]
