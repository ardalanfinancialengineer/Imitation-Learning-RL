"""Imitation learning: behavioural cloning, its metrics, and its acceptance checks."""

from .bc import (
    BCConfig,
    BCHistory,
    BehaviouralCloningTrainer,
    EpochRecord,
    TensorPairDataset,
    estimated_megabytes,
    materialise,
    seed_everything,
)
from .comparison import (
    BootstrapInterval,
    paired_bootstrap,
    per_path_tracking_errors,
    seed_summary,
)
from .critic import LIPSCHITZ_METHODS, CriticConfig, WassersteinCritic
from .dagger import (
    BETA_SCHEDULES,
    AggregatedDataset,
    AggregatedRound,
    DAggerConfig,
    DAggerHistory,
    DAggerTrainer,
    RoundRecord,
)
from .dagger_diagnostics import (
    DAggerDiagnostics,
    DAggerDiagnosticThresholds,
    evaluate_dagger,
    improvement_interval,
)
from .dataset import DemonstrationDataset, PathEntry
from .diagnostics import (
    PolicyDiagnostics,
    PolicyDiagnosticThresholds,
    evaluate_policy,
)
from .differentiable import (
    RolloutTape,
    SegmentRollout,
    build_tape,
    realised_net_returns,
    rollout_segment,
    rollout_tape,
)
from .gail import GAILConfig, GAILHistory, GAILRecord, GAILTrainer
from .gail_diagnostics import (
    GAILDiagnostics,
    GAILDiagnosticThresholds,
    compare_policies,
    evaluate_gail,
)
from .losses import (
    LOSS_NAMES,
    allocation_l1,
    allocation_mse,
    get_loss,
    weight_divergence,
)
from .rollout import (
    ExpertOracle,
    PathRollout,
    answer_rollouts,
    per_path_closed_loop_errors,
    rollout_metrics,
    rollout_policy,
)

__all__ = [
    "BETA_SCHEDULES",
    "LIPSCHITZ_METHODS",
    "LOSS_NAMES",
    "AggregatedDataset",
    "AggregatedRound",
    "BCConfig",
    "BCHistory",
    "BehaviouralCloningTrainer",
    "BootstrapInterval",
    "DAggerConfig",
    "DAggerDiagnosticThresholds",
    "DAggerDiagnostics",
    "DAggerHistory",
    "DAggerTrainer",
    "CriticConfig",
    "DemonstrationDataset",
    "EpochRecord",
    "ExpertOracle",
    "GAILConfig",
    "GAILDiagnosticThresholds",
    "GAILDiagnostics",
    "GAILHistory",
    "GAILRecord",
    "GAILTrainer",
    "PathEntry",
    "PathRollout",
    "PolicyDiagnosticThresholds",
    "PolicyDiagnostics",
    "RolloutTape",
    "RoundRecord",
    "SegmentRollout",
    "TensorPairDataset",
    "WassersteinCritic",
    "allocation_l1",
    "allocation_mse",
    "answer_rollouts",
    "build_tape",
    "compare_policies",
    "estimated_megabytes",
    "evaluate_gail",
    "evaluate_dagger",
    "evaluate_policy",
    "get_loss",
    "improvement_interval",
    "materialise",
    "paired_bootstrap",
    "per_path_closed_loop_errors",
    "per_path_tracking_errors",
    "realised_net_returns",
    "rollout_metrics",
    "rollout_policy",
    "rollout_segment",
    "rollout_tape",
    "seed_everything",
    "seed_summary",
    "weight_divergence",
]
