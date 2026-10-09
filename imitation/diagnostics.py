"""Acceptance checks for the policy and behavioural-cloning milestone.

The Weeks 17-21 work package asks for the first behavioural cloning models and
a comparison of "prediction quality and runtime". A validation loss alone would
not establish either, so the milestone is gated on seven deterministic checks.

``action_feasibility``
    every allocation the policy produces on held-out states satisfies the
    budget, the box bounds, and the leverage limit of Equation (5). A policy
    that proposes infeasible portfolios has not solved the problem regardless
    of its loss;
``determinism``
    running the policy twice on the same states gives bit-identical
    allocations. This is a real check rather than a formality, because the
    diffusion policy samples: it passes only because DDIM is deterministic
    given its initial noise and the generator is seeded;
``optimisation_sanity``
    the model can drive the loss on a single small batch far below its value at
    initialisation. This catches a dead architecture, a detached graph, or a
    projection that blocks gradients — failures that otherwise show up as a
    plateau that looks like a hard problem;
``beats_naive_baselines``
    held-out tracking error is below both the equal-weight rule and the
    never-trade rule, **and** the paired bootstrap interval for the difference
    excludes zero. The second baseline is the demanding one: ``w_{t-1}`` is a
    block of the state, so a policy could reproduce it by reading one slice,
    and under transaction costs it explains much of what the expert does.

    The interval is not decoration. Both learned heads are initialised to
    return ``w_{t-1}`` exactly, so every policy starts *at* the never-trade
    baseline and a bare inequality is cleared by noise alone. Requiring the
    paired interval of Section 5.4 to exclude zero is what makes this a
    statement that training accomplished something. See
    :mod:`src.imitation.comparison` and :mod:`src.policy.reference`;
``generalisation``
    the test tracking error does not exceed the *validation* tracking error by
    more than a stated factor, so the model chosen by early stopping is the one
    that also performs on data it never influenced.

    The comparison is deliberately validation-versus-test rather than
    train-versus-test. Whenever the state is a sufficient statistic for the
    expert's decision, a flexible enough policy drives its *training* error
    towards zero; a train-versus-test ratio then grows without bound for a good
    policy as easily as for a bad one, and gates nothing. Validation and test
    are both unseen paths drawn the same way, so a gap between them is instead a
    genuine failure of model selection. The train-to-test gap is still measured
    and reported as ``memorisation_gap``, in the same
    reported-not-asserted spirit as ``anchor_chain_sensitivity`` in Weeks 14-16;

    Sufficiency is a property of the configuration, not a given. Under the
    shipped ``regime_mixture`` forecaster the expert plans over ``H = 21`` using
    ``mu_hat_{tau+1|t}`` for every ``tau``, and those forecasts vary across the
    horizon by roughly 67% of the level in the mean and 43% in the covariance,
    while Equation (6) carries only the ``tau = 0`` forecast. The expert
    therefore conditions on information the policy cannot see, and imitation
    error has a floor no amount of training removes. A rolling-window
    forecaster is horizon-constant and does not have this gap. See
    ``docs/policy_behavioural_cloning.md``;
``inference_latency``
    the mean per-decision inference time is below the mean time to re-solve the
    rolling-horizon quadratic programme by at least a stated factor. This is
    research question 5 measured directly rather than asserted;
``policy_no_look_ahead``
    rewriting every return *after* a decision date leaves the policy's action at
    that date unchanged. Weeks 14-16 established this for the assembled state;
    here it is checked end to end, through the network, which is what closes off
    the possibility of a leak entering via the standardiser or the encoder.

The look-ahead check inherits the control-versus-treatment discipline measured
in Weeks 14-16: it compares two rebuilds that differ *only* in the future,
never a rebuild against the stored path. The stored path carries accumulated
solver noise from the transaction-cost anchor chain, which would force a
tolerance loose enough to hide a small genuine leak.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch

from ..dataset.demonstrations import DemonstrationDataset
from ..dataset.diagnostics import rebuild_truncated_path
from ..dataset.paths import PathData
from ..policy.base import AllocationPolicy
from ..policy.reference import default_references
from .bc import BehaviouralCloningTrainer, materialise, seed_everything
from .comparison import (
    BootstrapInterval,
    paired_bootstrap,
    per_path_tracking_errors,
)


@dataclass(frozen=True)
class PolicyDiagnosticThresholds:
    """Deterministic acceptance thresholds for one trained policy."""

    max_feasibility_violation: float = 1e-6
    max_determinism_difference: float = 0.0
    min_overfit_loss_reduction: float = 5.0
    max_generalisation_ratio: float = 1.5
    min_latency_speedup: float = 5.0
    max_look_ahead_difference: float = 0.0
    baseline_confidence_level: float = 0.95
    require_baseline_significance: bool = True

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if isinstance(value, bool):
                continue
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be nonnegative and finite.")
        if not 0.0 < self.baseline_confidence_level < 1.0:
            raise ValueError("baseline_confidence_level must lie strictly in (0, 1).")
        if self.min_overfit_loss_reduction < 1.0:
            raise ValueError("min_overfit_loss_reduction must be at least 1.")
        if self.max_generalisation_ratio < 1.0:
            raise ValueError("max_generalisation_ratio must be at least 1.")


@dataclass(frozen=True)
class PolicyDiagnostics:
    """Observed-versus-required behaviour of one trained policy."""

    checks: dict[str, bool]
    feasibility_violation: float
    determinism_difference: float
    overfit_loss_reduction: float
    train_tracking_error: float
    validation_tracking_error: float
    holdout_tracking_error: float
    generalisation_ratio: float
    memorisation_gap: float
    baseline_tracking_errors: dict[str, float]
    baseline_intervals: dict[str, BootstrapInterval]
    policy_seconds_per_decision: float
    expert_seconds_per_decision: float
    latency_speedup: float
    look_ahead_difference: float
    checked_decision_times: tuple[int, ...]
    holdout_metrics: dict[str, float]
    thresholds: PolicyDiagnosticThresholds

    @property
    def passed(self) -> bool:
        return all(self.checks.values())

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable report."""
        return {
            "passed": self.passed,
            "checks": dict(self.checks),
            "action_feasibility": {
                "max_violation": float(self.feasibility_violation)
            },
            "determinism": {"max_difference": float(self.determinism_difference)},
            "optimisation_sanity": {
                "loss_reduction_factor": float(self.overfit_loss_reduction)
            },
            "beats_naive_baselines": {
                "policy_tracking_error": float(self.holdout_tracking_error),
                "baselines": {
                    name: float(value)
                    for name, value in self.baseline_tracking_errors.items()
                },
                "paired_bootstrap": {
                    name: interval.to_dict()
                    for name, interval in self.baseline_intervals.items()
                },
                "note": (
                    "The interval is for baseline minus policy, paired path by "
                    "path (Section 5.4). Positive and excluding zero means the "
                    "policy is better than the rule by more than sampling noise. "
                    "Both learned heads start at w_{t-1} exactly, so a bare "
                    "inequality against never-trade is cleared by noise alone."
                ),
            },
            "generalisation": {
                "train_tracking_error": float(self.train_tracking_error),
                "validation_tracking_error": float(self.validation_tracking_error),
                "holdout_tracking_error": float(self.holdout_tracking_error),
                "test_over_validation": float(self.generalisation_ratio),
            },
            "memorisation_gap": {
                "test_over_train": float(self.memorisation_gap),
                "note": (
                    "Reported, not asserted. Where the state is a sufficient "
                    "statistic for the expert's decision, a flexible policy "
                    "drives its training error towards zero and this ratio grows "
                    "without bound whether or not the policy generalises. It "
                    "measures how much of the training fit is memorisation, and "
                    "is the clearest signal that more independent paths would "
                    "help. Under the shipped regime-mixture expert the state is "
                    "NOT sufficient: Equation (6) carries only the tau = 0 "
                    "forecast while the expert plans over H = 21 with "
                    "horizon-varying moments, so imitation error has a floor. "
                    "See docs/policy_behavioural_cloning.md."
                ),
            },
            "inference_latency": {
                "policy_seconds_per_decision": float(self.policy_seconds_per_decision),
                "expert_seconds_per_decision": float(self.expert_seconds_per_decision),
                "speedup": float(self.latency_speedup),
                "note": (
                    "Measured on a single CPU process, batch size one, with the "
                    "expert's CVXPY programme already compiled. It answers "
                    "research question 5 for this hardware only."
                ),
            },
            "policy_no_look_ahead": {
                "max_action_difference": float(self.look_ahead_difference),
                "checked_decision_times": list(self.checked_decision_times),
            },
            "holdout_metrics": {
                name: float(value) for name, value in self.holdout_metrics.items()
            },
            "thresholds": asdict(self.thresholds),
        }


def evaluate_policy(
    policy: AllocationPolicy,
    trainer: BehaviouralCloningTrainer,
    *,
    train_dataset: DemonstrationDataset,
    validation_dataset: DemonstrationDataset,
    holdout_dataset: DemonstrationDataset,
    look_ahead_path: PathData,
    preset: Any,
    latency_decisions: int = 20,
    look_ahead_samples: int = 2,
    thresholds: PolicyDiagnosticThresholds | None = None,
) -> PolicyDiagnostics:
    """Run every acceptance check for one trained policy."""
    limits = thresholds if thresholds is not None else PolicyDiagnosticThresholds()

    holdout_metrics = trainer.evaluate(holdout_dataset)
    train_metrics = trainer.evaluate(train_dataset)
    validation_metrics = trainer.evaluate(validation_dataset)

    determinism = _determinism_difference(policy, trainer, holdout_dataset)
    reduction = _overfit_reduction(policy, trainer, train_dataset)
    baselines, baseline_intervals = _baseline_comparison(
        policy, holdout_dataset, limits
    )
    policy_seconds, expert_seconds = _latency(
        policy, holdout_dataset, latency_decisions
    )
    look_ahead, checked = _look_ahead_difference(
        policy, holdout_dataset, look_ahead_path, preset, look_ahead_samples
    )

    holdout_tracking = float(holdout_metrics["tracking_error"])
    train_tracking = float(train_metrics["tracking_error"])
    validation_tracking = float(validation_metrics["tracking_error"])
    ratio = holdout_tracking / max(validation_tracking, 1e-12)
    memorisation = holdout_tracking / max(train_tracking, 1e-12)
    speedup = expert_seconds / max(policy_seconds, 1e-12)

    checks = {
        "action_feasibility": (
            float(holdout_metrics["feasibility_violation"])
            <= limits.max_feasibility_violation
        ),
        "determinism": determinism <= limits.max_determinism_difference,
        "optimisation_sanity": reduction >= limits.min_overfit_loss_reduction,
        "beats_naive_baselines": all(
            holdout_tracking < value for value in baselines.values()
        )
        and (
            not limits.require_baseline_significance
            or all(
                interval.mean > 0.0 and interval.low > 0.0
                for interval in baseline_intervals.values()
            )
        ),
        "generalisation": ratio <= limits.max_generalisation_ratio,
        "inference_latency": speedup >= limits.min_latency_speedup,
        "policy_no_look_ahead": look_ahead <= limits.max_look_ahead_difference,
    }

    return PolicyDiagnostics(
        checks=checks,
        feasibility_violation=float(holdout_metrics["feasibility_violation"]),
        determinism_difference=determinism,
        overfit_loss_reduction=reduction,
        train_tracking_error=train_tracking,
        validation_tracking_error=validation_tracking,
        holdout_tracking_error=holdout_tracking,
        generalisation_ratio=ratio,
        memorisation_gap=memorisation,
        baseline_tracking_errors=baselines,
        baseline_intervals=baseline_intervals,
        policy_seconds_per_decision=policy_seconds,
        expert_seconds_per_decision=expert_seconds,
        latency_speedup=speedup,
        look_ahead_difference=look_ahead,
        checked_decision_times=checked,
        holdout_metrics=holdout_metrics,
        thresholds=limits,
    )


def _determinism_difference(
    policy: AllocationPolicy,
    trainer: BehaviouralCloningTrainer,
    dataset: DemonstrationDataset,
    samples: int = 64,
) -> float:
    """Re-run the policy on the same states and measure the drift."""
    count = min(samples, len(dataset))
    states = np.stack([dataset.sample(index)[0] for index in range(count)])
    del trainer
    first = policy.act(states)
    second = policy.act(states)
    return float(np.abs(first - second).max())


def _reinitialise_parameters(module: torch.nn.Module) -> None:
    """Reset every submodule that knows how to re-initialise itself.

    The standardiser holds fitted buffers rather than parameters, so it is left
    alone and the probe keeps the real input scaling.
    """
    for child in module.modules():
        reset = getattr(child, "reset_parameters", None)
        if callable(reset):
            reset()


def _overfit_reduction(
    policy: AllocationPolicy,
    trainer: BehaviouralCloningTrainer,
    dataset: DemonstrationDataset,
    batch: int = 32,
    steps: int = 150,
) -> float:
    """Fit a re-initialised copy of the policy to one batch and report the reduction.

    The check exists to catch a dead architecture, a detached graph, or a
    projection that blocks gradients. It therefore probes the *architecture*,
    on a copy so the trained weights are never disturbed.

    The copy's network weights are re-initialised first, and that detail is
    load-bearing. An earlier version started from the trained state, which
    measured how much slack a converged policy happened to retain rather than
    whether the architecture can descend — and that quantity is not stable.
    Measured across three training seeds on the same fixture it came out as
    1.66, 1.48, and 1.07e7: the check was passing or failing on luck. The
    projection is why. Coordinates clamped at a bound receive exactly zero
    gradient, so once a policy has converged onto the expert's many exact
    zeros, the remaining loss on a training batch is largely irreducible and
    the ratio says nothing about gradient flow.

    From a re-initialised start the same fixture gives 17.7 on every seed, and
    a policy whose gradients do not reach its output still reports ~1.

    The fitted standardiser is deliberately kept: the probe should see the same
    input scaling the real policy does.
    """
    from copy import deepcopy

    count = min(batch, len(dataset))
    states = torch.stack(
        [torch.as_tensor(dataset.sample(i)[0], dtype=torch.float32) for i in range(count)]
    ).to(trainer.device)
    actions = torch.stack(
        [torch.as_tensor(dataset.sample(i)[1], dtype=torch.float32) for i in range(count)]
    ).to(trainer.device)

    seed_everything(trainer.config.seed + 1)
    clone = deepcopy(policy)
    _reinitialise_parameters(clone)
    clone.train()
    optimiser = torch.optim.AdamW(clone.parameters(), lr=trainer.config.learning_rate)

    initial = float(clone.bc_loss(states, actions, trainer.loss_fn).detach())
    final = initial
    for _ in range(steps):
        optimiser.zero_grad(set_to_none=True)
        loss = clone.bc_loss(states, actions, trainer.loss_fn)
        loss.backward()
        optimiser.step()
        final = float(loss.detach())

    if final <= 1e-12:
        return float("inf")
    return initial / final


def _baseline_tracking_errors(dataset: DemonstrationDataset) -> dict[str, float]:
    """Tracking error of the non-learned reference rules on the same data."""
    states, actions = materialise(dataset)
    raw_states = states.numpy().astype(np.float64)
    expert = actions.numpy().astype(np.float64)

    references = default_references(
        dataset.spec,
        n_assets=dataset.n_assets,
        n_regimes=dataset.assembler.n_regimes,
        constraints=_constraints_of(dataset),
    )
    return {
        name: float(np.mean(np.sum((rule.act(raw_states) - expert) ** 2, axis=1)))
        for name, rule in references.items()
    }


def _baseline_comparison(
    policy: AllocationPolicy,
    dataset: DemonstrationDataset,
    limits: PolicyDiagnosticThresholds,
) -> tuple[dict[str, float], dict[str, BootstrapInterval]]:
    """Compare the policy against each reference rule, paired path by path."""
    references = default_references(
        dataset.spec,
        n_assets=dataset.n_assets,
        n_regimes=dataset.assembler.n_regimes,
        constraints=_constraints_of(dataset),
    )
    policy_errors = per_path_tracking_errors(policy, dataset)

    # Sample-pooled means, so they are on the same footing as the policy's own
    # `holdout_tracking_error`, which the inequality is checked against. The
    # per-path errors are used only for the pairing, where equal weight per path
    # is what the bootstrap resamples over.
    means = _baseline_tracking_errors(dataset)
    intervals: dict[str, BootstrapInterval] = {}
    for name, rule in references.items():
        rule_errors = per_path_tracking_errors(rule, dataset)
        # Baseline minus policy: positive means the policy is the better one.
        intervals[name] = paired_bootstrap(
            rule_errors,
            policy_errors,
            level=limits.baseline_confidence_level,
        )
    return means, intervals


def _latency(
    policy: AllocationPolicy, dataset: DemonstrationDataset, decisions: int
) -> tuple[float, float]:
    """Time one policy inference against one expert quadratic-programme solve.

    Both are measured at batch size one, because a rebalancing date produces a
    single decision. Batching the policy would flatter it against an optimiser
    that cannot be batched at all.
    """
    entry = dataset.entries[0]
    count = min(decisions, int(entry.indices.size))
    if count == 0:
        return float("nan"), float("nan")

    positions = entry.indices[:count]
    states = np.stack(
        [dataset.assemble_state(entry, int(index)) for index in positions]
    )

    expert = entry.data.build_expert()
    # Warm up both: the first call compiles the CVXPY programme and populates
    # torch's kernel caches, and neither cost recurs per decision.
    expert.decide(
        int(entry.data.decision_times[int(positions[0])]),
        entry.data.previous_weights[int(positions[0])],
    )
    policy.act(states[0])

    started = time.perf_counter()
    for state in states:
        policy.act(state)
    policy_seconds = (time.perf_counter() - started) / count

    started = time.perf_counter()
    for index in positions:
        expert.decide(
            int(entry.data.decision_times[int(index)]),
            entry.data.previous_weights[int(index)],
        )
    expert_seconds = (time.perf_counter() - started) / count

    return policy_seconds, expert_seconds


def _look_ahead_difference(
    policy: AllocationPolicy,
    dataset: DemonstrationDataset,
    path: PathData,
    preset: Any,
    samples: int,
) -> tuple[float, tuple[int, ...]]:
    """Rewrite the future and measure how far the policy's action moves.

    Everything downstream of the returns is rebuilt from scratch on the altered
    array — the HMM is refitted and the expert re-solved — and the policy is
    then asked for an allocation at the same decision date. Comparison is
    against a control rebuild with the unaltered future, for the reason
    established in Weeks 14-16: the stored trajectory carries accumulated
    solver noise from the transaction-cost anchor chain, so comparing against
    it would need a tolerance loose enough to hide a real leak.
    """
    entry = next(
        (item for item in dataset.entries if item.data.path_id == path.path_id), None
    )
    if entry is None or entry.indices.size == 0:
        return 0.0, ()

    positions = np.unique(
        np.linspace(0, entry.indices.size - 1, min(samples, entry.indices.size))
        .round()
        .astype(int)
    )
    worst = 0.0
    checked: list[int] = []
    for position in positions:
        sample_index = int(entry.indices[position])
        t = int(path.decision_times[sample_index])

        altered = path.returns.copy()
        n_after = altered.shape[0] - t
        altered[t:] = path.returns[:n_after][::-1]

        control = rebuild_truncated_path(path, path.returns.copy(), preset, sample_index)
        treatment = rebuild_truncated_path(path, altered, preset, sample_index)

        difference = np.abs(
            _action_for(policy, dataset, treatment, sample_index)
            - _action_for(policy, dataset, control, sample_index)
        ).max()
        worst = max(worst, float(difference))
        checked.append(t)
    return worst, tuple(checked)


def _action_for(
    policy: AllocationPolicy,
    dataset: DemonstrationDataset,
    path: PathData,
    sample_index: int,
) -> np.ndarray:
    rebuilt = DemonstrationDataset([path], spec=dataset.spec)
    state = rebuilt.assemble_state(rebuilt.entries[0], sample_index)
    return np.asarray(policy.act(state), dtype=np.float64)


def _constraints_of(dataset: DemonstrationDataset):
    from ..expert.constraints import PortfolioConstraints

    settings = dataset.entries[0].data.metadata["expert"]["constraints"]
    return PortfolioConstraints(
        long_only=bool(settings["long_only"]),
        min_weight=float(settings["min_weight"]),
        max_weight=float(settings["max_weight"]),
        leverage_limit=(
            None
            if settings.get("leverage_limit") is None
            else float(settings["leverage_limit"])
        ),
    )
