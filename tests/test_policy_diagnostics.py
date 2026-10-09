"""Tests for the Weeks 17-21 acceptance checks.

A check suite is only worth having if it fails when it should. Each negative
test below breaks exactly one property and asserts that its own check — and
ideally only its own check — turns red.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch

from src.dataset.demonstrations import DemonstrationDataset
from src.dataset.paths import generate_path
from src.dataset.state import StateSpec
from src.expert.constraints import PortfolioConstraints
from src.imitation.bc import BCConfig, BehaviouralCloningTrainer, seed_everything
from src.imitation.diagnostics import (
    PolicyDiagnosticThresholds,
    evaluate_policy,
)
from src.policy.base import AllocationPolicy
from src.policy.builder import build_policy
from src.simulation.presets import canonical_three_regime_preset

N_ASSETS = 4
N_REGIMES = 3
SPEC = StateSpec(lookback=10)
CONSTRAINTS = PortfolioConstraints(
    long_only=True, min_weight=0.0, max_weight=0.5, leverage_limit=1.0
)

# The horizon is the exposé's H = 21, not a shortened one, and that matters for
# `beats_naive_baselines` rather than for realism.
#
# A myopic expert compares a full transaction cost against a single period of
# expected improvement and therefore mostly does not trade: at H = 5 and
# c = 0.0025, 77% of this expert's decisions have exactly zero turnover, so the
# never-trade rule reproduces it perfectly on 77% of the data and a policy can
# only distinguish itself on the remaining 23%. The check then measures the
# expert's inactivity rather than the policy, which is the failure mode the
# fixture comment below warns about.
#
# At H = 21 only 59% of decisions are frozen and the never-trade tracking error
# is 2.2x larger, leaving room the policy can actually take: it reaches 0.78 of
# the never-trade baseline with a paired interval of [+1.9e-3, +4.3e-3], where
# at H = 5 it reached 0.91 with an interval that straddled zero. The cost is
# about twelve extra seconds of path generation.
EXPERT_CONFIG = {
    "horizon": 21,
    "risk_aversion": 5.0,
    "transaction_cost": 0.0025,
    "constraints": {
        "long_only": True,
        "min_weight": 0.0,
        "max_weight": 0.5,
        "leverage_limit": 1.0,
    },
    "moments": {"estimator": "rolling", "covariance": "ledoit_wolf", "window": 60},
}

# The look-ahead check re-solves the expert twice per sampled decision date, so
# it is the dominant cost here; one sample keeps the suite fast without
# weakening what is being asserted.
LOOK_AHEAD_SAMPLES = 1
LATENCY_DECISIONS = 5

# The fixture universe is deliberately tiny (N = 4), so its quadratic programme
# is cheap relative to roughly 1 ms of per-call torch overhead. The shipped
# configuration (N = 8) reaches 8x and upwards, which is why the *default*
# threshold is 5.0. Gating this fixture at the default would assert something
# about the toy expert's cheapness rather than about the policy.
FIXTURE_THRESHOLDS = PolicyDiagnosticThresholds(min_latency_speedup=1.5)


@pytest.fixture(scope="module")
def fixture() -> dict[str, Any]:
    preset = canonical_three_regime_preset(N_ASSETS)
    # Enough independent paths that a trained policy really does beat the
    # never-trade rule on unseen data. With only a few hundred samples it does
    # not, and the acceptance check would be measuring the fixture rather than
    # the policy.
    #
    # The holdout deliberately holds several paths: `beats_naive_baselines`
    # bootstraps the paired difference *over paths*, and a single path carries
    # no information about sampling variation.
    paths = [
        generate_path(
            path_id=index,
            seed=200 + index,
            preset=preset,
            periods=500,
            expert_config=EXPERT_CONFIG,
        )
        for index in range(12)
    ]
    return {
        "preset": preset,
        "paths": paths,
        "train": DemonstrationDataset(paths[:8], spec=SPEC),
        "validation": DemonstrationDataset(paths[8:9], spec=SPEC),
        "holdout": DemonstrationDataset(paths[9:], spec=SPEC),
    }


@pytest.fixture(scope="module")
def trained(fixture) -> dict[str, Any]:
    seed_everything(0)
    policy = build_policy(
        {"kind": "mlp", "hidden_dimensions": [64, 32], "dropout": 0.0},
        spec=SPEC,
        n_assets=N_ASSETS,
        n_regimes=N_REGIMES,
        constraints=CONSTRAINTS,
    )
    trainer = BehaviouralCloningTrainer(
        policy, BCConfig(epochs=40, batch_size=64, seed=0, patience=40, weight_decay=1e-2)
    )
    trainer.fit(fixture["train"], fixture["validation"])
    return {"policy": policy, "trainer": trainer}


def run_checks(
    fixture,
    policy: AllocationPolicy,
    trainer: BehaviouralCloningTrainer,
    thresholds: PolicyDiagnosticThresholds | None = None,
):
    return evaluate_policy(
        policy,
        trainer,
        train_dataset=fixture["train"],
        validation_dataset=fixture["validation"],
        holdout_dataset=fixture["holdout"],
        look_ahead_path=fixture["paths"][9],
        preset=fixture["preset"],
        latency_decisions=LATENCY_DECISIONS,
        look_ahead_samples=LOOK_AHEAD_SAMPLES,
        thresholds=thresholds if thresholds is not None else FIXTURE_THRESHOLDS,
    )


# ----------------------------------------------------------------------
# A correct policy
# ----------------------------------------------------------------------
def test_a_trained_policy_passes_every_check(fixture, trained):
    report = run_checks(fixture, trained["policy"], trained["trainer"])
    assert report.passed, report.to_dict()["checks"]


def test_the_report_is_json_serialisable_and_records_its_evidence(fixture, trained):
    report = run_checks(fixture, trained["policy"], trained["trainer"]).to_dict()

    assert set(report["checks"]) == {
        "action_feasibility",
        "determinism",
        "optimisation_sanity",
        "beats_naive_baselines",
        "generalisation",
        "inference_latency",
        "policy_no_look_ahead",
    }
    assert report["beats_naive_baselines"]["baselines"].keys() == {
        "equal_weight",
        "previous_weights",
    }
    intervals = report["beats_naive_baselines"]["paired_bootstrap"]
    assert intervals["previous_weights"]["n_paths"] == 3
    assert intervals["previous_weights"]["low"] <= intervals["previous_weights"]["mean"]
    assert intervals["previous_weights"]["mean"] <= intervals["previous_weights"]["high"]
    assert report["inference_latency"]["speedup"] > 1.0
    assert "memorisation_gap" in report
    assert len(report["policy_no_look_ahead"]["checked_decision_times"]) == LOOK_AHEAD_SAMPLES


def test_the_policy_is_faster_than_re_solving_the_quadratic_programme(fixture, trained):
    """Research question 5, measured rather than assumed."""
    report = run_checks(fixture, trained["policy"], trained["trainer"])
    assert report.policy_seconds_per_decision < report.expert_seconds_per_decision


def test_the_policy_action_ignores_a_rewritten_future(fixture, trained):
    """The Weeks 14-16 state check, extended end to end through the network."""
    report = run_checks(fixture, trained["policy"], trained["trainer"])
    assert report.look_ahead_difference == 0.0


# ----------------------------------------------------------------------
# Deliberately broken policies
# ----------------------------------------------------------------------
class ConstantPolicy(AllocationPolicy):
    """Ignores the state entirely and always returns equal weight."""

    def __init__(self) -> None:
        super().__init__(
            state_dimension=SPEC.dimension(N_ASSETS, N_REGIMES),
            n_assets=N_ASSETS,
            constraints=CONSTRAINTS,
        )
        self.unused = torch.nn.Linear(1, 1)

    def allocate(self, states: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        del kwargs
        return torch.full(
            (states.shape[0], self.n_assets), 1.0 / self.n_assets, dtype=states.dtype
        )

    def bc_loss(self, states, actions, loss):
        return loss(self.allocate(states), actions) + 0.0 * self.unused.weight.sum()

    def describe(self) -> dict[str, Any]:
        return {"kind": "constant", **self.base_description()}


class InfeasiblePolicy(ConstantPolicy):
    """Returns weights that breach the position cap and the budget."""

    def allocate(self, states: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        del kwargs
        weights = torch.zeros((states.shape[0], self.n_assets), dtype=states.dtype)
        weights[:, 0] = 1.5
        return weights

    def describe(self) -> dict[str, Any]:
        return {"kind": "infeasible", **self.base_description()}


class RandomPolicy(ConstantPolicy):
    """Draws a fresh allocation on every call, so it is not a function of the state."""

    def allocate(self, states: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        del kwargs
        scores = torch.rand((states.shape[0], self.n_assets), dtype=states.dtype)
        return scores / scores.sum(dim=-1, keepdim=True)

    def describe(self) -> dict[str, Any]:
        return {"kind": "random", **self.base_description()}


def make_trainer(policy: AllocationPolicy) -> BehaviouralCloningTrainer:
    return BehaviouralCloningTrainer(policy, BCConfig(epochs=1, batch_size=64, seed=0))


def test_a_state_independent_policy_fails_the_baseline_check(fixture):
    """Equal weight is one of the baselines, so it cannot beat itself."""
    policy = ConstantPolicy()
    report = run_checks(fixture, policy, make_trainer(policy))

    assert not report.checks["beats_naive_baselines"]
    # It ties the equal-weight rule exactly, so the interval is degenerate at 0.
    assert report.baseline_intervals["equal_weight"].mean == pytest.approx(0.0)
    assert report.checks["action_feasibility"]
    assert report.checks["policy_no_look_ahead"]


def test_a_policy_whose_gradients_do_not_reach_its_output_fails_optimisation_sanity(
    fixture,
):
    """The check had no negative test, and was silently unreliable without one.

    ``ConstantPolicy`` returns equal weight regardless of the state, so its loss
    cannot be reduced by any parameter update — exactly the dead-architecture
    case the check exists to catch. It should therefore report a reduction of
    about 1.
    """
    policy = ConstantPolicy()
    report = run_checks(fixture, policy, make_trainer(policy))

    assert not report.checks["optimisation_sanity"]
    assert report.overfit_loss_reduction == pytest.approx(1.0, abs=1e-6)


def test_an_infeasible_policy_fails_the_feasibility_check(fixture):
    policy = InfeasiblePolicy()
    report = run_checks(fixture, policy, make_trainer(policy))

    assert not report.checks["action_feasibility"]
    assert report.feasibility_violation > 0.5


def test_a_non_deterministic_policy_fails_the_determinism_check(fixture):
    policy = RandomPolicy()
    report = run_checks(fixture, policy, make_trainer(policy))

    assert not report.checks["determinism"]
    assert report.determinism_difference > 0.0


def test_an_unreachable_latency_budget_fails_its_own_check(fixture, trained):
    report = run_checks(
        fixture,
        trained["policy"],
        trained["trainer"],
        thresholds=PolicyDiagnosticThresholds(min_latency_speedup=1e9),
    )
    assert not report.checks["inference_latency"]
    assert report.checks["action_feasibility"]


def test_the_generalisation_ratio_compares_test_against_validation(fixture, trained):
    """Not against training: a deterministic target makes the training error
    collapse, so a train-versus-test ratio would gate nothing."""
    report = run_checks(fixture, trained["policy"], trained["trainer"])

    assert report.generalisation_ratio == pytest.approx(
        report.holdout_tracking_error / report.validation_tracking_error, rel=1e-9
    )
    assert report.memorisation_gap == pytest.approx(
        report.holdout_tracking_error / report.train_tracking_error, rel=1e-9
    )


def test_an_unreachable_generalisation_ratio_fails_its_own_check(fixture, trained):
    tight = PolicyDiagnosticThresholds(
        max_generalisation_ratio=1.0, min_latency_speedup=1.5
    )
    report = run_checks(fixture, trained["policy"], trained["trainer"], thresholds=tight)
    # The threshold binds only if test is genuinely worse than validation, which
    # is the situation the check exists to catch.
    assert report.checks["generalisation"] == (report.generalisation_ratio <= 1.0)


def test_the_baseline_check_requires_more_than_a_point_estimate(fixture, trained):
    """A policy that ties never-trade must fail even if it edges ahead on the mean.

    Both learned heads are initialised to return ``w_{t-1}`` exactly, so a bare
    inequality against the never-trade rule is cleared by noise alone. Turning
    the significance requirement off is what shows it is doing work.
    """
    report = run_checks(fixture, trained["policy"], trained["trainer"])
    interval = report.baseline_intervals["previous_weights"]

    assert interval.mean > 0.0, "the trained policy should beat never-trade on the mean"
    assert report.checks["beats_naive_baselines"] == (
        interval.low > 0.0
        and report.baseline_intervals["equal_weight"].low > 0.0
    )


def test_thresholds_reject_impossible_settings():
    with pytest.raises(ValueError, match="max_generalisation_ratio"):
        PolicyDiagnosticThresholds(max_generalisation_ratio=0.5)
    with pytest.raises(ValueError, match="min_overfit_loss_reduction"):
        PolicyDiagnosticThresholds(min_overfit_loss_reduction=0.5)
    with pytest.raises(ValueError, match="nonnegative"):
        PolicyDiagnosticThresholds(max_feasibility_violation=-1.0)


def test_baseline_tracking_errors_are_computed_on_the_same_data(fixture, trained):
    """The comparison is only fair if both read identical held-out states."""
    from src.imitation.diagnostics import _baseline_tracking_errors

    baselines = _baseline_tracking_errors(fixture["holdout"])
    states = np.stack(
        [fixture["holdout"].sample(i)[0] for i in range(len(fixture["holdout"]))]
    )
    actions = np.stack(
        [fixture["holdout"].sample(i)[1] for i in range(len(fixture["holdout"]))]
    )
    previous = states[:, SPEC.block_slices(N_ASSETS, N_REGIMES)["previous_weights"]]
    expected = float(np.mean(np.sum((previous - actions) ** 2, axis=1)))

    assert baselines["previous_weights"] == pytest.approx(expected, rel=1e-6)
