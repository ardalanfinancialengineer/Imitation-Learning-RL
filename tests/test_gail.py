"""Tests for the GAIL loop and its acceptance checks."""

from __future__ import annotations

import numpy as np
import pytest

from src.dataset.demonstrations import DemonstrationDataset
from src.dataset.paths import generate_path
from src.dataset.state import StateSpec
from src.expert.constraints import PortfolioConstraints
from src.imitation.bc import BCConfig, BehaviouralCloningTrainer, seed_everything
from src.imitation.critic import CriticConfig
from src.imitation.gail import GAILConfig, GAILTrainer
from src.imitation.gail_diagnostics import (
    GAILDiagnosticThresholds,
    compare_policies,
    evaluate_gail,
)
from src.imitation.rollout import ExpertOracle
from src.policy.builder import build_policy
from src.simulation.presets import canonical_three_regime_preset

N_ASSETS = 4
SPEC = StateSpec(lookback=10)
CONSTRAINTS = PortfolioConstraints(long_only=True, min_weight=0.0, max_weight=0.4)
EXPERT_CONFIG = {
    "horizon": 5,
    "risk_aversion": 5.0,
    "transaction_cost": 0.0025,
    "constraints": {"long_only": True, "min_weight": 0.0, "max_weight": 0.4},
    "moments": {"estimator": "rolling", "covariance": "ledoit_wolf", "window": 60},
}


@pytest.fixture(scope="module")
def fixture():
    preset = canonical_three_regime_preset(N_ASSETS)
    paths = [
        generate_path(
            path_id=index,
            seed=42 + index,
            preset=preset,
            periods=280,
            expert_config=EXPERT_CONFIG,
        )
        for index in range(4)
    ]
    train = DemonstrationDataset(paths[:3], spec=SPEC)
    validation = DemonstrationDataset(paths[3:], spec=SPEC)

    seed_everything(0)
    policy = build_policy(
        {"kind": "mlp", "hidden_dimensions": [32, 16], "dropout": 0.0},
        spec=SPEC,
        n_assets=N_ASSETS,
        n_regimes=3,
        constraints=CONSTRAINTS,
    )
    BehaviouralCloningTrainer(policy, BCConfig(epochs=4, batch_size=64, seed=0)).fit(
        train, validation
    )
    return preset, policy, train, validation


def _trainer(policy, **kwargs) -> GAILTrainer:
    critic = kwargs.pop("critic", {})
    return GAILTrainer(
        policy,
        config=GAILConfig(
            iterations=kwargs.pop("iterations", 4),
            segment_length=8,
            evaluate_every=2,
            seed=0,
            **kwargs,
        ),
        critic_config=CriticConfig(steps_per_round=2, learning_rate=1e-3, **critic),
    )


def test_the_loop_records_every_iteration(fixture):
    _, policy, train, validation = fixture
    trainer = _trainer(policy)
    history = trainer.fit(train, validation)

    assert len(history.records) == 4
    assert [record.iteration for record in history.records] == [1, 2, 3, 4]
    assert not history.diverged


def test_every_record_carries_the_stability_signals(fixture):
    _, policy, train, validation = fixture
    history = _trainer(policy).fit(train, validation)
    record = history.records[0]

    for name in (
        "wasserstein_estimate",
        "gradient_penalty",
        "critic_gradient_norm",
        "policy_gradient_norm",
        "measured_lipschitz",
    ):
        assert np.isfinite(getattr(record, name))


def test_stability_summary_describes_the_wasserstein_trace(fixture):
    _, policy, train, validation = fixture
    history = _trainer(policy, iterations=8).fit(train, validation)
    stability = history.stability()

    assert stability["wasserstein_volatility"] >= 0.0
    assert np.isfinite(stability["wasserstein_final"])
    assert stability["max_measured_lipschitz"] > 0.0


def test_the_returned_policy_is_the_best_iteration(fixture):
    _, policy, train, validation = fixture
    trainer = _trainer(policy, iterations=6)
    history = trainer.fit(train, validation)

    final = trainer._evaluate(validation, ExpertOracle(validation))
    assert final["closed_loop_tracking_error"] == pytest.approx(
        history.best_closed_loop_error, rel=1e-9
    )


def test_gail_keeps_actions_feasible(fixture):
    _, policy, train, validation = fixture
    trainer = _trainer(policy)
    trainer.fit(train, validation)

    from src.imitation.rollout import rollout_policy

    for rollout in rollout_policy(trainer.policy, validation):
        assert CONSTRAINTS.is_feasible(rollout.policy_weights)


def test_the_run_is_reproducible(fixture):
    _, policy, train, validation = fixture
    from copy import deepcopy

    first = _trainer(deepcopy(policy)).fit(train, validation)
    second = _trainer(deepcopy(policy)).fit(train, validation)
    assert first.series("wasserstein_estimate").tolist() == pytest.approx(
        second.series("wasserstein_estimate").tolist()
    )


def test_both_lipschitz_methods_run(fixture):
    _, policy, train, validation = fixture
    from copy import deepcopy

    for method in ("gradient_penalty", "weight_clipping"):
        history = _trainer(
            deepcopy(policy), critic={"lipschitz": method}
        ).fit(train, validation)
        assert not history.diverged
        assert len(history.records) == 4


def test_occupancy_distance_is_zero_against_the_experts_own_actions(fixture):
    """A critic scores the gap; scoring a distribution against itself gives zero."""
    _, policy, train, validation = fixture
    trainer = _trainer(policy)
    trainer.fit(train, validation)

    import torch

    from src.imitation.differentiable import build_tape

    tape = build_tape(validation)
    expert_states = torch.cat([tape.expert_state(s) for s in range(tape.n_steps)])
    expert_actions = tape.expert_actions.reshape(-1, tape.n_assets)
    with torch.no_grad():
        estimate = trainer.critic.wasserstein_estimate(
            expert_states, expert_actions, expert_states, expert_actions
        )
    assert float(estimate) == pytest.approx(0.0, abs=1e-5)


def test_compare_policies_scores_everything_with_one_critic(fixture):
    _, policy, train, validation = fixture
    from copy import deepcopy

    trainer = _trainer(policy)
    trainer.fit(train, validation)
    original = trainer.policy

    distances = compare_policies(
        trainer, {"a": deepcopy(policy), "b": trainer.policy}, validation
    )
    assert set(distances) == {"a", "b"}
    assert all(np.isfinite(value) for value in distances.values())
    assert trainer.policy is original, "the trainer's own policy must be restored"


def test_rejects_invalid_configuration():
    with pytest.raises(ValueError):
        GAILConfig(iterations=0)
    with pytest.raises(ValueError):
        GAILConfig(segment_length=0)
    with pytest.raises(ValueError):
        GAILConfig(policy_learning_rate=0.0)
    with pytest.raises(ValueError):
        GAILConfig(bc_anchor_weight=-1.0)


# ----------------------------------------------------------------------
# Acceptance checks
# ----------------------------------------------------------------------
def test_a_correct_run_passes_every_check(fixture):
    preset, policy, train, validation = fixture
    trainer = _trainer(policy)
    history = trainer.fit(train, validation)

    diagnostics = evaluate_gail(
        trainer, history, train, validation, preset=preset, gradient_parameters=3
    )
    assert diagnostics.passed, diagnostics.checks
    assert set(diagnostics.checks) == {
        "rollout_gradient_correctness",
        "state_splice_exactness",
        "critic_lipschitz",
        "action_feasibility",
        "training_stability",
        "rollout_no_look_ahead",
        "rollout_determinism",
    }


def test_the_gradient_check_is_tight(fixture):
    """The claim that backprop replaces the policy gradient rests on this."""
    preset, policy, train, validation = fixture
    trainer = _trainer(policy)
    history = trainer.fit(train, validation)
    diagnostics = evaluate_gail(
        trainer, history, train, validation, preset=preset, gradient_parameters=4
    )
    assert diagnostics.gradient_relative_error < 1e-6
    assert diagnostics.splice_difference < 1e-9
    assert diagnostics.checked_parameters > 0


def test_a_diverged_run_fails_the_stability_check(fixture):
    preset, policy, train, validation = fixture
    trainer = _trainer(policy)
    history = trainer.fit(train, validation)
    history.diverged = True

    diagnostics = evaluate_gail(
        trainer, history, train, validation, preset=preset, gradient_parameters=2
    )
    assert not diagnostics.checks["training_stability"]
    assert not diagnostics.passed


def test_an_unreachable_gradient_tolerance_fails_its_own_check(fixture):
    preset, policy, train, validation = fixture
    trainer = _trainer(policy)
    history = trainer.fit(train, validation)

    diagnostics = evaluate_gail(
        trainer,
        history,
        train,
        validation,
        preset=preset,
        gradient_parameters=2,
        thresholds=GAILDiagnosticThresholds(max_gradient_relative_error=0.0),
    )
    assert not diagnostics.checks["rollout_gradient_correctness"]


def test_report_is_json_serialisable(fixture):
    import json

    preset, policy, train, validation = fixture
    trainer = _trainer(policy)
    history = trainer.fit(train, validation)
    report = evaluate_gail(
        trainer, history, train, validation, preset=preset, gradient_parameters=2
    ).to_dict()
    assert json.loads(json.dumps(report))["passed"] is True
