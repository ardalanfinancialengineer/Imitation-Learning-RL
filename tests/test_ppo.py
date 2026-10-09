"""Tests for the PPO trainer, the exact-gradient baseline, and their checks."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
import torch

from src.dataset.demonstrations import DemonstrationDataset
from src.dataset.paths import generate_path
from src.dataset.state import StateSpec
from src.expert.constraints import PortfolioConstraints
from src.imitation.bc import BCConfig, BehaviouralCloningTrainer, seed_everything
from src.imitation.differentiable import build_tape
from src.policy.builder import build_policy
from src.reinforcement.analytic import AnalyticConfig, AnalyticRewardTrainer
from src.reinforcement.diagnostics import (
    PPODiagnosticThresholds,
    evaluate_ppo,
)
from src.reinforcement.ppo import PPOConfig, PPOTrainer
from src.reinforcement.reward import RewardConfig
from src.reinforcement.stochastic import ActorCriticConfig, GaussianActorCritic
from src.simulation.presets import canonical_three_regime_preset

N_ASSETS = 4
SPEC = StateSpec(lookback=10)
CONSTRAINTS = PortfolioConstraints(long_only=True, min_weight=0.0, max_weight=0.4)
REWARD = RewardConfig(risk_aversion=5.0, transaction_cost=0.0025)
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
            periods=300,
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
    return policy, train, validation


def _trainer(policy, **kwargs) -> PPOTrainer:
    return PPOTrainer(
        GaussianActorCritic(deepcopy(policy), config=ActorCriticConfig()),
        REWARD,
        config=PPOConfig(iterations=kwargs.pop("iterations", 3), seed=0, **kwargs),
    )


# ----------------------------------------------------------------------
# The loop
# ----------------------------------------------------------------------
def test_the_loop_records_every_iteration(fixture):
    policy, train, validation = fixture
    history = _trainer(policy).fit(train, validation)

    assert len(history.records) == 3
    assert [record.iteration for record in history.records] == [1, 2, 3]


def test_every_record_carries_the_reported_measures(fixture):
    policy, train, validation = fixture
    record = _trainer(policy).fit(train, validation).records[0]

    for name in (
        "certainty_equivalent",
        "mean_turnover",
        "approximate_kl",
        "clip_fraction",
        "explained_variance",
        "feasibility_violation",
    ):
        assert np.isfinite(getattr(record, name))


def test_constraints_hold_at_every_iteration(fixture):
    """'Check turnover, constraints, and utility results' - the constraints part."""
    policy, train, validation = fixture
    history = _trainer(policy).fit(train, validation)
    assert all(record.feasibility_violation == 0.0 for record in history.records)


def test_the_returned_policy_is_the_best_iteration(fixture):
    policy, train, validation = fixture
    trainer = _trainer(policy, iterations=5)
    history = trainer.fit(train, validation)

    tape = build_tape(validation, with_covariances=True)
    assert trainer.evaluate(tape)["certainty_equivalent"] == pytest.approx(
        history.best_certainty_equivalent, rel=1e-9
    )


def test_the_run_is_reproducible(fixture):
    policy, train, validation = fixture
    first = _trainer(policy).fit(train, validation)
    second = _trainer(policy).fit(train, validation)
    assert first.series("certainty_equivalent").tolist() == pytest.approx(
        second.series("certainty_equivalent").tolist()
    )


def test_evaluation_uses_the_deterministic_policy(fixture):
    """Reporting the exploring policy would understate the deployed one."""
    policy, train, validation = fixture
    trainer = _trainer(policy)
    tape = build_tape(validation, with_covariances=True)
    assert trainer.evaluate(tape) == trainer.evaluate(tape)


def test_advantages_use_a_terminal_value_of_zero(fixture):
    """Each path is a complete episode, so the last step does not bootstrap."""
    policy, train, validation = fixture
    trainer = _trainer(policy)
    rewards = torch.zeros(3, 2)
    rewards[-1] = 1.0
    values = torch.zeros(3, 2)

    advantages, returns = trainer._advantages(rewards, values)
    assert torch.allclose(advantages[-1], torch.ones(2))
    assert torch.allclose(returns[-1], torch.ones(2))


def test_rejects_invalid_configuration():
    with pytest.raises(ValueError):
        PPOConfig(iterations=0)
    with pytest.raises(ValueError):
        PPOConfig(clip_range=0.0)
    with pytest.raises(ValueError):
        PPOConfig(clip_range=1.0)
    with pytest.raises(ValueError):
        PPOConfig(discount=1.5)
    with pytest.raises(ValueError):
        PPOConfig(learning_rate=0.0)


# ----------------------------------------------------------------------
# Exact-gradient baseline
# ----------------------------------------------------------------------
def test_the_exact_gradient_ascends_the_reward_it_optimises(fixture):
    """The claim behind the baseline: exact gradients climb the training reward.

    Measured on a *fixed* segment, because ``fit`` samples a random segment each
    iteration and then returns the policy selected on validation utility — so
    training reward after ``fit`` is not what this mechanism predicts and is not
    what should be asserted.
    """
    from src.imitation.differentiable import rollout_segment
    from src.reinforcement.reward import tape_rewards

    policy, train, _ = fixture
    probe = deepcopy(policy)
    tape = build_tape(train, with_covariances=True, max_steps=24)
    optimiser = torch.optim.AdamW(probe.parameters(), lr=1e-3)

    def segment_reward() -> torch.Tensor:
        rolled = rollout_segment(probe, tape, tape.expert_anchors[0], 0, 16)
        return tape_rewards(tape, rolled.actions, rolled.anchors, REWARD).mean()

    before = float(segment_reward().detach())
    for _ in range(25):
        optimiser.zero_grad(set_to_none=True)
        (-segment_reward()).backward()
        optimiser.step()
    assert float(segment_reward().detach()) > before


def test_the_analytic_baseline_keeps_actions_feasible(fixture):
    policy, train, validation = fixture
    trainer = AnalyticRewardTrainer(
        deepcopy(policy), REWARD, config=AnalyticConfig(iterations=10, seed=0)
    )
    trainer.fit(train, validation)

    from src.imitation.differentiable import rollout_tape

    tape = build_tape(validation, with_covariances=True)
    actions = rollout_tape(trainer.policy, tape).actions
    assert CONSTRAINTS.is_feasible(
        actions.reshape(-1, N_ASSETS).numpy().astype(np.float64)
    )


def test_analytic_rejects_invalid_configuration():
    with pytest.raises(ValueError):
        AnalyticConfig(iterations=0)
    with pytest.raises(ValueError):
        AnalyticConfig(learning_rate=-1.0)


# ----------------------------------------------------------------------
# Acceptance checks
# ----------------------------------------------------------------------
def test_a_correct_run_passes_every_check(fixture):
    policy, train, validation = fixture
    reference = deepcopy(policy)
    trainer = _trainer(policy)
    history = trainer.fit(train, validation)

    diagnostics = evaluate_ppo(trainer, history, validation, reference)
    assert diagnostics.passed, diagnostics.checks
    assert set(diagnostics.checks) == {
        "reward_matches_environment",
        "action_feasibility",
        "log_probability_correctness",
        "warm_start_fidelity",
        "training_stability",
        "rollout_determinism",
    }


def test_the_two_reward_implementations_agree_tightly(fixture):
    """Equation (11) exists in PortfolioEnv and over the tape; they must match."""
    policy, train, validation = fixture
    reference = deepcopy(policy)
    trainer = _trainer(policy)
    history = trainer.fit(train, validation)

    diagnostics = evaluate_ppo(trainer, history, validation, reference)
    assert diagnostics.reward_difference < 1e-9
    assert diagnostics.log_probability_difference < 1e-5
    assert diagnostics.warm_start_difference < 1e-6


def test_an_unreachable_reward_tolerance_fails_its_own_check(fixture):
    policy, train, validation = fixture
    reference = deepcopy(policy)
    trainer = _trainer(policy)
    history = trainer.fit(train, validation)

    diagnostics = evaluate_ppo(
        trainer,
        history,
        validation,
        reference,
        thresholds=PPODiagnosticThresholds(max_reward_difference=0.0),
    )
    assert diagnostics.checks["reward_matches_environment"] == (
        diagnostics.reward_difference == 0.0
    )


def test_a_diverged_run_fails_the_stability_check(fixture):
    policy, train, validation = fixture
    reference = deepcopy(policy)
    trainer = _trainer(policy)
    history = trainer.fit(train, validation)
    object.__setattr__(history.records[0], "approximate_kl", float("nan"))

    diagnostics = evaluate_ppo(trainer, history, validation, reference)
    assert not diagnostics.checks["training_stability"]


def test_report_is_json_serialisable(fixture):
    import json

    policy, train, validation = fixture
    reference = deepcopy(policy)
    trainer = _trainer(policy)
    history = trainer.fit(train, validation)
    report = evaluate_ppo(trainer, history, validation, reference).to_dict()
    assert json.loads(json.dumps(report))["passed"] is True
