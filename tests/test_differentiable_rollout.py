"""Tests for the differentiable closed-loop rollout.

The gradient test is the important one: replacing GAIL's policy gradient with
backpropagation is only legitimate if the backpropagated gradient is actually
right, so it is checked against central finite differences rather than argued
for.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.dataset.demonstrations import DemonstrationDataset
from src.dataset.paths import generate_path
from src.dataset.state import StateSpec
from src.expert.constraints import PortfolioConstraints
from src.imitation.bc import BCConfig, BehaviouralCloningTrainer, seed_everything
from src.imitation.differentiable import (
    build_tape,
    realised_net_returns,
    rollout_segment,
    rollout_tape,
)
from src.imitation.rollout import rollout_policy
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
def dataset() -> DemonstrationDataset:
    preset = canonical_three_regime_preset(N_ASSETS)
    paths = [
        generate_path(
            path_id=index,
            seed=42 + index,
            preset=preset,
            periods=260,
            expert_config=EXPERT_CONFIG,
        )
        for index in range(3)
    ]
    return DemonstrationDataset(paths, spec=SPEC)


@pytest.fixture(scope="module")
def policy(dataset):
    seed_everything(0)
    model = build_policy(
        {"kind": "mlp", "hidden_dimensions": [32, 16], "dropout": 0.0},
        spec=SPEC,
        n_assets=N_ASSETS,
        n_regimes=3,
        constraints=CONSTRAINTS,
    )
    BehaviouralCloningTrainer(model, BCConfig(epochs=3, batch_size=64, seed=0)).fit(
        dataset, dataset
    )
    return model


def test_tape_shapes_match_the_dataset(dataset):
    tape = build_tape(dataset)
    assert tape.n_paths == 3
    assert tape.n_assets == N_ASSETS
    assert tape.state_dimension == dataset.state_dimension
    assert tape.n_steps == dataset.entries[0].indices.size


def test_spliced_state_equals_the_assembler(dataset):
    """The splice is an optimisation; it must not change what the policy sees."""
    tape = build_tape(dataset, dtype=torch.float64)
    entry = dataset.entries[0]
    for step in (0, 5, tape.n_steps - 1):
        spliced = tape.expert_state(step)[0].numpy()
        reference = dataset.assemble_state(entry, int(entry.indices[step]))
        np.testing.assert_allclose(spliced, reference, rtol=0, atol=1e-12)


def test_splicing_a_different_anchor_changes_only_that_block(dataset):
    tape = build_tape(dataset, dtype=torch.float64)
    anchor = torch.full((tape.n_paths, N_ASSETS), 0.25, dtype=torch.float64)
    swapped = tape.state(0, anchor)[0].numpy()
    original = tape.expert_state(0)[0].numpy()

    block = dataset.assembler.block_slices()["previous_weights"]
    np.testing.assert_allclose(swapped[block], 0.25)
    np.testing.assert_allclose(swapped[: block.start], original[: block.start])
    np.testing.assert_allclose(swapped[block.stop :], original[block.stop :])


def test_rollout_gradients_match_finite_differences(dataset, policy):
    """The load-bearing check for replacing the policy gradient with backprop.

    Run in float64: in float32 the finite-difference estimate itself carries
    about 1e-2 relative error, which would hide a genuinely wrong gradient.
    """
    probe = build_policy(
        {"kind": "mlp", "hidden_dimensions": [32, 16], "dropout": 0.0},
        spec=SPEC,
        n_assets=N_ASSETS,
        n_regimes=3,
        constraints=CONSTRAINTS,
    )
    probe.load_state_dict(policy.state_dict())
    probe = probe.double()
    tape = build_tape(dataset, dtype=torch.float64, max_steps=12)

    def objective() -> torch.Tensor:
        rolled = rollout_segment(probe, tape, tape.initial_anchor, 0, 10)
        return (rolled.actions**2).sum() + rolled.turnover.sum()

    probe.zero_grad(set_to_none=True)
    objective().backward()

    target = max(
        (p for p in probe.parameters() if p.grad is not None and p.numel() > 1),
        key=lambda p: float(p.grad.abs().max()),
    )
    gradients = target.grad.detach().reshape(-1).clone()
    flat = target.data.reshape(-1)

    epsilon = 1e-6
    for index in torch.argsort(gradients.abs(), descending=True)[:4]:
        original = flat[index].item()
        flat[index] = original + epsilon
        plus = float(objective().detach())
        flat[index] = original - epsilon
        minus = float(objective().detach())
        flat[index] = original

        estimate = (plus - minus) / (2.0 * epsilon)
        analytic = gradients[index].item()
        assert abs(estimate - analytic) / max(abs(estimate), 1e-12) < 1e-6


def test_the_rollout_carries_gradient_through_the_anchor_chain(dataset, policy):
    """A later action must depend on an earlier one, or there is no BPTT at all."""
    tape = build_tape(dataset, dtype=torch.float64, max_steps=6)
    probe = build_policy(
        {"kind": "mlp", "hidden_dimensions": [32, 16], "dropout": 0.0},
        spec=SPEC,
        n_assets=N_ASSETS,
        n_regimes=3,
        constraints=CONSTRAINTS,
    )
    probe.load_state_dict(policy.state_dict())
    probe = probe.double()

    anchor = tape.initial_anchor.clone().requires_grad_(True)
    rolled = rollout_segment(probe, tape, anchor, 0, 5)
    # Not `actions.sum()`: the budget constraint pins that to 1, so it is a
    # constant with identically zero gradient no matter how well BPTT works.
    (rolled.actions[-1] ** 2).sum().backward()

    assert anchor.grad is not None
    assert float(anchor.grad.abs().sum()) > 0.0


def test_detached_rollout_matches_the_numpy_rollout(dataset, policy):
    """The tape rollout and the Weeks 22-25 rollout must agree on the trajectory."""
    tape = build_tape(dataset)
    rolled = rollout_tape(policy, tape)
    reference = rollout_policy(policy, dataset)

    for column, path_rollout in enumerate(reference):
        np.testing.assert_allclose(
            rolled.actions[:, column].numpy(),
            path_rollout.policy_weights,
            atol=1e-5,
        )


def test_detached_rollout_builds_no_graph(dataset, policy):
    rolled = rollout_tape(policy, build_tape(dataset))
    assert not rolled.actions.requires_grad


def test_realised_net_returns_charge_the_cost_once(dataset, policy):
    tape = build_tape(dataset)
    rolled = rollout_tape(policy, tape)
    net = realised_net_returns(tape, rolled, 0.0025)

    gross = (rolled.actions * tape.realised_returns).sum(dim=-1)
    expected = gross - 0.0025 * rolled.turnover
    assert torch.allclose(net, expected)


def test_build_tape_rejects_misaligned_paths(dataset):
    preset = canonical_three_regime_preset(N_ASSETS)
    shorter = generate_path(
        path_id=99,
        seed=7,
        preset=preset,
        periods=200,
        expert_config=EXPERT_CONFIG,
    )
    mixed = DemonstrationDataset(
        [dataset.entries[0].data, shorter], spec=SPEC
    )
    with pytest.raises(ValueError, match="equal decision counts"):
        build_tape(mixed)


def test_build_tape_requires_the_previous_weights_block(dataset):
    """Without the anchor block the rollout carries no action dependence."""
    spec = StateSpec(lookback=10, include_previous_weights=False)
    stripped = DemonstrationDataset([entry.data for entry in dataset.entries], spec=spec)
    with pytest.raises(ValueError, match="previous-weights block"):
        build_tape(stripped)


def test_segment_must_contain_a_step(dataset, policy):
    tape = build_tape(dataset)
    with pytest.raises(ValueError):
        rollout_segment(policy, tape, tape.initial_anchor, 0, 0)
