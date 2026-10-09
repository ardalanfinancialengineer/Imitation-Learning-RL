"""Tests for the differentiable projection onto the feasible allocation set."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.expert.constraints import PortfolioConstraints
from src.policy.simplex import project_box_simplex, project_simplex


def test_projection_matches_the_numpy_expert_implementation():
    """The policy head and the expert's safety net must agree exactly.

    They solve the same problem by the same method; a divergence would mean the
    policy is trained against one feasible set and evaluated against another.
    """
    generator = np.random.default_rng(0)
    scores = generator.normal(size=(64, 12))
    constraints = PortfolioConstraints(long_only=True, min_weight=0.0, max_weight=0.25)

    projected = project_box_simplex(
        torch.as_tensor(scores, dtype=torch.float64), lower=0.0, upper=0.25
    ).numpy()
    reference = np.stack([constraints.project(row) for row in scores])

    assert np.abs(projected - reference).max() < 1e-12


@pytest.mark.parametrize("upper", [1.0, 0.3, 0.125])
def test_projection_returns_feasible_allocations(upper):
    generator = np.random.default_rng(1)
    # Deliberately extreme inputs: an untrained head produces nothing so tame.
    scores = torch.as_tensor(
        generator.normal(scale=50.0, size=(32, 8)), dtype=torch.float64
    )
    weights = project_box_simplex(scores, lower=0.0, upper=upper).numpy()

    assert np.allclose(weights.sum(axis=1), 1.0, atol=1e-10)
    assert weights.min() >= -1e-12
    assert weights.max() <= upper + 1e-12


def test_projection_leaves_feasible_points_unchanged():
    weights = torch.as_tensor([[0.25, 0.25, 0.25, 0.25]], dtype=torch.float64)
    assert torch.allclose(project_simplex(weights), weights, atol=1e-12)


def test_projection_produces_exact_zeros():
    """Sparsity is the reason a projection is used instead of a softmax."""
    scores = torch.as_tensor([[5.0, 0.0, -5.0, -5.0]], dtype=torch.float64)
    weights = project_simplex(scores)
    assert int((weights == 0.0).sum()) >= 2


def test_projection_gradient_matches_finite_differences():
    """The closed-form threshold is what makes the backward pass correct."""
    torch.manual_seed(0)
    scores = torch.randn(1, 6, dtype=torch.float64, requires_grad=True)
    target = torch.rand(1, 6, dtype=torch.float64)

    def objective(values: torch.Tensor) -> torch.Tensor:
        return ((project_box_simplex(values, lower=0.0, upper=0.4) - target) ** 2).sum()

    objective(scores).backward()
    analytic = scores.grad.clone()

    step = 1e-6
    numeric = torch.zeros_like(analytic)
    with torch.no_grad():
        for index in range(scores.shape[1]):
            shifted = scores.clone()
            shifted[0, index] += step
            forward = objective(shifted)
            shifted[0, index] -= 2 * step
            backward = objective(shifted)
            numeric[0, index] = (forward - backward) / (2 * step)

    assert torch.allclose(analytic, numeric, atol=1e-6)


def test_projection_rejects_an_unfundable_box():
    scores = torch.zeros(1, 4)
    with pytest.raises(ValueError, match="fully invested"):
        project_box_simplex(scores, lower=0.0, upper=0.1)


def test_projection_handles_a_fully_clamped_row():
    """With ``u = 1/N`` every coordinate is pinned and the active set is empty."""
    scores = torch.as_tensor([[3.0, -1.0, 0.5, 0.2]], dtype=torch.float64)
    weights = project_box_simplex(scores, lower=0.25, upper=0.25)
    assert torch.allclose(weights, torch.full((1, 4), 0.25, dtype=torch.float64))
