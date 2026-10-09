"""Behavioural-cloning objectives.

Section 4.5, Stage 1 of the exposé gives the baseline objective

    L_BC(theta) = (1/M) sum_t || pi_theta(s_t) - w*_t ||_2^2 + lambda_theta Omega(theta),

and adds that "the experiments may consider alternatives that are more closely
related to portfolio performance, such as utility loss or portfolio-weight
divergence". Three allocation losses are therefore provided:

``mse``
    Equation L_BC exactly: the mean over samples of the *squared L2 norm*
    summed over assets. Note this is not ``torch.nn.MSELoss``, which would
    average over assets too and so shrink by a factor of ``N``; keeping the sum
    makes the number directly comparable with the tracking error of
    Equation (18).

``l1``
    Mean absolute allocation error, summed over assets. This is the geometry
    transaction costs are actually charged in — ``c ||w_t - w_{t-1}||_1`` — so
    it weights a large error in one position the same as the cost of fixing it,
    and unlike the squared loss it does not smear a sparse target across all
    ``N`` assets.

``weight_divergence``
    The "portfolio-weight divergence" the exposé names: a Kullback-Leibler
    divergence that reads both allocations as distributions over the asset
    universe. It is scale-free across ``N`` and punishes putting weight where
    the expert put none far harder than the squared loss does.

The regularisation term ``Omega(theta)`` is decoupled weight decay in the
optimiser rather than a term added here, so ``lambda_theta`` does not interact
with the loss scale.
"""

from __future__ import annotations

import torch

#: Floor applied inside the logarithm of the divergence loss. Expert
#: allocations contain exact zeros, so an unguarded log is not merely unstable
#: but undefined.
DIVERGENCE_EPSILON = 1e-8

LOSS_NAMES = ("mse", "l1", "weight_divergence")


def allocation_mse(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean squared L2 allocation error, the exposé's ``L_BC``."""
    _check(predicted, target)
    return torch.mean(torch.sum((predicted - target) ** 2, dim=-1))


def allocation_l1(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean absolute allocation error, summed over assets."""
    _check(predicted, target)
    return torch.mean(torch.sum((predicted - target).abs(), dim=-1))


def weight_divergence(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """``KL(w* || pi(s))`` with both allocations read as distributions."""
    _check(predicted, target)
    safe_target = target.clamp_min(DIVERGENCE_EPSILON)
    safe_predicted = predicted.clamp_min(DIVERGENCE_EPSILON)
    ratio = torch.log(safe_target) - torch.log(safe_predicted)
    # Coordinates where the expert holds nothing contribute nothing to the
    # divergence; masking is what keeps that true given the epsilon floor.
    contribution = torch.where(target > 0.0, target * ratio, torch.zeros_like(target))
    return torch.mean(torch.sum(contribution, dim=-1))


def get_loss(name: str):
    """Return the allocation loss registered under ``name``."""
    losses = {
        "mse": allocation_mse,
        "l1": allocation_l1,
        "weight_divergence": weight_divergence,
    }
    if name not in losses:
        raise ValueError(f"unknown loss {name!r}; expected one of {LOSS_NAMES}.")
    return losses[name]


def _check(predicted: torch.Tensor, target: torch.Tensor) -> None:
    if predicted.shape != target.shape:
        raise ValueError(
            f"predicted and target must have the same shape; got "
            f"{tuple(predicted.shape)} and {tuple(target.shape)}."
        )
    if predicted.ndim != 2:
        raise ValueError("allocations must have shape (batch, n_assets).")
