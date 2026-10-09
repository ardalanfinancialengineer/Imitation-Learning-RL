"""Input standardisation for the Equation (6) state, fitted on training data only.

The state concatenates quantities whose natural scales differ by four orders of
magnitude: daily returns of order ``1e-3``, covariance entries of order
``1e-4``, portfolio weights of order ``1e-1``, and a risk aversion of ``5``.
Feeding that directly into a transformer makes the risk-aversion constant
dominate every inner product, so the encoder is preceded by an affine
standardisation.

Two properties of the fit matter for the research design.

**Training-only statistics.** The mean and scale are estimated on the training
split alone and then reused unchanged for validation, test, and zero-shot
paths. Refitting per split would leak the evaluation distribution into the
policy's inputs. The same discipline already governs
:mod:`src.utils.data`.

**Block-aware pooling.** Return-window statistics are pooled across the
``lookback`` lags and estimated per asset, so the encoding of a return does not
depend on how far back it sits. Estimating each ``(lag, asset)`` cell
separately would break that time-invariance, which is exactly the structure the
transformer's positional encoding is meant to represent.

**Constant blocks.** ``c`` and ``gamma`` are fixed within one experiment, so
their sample variance is zero. Centring them would erase them, which is fine
today but silently wrong once a policy is trained across several cost or
risk-aversion settings. Instead constant coordinates are left uncentred and
divided by their own magnitude, which brings them to order one while keeping
the value itself informative.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn

from ..dataset.state import StateSpec

#: Coordinates whose standard deviation falls below this are treated as constant.
CONSTANT_TOLERANCE = 1e-12


class StateStandardiser(nn.Module):
    """Affine ``(x - mean) / scale`` normalisation of the Equation (6) state.

    The parameters are buffers rather than learnables, so they travel with the
    checkpoint and are restored exactly, but are never updated by the optimiser.
    """

    # Declared so type checking knows these are tensors. `register_buffer`
    # writes into `nn.Module.__setattr__`, whose return type is `Tensor | Module`.
    mean: torch.Tensor
    scale: torch.Tensor
    fitted: torch.Tensor

    def __init__(self, dimension: int) -> None:
        super().__init__()
        if not isinstance(dimension, int) or dimension < 1:
            raise ValueError("dimension must be a positive integer.")
        self.dimension = dimension
        self.register_buffer("mean", torch.zeros(dimension, dtype=torch.float32))
        self.register_buffer("scale", torch.ones(dimension, dtype=torch.float32))
        self.register_buffer("fitted", torch.zeros((), dtype=torch.bool))

    @property
    def is_fitted(self) -> bool:
        return bool(self.fitted.item())

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        if states.shape[-1] != self.dimension:
            raise ValueError(
                f"states have final dimension {states.shape[-1]}; expected {self.dimension}."
            )
        return (states - self.mean) / self.scale

    def inverse(self, standardised: torch.Tensor) -> torch.Tensor:
        """Undo the standardisation, for diagnostics that need raw blocks."""
        return standardised * self.scale + self.mean

    def fit(
        self,
        samples: np.ndarray,
        *,
        spec: StateSpec,
        n_assets: int,
        n_regimes: int,
    ) -> StateStandardiser:
        """Estimate the statistics from a matrix of training states.

        Args:
            samples: Shape ``(n_samples, dimension)``, drawn from the training
                split only.
            spec: The state specification, used for block-aware pooling.
            n_assets: Number of assets ``N``.
            n_regimes: Number of regimes ``K``.
        """
        values = np.asarray(samples, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] < 2:
            raise ValueError("samples must have shape (n_samples, dimension), n_samples >= 2.")
        if values.shape[1] != self.dimension:
            raise ValueError(
                f"samples have dimension {values.shape[1]}; expected {self.dimension}."
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("samples must contain only finite values.")

        mean = values.mean(axis=0)
        deviation = values.std(axis=0)

        slices = spec.block_slices(n_assets, n_regimes)
        window = slices["return_window"]
        # Pool the return window over lags so the encoding is time-invariant.
        pooled = values[:, window].reshape(-1, spec.lookback, n_assets)
        per_asset_mean = pooled.mean(axis=(0, 1))
        per_asset_std = pooled.std(axis=(0, 1))
        mean[window] = np.tile(per_asset_mean, spec.lookback)
        deviation[window] = np.tile(per_asset_std, spec.lookback)

        scale = deviation.copy()
        constant = deviation <= CONSTANT_TOLERANCE
        # Keep constants: rescale by magnitude instead of centring them away.
        mean[constant] = 0.0
        scale[constant] = np.maximum(np.abs(values[0, constant]), 1.0)

        self.mean.copy_(torch.as_tensor(mean, dtype=torch.float32))
        self.scale.copy_(torch.as_tensor(scale, dtype=torch.float32))
        self.fitted.fill_(True)
        return self

    def describe(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary for experiment metadata."""
        return {
            "dimension": self.dimension,
            "fitted": self.is_fitted,
            "mean_absolute_centre": float(self.mean.abs().mean()),
            "median_scale": float(self.scale.median()),
            "constant_coordinates": int((self.mean == 0.0).sum()),
        }
