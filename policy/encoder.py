"""Transformer encoder over the Equation (6) state.

Section 4.3 of the exposé notes the difficulty this module exists to solve: the
state vector is large — 2,380 numbers at ``N = 50``, 24,505 at ``N = 200`` —
and most of that bulk is the return window and the covariance. Section 4.4
prescribes the remedy: "use the transformer encoder to process the return
history and combine it with estimated moments, regime probabilities, previous
weights, and economic parameters".

That is taken literally here. The flat state is split back into its Equation (6)
blocks using ``StateSpec.block_slices`` — the same slices the dataset milestone
already publishes — and turned into a token sequence:

* the return window becomes ``L`` tokens, one per lag, each a linear embedding
  of that day's cross-section of ``N`` returns, plus a learned positional code;
* every remaining block (``mu_hat``, the covariance encoding, ``p_hat_t``,
  ``w_{t-1}``, and ``(c, gamma)``) becomes a single context token with its own
  projection and a learned type embedding;
* a learned ``[CLS]`` token is prepended and its output is the state encoding.

Attention over that sequence is what lets the moments and the previous
portfolio modulate how the return history is read, which is the economically
interesting interaction: the same return window implies a different allocation
depending on where the portfolio already sits and how costly trading is.

One scaling caveat is worth stating plainly. The covariance context token is a
single ``Linear`` over the covariance block, so under the default ``vech``
encoding it holds ``N(N+1)/2 * d_model`` parameters — about 5.1M at ``N = 200``.
That is precisely what the ``compact`` covariance encoding built in Weeks 14-16
is for: it keeps the block linear in ``N`` and shrinks that projection by two
orders of magnitude. The choice is a configuration flag, so it can be ablated
rather than assumed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from ..dataset.state import StateSpec

#: Block that is expanded into a token sequence rather than a single token.
SEQUENCE_BLOCK = "return_window"


@dataclass(frozen=True)
class EncoderConfig:
    """Architecture of the transformer state encoder."""

    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    feedforward_multiplier: int = 4
    dropout: float = 0.1

    def __post_init__(self) -> None:
        for name in ("d_model", "n_heads", "n_layers", "feedforward_multiplier"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model={self.d_model} must be divisible by n_heads={self.n_heads}."
            )
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1).")

    def to_dict(self) -> dict[str, Any]:
        return {
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
            "feedforward_multiplier": self.feedforward_multiplier,
            "dropout": self.dropout,
        }


class StateTokeniser(nn.Module):
    """Turn a flat Equation (6) state into a sequence of ``d_model`` tokens."""

    def __init__(
        self,
        spec: StateSpec,
        *,
        n_assets: int,
        n_regimes: int,
        d_model: int,
    ) -> None:
        super().__init__()
        self.spec = spec
        self.n_assets = n_assets
        self.n_regimes = n_regimes
        self.d_model = d_model

        slices = spec.block_slices(n_assets, n_regimes)
        sizes = spec.block_sizes(n_assets, n_regimes)
        self.dimension = int(sum(sizes.values()))
        self._window = slices[SEQUENCE_BLOCK]
        self.context_names = tuple(name for name in sizes if name != SEQUENCE_BLOCK)
        self._context_slices = {name: slices[name] for name in self.context_names}

        self.return_projection = nn.Linear(n_assets, d_model)
        self.context_projections = nn.ModuleDict(
            {name: nn.Linear(sizes[name], d_model) for name in self.context_names}
        )
        # Learned rather than sinusoidal: the lookback is short and fixed, so
        # there is nothing to extrapolate to and a free table is simpler.
        self.lag_embedding = nn.Parameter(torch.zeros(spec.lookback, d_model))
        self.context_embedding = nn.Parameter(
            torch.zeros(len(self.context_names), d_model)
        )
        self.class_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.lag_embedding, std=0.02)
        nn.init.normal_(self.context_embedding, std=0.02)
        nn.init.normal_(self.class_token, std=0.02)

    @property
    def n_tokens(self) -> int:
        """Sequence length seen by the transformer, including ``[CLS]``."""
        return 1 + self.spec.lookback + len(self.context_names)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """Map ``(batch, dimension)`` states to ``(batch, n_tokens, d_model)``."""
        if states.ndim != 2 or states.shape[1] != self.dimension:
            raise ValueError(
                f"states must have shape (batch, {self.dimension}); got {tuple(states.shape)}."
            )
        batch = states.shape[0]

        window = states[:, self._window].reshape(batch, self.spec.lookback, self.n_assets)
        tokens = [self.return_projection(window) + self.lag_embedding]

        for index, name in enumerate(self.context_names):
            block = states[:, self._context_slices[name]]
            projected = self.context_projections[name](block)
            tokens.append((projected + self.context_embedding[index]).unsqueeze(1))

        return torch.cat([self.class_token.expand(batch, -1, -1), *tokens], dim=1)


class TransformerStateEncoder(nn.Module):
    """Encode the state into a single ``d_model`` vector via self-attention."""

    def __init__(
        self,
        spec: StateSpec,
        *,
        n_assets: int,
        n_regimes: int,
        config: EncoderConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config if config is not None else EncoderConfig()
        self.tokeniser = StateTokeniser(
            spec, n_assets=n_assets, n_regimes=n_regimes, d_model=self.config.d_model
        )
        layer = nn.TransformerEncoderLayer(
            d_model=self.config.d_model,
            nhead=self.config.n_heads,
            dim_feedforward=self.config.d_model * self.config.feedforward_multiplier,
            dropout=self.config.dropout,
            activation="gelu",
            batch_first=True,
            # Pre-norm. Post-norm needs a warmup schedule to train stably at
            # this depth, and warmup is one more thing to get wrong.
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=self.config.n_layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(self.config.d_model)

    @property
    def output_dimension(self) -> int:
        return self.config.d_model

    @property
    def state_dimension(self) -> int:
        return self.tokeniser.dimension

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """Return the ``[CLS]`` encoding, shape ``(batch, d_model)``."""
        encoded = self.encoder(self.tokeniser(states))
        return self.norm(encoded[:, 0])

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "transformer",
            **self.config.to_dict(),
            "n_tokens": self.tokeniser.n_tokens,
            "context_blocks": list(self.tokeniser.context_names),
        }


class MLPStateEncoder(nn.Module):
    """Flat feed-forward encoder used as the architecture floor.

    Section 5.2 asks whether the transformer-diffusion policy is better than a
    deterministic alternative. That comparison is only informative if there is
    also a floor showing what the state alone buys without any sequence
    structure, so the same training loop can be pointed at this instead.
    """

    def __init__(
        self,
        spec: StateSpec,
        *,
        n_assets: int,
        n_regimes: int,
        hidden_dimensions: tuple[int, ...] = (256, 128),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if not hidden_dimensions:
            raise ValueError("hidden_dimensions must not be empty.")
        self.hidden_dimensions = tuple(int(value) for value in hidden_dimensions)
        self.dropout = float(dropout)
        self._state_dimension = spec.dimension(n_assets, n_regimes)

        layers: list[nn.Module] = []
        width = self._state_dimension
        for hidden in self.hidden_dimensions:
            layers += [nn.Linear(width, hidden), nn.LayerNorm(hidden), nn.GELU()]
            if self.dropout > 0.0:
                layers.append(nn.Dropout(self.dropout))
            width = hidden
        self.body = nn.Sequential(*layers)

    @property
    def output_dimension(self) -> int:
        return self.hidden_dimensions[-1]

    @property
    def state_dimension(self) -> int:
        return self._state_dimension

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        if states.ndim != 2 or states.shape[1] != self._state_dimension:
            raise ValueError(
                f"states must have shape (batch, {self._state_dimension}); "
                f"got {tuple(states.shape)}."
            )
        return self.body(states)

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "mlp",
            "hidden_dimensions": list(self.hidden_dimensions),
            "dropout": self.dropout,
        }
