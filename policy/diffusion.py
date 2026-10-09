"""Transformer-diffusion allocation policy (Section 4.4, Equation 8).

The policy pairs the transformer state encoder with a conditional denoising
diffusion model over portfolio weights [Ho et al., 2020; Chi et al., 2023].
Training corrupts the expert allocation with Gaussian noise at a random
diffusion step and asks a conditional network to recover it; sampling runs the
reverse chain, projecting onto the feasible set after every step, as
Equation (8) requires:

    w_tilde^{(j-1)} = f_theta( w_tilde^{(j)}, s_t, j ),
    w^{(j-1)}       = Pi_{Delta^{N-1}}( w_tilde^{(j-1)} ).

Three implementation choices are worth recording.

**The chain runs on the adjustment, not on the allocation.** Diffusion assumes a
target of roughly unit scale, and allocations are nothing of the sort: they live
on the simplex, average ``1/N``, and are bounded by the position cap. Running the
chain on raw weights puts the entire simplex inside a fraction of one noise
standard deviation and the denoiser learns almost nothing.

The chain therefore runs on ``x = (w - w_{t-1}) / s``, the expert's *adjustment*
to the portfolio already held, rescaled by a factor ``s`` calibrated from the
training demonstrations. This is the same reasoning that anchors the
deterministic head on ``w_{t-1}``, and it matters more here: under transaction
costs the expert adjusts gently, so the adjustment is roughly an order of
magnitude smaller than the allocation and far better conditioned as a diffusion
target. Anchoring on the fixed ``1/N`` centre instead leaves the model trying to
generate the whole portfolio from noise at every rebalancing date, which it does
markedly worse. ``w_{t-1}`` is read from the state's own ``previous_weights``
block, so this uses no information the policy did not already have under
``F_t``.

ε-prediction is also supported and is measurably worse here, because the
expert's adjustment is a sharply concentrated target and ε-prediction amplifies
the network's error by ``1 / sqrt(alpha_bar)`` at low signal-to-noise — 17.8x the
held-out tracking error, and nine times the expert's turnover, because residual
sampling noise turns into a spurious trade at every rebalancing date. See
``docs/policy_behavioural_cloning.md``.

**Projection inside the loop, not only at the end.** Equation (8) places the
projection at every reverse step, and it earns its place: it keeps the iterate
inside the region the denoiser was trained on. Projecting only the final sample
leaves the intermediate states off-distribution and measurably degrades the
result.

**Two samplers.** Ancestral DDPM sampling is the reference. DDIM is also
provided and is the default, for two reasons that matter to this project: it is
deterministic given the initial noise, so the reproducibility check is a
statement about the policy rather than about a random seed, and it reaches a
comparable sample in far fewer network evaluations, which is the whole point of
the inference-latency comparison in research question 5. A diffusion policy
that needs 50 forward passes per rebalancing date would lose that comparison to
the quadratic programme by construction.

**Deterministic by default, stochastic on request.** With no ``generator``, the
chain starts from one fixed noise vector, drawn from ``sample_seed`` and shared
across the batch, which makes the policy an ordinary deterministic function of
the state. Passing a ``generator`` switches to independent per-row noise and
recovers a genuine sample.

The default is not merely a convenience. Drawing fresh batch-shaped noise would
make an allocation depend on *how many other states were scored alongside it*:
``torch.randn`` does not produce the same leading values for ``(1, N)`` as for
``(B, N)``, because it switches to a vectorised path above a size threshold. A
policy whose answer for one rebalancing date changes with the evaluation batch
size cannot be compared against the expert, replayed through the environment,
or checked for reproducibility. Sharing one fixed vector removes that
dependence entirely.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from ..dataset.state import StateSpec
from ..expert.constraints import PortfolioConstraints
from .base import AllocationLoss, AllocationPolicy
from .encoder import EncoderConfig, MLPStateEncoder, TransformerStateEncoder
from .simplex import project_box_simplex

NOISE_SCHEDULES = ("cosine", "linear")
SAMPLERS = ("ddim", "ddpm")
PREDICTION_TYPES = ("sample", "epsilon")

#: Floor used wherever a schedule coefficient appears under a division.
EPSILON = 1e-8

#: Adjustment scale used before :meth:`TransformerDiffusionPolicy.calibrate`
#: replaces it with the value measured on the training demonstrations.
DEFAULT_ACTION_SCALE = 0.05

#: Floor on the calibrated scale. An expert that never trades would otherwise
#: produce a zero scale and a division by zero.
MINIMUM_ACTION_SCALE = 1e-4


@dataclass(frozen=True)
class DiffusionConfig:
    """Noise schedule, denoiser width, and sampling settings."""

    n_steps: int = 50
    schedule: str = "cosine"
    beta_start: float = 1e-4
    beta_end: float = 0.02
    hidden_dimension: int = 256
    n_blocks: int = 3
    time_dimension: int = 64
    dropout: float = 0.0
    prediction: str = "sample"
    sampler: str = "ddim"
    sampling_steps: int = 10
    n_samples: int = 1
    sample_seed: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.sample_seed, int) or isinstance(self.sample_seed, bool):
            raise ValueError("sample_seed must be an integer.")
        for name in (
            "n_steps",
            "hidden_dimension",
            "n_blocks",
            "time_dimension",
            "sampling_steps",
            "n_samples",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if self.prediction not in PREDICTION_TYPES:
            raise ValueError(
                f"unknown prediction {self.prediction!r}; "
                f"expected one of {PREDICTION_TYPES}."
            )
        if self.schedule not in NOISE_SCHEDULES:
            raise ValueError(
                f"unknown schedule {self.schedule!r}; expected one of {NOISE_SCHEDULES}."
            )
        if self.sampler not in SAMPLERS:
            raise ValueError(f"unknown sampler {self.sampler!r}; expected one of {SAMPLERS}.")
        if self.sampling_steps > self.n_steps:
            raise ValueError("sampling_steps must not exceed n_steps.")
        if not 0.0 < self.beta_start <= self.beta_end < 1.0:
            raise ValueError("betas must satisfy 0 < beta_start <= beta_end < 1.")
        if self.time_dimension % 2 != 0:
            raise ValueError("time_dimension must be even for the sinusoidal embedding.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1).")

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_steps": self.n_steps,
            "schedule": self.schedule,
            "beta_start": self.beta_start,
            "beta_end": self.beta_end,
            "hidden_dimension": self.hidden_dimension,
            "n_blocks": self.n_blocks,
            "time_dimension": self.time_dimension,
            "dropout": self.dropout,
            "prediction": self.prediction,
            "sampler": self.sampler,
            "sampling_steps": self.sampling_steps,
            "n_samples": self.n_samples,
            "sample_seed": self.sample_seed,
        }


def make_betas(config: DiffusionConfig) -> torch.Tensor:
    """Return the variance schedule ``beta_1..beta_J``."""
    if config.schedule == "linear":
        return torch.linspace(config.beta_start, config.beta_end, config.n_steps)
    # Cosine schedule of Nichol and Dhariwal. It keeps signal-to-noise usable in
    # the early steps, which matters when only a handful of steps are taken.
    steps = torch.arange(config.n_steps + 1, dtype=torch.float64) / config.n_steps
    alphas_bar = torch.cos((steps + 0.008) / 1.008 * math.pi / 2) ** 2
    alphas_bar = alphas_bar / alphas_bar[0]
    betas = 1.0 - alphas_bar[1:] / alphas_bar[:-1]
    return betas.clamp(1e-8, 0.999).to(torch.float32)


class SinusoidalTimeEmbedding(nn.Module):
    """Standard transformer positional encoding of the diffusion step index."""

    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.dimension = dimension

    def forward(self, steps: torch.Tensor) -> torch.Tensor:
        half = self.dimension // 2
        frequencies = torch.exp(
            -math.log(10_000.0)
            * torch.arange(half, device=steps.device, dtype=torch.float32)
            / max(half - 1, 1)
        )
        angles = steps.float().unsqueeze(-1) * frequencies
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


class ConditionalDenoiser(nn.Module):
    """Predict the noise added to an allocation, given state and step index."""

    def __init__(
        self,
        *,
        n_assets: int,
        condition_dimension: int,
        config: DiffusionConfig,
    ) -> None:
        super().__init__()
        self.time_embedding = SinusoidalTimeEmbedding(config.time_dimension)
        width = config.hidden_dimension
        self.input_projection = nn.Linear(
            n_assets + condition_dimension + config.time_dimension, width
        )
        self.blocks = nn.ModuleList(
            nn.Sequential(
                nn.LayerNorm(width),
                nn.Linear(width, width),
                nn.GELU(),
                nn.Dropout(config.dropout) if config.dropout > 0.0 else nn.Identity(),
                nn.Linear(width, width),
            )
            for _ in range(config.n_blocks)
        )
        self.output = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, n_assets))

    def forward(
        self, noisy: torch.Tensor, condition: torch.Tensor, steps: torch.Tensor
    ) -> torch.Tensor:
        features = torch.cat(
            [noisy, condition, self.time_embedding(steps)], dim=-1
        )
        hidden = self.input_projection(features)
        for block in self.blocks:
            hidden = hidden + block(hidden)
        return self.output(hidden)


class TransformerDiffusionPolicy(AllocationPolicy):
    """Conditional diffusion over allocations, conditioned on the encoded state."""

    # Declared so type checking knows these are tensors; see StateStandardiser.
    betas: torch.Tensor
    alphas_bar: torch.Tensor
    action_scale: torch.Tensor

    def __init__(
        self,
        spec: StateSpec,
        *,
        n_assets: int,
        n_regimes: int,
        constraints: PortfolioConstraints | None = None,
        encoder: str = "transformer",
        encoder_config: EncoderConfig | None = None,
        hidden_dimensions: tuple[int, ...] = (256, 128),
        diffusion: DiffusionConfig | None = None,
    ) -> None:
        super().__init__(
            state_dimension=spec.dimension(n_assets, n_regimes),
            n_assets=n_assets,
            constraints=constraints if constraints is not None else PortfolioConstraints(),
        )
        self.spec = spec
        self.n_regimes = int(n_regimes)
        self.encoder_kind = encoder
        self.diffusion = diffusion if diffusion is not None else DiffusionConfig()

        self.encoder: TransformerStateEncoder | MLPStateEncoder
        if encoder == "transformer":
            self.encoder = TransformerStateEncoder(
                spec, n_assets=n_assets, n_regimes=n_regimes, config=encoder_config
            )
        else:
            self.encoder = MLPStateEncoder(
                spec,
                n_assets=n_assets,
                n_regimes=n_regimes,
                hidden_dimensions=hidden_dimensions,
            )

        self.denoiser = ConditionalDenoiser(
            n_assets=n_assets,
            condition_dimension=self.encoder.output_dimension,
            config=self.diffusion,
        )

        betas = make_betas(self.diffusion)
        alphas_bar = torch.cumprod(1.0 - betas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_bar", alphas_bar)
        # Overwritten by calibrate() from the training demonstrations; the
        # default keeps an uncalibrated policy usable rather than degenerate.
        self.register_buffer("action_scale", torch.tensor(DEFAULT_ACTION_SCALE))

        slices = spec.block_slices(n_assets, n_regimes)
        if "previous_weights" not in slices:
            raise ValueError(
                "TransformerDiffusionPolicy requires a StateSpec that includes w_{t-1}."
            )
        self._previous_slice = slices["previous_weights"]

    # ------------------------------------------------------------------
    # Action-space transforms
    # ------------------------------------------------------------------
    def anchor(self, states: torch.Tensor) -> torch.Tensor:
        """Return ``w_{t-1}``, the portfolio the adjustment is measured from."""
        return states[:, self._previous_slice]

    def to_latent(self, weights: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
        return (weights - anchor) / self.action_scale

    def to_weights(self, latent: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
        """Map a latent iterate back to a feasible allocation (Equation 8)."""
        lower, upper = self.bounds
        return project_box_simplex(
            latent * self.action_scale + anchor, lower=lower, upper=upper
        )

    def condition(self, states: torch.Tensor) -> torch.Tensor:
        return self.encoder(self.standardiser(states))

    @torch.no_grad()
    def calibrate(self, states: torch.Tensor, actions: torch.Tensor) -> None:
        """Set the adjustment scale from the training demonstrations.

        The scale is the root-mean-square expert adjustment per asset, so the
        diffusion target has unit variance by construction. It is estimated on
        training data only, like the input standardiser, and stored as a buffer
        so it travels with the checkpoint.
        """
        deviation = actions - self.anchor(states)
        scale = float(torch.sqrt(torch.mean(deviation**2)))
        self.action_scale.fill_(max(scale, MINIMUM_ACTION_SCALE))

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def denoising_loss(
        self, states: torch.Tensor, actions: torch.Tensor, *, generator=None
    ) -> torch.Tensor:
        """Return the denoising objective for one batch.

        Under ``prediction="sample"`` the network regresses the clean
        adjustment; under ``"epsilon"`` it regresses the noise. Both are
        standard, and the difference matters here — see the module docstring.
        """
        batch = states.shape[0]
        latent = self.to_latent(actions, self.anchor(states))
        steps = torch.randint(
            0,
            self.diffusion.n_steps,
            (batch,),
            device=states.device,
            generator=generator,
        )
        noise = torch.randn(
            latent.shape, device=latent.device, dtype=latent.dtype, generator=generator
        )
        alpha_bar = self.alphas_bar[steps].unsqueeze(-1)
        noisy = alpha_bar.sqrt() * latent + (1.0 - alpha_bar).sqrt() * noise
        predicted = self.denoiser(noisy, self.condition(states), steps)
        target = latent if self.diffusion.prediction == "sample" else noise
        return torch.mean(torch.sum((predicted - target) ** 2, dim=-1))

    def _split_prediction(
        self, output: torch.Tensor, latent: torch.Tensor, alpha_bar: torch.Tensor
    ) -> torch.Tensor:
        """Return the clean-adjustment estimate implied by one network output."""
        if self.diffusion.prediction == "sample":
            return output
        return (latent - (1.0 - alpha_bar).clamp_min(EPSILON).sqrt() * output) / (
            alpha_bar.clamp_min(EPSILON).sqrt()
        )

    def bc_loss(
        self, states: torch.Tensor, actions: torch.Tensor, loss: AllocationLoss
    ) -> torch.Tensor:
        """Behavioural cloning for a diffusion policy is denoising score matching.

        The configured allocation loss is not used here — it cannot be, since a
        training step never produces an allocation. It is still applied to this
        policy at evaluation time, so the reported validation numbers remain
        directly comparable with the deterministic policy's.
        """
        del loss
        return self.denoising_loss(states, actions)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    def allocate(self, states: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        generator = kwargs.pop("generator", None)
        n_samples = int(kwargs.pop("n_samples", self.diffusion.n_samples))
        sampler = str(kwargs.pop("sampler", self.diffusion.sampler))
        if kwargs:
            raise TypeError(f"unexpected arguments: {sorted(kwargs)}")
        if n_samples < 1:
            raise ValueError("n_samples must be a positive integer.")

        condition = self.condition(states)
        anchor = self.anchor(states)
        draws = [
            self._sample(condition, anchor, sampler=sampler, generator=generator)
            for _ in range(n_samples)
        ]
        if n_samples == 1:
            return draws[0]
        # The mean of feasible allocations is feasible (the set is convex), so
        # averaging draws needs no further projection.
        return torch.stack(draws).mean(dim=0)

    def _sample(
        self, condition: torch.Tensor, anchor: torch.Tensor, *, sampler: str, generator
    ) -> torch.Tensor:
        if sampler not in SAMPLERS:
            raise ValueError(f"unknown sampler {sampler!r}; expected one of {SAMPLERS}.")
        shape = (condition.shape[0], self.n_assets)
        latent = self._noise(shape, condition, generator, offset=0)
        schedule = self._step_schedule(sampler)

        for position, step in enumerate(schedule):
            index = torch.full(
                (shape[0],), step, device=condition.device, dtype=torch.long
            )
            alpha_bar = self.alphas_bar[step]
            output = self.denoiser(latent, condition, index)
            start = self._split_prediction(output, latent, alpha_bar)
            # Equation (8): project the provisional allocation at every step.
            start = self.to_latent(self.to_weights(start, anchor), anchor)

            next_step = schedule[position + 1] if position + 1 < len(schedule) else -1
            if next_step < 0:
                latent = start
                break

            # Recover the noise *implied by the projected* adjustment rather than
            # reusing the network's own output. Otherwise the update mixes a
            # feasible x_0 with a direction that points back off the simplex, and
            # the projection is partly undone at every step.
            residual = (latent - alpha_bar.sqrt() * start) / (
                (1.0 - alpha_bar).clamp_min(EPSILON).sqrt()
            )
            alpha_bar_next = self.alphas_bar[next_step]
            if sampler == "ddim":
                latent = (
                    alpha_bar_next.sqrt() * start
                    + (1.0 - alpha_bar_next).sqrt() * residual
                )
            else:
                noise = self._noise(shape, condition, generator, offset=position + 1)
                variance = (
                    (1.0 - alpha_bar_next) / (1.0 - alpha_bar) * (1.0 - alpha_bar / alpha_bar_next)
                )
                latent = (
                    alpha_bar_next.sqrt() * start
                    + (1.0 - alpha_bar_next - variance).clamp(min=0.0).sqrt() * residual
                    + variance.clamp(min=0.0).sqrt() * noise
                )

        return self.to_weights(latent, anchor)

    def _noise(
        self,
        shape: tuple[int, int],
        condition: torch.Tensor,
        generator,
        *,
        offset: int,
    ) -> torch.Tensor:
        """Return the noise for one reverse step.

        With a ``generator`` this is independent per row: a genuine sample. With
        none it is a single fixed vector broadcast across the batch, so the
        policy is deterministic and its answer for one state does not depend on
        the batch it was scored in. ``offset`` keeps successive DDPM steps from
        reusing the same vector.
        """
        device = condition.device
        dtype = condition.dtype
        if generator is not None:
            return torch.randn(shape, generator=generator, device=device, dtype=dtype)
        fixed = torch.Generator(device=device)
        fixed.manual_seed(self.diffusion.sample_seed * 1_000_003 + offset)
        return torch.randn(
            (1, shape[1]), generator=fixed, device=device, dtype=dtype
        ).expand(shape)

    def _step_schedule(self, sampler: str) -> list[int]:
        """Return the decreasing diffusion indices visited during sampling."""
        total = self.diffusion.n_steps
        if sampler == "ddpm":
            return list(range(total - 1, -1, -1))
        count = min(self.diffusion.sampling_steps, total)
        positions = torch.linspace(total - 1, 0, count).round().long().tolist()
        # Deduplicate while preserving the strictly decreasing order.
        schedule: list[int] = []
        for value in positions:
            if not schedule or value < schedule[-1]:
                schedule.append(int(value))
        return schedule

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "transformer_diffusion",
            "encoder": self.encoder.describe(),
            "diffusion": self.diffusion.to_dict(),
            "action_space": (
                f"chain runs on (w - w_prev) / {float(self.action_scale):.4g}; "
                "projected onto the feasible set at every reverse step"
            ),
            "action_scale": float(self.action_scale),
            **self.base_description(),
        }
