"""Single place where policies are built from configuration.

This mirrors :mod:`src.expert.config`, and for the same reason: when every
variant is constructed through one function from one dictionary, an ablation
provably differs from the baseline in exactly the parameters that were changed.
Building the transformer in the demo script and the diffusion model in the test
suite would make that guarantee an assertion rather than a fact.

The three named variants line up with Section 5.2:

===================  ==============================================
``mlp``              architecture floor: flat encoder, projected head
``transformer``      strategy 3, the deterministic imitation policy
``diffusion``        strategy 4, the transformer-diffusion policy
===================  ==============================================
"""

from __future__ import annotations

from typing import Any

from ..dataset.state import StateSpec
from ..expert.constraints import PortfolioConstraints
from .base import AllocationPolicy
from .deterministic import DeterministicPolicy
from .diffusion import DiffusionConfig, TransformerDiffusionPolicy
from .encoder import EncoderConfig

POLICY_KINDS = ("mlp", "transformer", "diffusion")

_DETERMINISTIC_KEYS = {
    "hidden_dimensions",
    "head_dimension",
    "dropout",
    "residual_from_previous",
}
_DIFFUSION_KEYS = {"hidden_dimensions"}


def build_policy(
    config: dict[str, Any],
    *,
    spec: StateSpec,
    n_assets: int,
    n_regimes: int,
    constraints: PortfolioConstraints | None = None,
) -> AllocationPolicy:
    """Build one policy from a configuration mapping.

    Args:
        config: Must contain ``kind``; may contain ``encoder`` (architecture
            settings for the transformer), ``diffusion`` (noise schedule and
            sampler), and per-variant keys such as ``hidden_dimensions``.
        spec: The state specification the policy is trained against.
        n_assets: Number of assets ``N``.
        n_regimes: Number of regimes ``K``.
        constraints: Feasible set the projection targets. Defaults to the
            long-only simplex.
    """
    settings = dict(config or {})
    kind = str(settings.pop("kind", "transformer"))
    if kind not in POLICY_KINDS:
        raise ValueError(f"unknown policy kind {kind!r}; expected one of {POLICY_KINDS}.")

    encoder_config = EncoderConfig(**(settings.pop("encoder", None) or {}))
    diffusion_config = DiffusionConfig(**(settings.pop("diffusion", None) or {}))
    if "hidden_dimensions" in settings:
        settings["hidden_dimensions"] = tuple(int(v) for v in settings["hidden_dimensions"])

    if kind == "diffusion":
        _reject_unknown(settings, _DIFFUSION_KEYS, kind)
        return TransformerDiffusionPolicy(
            spec,
            n_assets=n_assets,
            n_regimes=n_regimes,
            constraints=constraints,
            encoder="transformer",
            encoder_config=encoder_config,
            diffusion=diffusion_config,
            **settings,
        )

    _reject_unknown(settings, _DETERMINISTIC_KEYS, kind)
    return DeterministicPolicy(
        spec,
        n_assets=n_assets,
        n_regimes=n_regimes,
        constraints=constraints,
        encoder="transformer" if kind == "transformer" else "mlp",
        encoder_config=encoder_config,
        **settings,
    )


def _reject_unknown(settings: dict[str, Any], allowed: set[str], kind: str) -> None:
    unknown = set(settings) - allowed
    if unknown:
        raise ValueError(
            f"unknown settings for policy kind {kind!r}: {sorted(unknown)}; "
            f"expected a subset of {sorted(allowed)}."
        )
