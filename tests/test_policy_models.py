"""Tests for the policy architectures: tokeniser, standardiser, and both heads."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.dataset.state import StateSpec
from src.expert.constraints import PortfolioConstraints
from src.policy.builder import build_policy
from src.policy.deterministic import DeterministicPolicy
from src.policy.diffusion import DiffusionConfig, TransformerDiffusionPolicy, make_betas
from src.policy.encoder import EncoderConfig, StateTokeniser, TransformerStateEncoder
from src.policy.reference import PreviousWeightsReference, default_references
from src.policy.standardise import StateStandardiser

N_ASSETS = 6
N_REGIMES = 3
SPEC = StateSpec(lookback=8)
CONSTRAINTS = PortfolioConstraints(
    long_only=True, min_weight=0.0, max_weight=0.35, leverage_limit=1.0
)
DIMENSION = SPEC.dimension(N_ASSETS, N_REGIMES)

SMALL_ENCODER = {"d_model": 32, "n_heads": 4, "n_layers": 1, "dropout": 0.0}
SMALL_DIFFUSION = {
    "n_steps": 20,
    "hidden_dimension": 64,
    "n_blocks": 2,
    "sampling_steps": 4,
}


def make_states(count: int, seed: int = 0) -> np.ndarray:
    """Return plausible raw states: a valid probability block and valid weights."""
    generator = np.random.default_rng(seed)
    states = generator.normal(scale=0.01, size=(count, DIMENSION))
    slices = SPEC.block_slices(N_ASSETS, N_REGIMES)

    probabilities = generator.random((count, N_REGIMES))
    states[:, slices["regime_probabilities"]] = probabilities / probabilities.sum(
        axis=1, keepdims=True
    )
    weights = generator.random((count, N_ASSETS))
    states[:, slices["previous_weights"]] = np.stack(
        [CONSTRAINTS.project(row / row.sum()) for row in weights]
    )
    states[:, slices["economic_parameters"]] = np.array([0.0025, 5.0])
    return states


def make_actions(count: int, seed: int = 1) -> np.ndarray:
    generator = np.random.default_rng(seed)
    raw = generator.random((count, N_ASSETS))
    return np.stack([CONSTRAINTS.project(row / row.sum()) for row in raw])


def build(kind: str) -> DeterministicPolicy | TransformerDiffusionPolicy:
    config = {"kind": kind, "encoder": SMALL_ENCODER}
    if kind == "diffusion":
        config["diffusion"] = SMALL_DIFFUSION
    elif kind == "mlp":
        config["hidden_dimensions"] = [32, 32]
    return build_policy(
        config,
        spec=SPEC,
        n_assets=N_ASSETS,
        n_regimes=N_REGIMES,
        constraints=CONSTRAINTS,
    )


# ----------------------------------------------------------------------
# Tokenisation
# ----------------------------------------------------------------------
def test_tokeniser_emits_one_token_per_lag_plus_context_and_class():
    tokeniser = StateTokeniser(SPEC, n_assets=N_ASSETS, n_regimes=N_REGIMES, d_model=16)
    # 1 class token + 8 lags + mu, covariance, regimes, previous weights, (c, gamma)
    assert tokeniser.n_tokens == 1 + SPEC.lookback + 5
    tokens = tokeniser(torch.as_tensor(make_states(4), dtype=torch.float32))
    assert tokens.shape == (4, tokeniser.n_tokens, 16)


def test_encoder_reads_the_return_window_as_a_sequence():
    """Permuting two lags must change the encoding, or the order carries nothing."""
    torch.manual_seed(0)
    encoder = TransformerStateEncoder(
        SPEC,
        n_assets=N_ASSETS,
        n_regimes=N_REGIMES,
        config=EncoderConfig(**SMALL_ENCODER),
    ).eval()

    states = torch.as_tensor(make_states(3), dtype=torch.float32)
    window = SPEC.block_slices(N_ASSETS, N_REGIMES)["return_window"]
    permuted = states.clone()
    block = permuted[:, window].reshape(3, SPEC.lookback, N_ASSETS)
    block[:, [0, 1]] = block[:, [1, 0]]
    permuted[:, window] = block.reshape(3, -1)

    with torch.no_grad():
        assert not torch.allclose(encoder(states), encoder(permuted), atol=1e-6)


def test_encoder_rejects_a_state_of_the_wrong_width():
    encoder = TransformerStateEncoder(
        SPEC, n_assets=N_ASSETS, n_regimes=N_REGIMES, config=EncoderConfig(**SMALL_ENCODER)
    )
    with pytest.raises(ValueError, match="shape"):
        encoder(torch.zeros(2, DIMENSION + 1))


# ----------------------------------------------------------------------
# Standardisation
# ----------------------------------------------------------------------
def test_standardiser_centres_and_scales_varying_coordinates():
    states = make_states(500, seed=7)
    standardiser = StateStandardiser(DIMENSION).fit(
        states, spec=SPEC, n_assets=N_ASSETS, n_regimes=N_REGIMES
    )
    output = standardiser(torch.as_tensor(states, dtype=torch.float32)).numpy()

    slices = SPEC.block_slices(N_ASSETS, N_REGIMES)
    varying = output[:, slices["expected_returns"]]
    assert np.abs(varying.mean(axis=0)).max() < 1e-4
    assert np.abs(varying.std(axis=0) - 1.0).max() < 1e-3


def test_standardiser_keeps_constant_coordinates_informative():
    """``c`` and ``gamma`` are constant here; centring them would erase them."""
    states = make_states(200, seed=8)
    standardiser = StateStandardiser(DIMENSION).fit(
        states, spec=SPEC, n_assets=N_ASSETS, n_regimes=N_REGIMES
    )
    economic = SPEC.block_slices(N_ASSETS, N_REGIMES)["economic_parameters"]
    output = standardiser(torch.as_tensor(states[:1], dtype=torch.float32)).numpy()

    assert not np.allclose(output[0, economic], 0.0)
    assert np.abs(output[0, economic]).max() <= 1.0 + 1e-6


def test_standardiser_pools_the_return_window_across_lags():
    """The same return must encode identically wherever it sits in the window."""
    states = make_states(300, seed=9)
    standardiser = StateStandardiser(DIMENSION).fit(
        states, spec=SPEC, n_assets=N_ASSETS, n_regimes=N_REGIMES
    )
    window = SPEC.block_slices(N_ASSETS, N_REGIMES)["return_window"]
    scale = standardiser.scale[window].reshape(SPEC.lookback, N_ASSETS)
    assert torch.allclose(scale[0], scale[-1])


def test_standardiser_inverse_recovers_the_raw_state():
    states = make_states(50, seed=10)
    standardiser = StateStandardiser(DIMENSION).fit(
        states, spec=SPEC, n_assets=N_ASSETS, n_regimes=N_REGIMES
    )
    tensor = torch.as_tensor(states, dtype=torch.float32)
    assert torch.allclose(standardiser.inverse(standardiser(tensor)), tensor, atol=1e-4)


# ----------------------------------------------------------------------
# Policies
# ----------------------------------------------------------------------
@pytest.mark.parametrize("kind", ["mlp", "transformer", "diffusion"])
def test_policies_produce_feasible_allocations(kind):
    torch.manual_seed(0)
    policy = build(kind).eval()
    weights = policy.act(make_states(16))

    assert weights.shape == (16, N_ASSETS)
    assert CONSTRAINTS.is_feasible(weights)


@pytest.mark.parametrize("kind", ["mlp", "transformer", "diffusion"])
def test_act_accepts_a_single_state(kind):
    torch.manual_seed(0)
    policy = build(kind).eval()
    states = make_states(4)
    single = policy.act(states[0])
    assert single.shape == (N_ASSETS,)
    assert np.allclose(single, policy.act(states)[0], atol=1e-6)


@pytest.mark.parametrize("kind", ["mlp", "transformer"])
def test_deterministic_policy_starts_by_holding_the_previous_portfolio(kind):
    """The read-out is zero-initialised, so training begins at "do not trade"."""
    policy = build(kind).eval()
    states = make_states(8)
    previous = states[:, SPEC.block_slices(N_ASSETS, N_REGIMES)["previous_weights"]]
    assert np.abs(policy.act(states) - previous).max() < 1e-6


def test_residual_parameterisation_can_be_switched_off():
    policy = DeterministicPolicy(
        SPEC,
        n_assets=N_ASSETS,
        n_regimes=N_REGIMES,
        constraints=CONSTRAINTS,
        encoder="mlp",
        hidden_dimensions=(16,),
        residual_from_previous=False,
    ).eval()
    states = make_states(4)
    previous = states[:, SPEC.block_slices(N_ASSETS, N_REGIMES)["previous_weights"]]
    # A zero read-out projects to equal weight, not to w_{t-1}.
    assert np.abs(policy.act(states) - previous).max() > 1e-3


@pytest.mark.parametrize("kind", ["mlp", "transformer", "diffusion"])
def test_bc_loss_produces_gradients_for_every_parameter_group(kind):
    torch.manual_seed(0)
    policy = build(kind)
    states = torch.as_tensor(make_states(8), dtype=torch.float32)
    actions = torch.as_tensor(make_actions(8), dtype=torch.float32)

    loss = policy.bc_loss(states, actions, lambda a, b: ((a - b) ** 2).sum(-1).mean())
    loss.backward()

    total = sum(
        float(parameter.grad.abs().sum())
        for parameter in policy.parameters()
        if parameter.grad is not None
    )
    assert np.isfinite(float(loss.detach()))
    assert total > 0.0


# ----------------------------------------------------------------------
# Diffusion specifics
# ----------------------------------------------------------------------
def test_cosine_schedule_is_a_valid_variance_schedule():
    betas = make_betas(DiffusionConfig(n_steps=50, schedule="cosine"))
    alphas_bar = torch.cumprod(1.0 - betas, dim=0)
    assert betas.shape == (50,)
    assert bool((betas > 0).all() and (betas < 1).all())
    assert bool((alphas_bar.diff() < 0).all())


def test_ddim_sampling_is_deterministic_given_a_seeded_generator():
    torch.manual_seed(0)
    policy = build("diffusion").eval()
    states = torch.as_tensor(make_states(6), dtype=torch.float32)

    with torch.no_grad():
        first = policy.allocate(states, generator=torch.Generator().manual_seed(3))
        second = policy.allocate(states, generator=torch.Generator().manual_seed(3))
    assert torch.equal(first, second)


def test_diffusion_policy_is_deterministic_without_a_generator():
    """The default must be a plain function of the state, not a fresh draw."""
    torch.manual_seed(0)
    policy = build("diffusion").eval()
    states = make_states(6)
    assert np.array_equal(policy.act(states), policy.act(states))


def test_diffusion_allocation_does_not_depend_on_the_batch_it_was_scored_in():
    """``torch.randn`` is not batch-invariant; the fixed default noise repairs that."""
    torch.manual_seed(0)
    policy = build("diffusion").eval()
    states = make_states(9)

    batched = policy.act(states)
    for index in (0, 4, 8):
        assert np.allclose(policy.act(states[index]), batched[index], atol=1e-6)


def test_ddpm_sampling_varies_with_the_seed():
    torch.manual_seed(0)
    policy = build("diffusion").eval()
    states = torch.as_tensor(make_states(6), dtype=torch.float32)

    with torch.no_grad():
        first = policy.allocate(
            states, sampler="ddpm", generator=torch.Generator().manual_seed(1)
        )
        second = policy.allocate(
            states, sampler="ddpm", generator=torch.Generator().manual_seed(2)
        )
    assert not torch.allclose(first, second)


def test_latent_and_weight_spaces_round_trip():
    policy = build("diffusion").eval()
    states = torch.as_tensor(make_states(5), dtype=torch.float32)
    weights = torch.as_tensor(make_actions(5), dtype=torch.float32)
    anchor = policy.anchor(states)
    round_tripped = policy.to_weights(policy.to_latent(weights, anchor), anchor)
    assert torch.allclose(round_tripped, weights, atol=1e-5)


def test_calibration_sets_the_scale_to_the_root_mean_square_adjustment():
    """The diffusion target must have unit variance, or the schedule is wrong."""
    policy = build("diffusion")
    states = torch.as_tensor(make_states(200), dtype=torch.float32)
    actions = torch.as_tensor(make_actions(200), dtype=torch.float32)

    policy.calibrate(states, actions)
    expected = float(torch.sqrt(torch.mean((actions - policy.anchor(states)) ** 2)))
    assert float(policy.action_scale) == pytest.approx(expected, rel=1e-6)

    latent = policy.to_latent(actions, policy.anchor(states))
    assert float(torch.sqrt(torch.mean(latent**2))) == pytest.approx(1.0, rel=1e-5)


def test_averaging_diffusion_draws_stays_feasible():
    """The feasible set is convex, so a mean of draws needs no reprojection."""
    torch.manual_seed(0)
    policy = build("diffusion").eval()
    weights = policy.act(
        make_states(8), n_samples=4, sampler="ddpm", generator=torch.Generator().manual_seed(0)
    )
    assert CONSTRAINTS.is_feasible(weights)


def test_ddim_schedule_is_strictly_decreasing_and_ends_at_zero():
    policy = build("diffusion")
    schedule = policy._step_schedule("ddim")
    assert schedule[-1] == 0
    assert all(later < earlier for earlier, later in zip(schedule, schedule[1:]))


# ----------------------------------------------------------------------
# Reference rules and the builder
# ----------------------------------------------------------------------
def test_previous_weights_reference_returns_the_state_block():
    reference = PreviousWeightsReference(
        SPEC, n_assets=N_ASSETS, n_regimes=N_REGIMES, constraints=CONSTRAINTS
    )
    states = make_states(10)
    previous = states[:, SPEC.block_slices(N_ASSETS, N_REGIMES)["previous_weights"]]
    assert np.abs(reference.act(states) - previous).max() < 1e-9


def test_default_references_are_feasible():
    references = default_references(
        SPEC, n_assets=N_ASSETS, n_regimes=N_REGIMES, constraints=CONSTRAINTS
    )
    states = make_states(12)
    for rule in references.values():
        assert CONSTRAINTS.is_feasible(rule.act(states))


def test_builder_rejects_unknown_kinds_and_settings():
    with pytest.raises(ValueError, match="unknown policy kind"):
        build_policy(
            {"kind": "wavelet"}, spec=SPEC, n_assets=N_ASSETS, n_regimes=N_REGIMES
        )
    with pytest.raises(ValueError, match="unknown settings"):
        build_policy(
            {"kind": "transformer", "learning_rate": 1e-3},
            spec=SPEC,
            n_assets=N_ASSETS,
            n_regimes=N_REGIMES,
        )


def test_encoder_config_rejects_indivisible_head_counts():
    with pytest.raises(ValueError, match="divisible"):
        EncoderConfig(d_model=30, n_heads=4)


def test_diffusion_config_rejects_more_sampling_steps_than_it_has():
    with pytest.raises(ValueError, match="sampling_steps"):
        DiffusionConfig(n_steps=10, sampling_steps=20)
