# Weeks 5–8 Simulation Environment

This document records the implemented data-generating process, transaction-cost
convention, automated acceptance checks, and reproducibility controls for the
simulation milestone.

## Canonical market regimes

`canonical_three_regime_preset(n_assets)` defines three named latent states:

1. **calm** — positive daily means, low volatility, 0.20 constant correlation;
2. **volatile** — mixed daily means, medium volatility, 0.40 correlation;
3. **stressed** — negative daily means, high volatility, 0.55 correlation.

The transition matrix is:

```text
           next calm  next volatile  next stressed
calm            0.92           0.07           0.01
volatile        0.08           0.84           0.08
stressed        0.03           0.17           0.80
```

All covariance matrices are constructed as
`diag(volatility) × correlation × diag(volatility)` and validated as symmetric
positive semidefinite before simulation. Means and volatilities are generated
across the asset dimension, so the same preset supports development runs and
the final `N ∈ {50, 100, 200}` universes.

Conditional on latent regime `z_t = k`, returns are sampled from the configured
multivariate Gaussian `N(mu_k, Sigma_k)`. Regime labels are simulator truth for
diagnostics and evaluation; they are not policy inputs.

## Reproducibility and performance

One user seed creates independent child random streams for the Markov path and
the return draws. Repeating a run with the same parameters and seed therefore
reproduces both arrays exactly.

Returns are sampled in one batch per regime and written back to their original
time positions. This preserves the conditional distribution while avoiding
thousands of repeated covariance factorisations at final scale.

Every run records its preset, dimensions, seed, regime names, transaction-cost
grid, diagnostics, and final-scale targets in `diagnostics.json`.

## Transaction costs

The implemented convention matches the exposé:

```text
cost_t = c × ||w_t - w_(t-1)||_1
net_return_(t+1) = gross_return_(t+1) - cost_t
```

The cost is deducted exactly once. The reusable model permits `c = 0` for
no-cost ablations, while experiment-grid validation enforces the prescribed
range `c ∈ [0.001, 0.005]`. The canonical grid is
`[0.001, 0.0025, 0.005]`.

## Quantitative acceptance checks

`evaluate_simulation` checks one generated path against the target parameters:

- **regime frequencies** — empirical occupancy versus the exact finite-horizon
  expectation implied by the initial distribution and transition matrix;
- **regime persistence and transitions** — every empirical transition
  probability versus its target, including diagonal self-transitions;
- **means** — deviations divided by the target Gaussian standard error;
- **variances** — deviations divided by the Gaussian sample-variance standard
  error;
- **covariances** — Frobenius error divided by its Gaussian/Wishart
  root-mean-square sampling error.

Standardising the moment checks by sampling uncertainty prevents tolerances
from depending arbitrarily on return units or asset count. The checked-in
thresholds are deterministic and configurable. A run passes only if all five
categories pass.

## Demonstration

Run:

```bash
uv run python scripts/run_simulation_demo.py --config configs/simulation_demo.yaml
```

The default run uses six assets, 10,000 periods, and seed 42. It creates a
dashboard plus machine-readable reports under `data/simulation_demo/`. The
process exits with a non-zero status if a diagnostic fails.

The final dimensions can be smoke-tested independently:

```bash
uv run python scripts/check_simulation_scale.py
```

This runs `T=10,000` once for each `N=50`, `N=100`, and `N=200`, then writes
shapes and elapsed wall-clock times to
`data/simulation_demo/scale_check.json`.

## Scope boundary

This milestone generates and validates market returns and defines transaction
costs. Hidden-regime filtering, the H=21 friction-aware Markowitz expert,
state/action assembly, and learned policies belong to subsequent work packages.
