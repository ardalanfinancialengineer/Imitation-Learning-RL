# Weeks 14–16 Dataset and Environment

This document records the state and action definitions, the storage strategy,
the Section 5.1 data separation protocol, the Gymnasium environment, and the
acceptance checks for the dataset milestone.

## State and action

Equation (6) of the exposé defines the state as

```text
s_t = [ r_{t-L+1:t}, mu_hat_{t+1|t}, vech(Sigma_hat_{t+1|t}),
        p_hat_t, w_{t-1}, c, gamma ]
```

and the action as `a_t = w_t` on the long-only simplex. All of it is available
under `F_t`: the return window is strictly past, the moments come from the
Weeks 9–13 causal forecaster, the regime probabilities are filtered, and
`w_{t-1}` is the previously implemented portfolio.

Blocks are assembled in the order above and `StateSpec.block_slices` reports
where each one lands, so a trained policy's attributions can be mapped back to
economically meaningful inputs.

### Dimensions

With `L = 20` and `K = 3`:

| `N` | `vech` | Full state | Compact state |
|----:|-------:|-----------:|--------------:|
| 8   |     36 |        217 |           195 |
| 50  |   1275 |       2380 |          1161 |
| 200 |  20100 |      24505 |          4611 |

The `N = 50` figure of 2,380 matches the number quoted in the exposé, which is
a useful check that the implementation reads Equation (6) the same way.

### Covariance encoding

`vech` is the default and follows Equation (6) exactly. The exposé also
mentions compact covariance summaries as a scalability option, so a `compact`
encoding is implemented and selectable: per-asset volatilities plus six
correlation summary statistics (mean, standard deviation, minimum, and maximum
off-diagonal correlation; the share of total variance in the leading
eigenvalue; and the effective rank). That captures the level and concentration
of co-movement — what the allocation actually responds to — at a cost linear
rather than quadratic in `N`. Having both means the choice can be ablated later
instead of assumed now.

## Storage: why states are not persisted

Materialising every state for the final experiment would cost roughly

```text
500 paths x 10,000 steps x 2,380 floats x 4 bytes = 47.6 GB
```

with the `vech` block alone accounting for about 26 GB. Instead only the raw
ingredients are stored per path — returns, latent regimes, expert weights and
anchors, realised performance, and the fitted HMM parameters with filtered
probabilities — at roughly 9 MB per path, and states are assembled on access.

This is exact, not approximate. Baum–Welch is deterministic here, so a restored
filter path is precisely what a refit would produce;
`RegimeMixtureForecaster.restore` reattaches it instead of paying for the HMM
again on every load. The demo reports the realised reduction factor.

Decisions whose full lookback is unavailable are **dropped, not zero-padded**.
Padding would inject a constant that a policy could learn to key on.

## Data separation protocol (Section 5.1)

All three evaluation settings are defined now, before any policy exists.
Fixing the protocol up front is what stops it being adjusted once results are
visible.

**In-distribution holdout** — separation is at the *path* level. Training,
validation, and test paths are disjoint sets of independent Monte Carlo
realisations, so no two splits ever share a stochastic realisation. Assignment
is deterministic by path index unless a seed is supplied.

**Temporal holdout** — within a path, segments are contiguous and ordered,
never interleaved. Shuffling decision dates across a split would leak
information through both the return window and `w_{t-1}`.

**Distribution shift / zero-shot** — five settings, each perturbing exactly one
aspect so a result can be attributed rather than merely observed:

| Shift | What changes |
|---|---|
| `low_persistence` | self-transition probabilities scaled to 0.75 |
| `high_correlation` | all regime correlations scaled by 1.6 |
| `high_volatility` | all regime volatilities scaled by 1.5 |
| `weak_drift` | regime means compressed towards zero |
| `high_cost` | `c` at the top of the prescribed range |

`shifted_three_regime_preset` renormalises the transition matrix after scaling
persistence, and the resulting covariances are validated as positive
semidefinite.

## Environment

`PortfolioEnv` is a Gymnasium environment over one stored path. It reuses the
demonstration dataset's state assembler, which guarantees the policy sees
exactly the same observation during RL as during behavioural cloning.

The reward is Equation (11):

```text
rho_t = w_t' r_{t+1} - c ||w_t - w_{t-1}||_1
        - (gamma/2) w_t' Sigma_hat_{t+1|t} w_t
        - lambda_E ||w_t - w*_t||_2^2
```

The transaction cost is charged exactly once, inside the realised net return.
Turnover is reported in `info` but never added to the reward again. `lambda_E`
defaults to zero, so the environment is a faithful market-plus-frictions
simulator out of the box; the expert-deviation penalty is switched on for the
PPO stage. Submitted actions are projected onto the feasible set, and
`expert_action()` exposes the expert's decision for the current state, which is
what DAgger will query.

One implementation note worth keeping: the state specification is stored as
`env.state_spec`, **not** `env.spec`. Gymnasium reserves `spec` for its
`EnvSpec`, and shadowing it breaks `gym.make` and the vectorised wrappers
stable-baselines3 uses. A type check caught this; it would have been an
obscure failure at the PPO stage.

## Quantitative acceptance checks

`evaluate_dataset` runs six deterministic checks:

- **state_no_look_ahead** — rewriting every return after a decision date leaves
  the assembled state at that date bit-identical;
- **state_integrity** — dimensions match the analytic formula, states are
  finite, and each block sits where `block_slices` says;
- **action_validity** — every expert action lies on the long-only simplex;
- **split_disjointness** — path sets are disjoint and known, temporal segments
  are contiguous and ordered, and every zero-shot shift really differs from the
  training preset;
- **environment_consistency** — replaying the expert's own actions through the
  environment reproduces the stored net returns and turnover;
- **store_roundtrip** — saving and reloading a path reproduces its states
  exactly.

### How the no-look-ahead check is controlled

The check rebuilds everything downstream of the returns from scratch on the
rewritten array — refitting the HMM, re-solving the expert, reassembling the
state. Reusing the stored fits would make it vacuous, since stored arrays
cannot change when the future does.

The comparison is against a **control** rebuild with the same run length and an
unaltered future, not against the originally stored path. That design came out
of a measurement. Comparing directly against the stored path showed a
difference of `1.07e-7`, and tracing it gave:

| State block | Difference |
|---|---:|
| return window | 0 |
| expected returns | 1.1e-19 |
| covariance | 0 |
| regime probabilities | 0 |
| previous weights | 2.8e-10 |

Only `previous_weights` moved. A control experiment settled it: rebuilding with
the *unaltered* future produced exactly the same discrepancy, and control
versus treatment differed by **exactly 0.0**. So there was no leak. The
residual is that the QP solve is reproducible only to about `1e-14` — BLAS
reductions are not order-deterministic — and the transaction-cost term chains
`w_t` to `w_{t-1}`, amplifying that last-bit noise to roughly `1e-5` after a
few hundred decisions.

Comparing two rebuilds that differ *only* in the future cancels this exactly
and restores a bit-identical (`0.0` tolerance) test of causality. The
alternative — loosening the tolerance to `1e-5` — would have been loose enough
to hide a small genuine leak.

### Anchor-chain sensitivity

The amplification is reported as `anchor_chain_sensitivity`, deliberately as a
number rather than a pass/fail. It is the practical reproducibility limit for
stored expert trajectories: two arithmetically equivalent runs of the same
expert on the same returns can drift apart by this much.

The consequence for later milestones is worth stating plainly. Expert
trajectories are **not** bitwise reproducible across machines, BLAS builds, or
thread counts, because the cost term makes the trajectory a chaotic map. Any
claim of exact reproducibility should be made about *stored* paths, which are
persisted and reloaded exactly, rather than about regenerating them.

## Demonstration

```bash
uv run python scripts/run_dataset_demo.py --config configs/dataset_demo.yaml
```

The default run builds 10 independent paths of 1,200 periods with eight assets.
Outputs land in `data/dataset_demo/` and the command exits non-zero if any
check fails:

- `dataset_dashboard.png` — state composition, path-level split, temporal
  holdout, expert replayed through the environment, action validity, and the
  check summary;
- `dataset_report.json` — state description, storage comparison, dataset sizes,
  the full split protocol, environment description, and pass/fail evidence;
- `dataset_splits.csv`, `state_blocks.csv`, `zero_shot_shifts.csv`;
- `paths/path_XXXX.npz` — the stored paths themselves.

## Scope boundary

This milestone defines the state and action variables, prepares the paths and
splits, builds the environment, and verifies causality. Policy networks,
behavioural cloning, DAgger, GAIL, and PPO belong to Weeks 17 onwards.
`src/imitation/dataset.py` now re-exports the real dataset so the
behavioural-cloning milestone has a natural import site.

Note also that the zero-shot shift *paths* are not generated here — only the
protocol that defines them. Generating them is cheap and belongs with the
robustness milestone (Weeks 34–37), by which point the training configuration
will be frozen.
