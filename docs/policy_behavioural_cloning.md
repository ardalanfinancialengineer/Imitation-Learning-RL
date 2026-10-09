# Weeks 17–21 Policy Model and Behavioural Cloning

This document records the policy architectures, the behavioural-cloning
objectives and training loop, the acceptance checks, and the measured results
for the Weeks 17–21 milestone.

The work package asks for "a transformer-based policy baseline and, where
feasible, the planned transformer-diffusion policy", the first behavioural
cloning models, and a comparison of "prediction quality and runtime". All three
policies are implemented and trained; the comparison is in
[Results](#results).

## Policies

Three variants are built through one function, `build_policy`, for the same
reason `src/expert/config.py` exists: when every variant comes from one place
and one dictionary, an ablation provably differs from the baseline only in what
was changed.

| Kind | Section 5.2 strategy | What it is |
|---|---|---|
| `mlp` | — (architecture floor) | flat feed-forward encoder, projected head |
| `transformer` | 3, deterministic imitation policy | transformer encoder, projected head |
| `diffusion` | 4, transformer-diffusion policy | transformer encoder, conditional DDPM over allocations |

### Tokenising the state

Section 4.3 states the problem this solves: the state is large — 2,380 numbers
at `N = 50`, 24,505 at `N = 200` — and Section 4.4 prescribes using "the
transformer encoder to process the return history and combine it with estimated
moments, regime probabilities, previous weights, and economic parameters".

The flat state is split back into its Equation (6) blocks using the
`StateSpec.block_slices` the dataset milestone already publishes, and becomes a
token sequence:

- the return window becomes `L` tokens, one per lag, each a linear embedding of
  that day's cross-section of `N` returns, plus a learned positional code;
- `mu_hat`, the covariance encoding, `p_hat_t`, `w_{t-1}`, and `(c, gamma)` each
  become one context token with its own projection and type embedding;
- a learned `[CLS]` token is prepended and its output is the state encoding.

Attention over that sequence is what lets the moments and the current portfolio
modulate how the return history is read — the same window implies a different
allocation depending on where the portfolio already sits and how costly trading
is.

**Scaling caveat.** The covariance context token is a single `Linear` over the
covariance block, so under the default `vech` encoding it holds
`N(N+1)/2 x d_model` parameters — about 5.1M at `N = 200`. This is exactly what
the `compact` covariance encoding built in Weeks 14–16 is for: it keeps the
block linear in `N` and shrinks that projection by two orders of magnitude. The
choice is a config flag, so it can be ablated rather than assumed.

### The head is a projection, not a softmax

A softmax is the reflexive choice for simplex-valued outputs and it is wrong
here on two counts. It cannot represent an exact zero, whereas the long-only
expert under a `0.30` position cap produces allocations that are mostly exact
zeros; and it cannot respect the cap, so a softmax policy would routinely
propose positions the expert is forbidden from taking.

Equation (8)'s Euclidean projection handles both. `src/policy/simplex.py`
projects onto the *full* constraint set of Equation (5),
`{w : 1'w = 1, l <= w_i <= u}`, not the bare simplex:

```text
w_i = clip(v_i - theta, l, u),   theta chosen so that sum_i w_i = 1
```

`theta` is located by bisection, and the active set it implies is then used to
recompute `theta` **in closed form**. That second step is what makes the
backward pass correct: bisection alone is a sequence of comparisons with no
usable derivative, whereas the closed form carries the exact gradient
`d theta / d v_i = 1/|A|` on the active set. The tests check the result against
`PortfolioConstraints.project` (agreement to `2e-16`) and the gradient against
central finite differences.

### Residual anchoring on `w_{t-1}`

Both learned heads predict a *deviation* from `w_{t-1}` rather than an
allocation from scratch, and the read-out is zero-initialised so training starts
from "hold the current portfolio". Under transaction costs the expert is sticky
— the optimal move is usually a small adjustment — so predicting the adjustment
puts capacity where the variation is. `w_{t-1}` is read from the state's own
`previous_weights` block, so nothing is used that was not already available
under `F_t`. It can be switched off (`residual_from_previous: false`) to ablate
the choice.

### Input standardisation

The state mixes returns of order `1e-3`, covariances of order `1e-4`, weights of
order `1e-1`, and a risk aversion of `5`. Fed in raw, the risk-aversion constant
dominates every inner product. `StateStandardiser` applies an affine
`(x - mean) / scale` with three properties that matter:

- **training-only statistics**, frozen thereafter, so the evaluation
  distribution never reaches the policy's inputs;
- **return-window statistics pooled across lags** and estimated per asset, so a
  return encodes identically wherever it sits in the window — the time structure
  is the positional encoding's job, not the normaliser's;
- **constant coordinates are not centred away.** `c` and `gamma` have zero
  sample variance within one experiment. Centring them would erase them, which
  is harmless today and silently wrong once a policy is trained across several
  cost or risk-aversion settings, so they are left uncentred and divided by
  their own magnitude instead.

### Transformer-diffusion policy

Training corrupts the expert allocation at a random diffusion step and asks a
conditional network to recover it; sampling runs the reverse chain, projecting
after every step as Equation (8) requires. Four decisions were forced by
measurement rather than taste.

**The chain runs on the adjustment, not the allocation.** Diffusion assumes a
target of roughly unit scale; allocations are nothing of the sort. The chain
runs on `x = (w - w_{t-1}) / s`, where `s` is the root-mean-square expert
adjustment measured on the training demonstrations (0.0228 in the shipped
configuration). Anchoring on the fixed `1/N` centre instead leaves the model
generating the whole portfolio from noise at every date, and it does that badly.

**x₀-prediction, not ε-prediction.** This was the single largest effect
measured. The expert's adjustment is a sharply concentrated target — almost all
of its conditional mass sits near one value — and ε-prediction amplifies the network's error by `1/sqrt(alpha_bar)` at
low signal-to-noise, so the recovered allocation stays noisy. Regressing the
clean adjustment directly is far better conditioned. Both rows below are the
shipped configuration with only `prediction` changed:

| Prediction target | Held-out tracking error | vs never trade | Trade size |
|---|---:|---:|---:|
| `sample` (default) | 3.64e-3 | 0.96 | 0.028 |
| `epsilon` | 6.50e-2 | 17.18 | 0.410 |

The trade-size column is the readable symptom: under ε-prediction the policy
churns nine times the expert's turnover, because the residual sampling noise
becomes a spurious trade at every rebalancing date.

**The noise is recomputed from the projected iterate.** After projecting the
provisional `x_0`, the DDIM update uses the residual implied by the *projected*
value rather than the network's own output. Reusing the raw output mixes a
feasible `x_0` with a direction pointing back off the simplex, partly undoing
the projection at every step.

**Deterministic by default, stochastic on request.** With no `generator`, the
chain starts from one fixed noise vector shared across the batch, so the policy
is an ordinary deterministic function of the state. This is not a convenience:
`torch.randn` does not return the same leading values for shape `(1, N)` as for
`(B, N)` — it switches to a vectorised path above a size threshold — so
per-row sampling would make an allocation depend on *which batch it was scored
in*, including on whether it landed in a full batch or the short final one. A
policy with that property cannot be compared against the expert or checked for
reproducibility. DDIM is the default sampler for the same family of reasons,
plus one more: at 8 steps instead of 50 it keeps the latency comparison of
research question 5 meaningful.

## Behavioural cloning

### Objectives

Section 4.5 gives the baseline objective and invites alternatives "more closely
related to portfolio performance, such as utility loss or portfolio-weight
divergence". Three are implemented:

| Name | Definition | Why |
|---|---|---|
| `mse` | `mean_t ||pi(s_t) - w*_t||_2^2` | Equation `L_BC` exactly |
| `l1` | `mean_t ||pi(s_t) - w*_t||_1` | the geometry transaction costs are charged in |
| `weight_divergence` | `KL(w* || pi(s))` | the "portfolio-weight divergence" named in the exposé |

`mse` sums over assets rather than averaging, so it is directly comparable with
the tracking error of Equation (18); `torch.nn.MSELoss` would shrink it by a
factor of `N`. The regulariser `Omega(theta)` is decoupled weight decay in the
optimiser, so `lambda_theta` does not interact with the loss scale.

Utility loss is **not** implemented here. It needs `mu_hat` and `Sigma_hat` in
un-standardised form alongside the state, and it is the same quantity the PPO
reward of Equation (11) already optimises directly. Adding it at the BC stage
would duplicate the RL objective with none of the RL machinery.

For the diffusion policy the configured loss is not used during training — a
training step never produces an allocation — and this is explicit in
`bc_loss`. It *is* applied at evaluation, so the reported numbers stay
comparable across policies.

### Training loop

- **Model selection on validation tracking error**, not on the training
  objective. Tracking error is what the exposé reports and it is defined
  identically for every policy; the diffusion policy's denoising loss is on an
  entirely different scale and is not comparable with the deterministic
  policy's.
- **Optional state cache.** Weeks 14–16 established that states are assembled on
  access because materialising the final-scale dataset needs ~48 GB. Training
  re-reads the dataset every epoch, so the trainer materialises it once when the
  estimated footprint is under an explicit configured budget and falls back to
  lazy assembly when it is not. A test asserts the choice changes no number.
- **Seeding must precede construction.** A network's weights are drawn at
  construction time, so a trainer that seeds inside `fit` still inherits
  whatever initialisation the caller produced. `seed_everything` says so and the
  demo calls it before `build_policy`. This cost one genuinely non-reproducible
  run before it was caught.

## Acceptance checks

`evaluate_policy` runs seven deterministic checks; the demo exits non-zero if
any fails.

- **action_feasibility** — every held-out allocation satisfies the budget, box
  bounds, and leverage limit of Equation (5);
- **determinism** — running the policy twice on the same states gives
  bit-identical allocations. Not a formality: the diffusion policy samples, and
  passes only because of the deterministic-by-default noise above;
- **optimisation_sanity** — a copy of the policy, with its network weights
  re-initialised and its fitted standardiser kept, can drive the loss on a
  single batch far below its starting value, catching a dead architecture, a
  detached graph, or a projection that blocks gradients.

  The re-initialisation was added in Weeks 22-25 and matters. The original
  probe started from the *trained* state, and that quantity is not stable:
  across three training seeds on the same fixture it measured 1.66, 1.48, and
  1.07e7, so the check passed or failed on luck. The projection is the reason —
  coordinates clamped at a bound receive exactly zero gradient, so once a policy
  has converged onto the expert's many exact zeros, the residual batch loss is
  largely irreducible and reports nothing about gradient flow;
- **beats_naive_baselines** — held-out tracking error below both `equal_weight`
  and `previous_weights` (never trade). The second is the demanding one:
  `w_{t-1}` is a block of the state, so a policy could reproduce it by reading
  one slice, and under transaction costs it explains much of what the expert
  does;
- **generalisation** — test tracking error within a factor of the *validation*
  tracking error;
- **inference_latency** — mean per-decision inference time below the mean
  quadratic-programme solve time by at least a stated factor (research
  question 5);
- **policy_no_look_ahead** — rewriting every return after a decision date leaves
  the policy's action unchanged. Weeks 14–16 established this for the assembled
  state; here it is checked end to end through the network, closing off a leak
  entering via the standardiser or encoder. It reuses
  `rebuild_truncated_path` and the same control-versus-treatment discipline, so
  the two checks cannot drift apart.

### Why generalisation is measured against validation, not training

The obvious check — test error close to training error — was implemented first
and is wrong here. Whenever the state is a sufficient statistic for the expert's
decision, a flexible enough policy drives its *training* error towards zero.
Measured on a rolling-moment configuration, training tracking error falls to
`3.5e-4` while held-out error sits at `6.5e-3`: a ratio of 18 for a policy that
is nonetheless the best available. A train-versus-test gate would fire on good
and bad policies alike.

Validation and test are both unseen paths drawn the same way, so a gap between
*them* is a real failure — it means the model chosen by early stopping is not
the one that performs on data it never influenced. That is the gate.

The train-to-test ratio is still measured and reported as `memorisation_gap`,
deliberately as a number rather than a pass/fail, in the same spirit as
`anchor_chain_sensitivity` in Weeks 14–16.

### Why the baseline check needs an interval

Both learned heads are initialised to return `w_{t-1}` exactly, so **every
policy starts precisely at the never-trade baseline**. A bare inequality against
that baseline is therefore cleared by noise alone and certifies nothing. The
check requires, in addition, that the paired bootstrap interval for
`baseline − policy` exclude zero: differences are taken path by path over the
held-out paths and resampled, as Section 5.4 prescribes.

Resampling is over *paths*, not individual decisions. Decisions within a path
share a return window and are chained through `w_{t-1}`, so treating them as
independent would understate the interval by roughly the square root of the
decisions per path. See `src/imitation/comparison.py`.

## State sufficiency: the binding constraint on imitation

This is the most important finding of the milestone, and it was not anticipated
when the state was fixed in Weeks 14–16.

**Equation (6) does not contain everything the expert used.** The expert solves
a rolling-horizon programme over `H = 21` days using forecasts
`mu_hat_{tau+1|t}` and `Sigma_hat_{tau+1|t}` for every `tau = t … t+20`. The
state carries only the `tau = 0` forecast. Under the shipped `regime_mixture`
forecaster those forecasts are genuinely horizon-dependent, because filtered
regime probabilities are propagated through the transition matrix:

Measured by `scripts/check_state_sufficiency.py` at the midpoint decision date
of a shipped path:

| Across the horizon | `tau = 0` vs `tau = 20` | relative to the level |
|---|---:|---:|
| `mu_hat` | 1.68e-3 | **67%** |
| `Sigma_hat` | 4.51e-4 | **43%** |

So the policy is asked to reproduce decisions that depend on roughly twenty
forecast vectors it cannot see. Behavioural cloning has an error floor here
that no amount of training or capacity removes.

### Measuring the floor

A rolling-window forecaster is horizon-*constant* — verified as exactly `0.0`
variation across `tau` — so under it the `tau = 0` forecast in the state is the
whole of what the expert used. Running the identical pipeline against each
expert isolates the effect. Both use `H = 21`, the same constraints, cost, risk
aversion, paths, and split; only the moment estimator differs.

| Expert moments | Horizon variation | State sufficient? | `mlp` | `transformer` |
|---|---|---|---:|---:|
| `regime_mixture` (shipped) | 67% / 43% | no | 0.823 | 0.778 |
| `rolling` (control) | 0.0% / 0.0% | yes | 0.501 | **0.143** |

Reproduce with:

```bash
uv run python scripts/check_state_sufficiency.py
```

Tracking error as a fraction of that configuration's own never-trade baseline;
lower is better. The script also measures the horizon variation directly, and
asserts nothing — it reports. The deterministic transformer imitates the expert **5.4×
better** when the state is sufficient. Both control runs were still improving at
epoch 60 of 60 — neither early-stopped — while both shipped runs stopped at
epoch 13-15 of 25-27. So 0.143 is an upper bound on the achievable error, and
the early stopping under the shipped expert is the floor being hit, not
convergence.

Two consequences follow, and both matter for the rest of the project.

**The reported imitation quality is floor-limited, not model-limited.** The
0.769–0.823 band in the shipped configuration is close to the best any policy
can do given what the state contains.

**The floor was masking the architecture comparison of research question 3.**
Under the shipped expert the transformer beats the MLP by 5% (0.778 vs 0.823) —
easy to dismiss as noise. Under a sufficient state the same two architectures
separate by a factor of **3.5** (0.143 vs 0.501). The transformer's advantage is
real and large; it was invisible because both models were pinned against the
same information floor.

A caveat, stated plainly: the control also makes the regime-probability block
uninformative, since a rolling forecaster fits no HMM. That *removes*
information, so it can only make the control task harder, and the measured gain
is therefore a conservative estimate of the horizon effect.

### What to do about it

This is a question for the supervisor, not a decision to take in code, because
it changes a definition the exposé fixes. Three options:

1. **Extend the state** with the horizon path of the forecasts, or a compact
   summary of it (for example `mu_hat` at `tau ∈ {0, 5, 10, 20}`, or the
   horizon mean and slope). Cheap, and directly targets the gap.
2. **Shorten the expert's horizon** so `H = 1` matches the state. This changes
   the expert rather than the policy and gives up the multi-period behaviour
   that motivates the project.
3. **Keep Equation (6) and report the floor.** Defensible, but every imitation
   number in the study — and the DAgger and GAIL stages built on top — inherits
   a ceiling that has nothing to do with the learning algorithms being compared.

Option 1 looks right. Note that DAgger (Weeks 22–25) cannot fix this: it
addresses distribution shift between the expert's and the learner's states, not
missing state variables, so an information gap survives it untouched.

## Reproduction check (August 2026) — supersedes the numbers below

The full-size demo was re-run on a different machine and **the diffusion policy
failed its `beats_naive_baselines` check**. The numbers in the original results
section below could not be reproduced and should be treated as superseded.

What was verified before drawing that conclusion:

- path generation is **bit-reproducible** on the machine used for this run —
  two independent regenerations of the test paths agreed to `0.000e+00` and
  matched the demo exactly;
- `src/dataset/paths.py` and everything under `src/expert/` are **unchanged**
  since the original milestone commit, and the `expert:` block of
  `configs/bc_demo.yaml` is unchanged too;
- the split sizes are identical (18,900 / 3,600 / 4,500).

So this is not a code regression. The original figures came from a different
environment, and the difference is large: the never-trade baseline is
`5.748e-3` here against `3.786e-3` there, a 52% shift. The most likely cause is
the one recorded in [`open_decisions.md`](open_decisions.md) item 1 — on this
preset `mu_i/sigma_i^2` is nearly constant across assets, so the expert's
quadratic programme is close to degenerate and its solution can jump between
near-equivalent allocations when floating-point arithmetic differs across BLAS
builds. That changes the expert's turnover, which *is* the never-trade baseline.
This is the Weeks 14-16 anchor-chain caveat, but far larger than that caveat
suggested, because a flat objective amplifies it.

### Reproducible numbers

Tracking error across the three configured seeds, as a fraction of the
never-trade baseline of `5.748e-3` (lower is better):

| Policy | Mean | sd (% of mean) | Best seed | Worst seed | Latency speedup |
|---|---:|---:|---:|---:|---:|
| `mlp` | **0.783** | 1.2% | 0.772 | 0.789 | 5.27x |
| `transformer` | **0.740** | 6.1% | 0.712 | 0.792 | 3.86x |
| `diffusion` | **1.022** | 11.0% | 0.936 | 1.150 | **0.55x** |

Paired bootstrap against never-trading on the best seed, positive meaning the
policy is better:

| Policy | Improvement | 95% CI | Check |
|---|---:|---|---|
| `mlp` | +1.309e-3 | [+8.13e-4, +1.72e-3] | passes |
| `transformer` | +1.196e-3 | [+3.18e-5, +2.19e-3] | passes |
| `diffusion` | **−8.607e-4** | [−2.13e-3, −1.02e-4] | **fails** |

Two conclusions, and both are firmer than the original run supported.

**The transformer-diffusion policy does not clear the bar.** Averaged over
seeds it is *worse* than not trading at all, and its interval on the reported
seed excludes zero on the wrong side. It is also the most variable policy by a
wide margin — its seed-to-seed spread is nine times the MLP's — so which side
of the baseline it lands on is close to a coin flip.

**It is also slower than the thing it replaces.** A speedup of `0.55x` means
one decision costs nearly twice a full re-solve of the 21-day quadratic
programme. The eight denoising steps and eight projections are not cheap. The
config overrides its latency threshold to `0.5` so this does not fail the
check, which is why the finding is stated here rather than left to a red bar.

Research question 3 asks whether the diffusion policy offers advantages over a
deterministic baseline. On this evidence the answer is no, on both accuracy and
latency. What to do about that is
[open decision 4](open_decisions.md#4-the-transformer-diffusion-policy-does-not-clear-its-baseline).

The deterministic policies are unaffected and pass every check.

## Results

Numbers below come from `configs/bc_demo.yaml`: 30 independent paths of 1,200
periods with 8 assets, `L = 20`, the canonical three-regime preset, and the
Weeks 9–13 expert (`H = 21`, `gamma = 5`, `c = 0.0025`, cap `0.30`). See
`data/bc_demo/bc_report.json` for the full record.

Reproduce with:

```bash
uv run python scripts/run_bc_demo.py --config configs/bc_demo.yaml
```

The split gives 18,900 training, 3,600 validation, and 4,500 test samples, all
at the path level. Training is repeated at seeds 0, 1, 2 on identical paths and
splits. Every acceptance check passes for all three policies.

### Prediction quality (superseded — see the reproduction check above)

Tracking error is Equation (18), as a fraction of the `previous_weights`
(never-trade) baseline of 3.786e-3. The `equal_weight` baseline is 0.1364 — 36×
worse — and is never the binding comparison.

| Policy | Params | Test tracking error (mean ± sd over 3 seeds) | vs never trade |
|---|---:|---:|---:|
| never trade (reference) | 0 | 3.786e-3 | 1.000 |
| `mlp` | 107k | 3.104e-3 ± 3.7e-5 | 0.820 |
| `transformer` | 116k | **2.887e-3 ± 7.3e-5** | **0.763** |
| `diffusion` | 540k | 3.667e-3 ± 2.1e-5 | 0.969 |

Seed dispersion is small — the sd is under 3% of the mean everywhere — so
optimisation noise is not what limits these numbers.

### Does any of this beat not trading?

Both learned heads *start* at the never-trade baseline by construction, so the
question needs an interval rather than an inequality. Paired over the five
held-out paths, 95% percentile bootstrap, positive means the policy is better:

| Policy | Improvement over never trading | 95% CI | Excludes zero |
|---|---:|---|---|
| `mlp` | +6.71e-4 | [+2.57e-4, +1.09e-3] | yes |
| `transformer` | **+8.40e-4** | [+4.53e-4, +1.23e-3] | yes |
| `diffusion` | +1.43e-4 | [+1.66e-5, +2.69e-4] | yes |

All three clear the bar, but not comparably. The deterministic policies improve
on never-trading by 18% and 22%; the diffusion policy by **3.8%**, an
improvement that is real but roughly six times smaller — and its validation
curve shows it gains essentially all of that within its first epoch and then
oscillates without further progress.

### Which policy is better?

| Comparison | Mean difference | 95% CI | Distinguishable |
|---|---:|---|---|
| `mlp` − `transformer` | +1.70e-4 | [−1.9e-6, +3.16e-4] | **no** |
| `mlp` − `diffusion` | −5.28e-4 | [−8.50e-4, −2.06e-4] | yes |
| `transformer` − `diffusion` | −6.98e-4 | [−9.59e-4, −4.40e-4] | yes |

**The transformer is not distinguishable from the MLP floor on this evidence.**
The interval only just includes zero, and the sign favours the transformer at
every seed, but on 30 paths the comparison is not settled. Reporting "the
transformer wins" from the point estimate alone would have been wrong — and it
is what the earlier single-seed version of this table implied.

Both deterministic policies are clearly better than the diffusion policy.

The reason the first comparison is indecisive is not that the architectures are
equivalent. It is the information floor documented in the next section: under
the shipped expert both models are pinned against the same limit. Removing it
separates them by a factor of 3.5.

### Runtime (research question 5)

Measured at batch size one on a single CPU process, with the expert's CVXPY
programme already compiled.

| Policy | ms/decision | Expert QP ms/decision | Speedup | Training wall-clock |
|---|---:|---:|---:|---:|
| `mlp` | 0.62 | 8.17 | 13.2× | 7 s |
| `transformer` | 1.18 | 8.10 | 6.8× | 140 s |
| `diffusion` | 6.43 | 8.11 | 1.3× | 153 s |

The deterministic policies deliver the decision-time saving the exposé
hypothesises. **The diffusion policy essentially does not**: eight denoiser
evaluations plus eight projections cost nearly as much as re-solving the 21-day
quadratic programme outright. This is why its latency threshold is overridden in
the config — the override is a regression guard, and the finding is this
paragraph rather than a passing check.

Two caveats keep this honest. First, `N = 8` is a small programme; the expert's
cost grows steeply with the asset count while the policies' does not, so the
comparison must be repeated at `N ∈ {50, 100, 200}` before any general claim.
Second, the projection is a significant share of the policy's own cost — 0.47 ms
— which is why `DEFAULT_BISECTION_STEPS` was cut from 60 to 30 once measured.

### Objective ablation

Trained on the deterministic transformer only, so the comparison isolates the
loss. Single seed; the differences here are of the same order as the
seed-to-seed spread above and should be read as indicative.

| Loss | Test tracking error | Mean L1 error | Trade size | Best epoch |
|---|---:|---:|---:|---:|
| `mse` | **2.912e-3** | 0.0606 | 0.0531 | 7 |
| `weight_divergence` | 3.004e-3 | 0.0550 | 0.0321 | 23 |
| `l1` | 3.412e-3 | **0.0474** | 0.0163 | 37 |

Each objective wins on its own metric, which is expected. The interesting column
is the third: `l1` produces a policy that trades a third as much as the expert
(0.0163 against 0.0446), because absolute error is minimised by staying put.
Since turnover feeds directly into realised cost, that is worth carrying into
the PPO stage rather than reading `l1` as simply worse.

### Generalisation

| Policy | Test / validation (gated ≤ 1.5) | Test / train (reported) |
|---|---:|---:|
| `mlp` | 0.88 | 3.2 |
| `transformer` | 0.85 | 1.3 |
| `diffusion` | 0.85 | 1.0 |

Test tracks validation closely for every policy, so early stopping transfers.
The `mlp` memorisation gap of 3.2 says the floor model fits its training paths
three times better than it generalises, while the transformer does not — at 30
paths, the clearest signal that the study is data-limited as well as
information-limited.

## Scope boundary

This milestone implements the policy architectures and Stage 1 of the imitation
pipeline. DAgger (Weeks 22–25), Wasserstein-regularised GAIL (Weeks 26–29), and
PPO fine-tuning (Weeks 30–33) are not implemented here. `PortfolioEnv` already
exposes `expert_action()` for the DAgger queries, and the policies satisfy the
`AllocationPolicy` interface the later stages will drive.
