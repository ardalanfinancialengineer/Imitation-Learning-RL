# Weeks 26–29 Wasserstein-Regularised GAIL

This document records the GAIL implementation, the one place it departs from the
textbook and why, the stability investigation the work package asks for, the
acceptance checks, and the comparison against behavioural cloning and DAgger.

## What GAIL changes

Behavioural cloning matches the expert's *action* at states the expert visited.
DAgger fixes the states but keeps the target the same. GAIL changes the target:
it matches the expert's **occupancy measure**, the joint distribution over
states and actions a policy induces. Section 4.5 of the exposé puts it as
matching "the overall behaviour of the expert rather than only predicting
individual expert actions", and extends it with a Wasserstein critic
(Equation 9) following Arjovsky et al. (2017) and Gulrajani et al. (2017):

```text
W1(d_E, d_pi) = sup_{||D||_Lip <= 1} ( E_{(s,a)~d_E}[D(s,a)]
                                       - E_{(s,a)~d_pi}[D(s,a)] )
```

The loop is the usual two-player alternation: roll out to collect `d_pi`, update
the critic to maximise the dual gap, update the policy to minimise it.

## The one departure, and why it is safe

Standard GAIL uses a policy-gradient method for the policy update, because the
environment is a black box. **Here it is not.** At decision `t` the Equation (6)
state is

```text
s_t = [ r_{t-L+1:t}, mu_hat, vech(Sigma_hat), p_hat_t, w_{t-1}, c, gamma ]
```

and every block except `w_{t-1}` is exogenous market data. The one
action-dependent block *is* the previous action, so the transition is the
identity map from action to next anchor, and the policy — including the
Equation (8) projection, which Weeks 17–21 built with exact gradients — is
differentiable. The closed-loop rollout is therefore a differentiable recurrence
and `dW1/dtheta` follows from backpropagation.

This is a different **estimator** of the same objective, not a different
objective. It matters for three reasons:

- it removes the policy-gradient variance that is the dominant source of GAIL's
  instability, which is what makes a stability study meaningful at 12 paths;
- it avoids pulling the Weeks 30–33 PPO implementation forward;
- it is exact, and exactness is checkable.

The last point is the important one. The claim is verified rather than argued:
the acceptance suite compares the backpropagated gradient against central finite
differences **in float64** and requires agreement to `1e-6`; the measured value
is about `1e-9`. Float32 will not do — there the finite-difference estimate
itself carries roughly `1e-2` relative error, which would comfortably hide a
genuinely wrong gradient.

Backpropagation is truncated: the rollout runs in segments (24 steps by
default), and the anchor is detached at each boundary. The gradient check is run
over a segment, so the truncation cannot be silently wrong.

### An implementation detail worth knowing

States are assembled by splicing a differentiable anchor between two
precomputed halves, `cat(left_i, w_prev, right_i)`, rather than by calling the
NumPy assembler each step. That is what makes the rollout affordable, and a
separate acceptance check confirms the splice reproduces the assembler exactly.

A related trap, found while writing the tests: any objective that depends on the
action only through its **sum** has identically zero gradient, because the
budget constraint pins `sum(w) = 1`. A gradient test written against
`actions.sum()` will report a dead graph on perfectly good code.

## Why the critic's job is unusually well posed here

At a given date the expert and the policy see the *same* market blocks. So
`d_E` and `d_pi` differ only in `w_{t-1}` and in the action taken, and `W1` is
effectively a distance over allocation trajectories conditional on identical
market conditions rather than over the full high-dimensional state. That is far
better conditioned than generic GAIL faces, and it is why interpolating between
an expert pair and a policy pair drawn at the *same* date — as the gradient
penalty does — produces a plausible state-action pair rather than a blend of
unrelated market conditions.

## Training stability

This is the work package's explicit task, and it produced the milestone's
clearest result.

### The critic budget is the setting that matters

With the initial configuration — five critic steps at `1e-4` — the run looked
superficially fine and was quietly meaningless:

| Symptom | Value |
|---|---|
| Reported `W1` estimate | went **negative** (−0.047) |
| Measured `max ||grad D||` | 0.15 → 0.38, against a target of 1 |
| Critic loss | 7.86, of which the gradient penalty contributed 7.9 |

The critic was optimising the penalty almost exclusively and had barely begun to
separate the two distributions, so the `W1` it reported was not a lower bound on
anything. A negative dual estimate is the tell: it means the critic scores
policy samples *above* expert samples, which an optimised critic never does.

Ten critic steps at `3e-3` fix it. The measured Lipschitz constant settles
around 1.1–1.4 and the estimate becomes positive and interpretable. The lesson
generalises: with a gradient penalty, **report the measured Lipschitz constant**,
because it tells you whether the critic is solving its problem at all.

### Gradient penalty versus weight clipping

Both methods the exposé cites, same run, everything else fixed:

| Enforcement | `W1` volatility | Max measured `||grad D||` | Best closed-loop error |
|---|---:|---:|---:|
| gradient penalty | **0.115** | **1.40** | **0.0484** |
| weight clipping | 0.446 | 10.74 | 0.0667 |

Weight clipping is 3.9× more volatile, and — the striking part — its realised
Lipschitz constant is **10.7**, so clipping the weights does not actually
enforce the constraint Equation (9) assumes. That is exactly Gulrajani et al.'s
critique of Arjovsky et al., reproduced here as a measurement rather than a
citation. Gradient penalty is the default; clipping is retained as a switchable
ablation because the comparison *is* the deliverable.

## Acceptance checks

`evaluate_gail` runs seven:

- **rollout_gradient_correctness** — backpropagation against float64 finite
  differences. The load-bearing check;
- **state_splice_exactness** — the spliced state equals the assembler's;
- **critic_lipschitz** — the measured constant is within tolerance of 1;
- **action_feasibility** — every rolled allocation satisfies Equation (5);
- **training_stability** — the run did not diverge and `W1` volatility is
  bounded;
- **rollout_no_look_ahead** — rewriting the future leaves the action unchanged,
  with the Weeks 14–16 control-versus-treatment discipline;
- **rollout_determinism** — repeating a rollout reproduces it exactly.

As in Weeks 22–25, "GAIL beats BC and DAgger" is **not** a check. The work
package says investigate and compare; gating on the outcome would invite tuning
until it passed.

## Model selection, and a tension worth naming

Selection is on validation **closed-loop tracking error**, not on the
Wasserstein estimate. `W1` is GAIL's own objective, but it is measured against a
moving critic and is not comparable across BC, DAgger, and GAIL, whereas
tracking error is defined identically for all three and is what the work package
asks to compare.

The cost is real: a policy that improves occupancy match but not per-decision
tracking would be selected against. The estimate is recorded every iteration so
the tension stays visible rather than buried.

For the same reason, when the demo reports an occupancy distance per policy, it
scores **every** policy with a single critic. Estimates produced by different
critics are not comparable, and putting them in one column would be wrong.

## Demonstration

```bash
uv run python scripts/run_gail_demo.py --config configs/gail_demo.yaml
```

Behaviour-clones a policy, runs DAgger on a copy, runs GAIL from each starting
point, and repeats under both Lipschitz methods. `--skip-ablation` and
`--iterations` give faster paths. Outputs land in `data/gail_demo/`:

- `gail_dashboard.png` — the `W1` trace, closed-loop error during training, the
  three-way comparison, the stability ablation, the measured Lipschitz constant,
  and the check summary;
- `gail_report.json` — full record;
- `gail_iterations.csv`, `gail_comparison.csv`, `gail_stability.csv`.

## Results

12 paths of 900 periods, 8 assets, 40 adversarial iterations, transformer
policy, shipped `regime_mixture` expert. **18 minutes**; every acceptance check
passes and the rollout gradient agrees with finite differences to `6.9e-11`.

### The three-way comparison

Closed-loop tracking error on held-out paths, against a never-trade baseline of
`1.0971e-1`:

| Policy | Closed-loop | vs never trade | Anchor drift | Turnover |
|---|---:|---:|---:|---:|
| behavioural cloning | 3.924e-2 | 0.36 | 0.408 | 0.059 |
| DAgger | **2.003e-2** | **0.18** | **0.306** | 0.079 |
| GAIL from BC | 3.778e-2 | 0.34 | 0.397 | 0.058 |
| GAIL from DAgger | 2.003e-2 | 0.18 | 0.306 | 0.079 |

Paired bootstrap over held-out paths, positive meaning the second policy is
better:

| Comparison | Mean | 95% CI | Excludes zero |
|---|---:|---|---|
| BC − DAgger | +1.921e-2 | [+1.72e-2, +2.12e-2] | yes |
| BC − GAIL from BC | +1.463e-3 | [+3.04e-4, +2.62e-3] | yes |
| DAgger − GAIL from BC | **−1.775e-2** | [−2.09e-2, −1.46e-2] | yes |

Three things follow, and the second and third are negative results.

**GAIL improves on behavioural cloning, but barely.** The 3.7% reduction is
statistically real — the interval excludes zero — and it is roughly an order of
magnitude smaller than DAgger's 49% on the same paths.

**GAIL does not improve on DAgger at all.** Started from the DAgger policy, the
best iteration was **0**: no adversarial iteration beat the starting point on
validation closed-loop error, so the loop returned the DAgger policy unchanged.
That is why the two rows above are identical — it is the selection rule working,
not a bug.

**DAgger decisively beats GAIL as a Stage-2 alternative**, by a factor of 1.9
with a comfortable interval.

### Why GAIL under-delivers here

Offered as hypotheses, not conclusions, since this milestone is not powered to
separate them.

The most likely explanation is an information argument. The occupancy gap in
this problem is concentrated almost entirely in the anchor block — the market
blocks are shared, as noted above. DAgger attacks exactly that gap with **exact
expert labels** at the anchors the learner reaches. GAIL has to recover the same
information through a scalar adversarial signal. When the discrepancy is
low-dimensional and an exact oracle is available and cheap, an adversarial
estimate of it is strictly less informative. GAIL's usual advantage — not
needing an interactive expert — is not an advantage here, because the expert is
a quadratic programme that can be queried at will.

The budget is also modest: 40 iterations with 24-step segments, at 12 paths.
The `W1` trace is still moving at iteration 40.

### Stability

The measurement the work package asks for, both enforcement methods, everything
else fixed:

| Enforcement | `W1` volatility | Max measured `‖∇D‖` | Best closed-loop | Diverged |
|---|---:|---:|---:|---|
| gradient penalty | **0.201** | **1.32** | 2.285e-2 | no |
| weight clipping | 0.368 | **16.88** | 2.446e-2 | no |

Weight clipping is 1.8× more volatile and its realised Lipschitz constant is
**16.9** — clipping does not enforce the constraint Equation (9) assumes, while
the gradient penalty holds the constant at 1.1–1.3 throughout. Neither diverged,
so the finding is about *quality* of the constraint rather than outright
failure.

### A limitation this run exposed, in my own reporting design

The demo scores every policy's occupancy distance with a **single** critic,
reasoning that Wasserstein estimates from different critics are not comparable.
That reasoning is right but insufficient, and the numbers show it:

| Policy | Closed-loop error | `W1` estimate under the shared critic |
|---|---:|---:|
| behavioural cloning | 3.924e-2 | **−0.106** |
| DAgger | 2.003e-2 (best) | **+0.107** (worst) |
| GAIL from BC | 3.778e-2 | −0.030 |

Two things are wrong on their face. The estimates are **negative** for two
policies, and a dual estimate `E_dE[D] − E_dpi[D]` below zero cannot be a lower
bound on a distance — it means the critic scores the policy's pairs *above* the
expert's. And DAgger, clearly the best policy by closed-loop error, is ranked
*worst* by this column.

The cause is that the critic was trained to discriminate the expert from **one
particular** policy — GAIL-from-BC's occupancy — and it is near-optimal only for
that pair. Applied to a policy it never saw, it is an arbitrary 1-Lipschitz
function, not an estimate of anything.

The `occupancy_distance` column in `gail_report.json` should therefore **not**
be read as a comparison between policies. Doing it properly needs a fresh critic
trained to convergence against each policy separately, with the sup in Equation
(9) actually approached in each case; that is a self-contained piece of work and
is the right way to produce the `W1(d_E,t, d_pi,t)` term the regret
decomposition of Equation (13) refers to. Left undone rather than left
misleading.

## Scope boundary

This completes Stage 3 of the imitation pipeline. PPO fine-tuning
(Weeks 30–33) is not implemented here, and the exposé hands it the policy this
stage produces.

Two pieces carry forward. The differentiable rollout gives PPO an exact-gradient
baseline to compare its policy-gradient estimator against — a comparison worth
making, since it is unusual to have both available on the same problem. And the
critic is a learned, state-dependent score on allocations, which is a natural
shaping term for the Equation (11) reward should the expert-deviation penalty
prove too blunt.

One limit inherited unchanged: the Weeks 17–21 information floor. GAIL matches
occupancy, but occupancy is a distribution over the *same* states the policy can
see, so it cannot recover the horizon forecasts Equation (6) omits. Like DAgger,
it addresses a different failure.
