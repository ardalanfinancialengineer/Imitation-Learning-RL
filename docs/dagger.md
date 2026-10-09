# Weeks 22–25 DAgger Training

This document records the DAgger loop, why closed-loop evaluation had to be
added before the stage could be measured at all, the acceptance checks, and the
results.

The work package asks to "extend the imitation-learning stage using DAgger by
collecting states visited by the learned policy, requesting expert actions for
these states, and retraining the model on the expanded dataset", and to
"investigate whether collecting examples from the learner's own trajectories
improves its ability to follow the expert over time". It is framed as an
investigation, so the acceptance checks gate the **mechanics** of the loop and
the improvement is reported with a paired bootstrap interval rather than
asserted.

## Why the problem makes DAgger exact here

At decision `t` the market blocks of Equation (6) — the return window,
`mu_hat_{t+1|t}`, `vech(Sigma_hat_{t+1|t})`, and `p_hat_t` — depend only on the
simulated path. They are identical whoever is trading. **Only `w_{t-1}`
differs** under the learner's own trajectory. Two consequences follow, and both
are unusually convenient:

- a learner-visited state is the stored state with one block substituted, so no
  re-simulation is needed; and
- the expert query is the Weeks 9–13 quadratic programme re-solved at the same
  date with the learner's anchor — the *same* programme the demonstrations came
  from, not a re-derived approximation of it.

It also means the covariate shift DAgger addresses is localised to exactly one
block, and can therefore be measured directly rather than argued for. That
measurement is `mean_anchor_drift`, the mean `||w_{t-1}^policy −
w_{t-1}^expert||_1` along a rollout.

The exposé anticipates why this matters: "the current allocation influences
future transaction costs and therefore future decisions". Here that is exact.
The anchor is both a block of the state *and* the point the expert's cost term
is anchored on, so a learner that drifts changes both what it sees and what the
right answer is.

## Closed-loop evaluation had to come first

Behavioural cloning is evaluated **teacher-forced**: every state carries the
expert's own `w_{t-1}`, so the policy is scored one decision at a time on the
expert's trajectory. That metric is structurally blind to the failure DAgger
exists to fix, because it never lets the learner's mistakes accumulate.

The gap is not subtle. On the shipped configuration the behaviour-cloned policy
scores a teacher-forced tracking error around `2e-3` and a **closed-loop** error
around `7e-2` — a factor of roughly 35. Reporting only the first would have made
DAgger look useless while it was in fact fixing the second.

Three quantities are now distinguished, and confusing them is easy:

| Metric | Anchor | Compared against |
|---|---|---|
| teacher-forced tracking error | expert's `w_{t-1}` | stored `w*_t` |
| trajectory tracking error | policy's own `w_{t-1}` | stored `w*_t` |
| **closed-loop tracking error** | policy's own `w_{t-1}` | **expert's answer at the policy's state** |

Only the third is the right target. The second compares two allocations that
are answers to *different* states and is reported only as a diagnostic.

## The loop

```text
D <- D_E                                  (behavioural-cloning demonstrations)
for i = 1 .. N:
    roll out beta_i * expert + (1 - beta_i) * policy    -> visited anchors
    query the expert at a sample of those anchors       -> (s, w*) pairs
    D <- D + pairs
    retrain the policy on D
```

Round 0 is the behaviour-cloned policy itself. Recording it is what makes the
improvement measurable: without it the first aggregated round has nothing to be
compared against.

### Decisions that carry weight

**Aggregated states are stored as anchors, not as states.** A round records, per
path, the queried decision indices, the learner's anchor at each, and the
expert's answer — three small arrays. States are reassembled on demand by
substituting the anchor block, exactly as Weeks 14–16 established for the base
dataset. Materialising them instead would defeat that design at the first round.

**The standardiser is frozen after round 0.** Input statistics stay those of the
round-0 training split. Refitting as the aggregate grows would move the input
scale under a warm-started network and make rounds incomparable. This is still
training-only data, so nothing leaks.

**The learning rate is *not* lowered for the rounds.** The instinct with a
warm start is to reduce it. Measured on the development configuration, keeping
the behavioural-cloning rate of `1e-3` took closed-loop error from `6.76e-2` to
`4.98e-2` over four rounds, whereas dropping it to `3e-4` made every round worse
than the starting point. The frozen standardiser is the likely reason: the
network has to move to fit anchors it was never standardised for.

**Only a fraction of visited states is queried** (`query_fraction`, 0.4 by
default). Consecutive states on a path share all but one day of the return
window and differ in the anchor by one small trade, so they are highly
redundant, and the quadratic programme is the entire cost of a round.

**The round is selected on closed-loop error.** Round-to-round variation is
real — the aggregate changes and the network is retrained each round — so
returning the final round would report a draw from that variation rather than
the method's result. The loop keeps the best round's weights.

**The beta branch replays the stored expert action.** With probability `beta_i`
a rollout step defers to the expert, which keeps early rounds near the
demonstrated distribution. That branch replays `w*_t` as stored, computed at the
expert's own anchor rather than the mixture's current one. Re-solving would cost
one quadratic programme per deferred step for a branch whose weight decays to
nothing. The approximation affects only *which* states get visited; the expert
answers subsequently collected at those states are always exact, taken at the
anchor actually reached.

Both `exponential` (Ross et al.'s `beta_i = beta_decay^i`) and `zero` (pure
DAgger) were measured over six rounds. Both improve substantially;
`exponential` was smoother and ended better, and is the default.

## Acceptance checks

`evaluate_dagger` runs six checks, five of them at **zero tolerance**:

- **expert_query_consistency** — querying the oracle at the expert's own anchor
  reproduces the stored expert action bit-identically. This is the check that
  the DAgger query really is the same programme the demonstrations came from;
- **state_patch_exactness** — assembling a state with the expert's own anchor
  substituted reproduces the stored state bit-identically, so the substitution
  changes only what it is supposed to change;
- **aggregation_growth** — every round adds samples, and the base
  demonstrations survive unchanged at the front of the aggregate;
- **closed_loop_feasibility** — every allocation taken during a rollout, and
  every expert answer collected, satisfies Equation (5);
- **rollout_no_look_ahead** — rewriting the returns after a decision date
  leaves the closed-loop action at that date unchanged. Weeks 14–16 established
  this for the assembled state and Weeks 17–21 for the single-step policy
  output; here it is checked through a *sequential* rollout, where a leak could
  also enter through the anchor chain. It reuses `rebuild_truncated_path` and
  the same control-versus-treatment discipline;
- **rollout_determinism** — repeating a rollout reproduces the trajectory
  exactly.

Deliberately **not** a check: "DAgger improves closed-loop error". The work
package says investigate. A gate on the result would either be vacuous or would
invite tuning until it passed.

## Results

Numbers come from `configs/dagger_demo.yaml`. Reproduce with:

```bash
uv run python scripts/run_dagger_demo.py --config configs/dagger_demo.yaml
```

Outputs land in `data/dagger_demo/`:

- `dagger_dashboard.png` — closed-loop error and covariate shift per round, the
  teacher-forced contrast, bootstrap improvement intervals, aggregate growth,
  and the check summary;
- `dagger_report.json` — full record for every policy and expert configuration;
- `dagger_rounds.csv`, `dagger_summary.csv`.

The shipped configuration is 12 independent paths of 900 periods with 8 assets,
`L = 20`, four DAgger rounds, `query_fraction = 0.3`, and the Weeks 9–13 expert
(`H = 21`, `gamma = 5`, `c = 0.0025`, cap `0.30`). It takes **29.5 minutes** and
issues 27,648 expert queries in total. `--skip-control` roughly halves it.

### Closed-loop tracking error, held-out paths

Improvement is `BC − DAgger`, paired path by path over the held-out paths,
95% percentile bootstrap. Positive means DAgger helped.

| Expert | Policy | BC | DAgger | Change | 95% CI on the improvement | Excludes zero |
|---|---|---:|---:|---:|---|---|
| shipped | transformer | 3.92e-2 | **2.00e-2** | −48.9% | [+1.72e-2, +2.12e-2] | yes |
| shipped | mlp | 1.125e-1 | 5.29e-2 | −53.0% | [+5.87e-2, +6.05e-2] | yes |
| control | transformer | 6.30e-3 | **4.34e-3** | −31.1% | [+1.66e-3, +2.26e-3] | yes |
| control | mlp | 3.99e-2 | 3.72e-2 | −6.7% | [+1.56e-3, +3.81e-3] | yes |

**All four improve, and all four intervals exclude zero.** The covariate shift
falls with the error, which is the mechanism rather than a coincidence: the
shipped transformer's anchor drift goes 0.408 → 0.306 and the shipped MLP's
0.786 → 0.473.

The MLP under the shipped expert is the clearest case. Behaviour-cloned, its
closed-loop error is 1.125e-1 against a never-trade baseline of 1.097e-1 — it is
**worse than not trading at all** once its own errors compound. Four rounds of
DAgger take it to 0.48 of that baseline.

### Teacher-forced error is blind to this

The same policies, scored the way behavioural cloning scores them:

| Expert | Policy | Closed-loop after DAgger | Teacher-forced after DAgger | Ratio |
|---|---|---:|---:|---:|
| shipped | transformer | 2.00e-2 | 6.15e-3 | 3.3x |
| shipped | mlp | 5.29e-2 | 6.70e-3 | 7.9x |
| control | transformer | 4.34e-3 | 3.61e-3 | 1.2x |
| control | mlp | 3.72e-2 | 1.04e-2 | 3.6x |

Teacher-forced error understates the closed-loop error by between 1.2x and 7.9x,
and the understatement is *largest for the worst policy*. A milestone reported
on teacher-forced numbers alone would have concluded DAgger does nothing.

### Two findings worth carrying forward

**DAgger helps more where the state is insufficient, but does not close the
gap.** The improvement is larger under the shipped expert (−49%, −53%) than
under the control (−31%, −6.7%), which looks backwards until the mechanism is
taken seriously: DAgger repairs distribution shift, and there is more shift to
repair when the policy is worse. Under the sufficient state the transformer
already tracks the expert closely (drift 0.106) and has little to gain.

The gap itself survives exactly as predicted. After DAgger, the transformer sits
at 4.34e-3 under the sufficient state and 2.00e-2 under the shipped one — still
**4.6x worse**. DAgger addressed the shift and left the missing horizon
forecasts untouched, which is what the Weeks 17–21 note said it would do.

**Closed-loop evaluation separates the architectures where teacher-forced could
not.** Weeks 17–21 concluded that "the transformer is not distinguishable from
the MLP floor on this evidence", its paired interval only just including zero.
Closed-loop, the transformer is better by a factor of 2.6 under the shipped
expert (2.00e-2 against 5.29e-2) and 8.6 under the control (4.34e-3 against
3.72e-2), before and after DAgger alike.

State this carefully: those are point estimates from one seed, and the
transformer-versus-MLP difference was **not** given its own paired bootstrap
here — the intervals above are for BC-versus-DAgger within a policy. The
magnitude and the consistency across both experts make it worth testing
properly, not worth asserting yet. Doing so is cheap: `per_path_closed_loop_errors`
already returns the per-path units and `paired_bootstrap` consumes them.

If it holds, the architecture comparison of research question 3 should be made
closed-loop rather than teacher-forced. A deterministic transformer's advantage
over a flat encoder is largely that it recovers better from states it was not
trained on, and a teacher-forced metric cannot see that by construction.

## The information floor, and what DAgger can and cannot fix

Weeks 17–21 established that Equation (6) does not contain everything the expert
used: the expert plans over `H = 21` days with horizon-dependent forecasts and
the state carries only the `tau = 0` one, a gap measured at 67% of the level for
`mu_hat` under the shipped `regime_mixture` forecaster.

**DAgger cannot close that gap.** It addresses the mismatch between the
distribution of states the expert visited and the distribution the learner
visits; it does not add state variables. An information gap survives it
untouched.

This is why the demo runs a second, `control` expert whose moments come from a
rolling window. Rolling moments are horizon-*constant*, so under that expert the
`tau = 0` forecast in the state is the whole of what the expert used and the
state is sufficient. Everything else — horizon, constraints, cost, risk
aversion, paths, split, policy, training — is identical. Running both is what
separates "DAgger under-delivers" from "DAgger is pinned against the same floor
that limited behavioural cloning".

The caveat from Weeks 17–21 carries over: the control also makes the
regime-probability block uninformative, because a rolling forecaster fits no
HMM. That removes information, so it can only make the control task harder, and
any measured gain is a conservative estimate.

## Scope boundary

This milestone implements Stage 2 of the imitation pipeline. Wasserstein-
regularised GAIL (Weeks 26–29) and PPO fine-tuning (Weeks 30–33) are not
implemented here.

Two pieces of this milestone are directly reusable by them.
`src/imitation/rollout.py` provides the closed-loop machinery GAIL needs to
compare occupancy distributions — its rollouts already record the state
trajectory, the anchors, and the realised returns. `ExpertOracle` provides
on-demand expert actions at arbitrary states, which is what a GAIL discriminator
and any DAgger-style correction inside PPO both need.

The transformer-diffusion policy is supported by the code but is not run in the
demo: at roughly 6.4 ms per decision it costs about five times as much per
rollout step, and Weeks 17–21 measured it as clearly weaker on both accuracy and
latency.
