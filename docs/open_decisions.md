# Open Decisions Before Freezing the Final Configurations

Section 5.1 of the exposé commits to a specific discipline:

> The final experiment consists of 500 independent Monte Carlo runs. Development
> runs with fewer paths are used for debugging and hyperparameter screening,
> **after which final evaluation configurations are frozen and recorded** before
> the large-scale comparison.

Once frozen, changing a configuration means tuning against the final results,
which invalidates them. Four decisions are still open. Each is cheap to settle
now and expensive — in one case impossible — to settle afterwards.

Only the fourth is a defect; the others came out of measurements the acceptance
checks and demos produced. All four are recorded in the milestone documents they
came from. This file collects them in one place so they can be decided together,
because they interact.

## Summary

| # | Decision | Whose call | Blocks | If deferred |
|---|---|---|---|---|
| 1 | Does the expert need a genuine edge? | Supervisor | Weeks 34–39 | Every measured number has to be regenerated |
| 2 | How to run `N = 200` at 500 paths | Yours | Weeks 38–39 | The largest universe cannot be run at all |
| 3 | Should the state carry the horizon forecasts? | Supervisor | Weeks 34–39 | Imitation results keep a ceiling unrelated to the methods compared |
| 4 | Keep, fix, or report the transformer-diffusion policy? | Supervisor | Weeks 38–39 | A named Section 5.2 strategy fails its own acceptance check |

Decisions 1 and 3 interact: changing either invalidates the imitation numbers,
so they should be taken in one conversation rather than sequentially. Decision 4
may resolve itself if 1 is resolved by changing the preset.

---

## 1. The preset makes the expert a weak benchmark

### What was measured

The canonical three-regime preset builds regime means and volatilities with
matched `linspace` ranges. The consequence is that `mu_i / sigma_i^2` is nearly
constant across assets — 3.06 to 3.33 in the calm regime — so the mean-variance
frontier is close to degenerate and equal weight sits almost on it.

Three independent measurements, from three different milestones:

**Weeks 9–13.** An *oracle* expert, given the simulator's true regime moments
rather than estimates, was swept over risk aversion on one 2,500-period path:

| `gamma` | Ann. return | Ann. vol | Sharpe | Mean weights |
|------:|---:|---:|---:|---|
| 0.5 | +8.10% | 29.4% | +0.275 | tilted to high-mean assets |
| 2 | +5.00% | 24.7% | +0.202 | **0.125 each — exactly equal weight** |
| 5 (shipped) | +3.39% | 23.0% | +0.147 | tilted to low-volatility assets |
| 20 | −0.08% | 21.1% | −0.004 | pinned at the cap on the two calmest |

Equal weight on the same path: +4.99%, 24.7%, Sharpe +0.202 — identical to the
oracle at `gamma = 2` to three decimals. With *estimated* moments the shipped
expert reaches Sharpe +0.006 against equal weight's +0.202.

**Weeks 22–29.** DAgger beats GAIL decisively, because the occupancy gap is
concentrated in one state block that an exact expert query addresses directly.
That is a property of the setup rather than of the methods.

**Weeks 30–33.** On the held-out paths, pure PPO — random initialisation, no
expert information anywhere — earns **+3.20%** annualised while the expert
loses **−9.78%**, and it does so trading 24× less. The imitation-initialised
hybrid recovers almost none of that gain because it stays anchored near the
expert.

### Why it matters for the final study

Section 5.2 lists six strategies, four of which exist to be compared against the
rolling Markowitz expert. That comparison presumes the expert is worth imitating.
On this preset it is not: it pays transaction costs for estimated moments that
buy it nothing.

So the final study, run as configured today, would answer "can a learned policy
beat the expert?" with *yes* — for the uninteresting reason that the expert is
weak, not because imitation or fine-tuning worked. And the hybrid arm
(strategy 5) is structurally handicapped rather than fairly tested.

### Options

**A — Reframe and keep everything.** State plainly that the expert is a
*computational* benchmark, and centre the study on research question 5
(decision-time cost) and on imitation fidelity rather than on out-performance.
Nothing built has to change; the claims get narrower and more defensible.

**B — Change the preset** so means and volatilities are not rank-matched and the
expert has a real cross-sectional edge. Better science and a fairer test of the
hybrid, but it invalidates every performance number measured so far and requires
regenerating all demonstrations. This is exactly why it cannot wait: after the
freeze it is not available at all.

### Evidence that could be gathered first

A preset variant with unmatched means and volatilities, plus one expert demo
run, would establish whether the expert then beats equal weight. Roughly an
hour of compute, and it turns the conversation above from argument into
measurement. Not started — say the word.

---

## 2. `N = 200` at 500 paths is not feasible serially

### What was measured

Per-decision solve times for the rolling-horizon quadratic programme, single
threaded, after the size-dependent solver fix in Weeks 9–13 (which already cut
`N = 200` from 10,573 ms to 2,697 ms):

| `N` | ms/decision | One path (`T = 10,000`) | 500 paths |
|---:|---:|---:|---:|
| 50 | 67 | ~11 min | ~4 days |
| 100 | 336 | ~56 min | ~19 days |
| 200 | 2,697 | ~7.5 h | **~156 days** |

The exposé specifies `N ∈ {50, 100, 200}`, `T = 10,000`, and 500 runs. The last
row does not happen without a decision.

This is not a defect — it is the premise of the project, and research question 5
exists because re-solving the programme at every rebalancing date is expensive.
But it constrains what the final study can contain.

### Options

- **Parallelise across paths.** They are independent Monte Carlo realisations,
  so this is embarrassingly parallel. Sixteen cores brings `N = 200` to roughly
  ten days. **Set `OMP_NUM_THREADS=1` per worker** — otherwise each worker's
  BLAS spawns its own threads, the machine oversubscribes, and the result is
  often slower than running serially.
- **Train at `N = 50`, evaluate `N = 200` on fewer paths.** Keeps the scaling
  claim but with a wider confidence interval at the largest universe.
- **Shorten `T` for the largest universe.** Cheapest, but the paths are then not
  comparable across `N`.

Whichever is chosen belongs in the frozen record, because it changes what the
study can claim about scaling.

Full context: [`docs/markowitz_expert.md`](markowitz_expert.md), "Runtime
implication for the final experiment".

---

## 3. Equation (6) omits the expert's horizon forecasts

### What was measured

The expert solves a rolling-horizon programme over `H = 21` days using forecasts
`mu_hat_{tau+1|t}` and `Sigma_hat_{tau+1|t}` for every `tau`. The state carries
only the `tau = 0` forecast. Under the shipped `regime_mixture` forecaster those
forecasts are genuinely horizon-dependent — measured by
`scripts/check_state_sufficiency.py` at the midpoint decision date:

| Across the horizon | `tau = 0` vs `tau = 20` | Relative to the level |
|---|---:|---:|
| `mu_hat` | 1.68e-3 | **67%** |
| `Sigma_hat` | 4.51e-4 | **43%** |

So the policy is asked to reproduce decisions that depend on roughly twenty
forecast vectors it cannot see. Behavioural cloning has an error floor here that
no amount of training or capacity removes.

A rolling-window forecaster is horizon-*constant* — verified as exactly `0.0`
variation — so running the identical pipeline against each expert isolates the
effect:

| Expert moments | State sufficient? | `mlp` | `transformer` |
|---|---|---:|---:|
| `regime_mixture` (shipped) | no | 0.823 | 0.778 |
| `rolling` (control) | yes | 0.501 | **0.143** |

Tracking error as a fraction of that configuration's own never-trade baseline;
lower is better.

### Why it matters for the final study

**It masks research question 3.** Under the shipped expert the transformer beats
the MLP by 5% — easy to dismiss as noise. Under a sufficient state the same two
architectures separate by a factor of **3.5**. The architecture comparison the
exposé asks for is currently being run against a ceiling that hides it.

**Neither DAgger nor GAIL can lift it.** Both were measured against this: DAgger
addresses distribution shift, and after four rounds the transformer was still
4.6× worse under the shipped expert than under the sufficient-state control.
GAIL matches occupancy over the same states the policy can see. An information
gap survives both untouched.

### Options

1. **Extend the state** with the horizon path of the forecasts, or a compact
   summary — `mu_hat` at `tau ∈ {0, 5, 10, 20}`, or the horizon mean and slope.
   Cheap, and directly targets the gap. Changes Equation (6), so it changes a
   definition the exposé fixes.
2. **Shorten the expert's horizon** to `H = 1` so the state matches. Changes the
   expert rather than the policy, and gives up the multi-period behaviour that
   motivates the project.
3. **Keep Equation (6) and report the floor.** Defensible, but every imitation
   number in the study inherits a ceiling that has nothing to do with the
   learning algorithms being compared.

Option 1 looks right. Full context:
[`docs/policy_behavioural_cloning.md`](policy_behavioural_cloning.md), "State
sufficiency: the binding constraint on imitation".

---

## 4. The transformer-diffusion policy does not clear its baseline

### What was measured

Re-running the Weeks 17-21 demo at full size, the diffusion policy fails its
`beats_naive_baselines` acceptance check. Across the three configured seeds,
tracking error as a fraction of the never-trade baseline:

| Policy | Mean | sd (% of mean) | Best seed | Worst seed | Latency speedup |
|---|---:|---:|---:|---:|---:|
| `mlp` | 0.783 | 1.2% | 0.772 | 0.789 | 5.27x |
| `transformer` | 0.740 | 6.1% | 0.712 | 0.792 | 3.86x |
| `diffusion` | **1.022** | 11.0% | 0.936 | 1.150 | **0.55x** |

Two separate problems:

- **It is not better than doing nothing.** Averaged over seeds it is slightly
  *worse* than never trading, and its paired bootstrap interval on the reported
  seed excludes zero on the wrong side. Its seed-to-seed spread is nine times
  the MLP's, so which side of the baseline it lands on is close to a coin flip.
- **It is slower than the expert it replaces.** A `0.55x` speedup means one
  decision costs nearly twice a full re-solve of the 21-day quadratic
  programme, which removes the entire practical motivation for a learned policy.

This is not a regression: path generation was verified bit-reproducible on the
machine used, and the path-generation code and expert configuration are
unchanged since the milestone. Full detail in
[`policy_behavioural_cloning.md`](policy_behavioural_cloning.md).

### Why it matters for the final study

The transformer-diffusion policy is **strategy 4 of Section 5.2** and one of the
exposé's headline contributions — Section 4.4 specifies it in detail, and
research question 3 asks whether it "offers advantages relative to a
deterministic baseline policy in allocation quality, turnover, robustness, or
generalisation". On current evidence the answer is no on quality and no on
latency.

Running the final 500-path study with a strategy that fails its own acceptance
check would put a known-broken arm in the headline comparison.

### Options

**A — Report it as a negative result.** Keep the implementation, run it in the
final study, and state plainly that the diffusion policy underperforms both the
deterministic policies and the never-trade rule at this scale. Defensible and
cheap; research question 3 gets a clear, if unflattering, answer.

**B — Tune it.** The Weeks 17-21 document already records that the largest
single effect was switching from ε-prediction to x0-prediction. Other levers are
untried: more sampling steps, a different noise schedule, a larger action scale,
longer training. This is real work with no guarantee, and it must happen *before*
the freeze — tuning after seeing final results is not available.

**C — Drop strategy 4.** Narrow the study to the deterministic policies and
record why. Cleanest, but it removes one of the exposé's stated contributions,
so it is a supervisor decision rather than a coding one.

Note that option B interacts with decision 1: if the preset changes so that the
expert has a genuine edge, the diffusion policy's numbers change too, and tuning
against the current preset would be wasted effort. Settle 1 first.

---

## Suggested sequencing

1. **Decide 2 first** — it is a planning call, needs no supervisor, and does not
   interact with the others.
2. **Take 1, 3 and 4 together** with your supervisor. All three change what the
   imitation numbers mean, and taking them separately risks regenerating
   everything more than once. Decision 1 should lead: if the preset changes, the
   diffusion policy's numbers change with it, so tuning it against the current
   preset (decision 4, option B) would be wasted effort.
3. **Then freeze**, and record the resolutions alongside the configurations, as
   Section 5.1 requires.

If it helps that conversation, the preset experiment in item 1 is the one piece
of evidence that would materially sharpen it, and it is about an hour of
compute.
