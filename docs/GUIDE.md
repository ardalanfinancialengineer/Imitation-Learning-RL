# Project Guide

This walks the whole pipeline end to end, from simulated market data to a
fine-tuned policy, and defines each domain term where it first matters. It
assumes you can read Python and know what a neural network is; it assumes
nothing about finance.

Read it once top to bottom and the rest of the repository should make sense.
There is a [glossary](#glossary) at the end for looking things up later.

- **See the results:** `data/index.html` (build it with
  `uv run python scripts/run_all_demos.py`)
- **What is still undecided:** [`open_decisions.md`](open_decisions.md)
- **Per-milestone detail:** the documents linked from each stage below

---

## 1. What the project is asking

Every day, an investor has to split their money across `N` assets. The split is
a vector of **weights** `w` that sums to 1 — mathematically a point on the
probability simplex, and computationally just a normalised vector of length `N`.
Tomorrow the assets return `r`, so the portfolio earns `w · r`.

Choosing `w` well is a constrained optimisation problem, and there is a
classical solution: **Markowitz mean-variance optimisation**. It trades expected
return against risk, and with realistic extras (trading is not free, you cannot
short, no single asset may exceed a cap) it becomes a quadratic programme that
has to be re-solved at every rebalancing date.

That is expensive. This project asks:

> Can a neural network learn to reproduce that optimiser's decisions — and then,
> with reinforcement learning, improve on them?

The word for "learning to copy a known-good decision-maker" is **imitation
learning**. The thing being copied is called the **expert**. In machine-learning
terms the expert is a labelling oracle: you can query it for the correct output
at any input, it is just slow.

The whole study runs on **simulated** data, so the true data-generating process
is known and every experiment is repeatable.

## 2. The pipeline

```
 Weeks 5-8      simulate a market            src/simulation/
     |                                       -> returns for N assets over T days
     v
 Weeks 9-13     the expert decides           src/expert/
     |                                       -> optimal weights at each date
     v
 Weeks 14-16    build the training data      src/dataset/
     |                                       -> (state, expert action) pairs
     v
 Weeks 17-21    copy the expert  (BC)        src/policy/, src/imitation/bc.py
     |
     v
 Weeks 22-25    fix compounding error        src/imitation/dagger.py
     |          (DAgger)
     v
 Weeks 26-29    match behaviour  (GAIL)      src/imitation/gail.py
     |
     v
 Weeks 30-33    optimise outcomes (PPO)      src/reinforcement/
```

Each stage has a demo script that produces a dashboard image and a JSON report,
and a document explaining its design decisions. `data/index.html` collects them.

---

## 3. Stage by stage

### Weeks 5-8 — Simulating a market

**Code:** `src/simulation/` · **Doc:** [`simulation_environment.md`](simulation_environment.md)
· **Demo:** `scripts/run_simulation_demo.py`

Real market data would confound the study: you cannot repeat 2008, and you never
know the true parameters. So returns are generated from a known process.

Markets behave differently at different times — calm stretches, volatile
stretches, crises. That is modelled with a **regime-switching** process: a hidden
variable `z` takes one of three values (calm, volatile, stressed), it moves
between them according to a Markov chain, and each regime draws returns from its
own multivariate normal distribution. If you have seen a hidden Markov model,
this is exactly that, used generatively.

Two terms you will see constantly:

- **Covariance matrix** `Σ` — an `N × N` matrix describing how assets move
  together. The diagonal is each asset's variance; the off-diagonal entries are
  co-movement. It is the feature-correlation structure of the return vector.
- **Transaction cost** `c` — trading is not free. Moving from `w_{t-1}` to `w_t`
  costs `c · ||w_t − w_{t-1}||₁`, proportional to how much you shifted. That L1
  distance is called **turnover**.

The demo checks that the generated data actually matches the specification —
regime frequencies, persistence, means, variances, covariances. Five checks, all
must pass.

### Weeks 9-13 — The expert

**Code:** `src/expert/` · **Doc:** [`markowitz_expert.md`](markowitz_expert.md)
· **Demo:** `scripts/run_expert_demo.py`

At each date the expert solves a constrained optimisation over the next `H = 21`
days:

```
maximise    Σ_τ [ wᵀμ̂  −  (γ/2)·wᵀΣ̂w  −  c·||w_τ − w_{τ-1}||₁ ]
subject to  weights sum to 1, each in [0, cap]
```

Reading the three terms: **expected return**, minus **risk** scaled by a
risk-aversion parameter `γ`, minus **trading cost**. It plans 21 days ahead but
only executes the first day's decision, then re-solves tomorrow — this is
receding-horizon control, the same idea as model-predictive control.

Two things the expert must estimate, because it does not get to see the truth:

- **`μ̂`, `Σ̂`** — forecasts of next period's mean and covariance. Estimated from
  past returns only.
- **Regime probabilities `p̂`** — since `z` is hidden, a hidden Markov model is
  fitted to past returns and gives `P(z_t = k | data up to t)`.

**Look-ahead bias** is the cardinal sin here. It is train/test leakage by another
name: if a decision at time `t` uses any information from after `t`, every result
is invalid. The codebase tests this aggressively — see §4.

The expert is the ceiling the imitation stages aim at, *and* the thing they aim
to be faster than. It takes 4.7 ms per decision at `N = 8`, and 2.7 seconds at
`N = 200`.

### Weeks 14-16 — Building the training data

**Code:** `src/dataset/` · **Doc:** [`dataset_environment.md`](dataset_environment.md)
· **Demo:** `scripts/run_dataset_demo.py`

Supervised learning needs `(input, label)` pairs. The label is the expert's
allocation `w*`. The input — the **state** — is everything the policy may look
at, defined by Equation (6) of the exposé:

| Block | What it is | Size |
|---|---|---|
| return window | the last `L` days of returns | `L × N` |
| `μ̂` | forecast mean return | `N` |
| `vech(Σ̂)` | the covariance's lower triangle, flattened | `N(N+1)/2` |
| `p̂` | regime probabilities | `K` |
| `w_{t-1}` | what the portfolio currently holds | `N` |
| `c`, `γ` | cost and risk-aversion settings | 2 |

At `N = 50` that is 2,380 numbers per decision. `vech` is just "take the lower
triangle of a symmetric matrix and flatten it" — the upper triangle is redundant.

**Why `w_{t-1}` is in the state:** because trading costs money, the right move
depends on where you already are. This single block turns out to drive most of
the interesting behaviour in later stages.

Two design decisions worth knowing:

- **States are not stored.** Materialising them for the final study would need
  ~48 GB. Instead the raw ingredients are stored (~9 MB per path) and states are
  rebuilt on demand.
- **Paths are split, not rows.** Train, validation, and test get *different
  simulated market histories*. Splitting rows would put adjacent days of the same
  history on both sides — a leak.

This stage also builds a Gymnasium **environment** (`PortfolioEnv`), the standard
interface reinforcement learning expects: `reset()`, `step(action)`, reward.

### Weeks 17-21 — Behavioural cloning

**Code:** `src/policy/`, `src/imitation/bc.py` ·
**Doc:** [`policy_behavioural_cloning.md`](policy_behavioural_cloning.md) ·
**Demo:** `scripts/run_bc_demo.py`

**Behavioural cloning** is the simplest imitation method: plain supervised
regression from state to expert action. Minimise `||π(s) − w*||²`.

The network is a **transformer** encoder over the state's blocks, which lets the
moment forecasts and the current portfolio modulate how the return history is
read. There is also a plain MLP as a floor, and a **diffusion** policy
(generates the allocation through iterative denoising, like an image model).

The output head is the one place finance constrains the architecture. A softmax
would be the reflexive choice for "vector that sums to 1", and it is wrong here:
it can never output an exact zero, and it cannot respect a per-asset cap. So the
head is a **Euclidean projection** onto the feasible set — find the closest legal
portfolio to whatever the network proposed. It is differentiable, so it trains
end to end.

The key evaluation term is **tracking error**: mean squared distance between the
policy's allocation and the expert's, `E||w − w*||²`. Lower is better.

The baseline to beat is **never trade** — just hold `w_{t-1}`. That sounds
trivial but it is strong, because transaction costs make the expert reluctant to
move, so "do nothing" is often nearly right.

### Weeks 22-25 — DAgger

**Code:** `src/imitation/dagger.py` · **Doc:** [`dagger.md`](dagger.md) ·
**Demo:** `scripts/run_dagger_demo.py`

Behavioural cloning has a known failure. It only ever sees states the *expert*
visited. When the policy runs on its own, small errors move it somewhere the
expert never went, where the policy was never trained, so it errs more — and the
errors compound. This is **covariate shift**: train and deployment distributions
differ *because of* the model's own actions.

**DAgger** (Dataset Aggregation) fixes it by iterating:

1. let the policy run and record the states it actually visits;
2. ask the expert what it would have done at those states;
3. add those pairs to the training set and retrain;
4. repeat.

Measuring this needs a metric behavioural cloning does not have. The BC number
is **teacher-forced**: each state hands the policy the expert's `w_{t-1}`, so
errors never accumulate. The honest metric is **closed-loop**: let the policy
chain its own decisions and compare against what the expert would do *there*.
Measured gap: 2e-3 teacher-forced versus 7e-2 closed-loop — a factor of 35.

Result: closed-loop error fell about 49%.

### Weeks 26-29 — GAIL

**Code:** `src/imitation/gail.py`, `src/imitation/critic.py` ·
**Doc:** [`gail.md`](gail.md) · **Demo:** `scripts/run_gail_demo.py`

**GAIL** (Generative Adversarial Imitation Learning) changes the target. Instead
of matching the expert's action at each state, it matches the expert's
**occupancy measure** — the distribution over `(state, action)` pairs that a
policy induces over a whole trajectory. Think of it as matching behaviour in
aggregate rather than answer by answer.

It works like a GAN. A **critic** scores `(state, action)` pairs, trained to
score expert pairs high and policy pairs low; the policy is trained to fool it.
The distance being minimised is the **Wasserstein distance** (earth-mover's
distance), which requires the critic to be **1-Lipschitz** — its output cannot
change faster than its input. Two ways to enforce that are compared, and the
measurement reproduces the known result: a gradient penalty works, weight
clipping does not.

Result: GAIL improved on behavioural cloning slightly, but did **not** beat
DAgger. The likely reason is in the doc — when an exact expert oracle is cheap
to query, learning from it directly beats inferring it adversarially.

### Weeks 30-33 — PPO fine-tuning

**Code:** `src/reinforcement/` · **Doc:** [`ppo.md`](ppo.md) ·
**Demo:** `scripts/run_ppo_demo.py`

Everything so far tried to *copy* the expert. This stage tries to *beat* it, by
optimising outcomes directly with reinforcement learning.

The **reward** at each step (Equation 11) is:

```
realised net return  −  (γ/2)·estimated risk  −  λ_E·distance from the expert
```

**PPO** (Proximal Policy Optimization) is a standard policy-gradient algorithm.
Its distinguishing feature is that it *clips* how far one update can move the
policy — which matters here because training starts from a carefully trained
imitation policy that is easy to wreck.

PPO needs a **stochastic** policy (it computes a probability ratio between old
and new), but ours is deterministic. The fix: put a Gaussian on the network's
raw output *before* the projection, and treat the projection as part of the
environment. The density stays exact and every executed allocation stays legal.

Two arms are compared, as the exposé requires:

- **hybrid** — PPO starting from the imitation policy;
- **pure PPO** — same everything, random initialisation, no expert information.

The headline metric is **certainty-equivalent utility**: `E[R] − (γ/2)·Var(R)`,
a single number trading average return against its variability.

Result, and it is a genuine surprise: **pure PPO wins by a wide margin.** Not
because PPO imitates well, but because on the current market preset the expert
is not very good — so a policy free to ignore it does better. This is the first
of the [open decisions](open_decisions.md).

---

## 4. Two ideas that recur everywhere

### No look-ahead

Every stage re-tests that decisions at time `t` cannot see data after `t`. The
test is always the same shape: rewrite all returns after `t` with different
values, recompute, and require the result to be **bit-identical**.

There is a subtlety worth knowing, because it bit this project twice. Comparing
a rewritten run against the *stored* original fails at ~1e-7 for a reason that
has nothing to do with leakage: floating-point arithmetic is not perfectly
reproducible, and the transaction-cost term chains each decision to the last, so
tiny differences amplify. The fix is to compare two *rebuilds* that differ only
in the future — then the numerical noise cancels and the test can demand exact
equality.

### Acceptance checks

Each milestone ships a set of deterministic pass/fail checks, and its demo exits
non-zero if any fail. They are not unit tests; they are properties the milestone
claims to have. Checks that gate *mechanics* (is the maths right, is there a
leak) rather than *results* (did the method win) — because gating on results
invites tuning until they pass.

They have earned their keep repeatedly. The Weeks 14-16 check caught a real
look-ahead leak, and the Weeks 30-33 checks caught two problems on first run.

---

## 5. Where things live

```
src/
  simulation/     market data generation                    (Weeks 5-8)
  expert/         the Markowitz optimiser and its inputs    (Weeks 9-13)
  dataset/        state definition, storage, splits         (Weeks 14-16)
  policy/         network architectures and the projection  (Weeks 17-21)
  imitation/      BC, DAgger, GAIL, and shared rollouts     (Weeks 17-29)
  reinforcement/  reward, environment, PPO                  (Weeks 14-16, 30-33)
  evaluation/     Section 5.3 performance measures
  utils/          legacy demonstration format; prefer src/dataset

scripts/          one demo per milestone, plus run_all_demos.py
configs/          one YAML per demo; every experiment setting lives here
docs/             this guide, one document per milestone, open_decisions.md
tests/            406 tests, one file per module
data/             demo outputs (gitignored) and index.html
```

**Naming conventions.** A `*Config` dataclass holds settings and validates them
in `__post_init__`. A `*Trainer` runs a training loop and returns a `*History`.
An `evaluate_*` function returns a `*Diagnostics` object with a `checks` dict.
`describe()` returns JSON-serialisable metadata for the report.

**Configuration.** Nothing is hard-coded in a script. Every experiment setting is
in `configs/*.yaml`, and the demo writes the config it used into its report, so
any dashboard can be traced back to what produced it.

## 6. Running things

```bash
uv sync --extra dev                                   # once

uv run pytest tests/ -q                               # all 406 tests (~12 min)
uv run python scripts/run_all_demos.py                # every demo, ~12 min
uv run python scripts/run_all_demos.py --full         # shipped sizes, ~90 min
uv run python scripts/run_all_demos.py --only ppo     # just one
uv run python scripts/run_all_demos.py --index-only   # rebuild the page
```

Then open `data/index.html`.

Individual demos take `--config`, `--output-dir`, and size overrides; run any of
them with `--help`.

---

## Glossary

**Action** — in this project, a portfolio allocation `w`. See *weights*.

**Advantage** — in reinforcement learning, how much better an action was than
the value function expected. PPO's updates are scaled by it.

**Behavioural cloning (BC)** — imitation by supervised regression from state to
expert action. The simplest method, and the baseline for everything after.

**Certainty-equivalent utility (CE)** — `E[R] − (γ/2)·Var(R)`. A single number
summarising a strategy, trading average return against variability. Higher is
better. Equation (16).

**Closed-loop** — evaluation where the policy's own decisions feed into its next
input, so errors accumulate. Contrast *teacher-forced*.

**Covariance matrix (`Σ`)** — `N × N` matrix of how asset returns move together.
Diagonal is variance, off-diagonal is co-movement.

**Covariate shift** — the deployment input distribution differs from the training
one *because of* the model's own actions. The problem DAgger addresses.

**DAgger** — Dataset Aggregation. Iteratively roll out the policy, ask the expert
what it would do at the visited states, add those pairs, retrain.

**Drawdown** — the largest peak-to-trough fall in cumulative wealth.

**Expert** — the Markowitz optimiser. A labelling oracle: correct but slow.

**GAIL** — Generative Adversarial Imitation Learning. Matches the expert's
*occupancy measure* adversarially rather than its individual actions.

**Lipschitz (1-Lipschitz)** — a function whose output cannot change faster than
its input. Required of the Wasserstein critic; enforced with a gradient penalty.

**Look-ahead bias** — using information from after the decision date. Train/test
leakage, in finance clothing. Invalidates any result it touches.

**Markowitz / mean-variance optimisation** — choosing weights to trade expected
return against variance. The classical solution to portfolio choice.

**Occupancy measure** — the distribution over `(state, action)` pairs a policy
induces over a trajectory. GAIL's target.

**PPO** — Proximal Policy Optimization. A policy-gradient RL algorithm that clips
how far each update may move the policy.

**Projection (onto the simplex)** — given any vector, find the closest one that
is a legal portfolio (sums to 1, within bounds). The policy's output head.

**Regime** — a hidden market state (calm, volatile, stressed) that switches
according to a Markov chain and determines the return distribution.

**Rebalancing** — changing portfolio weights, which costs money.

**Rolling horizon / receding horizon** — plan `H` steps ahead, execute only the
first, re-plan next step. Model-predictive control.

**Risk aversion (`γ`)** — how strongly the objective penalises variance. Higher
`γ` means a more cautious portfolio.

**Sharpe ratio** — annualised mean return divided by annualised volatility. The
standard risk-adjusted performance measure.

**Simplex** — the set of vectors that are non-negative and sum to 1. The space of
long-only portfolios.

**Sortino ratio** — like Sharpe, but divides by *downside* deviation only, so
upside variation is not penalised.

**State** — everything the policy may look at when deciding. Equation (6).

**Teacher-forced** — evaluation where each input carries the *expert's* previous
allocation, so the policy's errors never accumulate. Flattering; contrast
*closed-loop*.

**Tracking error** — mean squared distance between the policy's allocation and
the expert's. The main imitation-quality measure.

**Transaction cost (`c`)** — proportional cost of trading, charged on turnover.

**Turnover** — `||w_t − w_{t-1}||₁`, how much the portfolio changed. Reported,
never rewarded — the cost term already prices it.

**Utility loss** — `CE(expert) − CE(policy)`. Positive means worse than the
expert. Equation (17).

**vech** — flatten the lower triangle of a symmetric matrix into a vector.

**Wasserstein distance** — earth-mover's distance between two distributions. What
the GAIL critic estimates.

**Weights (`w`)** — the portfolio allocation. Non-negative, sums to 1, each below
a cap.
