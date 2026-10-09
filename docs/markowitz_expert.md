# Weeks 9–13 Rolling-Horizon Markowitz Expert

This document records the implemented expert benchmark: the multi-period
optimisation problem, the moment-estimation options, the causal regime filter,
the automated acceptance checks, and the runtime evidence.

## The optimisation problem

At each rebalancing date `t` the expert solves Equation (5) of the exposé over a
planning horizon of `H = 21` trading days:

```text
max   sum_{tau=t}^{t+H-1} [ w_tau' mu_hat_{tau+1|t}
                            - (gamma/2) w_tau' Sigma_hat_{tau+1|t} w_tau
                            - c ||w_tau - w_{tau-1}||_1 ]
s.t.  1' w_tau = 1
      l_i <= w_{i,tau} <= u_i
      ||w_tau||_1 <= L
```

`w_{t-1|t}` is anchored on the **previously implemented** portfolio, not on a
planned one. Only the first allocation `w*_t = w*_{t|t}` is executed; the
problem is re-solved at the next date. Later planning steps exist to price the
future benefit of a trade against its one-off cost, which is the whole reason
the exposé specifies a rolling *horizon* rather than a myopic optimiser.

In the long-only baseline `l_i = 0` and `||w_tau||_1 = 1`, so the leverage
constraint is redundant and is omitted from the compiled programme. It is
imposed as soon as `long_only` is disabled, which keeps the formulation ready
for the limited-shorting extension.

### Risk-aversion convention

`gamma` multiplies the **variance** term, exactly as in Equation (5). The
earlier single-period implementation on `arkan/complete-simulation-environment`
scaled the *mean* by `gamma` instead. The two are equivalent only up to a
reparameterisation, and the exposé convention is now used everywhere, including
inside the retained `RollingMarkowitzExpert` wrapper.

`gamma` is scale-invariant here: annualising both `mu` and `Sigma` multiplies
each by 252, so `mu / sigma^2` — and therefore the optimal allocation — is
unchanged. The textbook range `gamma in [1, 10]` applies directly to the daily
quantities, and the demo uses `gamma = 5`. See
"[Calibrating gamma](#calibrating-gamma)" for why.

### Solver

The problem is built once per `(H, N)` pair and re-solved through CVXPY
parameters. A parameterised `quad_form` is not DPP-compliant, so the risk term
is written as `||L' w||^2` with `Sigma = L L'` and `L'` supplied as a parameter.

Solver preference is **size-dependent**, based on measurements on this problem
shape (median milliseconds per decision, `H = 21`):

| Decision variables | CLARABEL | OSQP | SCS |
|---:|---:|---:|---:|
| 168 (`N=8`)   | **4.5** | 5.2 | 5.9 |
| 1050 (`N=50`) | **74.8** | 75.9 | 76.2 |
| 2100 (`N=100`) | 3457 | **517** | 343 |
| 4200 (`N=200`) | 10596 | **2141** | 1309 |

CLARABEL is an interior-point method: most accurate, and fastest until its
dense KKT factorisation starts to dominate. Above 1500 decision variables OSQP
leads and stays within about `2e-4` of CLARABEL's weights. SCS is faster still
but roughly an order of magnitude looser (about `1e-3` at small `N`), so it is
kept only as a fallback. The CVXPY default remains the last resort.

Making this switch cut `N = 100` from 3180 ms to 336 ms and `N = 200` from
10573 ms to 2697 ms per decision.

Warm starting is deliberately **off**. It would make each allocation depend on
the solves that preceded it, so replaying one decision in isolation — which
DAgger, the no-look-ahead check, and any audit of a single date all require —
would only reproduce it to solver tolerance rather than exactly. With the
programme compiled once, the saving was not worth that.

If the solver returns nothing usable, the expert falls back to the previous
portfolio when that is feasible (zero turnover), and to equal weights otherwise.
Fallbacks are counted and reported; the shipped configuration requires a
fallback rate of zero.

## Information structure

Index convention: `returns[t]` is the return realised *after* the decision taken
at index `t`. Decision `t` therefore conditions on `returns[:t]` and earns
`returns[t]`, which matches `F_t = sigma(r_1, ..., r_t)` in the exposé with
`returns[t] = r_{t+1}`.

Every moment forecaster obeys one contract: `forecast(t, horizon)` may use
`returns[:t]` and nothing else.

## Moment estimation

Three options are implemented, selectable from configuration.

**Rolling sample** — mean and covariance of the last `window` observations,
held constant across the horizon. The classical benchmark.

**Ledoit–Wolf shrinkage** — the same rolling window with a shrinkage covariance
estimator. This matters at the exposé's final scale: with `N = 50` and a
60-day window the sample covariance is singular, while the shrinkage estimator
stays well conditioned.

**Regime mixture** — the exposé's Section 4.1 construction. Filtered
probabilities are propagated through the estimated transition matrix and used
to mix the estimated regime-conditional Gaussians:

```text
q_h     = p_hat_t P^h
mu_h    = sum_k q_h(k) mu_k
Sigma_h = sum_k q_h(k) (Sigma_k + mu_k mu_k') - mu_h mu_h'
```

The covariance follows the law of total variance, so it also captures the
dispersion *between* regimes rather than only within-regime risk. This makes the
forecast covariance genuinely horizon-dependent: as `q_h` drifts towards the
stationary distribution, forecast risk widens with the planning step. A
configurable weight blends in rolling-window moments.

## Causal regime filtering

`GaussianHMM` estimates transition matrix, initial distribution, regime means,
and full regime covariances by Baum–Welch, with a scaled forward–backward
recursion. EM is seeded deterministically by ranking short blocks on
cross-sectional dispersion, so no random restarts are needed and repeated fits
on the same data are identical. Covariances are shrunk towards a diagonal
target and ridged, because per-regime effective sample sizes are small relative
to `N`.

`CausalRegimeFilter` wraps the model in a refit schedule. A refit at time `t`
uses `returns[:t]`; filtered probabilities at `s >= t` use that fit until the
next refit. Two properties make this honest:

1. **Every refit is retained.** A query at index `t` is answered with the most
   recent fit estimated at or before `t`. Keeping only the final fit is exactly
   the leak the acceptance checks caught during development: early decisions
   were silently using regime means estimated on later data.
2. **States are canonically ordered** by increasing average variance, so state
   0 is the calmest across all refits. EM leaves labels arbitrary, and without
   this the filtered probability vector would mean different things at
   different dates — a problem that would resurface when it becomes a policy
   input.

On the canonical preset with six assets, the filter recovers regimes at roughly
0.74 accuracy against a 0.33 chance baseline, and the fitted daily volatilities
(≈1.1%, 2.5%, 4.0%) track the preset targets.

Simulated regime labels are used only to score the filter. They never enter the
expert's information set.

## Quantitative acceptance checks

`evaluate_expert` runs six deterministic checks on one configuration:

- **feasibility** — every executed allocation satisfies budget, box, and
  leverage constraints;
- **solver_health** — the solver reports an optimal status at every date and the
  fallback rate stays within tolerance;
- **no_look_ahead** — re-deciding at sampled dates with the history *truncated
  at that date* reproduces the executed weights. A decision that used any future
  information cannot survive replacing `returns` with `returns[:t]`, so this is
  a direct test rather than a proxy;
- **determinism** — solving the same programme twice returns the same
  allocation;
- **transaction_cost_response** — average turnover falls as `c` rises, which is
  the economic signature of the cost term actually entering the objective;
- **runtime** — median solve time stays inside the configured budget.

A **regime_filter_accuracy** check is added when simulator labels are supplied.

## Horizon and cost interaction

The cost and horizon ablations show the behaviour the exposé motivates. At the
top of the prescribed cost range (`c = 0.005`), a myopic `H = 1` expert compares
a full transaction cost against a *single day* of expected improvement and stops
trading almost entirely (mean turnover ≈ 1e-3). The `H = 21` expert pays the
same cost once but collects the benefit over the whole horizon, and keeps
rebalancing. Within a fixed horizon, turnover falls monotonically as `c` rises.

## Demonstration

```bash
uv run python scripts/run_expert_demo.py --config configs/expert_demo.yaml
```

The default run uses eight assets, 2,500 periods, and seed 42. It writes a
dashboard plus machine-readable evidence to `data/expert_demo/`, and exits with
a non-zero status if any acceptance check fails:

- `expert_dashboard.png` — cumulative net return against equal weight, executed
  allocations, turnover across the cost grid, filtered regime probabilities,
  the solve-time distribution, and the check summary;
- `expert_report.json` — configuration, performance, ablations, and pass/fail
  evidence;
- `expert_decisions.csv` — per-date weights, turnover, returns, solve time, and
  solver status;
- `expert_ablations.csv` — one row per single-parameter ablation.

Decision-time cost at the final asset universes:

```bash
uv run python scripts/check_expert_scale.py
```

This times `N ∈ {50, 100, 200}` with `H = 21` and writes
`data/expert_demo/expert_scale_check.json`, including platform metadata so the
later policy-versus-optimiser latency comparison is interpretable.

### Runtime implication for the final experiment

Measured medians per decision after the solver switch: `N=50` 67 ms,
`N=100` 336 ms, `N=200` 2697 ms. Extrapolated to a full `T = 10,000` path:

| `N` | One path | 500 Monte Carlo paths |
|---:|---:|---:|
| 50  | ~11 min | ~4 days |
| 100 | ~56 min | ~19 days |
| 200 | ~7.5 h  | ~156 days |

Single-threaded on the recorded platform. This is not a defect — it is the
premise of the project, stated in the exposé's motivation and formalised as
research question 5: re-solving the quadratic programme at every rebalancing
date is exactly what becomes computationally demanding at scale, and it is what
the learned policy is meant to replace at inference time.

It does, however, constrain the plan. Generating expert demonstrations at
`N = 200` for 500 paths is not feasible as a serial computation. Options to
consider before Weeks 38–39: run the demonstration generation in parallel
across paths (they are independent), reserve `N = 200` for a smaller number of
evaluation paths while training on `N = 50`, or shorten `T` for the largest
universe. The decision should be recorded before the final configurations are
frozen.

## Calibrating gamma

An oracle check was used to separate "the optimiser is wrong" from "the
objective says that". Feeding the expert the **true** simulator moments
(regime-conditional `mu_k`, `Sigma_k` and the true `P`, conditioned on the true
`z_t`) and sweeping `gamma` on the demo path gives:

| gamma | Ann. return | Ann. vol | Sharpe | Mean weights |
|------:|------------:|---------:|-------:|--------------|
| 0.5   | +8.10%      | 29.4%    | +0.275 | tilted to high-mean assets |
| 1     | +6.73%      | 28.0%    | +0.240 | tilted to high-mean assets |
| 2     | +5.00%      | 24.7%    | +0.202 | **0.125 each — exactly equal weight** |
| 5     | +3.39%      | 23.0%    | +0.147 | tilted to low-volatility assets |
| 10    | +0.72%      | 21.4%    | +0.034 | concentrated in low-volatility assets |
| 20    | −0.08%      | 21.1%    | −0.004 | pinned at the cap on the two calmest |

Equal weight on the same path returns +4.99% at 24.7% volatility, Sharpe
+0.202. Three things follow.

1. **The optimiser is correct.** Return and volatility both fall monotonically
   in `gamma`; the run traces a proper efficient frontier.
2. **The preset makes equal weight the mean-variance optimum at `gamma ≈ 2`.**
   The canonical preset builds means and volatilities with matched `linspace`
   ranges, so `mu_i / sigma_i^2` is nearly constant across assets (3.06 to 3.33
   in the calm regime). The oracle reproducing equal weight to three decimals
   at `gamma = 2` is a strong joint check of the preset and the solver.
3. **High `gamma` is not a noise filter.** An earlier draft used `gamma = 20`
   because it suppressed noise-chasing under estimated means. That conflates
   risk aversion with estimation error: at `gamma = 20` even the *oracle* gives
   up all return for a 15% volatility reduction. Estimation noise is the job of
   the shrinkage estimator, not of `gamma`.

The gap between the oracle at `gamma = 5` (Sharpe +0.147, return +3.39%) and
the shipped configuration with estimated moments (Sharpe +0.006, return +0.17%)
is the **cost of estimation error** on this path. It is precisely the quantity
the exposé's regret decomposition (Equation 13, term `epsilon_t`) is about, and
it is worth carrying forward as a reference number: an imitation policy trained
on these demonstrations inherits that gap, and only the RL fine-tuning stage
has any mechanism to close it.

## Interpreting the performance numbers

The expert is a *benchmark*, not a trading strategy. On this preset it should be
judged on feasibility, stability, cost responsiveness, and runtime — and against
the oracle frontier above — rather than on beating equal weight, which the
preset construction places very close to the mean-variance optimum. Reported
returns are net of transaction costs; the equal-weight benchmark is deliberately
given the favourable treatment of paying none.

If a later milestone needs an expert with a genuine cross-sectional edge, the
preset should be given means and volatilities that are not rank-matched, rather
than the expert being retuned.

## Scope boundary

This milestone implements and validates the expert benchmark. State-vector
assembly `s_t = [r_{t-L+1:t}, mu_hat, vech(Sigma_hat), p_hat_t, w_{t-1}, c,
gamma]`, train/validation/test path splitting, the environment wrapper, and the
demonstration dataset for behavioural cloning belong to Weeks 14–16.
`scripts/run_expert.py` still emits the older single-period demonstration format
and is unchanged by this milestone.
