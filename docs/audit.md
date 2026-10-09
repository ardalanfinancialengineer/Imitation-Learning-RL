# Codebase Audit Against the Exposé — Weeks 1–33

A full check of what is built against what the exposé specifies, done at the
point where implementation pauses for the supervisor conversation in
[`open_decisions.md`](open_decisions.md).

**Verdict: the work plan is complete and correct through Week 33.** Every
milestone the exposé schedules up to that point is implemented, documented, and
covered by acceptance checks. Four gaps and inconsistencies were found; three
were fixed during the audit and one is correctly deferred.

**One acceptance check does fail at full size.** The transformer-diffusion
policy fails `beats_naive_baselines` in the Weeks 17-21 demo. It is a real
result rather than a code fault — path generation was verified bit-reproducible
and the relevant code and configuration are unchanged since the milestone. It is
recorded as [open decision 4](open_decisions.md). The deterministic policies and
every other milestone pass.

---

## 1. Work plan coverage

The exposé's Table 1 schedules the project by week.

| Weeks | Work package | Status |
|---|---|---|
| 1–4 | Literature and planning | Not code |
| 5–8 | Simulation environment | **Complete** — `src/simulation/`, 5 checks |
| 9–13 | Markowitz expert | **Complete** — `src/expert/`, 7 checks |
| 14–16 | Dataset and environment | **Complete** — `src/dataset/`, 6 checks |
| 17–21 | Policy model and behavioural cloning | **Complete** — `src/policy/`, `src/imitation/bc.py`, 7 checks |
| 22–25 | DAgger training | **Complete** — `src/imitation/dagger.py`, 6 checks |
| 26–29 | GAIL extension | **Complete** — `src/imitation/gail.py`, 7 checks |
| 30–33 | PPO fine-tuning | **Complete** — `src/reinforcement/`, 6 checks |
| 18–35 | Theoretical analysis | Written work, runs in parallel — not code |
| 34–37 | Robustness checks | Not started (correctly — blocked on open decisions) |
| 38–39 | Final evaluation | Not started (correctly) |
| 40 | Report and submission | Not started |

## 2. Methodology coverage

Every numbered element of Section 4 is implemented.

| Exposé | Element | Where |
|---|---|---|
| §4.1 | Regime-switching simulation | `src/simulation/regimes.py`, `returns.py` |
| §4.1 | HMM regime filtering, Eq. (4) | `src/expert/regime_filter.py` |
| §4.1 | Ledoit–Wolf shrinkage | `src/expert/moments.py` |
| §4.2 | Rolling-horizon expert, Eq. (5) | `src/expert/markowitz.py` |
| §4.3 | State and action, Eq. (6)–(7) | `src/dataset/state.py` |
| §4.4 | Transformer-diffusion policy, Eq. (8) | `src/policy/diffusion.py`, `simplex.py` |
| §4.5 | Behavioural cloning | `src/imitation/bc.py` |
| §4.5 | DAgger | `src/imitation/dagger.py` |
| §4.5 | Wasserstein GAIL, Eq. (9) | `src/imitation/gail.py`, `critic.py` |
| §4.6 | PPO reward, Eq. (10)–(11) | `src/reinforcement/reward.py`, `ppo.py` |
| §4.7 | Regret analysis, Eq. (12)–(13) | Theoretical; see gap 4 below |

**Section 5.2 strategies — all six exist and are runnable:**

1. equal weight — `EqualWeightReference`
2. rolling-horizon expert — `RollingHorizonMarkowitzExpert`
3. deterministic transformer policy — `build_policy(kind="transformer")`
4. transformer-diffusion policy — `build_policy(kind="diffusion")`
5. hybrid imitation-initialised PPO — `run_ppo_demo.py`, hybrid arm
6. pure PPO — `run_ppo_demo.py`, pure arm

**Section 5.1 splits** — all three settings implemented in `src/dataset/splits.py`:
path-level in-distribution holdout, contiguous temporal holdout, and five
zero-shot shift specifications. The shift *paths* are not generated, which is
correct: that belongs to Weeks 34–37, after the configuration freeze.

## 3. Gaps found

### 3.1 Sortino ratio was missing — **fixed**

Section 5.3 lists "annualised return, annualised volatility, Sharpe ratio,
Sortino ratio, maximum drawdown, turnover, certainty-equivalent utility, expert
tracking error, utility loss, and decision-time latency". `Sortino` was absent,
and annualised return and volatility were computed ad hoc inside demo scripts
rather than existing as functions.

Added to `src/evaluation/metrics.py` with tests: `sortino_ratio`,
`annualised_return`, `annualised_volatility`. The Sortino ratio matters more
here than it usually would: the transaction-cost term makes the return
distribution asymmetric, and the Sharpe ratio penalises upside variation
identically to downside.

### 3.2 Section 5.3 measures were split across two modules — **fixed**

Certainty-equivalent utility and utility loss (Equations 16 and 17) live in
`src/reinforcement/reward.py`, because they need the reward's risk-aversion
parameter, while the rest live in `src/evaluation/metrics.py`. Nothing was
wrong, but there was no single import site for "the Section 5.3 measures".

`src/evaluation/__init__.py` now re-exports the series-only measures and its
docstring says explicitly where the other two live and why.

### 3.3 Five dead stub modules — **fixed**

Five files contained only `NotImplementedError` bodies and were imported by
nothing. They dated from the initial scaffold and had been superseded by real
implementations, but a reader could not tell them apart from working code:

| Deleted | Superseded by |
|---|---|
| `src/reinforcement/agent.py` | `src/reinforcement/ppo.py` |
| `src/evaluation/benchmarks.py` | `src/policy/reference.py` |
| `src/utils/plotting.py` | the dashboard code in each demo |
| `scripts/run_rl.py` | `scripts/run_ppo_demo.py` |
| `scripts/run_evaluation.py` | each demo's own comparison section |

Removing them also cleared the three long-standing lint warnings; `ruff` is now
clean across the whole tree.

`src/imitation/agent.py` was **kept**: it contains no code, only a documented
pointer explaining where the behavioural-cloning implementation moved to and
why the original softmax-head design was abandoned. That is useful signposting.

### 3.4 Diebold–Mariano and Giacomini–White tests — **correctly deferred**

Section 5.4 names both, and Equation (19) defines the per-period loss
differential they would consume. Neither is implemented.

This is not a gap in Weeks 1–33. The exposé is conditional about them —
"predictive-ability tests will only be used when a suitable sequence of
per-period loss differences can be defined" — and they belong to the final
evaluation in Weeks 38–39. What *is* implemented is the machinery they would
build on: paired bootstrap confidence intervals over matched paths, in
`src/imitation/comparison.py`, which Section 5.4 requires and which every
milestone comparison already uses.

**Carry forward:** implement both in Weeks 38–39, consuming Equation (19)'s
`δ_t = ℓᴬ_t − ℓᴮ_t`, with an explicit statement of the loss sequence,
dependence correction, and hypothesis, as Section 5.4 demands.

## 4. Things checked and found sound

- **No look-ahead.** Every stage tests it, and the tests are strong: they rewrite
  the future and demand bit-identical results, using a control-versus-treatment
  comparison so floating-point noise cannot mask a real leak. The Weeks 14–16
  version of this check caught a genuine leak during development.
- **Constraints.** Equation (5)'s budget, box, and leverage constraints are
  enforced by projection rather than by penalty, so they hold by construction.
  Measured violation is exactly `0` for every policy in every milestone.
- **Reproducibility.** Seeds are set before construction, not only before
  training. The one genuine limit is documented: expert trajectories are not
  bitwise reproducible across different BLAS builds, because the cost term chains
  decisions and amplifies last-bit differences. Claims of exactness are made
  about *stored* paths.
- **Configuration hygiene.** No experiment setting is hard-coded in a script;
  each demo writes the config it used into its report.
- **Equation (11) implemented twice** — in `PortfolioEnv` and over the batched
  tape — which is a real duplication risk. It is guarded by an acceptance check
  that replays a trajectory through both; they agree to `6.9e-18`.
- **Test suite.** 406 tests before the audit, 416 after the additions. One
  file per module, and each acceptance check has a negative test proving it
  fails when it should.

## 5. Known technical debt

Not defects, but worth knowing before extending the code.

**The legacy demonstration path.** `src/utils/data.py`, `scripts/run_expert.py`,
`scripts/generate_test_dataset.py`, and the `RollingMarkowitzExpert` wrapper form
a parallel, older pipeline (about 700 lines plus three test files) that predates
`src/dataset/`. Nothing downstream consumes it. It was kept for backwards
compatibility during the milestones; that reason has now expired, and it is the
single largest source of "which of these two things should I use?" confusion in
the tree.

**Recommendation:** delete it once you are confident nothing outside this
repository depends on the old `.npz` demonstration format. It is a clean,
self-contained removal. Not done during this audit because deleting ~700 lines
of working, tested code is a bigger call than removing unambiguous stubs, and it
is your call rather than mine.

**Annualisation is duplicated.** The `× 252` scaling appears in several demo
scripts rather than routing through `src/evaluation/metrics.py`. Now that the
canonical functions exist, new code should use them. Existing demos were left
alone deliberately: rewiring them would change numbers in dashboards that the
milestone documents quote, for no correctness gain.

**Demo runtimes are long.** DAgger ~30 min, PPO ~23 min, GAIL ~18 min at shipped
sizes. `scripts/run_all_demos.py` provides a quick profile (~12 min for all
seven) for when you just need to see everything work.

## 6. What to do next

1. Resolve the three items in [`open_decisions.md`](open_decisions.md) with your
   supervisor. Weeks 34–39 cannot start honestly until they are settled, because
   two of them change what the study measures.
2. Then Weeks 34–37 (robustness) and 38–39 (final Monte Carlo), adding the
   Diebold–Mariano and Giacomini–White tests from §3.4.
3. Optionally, delete the legacy path in §5 before building further.
