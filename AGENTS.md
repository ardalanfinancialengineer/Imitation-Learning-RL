# AGENTS.md

## Project Context

This repository implements a university research project on:

**Imitation-Reinforcement Learning for Dynamic Multi-Period Mean-Variance Portfolio Optimization**

The project studies whether learned portfolio policies can imitate and potentially improve on a rolling Markowitz expert in a simulated, regime-switching financial market with transaction costs and portfolio constraints.

The implementation should follow the project exposé in `docs/expose.pdf`.

Core implementation order:

1. Regime-switching return simulation
2. Rolling-window / rolling-horizon Markowitz expert
3. Expert demonstration dataset
4. Behavioural cloning
5. DAgger / GAIL extensions
6. PPO fine-tuning
7. Evaluation, robustness checks, and reporting

Do not jump to advanced IL/RL models before the simulator, expert, and
pre-behavioural-cloning data pipeline are reliable.

## Documentation Context

Additional project context is available in:

- `docs/ProjectDescriptionIL.pdf`
- `docs/expose.pdf`

Coding agents should consult these documents when they need clarification about
the research motivation, modelling assumptions, intended pipeline, or evaluation
criteria.

These documents describe the broader project scope, including imitation learning
and reinforcement learning extensions. Do not implement later-stage methods early
just because they are mentioned there. Follow the staged implementation order in
this file and keep changes aligned with the current milestone.

## Engineering Priorities

Prioritize correctness, reproducibility, and no-look-ahead discipline over model complexity.

Important principles:

- All decisions at time `t` must use only information available up to time `t`.
- Simulated latent regimes may be used for diagnostics/evaluation, but should not be leaked into the main expert or policy inputs unless explicitly testing an oracle baseline.
- Prefer small, tested vertical slices over large unverified implementations.
- Keep APIs simple and stable because later IL/RL stages will depend on them.
- Use deterministic seeds where appropriate.
- Validate array shapes, probability vectors, covariance matrices, and portfolio constraints.

## Current Status

Work packages **Weeks 5-8 (simulation environment)**, **Weeks 9-13 (Markowitz
expert)**, **Weeks 14-16 (dataset and environment)**, **Weeks 17-21 (policy
model and behavioural cloning)**, **Weeks 22-25 (DAgger)**, **Weeks 26-29
(Wasserstein GAIL)**, and **Weeks 30-33 (PPO fine-tuning)** are implemented,
documented, and covered by focused tests. That completes the exposé's IL/RL
pipeline; what remains is the regret analysis, robustness checks, and the final
Monte Carlo study.

**New to this codebase? Read `docs/GUIDE.md` first.** It walks the pipeline end
to end and defines the domain vocabulary. `docs/audit.md` records what is built
against the exposé's work plan. `data/index.html` (built by
`scripts/run_all_demos.py`) shows every milestone's results in one page.

**Before the final configurations are frozen (Section 5.1), three decisions are
open — see `docs/open_decisions.md`.** They concern the preset making the expert
a weak benchmark, the infeasibility of `N = 200` at 500 paths serially, and
Equation (6) omitting the expert's horizon forecasts. Do not treat any of the
three as settled, and do not quietly work around them: each changes what the
study measures, and each is recorded with the measurements it came from.

Milestone documentation:

- `docs/simulation_environment.md` — Weeks 5-8
- `docs/markowitz_expert.md` — Weeks 9-13, including how `gamma` was calibrated
  against an oracle-moment run
- `docs/dataset_environment.md` — Weeks 14-16, including why states are
  assembled lazily and how the no-look-ahead check is controlled
- `docs/policy_behavioural_cloning.md` — Weeks 17-21, including why the head is
  a projection rather than a softmax, why the diffusion chain predicts the
  clean adjustment rather than the noise, and why generalisation is measured
  against validation rather than training
- `docs/dagger.md` — Weeks 22-25, including why closed-loop evaluation had to be
  added before the stage could be measured at all
- `docs/GUIDE.md` — end-to-end walkthrough and glossary; the entry point
- `docs/audit.md` — coverage of the exposé's work plan, gaps, and known debt
- `docs/open_decisions.md` — the three decisions outstanding before the freeze
- `docs/ppo.md` — Weeks 30-33, including why the pure-PPO baseline beats the
  hybrid on this preset and why the exact-gradient baseline is not an upper
  bound on utility
- `docs/gail.md` — Weeks 26-29, including why the policy gradient is replaced by
  backpropagation, the gradient-penalty versus weight-clipping stability result,
  and why the per-policy Wasserstein column must not be read as a comparison

Implemented files:

- `configs/base.yaml`
  - Defines a small readable base experiment with simulation size, seed, transition matrix, initial distribution, regime means, and constant-correlation covariance construction parameters.
  - Defines expert parameters and the default expert demonstration output directory.
- `src/simulation/regimes.py`
  - `RegimeSwitchingModel` validates regime counts, transition matrices, and initial distributions.
  - Regime sampling uses `np.random.default_rng` and returns integer regime labels with shape `(T,)`.
- `src/simulation/returns.py`
  - `ReturnSimulator` validates asset counts, regime means, covariance shapes, finite values, symmetry, and PSD-ness.
  - Return simulation produces a pandas `DataFrame` with shape `(T, n_assets)` and a regime-label array with shape `(T,)`.
- `src/expert/markowitz.py`
  - `RollingHorizonMarkowitzExpert` solves the exposé's multi-period programme (Equation 5) over `H = 21` days, with transaction costs inside the objective anchored on the previously implemented portfolio.
  - Only the first planned allocation is implemented; the programme is re-solved at the next date.
  - `risk_aversion` scales the **variance** term, matching the exposé. The earlier convention scaled the mean instead.
  - The CVXPY programme is compiled once per `(H, N)` and re-solved through parameters. Warm starting is off so a single decision is reproducible in isolation.
  - Numerical fallbacks return feasible portfolios when solver output is unusable, and are counted.
  - `RollingMarkowitzExpert` is retained as an `H = 1`, zero-cost wrapper so `scripts/run_expert.py` and the existing demonstration format keep working.
- `src/expert/constraints.py`
  - `PortfolioConstraints` validates and enforces budget, `l_i <= w_i <= u_i`, and leverage limits, and provides a Euclidean projection onto the feasible set.
- `src/expert/moments.py`
  - Covariance estimation options: rolling sample and Ledoit-Wolf shrinkage.
  - `RollingMomentForecaster` and `RegimeMixtureForecaster` both obey the contract that `forecast(t, horizon)` may use `returns[:t]` and nothing else.
  - Regime-mixture covariances follow the law of total variance, so they include between-regime dispersion and are genuinely horizon-dependent.
- `src/expert/regime_filter.py`
  - `GaussianHMM` fits by Baum-Welch with deterministic initialisation, so repeated fits on the same data are identical.
  - `CausalRegimeFilter` retains **every** refit; a query at `t` is answered with the most recent fit estimated at or before `t`. Keeping only the final fit was a real look-ahead leak that the acceptance checks caught.
  - Hidden states are canonically ordered by increasing average variance so the probability vector keeps a stable meaning across refits.
- `src/expert/diagnostics.py`
  - Six acceptance checks: feasibility, solver health, no-look-ahead replay from truncated history, determinism, turnover response to transaction cost, and runtime. A regime-filter accuracy check is added when simulator labels are supplied.
- `src/expert/config.py`
  - Single place where expert components are built from configuration, so ablations differ from the baseline in exactly one documented parameter.
- `scripts/run_expert_demo.py`
  - Weeks 9-13 visual report: dashboard, JSON evidence, per-decision CSV, and ablation CSV. Exits non-zero if any check fails.
- `scripts/check_state_sufficiency.py`
  - Measures the horizon variation of each moment forecaster and the
    behavioural-cloning error achieved against each, holding everything else
    fixed. This is what established that Equation (6) is **not** a sufficient
    statistic under the shipped `regime_mixture` expert.
- `scripts/check_expert_scale.py`
  - Per-decision solve times at `N in {50, 100, 200}` with platform metadata, for the later policy-versus-optimiser latency comparison.
- `src/imitation/rollout.py`
  - Closed-loop rollouts and `ExpertOracle`, which answers "what would the expert do here?" at an arbitrary anchor. That query is the Weeks 9-13 programme re-solved, not a re-derivation: an acceptance check asserts it reproduces the stored action bit-identically at the expert's own anchor.
  - Only `w_{t-1}` differs between a stored state and a learner-visited one, because the market blocks of Equation (6) depend on the path rather than on who is trading. So a visited state is the stored state with one block substituted, and the covariate shift is directly measurable as `mean_anchor_drift`.
  - Distinguishes teacher-forced, trajectory, and **closed-loop** tracking error. Only the last compares the policy against the expert's answer *at the state the policy reached*; the BC metric is teacher-forced and is structurally blind to what DAgger fixes.
- `src/imitation/dagger.py`
  - `AggregatedDataset` stores rounds as anchors plus expert answers and reassembles states on demand, preserving the Weeks 14-16 lazy design. It satisfies the `TrainingDataset` protocol rather than subclassing `DemonstrationDataset`.
  - The standardiser is frozen after round 0, and the learning rate is **not** lowered for warm-started rounds: measured, keeping `1e-3` improved closed-loop error while `3e-4` made every round worse than the starting point.
  - Round 0 records the behaviour-cloned policy, and the loop restores the best round by closed-loop error rather than returning the last.
- `src/imitation/dagger_diagnostics.py`
  - Six checks, five at zero tolerance: expert-query consistency, state-patch exactness, aggregation growth, closed-loop feasibility, rollout no-look-ahead, rollout determinism. The improvement itself is **reported with a paired bootstrap interval, not gated** — the work package says investigate, and a gate would invite tuning until it passed.
- `scripts/run_dagger_demo.py`
  - Weeks 22-25 visual report over two policies and two expert configurations. Exits non-zero if any check fails.
- `src/imitation/critic.py`
  - Wasserstein critic for Equation (9), with both Lipschitz methods the exposé cites. No batch normalisation anywhere: it would make a score depend on the batch a sample landed in.
  - Reports the **measured** Lipschitz constant. That number is what reveals whether the critic is solving its problem: an under-trained critic sits near 0.2 and returns a negative Wasserstein estimate, which cannot be a lower bound on a distance.
- `src/imitation/differentiable.py`
  - The closed-loop rollout is differentiable because only `w_{t-1}` carries action dependence, so `dW1/dtheta` comes from backpropagation rather than a policy gradient. Same objective, exact gradient, and no need for PPO before Weeks 30-33.
  - States are spliced (`cat(left, w_prev, right)`) rather than reassembled; an acceptance check confirms the splice equals the assembler.
  - Trap worth knowing: any objective depending on the action only through its **sum** has identically zero gradient, because the budget constraint pins `sum(w) = 1`.
- `src/imitation/gail.py`
  - Adversarial loop with truncated BPTT. Selection is on validation closed-loop tracking error rather than on `W1`, because `W1` is measured against a moving critic and is not comparable across BC, DAgger, and GAIL.
  - The critic budget is the setting that matters: too few or too slow critic steps and the gradient penalty dominates its loss entirely.
- `src/imitation/gail_diagnostics.py`
  - Seven checks. `rollout_gradient_correctness` is the load-bearing one and runs in **float64** — in float32 the finite-difference estimate itself carries about `1e-2` relative error and would hide a wrong gradient.
  - `compare_policies` scores several policies with one critic and is **not** a valid comparison; its docstring says so. The shipped run returns negative estimates and ranks the best policy worst, because a critic is near-optimal only for the pair it was trained on.
- `scripts/run_gail_demo.py`
  - Weeks 26-29 visual report over BC, DAgger, and GAIL from two starting points, plus the Lipschitz ablation. Exits non-zero if any check fails.
- `src/reinforcement/reward.py`
  - Equations (10), (11), (16), (17). Turnover is deliberately **not** a reward term: the exposé says the cost term already prices it.
  - Equation (11) is implemented here *and* in `PortfolioEnv`, because PPO needs batched rollouts. An acceptance check asserts the two agree; they match to 6.9e-18.
- `src/reinforcement/stochastic.py`
  - Gaussian placed on the **pre-projection** scores, with Equation (8) treated as part of the environment. That keeps `log pi(z|s)` exact, keeps allocations feasible, and warm-starts from imitation weights with no adapter.
- `src/reinforcement/ppo.py`
  - Clipped surrogate with GAE, a clipped value loss, and a KL budget. A **low** learning rate measured better (3e-5 beat 2e-4): the warm start is worth protecting.
- `src/reinforcement/analytic.py`
  - Exact-gradient baseline on the same reward. It bounds the **training** objective only — it differentiates through the realised return path and overfits it, improving train CE while worsening held-out CE.
- `src/reinforcement/diagnostics.py`
  - Six checks. `warm_start_fidelity` must be measured on a *fresh* wrapper around the reference policy; comparing the trainer's own wrapper measures how far PPO moved, which is the point of running it.
- `scripts/run_ppo_demo.py`
  - Weeks 30-33 visual report: hybrid PPO, pure PPO, and the exact-gradient baseline, with turnover, constraints, and utility.
- `src/dataset/state.py`
  - `StateSpec` and `StateAssembler` implement Equation (6). `block_slices` reports where each block sits, so policy attributions map back to economic inputs.
  - Two covariance encodings: full `vech` (default, faithful to Equation 6) and a `compact` summary that is linear rather than quadratic in `n_assets`.
- `src/dataset/paths.py`
  - `PathData` stores only raw ingredients — returns, regimes, expert weights and anchors, realised performance, and the HMM fits. States are **never** persisted; materialising them at final scale would need about 48 GB.
  - `build_expert()` reattaches the stored filter path instead of refitting, which is what makes reloading cheap. This is exact because Baum-Welch is deterministic here.
  - `build_path_from_returns` is split out so the no-look-ahead check can rebuild everything from modified returns.
- `src/dataset/splits.py`
  - All three Section 5.1 settings: path-level in-distribution holdout, contiguous ordered temporal holdout, and five named zero-shot shifts that each perturb exactly one aspect.
- `src/dataset/demonstrations.py`
  - `DemonstrationDataset` assembles states on access and works with a PyTorch `DataLoader`. Decisions without a full lookback are dropped, not zero-padded.
- `src/reinforcement/environment.py`
  - Gymnasium `PortfolioEnv` with the Equation (11) reward; the transaction cost is charged exactly once and turnover is reported but never re-added.
  - The state specification is `env.state_spec`, **not** `env.spec`: Gymnasium reserves `spec` for `EnvSpec` and shadowing it breaks `gym.make` and SB3's vector wrappers.
  - `expert_action()` exposes the expert decision for the current state, which DAgger will query.
- `src/dataset/diagnostics.py`
  - Six acceptance checks: state no-look-ahead, state integrity, action validity, split disjointness, environment consistency, and store roundtrip.
  - The no-look-ahead check compares two rebuilds that differ **only** in the future. Comparing against the stored path instead showed `1e-7`, which a control experiment traced to solver noise (`1e-14`) amplified by the `w_t`-to-`w_{t-1}` cost chain, not to a leak. The controlled comparison restores a `0.0` tolerance.
  - `anchor_chain_sensitivity` is reported, not asserted: expert trajectories are not bitwise reproducible across BLAS builds or thread counts, so reproducibility claims must be about *stored* paths.
- `scripts/run_dataset_demo.py`
  - Weeks 14-16 visual report: dashboard, JSON evidence, split/state/shift tables, and the stored paths. Exits non-zero if any check fails.
- `scripts/run_expert.py`
  - Loads `configs/base.yaml`, instantiates `RegimeSwitchingModel`, `ReturnSimulator`, and `RollingMarkowitzExpert`, and generates returns, regimes, observations, and actions.
  - Saves `returns.csv`, `regimes.csv`, `expert_actions.csv`, `demonstrations.npz`, and `metadata.json` to a configurable output directory.
  - Keeps the pipeline reproducible via the configured seed.
  - Saves regimes for diagnostics only; regimes must not be treated as ordinary policy inputs.
- `src/utils/data.py`
  - Provides validated demonstration loading for the current `.npz` format.
  - Provides chronological train/validation splitting without fitting statistics.
  - Provides return normalization helpers where fitted statistics can be computed on training data only and reused for validation/test data.
  - Includes a simple rolling-window observation-statistics helper.
- `src/evaluation/metrics.py`
  - Implements deterministic `sharpe_ratio`, `max_drawdown`, `calmar_ratio`, `turnover`, and `tracking_error` helpers with input validation and edge-case handling.
- `src/policy/simplex.py`
  - Differentiable Euclidean projection onto `{w : 1'w = 1, l <= w <= u}`, the
    full Equation (5) feasible set rather than the bare simplex.
  - Bisection localises the active set; the threshold is then recomputed in
    **closed form**, which is what makes the backward pass exact. Bisection
    alone carries no usable derivative.
  - `DEFAULT_BISECTION_STEPS = 30`, not more: the closed form does the precision
    work, and at batch size one the loop dominates inference cost.
- `src/policy/standardise.py`
  - Affine input standardisation fitted on **training data only**.
  - Return-window statistics are pooled across lags so a return encodes the same
    wherever it sits in the window.
  - Constant coordinates (`c`, `gamma`) are rescaled but **not** centred;
    centring erases them, which breaks silently once a policy is trained across
    several cost or risk-aversion settings.
- `src/policy/encoder.py`
  - `StateTokeniser` splits the flat state back into its Equation (6) blocks and
    emits `[CLS] + L` lag tokens `+ 5` context tokens.
  - `TransformerStateEncoder` (pre-norm) and `MLPStateEncoder` (the floor).
- `src/policy/deterministic.py`, `src/policy/diffusion.py`
  - Strategy 3 and strategy 4 of Section 5.2.
  - Both predict a **deviation from `w_{t-1}`**, read from the state's own
    block, with a zero-initialised read-out so training starts from "hold".
  - The diffusion chain regresses the **clean adjustment (`prediction:
    sample`)**, not the noise: ε-prediction amplifies error by
    `1/sqrt(alpha_bar)` against a sharply concentrated target and measured 17.8x
    worse on held-out tracking error, with nine times the expert's turnover.
  - Sampling is **deterministic without a `generator`** — one fixed noise vector
    shared across the batch. Per-row noise would make an allocation depend on
    which batch it was scored in, because `torch.randn` is not shape-invariant.
- `src/policy/reference.py`
  - `EqualWeightReference` and `PreviousWeightsReference` (never trade), the
    bar behavioural cloning must clear. The second is the demanding one.
- `src/policy/builder.py`
  - Single construction point, mirroring `src/expert/config.py`, so ablations
    provably differ only in what was changed.
- `src/imitation/losses.py`
  - `mse` (Equation `L_BC`, summed over assets so it matches Equation 18), `l1`,
    and `weight_divergence`. Utility loss is deliberately *not* here: it is the
    PPO reward of Equation (11).
- `src/imitation/bc.py`
  - Training loop, evaluation metrics, and checkpoint format.
  - Model selection on **validation tracking error**, not the training
    objective, which is not comparable across the two model families.
  - Optional state cache under a configured budget; a test asserts it changes
    no number.
  - `seed_everything` must be called **before** constructing the policy; weights
    are drawn there, not in `fit`.
- `src/imitation/diagnostics.py`
  - Seven acceptance checks: action feasibility, determinism, optimisation
    sanity, beats-naive-baselines, generalisation, inference latency, and
    end-to-end policy no-look-ahead.
  - Generalisation is **test versus validation**. Where the state is sufficient
    the training error collapses, so a train-versus-test ratio gates nothing;
    the train-to-test gap is reported as `memorisation_gap`, in the
    reported-not-asserted style of `anchor_chain_sensitivity`.
  - `beats_naive_baselines` requires a **paired bootstrap interval excluding
    zero**, not a bare inequality. Both heads initialise to `w_{t-1}`, i.e.
    exactly at the never-trade baseline, so a point estimate certifies nothing.
- `src/imitation/comparison.py`
  - Section 5.4 paired comparison: per-path tracking errors and a percentile
    bootstrap over **paths** (not decisions — decisions within a path are
    strongly dependent), plus seed-dispersion summaries.
  - **Known limitation of the milestone, documented in the milestone doc:**
    Equation (6) carries only the `tau = 0` moment forecast while the expert
    plans over `H = 21` with horizon-varying forecasts (67% variation in
    `mu_hat`, 43% in `Sigma_hat`). Imitation therefore has an error floor. A
    control run against a horizon-constant rolling forecaster reaches a
    tracking-error ratio of 0.143 against 0.778 for the shipped expert.
- `scripts/run_bc_demo.py`
  - Weeks 17-21 visual report: dashboard, JSON evidence, comparison and history
    CSVs, loss ablation, and checkpoints. Exits non-zero if any check fails.
  - Caches expert paths keyed on the simulation *and* expert configuration.
- `scripts/run_imitation.py`
  - Pipeline entry point: trains one policy from stored paths and writes a
    checkpoint plus metrics. No figures, no checks.
- `scripts/generate_test_dataset.py`
  - Generates a small smoke dataset plus diagnostics plot for local inspection.

Test coverage:

- `tests/test_simulation.py`
  - Reproducible regime sampling and return simulation.
  - Invalid transition matrices and invalid covariance inputs.
  - Output shapes and valid regime labels.
- `tests/test_expert.py`
  - Finite, nonnegative Markowitz weights that sum to one and respect `max_weight`.
  - Feasible fallback behavior.
  - Demonstration length and no-look-ahead rolling-window indexing.
- `tests/test_run_expert.py`
  - `scripts/run_expert.py` output files, shapes, feasible actions, metadata, decision times, and fixed-seed reproducibility.
- `tests/test_expert_constraints.py`
  - Bound validation, budget/box/leverage violation reporting, and projection feasibility.
- `tests/test_regime_filter.py`
  - Recovery of the calm/volatile/stressed volatility ordering, deterministic fits, valid probability vectors.
  - Two causality tests: the forward recursion at `t` ignores later returns, and truncating the path leaves earlier filtered probabilities unchanged.
  - `parameters_at(t)` returns the fit in force, not a later one.
- `tests/test_expert_moments.py`
  - Positive definiteness of both covariance estimators and better conditioning from shrinkage when `n_assets ~ window`.
  - Both forecasters are unchanged when returns after the decision date are corrupted.
  - Regime-mixture covariances are horizon-dependent and include between-regime variance.
- `tests/test_rolling_horizon_expert.py`
  - Feasibility of the executed weight and of every planned step.
  - Cost anchoring on the previous portfolio, and monotone turnover in `c`.
  - `H = 1` agrees with the single-period solution at zero cost; `H = 21` keeps trading where `H = 1` freezes.
  - No-look-ahead replay, reproducibility, solver health, leverage binding under shorting, and a 50-asset universe.
- `tests/test_expert_diagnostics.py`
  - A correct expert passes every check; broken cost monotonicity and an unreachable runtime budget fail theirs.
- `tests/test_run_expert_demo.py`
  - End-to-end run of the shipped configuration at reduced size: all checks pass, all artefacts written, report records the exposé settings.
- `tests/test_dataset_state.py`
  - Dimensions reproduce the exposé's 2380 figure at `N=50, L=20, K=3`; blocks are contiguous and land where the slices say; `vech` recovers the lower triangle; the compact encoding reports volatilities and correlation summaries.
- `tests/test_dataset_paths.py`
  - Path shapes, simplex-valid actions, save/load roundtrip, format-version rejection, and that a reloaded path reproduces forecasts exactly without refitting.
  - Dataset window is strictly past, targets match the decision, short windows are dropped, and the dataset works with a PyTorch `DataLoader`.
- `tests/test_dataset_splits.py`
  - Disjoint and complete path splits, contiguous non-interleaved temporal segments, and that every zero-shot shift really changes the data-generating process while staying a valid simulation.
- `tests/test_portfolio_environment.py`
  - Replaying the expert reproduces the stored trajectory exactly; the reward matches Equation (11); the cost is charged once; actions are projected; observations do not depend on the realised return.
- `tests/test_dataset_diagnostics.py`
  - A correct dataset passes every check at zero tolerance; corrupted actions and an inert zero-shot shift fail theirs.
- `tests/test_run_dataset_demo.py`
  - End-to-end run of the shipped configuration at reduced size.
- `tests/test_data_utils.py`
  - Demonstration loading and shape validation.
  - Chronological splitting.
  - Training-only return normalization behavior without validation leakage.
  - Rolling-window observation statistics.
- `tests/test_metrics.py`
  - Known-value checks for Sharpe ratio, maximum drawdown, Calmar ratio, turnover, and tracking error.
  - Input validation for malformed arrays.
- `tests/test_policy_simplex.py`
  - Agreement with `PortfolioConstraints.project` to `2e-16`, feasibility under
    extreme inputs, exact zeros, gradient against central finite differences,
    and the empty-active-set corner.
- `tests/test_policy_models.py`
  - Tokenisation shapes, sequence-order sensitivity, standardiser behaviour
    (including that constants survive and lags share statistics).
  - Feasible actions from all three policies; the deterministic policies start
    at `w_{t-1}`.
  - Diffusion: deterministic without a generator, **batch-invariant**,
    seed-reproducible with one, DDPM varies with the seed, and the calibrated
    scale gives a unit-variance target.
- `tests/test_behavioural_cloning.py`
  - Loss definitions, exact zeros in the divergence, cache-versus-lazy
    equivalence, reproducibility, training-only standardisation, early
    stopping, every objective, and checkpoint round-trip.
- `tests/test_policy_diagnostics.py`
  - A trained policy passes all seven; a state-independent policy fails the
    baseline check, an infeasible one fails feasibility, a random one fails
    determinism, and unreachable thresholds fail their own checks.
- `tests/test_run_bc_demo.py`
  - End-to-end run of the shipped configuration at reduced size, plus that the
    path cache is keyed on the expert configuration.
- `tests/test_policy_comparison.py`
  - A clear difference gives an interval excluding zero and coin-flip noise does
    not; the difference is taken path by path before averaging; a single path
    reports no interval rather than a fabricated one.

Verified command:

```bash
uv run pytest tests/ -q
```

Verified result:

```text
416 passed
```

Five stub modules that contained only `NotImplementedError` were deleted in the
Weeks 1-33 audit (`src/reinforcement/agent.py`, `src/evaluation/benchmarks.py`,
`src/utils/plotting.py`, `scripts/run_rl.py`, `scripts/run_evaluation.py`); each
had been superseded by a real implementation and was imported by nothing.
`src/imitation/agent.py` is deliberately kept as a signpost with no code.

Known debt, recorded in `docs/audit.md` rather than fixed: `src/utils/data.py`,
`scripts/run_expert.py`, `scripts/generate_test_dataset.py`, and the
`RollingMarkowitzExpert` wrapper form an older parallel pipeline that nothing
downstream consumes. Prefer `src/dataset/` for anything new.

Two defects in the Weeks 17-21 acceptance suite were found and fixed while
running it after the DAgger work. Both predate that work and reproduce on the
Weeks 17-21 commit.

**The fixture expert barely traded.** `tests/test_policy_diagnostics.py` used
`horizon = 5`, and a myopic expert compares a full transaction cost against a
single period of expected improvement: 77% of its decisions had exactly zero
turnover. The never-trade rule therefore reproduced the expert perfectly on 77%
of the data, and `beats_naive_baselines` was measuring the expert's inactivity
rather than the policy. The fixture now uses the exposé's `horizon = 21`, where
59% of decisions are frozen and the never-trade tracking error is 2.2x larger.
The policy reaches 0.78 of that baseline with a paired interval of
[+1.9e-3, +4.3e-3]; at `H = 5` it reached 0.91 with an interval straddling zero.

**`optimisation_sanity` was measuring luck.** It probed the loss reduction from
the *trained* policy's state. Across three training seeds on the same fixture
that came out as 1.66, 1.48, and 1.07e7 — seven orders of magnitude, so the
check passed or failed at random. The cause is the projection: coordinates
clamped at a bound receive exactly zero gradient, so once a policy has converged
onto the expert's many exact zeros the remaining batch loss is largely
irreducible and says nothing about gradient flow. The probe now re-initialises
the copy's network weights, keeping the fitted standardiser, which is what its
stated purpose ("a dead architecture, a detached graph, or a projection that
blocks gradients") actually calls for. The same fixture then gives 181.7, 27.9,
and 47.2 against a threshold of 5.0.

The check also had **no negative test**, which is why it could be unreliable
unnoticed; one was added using the state-independent policy, whose loss cannot
be reduced by any parameter update and which now correctly reports a reduction
of 1.0.

Not yet implemented:

- PPO or other reinforcement learning fine-tuning (Weeks 30-33)
- Full benchmark/evaluation pipeline scripts
- Generation of the zero-shot shift *paths*. The protocol that defines them is
  implemented; generating them belongs with the robustness milestone
  (Weeks 34-37), once the training configuration is frozen.

Note: `scripts/run_expert.py` and `src/utils/data.py` still serve the older
single-period demonstration format via the `RollingMarkowitzExpert` wrapper.
They are kept working for backward compatibility, but new imitation-learning
code should use `src/dataset/` and `src/dataset/demonstrations.py`, which build
the Equation (6) state from the rolling-horizon expert.

## Expected Commands

Use these commands from the repository root:

```bash
uv run pytest tests/ -q
```

```bash
uv run python scripts/run_simulation_demo.py --config configs/simulation_demo.yaml
```

```bash
uv run python scripts/run_expert_demo.py --config configs/expert_demo.yaml
```

```bash
uv run python scripts/run_dataset_demo.py --config configs/dataset_demo.yaml
```

```bash
uv run python scripts/run_dagger_demo.py --config configs/dagger_demo.yaml
```

```bash
uv run python scripts/run_gail_demo.py --config configs/gail_demo.yaml
```

```bash
uv run python scripts/run_ppo_demo.py --config configs/ppo_demo.yaml
```

```bash
uv run python scripts/run_all_demos.py
```

```bash
uv run python scripts/run_bc_demo.py --config configs/bc_demo.yaml
```
