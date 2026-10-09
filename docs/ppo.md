# Weeks 30–33 PPO Fine-Tuning

This document records the reward, the stochastic policy PPO required, the
hybrid-versus-pure comparison, the turnover and constraint results, and the
reading of research question 4 that the numbers actually support.

## The reward

Equation (10) gives the realised net return, charging the transaction cost
exactly once, and Equation (11) adds estimated risk and expert deviation:

```text
R_tilde_{t+1}(w_t) = w_t' r_{t+1} - c ||w_t - w_{t-1}||_1
rho_t = R_tilde_{t+1}(w_t) - (gamma/2) w_t' Sigma_hat_{t+1|t} w_t
        - lambda_E ||w_t - w*_t||_2^2
```

Turnover is **not** a separate term — the exposé is explicit that the cost term
already prices it, so adding it again would charge the same friction twice. It
is reported as an evaluation measure instead.

Equation (11) is now implemented twice: once in `PortfolioEnv` for the
Gymnasium interface, and once over the batched rollout tape, because PPO needs
many rollouts and stepping one path at a time is too slow. Two implementations
of one equation drift silently, so an acceptance check replays a trajectory
through both. They agree to `6.9e-18`.

## The stochastic policy PPO required

PPO's clipped surrogate needs a density, and every policy built so far is
deterministic. The Gaussian is placed in the **pre-projection** space:

```text
z ~ N(scores(s), diag(sigma^2)),   w = Pi(z)
```

The policy's `scores(s)` — its raw output before Equation (8) projects it — is
the mean, and the projection is treated as part of the environment. This keeps
`log pi(z|s)` exact, keeps every executed allocation feasible by construction,
and warm-starts from behavioural cloning with no adapter, which is what makes
"imitation-initialised" mean anything. An acceptance check confirms that
wrapping a policy does not change the allocation it produces.

The alternatives fail for concrete reasons. Sampling on the simplex then
clipping gives a density that is wrong wherever the clip binds — under a 0.30
cap that is most of the time, and precisely where the expert is most
distinctive. A Dirichlet has a clean density on the simplex but cannot represent
the expert's exact zeros or the position cap.

A measured detail: a **low** learning rate worked better. At `3e-5` the hybrid
reached a better certainty equivalent than at `2e-4`, despite a far smaller clip
fraction. The warm start is worth protecting, which is the reason the exposé
gives for choosing PPO in the first place.

## Results

12 paths of 900 periods, 8 assets, 40 PPO iterations, transformer policy,
shipped `regime_mixture` expert. **23 minutes**; every acceptance check passes.

Held-out paths. Utility loss is Equation (17), `CE(pi_E) - CE(pi)`, so
**negative means better than the expert**:

| Arm | CE (Eq. 16) | Utility loss | Ann. net return | Turnover | Constraint violation |
|---|---:|---:|---:|---:|---:|
| expert | −1.005e-3 | 0 | −9.78% | 0.0609 | 0 |
| behavioural cloning | −1.007e-3 | +2.5e-6 | −10.56% | 0.0588 | 0 |
| hybrid PPO | −1.002e-3 | −3.4e-6 | −10.32% | 0.0590 | 0 |
| **pure PPO** | **−5.51e-4** | **−4.54e-4** | **+3.20%** | **0.0025** | 0 |
| exact-gradient baseline | −8.59e-4 | −1.46e-4 | −6.96% | 0.0533 | 0 |

**Constraints hold everywhere**: zero violation for every arm at every
iteration, which the projection guarantees by construction rather than by
penalty.

**Turnover is not a concern** — it is the opposite of one. Pure PPO trades
**24× less** than the expert (0.0025 against 0.0609) and every other arm trades
slightly less than the expert. Nothing here needed a turnover penalty, which is
consistent with the exposé's decision not to add one.

### The pure baseline wins, and that is the finding

Pure PPO — random initialisation, `lambda_E = 0`, no expert information
anywhere — beats every other arm by a wide margin, and it is the only arm with a
**positive** annualised net return. The hybrid barely moves from behavioural
cloning at all.

The explanation is not that PPO is a better imitator. It is that **the expert is
not utility-optimal on this simulator**, and pure PPO is the only arm free to
notice. Weeks 9–13 established the reason: the canonical preset builds regime
means and volatilities with matched `linspace` ranges, so `mu_i / sigma_i^2` is
nearly constant across assets and equal weight sits essentially on the
mean-variance frontier; the expert's estimated moments then buy it nothing and
cost it transaction fees. Its annualised net return on these paths is −9.78%.

Pure PPO, optimising realised utility directly, finds a near-static allocation:
turnover 0.0025 means it picks a portfolio and holds it. That earns +3.20%
where the expert loses 9.78%.

So on this configuration the imitation warm start is a **handicap**, not an
advantage. The hybrid inherits the expert's behaviour, and PPO's trust region —
the very thing that makes fine-tuning safe — keeps it there.

### Reading research question 4 honestly

RQ4 asks whether PPO fine-tuning "can improve realised out-of-sample
certainty-equivalent utility without unacceptable turnover, drawdown, or
deviation from the expert benchmark".

On this evidence: **yes on utility and turnover, no on deviation** — and the two
are the same fact. The utility gain comes precisely *from* deviating from the
expert. An arm constrained to stay near the expert (the hybrid) captures almost
none of it: −3.4e-6 of utility loss recovered, against pure PPO's −4.54e-4, two
orders of magnitude apart.

The `lambda_E` ablation says the same thing more directly. Raising the
expert-deviation penalty from 0 to 0.1 made the hybrid slightly *worse*
(validation CE −5.595e-4 against −5.715e-4). Pulling towards an expert that is
underperforming does not help.

**This conclusion is configuration-specific and should not be generalised.** It
follows from the preset making equal weight near-optimal. On a preset where the
expert has a genuine edge — means and volatilities not rank-matched — the
ordering could reverse entirely, and that is the experiment worth running before
the final study fixes its configurations. It is the same preset limitation
flagged in `docs/markowitz_expert.md`.

### The exact-gradient baseline is not an upper bound on utility

It was included to bound what the reward can deliver, so that a weak PPO result
could be attributed to the estimator rather than to the objective. It does not
do that job, and the reason is instructive.

Measured on a separate run, exact gradients improve the **training** objective
while making held-out utility *worse*:

| | Train CE | Validation CE |
|---|---:|---:|
| start (behavioural cloning) | +2.774e-4 | −9.836e-4 |
| after exact-gradient optimisation | **+3.782e-4** | **−1.181e-3** |

The mechanism is specific to what makes the method possible: the exact gradient
differentiates through the **realised** return path, so it can tune the policy
to that particular sample path rather than to the distribution that generated
it. PPO's sampling noise acts as a regulariser it does not have.

So the baseline bounds the training reward, not achievable utility. On the
shipped run it lands between the hybrid and pure PPO (−8.59e-4), which is
consistent: it starts from the same imitation anchor as the hybrid and is
partly held there, while also overfitting.

## Acceptance checks

`evaluate_ppo` runs six:

- **reward_matches_environment** — the tape and `PortfolioEnv` implementations
  of Equation (11) agree. Run in float64, because in float32 the two disagree at
  about `1e-9` for representation reasons alone and the check could not then be
  set tight enough to catch a real drift;
- **action_feasibility** — every executed allocation satisfies Equation (5);
- **log_probability_correctness** — the hand-written Gaussian density matches
  `torch.distributions.Normal`, since that density is PPO's likelihood ratio;
- **warm_start_fidelity** — wrapping a policy in the actor-critic does not
  change its allocation. Checked on a *fresh* wrapper around the reference
  policy: comparing the trainer's own wrapper instead measures how much PPO
  moved the policy, which is the point of running it. That mistake was made
  first and the check duly reported 3.3e-3;
- **training_stability** — the approximate KL stays finite and bounded;
- **rollout_determinism** — repeating a deterministic rollout reproduces it.

As in the two preceding milestones, "PPO must beat the baseline" is **not** a
check. The work package says compare.

## Demonstration

```bash
uv run python scripts/run_ppo_demo.py --config configs/ppo_demo.yaml
```

About 23 minutes. `--skip-ablation` and `--iterations` are the fast paths.
Outputs land in `data/ppo_demo/`:

- `ppo_dashboard.png` — certainty equivalent and turnover per iteration for both
  arms, the held-out comparison, PPO's trust region in action, value-head
  explained variance, and the checks;
- `ppo_report.json`, `ppo_iterations.csv`, `ppo_comparison.csv`.

## Scope boundary

This completes the exposé's IL/RL pipeline. What remains is the theoretical
regret analysis (Weeks 18–35), robustness checks (Weeks 34–37), and the final
500-run Monte Carlo evaluation (Weeks 38–39).

Two things should be settled before those configurations are frozen. The preset
question above — whether the expert has a genuine edge — determines whether the
hybrid arm is worth carrying at all. And the runtime table in
`docs/markowitz_expert.md` still constrains the final study: the expert's
quadratic programme is the dominant cost and `N = 200` at 500 paths is not
feasible serially.
