# Imitation-Reinforcement Learning for Dynamic Portfolio Optimization

A university research project on:

**Imitation-Reinforcement Learning for Dynamic Multi-Period Mean-Variance Portfolio Optimization**

Can a neural network learn to reproduce the decisions of a classical portfolio
optimiser — and then, with reinforcement learning, improve on them? The study
runs entirely on simulated regime-switching markets, so the true data-generating
process is known and every experiment is repeatable.

## Start here

| I want to… | Go to |
|---|---|
| **See every result in one page** | `data/index.html` — build with `uv run python scripts/run_all_demos.py` |
| **Understand what the project does** | [`docs/GUIDE.md`](docs/GUIDE.md) — full walkthrough, defines every domain term |
| **Know what is still undecided** | [`docs/open_decisions.md`](docs/open_decisions.md) |
| **Check what is built vs. planned** | [`docs/audit.md`](docs/audit.md) |
| **Read one milestone in depth** | the per-milestone documents in [`docs/`](docs/) |

```bash
uv sync --extra dev                             # set up
uv run python scripts/run_all_demos.py          # all seven demos, ~12 min
# then open data/index.html
```

## Status

The exposé's imitation and reinforcement-learning pipeline is **complete through
Week 33**. Weeks 34–39 are paused pending the three decisions in
[`docs/open_decisions.md`](docs/open_decisions.md), two of which change what the
final study measures.

| Weeks | Milestone | Demo | Document |
|---|---|---|---|
| 5–8 | Market simulation | `run_simulation_demo.py` | [simulation_environment.md](docs/simulation_environment.md) |
| 9–13 | Rolling-horizon Markowitz expert | `run_expert_demo.py` | [markowitz_expert.md](docs/markowitz_expert.md) |
| 14–16 | State, dataset, environment | `run_dataset_demo.py` | [dataset_environment.md](docs/dataset_environment.md) |
| 17–21 | Policies and behavioural cloning | `run_bc_demo.py` | [policy_behavioural_cloning.md](docs/policy_behavioural_cloning.md) |
| 22–25 | DAgger | `run_dagger_demo.py` | [dagger.md](docs/dagger.md) |
| 26–29 | Wasserstein GAIL | `run_gail_demo.py` | [gail.md](docs/gail.md) |
| 30–33 | PPO fine-tuning | `run_ppo_demo.py` | [ppo.md](docs/ppo.md) |
| 34–39 | Robustness and final Monte Carlo | — | not started |

Each milestone ships deterministic acceptance checks, and its demo exits
non-zero if any fail. 416 tests, `ruff` and `mypy` clean.

## Repository layout

```text
src/
  simulation/     regime-switching market generation
  expert/         the Markowitz optimiser, its moment estimators and regime filter
  dataset/        Equation (6) state, path storage, Section 5.1 splits
  policy/         network architectures and the feasible-set projection
  imitation/      behavioural cloning, DAgger, GAIL, shared rollout machinery
  reinforcement/  Equation (11) reward, Gymnasium environment, PPO
  evaluation/     Section 5.3 performance measures
  utils/          legacy demonstration format; new code should use src/dataset
scripts/          one demo per milestone, plus run_all_demos.py
configs/          one YAML per demo; every experiment setting lives here
docs/             guide, audit, open decisions, one document per milestone
tests/            one file per module
data/             demo outputs (gitignored) and index.html
```

## Setup

This project uses `uv`.

Install `uv` if needed:

```bash
pip install uv
```

Clone and enter the repository:

```bash
git clone https://github.com/AMLDS-IL-portfolio-optimization/il-rl-portfolio-optimization.git
cd il-rl-portfolio-optimization
```

Create/sync the local environment:

```bash
uv sync --extra dev
```

## Verifying and running

```bash
uv run pytest tests/ -q          # 416 tests, about 12 minutes
uv run ruff check src scripts tests
uv run mypy src
```

### All demos at once

```bash
uv run python scripts/run_all_demos.py                # quick profile, ~12 min
uv run python scripts/run_all_demos.py --full         # shipped sizes, ~90 min
uv run python scripts/run_all_demos.py --only ppo     # a single milestone
uv run python scripts/run_all_demos.py --index-only   # rebuild the page only
```

Then open `data/index.html`. The quick profile reduces path counts, path
lengths, and training iterations so the whole pipeline can be seen working in
one sitting; the numbers quoted in the milestone documents come from `--full`.

### One demo at a time

Each writes a dashboard, a JSON report, and CSV tables to its own directory
under `data/`, and exits non-zero if any acceptance check fails.

| Demo | Command | Runtime |
|---|---|---|
| Simulation | `uv run python scripts/run_simulation_demo.py --config configs/simulation_demo.yaml` | ~1 min |
| Expert | `uv run python scripts/run_expert_demo.py --config configs/expert_demo.yaml` | ~3 min |
| Dataset | `uv run python scripts/run_dataset_demo.py --config configs/dataset_demo.yaml` | ~2 min |
| Behavioural cloning | `uv run python scripts/run_bc_demo.py --config configs/bc_demo.yaml` | ~6 min |
| DAgger | `uv run python scripts/run_dagger_demo.py --config configs/dagger_demo.yaml` | ~30 min |
| GAIL | `uv run python scripts/run_gail_demo.py --config configs/gail_demo.yaml` | ~18 min |
| PPO | `uv run python scripts/run_ppo_demo.py --config configs/ppo_demo.yaml` | ~23 min |

All accept `--output-dir` and size overrides; run any with `--help`.

### Scale and diagnostic checks

```bash
uv run python scripts/check_simulation_scale.py    # N in {50,100,200} simulation timing
uv run python scripts/check_expert_scale.py        # expert solve time at final scale
uv run python scripts/check_state_sufficiency.py   # is Equation (6) sufficient?
```

## Project notes

The specification is `docs/expose.pdf`; `AGENTS.md` records the working
conventions and the current implementation status. There is a sibling
documentation repository at `../il-rl-portfolio-docs`.

Two conventions that are easy to violate by accident:

- **Latent regimes are for diagnostics only.** The simulator returns the true
  regime labels so results can be scored against them. They must never be fed
  to the expert or a policy as an input, except when explicitly testing an
  oracle baseline.
- **Decisions at time `t` may use information up to `t` and no later.** Every
  milestone tests this, and the tests demand bit-identical results rather than
  approximate agreement. See "No look-ahead" in [`docs/GUIDE.md`](docs/GUIDE.md).

## License

MIT
