# Project State — Working Notes

*Last updated: 2026-09-23. Author of this note: working session with Nick.*
*This is a personal orientation/working document, separate from Arkan's formal
[`audit.md`](audit.md) and [`open_decisions.md`](open_decisions.md), which
remain the authoritative status. Read those two first; this collects what we
established in the session plus the things still to chase down.*

---

## 1. Where the code lives (read this first)

- **`main` is a skeleton.** Every function is `raise NotImplementedError`. None
  of the real work is merged to `main` as of 2026-09-23.
- **All real code is on one linear branch chain**, tipped by
  **`arkan/ppo-fine-tuning`**. Each branch is an ancestor of the next, so the
  tip is a superset of everything:

  ```
  main
   → feature/foundational-simulation-expert
   → feature/expert-data-pipeline-metrics
   → arkan/complete-simulation-environment   (Weeks 5–8)
   → arkan/markowitz-expert                  (Weeks 9–13)
   → arkan/dataset-environment               (Weeks 14–16)
   → arkan/policy-behavioural-cloning        (Weeks 17–21)
   → arkan/dagger-training                   (Weeks 22–25)
   → arkan/gail-extension                    (Weeks 26–29)
   → arkan/ppo-fine-tuning                   (Weeks 30–33)  ← TIP, has everything
  ```

- **We are currently checked out on `arkan/ppo-fine-tuning`** (HEAD `bd52fcd`).
- **Open TODO:** merge the tip into `main` (or open a PR). `main` being empty is
  the single most confusing thing about the repo for anyone new.

## 2. Progress against the exposé

Code-complete and acceptance-checked through **Week 33**. Not started (correctly
— blocked on open decisions): Weeks 34–37 (robustness), 38–39 (final 500-run
Monte Carlo), 40 (report).

| Weeks | Package | Status |
|---|---|---|
| 5–8 | Regime-switching simulation | Complete |
| 9–13 | Rolling-horizon Markowitz expert | Complete |
| 14–16 | State/dataset + `PortfolioEnv` | Complete |
| 17–21 | Transformer-diffusion policy + BC | Complete (see §5 caveat) |
| 22–25 | DAgger | Complete |
| 26–29 | Wasserstein GAIL | Complete |
| 30–33 | PPO fine-tuning (hybrid + pure) | Complete |
| 34–39 | Robustness + final study | **Blocked** on decisions below |

Toolchain: this repo uses **`uv`**, not bare `python`/`pip`. `uv sync --extra dev`
once, then `uv run pytest tests/ -q` and `uv run python scripts/run_all_demos.py`.

## 3. The four open decisions (must settle before the config freeze)

Full detail in [`open_decisions.md`](open_decisions.md). Section 5.1 requires
freezing final configs before the 500-run study; changing any of these
afterwards invalidates results.

1. **The expert is a weak benchmark** *(supervisor call)*. Canonical 3-regime
   preset has rank-matched means/vols → near-degenerate frontier, equal-weight ≈
   optimal. Pure PPO (+3.20%) beats the expert (−9.78%) on held-out paths. Fix:
   reframe the expert as a *computational* benchmark, or change the preset
   (invalidates all measured numbers). A ~1-hour preset experiment would settle
   it empirically.
2. **N=200 at 500 paths is infeasible serially** (~156 core-days) *(our call)*.
   Parallelise across paths (`OMP_NUM_THREADS=1` per worker). See §6 for the
   machine reality.
3. **State omits horizon forecasts** (Eq. 6 carries only τ=0) *(supervisor
   call)*. Caps imitation quality and masks the transformer-vs-MLP comparison
   (RQ3). Fix: extend the state.
4. **Transformer-diffusion policy** *(supervisor call)* — see §5, which is now
   more nuanced than the report draft states.

Sequencing: decide #2 ourselves; take #1, #3, #4 to the supervisor together
(#1 leads, since it changes the others' numbers).

## 4. The progress update to Ardalan

- Draft written: **`docs/progress_update_2026-09.tex`** (LaTeX, ~2–3 pages,
  fully candid, with an evidence appendix). Compiled PDF is at repo root
  (`progress_update.pdf`).
- **Requested by 30 Sep 2026**, Word or LaTeX. Meeting in early October.
- Authored jointly (Nick + Arkan) per the exposé authorship.
- **⚠ Needs a fix before sending** — see §5. The draft asserts the diffusion
  policy "fails its baseline"; our own `--full` run does not reproduce that at
  the size it runs. Resolve §5 first, and correct decision-#2's core estimate to
  the 15-core M5 reality (§6) — the draft's "~10 days on 16 cores" assumed 16
  equal cores.

## 5. ⚠ UNRESOLVED: diffusion policy — the run disagrees with the docs

**This is the most important open thread from the session and must be nailed
down before the supervisor update goes out.**

`open_decisions.md` decision #4 and the report draft say the diffusion policy
**fails** `beats_naive_baselines` at full size: tracking error 1.022 (worse than
never-trading), latency 0.55× (slower than the expert).

But our `run_all_demos.py --full` run on 2026-09-23 shows the diffusion policy
**passing every check**:

| Metric | Docs / report (decision #4) | Our `--full` run |
|---|---|---|
| `beats_naive_baselines` | fails | **passes** |
| tracking error | 1.022 | **0.00364** (beats `previous_weights` 0.00379) |
| latency speedup | 0.55× | **1.32×** |

**Why they don't actually contradict (what we found):**

- `run_all_demos.py --full` runs BC at the **development size** in
  `configs/bc_demo.yaml`: `n_assets=8, T=1200, n_paths=30, seeds=[0,1,2]`. This
  is **not** the exposé's full scale (N∈{50,100,200}, T=10000, 500 paths).
  Decision #4 was measured "re-running the Weeks 17-21 demo at full size,"
  which appears to mean a different/larger configuration than the orchestrator's
  `--full` profile.
- The diffusion `min_latency_speedup` threshold is **overridden to 0.5** (vs 3.0
  for the others) — present since the milestone commit `f25f34b`, a deliberate
  acknowledgment that a K-step diffusion chain can't hit the 3× bar. So "passes"
  here means "clears the lowered bar at the small dev size," not "is fast."
- **`latency_speedup` is machine-specific.** Our 1.32× is on the M5 Pro; Arkan's
  0.55× on his machine. This number is not portable and must never be quoted
  without stating the hardware.

**What is still genuinely open:**

- Which configuration produced decision #4's 1.022 / 0.55× numbers? Is it stale
  (config drifted since the audit), or simply a larger standalone
  `run_bc_demo.py` run that the orchestrator's `--full` doesn't replicate?
- **Action before editing the report:** run the diffusion policy at (or near)
  the intended full scale via the standalone `scripts/run_bc_demo.py` with the
  final-study sizes, on a known machine, and record it. Only then state decision
  #4 as fact. Right now the report over-claims a clean failure that our freshest
  evidence doesn't show.

## 6. Compute reality (feeds decision #2 and the report)

- **This machine: Apple M5 Pro, 15 cores = 5 performance + 10 efficiency,
  48 GB RAM.** No CUDA (Apple Silicon); MPS exists but see §7.
- E-cores run sustained FP work at ~⅓–½ a P-core, so realistic path-parallel
  speedup is **~6–9×, not 15×**. Applied to decision #2:
  - N=50: ~4 days serial → ~12 h
  - N=100: ~19 days → ~2.4 days
  - **N=200: ~156 days → ~20 days** — still infeasible on the laptop alone.
- **N=200 wants a cluster or cloud box**, or a fallback (train N=50 / evaluate
  N=200 on fewer paths). Options:
  - **KIT bwUniCluster / bwForCluster** (bwHPC) — free, SLURM array job maps
    perfectly onto path-parallel work. Ask Ardalan / the C-RAM chair.
  - **Cloud CPU rental** — a 64–96 vCPU box runs the whole study in a few days
    for ~$100–300 (less on spot); resumable since each path writes its own `.npz`.
- **Not yet built:** the path-parallel harness itself. Design agreed:
  `parallel_map(fn, args, workers=N)` helper, `spawn` context, BLAS pinned to 1
  thread/worker, `--workers` CLI flag, bit-identical-to-serial (seed lives in
  each path spec). Machine-agnostic — same code for laptop / Arkan's box /
  cluster. **TODO: implement.**

## 7. Device / Apple-Silicon changes made this session

Committed to the working tree (not yet a git commit):

- **New `src/utils/device.py`** with `resolve_device()`, wired into all four
  trainers (`bc`, `gail`, `ppo`, `analytic`). Adds a `device: "auto"` sentinel.
- **`"auto"` picks CUDA-if-present, else CPU — never MPS.** Deliberate: the RL
  tape, `PortfolioEnv`, and GAIL/RL diagnostics use **float64**, which MPS
  doesn't support. Auto-selecting MPS would *break* the RL stages, not just slow
  them. `"mps"` is honored only if asked for explicitly, with a warning.
- Unavailable explicit requests (e.g. `cuda` on the Mac) **warn and fall back to
  CPU** rather than crash — so a config written for one machine runs on another.
- Default stays `"cpu"` everywhere → no change to existing runs or acceptance
  numbers. New `tests/test_device.py` (6 tests) pins the "auto never picks MPS"
  invariant. All trainer tests still pass; ruff clean.
- **Practical rule: run the demos on `cpu` or `auto`. Never force `mps`** for the
  RL/GAIL stages.
- A read-only macOS-compatibility audit found the repo otherwise clean: no
  fork/multiprocessing, `pathlib` everywhere, no `shell=True`, no thread-env
  tampering, all CUDA touches guarded. The only hazard is the intended float64/MPS
  one above.

## 8. Latest full run (2026-09-23, this machine, all CPU)

`uv run python scripts/run_all_demos.py --full` — all 7 demos passed, 2398 s total.

| Stage | Time |
|---|---|
| [5-8] simulation | 1 s |
| [9-13] expert | 75 s |
| [14-16] dataset/env | 51 s |
| [17-21] policies + BC | 850 s |
| [22-25] DAgger | 585 s |
| [26-29] GAIL | 385 s |
| [30-33] PPO | 451 s |

Note these are **development sizes**, not final-study scale (§5). Dashboards and
JSON reports are under `data/` — open `data/index.html`.

Prior key milestone results (from Arkan's docs, dev sizes):
- DAgger beats BC: closed-loop tracking error 3.92e-2 → 2.00e-2 (−48.9%, CI
  excludes zero).
- GAIL barely beats BC (−3.7%), does **not** beat DAgger — reproduces the known
  "exact oracle beats adversarial" result.
- PPO: **pure PPO wins** (+3.20% vs expert −9.78%) — this is decision #1.

## 9. Immediate TODO list

1. **Resolve §5** — run diffusion at intended full scale on a known machine
   before the report claims it fails. Highest priority; blocks the update.
2. **Correct the report** (`docs/progress_update_2026-09.tex`): decision-#2
   estimate to the 15-core M5 reality; soften/repair decision #4 per §5; add the
   compute-resource ask (bwUniCluster vs cloud) to the next-steps section.
3. **Send the update to Ardalan by 30 Sep 2026.**
4. **Implement the path-parallel harness** (§6) — machine-agnostic, `--workers`.
5. **Merge the tip into `main`** (or PR) so real code is on the default branch.
6. Optional cleanup Arkan flagged: delete the ~700-line legacy demonstration path
   (`src/utils/data.py`, `scripts/run_expert.py`, `generate_test_dataset.py`,
   `RollingMarkowitzExpert`) once nothing external depends on the old `.npz`
   format.
7. Later (Weeks 38–39): implement Diebold–Mariano and Giacomini–White tests
   (deferred per §5.4).
