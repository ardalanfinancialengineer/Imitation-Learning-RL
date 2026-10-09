# Mid-scale verification attempt — what we learned (2026-09-25 → 26)

## TL;DR

We tried to verify open decisions **#3 (state sufficiency)** and **#4 (BC/diffusion
baselines)** at realistic mid-scale (N=50 assets, T=5000, 50 paths, PPO-fine-tuning
harness) instead of the small development sizes the demos ship with. **Both runs ran
for ~10 hours and were then killed by macOS for low memory, producing no final
results.** That failure is itself the headline finding: the current pipeline cannot
sustain a mid-scale run on this machine, for two independent reasons (CPU cost and
memory), long before we get anywhere near the exposé's full 500-path / N=200 study.

## What we ran

| Run | Script | Config | Wall / CPU before kill | Output produced |
|-----|--------|--------|------------------------|-----------------|
| **#3** state sufficiency | `check_state_sufficiency.py` | N=50, T=5000, 50 paths, 2 moment estimators (`regime_mixture`, `rolling`) | ~10.7 h / ~10 h CPU | 50 `regime_mixture` paths only — **never reached config 2**, no `state_sufficiency.json` |
| **#4** BC / diffusion | `run_bc_demo.py` | N=50, T=5000, 50 paths, mlp/transformer/diffusion | ~10 h / ~9.2 h CPU | 50 cached paths (`data/bc_demo/paths/`) — **no `bc_report.json`**, killed mid-training |

Both were launched on the new path-parallel harness (`--workers 8`), which itself
worked correctly — generation fanned across cores and was bit-identical to serial.
The failure is downstream of generation, in the training/evaluation phases.

## What we learned

### 1. Mid-scale is CPU-bound in the forecaster, not the neural net

A macOS `sample` of the #3 driver mid-run showed the active leaf frames were almost
entirely **`libLAPACK` / `libBLAS` (Apple Accelerate)** plus numpy `DOUBLE_*` ufuncs —
**not** torch matmuls (the autograd worker threads were parked idle). The cost is
dominated by the **`regime_mixture` forecaster's linear algebra**: Ledoit–Wolf
covariance estimation + regime-EM refits (`refit_every=250`, `burn_in=300`,
`n_regimes=3`) over T=5000 at N=50 — O(N²·T) work with periodic eigendecompositions,
run per path and during horizon-variation evaluation.

**Evidence of scale:** #3 spent **10 hours of CPU and still had not finished the first
of its two moment configs.** The `rolling` estimator (cheap: one window, no EM, no
eigendecomposition) never got a chance to run.

### 2. Mid-scale is also memory-unsustainable over long runs

Both processes were **OOM-killed by the OS simultaneously** (~2 AM), after the machine
had been swapping hard (~163k pageouts logged). Memory recovered to 55% free the
instant they died. On 48 GB, running these two N=50 jobs alongside a normal desktop
(VS Code, browser, etc.) exhausted memory over ~10 h.

**Open question we have NOT answered:** whether the footprint *grew over time* (a real
leak) or was simply *constant-but-too-large*. This must be resolved before any rerun —
it changes whether the fix is "close other apps" or "fix the code."

### 3. The two blockers compound

These are **independent** failure modes. Even if we fixed the memory issue, #3 would
still take 10 h+ of CPU; even if we sped up the forecaster, a long run might still leak.
The full study (500 paths, up to N=200) needs *both* resolved — plus the previously
identified `build_tape` dense-tensor OOM — before it is feasible.

## Incidental findings

- **`run_bc_demo.py` `--output-dir` does not redirect the path cache.** The cache dir
  is taken from the config file (`configs/bc_demo.yaml` → `data/bc_demo/paths`), which
  overrides `--output-dir`. So #4's paths landed in the shared demo cache, clobbering
  the shipped small-size dev cache. **No scientific impact** (`_matches` correctly
  rejected the stale N=8 paths and regenerated at N=50 — verified `returns.shape
  (5000, 50)`), but it's a footgun worth fixing so `--output-dir` also moves the cache.
- **Buffered stdout.** Both scripts' progress prints (`flush=True`) never reached the
  redirected task logs — they stayed 0 bytes until exit, so live progress had to be
  inferred from filesystem mtimes and CPU counters. A `PYTHONUNBUFFERED=1` / line-
  buffering fix would make long runs observable.
- **`caffeinate` + closed lid.** `caffeinate -s` prevents idle/system sleep but does
  **not** override clamshell (closed-lid) sleep without an external display. Overnight
  runs need the lid open (or a display attached).

## Recommended next step (not yet done — awaiting decision)

Do **not** simply relaunch the same 10 h job. Instead run a **short, instrumented
diagnostic** first:

1. Small N / few paths (or reduced `refit_every`) so it finishes in ~30–60 min, to get
   the actual #3/#4 numbers cheaply.
2. Log **RSS over time** to distinguish memory *leak* from steady-state *footprint*.
3. Then decide whether the forecaster-refit path needs caching / precompute / vectorising.

## What survives on disk

- `data/state_sufficiency_midscale_2026-09-25/paths/regime_mixture/` — 50 paths (N=50, T=5000)
- `data/bc_demo/paths/` — 50 paths (N=50, T=5000), #4's regenerated cache
- No result JSONs from either run.

## For the Ardalan status update (30 Sep)

Report plainly: mid-scale verification was **attempted** at N=50 and the runs were
**OOM-killed after 10 hours with no output**. This is a decisive, concrete
reinforcement of blocker #2 (scale infeasibility) — the pipeline needs specific,
identified fixes (forecaster-refit cost, memory footprint/leak, `build_tape` OOM)
before the final study is runnable. The mid-scale attempt did *not* produce the
tracking-error or diffusion-baseline numbers we hoped to verify.
