"""Path-parallel execution across CPU cores, bit-identical to a serial run.

The Monte Carlo paths of the final study are independent realisations — each
carries its own seed (``base_seed + path_id``) and reads no global RNG (see
``src/simulation/returns.py``) — so generating them is embarrassingly parallel.
This module maps a pure function over a sequence of argument objects, optionally
across worker processes.

Two properties are load-bearing and deliberately preserved:

* **``workers <= 1`` is a plain in-process loop.** No process pool, no pickling
  round-trip — byte-for-byte the behaviour that existed before this module. The
  parallel path is strictly opt-in, so no existing run changes unless a caller
  asks for it.
* **Workers pin BLAS to a single thread.** This is a *correctness* measure, not
  only a performance one: multi-threaded BLAS reorders floating-point reductions
  and would perturb the last bits of a ``float64`` result, breaking the
  bit-identity the acceptance checks depend on. It is also what stops the
  oversubscription that makes naive parallelism slower than serial (each worker's
  BLAS otherwise spawns its own threads). See ``docs/open_decisions.md`` #2.

The pool uses the ``spawn`` start method: it is the macOS default, and it avoids
the fork+BLAS deadlock that ``fork`` risks when the parent has already touched a
threaded BLAS library.
"""

from __future__ import annotations

import functools
import multiprocessing
import os
from collections.abc import Callable, Sequence
from typing import TypeVar

__all__ = ["parallel_map", "resolve_workers"]

T = TypeVar("T")
R = TypeVar("R")

_BLAS_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    # macOS ships Accelerate (vecLib) as the default BLAS, so this one is the
    # relevant cap on Apple Silicon.
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def resolve_workers(workers: int) -> int:
    """Turn a requested worker count into a concrete positive integer.

    ``0`` and negative values are a convenience sentinel for "use every core",
    resolving to :func:`os.cpu_count`. Everything else passes through unchanged,
    so ``1`` stays serial.
    """
    if workers <= 0:
        return os.cpu_count() or 1
    return workers


def _init_worker() -> None:
    """Pin BLAS to one thread inside a freshly spawned worker.

    Runs once per worker before any mapped call. The environment variables cap
    the BLAS libraries' own thread pools; ``torch.set_num_threads(1)`` is the
    authoritative clamp for anything routed through torch, and always takes
    effect even if torch was imported before this ran.
    """
    for var in _BLAS_THREAD_ENV_VARS:
        os.environ[var] = "1"
    import torch

    torch.set_num_threads(1)


def _call_indexed(indexed_arg: tuple[int, T], fn: Callable[[T], R]) -> tuple[int, R]:
    index, arg = indexed_arg
    return index, fn(arg)


def parallel_map(
    fn: Callable[[T], R],
    args: Sequence[T],
    *,
    workers: int = 1,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[R]:
    """Map ``fn`` over ``args``, returning results in ``args`` order.

    Args:
        fn: A picklable, side-effect-free callable applied to each element of
            ``args``. Must be importable by reference (a module-level function),
            not a lambda or closure, so it survives the ``spawn`` pickling.
        args: The inputs. Each element must be picklable.
        workers: ``<= 1`` runs ``fn`` serially in this process (the default and
            the exact pre-existing behaviour). ``> 1`` fans the work across that
            many spawned worker processes. ``0`` or negative means "every core"
            (see :func:`resolve_workers`).
        progress_callback: If given, called ``progress_callback(done, total)`` in
            *this* (parent) process each time an item completes, where ``done``
            counts from 1 to ``len(args)``. It never runs inside a worker — under
            ``spawn`` a worker cannot call back into the parent — so it is safe to
            log or sample process memory from it. Completion order is
            nondeterministic under ``workers > 1`` (``done`` still climbs
            monotonically); the *returned* list is always in input order.

    Returns:
        ``[fn(a) for a in args]`` — always in input order, regardless of which
        worker finishes first, so downstream ordering is deterministic.
    """
    total = len(args)
    resolved = resolve_workers(workers)
    if resolved <= 1 or total <= 1:
        results: list[R] = []
        for done, arg in enumerate(args, start=1):
            results.append(fn(arg))
            if progress_callback is not None:
                progress_callback(done, total)
        return results

    context = multiprocessing.get_context("spawn")
    # Never spawn more workers than there is work for them to do.
    pool_size = min(resolved, total)
    indexed = list(enumerate(args))
    with context.Pool(processes=pool_size, initializer=_init_worker) as pool:
        # imap_unordered yields each (index, result) to the parent as soon as it
        # is ready, so the parent can tick progress mid-run; starmap could not,
        # because it only returns once every item is done. Results carry their
        # index, so re-sorting below makes the returned order deterministic
        # regardless of which worker finished first.
        completed: list[tuple[int, R]] = []
        for pair in pool.imap_unordered(
            functools.partial(_call_indexed, fn=fn), indexed
        ):
            completed.append(pair)
            if progress_callback is not None:
                progress_callback(len(completed), total)
    completed.sort(key=lambda pair: pair[0])
    return [result for _, result in completed]
