"""Run observability: structured logging and memory instrumentation.

The long mid-scale runs used to be black boxes. Progress was printed with
ad-hoc ``print(..., flush=True)`` calls that, under a redirected pipe, stayed
buffered and produced a *zero-byte log until the process exited* — so a run that
was OOM-killed after ten hours left no record of how far it got or how much
memory it was using. This module fixes both halves of that:

* **Logging that reaches the file while the run is alive.** ``configure_logging``
  installs a single handler that *flushes after every record*, so ``tail -f`` on a
  redirected log shows progress in real time regardless of the OS pipe buffering.
  Only entry-point scripts call ``configure_logging``; library modules merely
  ``logging.getLogger(__name__)`` and emit, so importing this module has no side
  effects (matching how ``device.py`` and ``parallel.py`` avoid global state).
* **Memory sampling that answers the two questions an OOM raises.** Is *this
  process* growing without bound (a leak), or is its steady-state footprint simply
  too large for the machine? ``sample_memory`` reports process RSS (the leak
  signal) alongside system-available memory and swap (the "about to be killed"
  signal), and :class:`MemoryTracker` keeps the peak RSS across a phase for an
  end-of-phase summary line.

Memory is always sampled in the *parent* process. The path-generation workers
(:mod:`src.utils.parallel`) run under a spawn pool with BLAS pinned to one
thread; sampling inside them would measure the wrong process and pull psutil into
the pinned worker init, so callers surface worker progress via the parent-side
``progress_callback`` instead.
"""

from __future__ import annotations

import logging
import sys
from typing import TextIO

import psutil

__all__ = [
    "configure_logging",
    "get_logger",
    "sample_memory",
    "log_memory",
    "MemoryTracker",
]

# A single process handle, reused across samples. Under a spawn worker this
# module is re-imported in the child, so each process binds its own handle to
# its own PID — but by design we only ever sample in the parent (see docstring).
_PROCESS = psutil.Process()

# Marks the handler we install so re-calling configure_logging replaces it
# instead of stacking a second one.
_HANDLER_TAG = "_observability_handler"

_MIB = float(2**20)


class _FlushingStreamHandler(logging.StreamHandler):
    """A ``StreamHandler`` that flushes after every record.

    This is the load-bearing fix for the zero-byte-log bug: a redirected pipe is
    block-buffered, so without a per-record flush nothing reaches the file until
    the buffer fills or the process exits. Flushing here is self-contained — it
    does not depend on ``PYTHONUNBUFFERED`` being set in the environment.
    """

    def emit(self, record: logging.LogRecord) -> None:
        super().emit(record)
        self.flush()


def configure_logging(verbose: bool = False, *, stream: TextIO = sys.stderr) -> None:
    """Install one flushing handler on the root logger. Idempotent.

    Args:
        verbose: ``True`` sets the root level to ``DEBUG``; otherwise ``INFO``.
        stream: Where records are written (default stderr, the conventional home
            for diagnostics). The per-record flush makes the choice moot for
            buffering — either stream reaches disk immediately.

    Calling this again replaces the handler it previously installed rather than
    adding a second one, so a script that (re)configures logging never doubles
    every line.
    """
    level = logging.DEBUG if verbose else logging.INFO
    root = logging.getLogger()

    for existing in list(root.handlers):
        if getattr(existing, _HANDLER_TAG, False):
            root.removeHandler(existing)

    handler = _FlushingStreamHandler(stream)
    setattr(handler, _HANDLER_TAG, True)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)-5s %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    root.addHandler(handler)
    root.setLevel(level)


def get_logger(name: str | None = None) -> logging.Logger:
    """Thin wrapper over :func:`logging.getLogger` for a consistent import site."""
    return logging.getLogger(name)


def sample_memory() -> dict[str, float]:
    """Return one point-in-time memory snapshot, all values in MiB (floats).

    Fields (the set is fixed, so callers and tests can rely on the shape):

    * ``rss_mib`` — this process's resident set size; the leak signal.
    * ``available_mib`` — system memory available before swapping/killing.
    * ``system_percent`` — percent of system RAM in use.
    * ``swap_used_mib`` — swap in use; rises during the thrash before an OOM kill.
    * ``swap_percent`` — percent of swap in use.
    """
    rss = _PROCESS.memory_info().rss
    virtual = psutil.virtual_memory()
    swap = psutil.swap_memory()
    return {
        "rss_mib": rss / _MIB,
        "available_mib": virtual.available / _MIB,
        "system_percent": float(virtual.percent),
        "swap_used_mib": swap.used / _MIB,
        "swap_percent": float(swap.percent),
    }


def log_memory(
    logger: logging.Logger, tag: str, *, level: int = logging.INFO
) -> dict[str, float]:
    """Sample memory, log a one-line summary tagged ``tag``, and return the sample."""
    sample = sample_memory()
    logger.log(
        level,
        "mem[%s] rss=%.0fMiB avail=%.0fMiB sys=%.0f%% swap=%.0fMiB",
        tag,
        sample["rss_mib"],
        sample["available_mib"],
        sample["system_percent"],
        sample["swap_used_mib"],
    )
    return sample


class MemoryTracker:
    """Sample memory across a phase and report its peak RSS at the end.

    A phase is a coarse span of a run — path generation, or training one policy.
    Call :meth:`sample` at each fine-grained boundary (per epoch, per completed
    path) and :meth:`summary` once when the phase ends; the summary line names the
    highest RSS seen, which is what tells you whether the phase alone would fit in
    memory at a larger scale.
    """

    def __init__(self, logger: logging.Logger, phase: str) -> None:
        self.logger = logger
        self.phase = phase
        self.peak_rss = 0.0

    def sample(self, tag: str, *, level: int = logging.INFO) -> dict[str, float]:
        sample = log_memory(self.logger, f"{self.phase}:{tag}", level=level)
        self.peak_rss = max(self.peak_rss, sample["rss_mib"])
        return sample

    def summary(self, *, level: int = logging.INFO) -> None:
        self.logger.log(
            level, "mem[%s] peak rss=%.0fMiB", self.phase, self.peak_rss
        )
