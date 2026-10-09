"""Tests for :mod:`src.utils.observability`.

The behaviour that matters and must not regress:

* Verbosity maps to the root log level (INFO by default, DEBUG under ``verbose``).
* ``configure_logging`` is idempotent — re-calling it replaces its handler rather
  than stacking a second one that would double every line.
* The handler flushes after each record. This is the fix for the zero-byte-log
  bug, so it is a correctness property: a record must be readable from the stream
  *without* an explicit flush by the caller.
* ``sample_memory`` returns the fixed field set, all floats, with sane ranges.
* ``MemoryTracker`` reports the peak RSS it has seen.
"""

from __future__ import annotations

import io
import logging

from src.utils.observability import (
    MemoryTracker,
    configure_logging,
    log_memory,
    sample_memory,
)

_MEMORY_FIELDS = {
    "rss_mib",
    "available_mib",
    "system_percent",
    "swap_used_mib",
    "swap_percent",
}


def _our_handlers(root: logging.Logger) -> list[logging.Handler]:
    return [h for h in root.handlers if getattr(h, "_observability_handler", False)]


def test_verbose_maps_to_debug_else_info():
    configure_logging(False)
    assert logging.getLogger().level == logging.INFO
    configure_logging(True)
    assert logging.getLogger().level == logging.DEBUG


def test_configure_logging_is_idempotent():
    configure_logging(False)
    configure_logging(False)
    configure_logging(True)
    assert len(_our_handlers(logging.getLogger())) == 1


def test_handler_flushes_without_explicit_flush():
    stream = io.StringIO()
    configure_logging(False, stream=stream)
    logging.getLogger("test.flush").info("hello-observability")
    # No explicit flush here: the handler must have flushed on emit.
    assert "hello-observability" in stream.getvalue()


def test_sample_memory_dict_shape_and_ranges():
    sample = sample_memory()
    assert set(sample) == _MEMORY_FIELDS
    assert all(isinstance(v, float) for v in sample.values())
    assert sample["rss_mib"] > 0.0
    assert 0.0 <= sample["system_percent"] <= 100.0
    assert 0.0 <= sample["swap_percent"] <= 100.0


def test_log_memory_returns_sample_and_logs(caplog):
    logger = logging.getLogger("test.mem")
    with caplog.at_level(logging.INFO, logger="test.mem"):
        sample = log_memory(logger, "phase-a")
    assert set(sample) == _MEMORY_FIELDS
    assert any("mem[phase-a]" in message for message in caplog.messages)


def test_memory_tracker_peak_is_monotonic(caplog):
    logger = logging.getLogger("test.tracker")
    tracker = MemoryTracker(logger, "training")
    with caplog.at_level(logging.INFO, logger="test.tracker"):
        first = tracker.sample("step-1")
        second = tracker.sample("step-2")
        tracker.summary()
    assert tracker.peak_rss == max(first["rss_mib"], second["rss_mib"])
    assert any("peak rss" in message for message in caplog.messages)
