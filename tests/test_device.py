"""Tests for :func:`src.utils.device.resolve_device`.

The behaviour that matters and must not regress:

* ``"auto"`` resolves to CUDA when present, otherwise CPU, and **never** MPS —
  because the RL tape and several diagnostics use float64, which MPS does not
  support (see ``src/utils/device.py`` and ``src/reinforcement/diagnostics.py``).
* An unavailable explicit request warns and falls back to CPU rather than
  raising, so a config written for one machine still runs on another.
"""

from __future__ import annotations

import warnings

import torch

from src.utils.device import resolve_device


def test_cpu_is_passed_through():
    assert resolve_device("cpu") == torch.device("cpu")


def test_torch_device_instance_is_returned_unchanged():
    dev = torch.device("cpu")
    assert resolve_device(dev) is dev


def test_auto_resolves_to_cuda_or_cpu_never_mps():
    resolved = resolve_device("auto")
    expected = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    assert resolved == expected
    assert resolved.type != "mps"


def test_unavailable_cuda_warns_and_falls_back_to_cpu():
    if torch.cuda.is_available():
        # On a CUDA box the request is honoured; the fallback path is untestable.
        assert resolve_device("cuda") == torch.device("cuda")
        return
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        resolved = resolve_device("cuda")
    assert resolved == torch.device("cpu")
    assert any(issubclass(w.category, RuntimeWarning) for w in caught)


def test_explicit_mps_always_warns():
    """Even when MPS is available, the float64 hazard must be surfaced."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        resolve_device("mps")
    assert any(issubclass(w.category, RuntimeWarning) for w in caught)


def test_case_and_whitespace_insensitive():
    assert resolve_device("  AUTO ") == resolve_device("auto")
