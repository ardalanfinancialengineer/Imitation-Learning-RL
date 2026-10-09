"""Device selection that is safe across CUDA, Apple Silicon, and CPU-only hosts.

The trainers accept a ``device`` string in their config. In addition to the
literal torch device names (``"cpu"``, ``"cuda"``, ``"cuda:0"``, ``"mps"``),
they accept the sentinel ``"auto"``, resolved here.

Two deliberate choices:

* **``"auto"`` never selects MPS.** The reinforcement-learning tape, the
  environment, and several diagnostics run in ``float64`` on purpose (see
  ``src/reinforcement/diagnostics.py`` and ``environment.py``), and the MPS
  backend does not support double precision for many ops. Auto-selecting MPS
  would therefore *break* the RL stages rather than merely slow them, and it
  would also change numbers that the acceptance checks pin. Auto resolves to
  CUDA when present, otherwise CPU. MPS remains available, but only when a
  caller asks for it explicitly and knowingly.
* **An unavailable explicit request falls back with a warning, not a crash.**
  Asking for ``"cuda"`` on a machine without it, or ``"mps"`` on a non-Apple
  build, logs a warning and uses CPU, so a config written for one machine still
  runs on another.
"""

from __future__ import annotations

import warnings

import torch

__all__ = ["resolve_device"]


def _mps_available() -> bool:
    backend = getattr(torch.backends, "mps", None)
    return bool(backend is not None and backend.is_available())


def resolve_device(requested: str | torch.device = "cpu") -> torch.device:
    """Turn a requested device string into a concrete, available ``torch.device``.

    Args:
        requested: ``"cpu"``, ``"cuda"``/``"cuda:N"``, ``"mps"``, or ``"auto"``.
            ``"auto"`` picks CUDA if available, otherwise CPU — never MPS, see
            the module docstring.

    Returns:
        A ``torch.device`` that is present on this machine. Unavailable explicit
        requests warn and fall back to CPU.
    """
    if isinstance(requested, torch.device):
        return requested

    name = str(requested).strip().lower()

    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    if name.startswith("cuda"):
        if torch.cuda.is_available():
            return torch.device(name)
        warnings.warn(
            f"device={requested!r} requested but CUDA is not available; "
            "falling back to CPU.",
            RuntimeWarning,
            stacklevel=2,
        )
        return torch.device("cpu")

    if name == "mps":
        if _mps_available():
            warnings.warn(
                "device='mps' selected explicitly. The RL tape, environment, and "
                "several diagnostics use float64, which MPS does not fully "
                "support; those stages may error or diverge. Use 'cpu' or 'cuda' "
                "for the reinforcement-learning demos.",
                RuntimeWarning,
                stacklevel=2,
            )
            return torch.device("mps")
        warnings.warn(
            "device='mps' requested but the MPS backend is not available; "
            "falling back to CPU.",
            RuntimeWarning,
            stacklevel=2,
        )
        return torch.device("cpu")

    # Any other literal (e.g. "cpu") is passed through to torch, which raises a
    # clear error for genuinely invalid names.
    return torch.device(name)
