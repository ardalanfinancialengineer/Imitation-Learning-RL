"""Demonstration dataset for behavioural cloning and DAgger-style training.

The real implementation lives in :mod:`src.dataset.demonstrations`, built in the
Weeks 14-16 milestone. It assembles Equation (6) states lazily from stored
paths rather than holding pre-materialised feature matrices, which is what
makes the final-scale dataset tractable.

This module re-exports it so imitation-learning code has a natural import site.
"""

from __future__ import annotations

from ..dataset.demonstrations import DemonstrationDataset, PathEntry

__all__ = ["DemonstrationDataset", "PathEntry"]
