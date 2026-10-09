"""Backwards-compatible import site for the behavioural-cloning agent.

This module used to hold ``BCPolicyNet`` and ``BehaviouralCloningAgent`` as
unimplemented stubs. The Weeks 17-21 milestone replaced them with real code
that lives where the responsibilities actually split: policy architectures in
:mod:`src.policy`, the training loop in :mod:`src.imitation.bc`.

The stubs assumed a feed-forward network with a softmax head. Neither survived
contact with the requirements — a softmax cannot produce the exact zeros or
respect the position cap that Equation (5) imposes, so the head is a Euclidean
projection instead (:mod:`src.policy.simplex`), and the network is a transformer
encoder over the state's Equation (6) blocks.

New code should import from :mod:`src.policy` and :mod:`src.imitation.bc`
directly.
"""

from __future__ import annotations

from ..policy.base import AllocationPolicy
from ..policy.builder import build_policy
from ..policy.deterministic import DeterministicPolicy
from ..policy.diffusion import TransformerDiffusionPolicy
from .bc import BCConfig, BehaviouralCloningTrainer

#: Retained aliases so older call sites keep resolving.
BCPolicyNet = DeterministicPolicy
BehaviouralCloningAgent = BehaviouralCloningTrainer

__all__ = [
    "AllocationPolicy",
    "BCConfig",
    "BCPolicyNet",
    "BehaviouralCloningAgent",
    "BehaviouralCloningTrainer",
    "DeterministicPolicy",
    "TransformerDiffusionPolicy",
    "build_policy",
]
