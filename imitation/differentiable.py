"""Differentiable closed-loop rollout, for gradient-based occupancy matching.

Textbook GAIL optimises the policy with a policy-gradient method because the
environment is a black box: you can sample from it but not differentiate
through it. **Here it is not a black box.**

At decision ``t`` the Equation (6) state is

    s_t = [ r_{t-L+1:t}, mu_hat, vech(Sigma_hat), p_hat_t, w_{t-1}, c, gamma ]

and every block except ``w_{t-1}`` is exogenous market data, fixed by the
simulated path and independent of who is trading. The one action-dependent
block is the previous allocation, and ``w_{t-1}`` is exactly the previous
action. The transition is therefore the identity map from action to the next
state's anchor block, and the policy — including the Equation (8) projection,
which Weeks 17-21 built with exact gradients — is differentiable. So the whole
closed-loop rollout is a differentiable recurrence and ``dW1/dtheta`` can be
obtained by backpropagation rather than estimated with the score-function trick.

That removes the dominant source of GAIL's instability. It is a different
*estimator* of the same Equation (9) objective, not a different objective.

Two practical consequences shape the implementation.

**States are spliced, not rebuilt.** Because the anchor block is contiguous in
the state vector, a step's state is ``cat(left_i, w_prev, right_i)`` where the
two halves are precomputed once per path with no gradient. Reassembling the
state through the NumPy assembler every step would both cost more and break the
graph.

**Backpropagation is truncated.** Carrying the graph across a 600-step path
would be expensive and badly conditioned. The rollout runs in segments; at each
segment boundary the anchor is detached, so credit is assigned over a window
rather than the whole path. The window is a configuration value, and the
gradient check in the acceptance suite is run against finite differences over
one segment so the truncation cannot silently be wrong.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..dataset.demonstrations import DemonstrationDataset
from ..policy.base import AllocationPolicy


@dataclass(frozen=True)
class RolloutTape:
    """Precomputed, gradient-free state halves for one set of aligned paths.

    Paths must share their decision indices, which holds whenever they were
    generated with the same periods and warmup. That lets a rollout step every
    path at once, so the policy is called on a batch rather than per path.
    """

    left: torch.Tensor
    right: torch.Tensor
    expert_actions: torch.Tensor
    expert_anchors: torch.Tensor
    realised_returns: torch.Tensor
    initial_anchor: torch.Tensor
    path_ids: tuple[int, ...]
    anchor_slice: slice
    #: ``Sigma_hat_{t+1|t}`` per step and path, needed for the Equation (11)
    #: risk term. Built only when asked for, because it is the one quantity
    #: here that grows quadratically in the asset count.
    forecast_covariances: torch.Tensor | None = None

    @property
    def n_steps(self) -> int:
        return int(self.left.shape[0])

    @property
    def n_paths(self) -> int:
        return int(self.left.shape[1])

    @property
    def n_assets(self) -> int:
        return int(self.expert_actions.shape[-1])

    @property
    def state_dimension(self) -> int:
        return int(self.left.shape[-1] + self.n_assets + self.right.shape[-1])

    def state(self, step: int, anchor: torch.Tensor) -> torch.Tensor:
        """Assemble the batch of states at ``step`` for a differentiable anchor."""
        return torch.cat([self.left[step], anchor, self.right[step]], dim=-1)

    def expert_state(self, step: int) -> torch.Tensor:
        """Assemble the batch of states the *expert* occupied at ``step``."""
        return self.state(step, self.expert_anchors[step])

    def gather(
        self,
        steps: np.ndarray,
        columns: np.ndarray,
        anchors: torch.Tensor,
    ) -> torch.Tensor:
        """Assemble states at scattered ``(step, path)`` positions.

        Used for critic minibatches, which sample positions rather than sweeping
        the tape in order.
        """
        return torch.cat(
            [self.left[steps, columns], anchors, self.right[steps, columns]], dim=-1
        )

    def gather_expert(self, steps: np.ndarray, columns: np.ndarray) -> torch.Tensor:
        """Assemble the expert's states at scattered positions."""
        return self.gather(steps, columns, self.expert_anchors[steps, columns])


def build_tape(
    dataset: DemonstrationDataset,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    max_steps: int | None = None,
    with_covariances: bool = False,
) -> RolloutTape:
    """Precompute the gradient-free parts of every state in ``dataset``.

    ``with_covariances`` additionally stores ``Sigma_hat_{t+1|t}``, which the
    Equation (11) reward needs and nothing else does.
    """
    entries = dataset.entries
    if not entries:
        raise ValueError("dataset contains no paths.")

    lengths = {int(entry.indices.size) for entry in entries}
    if len(lengths) != 1:
        raise ValueError(
            "differentiable rollout requires paths with equal decision counts; "
            f"got {sorted(lengths)}."
        )
    reference = entries[0].indices
    for entry in entries[1:]:
        if not np.array_equal(entry.indices, reference):
            raise ValueError(
                "differentiable rollout requires paths aligned on the same "
                "decision indices."
            )

    n_steps = int(reference.size) if max_steps is None else min(int(reference.size), max_steps)
    anchor_slice = dataset.assembler.block_slices().get("previous_weights")
    if anchor_slice is None:
        raise ValueError(
            "the state must include the previous-weights block; without it the "
            "rollout carries no action dependence and GAIL has nothing to match."
        )

    states = np.empty((n_steps, len(entries), dataset.state_dimension))
    expert_actions = np.empty((n_steps, len(entries), dataset.n_assets))
    expert_anchors = np.empty_like(expert_actions)
    realised = np.empty((n_steps, len(entries), dataset.n_assets))
    covariances = (
        np.empty((n_steps, len(entries), dataset.n_assets, dataset.n_assets))
        if with_covariances
        else None
    )

    for column, entry in enumerate(entries):
        path = entry.data
        for step in range(n_steps):
            sample_index = int(entry.indices[step])
            states[step, column] = dataset.assemble_state(entry, sample_index)
            expert_actions[step, column] = path.expert_weights[sample_index]
            expert_anchors[step, column] = path.previous_weights[sample_index]
            realised[step, column] = path.returns[int(path.decision_times[sample_index])]
            if covariances is not None:
                forecast = entry.forecaster.forecast(
                    int(path.decision_times[sample_index]), entry.horizon
                )
                covariances[step, column] = forecast.covariances[0]

    tensor = lambda array: torch.as_tensor(array, dtype=dtype, device=device)  # noqa: E731
    return RolloutTape(
        left=tensor(states[..., : anchor_slice.start]),
        right=tensor(states[..., anchor_slice.stop :]),
        expert_actions=tensor(expert_actions),
        expert_anchors=tensor(expert_anchors),
        realised_returns=tensor(realised),
        initial_anchor=tensor(expert_anchors[0]),
        path_ids=tuple(entry.data.path_id for entry in entries),
        anchor_slice=anchor_slice,
        forecast_covariances=(
            None if covariances is None else tensor(covariances)
        ),
    )


@dataclass
class SegmentRollout:
    """One truncated segment of a differentiable rollout."""

    states: torch.Tensor
    actions: torch.Tensor
    anchors: torch.Tensor
    final_anchor: torch.Tensor

    def flat(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(states, actions)`` flattened over steps and paths."""
        return (
            self.states.reshape(-1, self.states.shape[-1]),
            self.actions.reshape(-1, self.actions.shape[-1]),
        )

    @property
    def turnover(self) -> torch.Tensor:
        return (self.actions - self.anchors).abs().sum(dim=-1)


def rollout_segment(
    policy: AllocationPolicy,
    tape: RolloutTape,
    anchor: torch.Tensor,
    start: int,
    length: int,
) -> SegmentRollout:
    """Roll the policy forward ``length`` steps, keeping the graph.

    The returned ``final_anchor`` still carries gradient; the caller detaches it
    when it wants the truncation boundary.
    """
    stop = min(start + length, tape.n_steps)
    if stop <= start:
        raise ValueError("segment must contain at least one step.")

    states, actions, anchors = [], [], []
    current = anchor
    for step in range(start, stop):
        state = tape.state(step, current)
        action = policy.allocate(state)
        states.append(state)
        actions.append(action)
        anchors.append(current)
        current = action

    return SegmentRollout(
        states=torch.stack(states),
        actions=torch.stack(actions),
        anchors=torch.stack(anchors),
        final_anchor=current,
    )


@torch.no_grad()
def rollout_tape(
    policy: AllocationPolicy,
    tape: RolloutTape,
    *,
    anchor: torch.Tensor | None = None,
) -> SegmentRollout:
    """Roll the policy over the whole tape without building a graph.

    Used to collect the policy occupancy the critic trains against, and to score
    a finished policy. Detached because neither needs gradients and the full
    path would be expensive to keep.
    """
    was_training = policy.training
    policy.eval()
    try:
        start = tape.initial_anchor if anchor is None else anchor
        segment = rollout_segment(policy, tape, start, 0, tape.n_steps)
    finally:
        policy.train(was_training)
    return segment


def realised_net_returns(
    tape: RolloutTape, segment: SegmentRollout, transaction_cost: float, start: int = 0
) -> torch.Tensor:
    """Return ``w'r_{t+1} - c||w_t - w_{t-1}||_1`` for a rolled segment."""
    stop = start + segment.actions.shape[0]
    gross = (segment.actions * tape.realised_returns[start:stop]).sum(dim=-1)
    return gross - transaction_cost * segment.turnover
