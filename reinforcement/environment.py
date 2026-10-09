"""Gymnasium portfolio environment over a stored expert path.

The environment replays one simulated market realisation. At step ``i`` the
agent observes the Equation (6) state for decision index ``t``, submits weights
``w_t``, and receives the exposé's reward (Equation 11)

    rho_t = R_tilde_{t+1}(w_t)
            - (gamma/2) w_t' Sigma_hat_{t+1|t} w_t
            - lambda_E ||w_t - w*_t||_2^2

where the realised net return charges the transaction cost exactly once,

    R_tilde_{t+1}(w_t) = w_t' r_{t+1} - c ||w_t - w_{t-1}||_1.

Turnover is reported in ``info`` but never added to the reward again, because
the cost term already prices it.

``lambda_E`` defaults to zero, so the environment is a faithful
market-plus-frictions simulator out of the box. The expert-deviation penalty is
switched on for the PPO fine-tuning stage.

The environment owns no look-ahead: observations come from the same causal
state assembler used by the demonstration dataset, and ``r_{t+1}`` is revealed
only after the action for ``t`` has been submitted.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..dataset.demonstrations import DemonstrationDataset
from ..dataset.paths import PathData
from ..dataset.state import StateSpec
from ..expert.constraints import PortfolioConstraints


class PortfolioEnv(gym.Env):
    """Multi-period portfolio environment with the exposé's reward."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        path: PathData,
        *,
        spec: StateSpec | None = None,
        decision_slice: slice | None = None,
        expert_deviation_penalty: float = 0.0,
        constraints: PortfolioConstraints | None = None,
    ) -> None:
        super().__init__()
        if not np.isfinite(expert_deviation_penalty) or expert_deviation_penalty < 0.0:
            raise ValueError("expert_deviation_penalty must be finite and nonnegative.")

        self.path = path
        # The dataset already owns causal state assembly and the restored
        # forecaster; reusing it guarantees the policy sees exactly the same
        # observation during RL as during behavioural cloning.
        self._dataset = DemonstrationDataset(
            [path], spec=spec, decision_slice=decision_slice
        )
        self._entry = self._dataset.entries[0]
        self._indices = self._entry.indices
        # Not `self.spec`: gym.Env reserves that name for its EnvSpec, and
        # shadowing it breaks gym.make and the vectorised wrappers SB3 uses.
        self.state_spec = self._dataset.spec

        expert_settings = path.metadata["expert"]
        self.transaction_cost = float(expert_settings["transaction_cost"])
        self.risk_aversion = float(expert_settings["risk_aversion"])
        self.expert_deviation_penalty = float(expert_deviation_penalty)
        self.constraints = (
            constraints
            if constraints is not None
            else _constraints_from(expert_settings["constraints"])
        )

        self.n_assets = path.n_assets
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._dataset.state_dimension,),
            dtype=np.float64,
        )
        self.action_space = spaces.Box(
            low=0.0 if self.constraints.long_only else -np.inf,
            high=np.inf,
            shape=(self.n_assets,),
            dtype=np.float64,
        )

        self._step = 0
        self._weights = self.constraints.default_weights(self.n_assets)

    @property
    def observation_dimension(self) -> int:
        return int(self._dataset.state_dimension)

    @property
    def n_steps(self) -> int:
        return int(self._indices.size)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset to the first decision of the path."""
        super().reset(seed=seed)
        self._step = 0
        initial = (options or {}).get("initial_weights")
        self._weights = (
            self.constraints.default_weights(self.n_assets)
            if initial is None
            else self.constraints.project(np.asarray(initial, dtype=float))
        )
        return self._observation(), self._info()

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Apply weights for the current decision date and advance one period."""
        if self._step >= self.n_steps:
            raise RuntimeError("step() called after the episode terminated.")

        weights = self.constraints.project(np.asarray(action, dtype=float))
        sample_index = int(self._indices[self._step])
        t = int(self.path.decision_times[sample_index])

        turnover = float(np.abs(weights - self._weights).sum())
        gross_return = float(weights @ self.path.returns[t])
        net_return = gross_return - self.transaction_cost * turnover

        forecast = self._entry.forecaster.forecast(t, self._entry.horizon)
        risk_penalty = 0.5 * self.risk_aversion * float(
            weights @ forecast.covariances[0] @ weights
        )
        expert_weights = self.path.expert_weights[sample_index]
        deviation = float(np.square(weights - expert_weights).sum())
        reward = net_return - risk_penalty - self.expert_deviation_penalty * deviation

        self._weights = weights
        self._step += 1
        terminated = self._step >= self.n_steps

        info = self._info() | {
            "decision_time": t,
            "sample_index": sample_index,
            "gross_return": gross_return,
            "net_return": net_return,
            "turnover": turnover,
            "risk_penalty": risk_penalty,
            "expert_deviation": deviation,
            "expert_weights": expert_weights.copy(),
        }
        observation = (
            np.zeros(self.observation_dimension) if terminated else self._observation()
        )
        return observation, reward, terminated, False, info

    def expert_action(self) -> np.ndarray:
        """Return the expert's implemented weights for the current step.

        Used to replay the expert through the environment and to answer DAgger
        queries on states the learner visited.
        """
        if self._step >= self.n_steps:
            raise RuntimeError("no expert action is available after termination.")
        return self.path.expert_weights[int(self._indices[self._step])].copy()

    def _observation(self) -> np.ndarray:
        return self._dataset.assemble_state(self._entry, int(self._indices[self._step]))

    def _info(self) -> dict[str, Any]:
        return {"step": self._step, "previous_weights": self._weights.copy()}

    def describe(self) -> dict[str, Any]:
        """Return a JSON-serialisable description for experiment metadata."""
        return {
            "path_id": self.path.path_id,
            "n_steps": self.n_steps,
            "n_assets": self.n_assets,
            "observation_dimension": self.observation_dimension,
            "transaction_cost": self.transaction_cost,
            "risk_aversion": self.risk_aversion,
            "expert_deviation_penalty": self.expert_deviation_penalty,
            "constraints": self.constraints.to_dict(),
            "reward": (
                "w'r_{t+1} - c||w_t - w_{t-1}||_1 "
                "- (gamma/2) w'Sigma_hat w - lambda_E ||w_t - w*_t||^2"
            ),
        }


def _constraints_from(settings: dict[str, Any]) -> PortfolioConstraints:
    return PortfolioConstraints(
        long_only=bool(settings["long_only"]),
        min_weight=float(settings["min_weight"]),
        max_weight=float(settings["max_weight"]),
        leverage_limit=(
            None if settings.get("leverage_limit") is None else float(settings["leverage_limit"])
        ),
    )
