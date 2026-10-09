"""Data loading, normalisation, and feature engineering helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Demonstrations:
    """Validated arrays produced by the expert data-generation scripts."""

    observations: np.ndarray
    actions: np.ndarray
    returns: np.ndarray | None = None
    regimes: np.ndarray | None = None
    decision_times: np.ndarray | None = None

    @property
    def n_samples(self) -> int:
        return int(self.observations.shape[0])

    @property
    def window(self) -> int:
        return int(self.observations.shape[1])

    @property
    def n_assets(self) -> int:
        return int(self.observations.shape[2])


@dataclass(frozen=True)
class ReturnNormalizer:
    """Return normalization statistics fitted on a training slice only."""

    method: str
    location: np.ndarray
    scale: np.ndarray
    columns: tuple[str, ...] | None = None

    def transform(self, returns: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
        values, columns, index = _coerce_returns(returns)
        if values.shape[-1] != self.location.shape[0]:
            raise ValueError(
                "returns last dimension does not match fitted normalizer: "
                f"{values.shape[-1]} != {self.location.shape[0]}."
            )
        if self.columns is not None and columns is not None and tuple(columns) != self.columns:
            raise ValueError("DataFrame columns do not match fitted normalizer columns.")

        transformed = (values - self.location) / self.scale
        if isinstance(returns, pd.DataFrame):
            return pd.DataFrame(transformed, columns=columns, index=index)
        return transformed


def build_observation(
    returns_window: np.ndarray,
    include_cov: bool = True,
) -> np.ndarray:
    """Flatten rolling statistics into a 1-D observation vector.

    Concatenates: per-asset means, per-asset variances, and (optionally) the
    upper-triangular elements of the sample correlation matrix.

    Args:
        returns_window: Array of shape (window, n_assets).
        include_cov: Whether to include correlation features.

    Returns:
        1-D feature vector.
    """
    values = np.asarray(returns_window, dtype=float)
    if values.ndim != 2:
        raise ValueError("returns_window must be a 2-D array.")
    if values.shape[0] < 2:
        raise ValueError("returns_window must contain at least two rows.")
    if values.shape[1] == 0:
        raise ValueError("returns_window must contain at least one asset.")
    if not np.all(np.isfinite(values)):
        raise ValueError("returns_window must contain only finite values.")

    means = values.mean(axis=0)
    variances = values.var(axis=0, ddof=1)
    features = [means, variances]

    if include_cov:
        n_assets = values.shape[1]
        corr = np.eye(n_assets, dtype=float)
        std = values.std(axis=0, ddof=1)
        active = std > 1e-12
        if np.count_nonzero(active) > 1:
            active_corr = np.corrcoef(values[:, active], rowvar=False)
            corr[np.ix_(active, active)] = active_corr
        upper_triangle = np.triu_indices(n_assets, k=1)
        features.append(corr[upper_triangle])

    return np.concatenate(features).astype(float)


def validate_demonstrations(
    observations: np.ndarray,
    actions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate the current rolling-window demonstration array format."""
    observations = np.asarray(observations, dtype=float)
    actions = np.asarray(actions, dtype=float)

    if observations.ndim != 3:
        raise ValueError("observations must have shape (n_samples, window, n_assets).")
    if actions.ndim != 2:
        raise ValueError("actions must have shape (n_samples, n_assets).")
    if observations.shape[0] == 0:
        raise ValueError("demonstrations must contain at least one sample.")
    if observations.shape[1] == 0 or observations.shape[2] == 0:
        raise ValueError("observations must have non-empty window and asset dimensions.")
    if actions.shape != (observations.shape[0], observations.shape[2]):
        raise ValueError(
            "actions must align with observations: expected "
            f"{(observations.shape[0], observations.shape[2])}, got {actions.shape}."
        )
    if not np.all(np.isfinite(observations)):
        raise ValueError("observations must contain only finite values.")
    if not np.all(np.isfinite(actions)):
        raise ValueError("actions must contain only finite values.")

    return observations, actions


def load_demonstrations(path: str | Path) -> Demonstrations:
    """Load and validate a ``demonstrations.npz`` file.

    Latent regimes are returned only as optional diagnostics; they are never
    folded into the observation tensor by this helper.
    """
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        missing = {"observations", "actions"} - set(data.files)
        if missing:
            raise ValueError(f"demonstration file is missing keys: {sorted(missing)}.")

        observations, actions = validate_demonstrations(data["observations"], data["actions"])
        returns = np.asarray(data["returns"], dtype=float) if "returns" in data.files else None
        regimes = np.asarray(data["regimes"]) if "regimes" in data.files else None
        decision_times = (
            np.asarray(data["decision_times"], dtype=int)
            if "decision_times" in data.files
            else None
        )

    if returns is not None:
        if returns.ndim != 2 or returns.shape[1] != observations.shape[2]:
            raise ValueError("returns must have shape (T, n_assets).")
        if not np.all(np.isfinite(returns)):
            raise ValueError("returns must contain only finite values.")
    if regimes is not None:
        if regimes.ndim != 1:
            raise ValueError("regimes must be a 1-D array.")
        if returns is not None and regimes.shape[0] != returns.shape[0]:
            raise ValueError("regimes length must match returns length.")
    if decision_times is not None and decision_times.shape != (observations.shape[0],):
        raise ValueError("decision_times must have shape (n_samples,).")

    return Demonstrations(
        observations=observations,
        actions=actions,
        returns=returns,
        regimes=regimes,
        decision_times=decision_times,
    )


def train_validation_split(
    demonstrations: Demonstrations,
    train_fraction: float = 0.8,
) -> tuple[Demonstrations, Demonstrations]:
    """Chronologically split demonstrations without fitting any statistics."""
    if not isinstance(demonstrations, Demonstrations):
        raise TypeError("demonstrations must be a Demonstrations instance.")
    if not np.isfinite(train_fraction) or not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be strictly between zero and one.")
    if demonstrations.n_samples < 2:
        raise ValueError("at least two samples are required for a train/validation split.")

    split_index = int(np.floor(demonstrations.n_samples * train_fraction))
    split_index = min(max(split_index, 1), demonstrations.n_samples - 1)

    def _slice(start: int, stop: int) -> Demonstrations:
        decision_times = demonstrations.decision_times
        return Demonstrations(
            observations=demonstrations.observations[start:stop],
            actions=demonstrations.actions[start:stop],
            returns=None,
            regimes=None,
            decision_times=None if decision_times is None else decision_times[start:stop],
        )

    return _slice(0, split_index), _slice(split_index, demonstrations.n_samples)


def fit_return_normalizer(
    returns: pd.DataFrame | np.ndarray,
    method: str = "zscore",
) -> ReturnNormalizer:
    """Fit return normalization statistics on the provided training data."""
    method = _validate_method(method)
    values, columns, _index = _coerce_returns(returns)
    flat = values.reshape(-1, values.shape[-1])

    if method == "zscore":
        location = flat.mean(axis=0)
        scale = flat.std(axis=0, ddof=0)
    else:
        location = flat.min(axis=0)
        scale = flat.max(axis=0) - location

    scale = np.where(scale > 1e-12, scale, 1.0)
    return ReturnNormalizer(
        method=method,
        location=location.astype(float),
        scale=scale.astype(float),
        columns=None if columns is None else tuple(columns),
    )


def normalise_returns(
    returns: pd.DataFrame | np.ndarray,
    method: str = "zscore",
    normalizer: ReturnNormalizer | None = None,
) -> pd.DataFrame | np.ndarray:
    """Normalise return series for neural network inputs.

    Args:
        returns: Raw return DataFrame or array. Arrays may be shaped
            ``(T, n_assets)`` or ``(n_samples, window, n_assets)``.
        method: ``"zscore"`` or ``"minmax"``. Ignored when ``normalizer`` is supplied.
        normalizer: Pre-fitted training-only normalizer. Pass this when
            transforming validation/test data to avoid look-ahead leakage.
    """
    if normalizer is None:
        normalizer = fit_return_normalizer(returns, method=method)
    return normalizer.transform(returns)


def normalize_returns(
    returns: pd.DataFrame | np.ndarray,
    method: str = "zscore",
    normalizer: ReturnNormalizer | None = None,
) -> pd.DataFrame | np.ndarray:
    """US-spelling alias for :func:`normalise_returns`."""
    return normalise_returns(returns, method=method, normalizer=normalizer)


def _validate_method(method: str) -> str:
    if method not in {"zscore", "minmax"}:
        raise ValueError("method must be either 'zscore' or 'minmax'.")
    return method


def _coerce_returns(
    returns: pd.DataFrame | np.ndarray,
) -> tuple[np.ndarray, pd.Index | None, pd.Index | None]:
    if isinstance(returns, pd.DataFrame):
        values = returns.to_numpy(dtype=float)
        columns = returns.columns
        index = returns.index
    else:
        values = np.asarray(returns, dtype=float)
        columns = None
        index = None

    if values.ndim not in {2, 3}:
        raise ValueError("returns must have shape (T, n_assets) or (n_samples, window, n_assets).")
    if any(dimension == 0 for dimension in values.shape[:-1]):
        raise ValueError("returns must have non-empty sample and window dimensions.")
    if values.shape[-1] == 0:
        raise ValueError("returns must contain at least one asset.")
    if not np.all(np.isfinite(values)):
        raise ValueError("returns must contain only finite values.")
    return values, columns, index
