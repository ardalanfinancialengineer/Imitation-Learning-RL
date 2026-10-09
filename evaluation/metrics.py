"""Section 5.3 performance measures.

The exposé lists annualised return, annualised volatility, Sharpe ratio, Sortino
ratio, maximum drawdown, turnover, certainty-equivalent utility, expert tracking
error, utility loss, and decision-time latency.

Everything that depends only on a return or weight series lives here.
Certainty-equivalent utility and utility loss need the reward's risk aversion,
so they live with the reward in :mod:`src.reinforcement.reward`; decision-time
latency is measured where it is incurred, by ``scripts/check_expert_scale.py``
and the policy acceptance checks.

Every function takes a *periodic* (daily) return series and annualises with an
explicit factor, rather than taking pre-annualised inputs. Mixing the two
conventions is the classic way to publish a Sharpe ratio that is wrong by
sqrt(252).
"""

from __future__ import annotations

import numpy as np

#: Trading days per year. The exposé's Equation (14) uses 252.
TRADING_DAYS = 252


def annualised_return(returns: np.ndarray, annualisation: int = TRADING_DAYS) -> float:
    """Mean periodic return scaled to a year.

    Arithmetic rather than geometric, matching the mean used by the Sharpe ratio
    of Equation (14) so the two are consistent with each other.
    """
    values = _as_1d_returns(returns)
    _check_annualisation(annualisation)
    return float(values.mean()) * float(annualisation)


def annualised_volatility(
    returns: np.ndarray, annualisation: int = TRADING_DAYS
) -> float:
    """Standard deviation of periodic returns scaled to a year."""
    values = _as_1d_returns(returns)
    _check_annualisation(annualisation)
    if values.size < 2:
        return 0.0
    return float(values.std(ddof=1)) * float(np.sqrt(annualisation))


def sortino_ratio(
    returns: np.ndarray,
    risk_free: float = 0.0,
    annualisation: int = TRADING_DAYS,
    target: float = 0.0,
) -> float:
    """Annualised Sortino ratio: excess return over *downside* deviation.

    The Sharpe ratio penalises upside and downside variation equally. The
    Sortino ratio divides by the deviation of returns *below* ``target`` only,
    which matters here because the transaction-cost term makes the return
    distribution asymmetric — a policy that avoids losses and occasionally gains
    is not the same as one that is merely volatile.

    Downside deviation follows the usual convention of averaging squared
    shortfalls over *all* periods, not only the losing ones, so that a strategy
    with few but severe losses is not flattered by a small denominator.
    """
    values = _as_1d_returns(returns)
    if not np.isfinite(risk_free) or not np.isfinite(target):
        raise ValueError("risk_free and target must be finite.")
    _check_annualisation(annualisation)

    excess = values - risk_free
    shortfall = np.minimum(excess - target, 0.0)
    downside = float(np.sqrt(np.mean(shortfall**2)))
    mean_excess = float(excess.mean())

    if downside <= 1e-12:
        # No period fell below the target: the ratio is unbounded above, and
        # zero only if there was no excess return either.
        if np.isclose(mean_excess, 0.0):
            return 0.0
        return float(np.sign(mean_excess) * np.inf)
    return mean_excess / downside * float(np.sqrt(annualisation))


def sharpe_ratio(returns: np.ndarray, risk_free: float = 0.0, annualisation: int = 252) -> float:
    """Annualised Sharpe ratio of a return series.

    Args:
        returns: 1-D array of periodic portfolio returns.
        risk_free: Periodic risk-free rate.
        annualisation: Number of periods per year.
    """
    values = _as_1d_returns(returns)
    if not np.isfinite(risk_free):
        raise ValueError("risk_free must be finite.")
    if not isinstance(annualisation, int) or annualisation <= 0:
        raise ValueError("annualisation must be a positive integer.")

    excess = values - risk_free
    mean_excess = float(excess.mean())
    volatility = float(excess.std(ddof=1)) if excess.size > 1 else 0.0
    if volatility <= 1e-12:
        if np.isclose(mean_excess, 0.0):
            return 0.0
        return float(np.sign(mean_excess) * np.inf)
    return mean_excess / volatility * float(np.sqrt(annualisation))


def max_drawdown(cumulative_returns: np.ndarray) -> float:
    """Maximum peak-to-trough drawdown of a cumulative return series."""
    cumulative_returns = _as_1d_array(cumulative_returns, "cumulative_returns")
    wealth = 1.0 + cumulative_returns
    if np.any(wealth < 0.0):
        raise ValueError("cumulative_returns imply negative wealth.")

    wealth = np.concatenate(([1.0], wealth))
    peaks = np.maximum.accumulate(wealth)
    drawdowns = 1.0 - wealth / peaks
    return float(drawdowns.max())


def calmar_ratio(returns: np.ndarray, annualisation: int = 252) -> float:
    """Calmar ratio: annualised return divided by maximum drawdown."""
    values = _as_1d_returns(returns)
    if not isinstance(annualisation, int) or annualisation <= 0:
        raise ValueError("annualisation must be a positive integer.")
    if np.any(values < -1.0):
        raise ValueError("returns must be at least -100%.")

    wealth = np.cumprod(1.0 + values)
    cumulative = wealth - 1.0
    annualised_return = float(wealth[-1] ** (annualisation / values.size) - 1.0)
    drawdown = max_drawdown(cumulative)
    if drawdown <= 1e-12:
        if np.isclose(annualised_return, 0.0):
            return 0.0
        return float(np.sign(annualised_return) * np.inf)
    return annualised_return / drawdown


def turnover(weights: np.ndarray) -> float:
    """Mean one-way portfolio turnover across rebalancing steps.

    Args:
        weights: Weight matrix of shape (T, n_assets).
    """
    weights = _as_2d_weights(weights, "weights")
    if weights.shape[0] < 2:
        return 0.0
    one_way_turnover = 0.5 * np.abs(np.diff(weights, axis=0)).sum(axis=1)
    return float(one_way_turnover.mean())


def tracking_error(weights_agent: np.ndarray, weights_expert: np.ndarray) -> float:
    """Mean L1 distance between agent and expert weight allocations."""
    weights_agent = _as_2d_weights(weights_agent, "weights_agent")
    weights_expert = _as_2d_weights(weights_expert, "weights_expert")
    if weights_agent.shape != weights_expert.shape:
        raise ValueError(
            "weights_agent and weights_expert must have the same shape; "
            f"got {weights_agent.shape} and {weights_expert.shape}."
        )
    return float(np.abs(weights_agent - weights_expert).sum(axis=1).mean())


def _check_annualisation(annualisation: int) -> None:
    if not isinstance(annualisation, int) or isinstance(annualisation, bool):
        raise ValueError("annualisation must be a positive integer.")
    if annualisation <= 0:
        raise ValueError("annualisation must be a positive integer.")


def _as_1d_returns(values: np.ndarray) -> np.ndarray:
    values = _as_1d_array(values, "returns")
    if np.any(values < -1.0):
        raise ValueError("returns must be at least -100%.")
    return values


def _as_1d_array(values: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1:
        raise ValueError(f"{name} must be a 1-D array.")
    if values.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values.")
    return values


def _as_2d_weights(values: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError(f"{name} must be a 2-D array.")
    if values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError(f"{name} must have non-empty time and asset dimensions.")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values.")
    return values
