"""
Individual metric computation functions.

All metrics operate on numpy arrays for performance.
Statistical significance testing uses scipy.stats.
"""

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(slots=True, frozen=True)
class MetricValue:
    """
    Container for computed metric with metadata.

    Stores value, sample size, statistical significance, and confidence intervals.
    """

    value: float
    sample_size: int
    p_value: float | None = None
    ci_lower: float | None = None
    ci_upper: float | None = None
    confidence_level: str | None = None  # 'low', 'moderate', 'high'


def compute_information_coefficient(
    predicted: np.ndarray,
    actual: np.ndarray,
    method: str = "spearman",
) -> tuple[float, float]:
    """
    Compute information coefficient between predictions and outcomes.

    Args:
        predicted: Predicted values (e.g., confidence scores)
        actual: Actual outcomes (e.g., realized returns)
        method: 'spearman' (default) or 'pearson'

    Returns:
        Tuple of (correlation, p_value)

    Raises:
        ValueError: If arrays have different lengths or invalid method
    """
    if len(predicted) != len(actual):
        raise ValueError(f"Array length mismatch: {len(predicted)} vs {len(actual)}")

    if method == "spearman":
        corr, p_val = stats.spearmanr(predicted, actual)
    elif method == "pearson":
        corr, p_val = stats.pearsonr(predicted, actual)
    else:
        raise ValueError(f"Invalid method '{method}', must be 'spearman' or 'pearson'")

    # Handle NaN from constant arrays
    if np.isnan(corr):
        return 0.0, 1.0

    return float(corr), float(p_val)


def compute_sharpe_ratio(
    returns: np.ndarray,
    annualization_factor: int = 365,
    risk_free_rate: float = 0.0,
) -> float:
    """
    Compute annualized Sharpe ratio.

    Args:
        returns: Array of period returns (not cumulative)
        annualization_factor: 365 for crypto, 252 for equities
        risk_free_rate: Annual risk-free rate (default 0.0 for crypto)

    Returns:
        Annualized Sharpe ratio

    Raises:
        ValueError: If returns array is empty or has zero std
    """
    if len(returns) == 0:
        raise ValueError("Returns array is empty")

    mean_return = np.mean(returns)
    std_return = np.std(returns, ddof=1)

    if std_return == 0:
        # All returns are identical - undefined Sharpe
        return 0.0 if mean_return == 0 else np.inf

    # Annualize both mean and std
    annualized_return = mean_return * annualization_factor
    annualized_std = std_return * np.sqrt(annualization_factor)

    sharpe = (annualized_return - risk_free_rate) / annualized_std
    return float(sharpe)


def compute_sortino_ratio(
    returns: np.ndarray,
    annualization_factor: int = 365,
    mar: float = 0.0,
) -> float:
    """
    Compute annualized Sortino ratio.

    Uses downside deviation (returns below MAR) instead of total volatility.

    Args:
        returns: Array of period returns
        annualization_factor: 365 for crypto, 252 for equities
        mar: Minimum acceptable return (typically 0.0)

    Returns:
        Annualized Sortino ratio

    Raises:
        ValueError: If returns array is empty
    """
    if len(returns) == 0:
        raise ValueError("Returns array is empty")

    mean_return = np.mean(returns)

    # Downside deviation: only penalize returns below MAR
    downside_returns = returns[returns < mar]

    if len(downside_returns) == 0:
        # No downside - infinite Sortino (all returns >= MAR)
        return np.inf if mean_return > mar else 0.0

    downside_dev = np.std(downside_returns, ddof=1)

    if downside_dev == 0:
        return np.inf if mean_return > mar else 0.0

    # Annualize
    annualized_return = mean_return * annualization_factor
    annualized_downside_dev = downside_dev * np.sqrt(annualization_factor)

    sortino = (annualized_return - mar) / annualized_downside_dev
    return float(sortino)


def compute_max_drawdown(cumulative_returns: np.ndarray) -> float:
    """
    Compute maximum drawdown from cumulative returns.

    Args:
        cumulative_returns: Array of cumulative returns (not log returns)

    Returns:
        Maximum drawdown as positive decimal (e.g., 0.25 for 25% drawdown)

    Raises:
        ValueError: If cumulative_returns is empty
    """
    if len(cumulative_returns) == 0:
        raise ValueError("Cumulative returns array is empty")

    # Convert to wealth curve (starting at 1.0)
    wealth = 1.0 + cumulative_returns

    # Running maximum
    running_max = np.maximum.accumulate(wealth)

    # Drawdown at each point
    drawdowns = (running_max - wealth) / running_max

    max_dd = np.max(drawdowns)
    return float(max_dd)


def compute_calmar_ratio(
    returns: np.ndarray,
    annualization_factor: int = 365,
) -> float:
    """
    Compute Calmar ratio (annualized return / max drawdown).

    Args:
        returns: Array of period returns
        annualization_factor: 365 for crypto, 252 for equities

    Returns:
        Calmar ratio (higher is better)

    Raises:
        ValueError: If returns array is empty
    """
    if len(returns) == 0:
        raise ValueError("Returns array is empty")

    # Compute annualized return
    mean_return = np.mean(returns)
    annualized_return = mean_return * annualization_factor

    # Compute max drawdown from cumulative returns
    cumulative = np.cumsum(returns)
    max_dd = compute_max_drawdown(cumulative)

    if max_dd == 0:
        # No drawdown - infinite Calmar if positive return
        return np.inf if annualized_return > 0 else 0.0

    calmar = annualized_return / max_dd
    return float(calmar)


def compute_win_rate(profitable_mask: np.ndarray) -> float:
    """
    Compute win rate (fraction of profitable trades).

    Args:
        profitable_mask: Boolean array where True = profitable

    Returns:
        Win rate as decimal (0.0 to 1.0)

    Raises:
        ValueError: If array is empty
    """
    if len(profitable_mask) == 0:
        raise ValueError("Profitable mask array is empty")

    win_rate = np.mean(profitable_mask)
    return float(win_rate)


def compute_profit_factor(returns: np.ndarray) -> float:
    """
    Compute profit factor (gross profits / gross losses).

    Args:
        returns: Array of trade returns

    Returns:
        Profit factor (>1.0 means profitable overall)
        Returns np.inf if no losing trades
        Returns 0.0 if no winning trades

    Raises:
        ValueError: If returns array is empty
    """
    if len(returns) == 0:
        raise ValueError("Returns array is empty")

    profits = returns[returns > 0]
    losses = returns[returns < 0]

    gross_profit = np.sum(profits) if len(profits) > 0 else 0.0
    gross_loss = np.abs(np.sum(losses)) if len(losses) > 0 else 0.0

    if gross_loss == 0:
        # No losses - infinite profit factor if any profit
        return np.inf if gross_profit > 0 else 0.0

    profit_factor = gross_profit / gross_loss
    return float(profit_factor)


def bootstrap_confidence_interval(
    data: np.ndarray,
    metric_fn: callable,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
) -> tuple[float, float]:
    """
    Compute bootstrap confidence interval for a metric.

    Args:
        data: Input data array
        metric_fn: Function that computes metric from data array
        n_bootstrap: Number of bootstrap iterations
        confidence_level: Confidence level (e.g., 0.95 for 95% CI)

    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    if len(data) == 0:
        raise ValueError("Data array is empty")

    bootstrapped_metrics = []
    rng = np.random.default_rng(seed=42)  # Reproducible

    for _ in range(n_bootstrap):
        # Resample with replacement
        resampled = rng.choice(data, size=len(data), replace=True)
        try:
            metric_value = metric_fn(resampled)
            if not np.isinf(metric_value):  # Exclude infinite values
                bootstrapped_metrics.append(metric_value)
        except (ValueError, ZeroDivisionError):
            # Skip bootstrap samples that fail metric computation
            continue

    if len(bootstrapped_metrics) == 0:
        return (np.nan, np.nan)

    # Compute percentiles for confidence interval
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    ci_lower = float(np.percentile(bootstrapped_metrics, lower_percentile))
    ci_upper = float(np.percentile(bootstrapped_metrics, upper_percentile))

    return (ci_lower, ci_upper)
