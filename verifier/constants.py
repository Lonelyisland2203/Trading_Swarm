"""
Constants for backtesting and verification.

Defines timeframe-specific horizons and transaction cost defaults.
"""

# Timeframe-adaptive forward measurement windows
# Maps timeframe -> number of bars to measure forward return
HORIZON_BARS: dict[str, int] = {
    "1m": 60,    # 1 hour  - scalping signals
    "5m": 48,    # 4 hours - intraday swing
    "15m": 24,   # 6 hours - intraday position
    "1h": 24,    # 24 hours - day trade
    "4h": 12,    # 48 hours - swing trade
    "1d": 5,     # 5 days - position trade
}

# Default transaction cost (conservative estimate)
# 0.1% = 10 basis points, covers:
# - Binance taker: 0.075-0.1%
# - Kraken taker: 0.16%
# - Coinbase Pro: 0.2-0.5%
DEFAULT_TXN_COST_PCT = 0.001  # 0.1%


def get_horizon_bars(timeframe: str) -> int:
    """
    Get forward measurement window for timeframe.

    Args:
        timeframe: Timeframe string (e.g., "1h", "15m")

    Returns:
        Number of bars to measure forward

    Raises:
        ValueError: Unknown timeframe

    Example:
        >>> get_horizon_bars("1h")
        24  # Measure 24 hours forward
    """
    if timeframe not in HORIZON_BARS:
        valid = ", ".join(sorted(HORIZON_BARS.keys()))
        raise ValueError(
            f"Unknown timeframe: '{timeframe}'. Valid timeframes: {valid}"
        )

    return HORIZON_BARS[timeframe]


def compute_holding_periods_8h(timeframe: str, horizon_bars: int) -> float:
    """
    Compute holding period in 8-hour units (funding periods).

    Args:
        timeframe: Timeframe string (e.g., "1m", "1h", "1d")
        horizon_bars: Number of bars to hold position

    Returns:
        Holding period as fraction of 8h periods

    Examples:
        >>> compute_holding_periods_8h("1m", 60)  # 60 minutes
        0.125  # 1/8 of a funding period
        >>> compute_holding_periods_8h("1h", 24)  # 24 hours
        3.0    # 3 funding periods
        >>> compute_holding_periods_8h("1d", 5)   # 5 days
        15.0   # 15 funding periods
    """
    # Parse timeframe to hours per bar
    unit = timeframe[-1]
    value = int(timeframe[:-1]) if len(timeframe) > 1 else 1

    if unit == "m":
        hours_per_bar = value / 60.0
    elif unit == "h":
        hours_per_bar = value
    elif unit == "d":
        hours_per_bar = value * 24.0
    else:
        raise ValueError(f"Unsupported timeframe unit: {unit}")

    total_hours = hours_per_bar * horizon_bars
    return total_hours / 8.0
