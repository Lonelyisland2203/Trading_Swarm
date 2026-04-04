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
