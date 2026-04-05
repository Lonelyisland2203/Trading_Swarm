"""
Historical window walking for training data generation.

Implements temporal window generation by walking backward from current time
with configurable stride. Validates data completeness to ensure quality.
"""

from dataclasses import dataclass
from typing import Optional

import pandas as pd
from loguru import logger

from .market_data import MarketDataService


# Timeframe to milliseconds mapping
TIMEFRAME_TO_MS = {
    "1m": 60_000,
    "5m": 300_000,
    "15m": 900_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}


@dataclass(slots=True, frozen=True)
class HistoricalWindow:
    """
    Represents a specific window in time for data fetching.

    Attributes:
        symbol: Trading pair (e.g., "BTC/USDT")
        timeframe: Candle timeframe (e.g., "1h")
        end_timestamp_ms: Window end time in milliseconds (as-of time)
        lookback_bars: Number of bars to fetch before end
        stride_index: Which window in the sequence (0=current, 1=first back, etc.)
    """

    symbol: str
    timeframe: str
    end_timestamp_ms: int
    lookback_bars: int
    stride_index: int


def timeframe_to_milliseconds(timeframe: str) -> int:
    """
    Convert timeframe string to milliseconds.

    Args:
        timeframe: Timeframe string (e.g., "1h", "4h", "1d")

    Returns:
        Milliseconds per bar

    Raises:
        ValueError: If timeframe is not supported
    """
    if timeframe not in TIMEFRAME_TO_MS:
        raise ValueError(
            f"Unsupported timeframe: {timeframe}. "
            f"Supported: {list(TIMEFRAME_TO_MS.keys())}"
        )

    return TIMEFRAME_TO_MS[timeframe]


def calculate_window_timestamps(
    current_timestamp_ms: int,
    timeframe: str,
    window_count: int,
    stride_bars: int,
) -> list[int]:
    """
    Calculate timestamps for historical windows walking backward.

    Windows are evenly spaced by stride_bars, walking backward from current time.
    This ensures temporal diversity across different market regimes.

    Args:
        current_timestamp_ms: Current/latest timestamp in milliseconds
        timeframe: Candle timeframe (e.g., "1h")
        window_count: Number of windows to generate
        stride_bars: Bars between each window

    Returns:
        List of end timestamps in descending order (newest first)

    Example:
        >>> # Generate 3 windows with 100-bar stride on 1h timeframe
        >>> timestamps = calculate_window_timestamps(
        ...     current_timestamp_ms=1704067200000,  # 2024-01-01 00:00
        ...     timeframe="1h",
        ...     window_count=3,
        ...     stride_bars=100,
        ... )
        >>> # Returns: [
        >>> #   1704067200000,  # Window 0: current
        >>> #   1703707200000,  # Window 1: -100 hours
        >>> #   1703347200000,  # Window 2: -200 hours
        >>> # ]

    Raises:
        ValueError: If window_count or stride_bars are invalid
    """
    if window_count < 1:
        raise ValueError(f"window_count must be >= 1, got {window_count}")

    if stride_bars < 1:
        raise ValueError(f"stride_bars must be >= 1, got {stride_bars}")

    # Convert stride from bars to milliseconds
    ms_per_bar = timeframe_to_milliseconds(timeframe)
    stride_ms = stride_bars * ms_per_bar

    # Generate timestamps walking backward
    timestamps = []
    for i in range(window_count):
        window_end_ms = current_timestamp_ms - (i * stride_ms)
        timestamps.append(window_end_ms)

    logger.debug(
        "Calculated window timestamps",
        timeframe=timeframe,
        window_count=window_count,
        stride_bars=stride_bars,
        stride_ms=stride_ms,
    )

    return timestamps


def calculate_data_completeness(
    df: pd.DataFrame,
    expected_bars: int,
) -> float:
    """
    Calculate data completeness ratio.

    Args:
        df: OHLCV DataFrame
        expected_bars: Expected number of bars

    Returns:
        Completeness ratio (0.0 to 1.0)
    """
    if expected_bars == 0:
        return 0.0

    actual_bars = len(df)
    return min(actual_bars / expected_bars, 1.0)


async def fetch_window_data(
    service: MarketDataService,
    window: HistoricalWindow,
    min_completeness: float = 0.95,
) -> Optional[pd.DataFrame]:
    """
    Fetch OHLCV data for specific historical window.

    Validates data completeness against threshold. Returns None if data
    quality is insufficient.

    Args:
        service: Market data service instance
        window: Historical window specification
        min_completeness: Minimum completeness ratio (default 0.95)

    Returns:
        OHLCV DataFrame if data quality sufficient, None otherwise

    Example:
        >>> window = HistoricalWindow(
        ...     symbol="BTC/USDT",
        ...     timeframe="1h",
        ...     end_timestamp_ms=1704067200000,
        ...     lookback_bars=100,
        ...     stride_index=0,
        ... )
        >>> df = await fetch_window_data(service, window)
        >>> if df is not None:
        ...     print(f"Fetched {len(df)} bars")
    """
    try:
        # Fetch data up to window end time
        # Parameter "as_of" is the point-in-time timestamp (ms) from market_data.py
        df = await service.get_ohlcv_as_of(
            symbol=window.symbol,
            timeframe=window.timeframe,
            as_of=window.end_timestamp_ms,
            lookback_bars=window.lookback_bars + 10,  # Extra buffer for filtering
        )

        if df is None or df.empty:
            logger.warning(
                "No data returned for window",
                symbol=window.symbol,
                timeframe=window.timeframe,
                stride_index=window.stride_index,
            )
            return None

        # Filter to exact window end (point-in-time safety)
        df = df[df["timestamp"] <= window.end_timestamp_ms]

        # Take only requested lookback bars
        df = df.tail(window.lookback_bars)

        # Check completeness
        completeness = calculate_data_completeness(df, window.lookback_bars)

        if completeness < min_completeness:
            logger.warning(
                "Insufficient data completeness",
                symbol=window.symbol,
                timeframe=window.timeframe,
                stride_index=window.stride_index,
                completeness=f"{completeness:.2%}",
                threshold=f"{min_completeness:.2%}",
                actual_bars=len(df),
                expected_bars=window.lookback_bars,
            )
            return None

        logger.debug(
            "Window data fetched",
            symbol=window.symbol,
            timeframe=window.timeframe,
            stride_index=window.stride_index,
            bars=len(df),
            completeness=f"{completeness:.2%}",
        )

        return df

    except Exception as e:
        logger.error(
            "Failed to fetch window data",
            symbol=window.symbol,
            timeframe=window.timeframe,
            stride_index=window.stride_index,
            error=str(e),
        )
        return None
