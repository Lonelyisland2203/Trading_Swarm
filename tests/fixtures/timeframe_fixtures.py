"""Test fixtures for multi-timeframe analysis."""

import pandas as pd
import numpy as np


def create_test_df_bullish(bars: int = 100, base_price: float = 50000.0) -> pd.DataFrame:
    """
    Create DataFrame with bullish indicator pattern.

    Pattern:
    - Uptrend: price increases ~1% per 10 bars
    - Above Ichimoku cloud (requires 52 bars warmup)
    - KAMA rising
    - Donchian breakout to upside
    - RSI 55-65 (neutral-bullish)

    Args:
        bars: Number of bars to generate
        base_price: Starting price

    Returns:
        OHLCV DataFrame with bullish pattern
    """
    np.random.seed(42)  # Reproducible

    end_time = pd.Timestamp("2024-01-01 00:00:00")
    timestamps = pd.date_range(end=end_time, periods=bars, freq="1h")
    timestamps_ms = (timestamps.astype(int) // 10**6).values

    # Generate uptrending prices
    trend = np.linspace(0, 0.15, bars)  # 15% uptrend over period
    noise = np.random.normal(0, 0.005, bars)  # 0.5% noise
    price_multiplier = 1 + trend + noise

    close_prices = base_price * price_multiplier

    # Generate OHLC with upward bias
    high_prices = close_prices * (1 + np.abs(np.random.normal(0, 0.01, bars)))
    low_prices = close_prices * (1 - np.abs(np.random.normal(0, 0.008, bars)))
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = close_prices[0] * 0.99

    volume = np.random.uniform(1000, 2000, bars)

    df = pd.DataFrame({
        "timestamp": timestamps_ms,
        "open": open_prices,
        "high": high_prices,
        "low": low_prices,
        "close": close_prices,
        "volume": volume,
    })

    # Ensure OHLC relationship
    df["high"] = df[["open", "close", "high"]].max(axis=1)
    df["low"] = df[["open", "close", "low"]].min(axis=1)

    return df


def create_test_df_bearish(bars: int = 100, base_price: float = 50000.0) -> pd.DataFrame:
    """
    Create DataFrame with bearish indicator pattern.

    Pattern:
    - Downtrend: price decreases ~1% per 10 bars
    - Below Ichimoku cloud
    - KAMA falling
    - Donchian breakout to downside
    - RSI 35-45 (neutral-bearish)

    Args:
        bars: Number of bars to generate
        base_price: Starting price

    Returns:
        OHLCV DataFrame with bearish pattern
    """
    np.random.seed(43)  # Reproducible, different from bullish

    end_time = pd.Timestamp("2024-01-01 00:00:00")
    timestamps = pd.date_range(end=end_time, periods=bars, freq="1h")
    timestamps_ms = (timestamps.astype(int) // 10**6).values

    # Generate downtrending prices
    trend = np.linspace(0, -0.15, bars)  # 15% downtrend over period
    noise = np.random.normal(0, 0.005, bars)
    price_multiplier = 1 + trend + noise

    close_prices = base_price * price_multiplier

    # Generate OHLC with downward bias
    high_prices = close_prices * (1 + np.abs(np.random.normal(0, 0.008, bars)))
    low_prices = close_prices * (1 - np.abs(np.random.normal(0, 0.01, bars)))
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = close_prices[0] * 1.01

    volume = np.random.uniform(1000, 2000, bars)

    df = pd.DataFrame({
        "timestamp": timestamps_ms,
        "open": open_prices,
        "high": high_prices,
        "low": low_prices,
        "close": close_prices,
        "volume": volume,
    })

    # Ensure OHLC relationship
    df["high"] = df[["open", "close", "high"]].max(axis=1)
    df["low"] = df[["open", "close", "low"]].min(axis=1)

    return df


def create_test_df_neutral(bars: int = 100, base_price: float = 50000.0) -> pd.DataFrame:
    """
    Create DataFrame with neutral indicator pattern.

    Pattern:
    - Sideways: price oscillates around base with no trend
    - Inside Ichimoku cloud
    - KAMA flat
    - Donchian middle range
    - RSI near 50

    Args:
        bars: Number of bars to generate
        base_price: Starting price

    Returns:
        OHLCV DataFrame with neutral pattern
    """
    np.random.seed(44)  # Reproducible

    end_time = pd.Timestamp("2024-01-01 00:00:00")
    timestamps = pd.date_range(end=end_time, periods=bars, freq="1h")
    timestamps_ms = (timestamps.astype(int) // 10**6).values

    # Generate sideways prices with mean reversion
    noise = np.random.normal(0, 0.01, bars)  # 1% oscillation
    close_prices = base_price * (1 + noise)

    # Generate OHLC with symmetric volatility
    high_prices = close_prices * (1 + np.abs(np.random.normal(0, 0.008, bars)))
    low_prices = close_prices * (1 - np.abs(np.random.normal(0, 0.008, bars)))
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = close_prices[0]

    volume = np.random.uniform(1000, 2000, bars)

    df = pd.DataFrame({
        "timestamp": timestamps_ms,
        "open": open_prices,
        "high": high_prices,
        "low": low_prices,
        "close": close_prices,
        "volume": volume,
    })

    # Ensure OHLC relationship
    df["high"] = df[["open", "close", "high"]].max(axis=1)
    df["low"] = df[["open", "close", "low"]].min(axis=1)

    return df
