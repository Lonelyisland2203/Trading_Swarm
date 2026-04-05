"""
Technical indicators computed using pure pandas/numpy.

No TA-Lib dependency - all indicators implemented from mathematical definitions.
"""

import pandas as pd
import numpy as np
from loguru import logger

# Numerical precision threshold for division safety
EPSILON = 1e-10


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Relative Strength Index using Wilder's smoothing method.

    Args:
        close: Close price series
        period: RSI period (default 14)

    Returns:
        RSI values (0-100)

    Note:
        Uses EWM with adjust=False for Wilder's smoothing (not SMA).
        First RSI value appears at index `period`.
    """
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    # Wilder's smoothing: alpha = 1/period
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

    # Avoid division by zero with epsilon threshold
    rs = avg_gain / avg_loss.where(avg_loss > EPSILON, np.nan)
    rsi = 100 - (100 / (1 + rs))

    # Handle edge cases only after warmup period (where we have valid averages)
    # Warmup condition: both avg_gain and avg_loss should be non-NaN
    has_data = avg_gain.notna() & avg_loss.notna()

    # Pure gains (no losses): avg_loss near zero but avg_gain positive -> RSI = 100
    pure_gains = has_data & (avg_loss <= EPSILON) & (avg_gain > EPSILON)
    rsi = rsi.where(~pure_gains, 100.0)

    # Pure losses (no gains): avg_gain near zero but avg_loss positive -> RSI = 0
    pure_losses = has_data & (avg_gain <= EPSILON) & (avg_loss > EPSILON)
    rsi = rsi.where(~pure_losses, 0.0)

    return rsi


def compute_macd(
    close: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    MACD (Moving Average Convergence Divergence) with standard EMA parameters.

    Args:
        close: Close price series
        fast_period: Fast EMA period (default 12)
        slow_period: Slow EMA period (default 26)
        signal_period: Signal line EMA period (default 9)

    Returns:
        Tuple of (macd_line, signal_line, histogram)
    """
    # Standard EMA (not Wilder's)
    ema_fast = close.ewm(span=fast_period, adjust=False).mean()
    ema_slow = close.ewm(span=slow_period, adjust=False).mean()

    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


def compute_bollinger_bands(
    close: pd.Series,
    period: int = 20,
    num_std: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Bollinger Bands with SMA middle band.

    Args:
        close: Close price series
        period: Moving average period (default 20)
        num_std: Number of standard deviations (default 2.0)

    Returns:
        Tuple of (upper_band, middle_band, lower_band)
    """
    middle = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()

    upper = middle + (std * num_std)
    lower = middle - (std * num_std)

    return upper, middle, lower


def compute_bb_position(close: pd.Series, period: int = 20, num_std: float = 2.0) -> pd.Series:
    """
    Calculate price position within Bollinger Bands.

    Args:
        close: Close price series
        period: BB period (default 20)
        num_std: BB standard deviations (default 2.0)

    Returns:
        Position ratio (0.0 = lower band, 0.5 = middle, 1.0 = upper band)
    """
    upper, middle, lower = compute_bollinger_bands(close, period, num_std)

    # Avoid division by zero with epsilon threshold
    band_width = upper - lower
    position = (close - lower) / band_width.where(band_width.abs() > EPSILON, np.nan)

    return position


def compute_donchian_channels(
    high: pd.Series,
    low: pd.Series,
    period: int = 20,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Compute Donchian Channels (highest high and lowest low over period).

    Args:
        high: High price series
        low: Low price series
        period: Lookback period (default: 20)

    Returns:
        Tuple of (upper_band, middle_band, lower_band)
    """
    upper = high.rolling(window=period).max()
    lower = low.rolling(window=period).min()
    middle = (upper + lower) / 2.0

    return upper, middle, lower


def compute_donchian_channels_long(
    high: pd.Series,
    low: pd.Series,
    period: int = 55,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Compute long-term Donchian Channels (55-period default).

    Args:
        high: High price series
        low: Low price series
        period: Lookback period (default: 55)

    Returns:
        Tuple of (upper_band, middle_band, lower_band)
    """
    return compute_donchian_channels(high, low, period=period)


def compute_ichimoku_cloud(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    tenkan_period: int = 9,
    kijun_period: int = 26,
    senkou_span_b_period: int = 52,
) -> dict[str, pd.Series]:
    """
    Compute Ichimoku Cloud components.

    Components:
    - Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    - Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    - Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2, shifted forward 26
    - Senkou Span B (Leading Span B): (52-period high + low) / 2, shifted forward 26
    - Chikou Span (Lagging Span): Close shifted backward 26

    Args:
        high: High price series
        low: Low price series
        close: Close price series
        tenkan_period: Conversion line period (default: 9)
        kijun_period: Base line period (default: 26)
        senkou_span_b_period: Leading Span B period (default: 52)

    Returns:
        Dict with keys: tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span
    """
    # Tenkan-sen (Conversion Line)
    tenkan_high = high.rolling(window=tenkan_period).max()
    tenkan_low = low.rolling(window=tenkan_period).min()
    tenkan_sen = (tenkan_high + tenkan_low) / 2.0

    # Kijun-sen (Base Line)
    kijun_high = high.rolling(window=kijun_period).max()
    kijun_low = low.rolling(window=kijun_period).min()
    kijun_sen = (kijun_high + kijun_low) / 2.0

    # Senkou Span A (Leading Span A) - shifted forward 26 periods
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2.0).shift(kijun_period)

    # Senkou Span B (Leading Span B) - shifted forward 26 periods
    senkou_high = high.rolling(window=senkou_span_b_period).max()
    senkou_low = low.rolling(window=senkou_span_b_period).min()
    senkou_span_b = ((senkou_high + senkou_low) / 2.0).shift(kijun_period)

    # Chikou Span (Lagging Span) - close shifted backward 26 periods
    chikou_span = close.shift(-kijun_period)

    return {
        'tenkan_sen': tenkan_sen,
        'kijun_sen': kijun_sen,
        'senkou_span_a': senkou_span_a,
        'senkou_span_b': senkou_span_b,
        'chikou_span': chikou_span,
    }


def validate_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate OHLCV data integrity and fix common issues.

    Args:
        df: DataFrame with OHLCV columns

    Returns:
        Validated and cleaned DataFrame

    Raises:
        ValueError: If required columns are missing or DataFrame is empty
    """
    # Check for empty DataFrame
    if df.empty:
        raise ValueError("OHLCV DataFrame is empty")

    # Check for required columns
    required = ["timestamp", "open", "high", "low", "close", "volume"]
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required OHLCV columns: {missing}")

    # Check OHLC relationship: low <= open,close <= high
    invalid_bars = df[
        (df["low"] > df["open"]) |
        (df["low"] > df["close"]) |
        (df["high"] < df["open"]) |
        (df["high"] < df["close"])
    ]

    if not invalid_bars.empty:
        logger.warning(
            "Invalid OHLC bars detected (dropping)",
            count=len(invalid_bars),
            first_timestamp=invalid_bars.iloc[0]["timestamp"] if len(invalid_bars) > 0 else None
        )
        # Drop invalid bars
        df = df.drop(invalid_bars.index)

    # Check for duplicate timestamps
    duplicates = df[df["timestamp"].duplicated()]
    if not duplicates.empty:
        logger.warning("Duplicate timestamps detected (keeping last)", count=len(duplicates))
        df = df.drop_duplicates(subset=["timestamp"], keep="last")

    # Sort by timestamp
    df = df.sort_values("timestamp").reset_index(drop=True)

    return df
