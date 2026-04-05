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


def compute_kama(
    close: pd.Series,
    period: int = 10,
    fast_ema: int = 2,
    slow_ema: int = 30,
) -> pd.Series:
    """
    Compute Kaufman Adaptive Moving Average.

    Adjusts smoothing based on market efficiency ratio:
    - High efficiency (trending) → fast smoothing (responsive)
    - Low efficiency (choppy) → slow smoothing (filters noise)

    Efficiency Ratio = |price_change| / sum(|bar_changes|)
    Smoothing Constant = [ER * (fast_sc - slow_sc) + slow_sc]^2
    KAMA = prev_KAMA + SC * (price - prev_KAMA)

    Args:
        close: Close price series
        period: Efficiency ratio period (default: 10)
        fast_ema: Fast EMA constant (default: 2)
        slow_ema: Slow EMA constant (default: 30)

    Returns:
        KAMA series
    """
    # Convert EMA periods to smoothing constants
    fast_sc = 2.0 / (fast_ema + 1)
    slow_sc = 2.0 / (slow_ema + 1)

    # Calculate price changes
    price_change = (close - close.shift(period)).abs()

    # Calculate sum of absolute bar-to-bar changes
    volatility = close.diff().abs().rolling(window=period).sum()

    # Efficiency Ratio (avoid division by zero)
    er = price_change / volatility.where(volatility > EPSILON, np.nan)

    # Handle constant price: when both numerator and denominator are zero
    # In this case, ER should be 0 (use slow smoothing, KAMA becomes stable)
    has_data = price_change.notna() & volatility.notna()
    constant_price = has_data & (price_change <= EPSILON) & (volatility <= EPSILON)
    er = er.where(~constant_price, 0.0)

    # Smoothing Constant
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2

    # Calculate KAMA iteratively
    kama_values = pd.Series(index=close.index, dtype=float)

    # First valid value: use SMA
    first_valid_idx = period
    if first_valid_idx < len(close):
        kama_values.iloc[first_valid_idx] = close.iloc[:first_valid_idx + 1].mean()

        # Iterate from first_valid + 1 to end
        for i in range(first_valid_idx + 1, len(close)):
            if pd.notna(sc.iloc[i]) and pd.notna(kama_values.iloc[i - 1]):
                kama_values.iloc[i] = (
                    kama_values.iloc[i - 1] +
                    sc.iloc[i] * (close.iloc[i] - kama_values.iloc[i - 1])
                )

    return kama_values


def compute_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    Compute On-Balance Volume.

    Cumulative volume with direction based on close:
    - Add volume if close > previous close
    - Subtract volume if close < previous close
    - No change if close == previous close

    Args:
        close: Close price series
        volume: Volume series

    Returns:
        Cumulative OBV series
    """
    # Determine direction: 1 (up), -1 (down), 0 (unchanged)
    direction = pd.Series(0, index=close.index)
    direction[close > close.shift(1)] = 1
    direction[close < close.shift(1)] = -1

    # Signed volume
    signed_volume = direction * volume

    # Cumulative sum
    return signed_volume.cumsum()


def compute_cmf(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    period: int = 20,
) -> pd.Series:
    """
    Compute Chaikin Money Flow.

    Weighted average of money flow volume over period.

    Money Flow Multiplier = ((close - low) - (high - close)) / (high - low)
    Money Flow Volume = multiplier * volume
    CMF = sum(MFV over period) / sum(volume over period)

    Args:
        high: High price series
        low: Low price series
        close: Close price series
        volume: Volume series
        period: Lookback period (default: 20)

    Returns:
        CMF series (range: -1.0 to +1.0)
    """
    # Money Flow Multiplier (avoid division by zero)
    range_hl = high - low
    mf_multiplier = ((close - low) - (high - close)) / range_hl.where(range_hl > EPSILON, np.nan)

    # Money Flow Volume
    mf_volume = mf_multiplier * volume

    # CMF: rolling sum of MFV / rolling sum of volume
    mfv_sum = mf_volume.rolling(window=period).sum()
    volume_sum = volume.rolling(window=period).sum()

    return mfv_sum / volume_sum.where(volume_sum > EPSILON, np.nan)


def compute_mfi(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    Compute Money Flow Index (volume-weighted RSI).

    Typical Price = (high + low + close) / 3
    Raw Money Flow = Typical Price * Volume

    Separate positive flow (TP increased) from negative flow (TP decreased)
    Money Ratio = sum(positive_flow) / sum(negative_flow)
    MFI = 100 - (100 / (1 + Money Ratio))

    Args:
        high: High price series
        low: Low price series
        close: Close price series
        volume: Volume series
        period: Lookback period (default: 14)

    Returns:
        MFI series (range: 0-100)
    """
    # Typical Price
    typical_price = (high + low + close) / 3.0

    # Raw Money Flow
    raw_money_flow = typical_price * volume

    # Separate positive and negative flows
    tp_delta = typical_price.diff()
    positive_flow = raw_money_flow.where(tp_delta > 0, 0.0)
    negative_flow = raw_money_flow.where(tp_delta < 0, 0.0)

    # Rolling sums
    positive_sum = positive_flow.rolling(window=period).sum()
    negative_sum = negative_flow.rolling(window=period).sum()

    # Money Ratio (avoid division by zero)
    money_ratio = positive_sum / negative_sum.where(negative_sum > EPSILON, np.nan)

    # MFI
    mfi_series = 100 - (100 / (1 + money_ratio))

    # Handle edge cases: pure positive or pure negative
    has_data = positive_sum.notna() & negative_sum.notna()
    pure_positive = has_data & (negative_sum <= EPSILON) & (positive_sum > EPSILON)
    pure_negative = has_data & (positive_sum <= EPSILON) & (negative_sum > EPSILON)

    mfi_series = mfi_series.where(~pure_positive, 100.0)
    mfi_series = mfi_series.where(~pure_negative, 0.0)

    return mfi_series


def compute_vwap(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
) -> pd.Series:
    """
    Compute Volume-Weighted Average Price.

    VWAP = cumulative(typical_price * volume) / cumulative(volume)
    Typical Price = (high + low + close) / 3

    Note: In production, VWAP resets at session boundaries.
    Here it's cumulative over the entire window.

    Args:
        high: High price series
        low: Low price series
        close: Close price series
        volume: Volume series

    Returns:
        Cumulative VWAP series
    """
    # Typical Price
    typical_price = (high + low + close) / 3.0

    # Cumulative sums
    cum_tp_volume = (typical_price * volume).cumsum()
    cum_volume = volume.cumsum()

    # VWAP (avoid division by zero)
    return cum_tp_volume / cum_volume.where(cum_volume > EPSILON, np.nan)


def compute_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    Compute Average True Range.

    True Range = max of:
    - high - low
    - |high - previous_close|
    - |low - previous_close|

    ATR = Wilder's smoothed average of True Range

    Args:
        high: High price series
        low: Low price series
        close: Close price series
        period: Smoothing period (default: 14)

    Returns:
        ATR series
    """
    prev_close = close.shift(1)

    # True Range components
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    # True Range = max of three components
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Wilder's smoothing: alpha = 1/period
    return true_range.ewm(alpha=1/period, min_periods=period, adjust=False).mean()


def compute_atr_normalized(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    Compute normalized ATR as percentage of close price.

    ATR Normalized = (ATR / close) * 100

    Useful for comparing volatility across different price levels or assets.

    Args:
        high: High price series
        low: Low price series
        close: Close price series
        period: ATR period (default: 14)

    Returns:
        ATR normalized as percentage series
    """
    atr_value = compute_atr(high, low, close, period)
    return (atr_value / close.where(close.abs() > EPSILON, np.nan)) * 100


def compute_bb_width(
    close: pd.Series,
    period: int = 20,
    num_std: float = 2.0,
) -> pd.Series:
    """
    Compute Bollinger Band Width as percentage.

    Width = (upper_band - lower_band) / middle_band * 100

    Measures volatility expansion (high width) vs contraction (low width).
    Squeeze occurs when width is historically low.

    Args:
        close: Close price series
        period: BB period (default: 20)
        num_std: Standard deviation multiplier (default: 2.0)

    Returns:
        BB width as percentage series
    """
    middle = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()

    upper = middle + (std * num_std)
    lower = middle - (std * num_std)

    # Width as percentage (avoid division by zero)
    return ((upper - lower) / middle.where(middle.abs() > EPSILON, np.nan)) * 100


def compute_keltner_channels(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    ema_period: int = 20,
    atr_period: int = 10,
    atr_multiplier: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Compute Keltner Channels (upper, middle, lower).

    Middle = EMA of close
    Upper = Middle + (ATR * multiplier)
    Lower = Middle - (ATR * multiplier)

    Similar to Bollinger Bands but uses ATR instead of standard deviation.
    More responsive to volatility changes.

    Args:
        high: High price series
        low: Low price series
        close: Close price series
        ema_period: EMA period for middle line (default: 20)
        atr_period: ATR period (default: 10)
        atr_multiplier: ATR multiplier (default: 2.0)

    Returns:
        Tuple of (upper_channel, middle_channel, lower_channel)
    """
    # Middle band: EMA of close
    middle = close.ewm(span=ema_period, adjust=False).mean()

    # ATR
    atr_series = compute_atr(high, low, close, period=atr_period)

    # Upper and lower bands
    upper = middle + (atr_series * atr_multiplier)
    lower = middle - (atr_series * atr_multiplier)

    return upper, middle, lower


def compute_donchian_width(
    high: pd.Series,
    low: pd.Series,
    period: int = 20,
) -> pd.Series:
    """
    Compute Donchian Channel Width as percentage.

    Width = (upper_channel - lower_channel) / middle_channel * 100

    Similar to BB Width but based on high/low extremes rather than
    standard deviation. Useful for breakout detection.

    Args:
        high: High price series
        low: Low price series
        period: Donchian period (default: 20)

    Returns:
        Donchian width as percentage series
    """
    upper, middle, lower = compute_donchian_channels(high, low, period=period)

    # Width as percentage (avoid division by zero)
    return ((upper - lower) / middle.where(middle.abs() > EPSILON, np.nan)) * 100


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
