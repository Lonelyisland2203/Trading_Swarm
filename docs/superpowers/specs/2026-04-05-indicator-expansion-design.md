# Technical Indicator Expansion - Design Specification

**Date:** 2026-04-05
**Status:** Approved
**Author:** Claude Code

## Problem Statement

The current `data/indicators.py` provides only three basic indicators (RSI, MACD, Bollinger Bands). This "poverty-level" feature set limits the generator's ability to analyze market conditions comprehensively.

**Current state:**
- 3 indicators total: RSI, MACD, Bollinger Bands
- No trend structure indicators (Donchian, Ichimoku, KAMA)
- No volume indicators (OBV, CMF, MFI, VWAP)
- No volatility indicators beyond BB (ATR, Keltner, squeeze detection)
- No market structure indicators (Fair Value Gaps, swing points)
- No crypto-specific indicators (funding rate, open interest)

**Impact:**
- Generator lacks context for trend identification
- Missing volume confirmation signals
- No institutional order flow detection (FVG)
- Prompts contain limited technical analysis depth

## Solution Overview

Expand `data/indicators.py` from 3 to 19 indicators across 5 groups, using pandas/numpy only (no paid libraries). Add aggregation function `compute_all_indicators()` that returns both scalar summaries (for prompt consumption) and full series (for advanced analysis).

**Groups:**
1. **Trend Structure** (4): Donchian Channels (short/long), Ichimoku Cloud, KAMA
2. **Volume** (4): OBV, CMF, MFI, VWAP
3. **Volatility** (5): ATR, ATR normalized, BB Width, Keltner Channels, Donchian Width, TTM Squeeze
4. **Market Structure** (2): Fair Value Gaps, Swing Points
5. **Crypto-Specific** (2): Funding Rate (stub), Open Interest (stub)

## Architecture

### Approach: Minimal Addition Pattern

**File structure:**
- Single file: `data/indicators.py` (grows from 182 to ~850 lines)
- New test file: `tests/test_indicators_extended.py` (~450 lines)
- Modified: `data/prompt_builder.py` (use `compute_all_indicators()`)

**Rationale:**
- Follows existing pure-function pattern (consistency)
- Minimal disruption to PromptBuilder integration
- TDD-friendly (each function independently testable)
- YAGNI: No premature abstraction into modules

## Function Signatures

### Group 1 - Trend Structure

```python
def donchian_channels(
    df: pd.DataFrame,
    period: int = 20
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Donchian Channels (upper, middle, lower).

    Upper = highest high over period
    Middle = (upper + lower) / 2
    Lower = lowest low over period

    Args:
        df: OHLCV DataFrame
        period: Lookback period (default 20)

    Returns:
        Tuple of (upper_channel, middle_channel, lower_channel)
    """

def donchian_channels_long(
    df: pd.DataFrame,
    period: int = 55
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Long-period Donchian Channels for trend identification.

    Alias of donchian_channels with default period=55.
    Commonly used for longer-term trend following.

    Args:
        df: OHLCV DataFrame
        period: Lookback period (default 55)

    Returns:
        Tuple of (upper_channel, middle_channel, lower_channel)
    """

def ichimoku_cloud(
    df: pd.DataFrame,
    tenkan_period: int = 9,
    kijun_period: int = 26,
    senkou_span_b_period: int = 52,
) -> dict[str, pd.Series]:
    """
    Ichimoku Cloud components.

    Components:
    - Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    - Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    - Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2, shifted forward 26
    - Senkou Span B (Leading Span B): (52-period high + low) / 2, shifted forward 26
    - Chikou Span (Lagging Span): Close shifted backward 26

    Args:
        df: OHLCV DataFrame
        tenkan_period: Conversion line period (default 9)
        kijun_period: Base line period (default 26)
        senkou_span_b_period: Leading Span B period (default 52)

    Returns:
        Dict with keys: tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span
    """

def kama(
    close: pd.Series,
    period: int = 10,
    fast_ema: int = 2,
    slow_ema: int = 30,
) -> pd.Series:
    """
    Kaufman Adaptive Moving Average.

    Adjusts smoothing based on market efficiency ratio:
    - High efficiency (trending) → fast smoothing (responsive)
    - Low efficiency (choppy) → slow smoothing (filters noise)

    Efficiency Ratio = |price_change| / sum(|bar_changes|)
    Smoothing Constant = [ER * (fast_sc - slow_sc) + slow_sc]^2
    KAMA = prev_KAMA + SC * (price - prev_KAMA)

    Args:
        close: Close price series
        period: Efficiency ratio period (default 10)
        fast_ema: Fast EMA constant (default 2)
        slow_ema: Slow EMA constant (default 30)

    Returns:
        KAMA series
    """
```

### Group 2 - Volume

```python
def obv(df: pd.DataFrame) -> pd.Series:
    """
    On-Balance Volume.

    Cumulative volume with direction based on close:
    - Add volume if close > previous close
    - Subtract volume if close < previous close
    - No change if close == previous close

    Args:
        df: OHLCV DataFrame with 'close' and 'volume' columns

    Returns:
        Cumulative OBV series
    """

def cmf(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    Chaikin Money Flow.

    Weighted average of money flow volume over period.

    Money Flow Multiplier = ((close - low) - (high - close)) / (high - low)
    Money Flow Volume = multiplier * volume
    CMF = sum(MFV over period) / sum(volume over period)

    Args:
        df: OHLCV DataFrame
        period: Lookback period (default 20)

    Returns:
        CMF series (range: -1.0 to +1.0)
    """

def mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Money Flow Index (volume-weighted RSI).

    Typical Price = (high + low + close) / 3
    Raw Money Flow = Typical Price * Volume

    Separate positive flow (TP increased) from negative flow (TP decreased)
    Money Ratio = sum(positive_flow) / sum(negative_flow)
    MFI = 100 - (100 / (1 + Money Ratio))

    Args:
        df: OHLCV DataFrame
        period: Lookback period (default 14)

    Returns:
        MFI series (range: 0-100)
    """

def vwap(df: pd.DataFrame) -> pd.Series:
    """
    Volume-Weighted Average Price.

    VWAP = cumulative(typical_price * volume) / cumulative(volume)
    Typical Price = (high + low + close) / 3

    Note: In production, VWAP resets at session boundaries.
    Here it's cumulative over the entire DataFrame window.

    Args:
        df: OHLCV DataFrame

    Returns:
        Cumulative VWAP series
    """
```

### Group 3 - Volatility

```python
def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Average True Range.

    True Range = max of:
    - high - low
    - |high - previous_close|
    - |low - previous_close|

    ATR = Wilder's smoothed average of True Range

    Args:
        df: OHLCV DataFrame
        period: Smoothing period (default 14)

    Returns:
        ATR series
    """

def bb_width(close: pd.Series, period: int = 20, num_std: float = 2.0) -> pd.Series:
    """
    Bollinger Band Width as percentage.

    Width = (upper_band - lower_band) / middle_band * 100

    Measures volatility expansion (high width) vs contraction (low width).
    Squeeze occurs when width is historically low.

    Args:
        close: Close price series
        period: BB period (default 20)
        num_std: Standard deviation multiplier (default 2.0)

    Returns:
        BB width as percentage series
    """

def keltner_channels(
    df: pd.DataFrame,
    ema_period: int = 20,
    atr_period: int = 10,
    atr_multiplier: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Keltner Channels (upper, middle, lower).

    Middle = EMA of close
    Upper = Middle + (ATR * multiplier)
    Lower = Middle - (ATR * multiplier)

    Similar to Bollinger Bands but uses ATR instead of standard deviation.
    More responsive to volatility changes.

    Args:
        df: OHLCV DataFrame
        ema_period: EMA period for middle line (default 20)
        atr_period: ATR period (default 10)
        atr_multiplier: ATR multiplier (default 2.0)

    Returns:
        Tuple of (upper_channel, middle_channel, lower_channel)
    """

def donchian_width(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    Donchian Channel Width as percentage.

    Width = (upper_channel - lower_channel) / middle_channel * 100

    Similar to BB Width but based on high/low extremes rather than
    standard deviation. Useful for breakout detection.

    Args:
        df: OHLCV DataFrame
        period: Donchian period (default 20)

    Returns:
        Donchian width as percentage series
    """

def ttm_squeeze(
    bb_upper: pd.Series,
    bb_lower: pd.Series,
    kc_upper: pd.Series,
    kc_lower: pd.Series,
) -> pd.Series:
    """
    TTM Squeeze indicator.

    Squeeze is ON when Bollinger Bands are inside Keltner Channels.
    This indicates very low volatility and potential for explosive move.

    Squeeze ON: BB completely inside KC (both bands)
    Squeeze OFF: BB outside KC

    Args:
        bb_upper: Bollinger Band upper series
        bb_lower: Bollinger Band lower series
        kc_upper: Keltner Channel upper series
        kc_lower: Keltner Channel lower series

    Returns:
        Boolean series: True when squeeze is ON, False when OFF
    """
```

### Group 4 - Market Structure

```python
def fair_value_gaps(
    df: pd.DataFrame,
    min_gap_pct: float = 0.1,
) -> list[dict]:
    """
    Fair Value Gaps (FVG) - Classic ICT definition.

    Three-candle pattern where candle 2's wick doesn't overlap
    with candle 1 and 3's bodies, indicating institutional order flow.

    Bullish FVG: Gap between candle1.high and candle3.low (candle2 between)
    Bearish FVG: Gap between candle1.low and candle3.high (candle2 between)

    Returns list of unfilled gaps only (price hasn't returned to fill).

    Args:
        df: OHLCV DataFrame
        min_gap_pct: Minimum gap size as % of price to filter noise (default 0.1%)

    Returns:
        List of dicts:
        [
            {
                'index': int,           # Index of gap candle (middle)
                'direction': 'bullish' | 'bearish',
                'gap_top': float,       # Top of gap zone
                'gap_bottom': float,    # Bottom of gap zone
                'gap_size_pct': float,  # Gap size as % of price
                'timestamp': int,       # Millisecond timestamp
            }
        ]
    """

def swing_points(
    df: pd.DataFrame,
    window: int = 5,
) -> dict[str, list[dict]]:
    """
    Swing high and swing low detection.

    Swing high: Local maximum where high > surrounding highs in window
    Swing low: Local minimum where low < surrounding lows in window

    Args:
        df: OHLCV DataFrame
        window: Window size for local extrema detection (default 5)

    Returns:
        Dict with keys 'highs' and 'lows':
        {
            'highs': [
                {
                    'index': int,
                    'price': float,
                    'timestamp': int,
                }
            ],
            'lows': [...]
        }
    """
```

### Group 5 - Crypto-Specific (Stubs)

```python
def funding_rate(symbol: str, timestamp_ms: int) -> float | None:
    """
    Funding rate for perpetual futures (stub).

    TODO: Implement by fetching from exchange API or local database.
    For now, returns None to avoid blocking indicator computation.

    Funding rate indicates long/short imbalance:
    - Positive: Longs pay shorts (bullish bias)
    - Negative: Shorts pay longs (bearish bias)

    Args:
        symbol: Trading pair (e.g., "BTC/USDT")
        timestamp_ms: Timestamp for historical lookup

    Returns:
        Funding rate as decimal (e.g., 0.0001 = 0.01%), or None
    """
    return None

def open_interest(symbol: str, timestamp_ms: int) -> float | None:
    """
    Open interest for perpetual futures (stub).

    TODO: Implement by fetching from exchange API or local database.
    For now, returns None to avoid blocking indicator computation.

    Open interest indicates total contracts outstanding:
    - Rising OI + rising price: Strong uptrend
    - Rising OI + falling price: Strong downtrend
    - Falling OI: Liquidations, trend weakening

    Args:
        symbol: Trading pair (e.g., "BTC/USDT")
        timestamp_ms: Timestamp for historical lookup

    Returns:
        Open interest in USD, or None
    """
    return None
```

## compute_all_indicators() Design

### Function Signature

```python
def compute_all_indicators(
    df: pd.DataFrame,
    include_volume: bool = True,
    include_structure: bool = True,
) -> dict:
    """
    Compute all technical indicators and return as dictionary.

    Computes scalar summaries (latest values) for prompt consumption
    and full series for advanced analysis.

    Args:
        df: OHLCV DataFrame with columns: timestamp, open, high, low, close, volume
        include_volume: Compute volume-based indicators (requires volume column)
        include_structure: Compute market structure indicators (FVG, swing points)

    Returns:
        Dictionary with indicator values and metadata

    Raises:
        ValueError: If df is empty or missing required columns
    """
```

### Output Structure

```python
{
    # ========== SCALAR OUTPUTS (for prompt consumption) ==========

    # Existing indicators
    'rsi': float,                    # Current RSI (0-100)
    'macd': float,                   # Current MACD line
    'macd_signal': float,            # Current MACD signal line
    'macd_histogram': float,         # Current MACD histogram
    'bb_upper': float,               # Current BB upper band price
    'bb_middle': float,              # Current BB middle band price
    'bb_lower': float,               # Current BB lower band price
    'bb_position': float,            # Current price position in BB (0.0-1.0)

    # Trend indicators
    'donchian_upper': float,         # Current Donchian upper (20-period)
    'donchian_middle': float,        # Current Donchian middle
    'donchian_lower': float,         # Current Donchian lower
    'donchian_long_upper': float,    # Current Donchian upper (55-period)
    'donchian_long_middle': float,   # Current Donchian middle
    'donchian_long_lower': float,    # Current Donchian lower
    'ichimoku_tenkan_sen': float,    # Current Tenkan-sen
    'ichimoku_kijun_sen': float,     # Current Kijun-sen
    'ichimoku_senkou_span_a': float, # Current Senkou Span A
    'ichimoku_senkou_span_b': float, # Current Senkou Span B
    'ichimoku_chikou_span': float,   # Current Chikou Span
    'kama': float,                   # Current KAMA

    # Volume indicators (None if include_volume=False)
    'obv': float | None,             # Current OBV
    'cmf': float | None,             # Current CMF (-1.0 to +1.0)
    'mfi': float | None,             # Current MFI (0-100)
    'vwap': float | None,            # Current VWAP price

    # Volatility indicators
    'atr': float,                    # Current ATR (absolute)
    'atr_normalized': float,         # Current ATR / close * 100 (as %)
    'bb_width': float,               # Current BB width (as %)
    'keltner_upper': float,          # Current Keltner upper
    'keltner_middle': float,         # Current Keltner middle
    'keltner_lower': float,          # Current Keltner lower
    'donchian_width': float,         # Current Donchian width (as %)
    'ttm_squeeze': bool,             # Current squeeze status (True = ON)

    # Market structure scalars (for prompt consumption)
    'nearest_bullish_fvg_pct': float | None,   # Distance to nearest bullish FVG (%)
    'nearest_bearish_fvg_pct': float | None,   # Distance to nearest bearish FVG (%)
    'open_fvg_count': int,                     # Count of unfilled FVGs
    'nearest_swing_high_pct': float | None,    # Distance to nearest swing high (%)
    'nearest_swing_low_pct': float | None,     # Distance to nearest swing low (%)

    # Crypto-specific (always None for now)
    'funding_rate': None,
    'open_interest': None,

    # ========== FULL SERIES (for advanced analysis) ==========

    'series': {
        'rsi': pd.Series,
        'macd': pd.Series,
        'macd_signal': pd.Series,
        'macd_histogram': pd.Series,
        'bb_upper': pd.Series,
        'bb_middle': pd.Series,
        'bb_lower': pd.Series,
        'bb_position': pd.Series,
        'bb_width': pd.Series,

        'donchian_upper': pd.Series,
        'donchian_middle': pd.Series,
        'donchian_lower': pd.Series,
        'donchian_long_upper': pd.Series,
        'donchian_long_middle': pd.Series,
        'donchian_long_lower': pd.Series,

        'ichimoku_tenkan_sen': pd.Series,
        'ichimoku_kijun_sen': pd.Series,
        'ichimoku_senkou_span_a': pd.Series,
        'ichimoku_senkou_span_b': pd.Series,
        'ichimoku_chikou_span': pd.Series,

        'kama': pd.Series,

        'obv': pd.Series | None,
        'cmf': pd.Series | None,
        'mfi': pd.Series | None,
        'vwap': pd.Series | None,

        'atr': pd.Series,
        'atr_normalized': pd.Series,
        'keltner_upper': pd.Series,
        'keltner_middle': pd.Series,
        'keltner_lower': pd.Series,
        'donchian_width': pd.Series,
        'ttm_squeeze': pd.Series,
    },

    # ========== RAW MARKET STRUCTURE DATA ==========

    'raw_fvgs': list[dict] | None,              # None if include_structure=False
    'raw_swing_points': dict | None,            # None if include_structure=False
}
```

### Scalar Summary Logic

**FVG Distance Calculation:**
```python
current_price = df['close'].iloc[-1]

# For each unfilled bullish FVG (gap_bottom > current_price):
distance_pct = (gap_bottom - current_price) / current_price * 100

# For each unfilled bearish FVG (gap_top < current_price):
distance_pct = (gap_top - current_price) / current_price * 100

# Return nearest (smallest absolute distance)
nearest_bullish_fvg_pct = min(distances) if bullish_fvgs else None
nearest_bearish_fvg_pct = max(distances) if bearish_fvgs else None  # negative value
```

**Swing Point Distance Calculation:**
```python
current_price = df['close'].iloc[-1]

# For swing highs above current price:
distance_pct = (swing_high - current_price) / current_price * 100

# For swing lows below current price:
distance_pct = (swing_low - current_price) / current_price * 100

# Return nearest
nearest_swing_high_pct = min(distances) if highs_above else None
nearest_swing_low_pct = max(distances) if lows_below else None  # negative value
```

## PromptBuilder Integration

### Import Changes

**Current:**
```python
from .indicators import compute_rsi, compute_macd, compute_bb_position
```

**New:**
```python
from .indicators import compute_all_indicators
```

### Usage Changes

**Current (in `build_prompt()`):**
```python
# Calculate common indicators
rsi_series = compute_rsi(df["close"])
rsi = rsi_series.iloc[-1] if not pd.isna(rsi_series.iloc[-1]) else 50.0

macd_line, macd_signal_series, _ = compute_macd(df["close"])
macd = macd_line.iloc[-1] if not pd.isna(macd_line.iloc[-1]) else 0.0
macd_signal = macd_signal_series.iloc[-1] if not pd.isna(macd_signal_series.iloc[-1]) else 0.0

bb_pos = compute_bb_position(df["close"]).iloc[-1]
bb_pos = bb_pos if not pd.isna(bb_pos) else 0.5
```

**New:**
```python
# Calculate all indicators once
indicators = compute_all_indicators(df, include_volume=True, include_structure=True)

# Access scalars directly
rsi = indicators['rsi']
macd = indicators['macd']
macd_signal = indicators['macd_signal']
bb_pos = indicators['bb_position']

# New indicators available:
atr_normalized = indicators['atr_normalized']
ttm_squeeze = indicators['ttm_squeeze']
nearest_swing_high_pct = indicators['nearest_swing_high_pct']
# ... etc
```

**Backward compatibility:** All existing indicator references continue to work, just accessed from dict instead of individual function calls.

## Testing Strategy

### Test File Structure

**New file:** `tests/test_indicators_extended.py`

**Test count:** Minimum 51 tests (42 per-indicator + 4 special + 5 integration)

### Test Categories

**1. Per-Indicator Tests (42 tests):**

Each of 14 main indicators gets 3 tests:
- Happy path: Standard calculation with sufficient data
- Edge case: Insufficient data / warmup period behavior
- Numerical stability: Extreme values / division by zero

**Examples:**

```python
# Donchian Channels
def test_donchian_channels_standard():
    """Standard Donchian calculation with 50 bars."""

def test_donchian_channels_insufficient_data():
    """Donchian with < period bars returns NaN."""

def test_donchian_channels_all_same_price():
    """Donchian with flat price returns same upper/middle/lower."""

# KAMA
def test_kama_trending_market():
    """KAMA in trending market stays close to fast EMA."""

def test_kama_choppy_market():
    """KAMA in choppy market smooths like slow EMA."""

def test_kama_warmup_period():
    """KAMA returns NaN until sufficient data."""

# Fair Value Gaps
def test_fvg_bullish_detection():
    """Detects bullish FVG when gap exists between candle1.high and candle3.low."""

def test_fvg_min_gap_filter():
    """Filters out gaps smaller than min_gap_pct."""

def test_fvg_filled_gaps_excluded():
    """Excludes FVGs that have been filled by subsequent price action."""
```

**2. Special Indicator Tests (4 tests):**

```python
def test_ttm_squeeze_on():
    """TTM Squeeze is ON when BB inside Keltner Channels."""

def test_ttm_squeeze_off():
    """TTM Squeeze is OFF when BB outside Keltner Channels."""

def test_funding_rate_stub():
    """funding_rate returns None (stub)."""

def test_open_interest_stub():
    """open_interest returns None (stub)."""
```

**3. Integration Tests (5 tests):**

```python
def test_compute_all_indicators_full():
    """Verify compute_all_indicators returns all expected keys."""

def test_compute_all_indicators_volume_flag():
    """Verify include_volume=False sets volume indicators to None."""

def test_compute_all_indicators_structure_flag():
    """Verify include_structure=False skips FVG/swing calculations."""

def test_compute_all_indicators_scalar_summaries():
    """Verify FVG and swing point distance calculations are correct."""

def test_compute_all_indicators_atr_normalized():
    """Verify atr_normalized = atr / close * 100."""
```

**4. Numerical Stability Tests (included in per-indicator tests):**

```python
def test_atr_zero_range_bars():
    """ATR handles bars with zero range (open=high=low=close)."""

def test_cmf_zero_range_bars():
    """CMF handles bars with zero range without division by zero."""

def test_mfi_zero_volume():
    """MFI handles zero volume bars gracefully."""
```

### Test Data Fixtures

**Reuse existing fixtures:**
- `sample_ohlcv_df` from `tests/test_indicators.py`

**Add new fixtures:**
```python
@pytest.fixture
def trending_ohlcv_df():
    """OHLCV data with clear uptrend for KAMA/Ichimoku testing."""

@pytest.fixture
def choppy_ohlcv_df():
    """OHLCV data with sideways chop for squeeze detection."""

@pytest.fixture
def fvg_pattern_df():
    """OHLCV data with known bullish and bearish FVG patterns."""
```

## Implementation Phases

### Phase 1: Core Indicator Functions (Groups 1-3)
- Implement 12 main indicator functions (trend, volume, volatility)
- Implement 1 TTM Squeeze function
- Implement 2 crypto stub functions
- Write 46 tests (36 main + 2 TTM + 2 stubs + 6 numerical stability)
- Commit after all tests pass

### Phase 2: Market Structure Functions (Group 4)
- Implement FVG detection
- Implement swing point detection
- Write 6 tests (3 per function)
- Commit after all tests pass

### Phase 3: Aggregation Function
- Implement `compute_all_indicators()`
- Implement scalar summary calculations (FVG/swing distances)
- Implement `atr_normalized` calculation
- Write 5 integration tests
- Commit after all tests pass

### Phase 4: PromptBuilder Integration
- Update imports in `data/prompt_builder.py`
- Replace individual indicator calls with `compute_all_indicators()`
- Use `atr_normalized` instead of raw `atr` in prompts
- Update template rendering to use new indicators (optional enhancement)
- Verify existing tests still pass
- Commit

## Success Criteria

1. **All 17 indicator functions implemented** with correct mathematical definitions
2. **Minimum 51 tests passing** (42 per-indicator + 4 special + 5 integration)
3. **No external dependencies added** (pandas/numpy only)
4. **Backward compatibility maintained** for existing PromptBuilder functionality
5. **Scalar summaries accurate** for FVG/swing point distance calculations
6. **Volume flag works** (`include_volume=False` sets volume indicators to None)
7. **Structure flag works** (`include_structure=False` skips FVG/swing calculations)
8. **ATR normalized added** (`atr_normalized = atr / close * 100`)
9. **PromptBuilder uses** `atr_normalized` instead of raw `atr`

## Non-Goals

- Paid indicator libraries (TA-Lib, pandas-ta)
- Optimization/vectorization beyond pandas native performance
- Custom indicator visualization
- Indicator parameter tuning/optimization
- Implementing crypto stubs (funding_rate, open_interest remain None)

## Future Enhancements

- Implement `funding_rate()` by fetching from Binance API
- Implement `open_interest()` by fetching from Binance API
- Add indicator caching for performance (if needed)
- Add more exotic indicators (Volume Profile, Market Profile)
- Expand prompt templates to use new indicators
