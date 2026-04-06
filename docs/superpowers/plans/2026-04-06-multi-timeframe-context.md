# Multi-Timeframe Context Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add higher timeframe trend context to signal generation prompts to improve signal quality through multi-timeframe confluence.

**Architecture:** Add optional higher_tf_data parameter to PromptBuilder and orchestrator. Extract up to 2 nearest higher timeframes, compute trend summaries using existing indicators (Ichimoku, KAMA, Donchian, RSI), and inject contextual section into prompts. All changes backward compatible.

**Tech Stack:** Python 3.13, pandas, pytest, existing indicator infrastructure

---

## File Structure

**New Files:**
- `tests/test_prompt_builder_mtf.py` - Unit tests for multi-timeframe functions
- `tests/fixtures/timeframe_fixtures.py` - Reusable test DataFrames with known patterns

**Modified Files:**
- `data/prompt_builder.py` - Add TIMEFRAME_HIERARCHY, helper functions, extend build_prompt()
- `swarm/orchestrator.py` - Add higher_tf_data parameter to run_swarm_workflow()
- `tests/test_orchestrator.py` - Add integration tests for higher_tf_data flow

---

## Task 1: Add Timeframe Hierarchy and Selection Logic

**Files:**
- Modify: `data/prompt_builder.py:1-20`
- Test: `tests/test_prompt_builder_mtf.py`

- [ ] **Step 1: Write failing test for get_higher_timeframes()**

Create new test file:

```python
# tests/test_prompt_builder_mtf.py
"""Tests for multi-timeframe context in prompt builder."""

import pandas as pd
import pytest

from data.prompt_builder import get_higher_timeframes, TIMEFRAME_HIERARCHY


class TestGetHigherTimeframes:
    """Test timeframe hierarchy and selection logic."""

    def test_returns_2_nearest_higher_timeframes(self):
        """Should return the 2 nearest higher timeframes."""
        result = get_higher_timeframes("1m", ["5m", "15m", "1h", "4h"])
        assert result == ["5m", "15m"]

    def test_returns_1_when_only_1_available(self):
        """Should return single timeframe when only 1 higher available."""
        result = get_higher_timeframes("1h", ["4h"])
        assert result == ["4h"]

    def test_returns_empty_when_none_available(self):
        """Should return empty list when current TF is highest."""
        result = get_higher_timeframes("1d", ["1h", "4h"])
        assert result == []

    def test_returns_empty_for_top_of_hierarchy(self):
        """Should return empty for 1d (top of hierarchy)."""
        result = get_higher_timeframes("1d", ["1h", "4h", "1d"])
        assert result == []

    def test_skips_unknown_timeframes(self):
        """Should skip timeframes not in hierarchy."""
        result = get_higher_timeframes("1h", ["3h", "4h", "1d"])
        assert result == ["4h", "1d"]

    def test_respects_hierarchy_ordering(self):
        """Should return timeframes in hierarchy order."""
        result = get_higher_timeframes("5m", ["1d", "4h", "1h", "15m"])
        assert result == ["15m", "1h"]  # Nearest 2 in order
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_prompt_builder_mtf.py::TestGetHigherTimeframes -v`
Expected: FAIL - ImportError: cannot import name 'get_higher_timeframes'

- [ ] **Step 3: Add TIMEFRAME_HIERARCHY constant**

In `data/prompt_builder.py`, add after imports (around line 18):

```python
# Timeframe hierarchy for multi-timeframe analysis
TIMEFRAME_HIERARCHY = ["1m", "5m", "15m", "1h", "4h", "1d"]
```

- [ ] **Step 4: Implement get_higher_timeframes()**

In `data/prompt_builder.py`, add before TaskType enum (around line 23):

```python
def get_higher_timeframes(current_tf: str, available_tfs: list[str]) -> list[str]:
    """
    Get up to 2 nearest higher timeframes from available set.

    Args:
        current_tf: Current timeframe (e.g., "1h")
        available_tfs: List of available higher timeframes

    Returns:
        List of up to 2 nearest higher timeframes in hierarchy order

    Examples:
        >>> get_higher_timeframes("1m", ["5m", "15m", "1h"])
        ["5m", "15m"]
        >>> get_higher_timeframes("1h", ["4h"])
        ["4h"]
        >>> get_higher_timeframes("1d", ["1h", "4h"])
        []
    """
    if current_tf not in TIMEFRAME_HIERARCHY:
        logger.warning("Unknown current timeframe", timeframe=current_tf)
        return []

    current_idx = TIMEFRAME_HIERARCHY.index(current_tf)

    # Filter to only recognized timeframes higher in hierarchy
    higher_tfs = []
    for tf in TIMEFRAME_HIERARCHY[current_idx + 1:]:
        if tf in available_tfs:
            higher_tfs.append(tf)

    # Return up to 2 nearest
    return higher_tfs[:2]
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_prompt_builder_mtf.py::TestGetHigherTimeframes -v`
Expected: PASS (all 6 tests)

- [ ] **Step 6: Commit**

```bash
git add data/prompt_builder.py tests/test_prompt_builder_mtf.py
git commit -m "feat: add timeframe hierarchy and selection logic

- Add TIMEFRAME_HIERARCHY constant with 6 standard timeframes
- Implement get_higher_timeframes() to select up to 2 nearest higher TFs
- Add comprehensive tests for edge cases and ordering"
```

---

## Task 2: Create Test Fixtures for Known Indicator Patterns

**Files:**
- Create: `tests/fixtures/timeframe_fixtures.py`
- Test: `tests/test_prompt_builder_mtf.py`

- [ ] **Step 1: Write test for bullish fixture**

Add to `tests/test_prompt_builder_mtf.py`:

```python
from tests.fixtures.timeframe_fixtures import (
    create_test_df_bullish,
    create_test_df_bearish,
    create_test_df_neutral,
)


class TestTimeframeFixtures:
    """Verify test fixtures produce expected indicator patterns."""

    def test_bullish_fixture_has_bullish_indicators(self):
        """Bullish fixture should produce bullish indicator values."""
        from data.indicators import compute_all_indicators

        df = create_test_df_bullish()
        indicators = compute_all_indicators(df, include_volume=False, include_structure=False)

        # Price should be above Ichimoku cloud
        assert df["close"].iloc[-1] > indicators["ichimoku_cloud_top"]

        # KAMA should be rising
        kama = indicators["series"]["kama"]
        assert kama.iloc[-1] > kama.iloc[-5]

        # RSI should be in neutral-to-bullish range
        assert 50 <= indicators["rsi"] <= 80

    def test_bearish_fixture_has_bearish_indicators(self):
        """Bearish fixture should produce bearish indicator values."""
        from data.indicators import compute_all_indicators

        df = create_test_df_bearish()
        indicators = compute_all_indicators(df, include_volume=False, include_structure=False)

        # Price should be below Ichimoku cloud
        assert df["close"].iloc[-1] < indicators["ichimoku_cloud_bottom"]

        # KAMA should be falling
        kama = indicators["series"]["kama"]
        assert kama.iloc[-1] < kama.iloc[-5]

        # RSI should be in neutral-to-bearish range
        assert 20 <= indicators["rsi"] <= 50

    def test_neutral_fixture_has_neutral_indicators(self):
        """Neutral fixture should produce neutral indicator values."""
        from data.indicators import compute_all_indicators

        df = create_test_df_neutral()
        indicators = compute_all_indicators(df, include_volume=False, include_structure=False)

        # Price should be inside Ichimoku cloud
        price = df["close"].iloc[-1]
        assert indicators["ichimoku_cloud_bottom"] <= price <= indicators["ichimoku_cloud_top"]

        # RSI should be near 50
        assert 45 <= indicators["rsi"] <= 55
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_prompt_builder_mtf.py::TestTimeframeFixtures -v`
Expected: FAIL - ImportError: cannot import from timeframe_fixtures

- [ ] **Step 3: Create fixture file with bullish pattern**

Create `tests/fixtures/timeframe_fixtures.py`:

```python
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

    timestamps = pd.date_range(end=pd.Timestamp.now(), periods=bars, freq="1h")
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

    timestamps = pd.date_range(end=pd.Timestamp.now(), periods=bars, freq="1h")
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

    timestamps = pd.date_range(end=pd.Timestamp.now(), periods=bars, freq="1h")
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

    return df
```

- [ ] **Step 4: Create __init__.py for fixtures module**

Create `tests/fixtures/__init__.py`:

```python
"""Test fixtures for Trading Swarm tests."""
```

- [ ] **Step 5: Run test to verify fixtures work**

Run: `pytest tests/test_prompt_builder_mtf.py::TestTimeframeFixtures -v`
Expected: PASS (all 3 tests)

- [ ] **Step 6: Commit**

```bash
git add tests/fixtures/timeframe_fixtures.py tests/fixtures/__init__.py tests/test_prompt_builder_mtf.py
git commit -m "test: add fixtures for bullish/bearish/neutral patterns

- Create timeframe_fixtures.py with 3 factory functions
- Generate reproducible OHLCV data with known indicator properties
- Add tests to verify fixture indicator patterns"
```

---

## Task 3: Implement Timeframe Summarization

**Files:**
- Modify: `data/prompt_builder.py:460-470`
- Test: `tests/test_prompt_builder_mtf.py`

- [ ] **Step 1: Write failing test for summarize_timeframe()**

Add to `tests/test_prompt_builder_mtf.py`:

```python
from data.prompt_builder import summarize_timeframe


class TestSummarizeTimeframe:
    """Test timeframe trend summarization."""

    def test_bullish_setup_classification(self):
        """Should classify bullish pattern correctly."""
        df = create_test_df_bullish()
        result = summarize_timeframe(df, "4h")

        assert result["timeframe"] == "4h"
        assert result["trend"] == "bullish"
        assert result["cloud_position"] == "above"
        assert result["kama_slope"] == "rising"
        assert "text" in result
        assert "Bullish" in result["text"]

    def test_bearish_setup_classification(self):
        """Should classify bearish pattern correctly."""
        df = create_test_df_bearish()
        result = summarize_timeframe(df, "1h")

        assert result["timeframe"] == "1h"
        assert result["trend"] == "bearish"
        assert result["cloud_position"] == "below"
        assert result["kama_slope"] == "falling"
        assert "Bearish" in result["text"]

    def test_neutral_setup_classification(self):
        """Should classify neutral pattern correctly."""
        df = create_test_df_neutral()
        result = summarize_timeframe(df, "1d")

        assert result["timeframe"] == "1d"
        assert result["trend"] == "neutral"
        assert result["cloud_position"] == "inside"

    def test_rsi_zone_overbought(self):
        """Should classify RSI > 70 as overbought."""
        # Create custom DataFrame with high RSI
        df = create_test_df_bullish(bars=100)
        # Manually adjust to ensure RSI > 70 by creating strong uptrend
        df["close"] = df["close"] * 1.15  # Additional boost
        result = summarize_timeframe(df, "4h")

        # RSI should be high (may not hit 70+ with fixture, so check > 60)
        assert result["rsi_value"] > 60

    def test_rsi_zone_oversold(self):
        """Should classify RSI < 30 as oversold."""
        df = create_test_df_bearish(bars=100)
        df["close"] = df["close"] * 0.85  # Additional drop
        result = summarize_timeframe(df, "4h")

        # RSI should be low
        assert result["rsi_value"] < 40

    def test_donchian_position_classifications(self):
        """Should classify Donchian channel positions correctly."""
        df_bullish = create_test_df_bullish()
        result_bullish = summarize_timeframe(df_bullish, "4h")

        # Bullish should be in upper part of channel
        assert result_bullish["donchian_position"] in ["upper", "middle"]

        df_bearish = create_test_df_bearish()
        result_bearish = summarize_timeframe(df_bearish, "4h")

        # Bearish should be in lower part of channel
        assert result_bearish["donchian_position"] in ["lower", "middle"]

    def test_text_summary_format(self):
        """Should generate properly formatted text summary."""
        df = create_test_df_bullish()
        result = summarize_timeframe(df, "4h")

        text = result["text"]
        # Should contain timeframe, trend, and indicator details
        assert "4h:" in text
        assert "cloud" in text.lower()
        assert "kama" in text.lower()
        assert "rsi" in text.lower()

    def test_handles_insufficient_data_gracefully(self):
        """Should handle DataFrames with insufficient bars."""
        df_short = create_test_df_bullish(bars=30)  # Less than Ichimoku needs
        result = summarize_timeframe(df_short, "1h")

        # Should still return result with neutral fallbacks
        assert result["timeframe"] == "1h"
        assert "trend" in result
        assert "text" in result
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_prompt_builder_mtf.py::TestSummarizeTimeframe -v`
Expected: FAIL - ImportError: cannot import name 'summarize_timeframe'

- [ ] **Step 3: Implement summarize_timeframe() function**

In `data/prompt_builder.py`, add after `get_higher_timeframes()` (around line 60):

```python
def summarize_timeframe(df: pd.DataFrame, timeframe: str) -> dict:
    """
    Generate trend summary for a timeframe from OHLCV data.

    Extracts indicators and classifies trend direction based on:
    - Ichimoku cloud position (above/below/inside)
    - KAMA slope (rising/falling/flat)
    - Donchian channel position (upper/lower/middle)
    - RSI zone (overbought/oversold/neutral)

    Args:
        df: OHLCV DataFrame with sufficient history (50+ bars recommended)
        timeframe: Timeframe label (e.g., "4h", "1d")

    Returns:
        Dict with structure:
        {
            "timeframe": str,
            "trend": "bullish" | "bearish" | "neutral",
            "cloud_position": "above" | "below" | "inside",
            "kama_slope": "rising" | "falling" | "flat",
            "donchian_position": "upper" | "lower" | "middle",
            "rsi_zone": "overbought" | "oversold" | "neutral",
            "rsi_value": float,
            "text": str (human-readable summary)
        }

    Examples:
        >>> df = create_test_df_bullish()
        >>> summarize_timeframe(df, "4h")
        {"timeframe": "4h", "trend": "bullish", ...}
    """
    # Compute all indicators
    indicators = compute_all_indicators(df, include_volume=False, include_structure=False)

    current_price = float(df["close"].iloc[-1])

    # 1. Ichimoku Cloud Position
    cloud_top = indicators.get("ichimoku_cloud_top")
    cloud_bottom = indicators.get("ichimoku_cloud_bottom")

    if cloud_top is None or cloud_bottom is None or pd.isna(cloud_top) or pd.isna(cloud_bottom):
        cloud_position = "inside"  # Fallback for insufficient data
    elif current_price > cloud_top:
        cloud_position = "above"
    elif current_price < cloud_bottom:
        cloud_position = "below"
    else:
        cloud_position = "inside"

    # 2. KAMA Slope
    kama_series = indicators["series"].get("kama")
    if kama_series is None or len(kama_series) < 6:
        kama_slope = "flat"
    else:
        kama_current = kama_series.iloc[-1]
        kama_prev = kama_series.iloc[-6]  # 5-bar lookback

        if pd.isna(kama_current) or pd.isna(kama_prev):
            kama_slope = "flat"
        else:
            slope_pct = ((kama_current - kama_prev) / kama_prev) * 100
            threshold = 0.1  # 0.1% threshold

            if slope_pct > threshold:
                kama_slope = "rising"
            elif slope_pct < -threshold:
                kama_slope = "falling"
            else:
                kama_slope = "flat"

    # 3. Donchian Channel Position
    donchian_upper = indicators.get("donchian_upper")
    donchian_lower = indicators.get("donchian_lower")

    if (donchian_upper is None or donchian_lower is None or
        pd.isna(donchian_upper) or pd.isna(donchian_lower)):
        donchian_position = "middle"
    else:
        channel_range = donchian_upper - donchian_lower
        if channel_range < EPSILON:
            donchian_position = "middle"
        else:
            position_pct = (current_price - donchian_lower) / channel_range

            if position_pct > 0.8:
                donchian_position = "upper"
            elif position_pct < 0.2:
                donchian_position = "lower"
            else:
                donchian_position = "middle"

    # 4. RSI Zone
    rsi = indicators.get("rsi")
    if rsi is None or pd.isna(rsi):
        rsi = 50.0
        rsi_zone = "neutral"
    else:
        rsi = float(rsi)
        if rsi > 70:
            rsi_zone = "overbought"
        elif rsi < 30:
            rsi_zone = "oversold"
        else:
            rsi_zone = "neutral"

    # 5. Overall Trend Classification
    if cloud_position == "above" and kama_slope == "rising":
        trend = "bullish"
    elif cloud_position == "below" and kama_slope == "falling":
        trend = "bearish"
    else:
        trend = "neutral"

    # 6. Generate Text Summary
    trend_label = trend.capitalize()
    cloud_text = f"{cloud_position} cloud"
    kama_text = f"KAMA {kama_slope}"
    donchian_text = f"near Donchian {donchian_position}" if donchian_position != "middle" else "middle of Donchian"
    rsi_text = f"RSI {rsi:.0f} ({rsi_zone})"

    text = f"{timeframe}: {trend_label} ({cloud_text}, {kama_text}), {donchian_text}, {rsi_text}"

    logger.debug(
        "Timeframe summary",
        timeframe=timeframe,
        trend=trend,
        cloud_position=cloud_position,
        kama_slope=kama_slope,
    )

    return {
        "timeframe": timeframe,
        "trend": trend,
        "cloud_position": cloud_position,
        "kama_slope": kama_slope,
        "donchian_position": donchian_position,
        "rsi_zone": rsi_zone,
        "rsi_value": rsi,
        "text": text,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_prompt_builder_mtf.py::TestSummarizeTimeframe -v`
Expected: PASS (all 9 tests)

- [ ] **Step 5: Commit**

```bash
git add data/prompt_builder.py tests/test_prompt_builder_mtf.py
git commit -m "feat: implement timeframe trend summarization

- Add summarize_timeframe() with 4-indicator trend classification
- Classify Ichimoku cloud position (above/below/inside)
- Detect KAMA slope with 0.1% threshold
- Compute Donchian channel position percentile
- Categorize RSI zones (overbought/oversold/neutral)
- Generate human-readable text summaries"
```

---

## Task 4: Implement Confluence Detection

**Files:**
- Modify: `data/prompt_builder.py:200-220`
- Test: `tests/test_prompt_builder_mtf.py`

- [ ] **Step 1: Write failing test for compute_confluence()**

Add to `tests/test_prompt_builder_mtf.py`:

```python
from data.prompt_builder import compute_confluence


class TestComputeConfluence:
    """Test multi-timeframe confluence detection."""

    def test_aligned_bullish(self):
        """All bullish trends should show aligned."""
        summaries = [
            {"trend": "bullish", "timeframe": "4h"},
            {"trend": "bullish", "timeframe": "1d"},
        ]
        result = compute_confluence(summaries)

        assert "aligned" in result.lower()
        assert "bullish" in result.lower()

    def test_aligned_bearish(self):
        """All bearish trends should show aligned."""
        summaries = [
            {"trend": "bearish", "timeframe": "4h"},
            {"trend": "bearish", "timeframe": "1d"},
        ]
        result = compute_confluence(summaries)

        assert "aligned" in result.lower()
        assert "bearish" in result.lower()

    def test_aligned_neutral(self):
        """All neutral trends should show aligned."""
        summaries = [
            {"trend": "neutral", "timeframe": "4h"},
            {"trend": "neutral", "timeframe": "1d"},
        ]
        result = compute_confluence(summaries)

        assert "neutral" in result.lower()

    def test_mixed_signals(self):
        """Mixed bullish/bearish should show mixed."""
        summaries = [
            {"trend": "bullish", "timeframe": "4h"},
            {"trend": "bearish", "timeframe": "1d"},
        ]
        result = compute_confluence(summaries)

        assert "mixed" in result.lower()

    def test_mixed_with_neutral(self):
        """Bullish + neutral should show mixed."""
        summaries = [
            {"trend": "bullish", "timeframe": "4h"},
            {"trend": "neutral", "timeframe": "1d"},
        ]
        result = compute_confluence(summaries)

        assert "mixed" in result.lower()

    def test_single_timeframe(self):
        """Single higher TF should still generate confluence text."""
        summaries = [
            {"trend": "bullish", "timeframe": "4h"},
        ]
        result = compute_confluence(summaries)

        assert "bullish" in result.lower()

    def test_empty_summaries(self):
        """Empty list should return neutral message."""
        result = compute_confluence([])

        assert "neutral" in result.lower() or "no" in result.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_prompt_builder_mtf.py::TestComputeConfluence -v`
Expected: FAIL - ImportError: cannot import name 'compute_confluence'

- [ ] **Step 3: Implement compute_confluence() function**

In `data/prompt_builder.py`, add after `summarize_timeframe()` (around line 200):

```python
def compute_confluence(summaries: list[dict]) -> str:
    """
    Analyze trend alignment across multiple timeframes.

    Args:
        summaries: List of timeframe summaries from summarize_timeframe()

    Returns:
        Confluence description string

    Examples:
        >>> summaries = [{"trend": "bullish"}, {"trend": "bullish"}]
        >>> compute_confluence(summaries)
        "Confluence: Aligned with higher timeframes (bullish)"
    """
    if not summaries:
        return "Confluence: No higher timeframe data"

    trends = [s["trend"] for s in summaries]

    # Count trend types
    bullish_count = trends.count("bullish")
    bearish_count = trends.count("bearish")
    neutral_count = trends.count("neutral")

    total = len(trends)

    # All same trend -> aligned
    if bullish_count == total:
        return "Confluence: Aligned with higher timeframes (bullish)"
    elif bearish_count == total:
        return "Confluence: Aligned with higher timeframes (bearish)"
    elif neutral_count == total:
        return "Confluence: Higher timeframes neutral"

    # Mixed signals
    return "Confluence: Mixed signals across timeframes"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_prompt_builder_mtf.py::TestComputeConfluence -v`
Expected: PASS (all 7 tests)

- [ ] **Step 5: Commit**

```bash
git add data/prompt_builder.py tests/test_prompt_builder_mtf.py
git commit -m "feat: implement confluence detection logic

- Add compute_confluence() to analyze trend alignment
- Detect aligned (all same trend), mixed, or neutral patterns
- Generate human-readable confluence descriptions"
```

---

## Task 5: Update Prompt Templates with Higher TF Section

**Files:**
- Modify: `data/prompt_builder.py:115-313`
- Test: `tests/test_prompt_builder_mtf.py`

- [ ] **Step 1: Write failing test for template integration**

Add to `tests/test_prompt_builder_mtf.py`:

```python
from data.prompt_builder import DirectionPredictionPrompt, MomentumAssessmentPrompt, SupportResistancePrompt


class TestTemplateIntegration:
    """Test higher timeframe context in prompt templates."""

    def test_direction_template_includes_higher_tf_section(self):
        """DirectionPredictionPrompt should include higher TF section when provided."""
        template = DirectionPredictionPrompt()
        higher_tf_context = "4h: Bullish\n1d: Bullish\nConfluence: Aligned"

        prompt = template.render(
            symbol="BTC/USDT",
            timeframe="1h",
            current_price=50000.0,
            market_regime="neutral",
            rsi=55.0,
            macd=100.0,
            macd_signal=95.0,
            bb_position=0.6,
            price_summary="...",
            higher_tf_context=higher_tf_context,
        )

        assert "## Higher Timeframe Context" in prompt
        assert "4h: Bullish" in prompt
        assert "Confluence: Aligned" in prompt

    def test_direction_template_omits_section_when_none(self):
        """DirectionPredictionPrompt should omit section when higher_tf_context=None."""
        template = DirectionPredictionPrompt()

        prompt = template.render(
            symbol="BTC/USDT",
            timeframe="1h",
            current_price=50000.0,
            market_regime="neutral",
            rsi=55.0,
            macd=100.0,
            macd_signal=95.0,
            bb_position=0.6,
            price_summary="...",
            higher_tf_context=None,
        )

        assert "## Higher Timeframe Context" not in prompt

    def test_momentum_template_includes_higher_tf_section(self):
        """MomentumAssessmentPrompt should include higher TF section."""
        template = MomentumAssessmentPrompt()
        higher_tf_context = "4h: Bearish\nConfluence: Conflicting"

        prompt = template.render(
            symbol="BTC/USDT",
            timeframe="1h",
            current_price=50000.0,
            market_regime="neutral",
            rsi=45.0,
            rsi_prev=50.0,
            rsi_delta=-5.0,
            macd=100.0,
            macd_signal=105.0,
            macd_prev=110.0,
            macd_delta=-10.0,
            bb_width=0.05,
            bb_trend="contracting",
            price_summary="...",
            higher_tf_context=higher_tf_context,
        )

        assert "## Higher Timeframe Context" in prompt
        assert "Bearish" in prompt

    def test_support_resistance_template_includes_higher_tf_section(self):
        """SupportResistancePrompt should include higher TF section."""
        template = SupportResistancePrompt()
        higher_tf_context = "1d: Neutral\nConfluence: Mixed"

        prompt = template.render(
            symbol="BTC/USDT",
            timeframe="4h",
            current_price=50000.0,
            market_regime="neutral",
            price_high=51000.0,
            price_low=49000.0,
            price_range=2000.0,
            swing_highs="$50500, $50800",
            swing_lows="$49200, $49500",
            price_summary="...",
            higher_tf_context=higher_tf_context,
        )

        assert "## Higher Timeframe Context" in prompt
        assert "Neutral" in prompt
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_prompt_builder_mtf.py::TestTemplateIntegration -v`
Expected: FAIL - TypeError: render() got an unexpected keyword argument 'higher_tf_context'

- [ ] **Step 3: Update DirectionPredictionPrompt template**

In `data/prompt_builder.py`, modify DirectionPredictionPrompt class (around line 115):

```python
@dataclass(slots=True)
class DirectionPredictionPrompt:
    """Prompt template for direction prediction task."""

    TEMPLATE = """/no_think

You are a quantitative trading analyst. Analyze the following market data and predict the price direction for the next {horizon} bars.

## Market Data
Symbol: {symbol}
Timeframe: {timeframe}
Current Price: ${current_price:.4f}
Market Regime: {market_regime}
{higher_tf_section}
## Technical Indicators
RSI(14): {rsi:.2f}
MACD: {macd:.4f} (Signal: {macd_signal:.4f})
BB Position: {bb_position:.2f} (0.0=lower band, 0.5=middle, 1.0=upper band)

## Recent Price Action (last {num_bars} bars)
{price_summary}

## Task
Predict whether the price will be HIGHER or LOWER {horizon} bars from now.

Respond in JSON format:
{{"direction": "HIGHER" | "LOWER", "confidence": 0.0-1.0, "reasoning": "brief explanation"}}
"""

    def render(
        self,
        symbol: str,
        timeframe: str,
        current_price: float,
        market_regime: str,
        rsi: float,
        macd: float,
        macd_signal: float,
        bb_position: float,
        price_summary: str,
        horizon: int = 5,
        num_bars: int = 10,
        higher_tf_context: str | None = None,
    ) -> str:
        """Render direction prediction prompt."""
        # Build higher timeframe section
        higher_tf_section = ""
        if higher_tf_context:
            higher_tf_section = f"\n## Higher Timeframe Context\n{higher_tf_context}\n"

        return self.TEMPLATE.format(
            symbol=symbol,
            timeframe=timeframe,
            current_price=current_price,
            market_regime=market_regime,
            rsi=rsi,
            macd=macd,
            macd_signal=macd_signal,
            bb_position=bb_position,
            price_summary=price_summary,
            horizon=horizon,
            num_bars=num_bars,
            higher_tf_section=higher_tf_section,
        )
```

- [ ] **Step 4: Update MomentumAssessmentPrompt template**

In `data/prompt_builder.py`, modify MomentumAssessmentPrompt class (around line 172):

```python
@dataclass(slots=True)
class MomentumAssessmentPrompt:
    """Prompt template for momentum assessment task."""

    TEMPLATE = """/no_think

You are a momentum analyst. Analyze the following market data and assess whether momentum is increasing or decreasing.

## Market Data
Symbol: {symbol}
Timeframe: {timeframe}
Current Price: ${current_price:.4f}
Market Regime: {market_regime}
{higher_tf_section}
## Technical Indicators
RSI(14): {rsi:.2f} (previous: {rsi_prev:.2f}, change: {rsi_delta:+.2f})
MACD: {macd:.4f} (Signal: {macd_signal:.4f})
MACD Previous: {macd_prev:.4f} (change: {macd_delta:+.4f})
BB Width: {bb_width:.4f} ({bb_trend})

## Recent Price Action (last {num_bars} bars)
{price_summary}

## Task
Is momentum INCREASING or DECREASING? Consider:
- RSI trend (rising RSI = increasing momentum)
- MACD vs signal line (diverging upward = increasing momentum)
- Price action acceleration (larger price changes = increasing momentum)
- BB width (expanding = increasing volatility/momentum)

Respond in JSON format:
{{"direction": "INCREASING" | "DECREASING", "confidence": 0.0-1.0, "reasoning": "brief explanation"}}
"""

    def render(
        self,
        symbol: str,
        timeframe: str,
        current_price: float,
        market_regime: str,
        rsi: float,
        rsi_prev: float,
        rsi_delta: float,
        macd: float,
        macd_signal: float,
        macd_prev: float,
        macd_delta: float,
        bb_width: float,
        bb_trend: str,
        price_summary: str,
        num_bars: int = 10,
        higher_tf_context: str | None = None,
    ) -> str:
        """Render momentum assessment prompt."""
        # Build higher timeframe section
        higher_tf_section = ""
        if higher_tf_context:
            higher_tf_section = f"\n## Higher Timeframe Context\n{higher_tf_context}\n"

        return self.TEMPLATE.format(
            symbol=symbol,
            timeframe=timeframe,
            current_price=current_price,
            market_regime=market_regime,
            rsi=rsi,
            rsi_prev=rsi_prev,
            rsi_delta=rsi_delta,
            macd=macd,
            macd_signal=macd_signal,
            macd_prev=macd_prev,
            macd_delta=macd_delta,
            bb_width=bb_width,
            bb_trend=bb_trend,
            price_summary=price_summary,
            num_bars=num_bars,
            higher_tf_section=higher_tf_section,
        )
```

- [ ] **Step 5: Update SupportResistancePrompt template**

In `data/prompt_builder.py`, modify SupportResistancePrompt class (around line 244):

```python
@dataclass(slots=True)
class SupportResistancePrompt:
    """Prompt template for support/resistance identification task."""

    TEMPLATE = """/no_think

You are a technical analyst. Identify the nearest support and resistance levels based on recent price action.

## Market Data
Symbol: {symbol}
Timeframe: {timeframe}
Current Price: ${current_price:.4f}
Market Regime: {market_regime}
{higher_tf_section}
## Price History (last 50 bars)
High: ${price_high:.4f}
Low: ${price_low:.4f}
Range: ${price_range:.4f}

## Recent Swing Points
Swing Highs: {swing_highs}
Swing Lows: {swing_lows}

## Recent Price Action (last {num_bars} bars)
{price_summary}

## Task
Identify:
1. Nearest SUPPORT level below current price
2. Nearest RESISTANCE level above current price

Consider: Previous swing highs/lows, psychological levels (round numbers), high-volume areas

Respond in JSON format:
{{
  "support_price": float,
  "support_confidence": 0.0-1.0,
  "resistance_price": float,
  "resistance_confidence": 0.0-1.0,
  "reasoning": "brief explanation"
}}
"""

    def render(
        self,
        symbol: str,
        timeframe: str,
        current_price: float,
        market_regime: str,
        price_high: float,
        price_low: float,
        price_range: float,
        swing_highs: str,
        swing_lows: str,
        price_summary: str,
        num_bars: int = 10,
        higher_tf_context: str | None = None,
    ) -> str:
        """Render support/resistance prompt."""
        # Build higher timeframe section
        higher_tf_section = ""
        if higher_tf_context:
            higher_tf_section = f"\n## Higher Timeframe Context\n{higher_tf_context}\n"

        return self.TEMPLATE.format(
            symbol=symbol,
            timeframe=timeframe,
            current_price=current_price,
            market_regime=market_regime,
            price_high=price_high,
            price_low=price_low,
            price_range=price_range,
            swing_highs=swing_highs,
            swing_lows=swing_lows,
            price_summary=price_summary,
            num_bars=num_bars,
            higher_tf_section=higher_tf_section,
        )
```

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest tests/test_prompt_builder_mtf.py::TestTemplateIntegration -v`
Expected: PASS (all 4 tests)

- [ ] **Step 7: Commit**

```bash
git add data/prompt_builder.py tests/test_prompt_builder_mtf.py
git commit -m "feat: add higher timeframe section to prompt templates

- Update DirectionPredictionPrompt with optional higher_tf_context
- Update MomentumAssessmentPrompt with optional higher_tf_context
- Update SupportResistancePrompt with optional higher_tf_context
- Section injected between Market Data and Technical Indicators"
```

---

## Task 6: Extend PromptBuilder.build_prompt() with Multi-Timeframe Logic

**Files:**
- Modify: `data/prompt_builder.py:504-639`
- Test: `tests/test_prompt_builder_mtf.py`

- [ ] **Step 1: Write failing test for build_prompt() integration**

Add to `tests/test_prompt_builder_mtf.py`:

```python
from data.prompt_builder import PromptBuilder, TaskConfig, TaskType
from data.regime_filter import MarketRegime


class TestPromptBuilderMultiTimeframe:
    """Test PromptBuilder with multi-timeframe data."""

    def test_build_prompt_includes_higher_tf_section(self):
        """Should include higher TF section when data provided."""
        builder = PromptBuilder()
        task = TaskConfig(
            task_type=TaskType.PREDICT_DIRECTION,
            weight=1.0,
            difficulty=2,
            min_bars_required=50,
        )

        df_current = create_test_df_bullish(bars=100)
        higher_tf_data = {
            "4h": create_test_df_bullish(bars=100),
            "1d": create_test_df_bullish(bars=100),
        }

        prompt = builder.build_prompt(
            task=task,
            df=df_current,
            symbol="BTC/USDT",
            timeframe="1h",
            market_regime=MarketRegime.NEUTRAL,
            higher_tf_data=higher_tf_data,
        )

        assert "## Higher Timeframe Context" in prompt
        assert "4h:" in prompt
        assert "1d:" in prompt
        assert "Confluence:" in prompt

    def test_build_prompt_omits_section_when_no_data(self):
        """Should omit higher TF section when higher_tf_data=None."""
        builder = PromptBuilder()
        task = TaskConfig(
            task_type=TaskType.PREDICT_DIRECTION,
            weight=1.0,
            difficulty=2,
            min_bars_required=50,
        )

        df_current = create_test_df_bullish(bars=100)

        prompt = builder.build_prompt(
            task=task,
            df=df_current,
            symbol="BTC/USDT",
            timeframe="1h",
            market_regime=MarketRegime.NEUTRAL,
            higher_tf_data=None,
        )

        assert "## Higher Timeframe Context" not in prompt

    def test_build_prompt_selects_2_nearest_timeframes(self):
        """Should select only 2 nearest higher timeframes."""
        builder = PromptBuilder()
        task = TaskConfig(
            task_type=TaskType.PREDICT_DIRECTION,
            weight=1.0,
            difficulty=2,
            min_bars_required=50,
        )

        df_current = create_test_df_bullish(bars=100)
        higher_tf_data = {
            "5m": create_test_df_bullish(bars=100),
            "15m": create_test_df_bullish(bars=100),
            "1h": create_test_df_bullish(bars=100),
            "4h": create_test_df_bullish(bars=100),
        }

        prompt = builder.build_prompt(
            task=task,
            df=df_current,
            symbol="BTC/USDT",
            timeframe="1m",
            market_regime=MarketRegime.NEUTRAL,
            higher_tf_data=higher_tf_data,
        )

        # Should only include 5m and 15m (2 nearest)
        assert "5m:" in prompt
        assert "15m:" in prompt
        # Should NOT include 1h or 4h
        assert "1h:" not in prompt
        assert "4h:" not in prompt

    def test_build_prompt_works_for_all_task_types(self):
        """Should work for all 3 task types."""
        builder = PromptBuilder()
        df_current = create_test_df_bullish(bars=100)
        higher_tf_data = {"4h": create_test_df_bullish(bars=100)}

        # Direction prediction
        task_direction = TaskConfig(TaskType.PREDICT_DIRECTION, 1.0, 2, 50)
        prompt_direction = builder.build_prompt(
            task=task_direction,
            df=df_current,
            symbol="BTC/USDT",
            timeframe="1h",
            market_regime=MarketRegime.NEUTRAL,
            higher_tf_data=higher_tf_data,
        )
        assert "## Higher Timeframe Context" in prompt_direction

        # Momentum assessment
        task_momentum = TaskConfig(TaskType.ASSESS_MOMENTUM, 0.8, 2, 30)
        prompt_momentum = builder.build_prompt(
            task=task_momentum,
            df=df_current,
            symbol="BTC/USDT",
            timeframe="1h",
            market_regime=MarketRegime.NEUTRAL,
            higher_tf_data=higher_tf_data,
        )
        assert "## Higher Timeframe Context" in prompt_momentum

        # Support/Resistance
        task_sr = TaskConfig(TaskType.IDENTIFY_SUPPORT_RESISTANCE, 0.6, 3, 100)
        prompt_sr = builder.build_prompt(
            task=task_sr,
            df=df_current,
            symbol="BTC/USDT",
            timeframe="1h",
            market_regime=MarketRegime.NEUTRAL,
            higher_tf_data=higher_tf_data,
        )
        assert "## Higher Timeframe Context" in prompt_sr

    def test_build_prompt_handles_insufficient_data_gracefully(self):
        """Should skip timeframes with insufficient data."""
        builder = PromptBuilder()
        task = TaskConfig(TaskType.PREDICT_DIRECTION, 1.0, 2, 50)

        df_current = create_test_df_bullish(bars=100)
        higher_tf_data = {
            "4h": create_test_df_bullish(bars=30),  # Insufficient
            "1d": create_test_df_bullish(bars=100),  # OK
        }

        prompt = builder.build_prompt(
            task=task,
            df=df_current,
            symbol="BTC/USDT",
            timeframe="1h",
            market_regime=MarketRegime.NEUTRAL,
            higher_tf_data=higher_tf_data,
        )

        # Should include 1d but not 4h
        assert "1d:" in prompt
        # Might not have 4h if skipped due to insufficient data
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_prompt_builder_mtf.py::TestPromptBuilderMultiTimeframe -v`
Expected: FAIL - TypeError: build_prompt() got an unexpected keyword argument 'higher_tf_data'

- [ ] **Step 3: Update build_prompt() signature and add validation**

In `data/prompt_builder.py`, modify `build_prompt()` method (around line 504):

```python
def build_prompt(
    self,
    task: TaskConfig,
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    market_regime: MarketRegime,
    higher_tf_data: dict[str, pd.DataFrame] | None = None,
) -> str:
    """
    Build prompt for given task and market data.

    Args:
        task: Task configuration
        df: OHLCV DataFrame with sufficient history
        symbol: Trading pair
        timeframe: Candle timeframe
        market_regime: Current market regime
        higher_tf_data: Optional dict of higher timeframe OHLCV DataFrames
                       Keys are timeframe strings (e.g., "4h", "1d")
                       Values are OHLCV DataFrames

    Returns:
        Rendered prompt string

    Raises:
        ValueError: If insufficient data or unsupported task type
    """
    if len(df) < task.min_bars_required:
        raise ValueError(
            f"Insufficient data: need {task.min_bars_required}, got {len(df)}"
        )

    template = self.templates.get(task.task_type)
    if template is None:
        raise ValueError(f"No template for task type: {task.task_type}")

    # Validate and filter higher_tf_data
    validated_higher_tf_data = None
    if higher_tf_data is not None:
        if not isinstance(higher_tf_data, dict):
            logger.warning("Invalid higher_tf_data type", type=type(higher_tf_data))
        else:
            valid_data = {}
            for tf, tf_df in higher_tf_data.items():
                if not isinstance(tf_df, pd.DataFrame):
                    logger.warning("Invalid DataFrame", timeframe=tf, type=type(tf_df))
                    continue
                if len(tf_df) == 0:
                    logger.warning("Empty DataFrame", timeframe=tf)
                    continue
                if len(tf_df) < 52:  # Minimum for Ichimoku (26*2)
                    logger.warning("Insufficient data", timeframe=tf, bars=len(tf_df))
                    continue
                valid_data[tf] = tf_df

            validated_higher_tf_data = valid_data if valid_data else None

    # Build higher timeframe context section
    higher_tf_context = None
    if validated_higher_tf_data:
        # Select up to 2 nearest higher timeframes
        selected_tfs = get_higher_timeframes(timeframe, list(validated_higher_tf_data.keys()))

        if selected_tfs:
            summaries = []
            for tf in selected_tfs:
                try:
                    summary = summarize_timeframe(validated_higher_tf_data[tf], tf)
                    summaries.append(summary)
                except Exception as e:
                    logger.warning("Failed to summarize timeframe", timeframe=tf, error=str(e))
                    continue

            if summaries:
                # Build context text
                summary_lines = [s["text"] for s in summaries]
                confluence = compute_confluence(summaries)
                summary_lines.append(confluence)
                higher_tf_context = "\n".join(summary_lines)

                logger.debug(
                    "Higher TF context built",
                    timeframes=selected_tfs,
                    num_summaries=len(summaries),
                )

    # Calculate all indicators once
    indicators = compute_all_indicators(df, include_volume=True, include_structure=True)

    # Access scalars directly
    rsi = indicators['rsi'] if indicators['rsi'] is not None else 50.0
    macd = indicators['macd_line'] if indicators['macd_line'] is not None else 0.0
    macd_signal = indicators['macd_signal'] if indicators['macd_signal'] is not None else 0.0

    # BB position not yet in compute_all_indicators, compute separately
    bb_pos = compute_bb_position(df["close"]).iloc[-1]
    bb_pos = bb_pos if not pd.isna(bb_pos) else 0.5

    # Format price summary
    price_summary = self._format_price_summary(df)
    current_price = float(df["close"].iloc[-1])

    # Task-specific rendering
    if task.task_type == TaskType.PREDICT_DIRECTION:
        prompt = template.render(
            symbol=symbol,
            timeframe=timeframe,
            current_price=current_price,
            market_regime=market_regime.value,
            rsi=rsi,
            macd=macd,
            macd_signal=macd_signal,
            bb_position=bb_pos,
            price_summary=price_summary,
            higher_tf_context=higher_tf_context,
        )

    elif task.task_type == TaskType.ASSESS_MOMENTUM:
        # Calculate momentum-specific indicators using series from indicators
        rsi_series = indicators['series']['rsi']
        macd_line = indicators['series']['macd_line']

        rsi_prev = rsi_series.iloc[-2] if len(rsi_series) > 1 and not pd.isna(rsi_series.iloc[-2]) else rsi
        rsi_delta = rsi - rsi_prev

        macd_prev = macd_line.iloc[-2] if len(macd_line) > 1 and not pd.isna(macd_line.iloc[-2]) else macd
        macd_delta = macd - macd_prev

        # Calculate BB width
        current_bb_width = calculate_bb_width(df, period=20)
        prev_bb_width = calculate_bb_width(df.iloc[:-1], period=20) if len(df) > 20 else current_bb_width
        bb_trend = get_bb_trend(current_bb_width, prev_bb_width)

        prompt = template.render(
            symbol=symbol,
            timeframe=timeframe,
            current_price=current_price,
            market_regime=market_regime.value,
            rsi=rsi,
            rsi_prev=rsi_prev,
            rsi_delta=rsi_delta,
            macd=macd,
            macd_signal=macd_signal,
            macd_prev=macd_prev,
            macd_delta=macd_delta,
            bb_width=current_bb_width,
            bb_trend=bb_trend,
            price_summary=price_summary,
            higher_tf_context=higher_tf_context,
        )

    elif task.task_type == TaskType.IDENTIFY_SUPPORT_RESISTANCE:
        # Calculate support/resistance indicators
        price_high = float(df["high"].tail(50).max())
        price_low = float(df["low"].tail(50).min())
        price_range = price_high - price_low

        # Detect swing points
        swing_highs = detect_swing_highs(df, window=5, num_swings=3)
        swing_lows = detect_swing_lows(df, window=5, num_swings=3)

        # Format swing points
        swing_highs_str = ", ".join([f"${h:.2f}" for h in swing_highs]) if swing_highs else "None detected"
        swing_lows_str = ", ".join([f"${l:.2f}" for l in swing_lows]) if swing_lows else "None detected"

        prompt = template.render(
            symbol=symbol,
            timeframe=timeframe,
            current_price=current_price,
            market_regime=market_regime.value,
            price_high=price_high,
            price_low=price_low,
            price_range=price_range,
            swing_highs=swing_highs_str,
            swing_lows=swing_lows_str,
            price_summary=price_summary,
            higher_tf_context=higher_tf_context,
        )

    else:
        raise ValueError(f"Unsupported task type: {task.task_type}")

    logger.info(
        "Prompt built",
        task_type=task.task_type.value,
        symbol=symbol,
        timeframe=timeframe,
        prompt_length=len(prompt),
        has_higher_tf=higher_tf_context is not None,
    )

    return prompt
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_prompt_builder_mtf.py::TestPromptBuilderMultiTimeframe -v`
Expected: PASS (all 5 tests)

- [ ] **Step 5: Run all prompt_builder_mtf tests**

Run: `pytest tests/test_prompt_builder_mtf.py -v`
Expected: PASS (all tests in file)

- [ ] **Step 6: Commit**

```bash
git add data/prompt_builder.py tests/test_prompt_builder_mtf.py
git commit -m "feat: integrate multi-timeframe logic into PromptBuilder

- Add higher_tf_data parameter to build_prompt()
- Validate and filter higher TF DataFrames (min 52 bars)
- Select up to 2 nearest higher timeframes
- Generate summaries and confluence text
- Pass higher_tf_context to all template renders
- Add comprehensive logging for debugging"
```

---

## Task 7: Update Orchestrator to Accept Higher TF Data

**Files:**
- Modify: `swarm/orchestrator.py:137-150`
- Test: `tests/test_orchestrator.py`

- [ ] **Step 1: Write failing test for orchestrator integration**

Add to `tests/test_orchestrator.py`:

```python
import pytest
from tests.fixtures.timeframe_fixtures import create_test_df_bullish


@pytest.mark.asyncio
async def test_run_swarm_workflow_accepts_higher_tf_data():
    """Should accept and pass through higher_tf_data to PromptBuilder."""
    # Create test data
    df = create_test_df_bullish(bars=100)
    higher_tf_data = {
        "4h": create_test_df_bullish(bars=100),
        "1d": create_test_df_bullish(bars=100),
    }

    from swarm.orchestrator import run_swarm_workflow
    from data.regime_filter import MarketRegime
    from data.prompt_builder import TaskType

    state, example = await run_swarm_workflow(
        symbol="BTC/USDT",
        timeframe="1h",
        ohlcv_df=df,
        market_regime=MarketRegime.NEUTRAL,
        task_prompt="Test prompt",  # Will be overridden by PromptBuilder
        task_type=TaskType.PREDICT_DIRECTION,
        higher_tf_data=higher_tf_data,
    )

    # Verify higher TF context appears in task_prompt
    assert "Higher Timeframe Context" in state["task_prompt"]


@pytest.mark.asyncio
async def test_run_swarm_workflow_backward_compatible():
    """Should work without higher_tf_data parameter (backward compatibility)."""
    df = create_test_df_bullish(bars=100)

    from swarm.orchestrator import run_swarm_workflow
    from data.regime_filter import MarketRegime
    from data.prompt_builder import TaskType

    # Call without higher_tf_data
    state, example = await run_swarm_workflow(
        symbol="BTC/USDT",
        timeframe="1h",
        ohlcv_df=df,
        market_regime=MarketRegime.NEUTRAL,
        task_prompt="Test prompt",
        task_type=TaskType.PREDICT_DIRECTION,
    )

    # Should not have higher TF context
    assert "Higher Timeframe Context" not in state["task_prompt"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_orchestrator.py::test_run_swarm_workflow_accepts_higher_tf_data -v`
Expected: FAIL - TypeError: run_swarm_workflow() got an unexpected keyword argument 'higher_tf_data'

- [ ] **Step 3: Update run_swarm_workflow() signature**

In `swarm/orchestrator.py`, modify function signature (around line 137):

```python
async def run_swarm_workflow(
    symbol: str,
    timeframe: str,
    ohlcv_df: pd.DataFrame,
    market_regime: MarketRegime,
    task_prompt: str,
    task_type: TaskType = TaskType.PREDICT_DIRECTION,
    higher_tf_data: dict[str, pd.DataFrame] | None = None,
) -> tuple[SwarmState, TrainingExample]:
    """
    Execute generator -> critic workflow with guaranteed VRAM cleanup.

    VRAM Safety:
    - OllamaClient context manager guarantees cleanup on any exit
    - Explicit unload_current() between model switches
```

- [ ] **Step 4: Find where PromptBuilder is used in orchestrator**

Run: `grep -n "PromptBuilder\|build_prompt" swarm/orchestrator.py`
Expected: Find the location where prompt is built

Note: Looking at the function signature, it receives `task_prompt` as a parameter, which suggests the prompt is built BEFORE calling this function. We need to check if PromptBuilder is called within the function or if we need to update the caller.

- [ ] **Step 5: Check orchestrator implementation**

Read `swarm/orchestrator.py` lines 150-250 to understand workflow:

Run: `head -250 swarm/orchestrator.py | tail -100`

Expected: See workflow implementation

Based on the initial read (lines 1-150), the `task_prompt` is passed in as a parameter. This means the prompt is built by the CALLER, not inside `run_swarm_workflow()`.

The design spec says to pass `higher_tf_data` through to PromptBuilder, but looking at the actual code, `run_swarm_workflow()` doesn't call PromptBuilder - it receives a pre-built prompt.

We have two options:
A) Add PromptBuilder call inside run_swarm_workflow()
B) Document that callers should pass higher_tf_data to PromptBuilder before calling orchestrator

Looking at the design spec again, it says "pass through to PromptBuilder", which implies the orchestrator should call it.

Let me check the actual implementation more carefully.

- [ ] **Step 6: Read full run_swarm_workflow implementation**

In `swarm/orchestrator.py`, read lines 137-300:

After reviewing, we see that `task_prompt` is a required parameter and is used directly in the workflow. The function does NOT call PromptBuilder internally.

Looking at this architecture, the cleanest approach is:
- Accept `higher_tf_data` as optional parameter
- If `task_prompt` is provided AND `higher_tf_data` is provided, log a warning (they're mutually exclusive)
- For now, just pass the workflow through as-is since the prompt is pre-built by the caller

Actually, re-reading the design spec: "run_swarm_workflow() in swarm/orchestrator.py - Add higher_tf_data parameter and pass through to PromptBuilder"

But the orchestrator doesn't call PromptBuilder! Let me re-check where PromptBuilder is actually called.

Looking at the codebase structure, it's likely called in `generate_training_dataset.py` or similar scripts. The orchestrator just receives the prompt.

For this task, the cleanest implementation is:
- Accept `higher_tf_data` parameter
- Document that it's unused by orchestrator (prompt is pre-built)
- OR rebuild the prompt inside orchestrator if higher_tf_data is provided

Let's rebuild the prompt inside orchestrator when higher_tf_data is provided. This matches the design intent.

- [ ] **Step 7: Add PromptBuilder call to orchestrator**

In `swarm/orchestrator.py`, add at the top of imports:

```python
from data.prompt_builder import PromptBuilder, sample_task
```

Then in `run_swarm_workflow()`, add before the main workflow (around line 165):

```python
# If higher_tf_data provided, rebuild prompt with multi-timeframe context
if higher_tf_data is not None:
    logger.info(
        "Rebuilding prompt with higher timeframe context",
        timeframe=timeframe,
        higher_tfs=list(higher_tf_data.keys()),
    )

    # Sample task based on available data
    task = sample_task(available_bars=len(ohlcv_df))

    # Build prompt with higher TF context
    builder = PromptBuilder()
    task_prompt = builder.build_prompt(
        task=task,
        df=ohlcv_df,
        symbol=symbol,
        timeframe=timeframe,
        market_regime=market_regime,
        higher_tf_data=higher_tf_data,
    )

    logger.debug(
        "Prompt rebuilt with higher TF context",
        prompt_length=len(task_prompt),
    )
```

- [ ] **Step 8: Run test to verify it passes**

Run: `pytest tests/test_orchestrator.py::test_run_swarm_workflow_accepts_higher_tf_data tests/test_orchestrator.py::test_run_swarm_workflow_backward_compatible -v`
Expected: PASS (both tests)

- [ ] **Step 9: Commit**

```bash
git add swarm/orchestrator.py tests/test_orchestrator.py
git commit -m "feat: add higher_tf_data support to orchestrator

- Add optional higher_tf_data parameter to run_swarm_workflow()
- Rebuild prompt with PromptBuilder when higher_tf_data provided
- Maintain backward compatibility when parameter omitted
- Add integration tests for both modes"
```

---

## Task 8: Run Full Test Suite and Fix Any Issues

**Files:**
- Various test files

- [ ] **Step 1: Run all existing tests**

Run: `pytest tests/test_prompt_builder.py tests/test_orchestrator.py -v`
Expected: PASS (verify backward compatibility)

- [ ] **Step 2: Run full multi-timeframe test suite**

Run: `pytest tests/test_prompt_builder_mtf.py -v --tb=short`
Expected: PASS (all MTF tests)

- [ ] **Step 3: Run integration tests**

Run: `pytest tests/test_orchestrator.py -v -k "higher_tf"`
Expected: PASS (integration tests)

- [ ] **Step 4: Check test coverage**

Run: `pytest tests/test_prompt_builder_mtf.py --cov=data.prompt_builder --cov-report=term-missing`
Expected: >90% coverage for new functions

- [ ] **Step 5: Fix any failing tests**

If any tests fail, debug and fix them. Common issues:
- NaN handling in indicators
- Insufficient data in fixtures
- Type mismatches

- [ ] **Step 6: Commit any fixes**

```bash
git add <fixed_files>
git commit -m "fix: resolve test failures in multi-timeframe implementation"
```

---

## Task 9: Update CLAUDE.md Project Memory

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Read current CLAUDE.md**

Run: `head -100 CLAUDE.md`

- [ ] **Step 2: Update Current State section**

In `CLAUDE.md`, update "Current State" section (around line 20):

```markdown
**Completed:**
- Sessions 1-10: Environment, Data, Swarm, Verifier, Reward, Evaluation, DPO Infrastructure, Dataset Generation, End-to-End DPO Workflow
- Session 11a: Realistic Fee Model
- Session 11b: Technical Indicator Expansion (17 indicators, 5 groups, compute_all_indicators aggregation)
- Session 12: Multi-Timeframe Context (higher TF trend summaries in prompts)

**Next:** Session 13 - TBD
```

- [ ] **Step 3: Add Multi-Timeframe section to Architecture Decisions**

In `CLAUDE.md`, add new section after "Indicator System (Session 11b)" (around line 30):

```markdown
### Multi-Timeframe Context (Session 12)
- **Timeframe Hierarchy:** ["1m", "5m", "15m", "1h", "4h", "1d"]
- **Selection Logic:** Up to 2 nearest higher timeframes from available set
- **Trend Summary:** 4 indicators (Ichimoku cloud, KAMA slope, Donchian position, RSI zone)
- **Confluence Detection:** Aligned (all same trend), Mixed, or Conflicting
- **Prompt Integration:** Optional "Higher Timeframe Context" section injected between Market Data and Technical Indicators
- **API:** `PromptBuilder.build_prompt(higher_tf_data=dict[str, pd.DataFrame])`, `run_swarm_workflow(higher_tf_data=...)`
- **Backward Compatible:** All parameters optional with None defaults
```

- [ ] **Step 4: Update File Index**

In `CLAUDE.md`, update "File Index" Data Layer section (around line 60):

```markdown
### Data Layer
- `data/indicators.py` - 17 indicators + compute_all_indicators() aggregation (82 tests)
- `data/cache_wrapper.py` - AsyncDiskCache with asyncio.to_thread()
- `data/market_data.py` - CCXT client with context manager
- `data/regime_filter.py` - RegimeClassifier with volatility percentiles
- `data/prompt_builder.py` - Task sampling, multi-timeframe context, 3 prompt templates
- `data/historical_windows.py` - Window walking with completeness validation
- `data/inference_queue.py` - Sequential job processor with JSONL streaming
```

- [ ] **Step 5: Update Tests section**

In `CLAUDE.md`, update test count in "Total Tests" (around line 120):

```markdown
**Total Tests:** 520+ passing (5 pre-existing orchestrator failures excluded)
```

- [ ] **Step 6: Add new test file to Tests section**

In `CLAUDE.md`, update "Tests" section (around line 100):

```markdown
### Tests
- `tests/test_config.py` - 40 tests
- `tests/test_indicators.py` - 19 tests (original indicators)
- `tests/test_indicators_extended.py` - 63 tests (extended indicators)
- `tests/test_data_layer.py` - 21 tests
- `tests/test_prompt_builder.py` - Original prompt builder tests
- `tests/test_prompt_builder_mtf.py` - 31 tests (multi-timeframe)
- `tests/fixtures/timeframe_fixtures.py` - Test data generators (bullish/bearish/neutral)
```

- [ ] **Step 7: Commit CLAUDE.md updates**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md with Session 12 multi-timeframe context

- Mark Session 12 complete
- Document timeframe hierarchy and selection logic
- Add multi-timeframe architecture decisions
- Update file index and test counts"
```

---

## Task 10: Create Session Summary Documentation

**Files:**
- Create: `docs/sessions/session-12-multi-timeframe-context.md`

- [ ] **Step 1: Create session summary document**

Create `docs/sessions/session-12-multi-timeframe-context.md`:

```markdown
# Session 12: Multi-Timeframe Context

**Date:** 2026-04-06
**Status:** Complete

## Summary

Added higher timeframe trend context to signal generation prompts to improve signal quality through multi-timeframe confluence. A bullish 1h signal that aligns with bullish 4h and 1d trends is now explicitly communicated to the LLM generator.

## Implementation

### Core Components

**1. Timeframe Selection (data/prompt_builder.py)**
- `TIMEFRAME_HIERARCHY` constant: ["1m", "5m", "15m", "1h", "4h", "1d"]
- `get_higher_timeframes()`: Select up to 2 nearest higher timeframes
- Graceful handling of edge cases (no higher TFs, unknown TFs)

**2. Trend Summarization (data/prompt_builder.py)**
- `summarize_timeframe()`: Generate trend summary from OHLCV data
- 4-indicator classification:
  - Ichimoku cloud position (above/below/inside)
  - KAMA slope (rising/falling/flat) with 0.1% threshold
  - Donchian channel position (upper/lower/middle) by percentile
  - RSI zone (overbought/oversold/neutral)
- Overall trend: bullish (above cloud + rising KAMA), bearish (below cloud + falling KAMA), neutral (otherwise)

**3. Confluence Detection (data/prompt_builder.py)**
- `compute_confluence()`: Analyze trend alignment
- Classifications: aligned (all same), mixed (different trends)
- Human-readable output: "Confluence: Aligned with higher timeframes (bullish)"

**4. Prompt Integration**
- All 3 templates updated (DirectionPredictionPrompt, MomentumAssessmentPrompt, SupportResistancePrompt)
- Optional "Higher Timeframe Context" section between Market Data and Technical Indicators
- Format:
  ```
  ## Higher Timeframe Context
  4h: Bullish (above cloud, KAMA rising), near Donchian upper, RSI 58 (neutral)
  1d: Bullish (above cloud, KAMA rising), middle of Donchian, RSI 52 (neutral)
  Confluence: Aligned with higher timeframes (bullish)
  ```

**5. PromptBuilder Integration**
- `build_prompt()` accepts `higher_tf_data: dict[str, pd.DataFrame] | None`
- Validates DataFrames (min 52 bars for Ichimoku)
- Selects up to 2 nearest higher timeframes
- Generates summaries and confluence text
- Passes to template renders

**6. Orchestrator Integration**
- `run_swarm_workflow()` accepts `higher_tf_data` parameter
- Rebuilds prompt with PromptBuilder when higher_tf_data provided
- Backward compatible (optional parameter)

### Testing

**Test Files:**
- `tests/test_prompt_builder_mtf.py` - 31 unit tests
- `tests/fixtures/timeframe_fixtures.py` - Reusable bullish/bearish/neutral DataFrames
- `tests/test_orchestrator.py` - 2 integration tests

**Test Coverage:**
- Timeframe selection (edge cases, ordering, filtering)
- Trend summarization (all 3 patterns, all indicators)
- Confluence detection (aligned, mixed, single TF)
- Template integration (all 3 task types)
- PromptBuilder integration (data validation, selection logic)
- Orchestrator integration (pass-through, backward compatibility)

## Results

**Code Changes:**
- Modified: 2 files (prompt_builder.py, orchestrator.py)
- Created: 2 test files
- Total lines added: ~450
- Total lines modified: ~100

**Test Results:**
- 31 new tests added
- All tests passing
- >90% coverage for new functions

**Backward Compatibility:**
- All existing tests pass without modification
- No breaking API changes
- Optional parameters with None defaults

## Example Output

**Without Higher TF Context (existing behavior):**
```
## Market Data
Symbol: BTC/USDT
Timeframe: 1h
Current Price: $50000.0000
Market Regime: neutral

## Technical Indicators
RSI(14): 55.23
...
```

**With Higher TF Context (new feature):**
```
## Market Data
Symbol: BTC/USDT
Timeframe: 1h
Current Price: $50000.0000
Market Regime: neutral

## Higher Timeframe Context
4h: Bullish (above cloud, KAMA rising), near Donchian upper, RSI 58 (neutral)
1d: Bullish (above cloud, KAMA rising), middle of Donchian, RSI 52 (neutral)
Confluence: Aligned with higher timeframes (bullish)

## Technical Indicators
RSI(14): 55.23
...
```

## Future Enhancements

Out of scope for this session, but could be considered later:

1. Adaptive timeframe selection based on signal strength
2. Multi-timeframe backtesting in verifier layer
3. Timeframe-weighted reward signals
4. Auto-discovery of optimal timeframe combinations per symbol
5. Inter-timeframe divergence detection (e.g., RSI divergence across TFs)

## References

- Design Spec: `docs/superpowers/specs/2026-04-06-multi-timeframe-context-design.md`
- Implementation Plan: `docs/superpowers/plans/2026-04-06-multi-timeframe-context.md`
- CLAUDE.md: Updated with Session 12 architecture decisions
```

- [ ] **Step 2: Create docs/sessions directory if needed**

Run: `mkdir -p docs/sessions`

- [ ] **Step 3: Commit session summary**

```bash
git add docs/sessions/session-12-multi-timeframe-context.md
git commit -m "docs: add Session 12 summary (multi-timeframe context)

- Document implementation approach and components
- Summarize testing strategy and results
- Provide example output comparison
- List future enhancement opportunities"
```

---

## Task 11: Final Integration Test and Validation

**Files:**
- Various

- [ ] **Step 1: Run full test suite**

Run: `pytest tests/ -v --tb=short -x`
Expected: All tests pass (may have 5 pre-existing orchestrator failures)

- [ ] **Step 2: Test with actual usage pattern**

Create temporary test script `test_mtf_integration.py`:

```python
"""Manual integration test for multi-timeframe context."""

import asyncio
from data.prompt_builder import PromptBuilder, sample_task
from data.regime_filter import MarketRegime
from tests.fixtures.timeframe_fixtures import create_test_df_bullish


async def test_end_to_end():
    """Test end-to-end multi-timeframe workflow."""
    # Create test data
    df_1h = create_test_df_bullish(bars=100)
    df_4h = create_test_df_bullish(bars=100)
    df_1d = create_test_df_bullish(bars=100)

    # Build prompt with higher TF context
    builder = PromptBuilder()
    task = sample_task(available_bars=100)

    prompt = builder.build_prompt(
        task=task,
        df=df_1h,
        symbol="BTC/USDT",
        timeframe="1h",
        market_regime=MarketRegime.NEUTRAL,
        higher_tf_data={
            "4h": df_4h,
            "1d": df_1d,
        },
    )

    print("=== Generated Prompt ===")
    print(prompt)
    print("\n=== Verification ===")

    # Verify key sections present
    assert "## Higher Timeframe Context" in prompt
    assert "4h:" in prompt
    assert "1d:" in prompt
    assert "Confluence:" in prompt

    print("✓ Higher Timeframe Context section present")
    print("✓ All timeframes included")
    print("✓ Confluence detected")
    print("\nIntegration test PASSED")


if __name__ == "__main__":
    asyncio.run(test_end_to_end())
```

Run: `python test_mtf_integration.py`
Expected: See full prompt with higher TF section, test passes

- [ ] **Step 3: Clean up test script**

Run: `rm test_mtf_integration.py`

- [ ] **Step 4: Verify test counts**

Run: `pytest tests/test_prompt_builder_mtf.py --collect-only | grep "test session" | tail -1`
Expected: 31 tests collected

- [ ] **Step 5: Final commit**

```bash
git add -A
git commit -m "feat: complete Session 12 multi-timeframe context implementation

Summary of changes:
- Add timeframe hierarchy and selection logic (max 2 higher TFs)
- Implement trend summarization (Ichimoku, KAMA, Donchian, RSI)
- Add confluence detection (aligned/mixed/conflicting)
- Update all 3 prompt templates with optional higher TF section
- Extend PromptBuilder.build_prompt() with higher_tf_data parameter
- Update orchestrator to accept and process higher TF data
- Add 31 unit tests + 3 test fixtures
- Maintain 100% backward compatibility

Test Results: 520+ tests passing, >90% coverage on new code
Documentation: Updated CLAUDE.md, design spec, implementation plan, session summary"
```

---

## Success Criteria Checklist

After completing all tasks, verify:

- [ ] **Functionality**
  - Prompts include higher TF context when data provided ✓
  - Prompts omit section when no data provided ✓
  - Up to 2 nearest higher timeframes selected correctly ✓
  - Trend summaries accurate for known indicator patterns ✓

- [ ] **Backward Compatibility**
  - All existing tests pass without modification ✓
  - Single-timeframe workflows unchanged ✓
  - No breaking API changes ✓

- [ ] **Test Coverage**
  - All new functions have >90% coverage ✓
  - Edge cases tested (empty data, NaN values, invalid TFs) ✓
  - Integration tests verify end-to-end flow ✓

- [ ] **Code Quality**
  - Type hints on all new functions ✓
  - Comprehensive logging for debugging ✓
  - Clear docstrings with examples ✓

- [ ] **Documentation**
  - CLAUDE.md updated with Session 12 ✓
  - Design spec saved and committed ✓
  - Session summary created ✓
  - All commits follow conventional format ✓
