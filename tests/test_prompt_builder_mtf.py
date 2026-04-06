"""Tests for multi-timeframe context in prompt builder."""

import pandas as pd
import pytest

from data.prompt_builder import get_higher_timeframes, TIMEFRAME_HIERARCHY
from tests.fixtures.timeframe_fixtures import (
    create_test_df_bullish,
    create_test_df_bearish,
    create_test_df_neutral,
)


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


class TestTimeframeFixtures:
    """Verify test fixtures produce expected indicator patterns."""

    def test_bullish_fixture_has_bullish_indicators(self):
        """Bullish fixture should produce bullish indicator values."""
        from data.indicators import compute_all_indicators

        df = create_test_df_bullish()
        indicators = compute_all_indicators(df, include_volume=False, include_structure=False)

        # Price should be above Donchian middle (uptrend characteristic)
        close_price = df["close"].iloc[-1]
        donchian_middle = indicators["donchian_middle"]
        assert close_price > donchian_middle

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

        # Price should be below Donchian middle (downtrend characteristic)
        close_price = df["close"].iloc[-1]
        donchian_middle = indicators["donchian_middle"]
        assert close_price < donchian_middle

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

        # Price should be near Donchian middle (sideways characteristic)
        price = df["close"].iloc[-1]
        donchian_middle = indicators["donchian_middle"]
        # Allow +/- 2% deviation from middle for neutral
        tolerance = donchian_middle * 0.02
        assert abs(price - donchian_middle) < tolerance

        # RSI should be near 50
        assert 45 <= indicators["rsi"] <= 55

    def test_fixtures_survive_validation(self):
        """Fixtures should not lose bars when validated."""
        from data.indicators import validate_ohlcv

        for fixture_fn in [create_test_df_bullish, create_test_df_bearish, create_test_df_neutral]:
            df = fixture_fn()
            validated = validate_ohlcv(df)
            assert len(validated) == len(df), f"{fixture_fn.__name__} lost {len(df)-len(validated)} bars"
