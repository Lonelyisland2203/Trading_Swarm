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
