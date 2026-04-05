"""Tests for historical window walking logic."""

import pytest
import pandas as pd
from datetime import datetime, timezone

from data.historical_windows import (
    HistoricalWindow,
    timeframe_to_milliseconds,
    calculate_window_timestamps,
    calculate_data_completeness,
    fetch_window_data,
)


class TestTimeframeConversion:
    """Test timeframe to milliseconds conversion."""

    def test_supported_timeframes(self):
        """Test all supported timeframes convert correctly."""
        assert timeframe_to_milliseconds("1m") == 60_000
        assert timeframe_to_milliseconds("5m") == 300_000
        assert timeframe_to_milliseconds("15m") == 900_000
        assert timeframe_to_milliseconds("1h") == 3_600_000
        assert timeframe_to_milliseconds("4h") == 14_400_000
        assert timeframe_to_milliseconds("1d") == 86_400_000

    def test_unsupported_timeframe_raises(self):
        """Test unsupported timeframe raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported timeframe"):
            timeframe_to_milliseconds("3h")

        with pytest.raises(ValueError, match="Unsupported timeframe"):
            timeframe_to_milliseconds("invalid")


class TestWindowTimestampCalculation:
    """Test historical window timestamp calculation."""

    def test_single_window(self):
        """Test calculation with single window."""
        current_ts = 1704067200000  # 2024-01-01 00:00:00 UTC
        timestamps = calculate_window_timestamps(
            current_timestamp_ms=current_ts,
            timeframe="1h",
            window_count=1,
            stride_bars=100,
        )

        assert len(timestamps) == 1
        assert timestamps[0] == current_ts

    def test_multiple_windows(self):
        """Test calculation with multiple windows."""
        current_ts = 1704067200000  # 2024-01-01 00:00:00 UTC
        timestamps = calculate_window_timestamps(
            current_timestamp_ms=current_ts,
            timeframe="1h",
            window_count=3,
            stride_bars=100,
        )

        assert len(timestamps) == 3

        # Window 0: current
        assert timestamps[0] == current_ts

        # Window 1: -100 hours (100 bars * 3600000 ms/bar)
        assert timestamps[1] == current_ts - (100 * 3_600_000)

        # Window 2: -200 hours
        assert timestamps[2] == current_ts - (200 * 3_600_000)

    def test_descending_order(self):
        """Test timestamps are in descending order (newest first)."""
        current_ts = 1704067200000
        timestamps = calculate_window_timestamps(
            current_timestamp_ms=current_ts,
            timeframe="1h",
            window_count=5,
            stride_bars=50,
        )

        assert timestamps == sorted(timestamps, reverse=True)

    def test_different_timeframes(self):
        """Test calculation with different timeframes."""
        current_ts = 1704067200000

        # 1m timeframe
        ts_1m = calculate_window_timestamps(
            current_ts, "1m", window_count=2, stride_bars=100
        )
        assert ts_1m[1] == current_ts - (100 * 60_000)

        # 1d timeframe
        ts_1d = calculate_window_timestamps(
            current_ts, "1d", window_count=2, stride_bars=10
        )
        assert ts_1d[1] == current_ts - (10 * 86_400_000)

    def test_invalid_window_count_raises(self):
        """Test invalid window_count raises ValueError."""
        with pytest.raises(ValueError, match="window_count must be >= 1"):
            calculate_window_timestamps(
                1704067200000, "1h", window_count=0, stride_bars=100
            )

        with pytest.raises(ValueError, match="window_count must be >= 1"):
            calculate_window_timestamps(
                1704067200000, "1h", window_count=-1, stride_bars=100
            )

    def test_invalid_stride_raises(self):
        """Test invalid stride_bars raises ValueError."""
        with pytest.raises(ValueError, match="stride_bars must be >= 1"):
            calculate_window_timestamps(
                1704067200000, "1h", window_count=3, stride_bars=0
            )

        with pytest.raises(ValueError, match="stride_bars must be >= 1"):
            calculate_window_timestamps(
                1704067200000, "1h", window_count=3, stride_bars=-10
            )


class TestDataCompleteness:
    """Test data completeness calculation."""

    def test_perfect_completeness(self):
        """Test 100% completeness."""
        df = pd.DataFrame({"close": range(100)})
        completeness = calculate_data_completeness(df, expected_bars=100)
        assert completeness == 1.0

    def test_partial_completeness(self):
        """Test partial completeness."""
        df = pd.DataFrame({"close": range(95)})
        completeness = calculate_data_completeness(df, expected_bars=100)
        assert completeness == 0.95

    def test_over_completeness_capped(self):
        """Test completeness capped at 1.0."""
        df = pd.DataFrame({"close": range(110)})
        completeness = calculate_data_completeness(df, expected_bars=100)
        assert completeness == 1.0

    def test_zero_expected_bars(self):
        """Test zero expected bars returns 0.0."""
        df = pd.DataFrame({"close": range(50)})
        completeness = calculate_data_completeness(df, expected_bars=0)
        assert completeness == 0.0


@pytest.mark.asyncio
class TestFetchWindowData:
    """Test window data fetching with completeness validation."""

    async def test_successful_fetch(self, mock_market_data_service):
        """Test successful window data fetch."""
        # Create mock data
        timestamps = [1704067200000 + i * 3_600_000 for i in range(100)]
        mock_df = pd.DataFrame({
            "timestamp": timestamps,
            "open": [50000.0] * 100,
            "high": [50100.0] * 100,
            "low": [49900.0] * 100,
            "close": [50000.0 + i * 10 for i in range(100)],
            "volume": [100.0] * 100,
        })

        mock_service = mock_market_data_service(mock_df)

        window = HistoricalWindow(
            symbol="BTC/USDT",
            timeframe="1h",
            end_timestamp_ms=timestamps[-1],
            lookback_bars=100,
            stride_index=0,
        )

        df = await fetch_window_data(mock_service, window)

        assert df is not None
        assert len(df) == 100
        assert df["timestamp"].iloc[-1] <= window.end_timestamp_ms

    async def test_insufficient_completeness(self, mock_market_data_service):
        """Test returns None when completeness below threshold."""
        # Create incomplete data (only 90 bars)
        timestamps = [1704067200000 + i * 3_600_000 for i in range(90)]
        mock_df = pd.DataFrame({
            "timestamp": timestamps,
            "open": [50000.0] * 90,
            "high": [50100.0] * 90,
            "low": [49900.0] * 90,
            "close": [50000.0 + i * 10 for i in range(90)],
            "volume": [100.0] * 90,
        })

        mock_service = mock_market_data_service(mock_df)

        window = HistoricalWindow(
            symbol="BTC/USDT",
            timeframe="1h",
            end_timestamp_ms=timestamps[-1],
            lookback_bars=100,
            stride_index=0,
        )

        # Should return None (90/100 = 0.9 < 0.95 default threshold)
        df = await fetch_window_data(mock_service, window, min_completeness=0.95)
        assert df is None

    async def test_custom_completeness_threshold(self, mock_market_data_service):
        """Test custom completeness threshold."""
        # Create data with 90% completeness
        timestamps = [1704067200000 + i * 3_600_000 for i in range(90)]
        mock_df = pd.DataFrame({
            "timestamp": timestamps,
            "open": [50000.0] * 90,
            "high": [50100.0] * 90,
            "low": [49900.0] * 90,
            "close": [50000.0 + i * 10 for i in range(90)],
            "volume": [100.0] * 90,
        })

        mock_service = mock_market_data_service(mock_df)

        window = HistoricalWindow(
            symbol="BTC/USDT",
            timeframe="1h",
            end_timestamp_ms=timestamps[-1],
            lookback_bars=100,
            stride_index=0,
        )

        # Should succeed with lower threshold
        df = await fetch_window_data(mock_service, window, min_completeness=0.85)
        assert df is not None
        assert len(df) == 90

    async def test_empty_data_returns_none(self, mock_market_data_service):
        """Test returns None when no data available."""
        mock_df = pd.DataFrame()
        mock_service = mock_market_data_service(mock_df)

        window = HistoricalWindow(
            symbol="BTC/USDT",
            timeframe="1h",
            end_timestamp_ms=1704067200000,
            lookback_bars=100,
            stride_index=0,
        )

        df = await fetch_window_data(mock_service, window)
        assert df is None

    async def test_service_exception_returns_none(self, mock_market_data_service_exception):
        """Test returns None when service raises exception."""
        mock_service = mock_market_data_service_exception(
            Exception("Connection error")
        )

        window = HistoricalWindow(
            symbol="BTC/USDT",
            timeframe="1h",
            end_timestamp_ms=1704067200000,
            lookback_bars=100,
            stride_index=0,
        )

        df = await fetch_window_data(mock_service, window)
        assert df is None
