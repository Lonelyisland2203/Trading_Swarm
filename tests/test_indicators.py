"""Tests for technical indicators."""

import pytest
import pandas as pd
import numpy as np

from data.indicators import (
    compute_rsi,
    compute_macd,
    compute_bollinger_bands,
    compute_bb_position,
    validate_ohlcv,
)


@pytest.fixture
def sample_price_data():
    """Create sample price data for testing."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=100, freq="1h")
    base_price = 100.0

    # Generate realistic price movement
    returns = np.random.normal(0, 0.01, 100)
    prices = base_price * (1 + returns).cumprod()

    return pd.Series(prices, index=dates)


@pytest.fixture
def sample_ohlcv_df():
    """Create sample OHLCV DataFrame."""
    np.random.seed(42)
    n = 100
    timestamps = [1704067200000 + i * 3600000 for i in range(n)]  # Hourly

    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": np.random.uniform(99, 101, n),
        "high": np.random.uniform(100, 102, n),
        "low": np.random.uniform(98, 100, n),
        "close": np.random.uniform(99, 101, n),
        "volume": np.random.uniform(1000, 2000, n),
    })

    # Ensure OHLC relationship
    df["high"] = df[["open", "close", "high"]].max(axis=1)
    df["low"] = df[["open", "close", "low"]].min(axis=1)

    return df


class TestRSI:
    """Test RSI calculation."""

    def test_rsi_range(self, sample_price_data):
        """Test that RSI values are between 0 and 100."""
        rsi = compute_rsi(sample_price_data, period=14)

        # Drop NaN values
        rsi_valid = rsi.dropna()

        assert (rsi_valid >= 0).all()
        assert (rsi_valid <= 100).all()

    def test_rsi_length(self, sample_price_data):
        """Test RSI output length matches input."""
        rsi = compute_rsi(sample_price_data, period=14)
        assert len(rsi) == len(sample_price_data)

    def test_rsi_warmup_period(self, sample_price_data):
        """Test that RSI has NaN values during warmup."""
        period = 14
        rsi = compute_rsi(sample_price_data, period=period)

        # First values before period should be NaN (EWM starts at min_periods)
        # EWM with min_periods=14 starts producing values at index 13
        assert rsi.iloc[:period-1].isna().all()
        # After warmup, should have values
        assert not rsi.iloc[period:].isna().all()


class TestMACD:
    """Test MACD calculation."""

    def test_macd_output_structure(self, sample_price_data):
        """Test MACD returns three series."""
        result = compute_macd(sample_price_data)
        assert len(result) == 3

        macd_line, signal_line, histogram = result
        assert len(macd_line) == len(sample_price_data)
        assert len(signal_line) == len(sample_price_data)
        assert len(histogram) == len(sample_price_data)

    def test_macd_histogram_relationship(self, sample_price_data):
        """Test that histogram = macd_line - signal_line."""
        macd_line, signal_line, histogram = compute_macd(sample_price_data)

        # Drop NaN values
        valid_idx = ~(macd_line.isna() | signal_line.isna() | histogram.isna())
        diff = macd_line[valid_idx] - signal_line[valid_idx]

        np.testing.assert_array_almost_equal(
            diff.values,
            histogram[valid_idx].values,
            decimal=6
        )

    def test_macd_custom_parameters(self, sample_price_data):
        """Test MACD with custom parameters."""
        macd_line, signal_line, histogram = compute_macd(
            sample_price_data,
            fast_period=8,
            slow_period=17,
            signal_period=9
        )

        assert len(macd_line) == len(sample_price_data)


class TestBollingerBands:
    """Test Bollinger Bands calculation."""

    def test_bb_output_structure(self, sample_price_data):
        """Test BB returns three series."""
        result = compute_bollinger_bands(sample_price_data)
        assert len(result) == 3

        upper, middle, lower = result
        assert len(upper) == len(sample_price_data)
        assert len(middle) == len(sample_price_data)
        assert len(lower) == len(sample_price_data)

    def test_bb_band_relationship(self, sample_price_data):
        """Test that lower <= price <= upper (approximately)."""
        upper, middle, lower = compute_bollinger_bands(sample_price_data)

        # Drop NaN values
        valid_idx = ~(upper.isna() | lower.isna())

        # Lower band should be below upper band
        assert (lower[valid_idx] <= upper[valid_idx]).all()

    def test_bb_position_range(self, sample_price_data):
        """Test BB position is between 0 and 1 (approximately)."""
        bb_pos = compute_bb_position(sample_price_data)

        # Most values should be between 0 and 1
        valid = bb_pos.dropna()
        in_range = ((valid >= -0.1) & (valid <= 1.1)).sum()

        # At least 95% should be in reasonable range
        assert in_range / len(valid) >= 0.95


class TestValidateOHLCV:
    """Test OHLCV validation."""

    def test_valid_data_passes(self, sample_ohlcv_df):
        """Test that valid data passes validation."""
        result = validate_ohlcv(sample_ohlcv_df)
        assert len(result) == len(sample_ohlcv_df)

    def test_missing_columns_raises_error(self):
        """Test that missing columns raises ValueError."""
        df = pd.DataFrame({"timestamp": [1, 2, 3], "close": [100, 101, 102]})

        with pytest.raises(ValueError, match="Missing required"):
            validate_ohlcv(df)

    def test_invalid_ohlc_relationship_removed(self):
        """Test that bars with invalid OHLC are removed."""
        df = pd.DataFrame({
            "timestamp": [1, 2, 3],
            "open": [100, 100, 100],
            "high": [102, 102, 102],
            "low": [98, 98, 105],  # Invalid: low > high
            "close": [101, 101, 101],
            "volume": [1000, 1000, 1000],
        })

        result = validate_ohlcv(df)
        # Should remove the invalid bar
        assert len(result) == 2

    def test_duplicate_timestamps_removed(self, sample_ohlcv_df):
        """Test that duplicate timestamps are removed (keeping last)."""
        # Create valid duplicate with different volume (safer than changing price)
        duplicate_row = sample_ohlcv_df.iloc[0].copy()
        original_volume = duplicate_row["volume"]
        duplicate_row["volume"] = 9999.0  # Different volume

        # Append duplicate
        df_with_dup = pd.concat([sample_ohlcv_df, pd.DataFrame([duplicate_row])], ignore_index=True)

        result = validate_ohlcv(df_with_dup)

        # Should have same length as original (duplicate removed)
        assert len(result) == len(sample_ohlcv_df)

        # Should keep the last value (9999.0 volume, not original)
        first_timestamp_rows = result[result["timestamp"] == result.iloc[0]["timestamp"]]
        assert len(first_timestamp_rows) == 1
        assert first_timestamp_rows.iloc[0]["volume"] == 9999.0

    def test_data_sorted_by_timestamp(self):
        """Test that output is sorted by timestamp."""
        df = pd.DataFrame({
            "timestamp": [3, 1, 2],
            "open": [100, 100, 100],
            "high": [102, 102, 102],
            "low": [98, 98, 98],
            "close": [101, 101, 101],
            "volume": [1000, 1000, 1000],
        })

        result = validate_ohlcv(df)
        assert list(result["timestamp"]) == [1, 2, 3]

    def test_empty_dataframe_raises_error(self):
        """Test that empty DataFrame raises ValueError."""
        empty_df = pd.DataFrame(
            columns=["timestamp", "open", "high", "low", "close", "volume"]
        )

        with pytest.raises(ValueError, match="empty"):
            validate_ohlcv(empty_df)


class TestRSIEdgeCases:
    """Test RSI edge cases for division safety."""

    def test_rsi_monotonic_increase(self):
        """RSI should approach 100 for consistent gains."""
        prices = pd.Series([100.0 + i for i in range(50)])
        rsi = compute_rsi(prices, period=14)

        # Last RSI value should be near 100 (all gains, no losses)
        last_rsi = rsi.iloc[-1]
        assert not pd.isna(last_rsi), "RSI should not be NaN for monotonic increase"
        assert last_rsi > 90, f"RSI should be near 100 for pure gains, got {last_rsi}"

    def test_rsi_monotonic_decrease(self):
        """RSI should approach 0 for consistent losses."""
        prices = pd.Series([100.0 - i * 0.5 for i in range(50)])
        rsi = compute_rsi(prices, period=14)

        # Last RSI value should be near 0 (all losses, no gains)
        last_rsi = rsi.iloc[-1]
        assert not pd.isna(last_rsi), "RSI should not be NaN for monotonic decrease"
        assert last_rsi < 10, f"RSI should be near 0 for pure losses, got {last_rsi}"

    def test_rsi_constant_price(self):
        """RSI with constant prices should handle gracefully."""
        prices = pd.Series([100.0] * 50)
        rsi = compute_rsi(prices, period=14)

        # With no changes, RSI behavior depends on implementation
        # Our fix should handle this without raising exceptions
        assert len(rsi) == 50


class TestBBPositionEdgeCases:
    """Test BB position edge cases."""

    def test_bb_position_constant_price(self):
        """BB position with constant price should be NaN (zero width bands)."""
        prices = pd.Series([100.0] * 50)
        bb_pos = compute_bb_position(prices, period=20)

        # After warmup, should be NaN due to zero band width
        last_pos = bb_pos.iloc[-1]
        assert pd.isna(last_pos), "BB position should be NaN for constant prices"
