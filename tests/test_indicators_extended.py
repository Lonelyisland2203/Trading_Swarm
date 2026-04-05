"""Extended indicator tests for trading system."""

import pandas as pd
import pytest


@pytest.fixture
def sample_ohlcv():
    """50-bar sample OHLCV data for testing."""
    dates = pd.date_range(start="2024-01-01", periods=50, freq="1h")
    return pd.DataFrame({
        "timestamp": dates,
        "open": [100.0 + i * 0.5 for i in range(50)],
        "high": [102.0 + i * 0.5 for i in range(50)],
        "low": [98.0 + i * 0.5 for i in range(50)],
        "close": [101.0 + i * 0.5 for i in range(50)],
        "volume": [1000.0 for _ in range(50)],
    })


class TestDonchianChannels:
    """Tests for Donchian Channels indicators."""

    def test_donchian_channels_default_period(self, sample_ohlcv):
        """Test Donchian Channels with default 20-period."""
        from data.indicators import donchian_channels

        upper, middle, lower = donchian_channels(sample_ohlcv)

        assert isinstance(upper, pd.Series)
        assert isinstance(middle, pd.Series)
        assert isinstance(lower, pd.Series)
        assert len(upper) == len(sample_ohlcv)

        # First 19 values should be NaN
        assert upper.iloc[:19].isna().all()
        assert middle.iloc[:19].isna().all()
        assert lower.iloc[:19].isna().all()

        # Bar 19 (index 19, 20 bars total)
        assert upper.iloc[19] == sample_ohlcv["high"].iloc[:20].max()
        assert lower.iloc[19] == sample_ohlcv["low"].iloc[:20].min()
        assert middle.iloc[19] == (upper.iloc[19] + lower.iloc[19]) / 2.0

    def test_donchian_channels_custom_period(self, sample_ohlcv):
        """Test Donchian Channels with custom period."""
        from data.indicators import donchian_channels

        upper, middle, lower = donchian_channels(sample_ohlcv, period=10)

        # First 9 values should be NaN
        assert upper.iloc[:9].isna().all()

        # Bar 9 (index 9, 10 bars total)
        assert upper.iloc[9] == sample_ohlcv["high"].iloc[:10].max()
        assert lower.iloc[9] == sample_ohlcv["low"].iloc[:10].min()

    def test_donchian_channels_long_default(self, sample_ohlcv):
        """Test donchian_channels_long with 55-period default."""
        from data.indicators import donchian_channels_long

        upper, middle, lower = donchian_channels_long(sample_ohlcv)

        # First 54 values should be NaN (need 55 bars)
        # But we only have 50 bars, so all should be NaN
        assert upper.isna().all()
        assert middle.isna().all()
        assert lower.isna().all()

    def test_donchian_channels_trending_market(self):
        """Test Donchian Channels in trending market."""
        from data.indicators import donchian_channels

        # Uptrend: steadily increasing highs
        df = pd.DataFrame({
            "high": [100 + i for i in range(30)],
            "low": [98 + i for i in range(30)],
        })

        upper, middle, lower = donchian_channels(df, period=10)

        # Upper should be increasing in uptrend
        assert upper.iloc[20] > upper.iloc[10]
        # Lower should lag behind upper
        assert lower.iloc[20] < upper.iloc[20]
