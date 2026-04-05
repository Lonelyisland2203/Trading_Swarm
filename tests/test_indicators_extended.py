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
        from data.indicators import compute_donchian_channels

        upper, middle, lower = compute_donchian_channels(
            sample_ohlcv["high"], sample_ohlcv["low"]
        )

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
        from data.indicators import compute_donchian_channels

        upper, middle, lower = compute_donchian_channels(
            sample_ohlcv["high"], sample_ohlcv["low"], period=10
        )

        # First 9 values should be NaN
        assert upper.iloc[:9].isna().all()

        # Bar 9 (index 9, 10 bars total)
        assert upper.iloc[9] == sample_ohlcv["high"].iloc[:10].max()
        assert lower.iloc[9] == sample_ohlcv["low"].iloc[:10].min()

    def test_donchian_channels_long_default(self, sample_ohlcv):
        """Test compute_donchian_channels_long with 55-period default."""
        from data.indicators import compute_donchian_channels_long

        upper, middle, lower = compute_donchian_channels_long(
            sample_ohlcv["high"], sample_ohlcv["low"]
        )

        # First 54 values should be NaN (need 55 bars)
        # But we only have 50 bars, so all should be NaN
        assert upper.isna().all()
        assert middle.isna().all()
        assert lower.isna().all()

    def test_donchian_channels_trending_market(self):
        """Test Donchian Channels in trending market."""
        from data.indicators import compute_donchian_channels

        # Uptrend: steadily increasing highs
        high = pd.Series([100 + i for i in range(30)])
        low = pd.Series([98 + i for i in range(30)])

        upper, middle, lower = compute_donchian_channels(high, low, period=10)

        # Upper should be increasing in uptrend
        assert upper.iloc[20] > upper.iloc[10]
        # Lower should lag behind upper
        assert lower.iloc[20] < upper.iloc[20]

    def test_donchian_channels_constant_price(self):
        """Test Donchian Channels with constant price (no volatility)."""
        from data.indicators import compute_donchian_channels

        high = pd.Series([100.0] * 30)
        low = pd.Series([100.0] * 30)

        upper, middle, lower = compute_donchian_channels(high, low, period=10)

        # All bands should converge to same price when no volatility
        assert not pd.isna(upper.iloc[9])
        assert upper.iloc[9] == 100.0
        assert middle.iloc[9] == 100.0
        assert lower.iloc[9] == 100.0


class TestIchimokuCloud:
    """Tests for Ichimoku Cloud indicator."""

    def test_ichimoku_cloud_standard(self, sample_ohlcv):
        """Ichimoku cloud with standard parameters on 50 bars."""
        from data.indicators import compute_ichimoku_cloud

        components = compute_ichimoku_cloud(
            sample_ohlcv["high"],
            sample_ohlcv["low"],
            sample_ohlcv["close"]
        )

        # Check return type
        assert isinstance(components, dict)

        # Check all components present
        expected_keys = {
            'tenkan_sen', 'kijun_sen', 'senkou_span_a',
            'senkou_span_b', 'chikou_span'
        }
        assert set(components.keys()) == expected_keys

        # Check all are Series
        for key, series in components.items():
            assert isinstance(series, pd.Series)
            assert len(series) == 50

        # Tenkan-sen and Kijun-sen should not be NaN at end (sufficient data)
        assert not pd.isna(components['tenkan_sen'].iloc[-1])
        assert not pd.isna(components['kijun_sen'].iloc[-1])

    def test_ichimoku_cloud_insufficient_data(self):
        """Ichimoku with minimal data returns NaN."""
        from data.indicators import compute_ichimoku_cloud

        high = pd.Series([100, 101, 102])
        low = pd.Series([98, 99, 100])
        close = pd.Series([99, 100, 101])

        components = compute_ichimoku_cloud(high, low, close)

        # Tenkan (9-period) should be NaN with only 3 bars
        assert pd.isna(components['tenkan_sen'].iloc[-1])

    def test_ichimoku_cloud_flat_price(self):
        """Ichimoku with flat price returns same values for all lines."""
        from data.indicators import compute_ichimoku_cloud

        high = pd.Series([100.0] * 60)
        low = pd.Series([100.0] * 60)
        close = pd.Series([100.0] * 60)

        components = compute_ichimoku_cloud(high, low, close)

        # All lines should equal 100 (except shifted ones which may be NaN)
        assert abs(components['tenkan_sen'].iloc[-1] - 100.0) < 1e-6
        assert abs(components['kijun_sen'].iloc[-1] - 100.0) < 1e-6
