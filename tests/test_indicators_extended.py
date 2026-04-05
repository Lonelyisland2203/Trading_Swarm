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

        # Verify Tenkan-sen calculation at bar 8 (first non-NaN)
        expected_tenkan = (
            sample_ohlcv["high"].iloc[0:9].max() +
            sample_ohlcv["low"].iloc[0:9].min()
        ) / 2.0
        assert abs(components['tenkan_sen'].iloc[8] - expected_tenkan) < 1e-6

        # Verify Kijun-sen calculation at bar 25 (first non-NaN)
        expected_kijun = (
            sample_ohlcv["high"].iloc[0:26].max() +
            sample_ohlcv["low"].iloc[0:26].min()
        ) / 2.0
        assert abs(components['kijun_sen'].iloc[25] - expected_kijun) < 1e-6

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

        # All components should equal 100.0 where not NaN
        assert abs(components['tenkan_sen'].iloc[-1] - 100.0) < 1e-6
        assert abs(components['kijun_sen'].iloc[-1] - 100.0) < 1e-6

        # Senkou spans shifted forward 26: first valid at index 51
        # (senkou_span_a is (tenkan+kijun)/2 first valid at 25, then shifted 26 -> index 51)
        assert pd.isna(components['senkou_span_a'].iloc[50])
        assert abs(components['senkou_span_a'].iloc[51] - 100.0) < 1e-6

        # Chikou span shifted backward 26: last 26 are NaN
        assert pd.isna(components['chikou_span'].iloc[-1])
        assert abs(components['chikou_span'].iloc[0] - 100.0) < 1e-6

    def test_ichimoku_cloud_custom_periods(self, sample_ohlcv):
        """Test Ichimoku with non-standard periods."""
        from data.indicators import compute_ichimoku_cloud

        # Use shorter periods to test parameter handling
        components = compute_ichimoku_cloud(
            sample_ohlcv["high"],
            sample_ohlcv["low"],
            sample_ohlcv["close"],
            tenkan_period=7,
            kijun_period=22,
            senkou_span_b_period=44
        )

        # Verify periods were actually used by checking NaN positions
        # Tenkan (7-period) should have first 6 NaN, then valid at index 6
        assert components['tenkan_sen'].iloc[:6].isna().all()
        assert not pd.isna(components['tenkan_sen'].iloc[6])

        # Kijun (22-period) should have first 21 NaN, then valid at index 21
        assert components['kijun_sen'].iloc[:21].isna().all()
        assert not pd.isna(components['kijun_sen'].iloc[21])


@pytest.fixture
def trending_series():
    """Trending price series for KAMA testing."""
    # Strong uptrend: high efficiency ratio
    return pd.Series([100 + i for i in range(30)])


@pytest.fixture
def choppy_series():
    """Choppy price series for KAMA testing."""
    # Alternating up/down: low efficiency ratio
    return pd.Series([100 + (i % 2) * 2 for i in range(30)])


class TestKAMA:
    """Tests for Kaufman Adaptive Moving Average."""

    def test_kama_trending_market(self, trending_series):
        """KAMA in trending market stays responsive (close to price)."""
        from data.indicators import compute_kama

        result = compute_kama(trending_series, period=10)

        # Check return type
        assert isinstance(result, pd.Series)
        assert len(result) == 30

        # Last value should not be NaN (sufficient data)
        assert not pd.isna(result.iloc[-1])

        # In strong trend, KAMA should be close to current price
        # (efficiency ratio high, smoothing constant closer to fast)
        assert abs(result.iloc[-1] - trending_series.iloc[-1]) < 5.0

        # Verify SMA initialization at first valid index (bar 10)
        expected_init = trending_series.iloc[:11].mean()  # bars 0-10 inclusive
        assert abs(result.iloc[10] - expected_init) < 1e-6

    def test_kama_choppy_market(self, choppy_series):
        """KAMA in choppy market smooths aggressively."""
        from data.indicators import compute_kama

        result = compute_kama(choppy_series, period=10)

        # Last value should not be NaN
        assert not pd.isna(result.iloc[-1])

        # In choppy market, KAMA should lag behind current price
        # (efficiency ratio low, smoothing constant closer to slow)
        # Current price alternates 100/102, KAMA should be between them
        assert 99.0 < result.iloc[-1] < 103.0

        # Verify SMA initialization at first valid index (bar 10)
        expected_init = choppy_series.iloc[:11].mean()  # bars 0-10 inclusive
        assert abs(result.iloc[10] - expected_init) < 1e-6

    def test_kama_warmup_period(self):
        """KAMA with insufficient data returns NaN."""
        from data.indicators import compute_kama

        short_series = pd.Series([100, 101, 102])
        result = compute_kama(short_series, period=10)

        # Not enough data for period=10
        assert pd.isna(result.iloc[-1])

    def test_kama_constant_price(self):
        """KAMA with constant price should equal that price."""
        from data.indicators import compute_kama

        constant = pd.Series([100.0] * 30)
        result = compute_kama(constant, period=10)

        # After warmup (period + 1), KAMA should converge to constant price
        assert not pd.isna(result.iloc[-1])
        assert abs(result.iloc[-1] - 100.0) < 1e-6

        # First valid value should also be 100 (SMA of 100s = 100)
        assert abs(result.iloc[10] - 100.0) < 1e-6

    def test_kama_custom_parameters(self, trending_series):
        """Test KAMA with non-default parameters."""
        from data.indicators import compute_kama

        # Very fast smoothing (fast_ema=1 gives higher SC)
        fast_result = compute_kama(trending_series, period=5, fast_ema=1, slow_ema=30)

        # Should have valid data after period=5
        assert fast_result.iloc[:5].isna().all()
        assert not pd.isna(fast_result.iloc[5])

        # Faster smoothing should track price more closely in trends
        assert abs(fast_result.iloc[-1] - trending_series.iloc[-1]) < 3.0
