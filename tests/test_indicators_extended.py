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


class TestVolumeIndicators:
    """Tests for volume-based indicators."""

    def test_obv_accumulation(self, sample_ohlcv):
        """OBV accumulates volume on up days."""
        from data.indicators import compute_obv

        result = compute_obv(sample_ohlcv["close"], sample_ohlcv["volume"])

        assert isinstance(result, pd.Series)
        assert len(result) == 50
        assert result.iloc[0] == 0
        assert not pd.isna(result.iloc[-1])

    def test_obv_flat_price(self):
        """OBV with flat price stays at zero."""
        from data.indicators import compute_obv

        close = pd.Series([100.0] * 30)
        volume = pd.Series([1000.0] * 30)

        result = compute_obv(close, volume)
        assert result.iloc[-1] == 0

    def test_obv_zero_volume(self):
        """OBV handles zero volume gracefully."""
        from data.indicators import compute_obv

        close = pd.Series([100 + i for i in range(30)])
        volume = pd.Series([0.0] * 30)

        result = compute_obv(close, volume)
        assert result.iloc[-1] == 0

    def test_cmf_standard(self, sample_ohlcv):
        """CMF with 20-period on 50 bars."""
        from data.indicators import compute_cmf

        result = compute_cmf(
            sample_ohlcv["high"],
            sample_ohlcv["low"],
            sample_ohlcv["close"],
            sample_ohlcv["volume"],
            period=20
        )

        assert isinstance(result, pd.Series)
        assert len(result) == 50
        assert -1.0 <= result.iloc[-1] <= 1.0

    def test_cmf_zero_range_bars(self):
        """CMF handles bars with zero range (high=low)."""
        from data.indicators import compute_cmf

        high = pd.Series([100.0] * 30)
        low = pd.Series([100.0] * 30)
        close = pd.Series([100.0] * 30)
        volume = pd.Series([1000.0] * 30)

        result = compute_cmf(high, low, close, volume, period=20)
        assert pd.isna(result.iloc[-1]) or result.iloc[-1] == 0.0

    def test_cmf_insufficient_data(self):
        """CMF with less than period bars returns NaN."""
        from data.indicators import compute_cmf

        high = pd.Series([102, 103, 104])
        low = pd.Series([98, 99, 100])
        close = pd.Series([100, 101, 102])
        volume = pd.Series([1000, 1100, 1200])

        result = compute_cmf(high, low, close, volume, period=20)
        assert pd.isna(result.iloc[-1])

    def test_mfi_standard(self, sample_ohlcv):
        """MFI with 14-period on 50 bars."""
        from data.indicators import compute_mfi

        result = compute_mfi(
            sample_ohlcv["high"],
            sample_ohlcv["low"],
            sample_ohlcv["close"],
            sample_ohlcv["volume"],
            period=14
        )

        assert isinstance(result, pd.Series)
        assert len(result) == 50
        assert 0.0 <= result.iloc[-1] <= 100.0

    def test_mfi_zero_volume(self):
        """MFI handles zero volume gracefully."""
        from data.indicators import compute_mfi

        high = pd.Series([102] * 30)
        low = pd.Series([98] * 30)
        close = pd.Series([100 + i * 0.1 for i in range(30)])
        volume = pd.Series([0.0] * 30)

        result = compute_mfi(high, low, close, volume, period=14)
        assert pd.isna(result.iloc[-1])

    def test_mfi_insufficient_data(self):
        """MFI with less than period bars returns NaN."""
        from data.indicators import compute_mfi

        high = pd.Series([102, 103, 104])
        low = pd.Series([98, 99, 100])
        close = pd.Series([100, 101, 102])
        volume = pd.Series([1000, 1100, 1200])

        result = compute_mfi(high, low, close, volume, period=14)
        assert pd.isna(result.iloc[-1])

    def test_vwap_standard(self, sample_ohlcv):
        """VWAP cumulative calculation on 50 bars."""
        from data.indicators import compute_vwap

        result = compute_vwap(
            sample_ohlcv["high"],
            sample_ohlcv["low"],
            sample_ohlcv["close"],
            sample_ohlcv["volume"]
        )

        assert isinstance(result, pd.Series)
        assert len(result) == 50
        assert not pd.isna(result.iloc[0])

        # VWAP should be within reasonable range
        typical_price = (
            sample_ohlcv["high"].iloc[-1] +
            sample_ohlcv["low"].iloc[-1] +
            sample_ohlcv["close"].iloc[-1]
        ) / 3.0
        assert abs(result.iloc[-1] - typical_price) < 20.0

    def test_vwap_zero_volume(self):
        """VWAP handles zero volume gracefully."""
        from data.indicators import compute_vwap

        high = pd.Series([102] * 30)
        low = pd.Series([98] * 30)
        close = pd.Series([100] * 30)
        volume = pd.Series([0.0] * 30)

        result = compute_vwap(high, low, close, volume)
        assert pd.isna(result.iloc[0])

    def test_vwap_single_bar(self):
        """VWAP with single bar equals typical price."""
        from data.indicators import compute_vwap

        high = pd.Series([102.0])
        low = pd.Series([98.0])
        close = pd.Series([100.0])
        volume = pd.Series([1000.0])

        result = compute_vwap(high, low, close, volume)

        expected = (102.0 + 98.0 + 100.0) / 3.0
        assert abs(result.iloc[0] - expected) < 1e-6

    def test_cmf_concrete_calculation(self):
        """Verify CMF calculation with known values."""
        from data.indicators import compute_cmf

        # Simple scenario with manual calculation
        high = pd.Series([105, 110, 108, 112, 115])
        low = pd.Series([95, 100, 102, 104, 105])
        close = pd.Series([100, 108, 103, 110, 112])
        volume = pd.Series([1000, 1500, 1200, 1800, 1600])

        result = compute_cmf(high, low, close, volume, period=3)

        # Bar 2 (first valid with period=3): bars 0-2
        # MFM[0] = ((100-95)-(105-100))/(105-95) = 0.0
        # MFM[1] = ((108-100)-(110-108))/(110-100) = 0.6
        # MFM[2] = ((103-102)-(108-103))/(108-102) = -0.6667
        # CMF = sum(MFM*vol) / sum(vol)
        # = (0*1000 + 0.6*1500 + -0.6667*1200) / (1000+1500+1200)
        # = (0 + 900 - 800) / 3700 = 100 / 3700 = 0.027027
        expected_cmf = 0.027027
        assert abs(result.iloc[2] - expected_cmf) < 1e-5

    def test_vwap_first_value_precision(self, sample_ohlcv):
        """Verify VWAP first value equals typical price."""
        from data.indicators import compute_vwap

        result = compute_vwap(
            sample_ohlcv["high"],
            sample_ohlcv["low"],
            sample_ohlcv["close"],
            sample_ohlcv["volume"]
        )

        # Verify first value equals typical price on bar 0
        expected_first = (
            sample_ohlcv["high"].iloc[0] +
            sample_ohlcv["low"].iloc[0] +
            sample_ohlcv["close"].iloc[0]
        ) / 3.0
        assert abs(result.iloc[0] - expected_first) < 1e-6

    def test_cmf_custom_period(self, sample_ohlcv):
        """Test CMF with custom period."""
        from data.indicators import compute_cmf

        result = compute_cmf(
            sample_ohlcv["high"],
            sample_ohlcv["low"],
            sample_ohlcv["close"],
            sample_ohlcv["volume"],
            period=10
        )

        # First 9 values should be NaN (need 10 bars)
        assert result.iloc[:9].isna().all()
        assert not pd.isna(result.iloc[9])

    def test_mfi_custom_period(self, sample_ohlcv):
        """Test MFI with custom period."""
        from data.indicators import compute_mfi

        result = compute_mfi(
            sample_ohlcv["high"],
            sample_ohlcv["low"],
            sample_ohlcv["close"],
            sample_ohlcv["volume"],
            period=20
        )

        # First 19 values should be NaN (need 20 bars for rolling window)
        assert result.iloc[:19].isna().all()
        assert not pd.isna(result.iloc[19])

    def test_cmf_extreme_values(self):
        """Test CMF with extreme close positions."""
        from data.indicators import compute_cmf

        # All closes at high (maximum bullish) should give CMF near +1
        high = pd.Series([110, 115, 120, 125, 130])
        low = pd.Series([100, 105, 110, 115, 120])
        close = pd.Series([110, 115, 120, 125, 130])  # All at high
        volume = pd.Series([1000] * 5)

        result = compute_cmf(high, low, close, volume, period=3)

        # MFM when close=high is +1.0, so CMF should be +1.0
        assert abs(result.iloc[-1] - 1.0) < 1e-6


class TestVolatilityIndicators:
    """Tests for volatility-based indicators."""

    def test_atr_standard(self, sample_ohlcv):
        """ATR with 14-period on 50 bars."""
        from data.indicators import compute_atr

        result = compute_atr(
            sample_ohlcv["high"],
            sample_ohlcv["low"],
            sample_ohlcv["close"],
            period=14
        )

        assert isinstance(result, pd.Series)
        assert len(result) == 50
        assert result.iloc[-1] > 0

    def test_atr_zero_range_bars(self):
        """ATR handles bars with zero range gracefully."""
        from data.indicators import compute_atr

        high = pd.Series([100.0] * 30)
        low = pd.Series([100.0] * 30)
        close = pd.Series([100.0] * 30)

        result = compute_atr(high, low, close, period=14)
        assert result.iloc[-1] < 1e-6

    def test_atr_single_bar(self):
        """ATR with single bar returns NaN."""
        from data.indicators import compute_atr

        high = pd.Series([102.0])
        low = pd.Series([98.0])
        close = pd.Series([100.0])

        result = compute_atr(high, low, close, period=14)
        assert pd.isna(result.iloc[0])

    def test_atr_normalized_standard(self, sample_ohlcv):
        """ATR normalized with 14-period on 50 bars."""
        from data.indicators import compute_atr_normalized

        result = compute_atr_normalized(
            sample_ohlcv["high"],
            sample_ohlcv["low"],
            sample_ohlcv["close"],
            period=14
        )

        assert isinstance(result, pd.Series)
        assert len(result) == 50
        assert result.iloc[-1] > 0

    def test_bb_width_standard(self):
        """BB width calculation on 50 bars."""
        from data.indicators import compute_bb_width

        close = pd.Series([100 + i * 0.5 for i in range(50)])
        result = compute_bb_width(close, period=20)

        assert isinstance(result, pd.Series)
        assert len(result) == 50
        assert result.iloc[-1] > 0

    def test_bb_width_flat_price(self):
        """BB width with flat price returns zero."""
        from data.indicators import compute_bb_width

        close = pd.Series([100.0] * 30)
        result = compute_bb_width(close, period=20)
        assert result.iloc[-1] < 1e-6

    def test_bb_width_insufficient_data(self):
        """BB width with less than period bars returns NaN."""
        from data.indicators import compute_bb_width

        close = pd.Series([100, 101, 102])
        result = compute_bb_width(close, period=20)
        assert pd.isna(result.iloc[-1])

    def test_keltner_channels_standard(self, sample_ohlcv):
        """Keltner channels with standard parameters."""
        from data.indicators import compute_keltner_channels

        upper, middle, lower = compute_keltner_channels(
            sample_ohlcv["high"],
            sample_ohlcv["low"],
            sample_ohlcv["close"]
        )

        assert isinstance(upper, pd.Series)
        assert isinstance(middle, pd.Series)
        assert isinstance(lower, pd.Series)
        assert upper.iloc[-1] > middle.iloc[-1]
        assert middle.iloc[-1] > lower.iloc[-1]

    def test_keltner_channels_insufficient_data(self):
        """Keltner with minimal data returns NaN."""
        from data.indicators import compute_keltner_channels

        high = pd.Series([102, 103, 104])
        low = pd.Series([98, 99, 100])
        close = pd.Series([100, 101, 102])

        upper, middle, lower = compute_keltner_channels(
            high, low, close,
            ema_period=20, atr_period=10
        )
        assert pd.isna(upper.iloc[-1])

    def test_keltner_channels_zero_atr(self):
        """Keltner with zero ATR has upper=middle=lower."""
        from data.indicators import compute_keltner_channels

        high = pd.Series([100.0] * 30)
        low = pd.Series([100.0] * 30)
        close = pd.Series([100.0] * 30)

        upper, middle, lower = compute_keltner_channels(
            high, low, close,
            ema_period=20, atr_period=10
        )

        assert abs(upper.iloc[-1] - middle.iloc[-1]) < 1e-6
        assert abs(middle.iloc[-1] - lower.iloc[-1]) < 1e-6

    def test_donchian_width_standard(self, sample_ohlcv):
        """Donchian width calculation on 50 bars."""
        from data.indicators import compute_donchian_width

        result = compute_donchian_width(
            sample_ohlcv["high"],
            sample_ohlcv["low"],
            period=20
        )

        assert isinstance(result, pd.Series)
        assert len(result) == 50
        assert result.iloc[-1] > 0

    def test_donchian_width_flat_price(self):
        """Donchian width with flat price returns zero."""
        from data.indicators import compute_donchian_width

        high = pd.Series([100.0] * 30)
        low = pd.Series([100.0] * 30)

        result = compute_donchian_width(high, low, period=20)
        assert result.iloc[-1] < 1e-6

    def test_donchian_width_insufficient_data(self):
        """Donchian width with less than period bars returns NaN."""
        from data.indicators import compute_donchian_width

        high = pd.Series([102, 103, 104])
        low = pd.Series([98, 99, 100])

        result = compute_donchian_width(high, low, period=20)
        assert pd.isna(result.iloc[-1])


class TestTTMSqueeze:
    """Tests for TTM Squeeze indicator."""

    def test_ttm_squeeze_on(self):
        """Scenario: Flat price (tight BB) + expanding range (wide KC) = squeeze ON."""
        from data.indicators import compute_bollinger_bands, compute_keltner_channels, ttm_squeeze

        # Flat price series creates tight Bollinger Bands
        close = pd.Series([100.0] * 30)

        # Expanding range creates wide Keltner Channels
        # Inject high volatility by manipulating ATR via high-low range
        high_volatile = pd.Series([100.0 + i for i in range(30)])
        low_volatile = pd.Series([99.0 - i for i in range(30)])

        # Calculate BB on flat price (tight bands)
        bb_upper, bb_middle, bb_lower = compute_bollinger_bands(close, period=20)

        # Calculate KC on volatile range (wide channels)
        kc_upper, kc_middle, kc_lower = compute_keltner_channels(
            high_volatile, low_volatile, close, ema_period=20, atr_period=10
        )

        # TTM Squeeze: BB inside KC = squeeze ON
        result = ttm_squeeze(bb_upper, bb_lower, kc_upper, kc_lower)

        assert isinstance(result, pd.Series)
        assert len(result) == 30

        # Last value should be True (squeeze ON: BB inside KC)
        # BB is tight (close to 100), KC is wide (expanding)
        assert result.iloc[-1] == True

    def test_ttm_squeeze_off(self):
        """Scenario: Volatile price (wide BB) + narrow range (tight KC) = squeeze OFF."""
        from data.indicators import compute_bollinger_bands, compute_keltner_channels, ttm_squeeze

        # Volatile price creates wide Bollinger Bands
        close = pd.Series([100.0 + (i % 2) * 10 for i in range(30)])  # Oscillating 100, 110, 100, 110...

        # Calculate BB on volatile price (wide bands)
        bb_upper, bb_middle, bb_lower = compute_bollinger_bands(close, period=20)

        # For KC, use flat price with very tight range to get narrow channels
        # Key: flat close and minimal high-low range creates minimal ATR
        close_flat = pd.Series([105.0] * 30)  # Centered near middle of oscillation
        high_narrow = pd.Series([105.1] * 30)
        low_narrow = pd.Series([104.9] * 30)

        # Calculate KC on flat price with minimal range (tight channels)
        kc_upper, kc_middle, kc_lower = compute_keltner_channels(
            high_narrow, low_narrow, close_flat, ema_period=20, atr_period=10
        )

        # TTM Squeeze: BB outside KC = squeeze OFF
        result = ttm_squeeze(bb_upper, bb_lower, kc_upper, kc_lower)

        assert isinstance(result, pd.Series)
        assert len(result) == 30

        # Last value should be False (squeeze OFF: BB outside KC)
        # BB is wide (volatile price, ~100-110 range)
        # KC is tight (flat price, ~105 +/- tiny ATR)
        assert result.iloc[-1] == False

    def test_ttm_squeeze_explicit_values(self):
        """Verify TTM squeeze with explicit band values."""
        from data.indicators import ttm_squeeze

        # Explicit test data - no dependency on other indicator functions
        bb_upper = pd.Series([105.0, 108.0, 103.0, 106.0])
        bb_lower = pd.Series([95.0, 92.0, 97.0, 94.0])
        kc_upper = pd.Series([110.0, 107.0, 110.0, 105.0])
        kc_lower = pd.Series([90.0, 93.0, 90.0, 95.0])

        result = ttm_squeeze(bb_upper, bb_lower, kc_upper, kc_lower)

        # Bar 0: BB [95, 105] inside KC [90, 110] -> True
        assert result.iloc[0] == True
        # Bar 1: BB lower 92 < KC lower 93 -> False (not completely inside)
        assert result.iloc[1] == False
        # Bar 2: BB [97, 103] inside KC [90, 110] -> True
        assert result.iloc[2] == True
        # Bar 3: BB upper 106 > KC upper 105 -> False (not completely inside)
        assert result.iloc[3] == False

    def test_ttm_squeeze_boundary_equal_values(self):
        """When BB touches KC exactly, squeeze is OFF (not strictly inside)."""
        from data.indicators import ttm_squeeze

        # BB upper equals KC upper exactly
        bb_upper = pd.Series([110.0, 109.0])
        bb_lower = pd.Series([95.0, 95.0])
        kc_upper = pd.Series([110.0, 110.0])
        kc_lower = pd.Series([90.0, 90.0])

        result = ttm_squeeze(bb_upper, bb_lower, kc_upper, kc_lower)

        # Bar 0: bb_upper == kc_upper, strict < fails -> False
        assert result.iloc[0] == False
        # Bar 1: bb_upper < kc_upper, bb_lower > kc_lower -> True
        assert result.iloc[1] == True
