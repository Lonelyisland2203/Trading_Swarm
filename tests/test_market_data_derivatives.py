"""Tests for derivatives data fetching (funding rates and open interest)."""

import pandas as pd
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from data.market_data import MarketDataService, ExchangeClient


class TestAdaptiveTTL:
    """Test adaptive cache TTL calculation."""

    @patch("data.market_data.ExchangeClient")
    @patch("data.market_data.AsyncDiskCache")
    def test_historical_data_gets_long_ttl(self, mock_cache, mock_exchange):
        """Data older than 7 days gets 24-hour TTL."""
        service = MarketDataService()
        now = datetime.now(timezone.utc)
        old_timestamp = now - timedelta(days=10)

        ttl = service._compute_adaptive_ttl(old_timestamp, now)

        assert ttl == 86400  # 24 hours in seconds

    @patch("data.market_data.ExchangeClient")
    @patch("data.market_data.AsyncDiskCache")
    def test_recent_data_gets_medium_ttl(self, mock_cache, mock_exchange):
        """Data between 1 hour and 7 days old gets 2-hour TTL."""
        service = MarketDataService()
        now = datetime.now(timezone.utc)
        recent_timestamp = now - timedelta(hours=12)

        ttl = service._compute_adaptive_ttl(recent_timestamp, now)

        assert ttl == 7200  # 2 hours in seconds

    @patch("data.market_data.ExchangeClient")
    @patch("data.market_data.AsyncDiskCache")
    def test_live_data_gets_short_ttl(self, mock_cache, mock_exchange):
        """Data less than 1 hour old gets 30-minute TTL."""
        service = MarketDataService()
        now = datetime.now(timezone.utc)
        live_timestamp = now - timedelta(minutes=15)

        ttl = service._compute_adaptive_ttl(live_timestamp, now)

        assert ttl == 1800  # 30 minutes in seconds


class TestPerpetualMapping:
    """Test spot to perpetual symbol mapping."""

    @patch("data.market_data.ExchangeClient")
    @patch("data.market_data.AsyncDiskCache")
    async def test_load_perpetual_markets_caches_mapping(self, mock_cache, mock_exchange):
        """First call to load markets gets from API, subsequent calls use cache."""
        # Mock load_markets response (synchronous method)
        mock_exchange_instance = MagicMock()
        mock_exchange.return_value = mock_exchange_instance
        mock_exchange_instance.load_markets.return_value = {
            "BTC/USDT": {"type": "spot", "symbol": "BTC/USDT"},
            "BTC/USDT:USDT": {"type": "swap", "symbol": "BTC/USDT:USDT", "settle": "USDT"},
            "ETH/USDT": {"type": "spot", "symbol": "ETH/USDT"},
            "ETH/USDT:USDT": {"type": "swap", "symbol": "ETH/USDT:USDT", "settle": "USDT"},
        }

        # Mock cache get/set (async methods)
        mock_cache_instance = MagicMock()
        mock_cache.return_value = mock_cache_instance
        mock_cache_instance.get = AsyncMock(return_value=None)  # Not cached yet
        mock_cache_instance.set = AsyncMock()

        service = MarketDataService()

        # First call - should fetch from API
        mapping = await service._load_perpetual_markets()

        assert mapping == {"BTC/USDT": "BTC/USDT:USDT", "ETH/USDT": "ETH/USDT:USDT"}
        mock_exchange_instance.load_markets.assert_called_once()

        # Should have cached the result
        mock_cache_instance.set.assert_called_once()

    @patch("data.market_data.ExchangeClient")
    @patch("data.market_data.AsyncDiskCache")
    async def test_get_perpetual_symbol_returns_perp_for_spot(self, mock_cache, mock_exchange):
        """Spot symbols are mapped to their perpetual equivalents."""
        mock_exchange_instance = MagicMock()
        mock_exchange.return_value = mock_exchange_instance

        mock_cache_instance = MagicMock()
        mock_cache.return_value = mock_cache_instance
        mock_cache_instance.get = AsyncMock(
            return_value={"BTC/USDT": "BTC/USDT:USDT", "ETH/USDT": "ETH/USDT:USDT"}
        )

        service = MarketDataService()

        perp_symbol = await service._get_perpetual_symbol("BTC/USDT")

        assert perp_symbol == "BTC/USDT:USDT"

    @patch("data.market_data.ExchangeClient")
    @patch("data.market_data.AsyncDiskCache")
    async def test_get_perpetual_symbol_returns_none_if_no_mapping(self, mock_cache, mock_exchange):
        """Returns None if perpetual not found for spot symbol."""
        mock_exchange_instance = MagicMock()
        mock_exchange.return_value = mock_exchange_instance

        mock_cache_instance = MagicMock()
        mock_cache.return_value = mock_cache_instance
        mock_cache_instance.get = AsyncMock(return_value={"BTC/USDT": "BTC/USDT:USDT"})

        service = MarketDataService()

        perp_symbol = await service._get_perpetual_symbol("UNKNOWN/USDT")

        assert perp_symbol is None


class TestExchangeClientDerivatives:
    """Test ExchangeClient methods for derivatives data."""

    @patch("data.market_data.ccxt.binance")
    async def test_fetch_funding_rate_history_success(self, mock_binance_class):
        """Successful funding rate history fetch."""
        mock_exchange = MagicMock()
        mock_binance_class.return_value = mock_exchange

        # Mock CCXT response
        mock_exchange.fetch_funding_rate_history.return_value = [
            {"timestamp": 1609459200000, "fundingRate": 0.0001},
            {"timestamp": 1609488000000, "fundingRate": 0.00015},
        ]

        client = ExchangeClient(exchange_id="binance")

        result = await client.fetch_funding_rate_history(
            "BTC/USDT:USDT", since=1609459200000, limit=100
        )

        assert len(result) == 2
        assert result[0]["fundingRate"] == 0.0001
        mock_exchange.fetch_funding_rate_history.assert_called_once_with(
            "BTC/USDT:USDT", since=1609459200000, limit=100
        )

    @patch("data.market_data.ccxt.binance")
    async def test_fetch_open_interest_history_success(self, mock_binance_class):
        """Successful open interest history fetch."""
        mock_exchange = MagicMock()
        mock_binance_class.return_value = mock_exchange

        # Mock CCXT response
        mock_exchange.fetch_open_interest_history.return_value = [
            {"timestamp": 1609459200000, "openInterestValue": 1000000000},
            {"timestamp": 1609488000000, "openInterestValue": 1050000000},
        ]

        client = ExchangeClient(exchange_id="binance")

        result = await client.fetch_open_interest_history(
            "BTC/USDT:USDT", timeframe="1h", since=1609459200000, limit=100
        )

        assert len(result) == 2
        assert result[0]["openInterestValue"] == 1000000000
        mock_exchange.fetch_open_interest_history.assert_called_once_with(
            "BTC/USDT:USDT", timeframe="1h", since=1609459200000, limit=100
        )


class TestFundingRateFetching:
    """Test funding rate fetching with capability checking."""

    @patch("data.market_data.ExchangeClient")
    @patch("data.market_data.AsyncDiskCache")
    async def test_fetch_funding_rates_checks_capability(self, mock_cache, mock_exchange_class):
        """Method checks exchange capability before fetching."""
        mock_exchange = AsyncMock()
        mock_exchange_class.return_value = mock_exchange
        mock_exchange.exchange.has = {"fetchFundingRateHistory": False}

        mock_cache_instance = AsyncMock()
        mock_cache.return_value = mock_cache_instance
        mock_cache_instance.get.return_value = {"BTC/USDT": "BTC/USDT:USDT"}

        service = MarketDataService()

        result = await service.fetch_funding_rates("BTC/USDT")

        # Should return None when capability not supported
        assert result is None
        # Should NOT call fetch_funding_rate_history
        mock_exchange.fetch_funding_rate_history.assert_not_called()

    @patch("data.market_data.ExchangeClient")
    @patch("data.market_data.AsyncDiskCache")
    async def test_fetch_funding_rates_success(self, mock_cache, mock_exchange_class):
        """Successful funding rate fetch returns DataFrame."""
        mock_exchange = AsyncMock()
        mock_exchange_class.return_value = mock_exchange
        mock_exchange.exchange.has = {"fetchFundingRateHistory": True}
        mock_exchange.fetch_funding_rate_history.return_value = [
            {"timestamp": 1609459200000, "fundingRate": 0.0001},
            {"timestamp": 1609488000000, "fundingRate": 0.00015},
        ]

        mock_cache_instance = AsyncMock()
        mock_cache.return_value = mock_cache_instance
        mock_cache_instance.get.side_effect = [
            {"BTC/USDT": "BTC/USDT:USDT"},  # perpetual mapping
            None,  # funding rates cache miss
        ]

        service = MarketDataService()

        result = await service.fetch_funding_rates("BTC/USDT")

        assert result is not None
        assert len(result) == 2
        assert "timestamp" in result.columns
        assert "funding_rate" in result.columns
        assert result["funding_rate"].iloc[0] == 0.0001

    @patch("data.market_data.ExchangeClient")
    @patch("data.market_data.AsyncDiskCache")
    async def test_fetch_funding_rates_applies_point_in_time_filter(
        self, mock_cache, mock_exchange_class
    ):
        """Point-in-time filter excludes future funding rates."""
        mock_exchange = AsyncMock()
        mock_exchange_class.return_value = mock_exchange
        mock_exchange.exchange.has = {"fetchFundingRateHistory": True}

        # Create mock DataFrame
        import pandas as pd

        mock_df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime([1609459200000, 1609488000000], unit="ms", utc=True),
                "funding_rate": [0.0001, 0.00015],
            }
        )

        mock_cache_instance = AsyncMock()
        mock_cache.return_value = mock_cache_instance
        mock_cache_instance.get.side_effect = [
            {"BTC/USDT": "BTC/USDT:USDT"},  # perpetual mapping
            mock_df,  # funding rates cache hit
        ]

        service = MarketDataService()

        # Filter to only include first timestamp
        as_of = pd.to_datetime(1609459200000, unit="ms", utc=True)
        result = await service.fetch_funding_rates("BTC/USDT", as_of=as_of)

        assert len(result) == 1
        assert result["funding_rate"].iloc[0] == 0.0001

    @patch("data.market_data.ExchangeClient")
    @patch("data.market_data.AsyncDiskCache")
    async def test_fetch_funding_rates_returns_none_for_missing_perpetual(
        self, mock_cache, mock_exchange_class
    ):
        """Returns None if no perpetual symbol found."""
        mock_exchange = AsyncMock()
        mock_exchange_class.return_value = mock_exchange
        mock_exchange.exchange.has = {"fetchFundingRateHistory": True}

        mock_cache_instance = AsyncMock()
        mock_cache.return_value = mock_cache_instance
        mock_cache_instance.get.return_value = {}  # Empty mapping - no perpetual

        service = MarketDataService()

        result = await service.fetch_funding_rates("UNKNOWN/USDT")

        assert result is None

    @patch("data.market_data.ExchangeClient")
    @patch("data.market_data.AsyncDiskCache")
    async def test_fetch_funding_rates_uses_cache(self, mock_cache, mock_exchange_class):
        """Cached funding rates are returned without exchange call."""
        mock_exchange = AsyncMock()
        mock_exchange_class.return_value = mock_exchange
        mock_exchange.exchange.has = {"fetchFundingRateHistory": True}

        # Create mock DataFrame
        import pandas as pd

        mock_df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime([1609459200000], unit="ms", utc=True),
                "funding_rate": [0.0001],
            }
        )

        mock_cache_instance = AsyncMock()
        mock_cache.return_value = mock_cache_instance
        mock_cache_instance.get.side_effect = [
            {"BTC/USDT": "BTC/USDT:USDT"},  # perpetual mapping
            mock_df,  # funding rates cache hit
        ]

        service = MarketDataService()

        result = await service.fetch_funding_rates("BTC/USDT")

        assert result is not None
        assert len(result) == 1
        # Should NOT call exchange since we got cache hit
        mock_exchange.fetch_funding_rate_history.assert_not_called()


class TestOpenInterestFetching:
    """Test open interest fetching with capability checking."""

    @patch("data.market_data.ExchangeClient")
    @patch("data.market_data.AsyncDiskCache")
    async def test_fetch_open_interest_checks_capability(self, mock_cache, mock_exchange_class):
        """Method checks exchange capability before fetching."""
        mock_exchange = AsyncMock()
        mock_exchange_class.return_value = mock_exchange
        mock_exchange.exchange.has = {"fetchOpenInterestHistory": False}

        mock_cache_instance = AsyncMock()
        mock_cache.return_value = mock_cache_instance
        mock_cache_instance.get.return_value = {"BTC/USDT": "BTC/USDT:USDT"}

        service = MarketDataService()

        result = await service.fetch_open_interest("BTC/USDT", timeframe="1h")

        # Should return None when capability not supported
        assert result is None
        mock_exchange.fetch_open_interest_history.assert_not_called()

    @patch("data.market_data.ExchangeClient")
    @patch("data.market_data.AsyncDiskCache")
    async def test_fetch_open_interest_success(self, mock_cache, mock_exchange_class):
        """Successful open interest fetch returns DataFrame."""
        mock_exchange = AsyncMock()
        mock_exchange_class.return_value = mock_exchange
        mock_exchange.exchange.has = {"fetchOpenInterestHistory": True}
        mock_exchange.fetch_open_interest_history.return_value = [
            {"timestamp": 1609459200000, "openInterestValue": 1000000000, "openInterest": 50000},
            {"timestamp": 1609488000000, "openInterestValue": 1050000000, "openInterest": 51000},
        ]

        mock_cache_instance = AsyncMock()
        mock_cache.return_value = mock_cache_instance
        mock_cache_instance.get.side_effect = [
            {"BTC/USDT": "BTC/USDT:USDT"},  # perpetual mapping
            None,  # open interest cache miss
        ]

        service = MarketDataService()

        result = await service.fetch_open_interest("BTC/USDT", timeframe="1h")

        assert result is not None
        assert len(result) == 2
        assert "timestamp" in result.columns
        assert "open_interest_value" in result.columns
        assert "open_interest" in result.columns
        assert result["open_interest_value"].iloc[0] == 1000000000
        assert result["open_interest"].iloc[0] == 50000

    @patch("data.market_data.ExchangeClient")
    @patch("data.market_data.AsyncDiskCache")
    async def test_fetch_open_interest_applies_point_in_time_filter(
        self, mock_cache, mock_exchange_class
    ):
        """Point-in-time filter excludes future open interest data."""
        mock_exchange = AsyncMock()
        mock_exchange_class.return_value = mock_exchange
        mock_exchange.exchange.has = {"fetchOpenInterestHistory": True}

        # Create mock DataFrame
        import pandas as pd

        mock_df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime([1609459200000, 1609488000000], unit="ms", utc=True),
                "open_interest_value": [1000000000, 1050000000],
                "open_interest": [50000, 51000],
            }
        )

        mock_cache_instance = AsyncMock()
        mock_cache.return_value = mock_cache_instance
        mock_cache_instance.get.side_effect = [
            {"BTC/USDT": "BTC/USDT:USDT"},  # perpetual mapping
            mock_df,  # open interest cache hit
        ]

        service = MarketDataService()

        # Filter to only include first timestamp
        as_of = pd.to_datetime(1609459200000, unit="ms", utc=True)
        result = await service.fetch_open_interest("BTC/USDT", timeframe="1h", as_of=as_of)

        assert len(result) == 1
        assert result["open_interest_value"].iloc[0] == 1000000000

    @patch("data.market_data.ExchangeClient")
    @patch("data.market_data.AsyncDiskCache")
    async def test_fetch_open_interest_returns_none_for_missing_perpetual(
        self, mock_cache, mock_exchange_class
    ):
        """Returns None if no perpetual symbol found."""
        mock_exchange = AsyncMock()
        mock_exchange_class.return_value = mock_exchange
        mock_exchange.exchange.has = {"fetchOpenInterestHistory": True}

        mock_cache_instance = AsyncMock()
        mock_cache.return_value = mock_cache_instance
        mock_cache_instance.get.return_value = {}  # Empty mapping - no perpetual

        service = MarketDataService()

        result = await service.fetch_open_interest("UNKNOWN/USDT", timeframe="1h")

        assert result is None

    @patch("data.market_data.ExchangeClient")
    @patch("data.market_data.AsyncDiskCache")
    async def test_fetch_open_interest_uses_cache(self, mock_cache, mock_exchange_class):
        """Cached open interest data is returned without exchange call."""
        mock_exchange = AsyncMock()
        mock_exchange_class.return_value = mock_exchange
        mock_exchange.exchange.has = {"fetchOpenInterestHistory": True}

        # Create mock DataFrame
        import pandas as pd

        mock_df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime([1609459200000], unit="ms", utc=True),
                "open_interest_value": [1000000000],
                "open_interest": [50000],
            }
        )

        mock_cache_instance = AsyncMock()
        mock_cache.return_value = mock_cache_instance
        mock_cache_instance.get.side_effect = [
            {"BTC/USDT": "BTC/USDT:USDT"},  # perpetual mapping
            mock_df,  # open interest cache hit
        ]

        service = MarketDataService()

        result = await service.fetch_open_interest("BTC/USDT", timeframe="1h")

        assert result is not None
        assert len(result) == 1
        # Should NOT call exchange since we got cache hit
        mock_exchange.fetch_open_interest_history.assert_not_called()


class TestGetMarketContext:
    """Test get_market_context() unified API."""

    @patch("data.market_data.ExchangeClient")
    @patch("data.market_data.AsyncDiskCache")
    async def test_get_market_context_returns_all_data_types(self, mock_cache, mock_exchange_class):
        """Returns dict with OHLCV, funding rates, and open interest."""
        # Setup mocks for all data types
        mock_exchange = AsyncMock()
        mock_exchange_class.return_value = mock_exchange

        # Exchange supports all features
        mock_exchange.exchange.has = {
            "fetchOHLCV": True,
            "fetchFundingRateHistory": True,
            "fetchOpenInterestHistory": True,
        }

        # Mock OHLCV response
        mock_exchange.fetch_ohlcv.return_value = [
            [1609459200000, 29000, 29500, 28800, 29200, 1000],
        ]

        # Mock funding rate response
        mock_exchange.fetch_funding_rate_history.return_value = [
            {"timestamp": 1609459200000, "fundingRate": 0.0001},
        ]

        # Mock open interest response
        mock_exchange.fetch_open_interest_history.return_value = [
            {"timestamp": 1609459200000, "openInterestValue": 1000000000},
        ]

        mock_cache_instance = AsyncMock()
        mock_cache.return_value = mock_cache_instance

        # Mock OHLCV cache response (list of dicts format)
        ohlcv_cache = [
            {
                "timestamp": 1609459200000,
                "open": 29000,
                "high": 29500,
                "low": 28800,
                "close": 29200,
                "volume": 1000,
            }
        ]

        # Return perp mapping, then cache responses for all data
        mock_cache_instance.get.side_effect = [
            ohlcv_cache,  # OHLCV cache hit
            {"BTC/USDT": "BTC/USDT:USDT"},  # perp mapping for funding rates
            None,  # funding rates cache miss
            {"BTC/USDT": "BTC/USDT:USDT"},  # perp mapping for open interest
            None,  # open interest cache miss
        ]

        service = MarketDataService()

        result = await service.get_market_context(symbol="BTC/USDT", timeframe="1h", limit=100)

        # Should return dict with all components
        assert "ohlcv_df" in result
        assert "funding_rate" in result
        assert "funding_rate_history" in result
        assert "open_interest" in result
        assert "open_interest_change_pct" in result

        # Verify data types
        import pandas as pd

        assert isinstance(result["ohlcv_df"], pd.DataFrame)
        assert isinstance(result["funding_rate_history"], pd.DataFrame)
        assert isinstance(result["funding_rate"], float)

    @patch("data.market_data.ExchangeClient")
    @patch("data.market_data.AsyncDiskCache")
    async def test_get_market_context_spot_only_fallback(self, mock_cache, mock_exchange_class):
        """Spot-only symbols return None for derivatives data."""
        mock_exchange = AsyncMock()
        mock_exchange_class.return_value = mock_exchange

        # Exchange supports derivatives but symbol has no perpetual
        mock_exchange.exchange.has = {
            "fetchOHLCV": True,
            "fetchFundingRateHistory": True,
            "fetchOpenInterestHistory": True,
        }

        # Mock OHLCV cache response
        ohlcv_cache = [
            {
                "timestamp": 1609459200000,
                "open": 29000,
                "high": 29500,
                "low": 28800,
                "close": 29200,
                "volume": 1000,
            }
        ]

        mock_cache_instance = AsyncMock()
        mock_cache.return_value = mock_cache_instance
        mock_cache_instance.get.side_effect = [
            ohlcv_cache,  # OHLCV cache hit
            {},  # Empty perp mapping - no perpetual for this symbol (funding)
            {},  # Empty perp mapping - no perpetual for this symbol (OI)
        ]

        service = MarketDataService()

        result = await service.get_market_context(symbol="UNKNOWN/USDT", timeframe="1h", limit=100)

        # OHLCV should be present
        assert isinstance(result["ohlcv_df"], pd.DataFrame)

        # Derivatives data should be None
        assert result["funding_rate"] is None
        assert result["funding_rate_history"] is None
        assert result["open_interest"] is None
        assert result["open_interest_change_pct"] is None

    @patch("data.market_data.ExchangeClient")
    @patch("data.market_data.AsyncDiskCache")
    async def test_get_market_context_with_point_in_time(self, mock_cache, mock_exchange_class):
        """Point-in-time filtering works across all data types."""
        mock_exchange = AsyncMock()
        mock_exchange_class.return_value = mock_exchange

        mock_exchange.exchange.has = {
            "fetchOHLCV": True,
            "fetchFundingRateHistory": True,
            "fetchOpenInterestHistory": True,
        }

        # Mock OHLCV fetch (for as_of case)
        # Important: First bar should close BEFORE as_of, second bar AFTER
        # For 1h timeframe, bar at 1609459200000 closes at 1609459200000 + 3600000 = 1609462800000
        mock_exchange.fetch_ohlcv.return_value = [
            [
                1609452000000,
                28800,
                29100,
                28700,
                29000,
                900,
            ],  # Closes at 1609455600000 (before as_of)
            [
                1609455600000,
                29000,
                29500,
                28800,
                29200,
                1000,
            ],  # Closes at 1609459200000 (exactly at as_of)
            [
                1609459200000,
                29200,
                29600,
                29100,
                29400,
                1100,
            ],  # Closes at 1609462800000 (after as_of)
        ]

        # Mock funding rate cache with multiple timestamps
        funding_cache = pd.DataFrame(
            {
                "timestamp": pd.to_datetime([1609455600000, 1609488000000], unit="ms", utc=True),
                "funding_rate": [0.0001, 0.00015],
            }
        )

        # Mock OI cache with multiple timestamps
        oi_cache = pd.DataFrame(
            {
                "timestamp": pd.to_datetime([1609455600000, 1609488000000], unit="ms", utc=True),
                "open_interest_value": [1000000000, 1050000000],
            }
        )

        mock_cache_instance = AsyncMock()
        mock_cache.return_value = mock_cache_instance
        mock_cache_instance.get.side_effect = [
            None,  # OHLCV cache miss
            {"BTC/USDT": "BTC/USDT:USDT"},  # perp mapping for funding
            funding_cache,  # funding rates cache hit
            {"BTC/USDT": "BTC/USDT:USDT"},  # perp mapping for OI
            oi_cache,  # OI cache hit
        ]

        service = MarketDataService()

        # Filter to as_of = 1609459200000 (bars closing at or before this time)
        as_of = pd.to_datetime(1609459200000, unit="ms", utc=True)
        result = await service.get_market_context(
            symbol="BTC/USDT", timeframe="1h", as_of=as_of, limit=100
        )

        # Should have 2 OHLCV bars (closing at 1609455600000 and 1609459200000)
        assert len(result["ohlcv_df"]) == 2
        # Should have 1 funding rate (at 1609455600000, second one is after as_of)
        assert len(result["funding_rate_history"]) == 1
        assert result["funding_rate"] == 0.0001

    @patch("data.market_data.ExchangeClient")
    @patch("data.market_data.AsyncDiskCache")
    async def test_get_market_context_partial_derivatives_data(
        self, mock_cache, mock_exchange_class
    ):
        """Handles case where only funding rates available, not open interest."""
        mock_exchange = AsyncMock()
        mock_exchange_class.return_value = mock_exchange

        # Exchange supports funding but not OI
        mock_exchange.exchange.has = {
            "fetchOHLCV": True,
            "fetchFundingRateHistory": True,
            "fetchOpenInterestHistory": False,  # Not supported
        }

        # Mock OHLCV cache
        ohlcv_cache = [
            {
                "timestamp": 1609459200000,
                "open": 29000,
                "high": 29500,
                "low": 28800,
                "close": 29200,
                "volume": 1000,
            }
        ]

        mock_cache_instance = AsyncMock()
        mock_cache.return_value = mock_cache_instance
        mock_cache_instance.get.side_effect = [
            ohlcv_cache,  # OHLCV cache hit
            {"BTC/USDT": "BTC/USDT:USDT"},  # perp mapping for funding
            None,  # funding cache miss
        ]

        mock_exchange.fetch_funding_rate_history.return_value = [
            {"timestamp": 1609459200000, "fundingRate": 0.0001}
        ]

        service = MarketDataService()

        result = await service.get_market_context(symbol="BTC/USDT", timeframe="1h", limit=100)

        # OHLCV and funding should be present
        assert isinstance(result["ohlcv_df"], pd.DataFrame)
        assert result["funding_rate"] == 0.0001

        # Open interest should be None (not supported)
        assert result["open_interest"] is None
        assert result["open_interest_change_pct"] is None

    @patch("data.market_data.ExchangeClient")
    @patch("data.market_data.AsyncDiskCache")
    async def test_get_market_context_calculates_oi_change(self, mock_cache, mock_exchange_class):
        """Calculates 24-hour open interest change correctly for different timeframes."""
        mock_exchange = AsyncMock()
        mock_exchange_class.return_value = mock_exchange

        mock_exchange.exchange.has = {
            "fetchOHLCV": True,
            "fetchFundingRateHistory": True,
            "fetchOpenInterestHistory": True,
        }

        # Test multiple timeframes
        test_cases = [
            # (timeframe, bar_duration_ms, num_bars, bars_for_24h)
            ("1h", 3600000, 30, 24),  # 1h bars: need 24 bars for 24h
            ("4h", 14400000, 10, 6),  # 4h bars: need 6 bars for 24h
            ("1m", 60000, 1500, 1440),  # 1m bars: need 1440 bars for 24h
        ]

        for timeframe, bar_duration_ms, num_bars, bars_for_24h in test_cases:
            # Mock OHLCV cache
            ohlcv_cache = [
                {
                    "timestamp": 1609459200000,
                    "open": 29000,
                    "high": 29500,
                    "low": 28800,
                    "close": 29200,
                    "volume": 1000,
                }
            ]

            # Create OI data with enough bars for this timeframe
            # Index -1 (last) is most recent, index -bars_for_24h is 24 hours ago
            oi_timestamps = [1609459200000 + (i * bar_duration_ms) for i in range(num_bars)]
            oi_values = [1000000000 + (i * 1000000) for i in range(num_bars)]  # Increasing OI
            oi_cache = pd.DataFrame(
                {
                    "timestamp": pd.to_datetime(oi_timestamps, unit="ms", utc=True),
                    "open_interest_value": oi_values,
                }
            )

            mock_cache_instance = AsyncMock()
            mock_cache.return_value = mock_cache_instance
            mock_cache_instance.get.side_effect = [
                ohlcv_cache,  # OHLCV cache hit
                {"BTC/USDT": "BTC/USDT:USDT"},  # perp mapping for funding
                None,  # funding cache miss (skip funding for this test)
                {"BTC/USDT": "BTC/USDT:USDT"},  # perp mapping for OI
                oi_cache,  # OI cache hit
            ]

            service = MarketDataService()

            result = await service.get_market_context(
                symbol="BTC/USDT", timeframe=timeframe, limit=100
            )

            # Verify OI values
            # Latest (index -1): 1000000000 + (num_bars-1)*1000000
            # 24h ago: index from end is -bars_for_24h, which is index (num_bars - bars_for_24h)
            latest_oi = 1000000000 + (num_bars - 1) * 1000000
            oi_24h_ago = 1000000000 + (num_bars - bars_for_24h) * 1000000

            assert result["open_interest"] == latest_oi, f"Failed for {timeframe}"
            assert result["open_interest_change_pct"] is not None, f"Failed for {timeframe}"

            # Calculate expected change
            expected_change = ((latest_oi - oi_24h_ago) / oi_24h_ago) * 100
            assert abs(result["open_interest_change_pct"] - expected_change) < 0.01, (
                f"Failed for {timeframe}: expected {expected_change}, got {result['open_interest_change_pct']}"
            )
