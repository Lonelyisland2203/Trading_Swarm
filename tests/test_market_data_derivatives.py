"""Tests for derivatives data fetching (funding rates and open interest)."""
import asyncio
import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from data.market_data import MarketDataService, ExchangeClient

class TestAdaptiveTTL:
    """Test adaptive cache TTL calculation."""

    @patch('data.market_data.ExchangeClient')
    @patch('data.market_data.AsyncDiskCache')
    def test_historical_data_gets_long_ttl(self, mock_cache, mock_exchange):
        """Data older than 7 days gets 24-hour TTL."""
        service = MarketDataService()
        now = datetime.now(timezone.utc)
        old_timestamp = now - timedelta(days=10)

        ttl = service._compute_adaptive_ttl(old_timestamp, now)

        assert ttl == 86400  # 24 hours in seconds

    @patch('data.market_data.ExchangeClient')
    @patch('data.market_data.AsyncDiskCache')
    def test_recent_data_gets_medium_ttl(self, mock_cache, mock_exchange):
        """Data between 1 hour and 7 days old gets 2-hour TTL."""
        service = MarketDataService()
        now = datetime.now(timezone.utc)
        recent_timestamp = now - timedelta(hours=12)

        ttl = service._compute_adaptive_ttl(recent_timestamp, now)

        assert ttl == 7200  # 2 hours in seconds

    @patch('data.market_data.ExchangeClient')
    @patch('data.market_data.AsyncDiskCache')
    def test_live_data_gets_short_ttl(self, mock_cache, mock_exchange):
        """Data less than 1 hour old gets 30-minute TTL."""
        service = MarketDataService()
        now = datetime.now(timezone.utc)
        live_timestamp = now - timedelta(minutes=15)

        ttl = service._compute_adaptive_ttl(live_timestamp, now)

        assert ttl == 1800  # 30 minutes in seconds


class TestPerpetualMapping:
    """Test spot to perpetual symbol mapping."""

    @patch('data.market_data.ExchangeClient')
    @patch('data.market_data.AsyncDiskCache')
    async def test_load_perpetual_markets_caches_mapping(
        self, mock_cache, mock_exchange
    ):
        """First call to load markets gets from API, subsequent calls use cache."""
        # Mock load_markets response (synchronous method)
        mock_exchange_instance = MagicMock()
        mock_exchange.return_value = mock_exchange_instance
        mock_exchange_instance.load_markets.return_value = {
            'BTC/USDT': {'type': 'spot', 'symbol': 'BTC/USDT'},
            'BTC/USDT:USDT': {'type': 'swap', 'symbol': 'BTC/USDT:USDT', 'settle': 'USDT'},
            'ETH/USDT': {'type': 'spot', 'symbol': 'ETH/USDT'},
            'ETH/USDT:USDT': {'type': 'swap', 'symbol': 'ETH/USDT:USDT', 'settle': 'USDT'},
        }

        # Mock cache get/set (async methods)
        mock_cache_instance = MagicMock()
        mock_cache.return_value = mock_cache_instance
        mock_cache_instance.get = AsyncMock(return_value=None)  # Not cached yet
        mock_cache_instance.set = AsyncMock()

        service = MarketDataService()

        # First call - should fetch from API
        mapping = await service._load_perpetual_markets()

        assert mapping == {
            'BTC/USDT': 'BTC/USDT:USDT',
            'ETH/USDT': 'ETH/USDT:USDT'
        }
        mock_exchange_instance.load_markets.assert_called_once()

        # Should have cached the result
        mock_cache_instance.set.assert_called_once()

    @patch('data.market_data.ExchangeClient')
    @patch('data.market_data.AsyncDiskCache')
    async def test_get_perpetual_symbol_returns_perp_for_spot(
        self, mock_cache, mock_exchange
    ):
        """Spot symbols are mapped to their perpetual equivalents."""
        mock_exchange_instance = MagicMock()
        mock_exchange.return_value = mock_exchange_instance

        mock_cache_instance = MagicMock()
        mock_cache.return_value = mock_cache_instance
        mock_cache_instance.get = AsyncMock(return_value={
            'BTC/USDT': 'BTC/USDT:USDT',
            'ETH/USDT': 'ETH/USDT:USDT'
        })

        service = MarketDataService()

        perp_symbol = await service._get_perpetual_symbol('BTC/USDT')

        assert perp_symbol == 'BTC/USDT:USDT'

    @patch('data.market_data.ExchangeClient')
    @patch('data.market_data.AsyncDiskCache')
    async def test_get_perpetual_symbol_returns_none_if_no_mapping(
        self, mock_cache, mock_exchange
    ):
        """Returns None if perpetual not found for spot symbol."""
        mock_exchange_instance = MagicMock()
        mock_exchange.return_value = mock_exchange_instance

        mock_cache_instance = MagicMock()
        mock_cache.return_value = mock_cache_instance
        mock_cache_instance.get = AsyncMock(return_value={
            'BTC/USDT': 'BTC/USDT:USDT'
        })

        service = MarketDataService()

        perp_symbol = await service._get_perpetual_symbol('UNKNOWN/USDT')

        assert perp_symbol is None


class TestExchangeClientDerivatives:
    """Test ExchangeClient methods for derivatives data."""

    @patch('data.market_data.ccxt.binance')
    async def test_fetch_funding_rate_history_success(self, mock_binance_class):
        """Successful funding rate history fetch."""
        mock_exchange = MagicMock()
        mock_binance_class.return_value = mock_exchange

        # Mock CCXT response
        mock_exchange.fetch_funding_rate_history.return_value = [
            {'timestamp': 1609459200000, 'fundingRate': 0.0001},
            {'timestamp': 1609488000000, 'fundingRate': 0.00015},
        ]

        client = ExchangeClient(exchange_id='binance')

        result = await client.fetch_funding_rate_history(
            'BTC/USDT:USDT',
            since=1609459200000,
            limit=100
        )

        assert len(result) == 2
        assert result[0]['fundingRate'] == 0.0001
        mock_exchange.fetch_funding_rate_history.assert_called_once_with(
            'BTC/USDT:USDT',
            since=1609459200000,
            limit=100
        )

    @patch('data.market_data.ccxt.binance')
    async def test_fetch_open_interest_history_success(self, mock_binance_class):
        """Successful open interest history fetch."""
        mock_exchange = MagicMock()
        mock_binance_class.return_value = mock_exchange

        # Mock CCXT response
        mock_exchange.fetch_open_interest_history.return_value = [
            {'timestamp': 1609459200000, 'openInterestValue': 1000000000},
            {'timestamp': 1609488000000, 'openInterestValue': 1050000000},
        ]

        client = ExchangeClient(exchange_id='binance')

        result = await client.fetch_open_interest_history(
            'BTC/USDT:USDT',
            timeframe='1h',
            since=1609459200000,
            limit=100
        )

        assert len(result) == 2
        assert result[0]['openInterestValue'] == 1000000000
        mock_exchange.fetch_open_interest_history.assert_called_once_with(
            'BTC/USDT:USDT',
            timeframe='1h',
            since=1609459200000,
            limit=100
        )


class TestFundingRateFetching:
    """Test funding rate fetching with capability checking."""

    @patch('data.market_data.ExchangeClient')
    @patch('data.market_data.AsyncDiskCache')
    async def test_fetch_funding_rates_checks_capability(
        self, mock_cache, mock_exchange_class
    ):
        """Method checks exchange capability before fetching."""
        mock_exchange = AsyncMock()
        mock_exchange_class.return_value = mock_exchange
        mock_exchange.exchange.has = {'fetchFundingRateHistory': False}

        mock_cache_instance = AsyncMock()
        mock_cache.return_value = mock_cache_instance
        mock_cache_instance.get.return_value = {'BTC/USDT': 'BTC/USDT:USDT'}

        service = MarketDataService()

        result = await service.fetch_funding_rates('BTC/USDT')

        # Should return None when capability not supported
        assert result is None
        # Should NOT call fetch_funding_rate_history
        mock_exchange.fetch_funding_rate_history.assert_not_called()

    @patch('data.market_data.ExchangeClient')
    @patch('data.market_data.AsyncDiskCache')
    async def test_fetch_funding_rates_success(
        self, mock_cache, mock_exchange_class
    ):
        """Successful funding rate fetch returns DataFrame."""
        mock_exchange = AsyncMock()
        mock_exchange_class.return_value = mock_exchange
        mock_exchange.exchange.has = {'fetchFundingRateHistory': True}
        mock_exchange.fetch_funding_rate_history.return_value = [
            {'timestamp': 1609459200000, 'fundingRate': 0.0001},
            {'timestamp': 1609488000000, 'fundingRate': 0.00015},
        ]

        mock_cache_instance = AsyncMock()
        mock_cache.return_value = mock_cache_instance
        mock_cache_instance.get.side_effect = [
            {'BTC/USDT': 'BTC/USDT:USDT'},  # perpetual mapping
            None  # funding rates cache miss
        ]

        service = MarketDataService()

        result = await service.fetch_funding_rates('BTC/USDT')

        assert result is not None
        assert len(result) == 2
        assert 'timestamp' in result.columns
        assert 'funding_rate' in result.columns
        assert result['funding_rate'].iloc[0] == 0.0001

    @patch('data.market_data.ExchangeClient')
    @patch('data.market_data.AsyncDiskCache')
    async def test_fetch_funding_rates_applies_point_in_time_filter(
        self, mock_cache, mock_exchange_class
    ):
        """Point-in-time filter excludes future funding rates."""
        mock_exchange = AsyncMock()
        mock_exchange_class.return_value = mock_exchange
        mock_exchange.exchange.has = {'fetchFundingRateHistory': True}

        # Create mock DataFrame
        import pandas as pd
        from datetime import timezone
        mock_df = pd.DataFrame({
            'timestamp': pd.to_datetime([1609459200000, 1609488000000], unit='ms', utc=True),
            'funding_rate': [0.0001, 0.00015]
        })

        mock_cache_instance = AsyncMock()
        mock_cache.return_value = mock_cache_instance
        mock_cache_instance.get.side_effect = [
            {'BTC/USDT': 'BTC/USDT:USDT'},  # perpetual mapping
            mock_df  # funding rates cache hit
        ]

        service = MarketDataService()

        # Filter to only include first timestamp
        as_of = pd.to_datetime(1609459200000, unit='ms', utc=True)
        result = await service.fetch_funding_rates('BTC/USDT', as_of=as_of)

        assert len(result) == 1
        assert result['funding_rate'].iloc[0] == 0.0001

    @patch('data.market_data.ExchangeClient')
    @patch('data.market_data.AsyncDiskCache')
    async def test_fetch_funding_rates_returns_none_for_missing_perpetual(
        self, mock_cache, mock_exchange_class
    ):
        """Returns None if no perpetual symbol found."""
        mock_exchange = AsyncMock()
        mock_exchange_class.return_value = mock_exchange
        mock_exchange.exchange.has = {'fetchFundingRateHistory': True}

        mock_cache_instance = AsyncMock()
        mock_cache.return_value = mock_cache_instance
        mock_cache_instance.get.return_value = {}  # Empty mapping - no perpetual

        service = MarketDataService()

        result = await service.fetch_funding_rates('UNKNOWN/USDT')

        assert result is None

    @patch('data.market_data.ExchangeClient')
    @patch('data.market_data.AsyncDiskCache')
    async def test_fetch_funding_rates_uses_cache(
        self, mock_cache, mock_exchange_class
    ):
        """Cached funding rates are returned without exchange call."""
        mock_exchange = AsyncMock()
        mock_exchange_class.return_value = mock_exchange
        mock_exchange.exchange.has = {'fetchFundingRateHistory': True}

        # Create mock DataFrame
        import pandas as pd
        mock_df = pd.DataFrame({
            'timestamp': pd.to_datetime([1609459200000], unit='ms', utc=True),
            'funding_rate': [0.0001]
        })

        mock_cache_instance = AsyncMock()
        mock_cache.return_value = mock_cache_instance
        mock_cache_instance.get.side_effect = [
            {'BTC/USDT': 'BTC/USDT:USDT'},  # perpetual mapping
            mock_df  # funding rates cache hit
        ]

        service = MarketDataService()

        result = await service.fetch_funding_rates('BTC/USDT')

        assert result is not None
        assert len(result) == 1
        # Should NOT call exchange since we got cache hit
        mock_exchange.fetch_funding_rate_history.assert_not_called()


class TestOpenInterestFetching:
    """Test open interest fetching with capability checking."""

    @patch('data.market_data.ExchangeClient')
    @patch('data.market_data.AsyncDiskCache')
    async def test_fetch_open_interest_checks_capability(
        self, mock_cache, mock_exchange_class
    ):
        """Method checks exchange capability before fetching."""
        mock_exchange = AsyncMock()
        mock_exchange_class.return_value = mock_exchange
        mock_exchange.exchange.has = {'fetchOpenInterestHistory': False}

        mock_cache_instance = AsyncMock()
        mock_cache.return_value = mock_cache_instance
        mock_cache_instance.get.return_value = {'BTC/USDT': 'BTC/USDT:USDT'}

        service = MarketDataService()

        result = await service.fetch_open_interest('BTC/USDT', timeframe='1h')

        # Should return None when capability not supported
        assert result is None
        mock_exchange.fetch_open_interest_history.assert_not_called()

    @patch('data.market_data.ExchangeClient')
    @patch('data.market_data.AsyncDiskCache')
    async def test_fetch_open_interest_success(
        self, mock_cache, mock_exchange_class
    ):
        """Successful open interest fetch returns DataFrame."""
        mock_exchange = AsyncMock()
        mock_exchange_class.return_value = mock_exchange
        mock_exchange.exchange.has = {'fetchOpenInterestHistory': True}
        mock_exchange.fetch_open_interest_history.return_value = [
            {'timestamp': 1609459200000, 'openInterestValue': 1000000000, 'openInterest': 50000},
            {'timestamp': 1609488000000, 'openInterestValue': 1050000000, 'openInterest': 51000},
        ]

        mock_cache_instance = AsyncMock()
        mock_cache.return_value = mock_cache_instance
        mock_cache_instance.get.side_effect = [
            {'BTC/USDT': 'BTC/USDT:USDT'},  # perpetual mapping
            None  # open interest cache miss
        ]

        service = MarketDataService()

        result = await service.fetch_open_interest('BTC/USDT', timeframe='1h')

        assert result is not None
        assert len(result) == 2
        assert 'timestamp' in result.columns
        assert 'open_interest_value' in result.columns
        assert 'open_interest' in result.columns
        assert result['open_interest_value'].iloc[0] == 1000000000
        assert result['open_interest'].iloc[0] == 50000

    @patch('data.market_data.ExchangeClient')
    @patch('data.market_data.AsyncDiskCache')
    async def test_fetch_open_interest_applies_point_in_time_filter(
        self, mock_cache, mock_exchange_class
    ):
        """Point-in-time filter excludes future open interest data."""
        mock_exchange = AsyncMock()
        mock_exchange_class.return_value = mock_exchange
        mock_exchange.exchange.has = {'fetchOpenInterestHistory': True}

        # Create mock DataFrame
        import pandas as pd
        from datetime import timezone
        mock_df = pd.DataFrame({
            'timestamp': pd.to_datetime([1609459200000, 1609488000000], unit='ms', utc=True),
            'open_interest_value': [1000000000, 1050000000],
            'open_interest': [50000, 51000]
        })

        mock_cache_instance = AsyncMock()
        mock_cache.return_value = mock_cache_instance
        mock_cache_instance.get.side_effect = [
            {'BTC/USDT': 'BTC/USDT:USDT'},  # perpetual mapping
            mock_df  # open interest cache hit
        ]

        service = MarketDataService()

        # Filter to only include first timestamp
        as_of = pd.to_datetime(1609459200000, unit='ms', utc=True)
        result = await service.fetch_open_interest('BTC/USDT', timeframe='1h', as_of=as_of)

        assert len(result) == 1
        assert result['open_interest_value'].iloc[0] == 1000000000

    @patch('data.market_data.ExchangeClient')
    @patch('data.market_data.AsyncDiskCache')
    async def test_fetch_open_interest_returns_none_for_missing_perpetual(
        self, mock_cache, mock_exchange_class
    ):
        """Returns None if no perpetual symbol found."""
        mock_exchange = AsyncMock()
        mock_exchange_class.return_value = mock_exchange
        mock_exchange.exchange.has = {'fetchOpenInterestHistory': True}

        mock_cache_instance = AsyncMock()
        mock_cache.return_value = mock_cache_instance
        mock_cache_instance.get.return_value = {}  # Empty mapping - no perpetual

        service = MarketDataService()

        result = await service.fetch_open_interest('UNKNOWN/USDT', timeframe='1h')

        assert result is None

    @patch('data.market_data.ExchangeClient')
    @patch('data.market_data.AsyncDiskCache')
    async def test_fetch_open_interest_uses_cache(
        self, mock_cache, mock_exchange_class
    ):
        """Cached open interest data is returned without exchange call."""
        mock_exchange = AsyncMock()
        mock_exchange_class.return_value = mock_exchange
        mock_exchange.exchange.has = {'fetchOpenInterestHistory': True}

        # Create mock DataFrame
        import pandas as pd
        mock_df = pd.DataFrame({
            'timestamp': pd.to_datetime([1609459200000], unit='ms', utc=True),
            'open_interest_value': [1000000000],
            'open_interest': [50000]
        })

        mock_cache_instance = AsyncMock()
        mock_cache.return_value = mock_cache_instance
        mock_cache_instance.get.side_effect = [
            {'BTC/USDT': 'BTC/USDT:USDT'},  # perpetual mapping
            mock_df  # open interest cache hit
        ]

        service = MarketDataService()

        result = await service.fetch_open_interest('BTC/USDT', timeframe='1h')

        assert result is not None
        assert len(result) == 1
        # Should NOT call exchange since we got cache hit
        mock_exchange.fetch_open_interest_history.assert_not_called()
