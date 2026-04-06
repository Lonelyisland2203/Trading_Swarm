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

        result = await asyncio.to_thread(
            client.fetch_funding_rate_history,
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

        result = await asyncio.to_thread(
            client.fetch_open_interest_history,
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
