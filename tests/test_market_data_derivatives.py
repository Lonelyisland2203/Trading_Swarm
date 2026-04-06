"""Tests for derivatives data fetching (funding rates and open interest)."""
import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from data.market_data import MarketDataService

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
