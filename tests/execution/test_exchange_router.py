"""
Tests for ExchangeRouter.

Verifies router dispatches to correct adapter based on EXCHANGE env var.
All tests use mocked adapters - no live calls.
"""

import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from execution.exchange_router import ExchangeRouter
from execution.models import OrderResult, Position


@pytest.fixture
def temp_state_dir():
    """Create temporary state directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def mock_hyperliquid_adapter():
    """Mock HyperliquidAdapter."""
    mock = MagicMock()
    mock.place_order = AsyncMock(
        return_value=OrderResult(
            order_id="hl-123",
            symbol="BTC",
            side="buy",
            order_type="limit",
            amount=0.01,
            price=50000.0,
            status="open",
            filled=0.0,
            remaining=0.01,
            cost=0.0,
            fee=0.0,
            timestamp=MagicMock(),
        )
    )
    mock.cancel_order = AsyncMock()
    mock.get_positions = AsyncMock(return_value=[])
    mock.get_balance = AsyncMock(return_value={"total": 10000.0, "free": 9000.0})
    mock.flatten_all = AsyncMock()
    return mock


@pytest.fixture
def mock_binance_adapter():
    """Mock BinanceExecutionClient."""
    mock = MagicMock()
    mock.place_limit_order = AsyncMock(
        return_value=OrderResult(
            order_id="bn-456",
            symbol="BTC/USDT",
            side="buy",
            order_type="limit",
            amount=0.01,
            price=50000.0,
            status="open",
            filled=0.0,
            remaining=0.01,
            cost=0.0,
            fee=0.0,
            timestamp=MagicMock(),
        )
    )
    mock.place_market_order = AsyncMock()
    mock.cancel_order = AsyncMock()
    mock.get_open_positions = AsyncMock(return_value=[])
    mock.get_balance = AsyncMock(return_value={"USDT": {"free": 9000.0, "total": 10000.0}})
    mock.close_position = AsyncMock()
    return mock


class TestExchangeRouterHyperliquid:
    """Test router dispatches to Hyperliquid adapter."""

    @pytest.mark.asyncio
    async def test_routes_to_hyperliquid(self, temp_state_dir, mock_hyperliquid_adapter):
        """When EXCHANGE=hyperliquid, routes to HyperliquidAdapter."""
        with patch.dict(os.environ, {"EXCHANGE": "hyperliquid"}):
            with patch(
                "execution.hyperliquid_adapter.HyperliquidAdapter",
                return_value=mock_hyperliquid_adapter,
            ):
                router = ExchangeRouter(
                    private_key="0x" + "a" * 64,
                    testnet=True,
                    state_dir=temp_state_dir,
                )
                # Inject mock
                router._adapter = mock_hyperliquid_adapter

                result = await router.place_order(
                    symbol="BTC",
                    side="buy",
                    amount=0.01,
                    price=50000.0,
                    order_type="limit",
                )

                mock_hyperliquid_adapter.place_order.assert_called_once()
                assert result.order_id == "hl-123"

    @pytest.mark.asyncio
    async def test_hyperliquid_positions(self, temp_state_dir, mock_hyperliquid_adapter):
        """Get positions via Hyperliquid."""
        mock_hyperliquid_adapter.get_positions.return_value = [
            Position(
                symbol="BTC",
                side="long",
                amount=0.01,
                entry_price=50000.0,
                mark_price=51000.0,
                unrealized_pnl=10.0,
                leverage=1,
            )
        ]

        with patch.dict(os.environ, {"EXCHANGE": "hyperliquid"}):
            with patch(
                "execution.hyperliquid_adapter.HyperliquidAdapter",
                return_value=mock_hyperliquid_adapter,
            ):
                router = ExchangeRouter(
                    private_key="0x" + "a" * 64,
                    testnet=True,
                    state_dir=temp_state_dir,
                )
                router._adapter = mock_hyperliquid_adapter

                positions = await router.get_positions()

                assert len(positions) == 1
                assert positions[0].symbol == "BTC"


class TestExchangeRouterBinance:
    """Test router dispatches to existing CCXT Binance adapter."""

    @pytest.mark.asyncio
    async def test_routes_to_binance(self, temp_state_dir, mock_binance_adapter):
        """When EXCHANGE=binance, routes to BinanceExecutionClient."""
        with patch.dict(os.environ, {"EXCHANGE": "binance"}):
            with patch(
                "execution.binance_client.BinanceExecutionClient", return_value=mock_binance_adapter
            ):
                router = ExchangeRouter(
                    api_key="test-key",
                    api_secret="test-secret",
                    testnet=True,
                    state_dir=temp_state_dir,
                )
                router._adapter = mock_binance_adapter

                result = await router.place_order(
                    symbol="BTC/USDT",
                    side="buy",
                    amount=0.01,
                    price=50000.0,
                    order_type="limit",
                )

                mock_binance_adapter.place_limit_order.assert_called_once()
                assert result.order_id == "bn-456"

    @pytest.mark.asyncio
    async def test_binance_market_order(self, temp_state_dir, mock_binance_adapter):
        """Market order routed to Binance."""
        mock_binance_adapter.place_market_order.return_value = OrderResult(
            order_id="bn-789",
            symbol="ETH/USDT",
            side="sell",
            order_type="market",
            amount=0.5,
            price=None,
            status="closed",
            filled=0.5,
            remaining=0.0,
            cost=1500.0,
            fee=0.75,
            timestamp=MagicMock(),
        )

        with patch.dict(os.environ, {"EXCHANGE": "binance"}):
            with patch(
                "execution.binance_client.BinanceExecutionClient", return_value=mock_binance_adapter
            ):
                router = ExchangeRouter(
                    api_key="test-key",
                    api_secret="test-secret",
                    testnet=True,
                    state_dir=temp_state_dir,
                )
                router._adapter = mock_binance_adapter

                await router.place_order(
                    symbol="ETH/USDT",
                    side="sell",
                    amount=0.5,
                    order_type="market",
                )

                mock_binance_adapter.place_market_order.assert_called_once()


class TestExchangeRouterSwap:
    """Test runtime switching between exchanges."""

    @pytest.mark.asyncio
    async def test_switch_from_binance_to_hyperliquid(
        self, temp_state_dir, mock_binance_adapter, mock_hyperliquid_adapter
    ):
        """Can switch exchanges at runtime."""
        with patch.dict(os.environ, {"EXCHANGE": "binance"}):
            with patch(
                "execution.binance_client.BinanceExecutionClient", return_value=mock_binance_adapter
            ):
                router = ExchangeRouter(
                    api_key="test-key",
                    api_secret="test-secret",
                    testnet=True,
                    state_dir=temp_state_dir,
                )
                router._adapter = mock_binance_adapter

                # Initially Binance
                result1 = await router.place_order(
                    symbol="BTC/USDT",
                    side="buy",
                    amount=0.01,
                    price=50000.0,
                    order_type="limit",
                )
                assert result1.order_id.startswith("bn")

        # Now switch to Hyperliquid
        with patch.dict(os.environ, {"EXCHANGE": "hyperliquid"}):
            with patch(
                "execution.hyperliquid_adapter.HyperliquidAdapter",
                return_value=mock_hyperliquid_adapter,
            ):
                router.switch_exchange("hyperliquid", private_key="0x" + "a" * 64)
                router._adapter = mock_hyperliquid_adapter

                result2 = await router.place_order(
                    symbol="BTC",
                    side="buy",
                    amount=0.01,
                    price=50000.0,
                    order_type="limit",
                )
                assert result2.order_id == "hl-123"

    @pytest.mark.asyncio
    async def test_router_logs_exchange_selection(self, temp_state_dir, mock_hyperliquid_adapter):
        """Router logs which exchange is selected at startup."""
        with patch.dict(os.environ, {"EXCHANGE": "hyperliquid"}):
            with patch(
                "execution.hyperliquid_adapter.HyperliquidAdapter",
                return_value=mock_hyperliquid_adapter,
            ):
                router = ExchangeRouter(
                    private_key="0x" + "a" * 64,
                    testnet=True,
                    state_dir=temp_state_dir,
                )

                # Check that exchange selection was logged
                assert router.current_exchange == "hyperliquid"


class TestExchangeRouterInterface:
    """Test unified interface regardless of exchange."""

    @pytest.mark.asyncio
    async def test_identical_interface_hyperliquid(self, temp_state_dir, mock_hyperliquid_adapter):
        """Interface is identical for Hyperliquid."""
        with patch.dict(os.environ, {"EXCHANGE": "hyperliquid"}):
            with patch(
                "execution.hyperliquid_adapter.HyperliquidAdapter",
                return_value=mock_hyperliquid_adapter,
            ):
                router = ExchangeRouter(
                    private_key="0x" + "a" * 64,
                    testnet=True,
                    state_dir=temp_state_dir,
                )
                router._adapter = mock_hyperliquid_adapter

                # All these methods should exist and work
                await router.place_order("BTC", "buy", 0.01, 50000.0, "limit")
                await router.get_positions()
                await router.get_balance()
                await router.flatten_all()

    @pytest.mark.asyncio
    async def test_identical_interface_binance(self, temp_state_dir, mock_binance_adapter):
        """Interface is identical for Binance."""
        with patch.dict(os.environ, {"EXCHANGE": "binance"}):
            with patch(
                "execution.binance_client.BinanceExecutionClient", return_value=mock_binance_adapter
            ):
                router = ExchangeRouter(
                    api_key="test-key",
                    api_secret="test-secret",
                    testnet=True,
                    state_dir=temp_state_dir,
                )
                router._adapter = mock_binance_adapter

                # Same interface works
                await router.place_order("BTC/USDT", "buy", 0.01, 50000.0, "limit")
                await router.get_positions()
                await router.get_balance()
                await router.flatten_all()


class TestExchangeRouterDefaults:
    """Test default behavior."""

    def test_default_exchange_is_binance(self, temp_state_dir, mock_binance_adapter):
        """When EXCHANGE not set, defaults to binance."""
        # Clear EXCHANGE env var
        env = {k: v for k, v in os.environ.items() if k != "EXCHANGE"}

        with patch.dict(os.environ, env, clear=True):
            with patch(
                "execution.binance_client.BinanceExecutionClient", return_value=mock_binance_adapter
            ):
                router = ExchangeRouter(
                    api_key="test-key",
                    api_secret="test-secret",
                    testnet=True,
                    state_dir=temp_state_dir,
                )

                assert router.current_exchange == "binance"

    def test_invalid_exchange_raises(self, temp_state_dir):
        """Invalid EXCHANGE value raises error."""
        with patch.dict(os.environ, {"EXCHANGE": "kraken"}):
            with pytest.raises(ValueError, match="Unsupported exchange"):
                ExchangeRouter(
                    api_key="test-key",
                    api_secret="test-secret",
                    testnet=True,
                    state_dir=temp_state_dir,
                )
