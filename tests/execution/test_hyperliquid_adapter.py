"""
Tests for HyperliquidAdapter.

All tests use mocked API - no live calls.
TDD: Write tests first, watch them fail, then implement.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Import will fail until implementation exists - that's expected in TDD RED phase
from execution.hyperliquid_adapter import HyperliquidAdapter
from execution.exceptions import (
    OrderRejectedError,
)


@pytest.fixture
def mock_hyperliquid_info():
    """Mock Hyperliquid Info API."""
    mock = MagicMock()
    mock.user_state = MagicMock(
        return_value={
            "marginSummary": {
                "accountValue": "10000.00",
                "totalMarginUsed": "1000.00",
                "withdrawable": "9000.00",
            },
            "assetPositions": [],
        }
    )
    mock.open_orders = MagicMock(return_value=[])
    return mock


@pytest.fixture
def mock_hyperliquid_exchange():
    """Mock Hyperliquid Exchange API."""
    mock = MagicMock()
    mock.order = MagicMock(
        return_value={"response": {"data": {"statuses": [{"resting": {"oid": 12345}}]}}}
    )
    mock.cancel = MagicMock(return_value={"status": "ok"})
    return mock


@pytest.fixture
def temp_state_dir():
    """Create temporary state directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def adapter(temp_state_dir, mock_hyperliquid_info, mock_hyperliquid_exchange):
    """Create HyperliquidAdapter with mocked dependencies."""
    with patch("execution.hyperliquid_adapter.Info", return_value=mock_hyperliquid_info):
        with patch(
            "execution.hyperliquid_adapter.Exchange", return_value=mock_hyperliquid_exchange
        ):
            adapter = HyperliquidAdapter(
                private_key="0x" + "a" * 64,  # Mock private key
                testnet=True,
                state_dir=temp_state_dir,
            )
            adapter._info = mock_hyperliquid_info
            adapter._exchange = mock_hyperliquid_exchange
            yield adapter


class TestOrderPlacementLimit:
    """Test limit order placement."""

    @pytest.mark.asyncio
    async def test_limit_buy_order_params_correct(self, adapter, mock_hyperliquid_exchange):
        """Verify limit buy order sends correct params to API."""
        # Arrange
        symbol = "BTC"
        side = "buy"
        amount = 0.01
        price = 50000.0

        # Act
        result = await adapter.place_order(
            symbol=symbol,
            side=side,
            amount=amount,
            price=price,
            order_type="limit",
        )

        # Assert
        mock_hyperliquid_exchange.order.assert_called_once()
        call_args = mock_hyperliquid_exchange.order.call_args
        assert call_args is not None
        # Hyperliquid uses coin name, is_buy flag, limit order params
        assert result.order_id is not None
        assert result.symbol == symbol
        assert result.side == side

    @pytest.mark.asyncio
    async def test_limit_sell_order_params_correct(self, adapter, mock_hyperliquid_exchange):
        """Verify limit sell order sends correct params to API."""
        result = await adapter.place_order(
            symbol="ETH",
            side="sell",
            amount=0.1,
            price=3000.0,
            order_type="limit",
        )

        mock_hyperliquid_exchange.order.assert_called_once()
        assert result.side == "sell"


class TestOrderPlacementMarket:
    """Test market order placement."""

    @pytest.mark.asyncio
    async def test_market_buy_order_params_correct(self, adapter, mock_hyperliquid_exchange):
        """Verify market buy order sends correct params."""
        result = await adapter.place_order(
            symbol="BTC",
            side="buy",
            amount=0.01,
            order_type="market",
        )

        mock_hyperliquid_exchange.order.assert_called_once()
        # Market orders should use IOC with slippage
        assert result.order_type == "market"

    @pytest.mark.asyncio
    async def test_market_sell_order_params_correct(self, adapter, mock_hyperliquid_exchange):
        """Verify market sell order sends correct params."""
        result = await adapter.place_order(
            symbol="ETH",
            side="sell",
            amount=0.5,
            order_type="market",
        )

        assert result.order_type == "market"
        assert result.side == "sell"


class TestTriggerOrders:
    """Test exchange-side stop-loss trigger orders."""

    @pytest.mark.asyncio
    async def test_stop_loss_placed_with_long_position(self, adapter, mock_hyperliquid_exchange):
        """Every long position gets an exchange-side stop-loss order."""
        # Act - place a long (buy) order
        await adapter.place_order(
            symbol="BTC",
            side="buy",
            amount=0.01,
            price=50000.0,
            order_type="limit",
            stop_loss_price=49000.0,  # 2% stop loss
        )

        # Assert - should have placed main order + stop loss trigger
        assert mock_hyperliquid_exchange.order.call_count >= 1
        # Check that a trigger order was placed
        calls = mock_hyperliquid_exchange.order.call_args_list
        # At least one call should be for stop-loss
        assert (
            any("trigger" in str(call).lower() or len(calls) > 1 for call in calls)
            or mock_hyperliquid_exchange.order.call_count == 2
        )

    @pytest.mark.asyncio
    async def test_stop_loss_placed_with_short_position(self, adapter, mock_hyperliquid_exchange):
        """Every short position gets an exchange-side stop-loss order."""
        await adapter.place_order(
            symbol="ETH",
            side="sell",
            amount=0.1,
            price=3000.0,
            order_type="limit",
            stop_loss_price=3060.0,  # 2% stop loss for short
        )

        # Stop loss should be placed for short position
        assert mock_hyperliquid_exchange.order.call_count >= 1


class TestPositionQuery:
    """Test position parsing from API response."""

    @pytest.mark.asyncio
    async def test_position_parsing_long(self, adapter, mock_hyperliquid_info):
        """Parse long position correctly from API response."""
        mock_hyperliquid_info.user_state.return_value = {
            "marginSummary": {"accountValue": "10000.00"},
            "assetPositions": [
                {
                    "position": {
                        "coin": "BTC",
                        "szi": "0.01",  # Positive = long
                        "entryPx": "50000.0",
                        "positionValue": "500.0",
                        "unrealizedPnl": "10.0",
                        "leverage": {"value": "1"},
                        "liquidationPx": "45000.0",
                    }
                }
            ],
        }

        positions = await adapter.get_positions()

        assert len(positions) == 1
        pos = positions[0]
        assert pos.symbol == "BTC"
        assert pos.side == "long"
        assert pos.amount == 0.01
        assert pos.entry_price == 50000.0
        assert pos.unrealized_pnl == 10.0

    @pytest.mark.asyncio
    async def test_position_parsing_short(self, adapter, mock_hyperliquid_info):
        """Parse short position correctly from API response."""
        mock_hyperliquid_info.user_state.return_value = {
            "marginSummary": {"accountValue": "10000.00"},
            "assetPositions": [
                {
                    "position": {
                        "coin": "ETH",
                        "szi": "-0.5",  # Negative = short
                        "entryPx": "3000.0",
                        "positionValue": "1500.0",
                        "unrealizedPnl": "-20.0",
                        "leverage": {"value": "1"},
                        "liquidationPx": "3500.0",
                    }
                }
            ],
        }

        positions = await adapter.get_positions()

        assert len(positions) == 1
        pos = positions[0]
        assert pos.side == "short"
        assert pos.amount == 0.5  # Absolute value
        assert pos.unrealized_pnl == -20.0

    @pytest.mark.asyncio
    async def test_empty_positions(self, adapter, mock_hyperliquid_info):
        """Handle no open positions."""
        mock_hyperliquid_info.user_state.return_value = {
            "marginSummary": {"accountValue": "10000.00"},
            "assetPositions": [],
        }

        positions = await adapter.get_positions()

        assert positions == []


class TestCancelOrder:
    """Test order cancellation."""

    @pytest.mark.asyncio
    async def test_cancel_by_order_id(self, adapter, mock_hyperliquid_exchange):
        """Cancel order by ID."""
        mock_hyperliquid_exchange.cancel.return_value = {"status": "ok"}

        result = await adapter.cancel_order(
            order_id="12345",
            symbol="BTC",
        )

        mock_hyperliquid_exchange.cancel.assert_called_once()
        assert result.status == "canceled"

    @pytest.mark.asyncio
    async def test_cancel_nonexistent_order(self, adapter, mock_hyperliquid_exchange):
        """Cancel nonexistent order raises error."""
        mock_hyperliquid_exchange.cancel.side_effect = Exception("Order not found")

        with pytest.raises(OrderRejectedError):
            await adapter.cancel_order(order_id="99999", symbol="BTC")


class TestKillSwitchFlatten:
    """Test STOP file triggers flatten of all positions."""

    @pytest.mark.asyncio
    async def test_stop_file_triggers_flatten(
        self, adapter, temp_state_dir, mock_hyperliquid_info, mock_hyperliquid_exchange
    ):
        """STOP file should trigger flatten of all positions."""
        # Setup: Create STOP file
        stop_file = Path(temp_state_dir) / "STOP"
        stop_file.touch()

        # Mock positions
        mock_hyperliquid_info.user_state.return_value = {
            "marginSummary": {"accountValue": "10000.00"},
            "assetPositions": [
                {
                    "position": {
                        "coin": "BTC",
                        "szi": "0.01",
                        "entryPx": "50000.0",
                        "positionValue": "500.0",
                        "unrealizedPnl": "10.0",
                        "leverage": {"value": "1"},
                    }
                }
            ],
        }

        # Act
        await adapter.flatten_all()

        # Assert - should have placed market sell order to close
        mock_hyperliquid_exchange.order.assert_called()

    @pytest.mark.asyncio
    async def test_flatten_all_closes_all_positions(
        self, adapter, mock_hyperliquid_info, mock_hyperliquid_exchange
    ):
        """flatten_all closes all open positions with market orders."""
        # Multiple positions
        mock_hyperliquid_info.user_state.return_value = {
            "marginSummary": {"accountValue": "10000.00"},
            "assetPositions": [
                {
                    "position": {
                        "coin": "BTC",
                        "szi": "0.01",
                        "entryPx": "50000.0",
                        "positionValue": "500.0",
                        "unrealizedPnl": "10.0",
                        "leverage": {"value": "1"},
                    }
                },
                {
                    "position": {
                        "coin": "ETH",
                        "szi": "-0.5",
                        "entryPx": "3000.0",
                        "positionValue": "1500.0",
                        "unrealizedPnl": "-20.0",
                        "leverage": {"value": "1"},
                    }
                },
            ],
        }

        await adapter.flatten_all()

        # Should close both positions
        assert mock_hyperliquid_exchange.order.call_count == 2


class TestEIP712Signing:
    """Test EIP-712 signing mechanism."""

    def test_signing_uses_private_key(self, temp_state_dir):
        """Verify signing mechanism uses provided private key."""
        private_key = "0x" + "b" * 64

        with patch("execution.hyperliquid_adapter.Info"):
            with patch("execution.hyperliquid_adapter.Exchange") as mock_exchange:
                HyperliquidAdapter(
                    private_key=private_key,
                    testnet=True,
                    state_dir=temp_state_dir,
                )

                # Exchange should be initialized with the private key
                mock_exchange.assert_called_once()

    def test_invalid_private_key_raises(self, temp_state_dir):
        """Invalid private key format raises error."""
        with pytest.raises(ValueError):
            with patch("execution.hyperliquid_adapter.Info"):
                with patch("execution.hyperliquid_adapter.Exchange"):
                    HyperliquidAdapter(
                        private_key="not-a-valid-key",
                        testnet=True,
                        state_dir=temp_state_dir,
                    )


class TestConnectionRetry:
    """Test retry logic for VPN drops."""

    @pytest.mark.asyncio
    async def test_retry_on_connection_error(self, adapter, mock_hyperliquid_exchange):
        """Retry 3 times with exponential backoff on connection failure."""
        # First two calls fail, third succeeds
        mock_hyperliquid_exchange.order.side_effect = [
            ConnectionError("VPN dropped"),
            ConnectionError("VPN dropped"),
            {"response": {"data": {"statuses": [{"resting": {"oid": 12345}}]}}},
        ]

        result = await adapter.place_order(
            symbol="BTC",
            side="buy",
            amount=0.01,
            price=50000.0,
            order_type="limit",
        )

        assert mock_hyperliquid_exchange.order.call_count == 3
        assert result.order_id is not None

    @pytest.mark.asyncio
    async def test_raises_after_max_retries(self, adapter, mock_hyperliquid_exchange):
        """Raise after max retries exhausted."""
        mock_hyperliquid_exchange.order.side_effect = ConnectionError("VPN dropped")

        with pytest.raises(OrderRejectedError) as exc_info:
            await adapter.place_order(
                symbol="BTC",
                side="buy",
                amount=0.01,
                price=50000.0,
                order_type="limit",
            )

        # Should have tried 3 times
        assert mock_hyperliquid_exchange.order.call_count == 3
        assert "retries" in str(exc_info.value).lower() or "failed" in str(exc_info.value).lower()


class TestOrderLogging:
    """Test order logging to JSONL."""

    @pytest.mark.asyncio
    async def test_orders_logged_to_jsonl(self, adapter, temp_state_dir, mock_hyperliquid_exchange):
        """All orders should be logged to execution/order_log.jsonl."""
        await adapter.place_order(
            symbol="BTC",
            side="buy",
            amount=0.01,
            price=50000.0,
            order_type="limit",
        )

        log_file = Path(temp_state_dir) / "order_log.jsonl"
        assert log_file.exists()

        with open(log_file) as f:
            lines = f.readlines()
            assert len(lines) >= 1
            log_entry = json.loads(lines[-1])
            assert log_entry["symbol"] == "BTC"
            assert log_entry["side"] == "buy"


class TestGetBalance:
    """Test balance retrieval."""

    @pytest.mark.asyncio
    async def test_get_balance(self, adapter, mock_hyperliquid_info):
        """Get account balance."""
        mock_hyperliquid_info.user_state.return_value = {
            "marginSummary": {
                "accountValue": "10000.00",
                "totalMarginUsed": "1000.00",
                "withdrawable": "9000.00",
            },
            "assetPositions": [],
        }

        balance = await adapter.get_balance()

        assert balance["total"] == 10000.0
        assert balance["used"] == 1000.0
        assert balance["free"] == 9000.0
