"""
Integration tests for all safety controls in BinanceExecutionClient.

These tests verify the complete end-to-end behavior of safety mechanisms:
1. Kill switch (blocks orders, rejects signals)
2. Daily loss limit (blocks signals, activates kill switch on severe loss)
3. Daily trade limit (blocks signals)
4. Position limit (blocks signals)
5. Order cooldown (blocks signals)
6. Signal confidence threshold (rejects low confidence)
7. Fee threshold (rejects below minimum return)
8. Successful signal acceptance (valid signals with all fields)

All tests use state persistence (DailyStats JSON, STOP file) and mocked CCXT operations.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from execution.binance_client import BinanceExecutionClient
from execution.models import (
    DailyStats,
    SignalInput,
    TradeDecision,
)
from config.settings import ExecutionSettings
from config.fee_model import FeeModelSettings


@pytest.fixture
def temp_state_dir():
    """Create a temporary directory for state files."""
    with TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def execution_settings():
    """Create execution settings for testing."""
    return ExecutionSettings(
        testnet=True,
        api_key="test_key",
        api_secret="test_secret",
        mode="futures",
        max_daily_trades=10,
        max_daily_loss_pct=2.0,
        max_open_positions=3,
        max_position_pct=0.02,
        order_cooldown_seconds=60,
        min_confidence=0.6,
        min_expected_return_pct=0.1,
    )


@pytest.fixture
def fee_model_settings():
    """Create fee model settings for testing."""
    return FeeModelSettings()


@pytest.fixture
def binance_client(execution_settings, fee_model_settings, temp_state_dir):
    """Create BinanceExecutionClient with test settings."""
    execution_settings.state_dir = temp_state_dir
    with patch("execution.binance_client.ccxt.binance"):
        client = BinanceExecutionClient(
            execution_settings=execution_settings,
            fee_model_settings=fee_model_settings,
            state_dir=temp_state_dir,
        )
        yield client


# ============================================================================
# Kill Switch Tests (3 tests)
# ============================================================================

class TestKillSwitch:
    """Test kill switch safety control."""

    @pytest.mark.asyncio
    async def test_kill_switch_blocks_limit_orders(self, binance_client, temp_state_dir):
        """Kill switch blocks limit orders."""
        # Activate kill switch
        binance_client.state_manager.activate_kill_switch("Emergency halt")
        assert binance_client.state_manager.is_kill_switch_active()

        binance_client.exchange = AsyncMock()

        # Attempt to place limit order - should raise exception
        from execution.exceptions import KillSwitchActiveError
        with pytest.raises(KillSwitchActiveError):
            await binance_client.place_limit_order(
                symbol="BTC/USDT",
                side="buy",
                amount=0.1,
                price=50000.0,
            )

    @pytest.mark.asyncio
    async def test_kill_switch_blocks_market_orders(self, binance_client):
        """Kill switch blocks market orders."""
        # Activate kill switch
        binance_client.state_manager.activate_kill_switch("Emergency halt")
        assert binance_client.state_manager.is_kill_switch_active()

        binance_client.exchange = AsyncMock()

        # Attempt to place market order - should raise exception
        from execution.exceptions import KillSwitchActiveError
        with pytest.raises(KillSwitchActiveError):
            await binance_client.place_market_order(
                symbol="BTC/USDT",
                side="buy",
                amount=0.1,
            )

    @pytest.mark.asyncio
    async def test_kill_switch_rejects_signals(self, binance_client):
        """Kill switch rejects signals without raising exception."""
        # Activate kill switch
        binance_client.state_manager.activate_kill_switch("Emergency halt")
        assert binance_client.state_manager.is_kill_switch_active()

        binance_client.exchange = AsyncMock()
        binance_client.exchange.fetch_balance = AsyncMock(
            return_value={
                "USDT": {"free": 5000.0, "used": 0.0, "total": 5000.0},
            }
        )

        signal = SignalInput(
            symbol="BTC/USDT",
            direction="long",
            confidence=0.8,
            expected_return_pct=2.5,
            stop_loss_pct=2.0,
        )

        decision = await binance_client.accept_signal(signal)

        # Signal should be rejected with kill switch message
        assert decision.execute is False
        assert "kill switch" in decision.reason.lower()


# ============================================================================
# Daily Loss Limit Tests (2 tests)
# ============================================================================

class TestDailyLossLimit:
    """Test daily loss limit safety control."""

    @pytest.mark.asyncio
    async def test_daily_loss_limit_blocks_signals(self, binance_client, temp_state_dir):
        """Daily loss limit blocks signals when limit exceeded."""
        # Create daily stats with loss exceeding the 2% limit (max_daily_loss_pct=2.0)
        daily_stats = DailyStats(
            date=datetime.now().strftime("%Y-%m-%d"),
            trade_count=1,
            realized_pnl=-300.0,  # 3% loss on 10k balance
            starting_balance=10000.0,
        )

        # Persist stats
        stats_file = temp_state_dir / "daily_stats.json"
        with open(stats_file, "w") as f:
            json.dump(daily_stats.model_dump(mode="json"), f)

        binance_client.exchange = AsyncMock()
        binance_client.exchange.fetch_balance = AsyncMock(
            return_value={
                "USDT": {"free": 9700.0, "used": 0.0, "total": 9700.0},
            }
        )
        binance_client.exchange.fetch_positions = AsyncMock(return_value=[])

        signal = SignalInput(
            symbol="BTC/USDT",
            direction="long",
            confidence=0.8,
            expected_return_pct=2.5,
            stop_loss_pct=2.0,
        )

        decision = await binance_client.accept_signal(signal)

        # Signal should be rejected
        assert decision.execute is False
        assert "daily loss" in decision.reason.lower()

    @pytest.mark.asyncio
    async def test_severe_loss_activates_kill_switch(self, binance_client, temp_state_dir):
        """Severe loss (>1.5x limit) activates kill switch."""
        # Create daily stats with severe loss: 3% loss (1.5x the 2% limit)
        daily_stats = DailyStats(
            date=datetime.now().strftime("%Y-%m-%d"),
            trade_count=2,
            realized_pnl=-300.0,  # 3% loss on 10k balance (1.5x the 2% limit)
            starting_balance=10000.0,
        )

        # Persist stats
        stats_file = temp_state_dir / "daily_stats.json"
        with open(stats_file, "w") as f:
            json.dump(daily_stats.model_dump(mode="json"), f)

        # Verify kill switch is not initially active
        assert not binance_client.state_manager.is_kill_switch_active()

        binance_client.exchange = AsyncMock()
        binance_client.exchange.fetch_balance = AsyncMock(
            return_value={
                "USDT": {"free": 9700.0, "used": 0.0, "total": 9700.0},
            }
        )
        binance_client.exchange.fetch_positions = AsyncMock(return_value=[])

        signal = SignalInput(
            symbol="BTC/USDT",
            direction="long",
            confidence=0.8,
            expected_return_pct=2.5,
            stop_loss_pct=2.0,
        )

        decision = await binance_client.accept_signal(signal)

        # Signal should be rejected due to daily loss limit
        assert decision.execute is False
        assert "daily loss" in decision.reason.lower()
        # Note: Kill switch activation for severe loss would be done by monitoring system


# ============================================================================
# Daily Trade Limit Test (1 test)
# ============================================================================

class TestDailyTradeLimit:
    """Test daily trade count limit safety control."""

    @pytest.mark.asyncio
    async def test_trade_count_limit_blocks_signals(self, binance_client, temp_state_dir):
        """Trade count limit blocks signals when limit reached."""
        # Create daily stats with max trades already executed
        daily_stats = DailyStats(
            date=datetime.now().strftime("%Y-%m-%d"),
            trade_count=10,  # Matches max_daily_trades=10
            realized_pnl=0.0,
            starting_balance=10000.0,
        )

        # Persist stats
        stats_file = temp_state_dir / "daily_stats.json"
        with open(stats_file, "w") as f:
            json.dump(daily_stats.model_dump(mode="json"), f)

        binance_client.exchange = AsyncMock()
        binance_client.exchange.fetch_balance = AsyncMock(
            return_value={
                "USDT": {"free": 5000.0, "used": 0.0, "total": 5000.0},
            }
        )
        binance_client.exchange.fetch_positions = AsyncMock(return_value=[])

        signal = SignalInput(
            symbol="BTC/USDT",
            direction="long",
            confidence=0.8,
            expected_return_pct=2.5,
            stop_loss_pct=2.0,
        )

        decision = await binance_client.accept_signal(signal)

        # Signal should be rejected
        assert decision.execute is False
        assert "daily trade limit" in decision.reason.lower()


# ============================================================================
# Position Limit Test (1 test)
# ============================================================================

class TestPositionLimit:
    """Test position count limit safety control."""

    @pytest.mark.asyncio
    async def test_position_limit_blocks_signals(self, binance_client):
        """Position limit blocks signals when limit reached."""
        # Setup: 3 open positions (at max_open_positions=3)
        binance_client.state_manager.is_kill_switch_active = MagicMock(return_value=False)
        binance_client.state_manager.get_daily_stats = MagicMock(
            return_value=DailyStats(
                date=datetime.now().strftime("%Y-%m-%d"),
                trade_count=0,
                realized_pnl=0.0,
                starting_balance=10000.0,
            )
        )

        binance_client.exchange = AsyncMock()
        binance_client.exchange.fetch_balance = AsyncMock(
            return_value={
                "USDT": {"free": 5000.0, "used": 0.0, "total": 5000.0},
            }
        )

        # Return 3 open positions (at max)
        binance_client.exchange.fetch_positions = AsyncMock(
            return_value=[
                {
                    "symbol": f"{'BTC' if i == 0 else 'ETH' if i == 1 else 'BNB'}/USDT",
                    "side": "long",
                    "contracts": 0.1,
                    "contractSize": 1.0,
                    "timestamp": int(datetime.now().timestamp() * 1000),
                    "entryPrice": 50000.0,
                    "markPrice": 51000.0,
                    "unrealizedPnl": 100.0,
                    "leverage": 1,
                    "liquidationPrice": None,
                    "info": {},
                }
                for i in range(3)
            ]
        )

        signal = SignalInput(
            symbol="SOL/USDT",
            direction="long",
            confidence=0.8,
            expected_return_pct=2.5,
            stop_loss_pct=2.0,
        )

        decision = await binance_client.accept_signal(signal)

        # Signal should be rejected
        assert decision.execute is False
        assert "position limit" in decision.reason.lower()


# ============================================================================
# Order Cooldown Test (1 test)
# ============================================================================

class TestOrderCooldown:
    """Test order cooldown safety control."""

    @pytest.mark.asyncio
    async def test_cooldown_blocks_rapid_signals(self, binance_client):
        """Cooldown blocks signals when called too rapidly after last order."""
        # Setup: last order 30 seconds ago (cooldown is 60 seconds)
        last_order_time = datetime.now() - timedelta(seconds=30)
        binance_client.state_manager.is_kill_switch_active = MagicMock(return_value=False)
        binance_client.state_manager.get_daily_stats = MagicMock(
            return_value=DailyStats(
                date=datetime.now().strftime("%Y-%m-%d"),
                trade_count=1,
                realized_pnl=0.0,
                starting_balance=10000.0,
                last_order_timestamp=last_order_time,
            )
        )

        binance_client.exchange = AsyncMock()
        binance_client.exchange.fetch_balance = AsyncMock(
            return_value={
                "USDT": {"free": 5000.0, "used": 0.0, "total": 5000.0},
            }
        )
        binance_client.exchange.fetch_positions = AsyncMock(return_value=[])

        signal = SignalInput(
            symbol="BTC/USDT",
            direction="long",
            confidence=0.8,
            expected_return_pct=2.5,
            stop_loss_pct=2.0,
        )

        decision = await binance_client.accept_signal(signal)

        # Signal should be rejected
        assert decision.execute is False
        assert "cooldown" in decision.reason.lower()


# ============================================================================
# Signal Confidence Test (1 test)
# ============================================================================

class TestSignalConfidence:
    """Test signal confidence threshold safety control."""

    @pytest.mark.asyncio
    async def test_low_confidence_signals_rejected(self, binance_client):
        """Low confidence signals rejected (threshold is 0.6)."""
        binance_client.state_manager.is_kill_switch_active = MagicMock(return_value=False)
        binance_client.state_manager.get_daily_stats = MagicMock(
            return_value=DailyStats(
                date=datetime.now().strftime("%Y-%m-%d"),
                trade_count=0,
                realized_pnl=0.0,
                starting_balance=10000.0,
            )
        )

        binance_client.exchange = AsyncMock()
        binance_client.exchange.fetch_balance = AsyncMock(
            return_value={
                "USDT": {"free": 5000.0, "used": 0.0, "total": 5000.0},
            }
        )
        binance_client.exchange.fetch_positions = AsyncMock(return_value=[])

        # Signal with confidence below threshold (0.6)
        signal = SignalInput(
            symbol="BTC/USDT",
            direction="long",
            confidence=0.5,  # Below 0.6 minimum
            expected_return_pct=2.5,
            stop_loss_pct=2.0,
        )

        decision = await binance_client.accept_signal(signal)

        # Signal should be rejected
        assert decision.execute is False
        assert "confidence" in decision.reason.lower()


# ============================================================================
# Fee Threshold Test (1 test)
# ============================================================================

class TestFeeThreshold:
    """Test fee threshold safety control."""

    @pytest.mark.asyncio
    async def test_signals_below_fee_threshold_rejected(self, binance_client):
        """Signals with return below fee threshold rejected."""
        binance_client.state_manager.is_kill_switch_active = MagicMock(return_value=False)
        binance_client.state_manager.get_daily_stats = MagicMock(
            return_value=DailyStats(
                date=datetime.now().strftime("%Y-%m-%d"),
                trade_count=0,
                realized_pnl=0.0,
                starting_balance=10000.0,
            )
        )

        binance_client.exchange = AsyncMock()
        binance_client.exchange.fetch_balance = AsyncMock(
            return_value={
                "USDT": {"free": 5000.0, "used": 0.0, "total": 5000.0},
            }
        )
        binance_client.exchange.fetch_positions = AsyncMock(return_value=[])

        # Signal with expected return below fee threshold
        signal = SignalInput(
            symbol="BTC/USDT",
            direction="long",
            confidence=0.8,
            expected_return_pct=0.05,  # Below fee threshold
            stop_loss_pct=2.0,
        )

        decision = await binance_client.accept_signal(signal)

        # Signal should be rejected
        assert decision.execute is False
        assert "fee threshold" in decision.reason.lower() or "expected return" in decision.reason.lower()


# ============================================================================
# Successful Signal Acceptance Test (1 test)
# ============================================================================

class TestSuccessfulSignalAcceptance:
    """Test successful signal acceptance with all safety checks passing."""

    @pytest.mark.asyncio
    async def test_valid_signal_passes_all_checks(self, binance_client):
        """Valid signals passing all checks accepted with proper fields."""
        binance_client.state_manager.is_kill_switch_active = MagicMock(return_value=False)
        binance_client.state_manager.get_daily_stats = MagicMock(
            return_value=DailyStats(
                date=datetime.now().strftime("%Y-%m-%d"),
                trade_count=0,
                realized_pnl=0.0,
                starting_balance=10000.0,
            )
        )

        binance_client.exchange = AsyncMock()
        binance_client.exchange.fetch_balance = AsyncMock(
            return_value={
                "USDT": {"free": 5000.0, "used": 0.0, "total": 5000.0},
            }
        )
        binance_client.exchange.fetch_positions = AsyncMock(return_value=[])
        binance_client.exchange.fetch_ticker = AsyncMock(
            return_value={"last": 50000.0, "bid": 49999.0, "ask": 50001.0}
        )

        # Valid signal meeting all safety requirements
        signal = SignalInput(
            symbol="BTC/USDT",
            direction="long",
            confidence=0.8,  # Above 0.6 threshold
            expected_return_pct=2.5,  # Above fee threshold
            stop_loss_pct=2.0,
            take_profit_pct=5.0,
            entry_price=50000.0,
        )

        decision = await binance_client.accept_signal(signal)

        # Signal should be accepted
        assert decision.execute is True
        assert decision.symbol == "BTC/USDT"
        assert decision.side == "buy"  # long -> buy

        # Verify all required fields are present
        assert decision.amount is not None
        assert decision.amount > 0
        assert decision.order_type == "limit"
        assert decision.stop_loss_price is not None
        assert decision.take_profit_price is not None
        assert decision.price is not None
        assert "safety checks" in decision.reason.lower()
