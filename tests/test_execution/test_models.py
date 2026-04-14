"""
Comprehensive tests for execution layer Pydantic models.

Tests all models:
- OrderResult: Order placement results
- OrderStatus: Current order status tracking
- Position: Open position information
- TradeDecision: Accept/reject signal decisions
- DailyStats: Daily trading statistics with circuit breaker tracking
- SignalInput: Input signals from generation pipeline
"""

from datetime import datetime
import pytest
from pydantic import ValidationError

from execution.models import (
    OrderResult,
    OrderStatus,
    Position,
    TradeDecision,
    DailyStats,
    SignalInput,
)


class TestOrderResult:
    """Tests for OrderResult model."""

    def test_valid_creation(self):
        """Test OrderResult can be created with valid data."""
        order = OrderResult(
            order_id="12345",
            symbol="BTCUSDT",
            side="buy",
            order_type="limit",
            amount=0.5,
            price=43000.0,
            status="closed",
            filled=0.5,
            remaining=0.0,
            cost=21500.0,
            fee=10.75,
            timestamp=datetime.now(),
        )
        assert order.order_id == "12345"
        assert order.symbol == "BTCUSDT"
        assert order.side == "buy"
        assert order.filled == 0.5
        assert order.fee == 10.75

    def test_valid_with_none_price(self):
        """Test OrderResult accepts None for price (market orders)."""
        order = OrderResult(
            order_id="12346",
            symbol="ETHUSDT",
            side="sell",
            order_type="market",
            amount=1.0,
            price=None,
            status="closed",
            filled=1.0,
            remaining=0.0,
            cost=2000.0,
            fee=1.0,
            timestamp=datetime.now(),
        )
        assert order.price is None
        assert order.order_type == "market"

    def test_side_validation_buy(self):
        """Test side only accepts 'buy' or 'sell'."""
        order = OrderResult(
            order_id="12347",
            symbol="BTCUSDT",
            side="buy",
            order_type="limit",
            amount=0.5,
            price=43000.0,
            status="open",
            filled=0.0,
            remaining=0.5,
            cost=0.0,
            fee=0.0,
            timestamp=datetime.now(),
        )
        assert order.side == "buy"

    def test_side_validation_sell(self):
        """Test side accepts 'sell'."""
        order = OrderResult(
            order_id="12348",
            symbol="BTCUSDT",
            side="sell",
            order_type="limit",
            amount=0.5,
            price=43000.0,
            status="open",
            filled=0.0,
            remaining=0.5,
            cost=0.0,
            fee=0.0,
            timestamp=datetime.now(),
        )
        assert order.side == "sell"

    def test_side_validation_invalid(self):
        """Test side rejects invalid values."""
        with pytest.raises(ValidationError) as exc_info:
            OrderResult(
                order_id="12349",
                symbol="BTCUSDT",
                side="invalid",
                order_type="limit",
                amount=0.5,
                price=43000.0,
                status="open",
                filled=0.0,
                remaining=0.5,
                cost=0.0,
                fee=0.0,
                timestamp=datetime.now(),
            )
        assert "side" in str(exc_info.value).lower()

    def test_amount_must_be_non_negative(self):
        """Test amount cannot be negative."""
        with pytest.raises(ValidationError):
            OrderResult(
                order_id="12350",
                symbol="BTCUSDT",
                side="buy",
                order_type="limit",
                amount=-0.5,
                price=43000.0,
                status="open",
                filled=0.0,
                remaining=0.5,
                cost=0.0,
                fee=0.0,
                timestamp=datetime.now(),
            )

    def test_filled_remaining_consistency(self):
        """Test filled and remaining fields."""
        order = OrderResult(
            order_id="12351",
            symbol="BTCUSDT",
            side="buy",
            order_type="limit",
            amount=1.0,
            price=43000.0,
            status="partially_filled",
            filled=0.6,
            remaining=0.4,
            cost=25800.0,
            fee=12.9,
            timestamp=datetime.now(),
        )
        assert order.filled == 0.6
        assert order.remaining == 0.4
        assert order.filled + order.remaining == pytest.approx(order.amount)


class TestOrderStatus:
    """Tests for OrderStatus model."""

    def test_valid_creation(self):
        """Test OrderStatus can be created with valid data."""
        status = OrderStatus(
            order_id="12345",
            symbol="BTCUSDT",
            status="open",
            filled=0.0,
            remaining=0.5,
            average_price=None,
        )
        assert status.order_id == "12345"
        assert status.status == "open"

    def test_is_complete_open_order(self):
        """Test is_complete is False for open orders."""
        status = OrderStatus(
            order_id="12345",
            symbol="BTCUSDT",
            status="open",
            filled=0.0,
            remaining=0.5,
            average_price=None,
        )
        assert status.is_complete is False

    def test_is_complete_closed_order(self):
        """Test is_complete is True for closed orders."""
        status = OrderStatus(
            order_id="12345",
            symbol="BTCUSDT",
            status="closed",
            filled=0.5,
            remaining=0.0,
            average_price=43000.0,
        )
        assert status.is_complete is True

    def test_is_complete_canceled_order(self):
        """Test is_complete is True for canceled orders."""
        status = OrderStatus(
            order_id="12345",
            symbol="BTCUSDT",
            status="canceled",
            filled=0.2,
            remaining=0.3,
            average_price=43000.0,
        )
        assert status.is_complete is True

    def test_is_complete_expired_order(self):
        """Test is_complete is True for expired orders."""
        status = OrderStatus(
            order_id="12345",
            symbol="BTCUSDT",
            status="expired",
            filled=0.0,
            remaining=0.5,
            average_price=None,
        )
        assert status.is_complete is True

    def test_is_complete_filled_order(self):
        """Test is_complete is True for filled orders."""
        status = OrderStatus(
            order_id="12345",
            symbol="BTCUSDT",
            status="filled",
            filled=0.5,
            remaining=0.0,
            average_price=43000.0,
        )
        assert status.is_complete is True

    def test_is_complete_partially_filled(self):
        """Test is_complete is False for partially filled orders."""
        status = OrderStatus(
            order_id="12345",
            symbol="BTCUSDT",
            status="partially_filled",
            filled=0.3,
            remaining=0.2,
            average_price=43000.0,
        )
        assert status.is_complete is False


class TestPosition:
    """Tests for Position model."""

    def test_long_position(self):
        """Test creating a long position."""
        pos = Position(
            symbol="BTCUSDT",
            side="long",
            amount=0.5,
            entry_price=43000.0,
            mark_price=44000.0,
            unrealized_pnl=500.0,
            leverage=2,
        )
        assert pos.side == "long"
        assert pos.amount == 0.5
        assert pos.leverage == 2

    def test_short_position(self):
        """Test creating a short position."""
        pos = Position(
            symbol="ETHUSDT",
            side="short",
            amount=2.0,
            entry_price=2500.0,
            mark_price=2450.0,
            unrealized_pnl=100.0,
            leverage=3,
        )
        assert pos.side == "short"
        assert pos.unrealized_pnl == 100.0

    def test_position_with_liquidation_price(self):
        """Test position with liquidation price."""
        pos = Position(
            symbol="BTCUSDT",
            side="long",
            amount=0.5,
            entry_price=43000.0,
            mark_price=44000.0,
            unrealized_pnl=500.0,
            leverage=5,
            liquidation_price=40000.0,
        )
        assert pos.liquidation_price == 40000.0

    def test_position_without_liquidation_price(self):
        """Test position defaults liquidation_price to None."""
        pos = Position(
            symbol="BTCUSDT",
            side="long",
            amount=0.5,
            entry_price=43000.0,
            mark_price=44000.0,
            unrealized_pnl=500.0,
            leverage=1,
        )
        assert pos.liquidation_price is None

    def test_amount_must_be_non_negative(self):
        """Test amount cannot be negative."""
        with pytest.raises(ValidationError):
            Position(
                symbol="BTCUSDT",
                side="long",
                amount=-0.5,
                entry_price=43000.0,
                mark_price=44000.0,
                unrealized_pnl=500.0,
                leverage=1,
            )

    def test_entry_price_must_be_positive(self):
        """Test entry_price must be greater than 0."""
        with pytest.raises(ValidationError):
            Position(
                symbol="BTCUSDT",
                side="long",
                amount=0.5,
                entry_price=0.0,
                mark_price=44000.0,
                unrealized_pnl=500.0,
                leverage=1,
            )

    def test_mark_price_must_be_positive(self):
        """Test mark_price must be greater than 0."""
        with pytest.raises(ValidationError):
            Position(
                symbol="BTCUSDT",
                side="long",
                amount=0.5,
                entry_price=43000.0,
                mark_price=-1000.0,
                unrealized_pnl=500.0,
                leverage=1,
            )

    def test_leverage_bounds(self):
        """Test leverage is between 1 and 125."""
        # Valid: leverage=1
        pos = Position(
            symbol="BTCUSDT",
            side="long",
            amount=0.5,
            entry_price=43000.0,
            mark_price=44000.0,
            unrealized_pnl=500.0,
            leverage=1,
        )
        assert pos.leverage == 1

        # Valid: leverage=125
        pos = Position(
            symbol="BTCUSDT",
            side="long",
            amount=0.5,
            entry_price=43000.0,
            mark_price=44000.0,
            unrealized_pnl=500.0,
            leverage=125,
        )
        assert pos.leverage == 125

        # Invalid: leverage < 1
        with pytest.raises(ValidationError):
            Position(
                symbol="BTCUSDT",
                side="long",
                amount=0.5,
                entry_price=43000.0,
                mark_price=44000.0,
                unrealized_pnl=500.0,
                leverage=0,
            )

        # Invalid: leverage > 125
        with pytest.raises(ValidationError):
            Position(
                symbol="BTCUSDT",
                side="long",
                amount=0.5,
                entry_price=43000.0,
                mark_price=44000.0,
                unrealized_pnl=500.0,
                leverage=126,
            )


class TestTradeDecision:
    """Tests for TradeDecision model."""

    def test_execute_decision(self):
        """Test creating an execute decision."""
        decision = TradeDecision(
            execute=True,
            reason="Signal passed all checks",
            symbol="BTCUSDT",
            side="buy",
            amount=0.5,
            price=43000.0,
            order_type="limit",
            stop_loss_price=42000.0,
            take_profit_price=45000.0,
        )
        assert decision.execute is True
        assert decision.reason == "Signal passed all checks"
        assert decision.symbol == "BTCUSDT"
        assert decision.side == "buy"

    def test_reject_decision(self):
        """Test creating a reject decision."""
        decision = TradeDecision(
            execute=False,
            reason="Insufficient balance",
        )
        assert decision.execute is False
        assert decision.reason == "Insufficient balance"
        assert decision.symbol is None
        assert decision.side is None
        assert decision.amount is None
        assert decision.price is None
        assert decision.order_type is None
        assert decision.stop_loss_price is None
        assert decision.take_profit_price is None

    def test_execute_decision_with_market_order(self):
        """Test execute decision with market order (no price)."""
        decision = TradeDecision(
            execute=True,
            reason="Market order approved",
            symbol="ETHUSDT",
            side="sell",
            amount=2.0,
            price=None,
            order_type="market",
        )
        assert decision.order_type == "market"
        assert decision.price is None

    def test_side_validation(self):
        """Test side only accepts 'buy' or 'sell'."""
        # Valid: buy
        decision = TradeDecision(
            execute=True,
            reason="OK",
            symbol="BTCUSDT",
            side="buy",
            amount=0.5,
        )
        assert decision.side == "buy"

        # Valid: sell
        decision = TradeDecision(
            execute=True,
            reason="OK",
            symbol="BTCUSDT",
            side="sell",
            amount=0.5,
        )
        assert decision.side == "sell"

        # Invalid side
        with pytest.raises(ValidationError):
            TradeDecision(
                execute=True,
                reason="OK",
                symbol="BTCUSDT",
                side="invalid",
                amount=0.5,
            )

    def test_amount_must_be_non_negative(self):
        """Test amount cannot be negative."""
        with pytest.raises(ValidationError):
            TradeDecision(
                execute=True,
                reason="OK",
                symbol="BTCUSDT",
                side="buy",
                amount=-0.5,
            )


class TestDailyStats:
    """Tests for DailyStats model."""

    def test_valid_creation(self):
        """Test DailyStats can be created with valid data."""
        stats = DailyStats(
            date="2026-04-06",
            trade_count=5,
            realized_pnl=150.0,
            starting_balance=10000.0,
        )
        assert stats.date == "2026-04-06"
        assert stats.trade_count == 5
        assert stats.realized_pnl == 150.0
        assert stats.starting_balance == 10000.0

    def test_daily_loss_pct_positive_return(self):
        """Test daily_loss_pct is 0.0 when realized_pnl is positive."""
        stats = DailyStats(
            date="2026-04-06",
            trade_count=3,
            realized_pnl=500.0,
            starting_balance=10000.0,
        )
        assert stats.daily_loss_pct == 0.0

    def test_daily_loss_pct_zero_return(self):
        """Test daily_loss_pct is 0.0 when realized_pnl is zero."""
        stats = DailyStats(
            date="2026-04-06",
            trade_count=2,
            realized_pnl=0.0,
            starting_balance=10000.0,
        )
        assert stats.daily_loss_pct == 0.0

    def test_daily_loss_pct_negative_return(self):
        """Test daily_loss_pct calculation for losses."""
        stats = DailyStats(
            date="2026-04-06",
            trade_count=4,
            realized_pnl=-250.0,
            starting_balance=10000.0,
        )
        # daily_loss_pct = abs(-250) / 10000 * 100 = 2.5
        assert stats.daily_loss_pct == 2.5

    def test_daily_loss_pct_zero_balance(self):
        """Test daily_loss_pct returns 0.0 when starting_balance is zero."""
        stats = DailyStats(
            date="2026-04-06",
            trade_count=1,
            realized_pnl=-100.0,
            starting_balance=0.0,
        )
        assert stats.daily_loss_pct == 0.0

    def test_daily_loss_pct_small_loss(self):
        """Test daily_loss_pct with small loss percentage."""
        stats = DailyStats(
            date="2026-04-06",
            trade_count=1,
            realized_pnl=-10.0,
            starting_balance=10000.0,
        )
        # daily_loss_pct = 10 / 10000 * 100 = 0.1%
        assert stats.daily_loss_pct == pytest.approx(0.1)

    def test_trade_count_default_zero(self):
        """Test trade_count defaults to 0."""
        stats = DailyStats(
            date="2026-04-06",
            realized_pnl=0.0,
            starting_balance=10000.0,
        )
        assert stats.trade_count == 0

    def test_last_order_timestamp_none_by_default(self):
        """Test last_order_timestamp defaults to None."""
        stats = DailyStats(
            date="2026-04-06",
            realized_pnl=0.0,
            starting_balance=10000.0,
        )
        assert stats.last_order_timestamp is None

    def test_last_order_timestamp_set(self):
        """Test last_order_timestamp can be set."""
        now = datetime.now()
        stats = DailyStats(
            date="2026-04-06",
            realized_pnl=0.0,
            starting_balance=10000.0,
            last_order_timestamp=now,
        )
        assert stats.last_order_timestamp == now


class TestSignalInput:
    """Tests for SignalInput model."""

    def test_valid_long_signal(self):
        """Test creating a valid long signal."""
        signal = SignalInput(
            symbol="BTCUSDT",
            direction="long",
            confidence=0.75,
            expected_return_pct=1.5,
            stop_loss_pct=0.5,
        )
        assert signal.symbol == "BTCUSDT"
        assert signal.direction == "long"
        assert signal.confidence == 0.75

    def test_valid_short_signal(self):
        """Test creating a valid short signal."""
        signal = SignalInput(
            symbol="ETHUSDT",
            direction="short",
            confidence=0.65,
            expected_return_pct=2.0,
            stop_loss_pct=1.0,
        )
        assert signal.direction == "short"
        assert signal.confidence == 0.65

    def test_confidence_range_validation_valid(self):
        """Test confidence accepts values in [0, 1]."""
        # Min: 0.0
        signal = SignalInput(
            symbol="BTCUSDT",
            direction="long",
            confidence=0.0,
            expected_return_pct=1.0,
            stop_loss_pct=0.5,
        )
        assert signal.confidence == 0.0

        # Max: 1.0
        signal = SignalInput(
            symbol="BTCUSDT",
            direction="long",
            confidence=1.0,
            expected_return_pct=1.0,
            stop_loss_pct=0.5,
        )
        assert signal.confidence == 1.0

    def test_confidence_range_validation_too_low(self):
        """Test confidence rejects values below 0."""
        with pytest.raises(ValidationError):
            SignalInput(
                symbol="BTCUSDT",
                direction="long",
                confidence=-0.1,
                expected_return_pct=1.0,
                stop_loss_pct=0.5,
            )

    def test_confidence_range_validation_too_high(self):
        """Test confidence rejects values above 1."""
        with pytest.raises(ValidationError):
            SignalInput(
                symbol="BTCUSDT",
                direction="long",
                confidence=1.1,
                expected_return_pct=1.0,
                stop_loss_pct=0.5,
            )

    def test_direction_validation_long(self):
        """Test direction accepts 'long'."""
        signal = SignalInput(
            symbol="BTCUSDT",
            direction="long",
            confidence=0.5,
            expected_return_pct=1.0,
            stop_loss_pct=0.5,
        )
        assert signal.direction == "long"

    def test_direction_validation_short(self):
        """Test direction accepts 'short'."""
        signal = SignalInput(
            symbol="BTCUSDT",
            direction="short",
            confidence=0.5,
            expected_return_pct=1.0,
            stop_loss_pct=0.5,
        )
        assert signal.direction == "short"

    def test_direction_validation_invalid(self):
        """Test direction rejects invalid values."""
        with pytest.raises(ValidationError):
            SignalInput(
                symbol="BTCUSDT",
                direction="invalid",
                confidence=0.5,
                expected_return_pct=1.0,
                stop_loss_pct=0.5,
            )

    def test_stop_loss_pct_must_be_positive(self):
        """Test stop_loss_pct must be greater than 0."""
        with pytest.raises(ValidationError):
            SignalInput(
                symbol="BTCUSDT",
                direction="long",
                confidence=0.5,
                expected_return_pct=1.0,
                stop_loss_pct=0.0,
            )

    def test_with_optional_fields(self):
        """Test SignalInput with all optional fields."""
        signal = SignalInput(
            symbol="BTCUSDT",
            direction="long",
            confidence=0.8,
            expected_return_pct=2.5,
            stop_loss_pct=1.0,
            take_profit_pct=5.0,
            timeframe="4h",
            entry_price=44000.0,
            metadata={"indicator": "rsi", "value": 72},
        )
        assert signal.take_profit_pct == 5.0
        assert signal.timeframe == "4h"
        assert signal.entry_price == 44000.0
        assert signal.metadata == {"indicator": "rsi", "value": 72}

    def test_timeframe_default(self):
        """Test timeframe defaults to '1h'."""
        signal = SignalInput(
            symbol="BTCUSDT",
            direction="long",
            confidence=0.5,
            expected_return_pct=1.0,
            stop_loss_pct=0.5,
        )
        assert signal.timeframe == "1h"

    def test_entry_price_none_by_default(self):
        """Test entry_price defaults to None."""
        signal = SignalInput(
            symbol="BTCUSDT",
            direction="long",
            confidence=0.5,
            expected_return_pct=1.0,
            stop_loss_pct=0.5,
        )
        assert signal.entry_price is None

    def test_metadata_none_by_default(self):
        """Test metadata defaults to None."""
        signal = SignalInput(
            symbol="BTCUSDT",
            direction="long",
            confidence=0.5,
            expected_return_pct=1.0,
            stop_loss_pct=0.5,
        )
        assert signal.metadata is None

    def test_negative_expected_return(self):
        """Test negative expected_return_pct is allowed."""
        signal = SignalInput(
            symbol="BTCUSDT",
            direction="long",
            confidence=0.5,
            expected_return_pct=-1.0,
            stop_loss_pct=0.5,
        )
        assert signal.expected_return_pct == -1.0
