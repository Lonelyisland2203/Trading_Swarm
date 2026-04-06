"""
Tests for execution layer exception hierarchy.

Verifies that all execution exceptions properly inherit from ExecutionError
and correctly store and format their parameters.
"""

import pytest

from execution.exceptions import (
    CooldownActiveError,
    DailyLossLimitError,
    DailyTradeCountError,
    ExecutionError,
    InsufficientBalanceError,
    KillSwitchActiveError,
    LiveTradingNotAllowedError,
    OrderRejectedError,
    PositionLimitError,
    SignalRejectedError,
)


class TestExceptionHierarchy:
    """Verify all execution exceptions inherit from ExecutionError."""

    def test_kill_switch_active_error_inherits_from_execution_error(self):
        """KillSwitchActiveError should inherit from ExecutionError."""
        err = KillSwitchActiveError()
        assert isinstance(err, ExecutionError)

    def test_daily_loss_limit_error_inherits_from_execution_error(self):
        """DailyLossLimitError should inherit from ExecutionError."""
        err = DailyLossLimitError(5.0, 10.0)
        assert isinstance(err, ExecutionError)

    def test_position_limit_error_inherits_from_execution_error(self):
        """PositionLimitError should inherit from ExecutionError."""
        err = PositionLimitError(5, 10)
        assert isinstance(err, ExecutionError)

    def test_cooldown_active_error_inherits_from_execution_error(self):
        """CooldownActiveError should inherit from ExecutionError."""
        err = CooldownActiveError(30.5)
        assert isinstance(err, ExecutionError)

    def test_daily_trade_count_error_inherits_from_execution_error(self):
        """DailyTradeCountError should inherit from ExecutionError."""
        err = DailyTradeCountError(10, 20)
        assert isinstance(err, ExecutionError)

    def test_insufficient_balance_error_inherits_from_execution_error(self):
        """InsufficientBalanceError should inherit from ExecutionError."""
        err = InsufficientBalanceError(100.0, 50.0, "USDT")
        assert isinstance(err, ExecutionError)

    def test_live_trading_not_allowed_error_inherits_from_execution_error(self):
        """LiveTradingNotAllowedError should inherit from ExecutionError."""
        err = LiveTradingNotAllowedError()
        assert isinstance(err, ExecutionError)

    def test_order_rejected_error_inherits_from_execution_error(self):
        """OrderRejectedError should inherit from ExecutionError."""
        err = OrderRejectedError("Invalid price")
        assert isinstance(err, ExecutionError)

    def test_signal_rejected_error_inherits_from_execution_error(self):
        """SignalRejectedError should inherit from ExecutionError."""
        err = SignalRejectedError("Confidence too low")
        assert isinstance(err, ExecutionError)


class TestDailyLossLimitError:
    """Test DailyLossLimitError parameter storage and formatting."""

    def test_stores_current_loss_pct(self):
        """Should store current_loss_pct as instance attribute."""
        err = DailyLossLimitError(5.25, 10.0)
        assert err.current_loss_pct == 5.25

    def test_stores_limit_pct(self):
        """Should store limit_pct as instance attribute."""
        err = DailyLossLimitError(5.0, 10.5)
        assert err.limit_pct == 10.5

    def test_formats_message_with_percentages(self):
        """Error message should include both percentages."""
        err = DailyLossLimitError(5.25, 10.5)
        msg = str(err)
        assert "5.25%" in msg
        assert "10.50%" in msg
        assert "Daily loss limit exceeded" in msg

    def test_message_with_zero_loss(self):
        """Should handle zero loss correctly."""
        err = DailyLossLimitError(0.0, 10.0)
        msg = str(err)
        assert "0.00%" in msg

    def test_message_with_large_percentages(self):
        """Should handle large percentage values."""
        err = DailyLossLimitError(95.5, 100.0)
        msg = str(err)
        assert "95.50%" in msg
        assert "100.00%" in msg


class TestPositionLimitError:
    """Test PositionLimitError parameter storage and formatting."""

    def test_stores_current_positions(self):
        """Should store current_positions as instance attribute."""
        err = PositionLimitError(5, 10)
        assert err.current_positions == 5

    def test_stores_max_positions(self):
        """Should store max_positions as instance attribute."""
        err = PositionLimitError(5, 15)
        assert err.max_positions == 15

    def test_formats_message_with_counts(self):
        """Error message should include both position counts."""
        err = PositionLimitError(5, 10)
        msg = str(err)
        assert "5/10" in msg
        assert "Position limit reached" in msg

    def test_message_with_zero_positions(self):
        """Should handle zero current positions."""
        err = PositionLimitError(0, 10)
        msg = str(err)
        assert "0/10" in msg

    def test_message_with_single_position(self):
        """Should handle single position correctly."""
        err = PositionLimitError(1, 5)
        msg = str(err)
        assert "1/5" in msg


class TestCooldownActiveError:
    """Test CooldownActiveError parameter storage and formatting."""

    def test_stores_seconds_remaining(self):
        """Should store seconds_remaining as instance attribute."""
        err = CooldownActiveError(30.5)
        assert err.seconds_remaining == 30.5

    def test_formats_message_with_seconds(self):
        """Error message should include remaining seconds."""
        err = CooldownActiveError(30.5)
        msg = str(err)
        assert "30.5s" in msg
        assert "cooldown" in msg.lower()

    def test_message_with_zero_seconds(self):
        """Should handle zero seconds remaining."""
        err = CooldownActiveError(0.0)
        msg = str(err)
        assert "0.0s" in msg

    def test_message_with_fractional_seconds(self):
        """Should format fractional seconds correctly."""
        err = CooldownActiveError(45.123)
        msg = str(err)
        assert "45.1s" in msg

    def test_message_with_large_seconds(self):
        """Should handle large second values."""
        err = CooldownActiveError(3661.5)
        msg = str(err)
        assert "3661.5s" in msg


class TestDailyTradeCountError:
    """Test DailyTradeCountError parameter storage and formatting."""

    def test_stores_current_count(self):
        """Should store current_count as instance attribute."""
        err = DailyTradeCountError(10, 20)
        assert err.current_count == 10

    def test_stores_max_count(self):
        """Should store max_count as instance attribute."""
        err = DailyTradeCountError(10, 25)
        assert err.max_count == 25

    def test_formats_message_with_counts(self):
        """Error message should include both trade counts."""
        err = DailyTradeCountError(10, 20)
        msg = str(err)
        assert "10/20" in msg
        assert "Daily trade limit reached" in msg

    def test_message_with_zero_trades(self):
        """Should handle zero trades correctly."""
        err = DailyTradeCountError(0, 20)
        msg = str(err)
        assert "0/20" in msg

    def test_message_with_max_trades_reached(self):
        """Should handle when max trades are reached."""
        err = DailyTradeCountError(20, 20)
        msg = str(err)
        assert "20/20" in msg


class TestInsufficientBalanceError:
    """Test InsufficientBalanceError parameter storage and formatting."""

    def test_stores_required(self):
        """Should store required balance as instance attribute."""
        err = InsufficientBalanceError(100.5, 50.0, "USDT")
        assert err.required == 100.5

    def test_stores_available(self):
        """Should store available balance as instance attribute."""
        err = InsufficientBalanceError(100.0, 75.5, "USDT")
        assert err.available == 75.5

    def test_stores_asset(self):
        """Should store asset symbol as instance attribute."""
        err = InsufficientBalanceError(100.0, 50.0, "BTC")
        assert err.asset == "BTC"

    def test_formats_message_with_all_details(self):
        """Error message should include required, available, and asset."""
        err = InsufficientBalanceError(100.5, 75.25, "USDT")
        msg = str(err)
        assert "USDT" in msg
        assert "100.5" in msg or "100.50000000" in msg
        assert "75.25" in msg or "75.25000000" in msg

    def test_message_with_different_assets(self):
        """Should handle different asset symbols."""
        err_btc = InsufficientBalanceError(1.5, 0.5, "BTC")
        assert "BTC" in str(err_btc)

        err_eth = InsufficientBalanceError(10.0, 5.0, "ETH")
        assert "ETH" in str(err_eth)

    def test_message_with_small_balances(self):
        """Should handle very small balance values."""
        err = InsufficientBalanceError(0.00000001, 0.000000005, "BTC")
        msg = str(err)
        assert "BTC" in msg

    def test_message_with_zero_available(self):
        """Should handle zero available balance."""
        err = InsufficientBalanceError(100.0, 0.0, "USDT")
        msg = str(err)
        assert "0.00000000" in msg


class TestOrderRejectedError:
    """Test OrderRejectedError parameter storage and formatting."""

    def test_stores_reason(self):
        """Should store reason as instance attribute."""
        err = OrderRejectedError("Invalid price")
        assert err.reason == "Invalid price"

    def test_stores_exchange_error(self):
        """Should store exchange_error as instance attribute."""
        original_error = ValueError("Test error")
        err = OrderRejectedError("Order failed", original_error)
        assert err.exchange_error is original_error

    def test_message_with_reason_only(self):
        """Error message should include reason without exchange error."""
        err = OrderRejectedError("Invalid price")
        msg = str(err)
        assert "Order rejected" in msg
        assert "Invalid price" in msg
        assert "Exchange error" not in msg

    def test_message_with_reason_and_exchange_error(self):
        """Error message should include both reason and exchange error."""
        original_error = ValueError("API returned 400")
        err = OrderRejectedError("Order validation failed", original_error)
        msg = str(err)
        assert "Order rejected" in msg
        assert "Order validation failed" in msg
        assert "Exchange error" in msg
        assert "API returned 400" in msg

    def test_exchange_error_none_by_default(self):
        """exchange_error should default to None."""
        err = OrderRejectedError("Invalid amount")
        assert err.exchange_error is None

    def test_message_with_different_exchange_errors(self):
        """Should handle different types of exchange errors."""
        err_value = OrderRejectedError("Test", ValueError("val error"))
        assert "val error" in str(err_value)

        err_runtime = OrderRejectedError("Test", RuntimeError("runtime error"))
        assert "runtime error" in str(err_runtime)


class TestSignalRejectedError:
    """Test SignalRejectedError parameter storage and formatting."""

    def test_stores_reason(self):
        """Should store reason as instance attribute."""
        err = SignalRejectedError("Confidence too low")
        assert err.reason == "Confidence too low"

    def test_formats_message_with_reason(self):
        """Error message should include rejection reason."""
        err = SignalRejectedError("Confidence too low")
        msg = str(err)
        assert "Signal rejected" in msg
        assert "Confidence too low" in msg

    def test_message_with_different_reasons(self):
        """Should handle various rejection reasons."""
        reasons = [
            "Confidence too low",
            "Volatility too high",
            "Time to expiration too short",
            "Insufficient volume",
        ]
        for reason in reasons:
            err = SignalRejectedError(reason)
            msg = str(err)
            assert reason in msg
            assert "Signal rejected" in msg


class TestLiveTradingNotAllowedError:
    """Test LiveTradingNotAllowedError."""

    def test_instantiation(self):
        """Should instantiate without parameters."""
        err = LiveTradingNotAllowedError()
        assert isinstance(err, ExecutionError)

    def test_message_is_meaningful(self):
        """Should have a meaningful error message."""
        err = LiveTradingNotAllowedError()
        msg = str(err)
        assert len(msg) > 0


class TestKillSwitchActiveError:
    """Test KillSwitchActiveError."""

    def test_instantiation(self):
        """Should instantiate without parameters."""
        err = KillSwitchActiveError()
        assert isinstance(err, ExecutionError)

    def test_message_is_meaningful(self):
        """Should have a meaningful error message."""
        err = KillSwitchActiveError()
        msg = str(err)
        assert len(msg) > 0
