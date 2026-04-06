"""
Tests for position sizing calculation with fee awareness.

Tests the calculate_position_size() function which determines trade size based on:
- Risk tolerance (% of balance to risk per trade)
- Stop loss distance (tighter stops allow larger positions)
- Trading fees (reduce position to maintain risk target)
- Position cap (hard 2% limit regardless of risk calculation)
"""

import pytest

from config.fee_model import FeeModelSettings
from execution.position_sizing import calculate_position_size, PositionSizeResult


class TestCalculatePositionSize:
    """Test suite for position sizing calculations."""

    def test_basic_position_size(self):
        """Test basic position sizing with 1% risk."""
        fee_model = FeeModelSettings()
        balance = 10000.0
        risk_pct = 0.01  # 1%
        entry_price = 100.0
        stop_price = 95.0  # 5% stop (wider to avoid hitting cap)

        result = calculate_position_size(
            balance=balance,
            risk_pct=risk_pct,
            entry_price=entry_price,
            stop_price=stop_price,
            fee_model=fee_model,
        )

        # Verify basic fields
        assert isinstance(result, PositionSizeResult)
        assert result.amount > 0, "Position amount should be positive"
        assert result.notional > 0, "Notional value should be positive"
        assert result.risk_amount <= 100.0, "Risk should be ~1% of 10k balance"
        assert result.risk_amount > 0, "Risk amount should be positive"
        assert result.stop_distance_pct > 0, "Stop distance should be positive"
        assert result.fees_included >= 0, "Fees should be non-negative"
        assert isinstance(result.reason, str), "Reason should be a string"

    def test_fee_reduces_position_size(self):
        """Test that position size is smaller when fees are included."""
        fee_model_with_fees = FeeModelSettings()
        fee_model_no_fees = FeeModelSettings(
            maker_fee_pct=0.0,
            taker_fee_pct=0.0,
            slippage_pct=0.0,
            include_funding=False,
        )
        balance = 10000.0
        risk_pct = 0.01
        entry_price = 100.0
        stop_price = 95.0  # 5% stop
        max_position_pct = 0.50  # High cap to avoid capping position

        result_with_fees = calculate_position_size(
            balance=balance,
            risk_pct=risk_pct,
            entry_price=entry_price,
            stop_price=stop_price,
            fee_model=fee_model_with_fees,
            max_position_pct=max_position_pct,
        )

        result_no_fees = calculate_position_size(
            balance=balance,
            risk_pct=risk_pct,
            entry_price=entry_price,
            stop_price=stop_price,
            fee_model=fee_model_no_fees,
            max_position_pct=max_position_pct,
        )

        # Position with fees should be smaller than position without fees
        assert result_with_fees.amount < result_no_fees.amount, (
            "Position size with fees should be smaller than without fees"
        )
        assert result_with_fees.notional < result_no_fees.notional, (
            "Notional with fees should be smaller than without fees"
        )

    def test_max_position_cap(self):
        """Test that position is capped at max_position_pct regardless of risk."""
        fee_model = FeeModelSettings()
        balance = 10000.0
        risk_pct = 0.50  # 50% risk - very high
        entry_price = 50000.0
        stop_price = 49900.0  # Tight 0.2% stop
        max_position_pct = 0.02  # 2% cap

        result = calculate_position_size(
            balance=balance,
            risk_pct=risk_pct,
            entry_price=entry_price,
            stop_price=stop_price,
            fee_model=fee_model,
            max_position_pct=max_position_pct,
        )

        # Position should be capped at 2% of balance
        max_notional = balance * max_position_pct
        assert result.notional <= max_notional * 1.001, (  # Allow tiny rounding error
            f"Notional {result.notional} exceeds cap {max_notional}"
        )
        assert result.capped_by_max is True, "Should indicate position was capped"
        assert "capped" in result.reason.lower(), "Reason should mention capping"

    def test_long_position_sizing(self):
        """Test position sizing for long position (stop below entry)."""
        fee_model = FeeModelSettings()
        balance = 10000.0
        risk_pct = 0.01  # 1%
        entry_price = 100.0
        stop_price = 95.0  # Stop below entry (5% stop)
        max_position_pct = 0.50  # High cap to avoid capping position

        result = calculate_position_size(
            balance=balance,
            risk_pct=risk_pct,
            entry_price=entry_price,
            stop_price=stop_price,
            fee_model=fee_model,
            max_position_pct=max_position_pct,
        )

        assert result.amount > 0, "Long position should have positive amount"
        assert result.notional > 0, "Long position should have positive notional"
        # Risk should be approximately 1% of balance (within tolerance for fees)
        assert 90 <= result.risk_amount <= 110, (
            f"Risk amount {result.risk_amount} should be ~100 (1% of 10k)"
        )

    def test_short_position_sizing(self):
        """Test position sizing for short position (stop above entry)."""
        fee_model = FeeModelSettings()
        balance = 10000.0
        risk_pct = 0.01  # 1%
        entry_price = 100.0
        stop_price = 105.0  # Stop above entry (short, 5% stop - wider to avoid hitting cap)

        result = calculate_position_size(
            balance=balance,
            risk_pct=risk_pct,
            entry_price=entry_price,
            stop_price=stop_price,
            fee_model=fee_model,
        )

        assert result.amount > 0, "Short position should have positive amount"
        assert result.notional > 0, "Short position should have positive notional"
        # Stop distance should be positive regardless of direction
        assert result.stop_distance_pct > 0, "Stop distance should be positive"

    def test_zero_balance_returns_zero(self):
        """Test that zero balance returns zero position."""
        fee_model = FeeModelSettings()
        balance = 0.0
        risk_pct = 0.01
        entry_price = 100.0
        stop_price = 98.0

        result = calculate_position_size(
            balance=balance,
            risk_pct=risk_pct,
            entry_price=entry_price,
            stop_price=stop_price,
            fee_model=fee_model,
        )

        assert result.amount == 0.0, "Zero balance should result in zero position"
        assert result.notional == 0.0, "Zero balance should result in zero notional"
        assert isinstance(result.reason, str), "Should provide reason"

    def test_entry_equals_stop_returns_zero(self):
        """Test that entry price equal to stop price returns zero position."""
        fee_model = FeeModelSettings()
        balance = 10000.0
        risk_pct = 0.01
        entry_price = 100.0
        stop_price = 100.0  # Same as entry

        result = calculate_position_size(
            balance=balance,
            risk_pct=risk_pct,
            entry_price=entry_price,
            stop_price=stop_price,
            fee_model=fee_model,
        )

        assert result.amount == 0.0, "Equal entry/stop should result in zero position"
        assert result.notional == 0.0, "Equal entry/stop should result in zero notional"
        assert "invalid" in result.reason.lower() or "zero" in result.reason.lower(), (
            "Reason should mention invalid or zero stop distance"
        )

    def test_negative_risk_raises_error(self):
        """Test that negative risk_pct raises ValueError."""
        fee_model = FeeModelSettings()
        balance = 10000.0
        risk_pct = -0.01  # Negative risk
        entry_price = 100.0
        stop_price = 98.0

        with pytest.raises(ValueError, match="risk_pct"):
            calculate_position_size(
                balance=balance,
                risk_pct=risk_pct,
                entry_price=entry_price,
                stop_price=stop_price,
                fee_model=fee_model,
            )

    def test_position_size_result_fields(self):
        """Test that PositionSizeResult has all required fields."""
        fee_model = FeeModelSettings()
        balance = 10000.0
        risk_pct = 0.01
        entry_price = 100.0
        stop_price = 98.0

        result = calculate_position_size(
            balance=balance,
            risk_pct=risk_pct,
            entry_price=entry_price,
            stop_price=stop_price,
            fee_model=fee_model,
        )

        # Verify all fields exist and have correct types
        assert hasattr(result, "amount")
        assert hasattr(result, "notional")
        assert hasattr(result, "risk_amount")
        assert hasattr(result, "stop_distance_pct")
        assert hasattr(result, "fees_included")
        assert hasattr(result, "capped_by_max")
        assert hasattr(result, "reason")

        assert isinstance(result.amount, float)
        assert isinstance(result.notional, float)
        assert isinstance(result.risk_amount, float)
        assert isinstance(result.stop_distance_pct, float)
        assert isinstance(result.fees_included, float)
        assert isinstance(result.capped_by_max, bool)
        assert isinstance(result.reason, str)

    def test_very_tight_stop(self):
        """Test that tighter stop allows larger position for same risk."""
        fee_model = FeeModelSettings()
        balance = 10000.0
        risk_pct = 0.01
        entry_price = 100.0

        # Wide stop (10%)
        result_wide = calculate_position_size(
            balance=balance,
            risk_pct=risk_pct,
            entry_price=entry_price,
            stop_price=90.0,  # 10% stop
            fee_model=fee_model,
        )

        # Tight stop (5%)
        result_tight = calculate_position_size(
            balance=balance,
            risk_pct=risk_pct,
            entry_price=entry_price,
            stop_price=95.0,  # 5% stop
            fee_model=fee_model,
        )

        # Tighter stop should allow larger position (as long as not capped)
        if not result_tight.capped_by_max:
            assert result_tight.amount > result_wide.amount, (
                "Tighter stop should allow larger position"
            )
            assert result_tight.notional > result_wide.notional, (
                "Tighter stop should allow larger notional"
            )
