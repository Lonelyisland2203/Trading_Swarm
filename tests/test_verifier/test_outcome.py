"""Tests for outcome computation functions."""

import math

import pandas as pd
import pytest

from verifier.outcome import (
    VerifiedOutcome,
    compute_log_return,
    compute_mae,
    compute_net_return,
    determine_direction,
)


class TestComputeLogReturn:
    """Test log return computation."""
    
    def test_positive_return(self):
        """Test log return for price increase."""
        # 10% gain: exit = 110, entry = 100
        log_ret = compute_log_return(100.0, 110.0)
        
        # ln(1.1) ≈ 0.0953
        assert 0.095 < log_ret < 0.096
    
    def test_negative_return(self):
        """Test log return for price decrease."""
        # 10% loss: exit = 90, entry = 100
        log_ret = compute_log_return(100.0, 90.0)
        
        # ln(0.9) ≈ -0.1054
        assert -0.106 < log_ret < -0.105
    
    def test_zero_return(self):
        """Test log return for no price change."""
        log_ret = compute_log_return(100.0, 100.0)
        
        # ln(1) = 0
        assert log_ret == 0.0
    
    def test_symmetry_of_returns(self):
        """Test that +X% and -X% log returns have similar magnitude."""
        gain = compute_log_return(100.0, 110.0)  # +10%
        loss = compute_log_return(100.0, 90.0)   # -10%

        # Magnitudes should be similar (approximately 1% difference for 10% moves)
        assert abs(abs(gain) - abs(loss)) < 0.011
    
    def test_additivity(self):
        """Test that log returns add across periods."""
        # Period 1: 100 -> 110 (+10%)
        ret1 = compute_log_return(100.0, 110.0)
        
        # Period 2: 110 -> 121 (+10%)
        ret2 = compute_log_return(110.0, 121.0)
        
        # Total: 100 -> 121 (+21%)
        total = compute_log_return(100.0, 121.0)
        
        # Should satisfy: ret1 + ret2 ≈ total
        assert abs((ret1 + ret2) - total) < 0.001
    
    def test_raises_on_negative_entry_price(self):
        """Test that negative entry price raises ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            compute_log_return(-100.0, 110.0)
    
    def test_raises_on_negative_exit_price(self):
        """Test that negative exit price raises ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            compute_log_return(100.0, -110.0)
    
    def test_raises_on_zero_price(self):
        """Test that zero price raises ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            compute_log_return(0.0, 110.0)


class TestComputeNetReturn:
    """Test net return after transaction costs."""
    
    def test_default_txn_cost(self):
        """Test that default 0.1% cost reduces return correctly."""
        gross_return = compute_log_return(100.0, 105.0)  # ~0.0488
        net = compute_net_return(gross_return)
        
        # Should be reduced by ~0.2% (2 trades * 0.1%)
        # But in log space: ln((1-0.001)^2) ≈ -0.002
        expected_reduction = math.log((1 - 0.001) ** 2)
        expected = gross_return + expected_reduction
        
        assert abs(net - expected) < 0.0001
    
    def test_custom_txn_cost(self):
        """Test with custom transaction cost."""
        gross_return = 0.05
        net = compute_net_return(gross_return, txn_cost_pct=0.002, num_trades=2)
        
        # Cost: ln((1-0.002)^2) ≈ -0.004
        expected = gross_return + math.log((1 - 0.002) ** 2)
        
        assert abs(net - expected) < 0.0001
    
    def test_single_trade_cost(self):
        """Test with only one trade (e.g., entry only, exit not yet)."""
        gross_return = 0.03
        net = compute_net_return(gross_return, num_trades=1)
        
        # Cost: ln(1-0.001) ≈ -0.001
        expected = gross_return + math.log(1 - 0.001)
        
        assert abs(net - expected) < 0.0001
    
    def test_zero_cost(self):
        """Test with zero transaction cost."""
        gross_return = 0.04
        net = compute_net_return(gross_return, txn_cost_pct=0.0)
        
        # Net should equal gross
        assert net == gross_return
    
    def test_high_cost_can_make_return_negative(self):
        """Test that high costs can turn positive return negative."""
        small_gain = compute_log_return(100.0, 100.5)  # 0.5% gain
        
        # 1% cost per trade * 2 trades = 2% total cost
        net = compute_net_return(small_gain, txn_cost_pct=0.01, num_trades=2)
        
        # Should be negative after costs
        assert net < 0


class TestComputeMAE:
    """Test Max Adverse Excursion computation."""
    
    def test_mae_for_higher_signal_with_drawdown(self):
        """Test MAE for HIGHER signal that experienced drawdown."""
        # Entry at 100, signal predicts HIGHER
        # Price dropped to 95 (5% adverse) then recovered
        df = pd.DataFrame({
            "open": [100.0, 98.0, 95.0, 97.0],
            "high": [101.0, 99.0, 96.0, 105.0],
            "low": [99.0, 95.0, 94.0, 96.0],
            "close": [98.0, 95.5, 97.0, 105.0],
        })
        
        mae = compute_mae(df, "HIGHER", entry_price=100.0)
        
        # Worst case: low of 94.0
        # MAE = (94 - 100) / 100 = -0.06
        assert mae == pytest.approx(-0.06, abs=0.001)
    
    def test_mae_for_higher_signal_without_drawdown(self):
        """Test MAE for HIGHER signal that never went adverse."""
        # Entry at 100, went straight up
        df = pd.DataFrame({
            "open": [100.0, 102.0, 104.0],
            "high": [102.0, 104.0, 106.0],
            "low": [100.0, 102.0, 104.0],
            "close": [102.0, 104.0, 106.0],
        })
        
        mae = compute_mae(df, "HIGHER", entry_price=100.0)
        
        # Never went below entry → MAE = 0
        assert mae == 0.0
    
    def test_mae_for_lower_signal_with_drawdown(self):
        """Test MAE for LOWER signal that experienced drawdown."""
        # Entry at 100, signal predicts LOWER
        # Price rose to 108 (8% adverse) then dropped
        df = pd.DataFrame({
            "open": [100.0, 102.0, 108.0, 95.0],
            "high": [102.0, 108.0, 109.0, 96.0],
            "low": [99.0, 101.0, 105.0, 94.0],
            "close": [102.0, 107.0, 95.0, 95.0],
        })
        
        mae = compute_mae(df, "LOWER", entry_price=100.0)
        
        # Worst case: high of 109.0
        # MAE = (100 - 109) / 100 = -0.09
        assert mae == pytest.approx(-0.09, abs=0.001)
    
    def test_mae_for_lower_signal_without_drawdown(self):
        """Test MAE for LOWER signal that never went adverse."""
        # Entry at 100, went straight down
        df = pd.DataFrame({
            "open": [100.0, 98.0, 96.0],
            "high": [100.0, 98.0, 96.0],
            "low": [98.0, 96.0, 94.0],
            "close": [98.0, 96.0, 94.0],
        })
        
        mae = compute_mae(df, "LOWER", entry_price=100.0)
        
        # Never went above entry → MAE = 0
        assert mae == 0.0
    
    def test_mae_is_always_non_positive(self):
        """Test that MAE is always <= 0 (convention)."""
        # Various scenarios
        scenarios = [
            (pd.DataFrame({
                "low": [95.0, 90.0], "high": [105.0, 110.0],
                "open": [100.0, 95.0], "close": [95.0, 110.0]
            }), "HIGHER", 100.0),
            (pd.DataFrame({
                "low": [95.0, 90.0], "high": [105.0, 110.0],
                "open": [100.0, 105.0], "close": [105.0, 90.0]
            }), "LOWER", 100.0),
        ]
        
        for df, direction, entry in scenarios:
            mae = compute_mae(df, direction, entry)
            assert mae <= 0.0
    
    def test_mae_raises_on_empty_dataframe(self):
        """Test that empty DataFrame raises ValueError."""
        df = pd.DataFrame()
        
        with pytest.raises(ValueError, match="empty"):
            compute_mae(df, "HIGHER", 100.0)
    
    def test_mae_raises_on_invalid_direction(self):
        """Test that invalid direction raises ValueError."""
        df = pd.DataFrame({
            "low": [95.0], "high": [105.0],
            "open": [100.0], "close": [102.0]
        })
        
        with pytest.raises(ValueError, match="Invalid direction"):
            compute_mae(df, "FLAT", 100.0)


class TestDetermineDirection:
    """Test direction determination from returns."""
    
    def test_positive_return_is_higher(self):
        """Test that positive return gives HIGHER."""
        assert determine_direction(0.05) == "HIGHER"
        assert determine_direction(0.001) == "HIGHER"
    
    def test_negative_return_is_lower(self):
        """Test that negative return gives LOWER."""
        assert determine_direction(-0.05) == "LOWER"
        assert determine_direction(-0.001) == "LOWER"
    
    def test_tiny_return_is_flat(self):
        """Test that very small return is classified as FLAT."""
        # Default threshold is 0.0001 (0.01%)
        assert determine_direction(0.00005) == "FLAT"
        assert determine_direction(-0.00005) == "FLAT"
        assert determine_direction(0.0) == "FLAT"
    
    def test_custom_threshold(self):
        """Test with custom threshold."""
        # 0.5% threshold
        assert determine_direction(0.003, threshold=0.005) == "FLAT"
        assert determine_direction(0.006, threshold=0.005) == "HIGHER"
        assert determine_direction(-0.006, threshold=0.005) == "LOWER"


class TestVerifiedOutcome:
    """Test VerifiedOutcome dataclass."""
    
    def test_create_outcome(self):
        """Test creating a verified outcome."""
        outcome = VerifiedOutcome(
            example_id="test-123",
            actual_direction="HIGHER",
            realized_return=0.05,
            max_adverse_excursion=-0.02,
            net_return=0.048,
            entry_price=100.0,
            exit_price=105.0,
            bars_held=24,
        )
        
        assert outcome.example_id == "test-123"
        assert outcome.actual_direction == "HIGHER"
        assert outcome.realized_return == 0.05
        assert outcome.bars_held == 24
    
    def test_outcome_is_frozen(self):
        """Test that VerifiedOutcome is immutable."""
        outcome = VerifiedOutcome(
            example_id="test",
            actual_direction="HIGHER",
            realized_return=0.05,
            max_adverse_excursion=0.0,
            net_return=0.048,
            entry_price=100.0,
            exit_price=105.0,
            bars_held=24,
        )
        
        # Should not be able to modify
        with pytest.raises(AttributeError):
            outcome.realized_return = 0.10  # type: ignore
