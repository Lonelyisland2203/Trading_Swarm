"""Tests for verifier constants and utilities."""

import pytest

from verifier.constants import HORIZON_BARS, DEFAULT_TXN_COST_PCT, get_horizon_bars, compute_holding_periods_8h


class TestHorizonBars:
    """Test timeframe horizon configuration."""
    
    def test_all_common_timeframes_defined(self):
        """Test that all common trading timeframes have horizons."""
        common_timeframes = ["1m", "5m", "15m", "1h", "4h", "1d"]
        
        for tf in common_timeframes:
            assert tf in HORIZON_BARS, f"Missing horizon for {tf}"
    
    def test_horizons_are_positive(self):
        """Test that all horizons are positive integers."""
        for timeframe, bars in HORIZON_BARS.items():
            assert isinstance(bars, int), f"{timeframe} horizon is not int"
            assert bars > 0, f"{timeframe} horizon must be positive"
    
    def test_shorter_timeframes_have_more_bars(self):
        """Test that shorter timeframes use more bars (to cover similar real time)."""
        # 1m should have more bars than 1h for similar real-time coverage
        assert HORIZON_BARS["1m"] > HORIZON_BARS["1h"]
        assert HORIZON_BARS["5m"] > HORIZON_BARS["4h"]
    
    def test_get_horizon_bars_valid_timeframe(self):
        """Test getting horizon for valid timeframe."""
        assert get_horizon_bars("1h") == 24
        assert get_horizon_bars("15m") == 24
        assert get_horizon_bars("1d") == 5
    
    def test_get_horizon_bars_invalid_timeframe(self):
        """Test that invalid timeframe raises ValueError."""
        with pytest.raises(ValueError, match="Unknown timeframe"):
            get_horizon_bars("invalid")
        
        with pytest.raises(ValueError, match="Unknown timeframe"):
            get_horizon_bars("2h")  # Not in our mapping


class TestTransactionCost:
    """Test transaction cost defaults."""

    def test_default_txn_cost_is_reasonable(self):
        """Test that default cost is in reasonable range (0.05% - 0.5%)."""
        assert 0.0005 <= DEFAULT_TXN_COST_PCT <= 0.005

    def test_default_matches_typical_exchange_fees(self):
        """Test that default is conservative estimate of exchange fees."""
        # 0.1% is conservative covering:
        # - Binance taker: 0.075-0.1%
        # - Kraken taker: 0.16%
        # Should be <= Kraken but >= Binance
        assert DEFAULT_TXN_COST_PCT >= 0.0007  # Above Binance with BNB
        assert DEFAULT_TXN_COST_PCT <= 0.002   # Below Kraken/Coinbase Pro


def test_compute_holding_periods_8h_1m():
    """Test holding period calculation for 1m timeframe."""
    # 60 bars at 1 minute = 60 minutes = 1 hour
    periods = compute_holding_periods_8h("1m", 60)
    assert abs(periods - 0.125) < 1e-9  # 1/8 of a funding period


def test_compute_holding_periods_8h_1h():
    """Test holding period for 1h timeframe."""
    # 24 bars at 1 hour = 24 hours = 3 funding periods
    periods = compute_holding_periods_8h("1h", 24)
    assert abs(periods - 3.0) < 1e-9


def test_compute_holding_periods_8h_1d():
    """Test holding period for 1d timeframe."""
    # 5 bars at 1 day = 5 days = 15 funding periods
    periods = compute_holding_periods_8h("1d", 5)
    assert abs(periods - 15.0) < 1e-9


def test_compute_holding_periods_8h_5m():
    """Test holding period for 5m timeframe."""
    # 96 bars at 5 minutes = 480 minutes = 8 hours = 1 funding period
    periods = compute_holding_periods_8h("5m", 96)
    assert abs(periods - 1.0) < 1e-9


def test_compute_holding_periods_8h_15m():
    """Test holding period for 15m timeframe."""
    # 32 bars at 15 minutes = 480 minutes = 8 hours = 1 funding period
    periods = compute_holding_periods_8h("15m", 32)
    assert abs(periods - 1.0) < 1e-9


def test_compute_holding_periods_8h_4h():
    """Test holding period for 4h timeframe."""
    # 2 bars at 4 hours = 8 hours = 1 funding period
    periods = compute_holding_periods_8h("4h", 2)
    assert abs(periods - 1.0) < 1e-9


def test_compute_holding_periods_8h_invalid_unit():
    """Test that invalid timeframe unit raises ValueError."""
    with pytest.raises(ValueError, match="Unsupported timeframe unit"):
        compute_holding_periods_8h("1w", 1)


def test_compute_holding_periods_8h_zero_bars():
    """Test holding period with zero bars (valid edge case)."""
    periods = compute_holding_periods_8h("1h", 0)
    assert abs(periods - 0.0) < 1e-9
