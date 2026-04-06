"""
End-to-end integration test for realistic fee model.

Tests that the fee model flows correctly through the entire pipeline:
Config → Verifier → DPO Training
"""
import math
import pytest
from config.settings import AppSettings
from config.fee_model import FeeModelSettings
from verifier.constants import compute_holding_periods_8h, get_horizon_bars
from verifier.outcome import apply_fee_model

# Import create_fee_model from both scripts
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from run_dpo_training import create_fee_model as create_fee_model_dpo
from generate_training_dataset import create_fee_model as create_fee_model_dataset


def test_fee_model_in_app_settings():
    """Test fee model is properly integrated into AppSettings."""
    settings = AppSettings()

    assert hasattr(settings, 'fee_model')
    assert isinstance(settings.fee_model, FeeModelSettings)

    # Verify defaults
    assert settings.fee_model.maker_fee_pct == 0.02
    assert settings.fee_model.taker_fee_pct == 0.05
    assert settings.fee_model.bnb_discount_enabled == True


def test_holding_period_calculation_all_timeframes():
    """Test holding period calculation for all supported timeframes."""
    timeframes_and_expected = [
        ("1m", 60, 0.125),   # 60 bars at 1m = 1 hour = 1/8 funding period
        ("5m", 96, 1.0),     # 96 bars at 5m = 8 hours = 1 funding period
        ("15m", 32, 1.0),    # 32 bars at 15m = 8 hours = 1 funding period
        ("1h", 24, 3.0),     # 24 bars at 1h = 24 hours = 3 funding periods
        ("4h", 2, 1.0),      # 2 bars at 4h = 8 hours = 1 funding period
        ("1d", 5, 15.0),     # 5 bars at 1d = 5 days = 15 funding periods
    ]

    for timeframe, horizon_bars, expected_periods in timeframes_and_expected:
        periods = compute_holding_periods_8h(timeframe, horizon_bars)
        assert abs(periods - expected_periods) < 1e-9, \
            f"Failed for {timeframe}: expected {expected_periods}, got {periods}"


def test_apply_fee_model_reduces_returns():
    """Test that apply_fee_model consistently reduces returns."""
    fee_model = FeeModelSettings()

    test_cases = [
        (0.15, 0, True),    # +0.15% gross, 0 funding → should be positive but reduced
        (0.08, 0, False),   # +0.08% gross, 0 funding → should flip negative
        (0.20, 3, True),    # +0.20% gross, 3 funding periods → should be positive
        (-0.10, 0, False),  # -0.10% gross → should be more negative
    ]

    for gross_pct, holding_periods, should_be_positive in test_cases:
        gross_log = math.log(1 + gross_pct / 100)
        net_log = apply_fee_model(gross_log, fee_model, holding_periods)
        net_pct = (math.exp(net_log) - 1) * 100

        # Net should always be less than gross (fees reduce returns)
        assert net_pct < gross_pct, \
            f"Fees should reduce returns: gross={gross_pct}, net={net_pct}"

        if should_be_positive:
            assert net_pct > 0, f"Expected positive net for gross={gross_pct}"
        else:
            assert net_pct < 0, f"Expected negative net for gross={gross_pct}"


def test_fee_model_exact_conversions():
    """Test that fee model uses exact log conversions, not approximations."""
    fee_model = FeeModelSettings()

    # Large return where linear approximation would differ significantly
    gross_pct = 5.0  # +5%
    gross_log = math.log(1 + gross_pct / 100)

    net_log = apply_fee_model(gross_log, fee_model, holding_periods_8h=0)
    net_pct = (math.exp(net_log) - 1) * 100

    # Exact: 5.0 - 0.083 = 4.917%
    expected_exact = 4.917

    # Verify we match exact calculation (tolerance for floating point)
    assert abs(net_pct - expected_exact) < 1e-3, \
        f"Expected exact conversion: {expected_exact}%, got {net_pct}%"


def test_fee_model_env_var_override():
    """Test that environment variables can override fee model defaults."""
    import os

    # Set custom fees
    os.environ["FEE_MAKER_PCT"] = "0.01"
    os.environ["FEE_TAKER_PCT"] = "0.03"
    os.environ["FEE_BNB_DISCOUNT"] = "false"

    settings = AppSettings()

    assert settings.fee_model.maker_fee_pct == 0.01
    assert settings.fee_model.taker_fee_pct == 0.03
    assert settings.fee_model.bnb_discount_enabled == False

    # Cleanup
    del os.environ["FEE_MAKER_PCT"]
    del os.environ["FEE_TAKER_PCT"]
    del os.environ["FEE_BNB_DISCOUNT"]


def test_end_to_end_fee_impact():
    """
    End-to-end test: verify that a signal with +0.10% gross return
    becomes unprofitable after Binance Futures fees.

    This is the core problem the fee model solves - preventing training
    on poisoned labels where signals appear profitable under flat 0.1%
    fees but are actually losses under realistic fees.
    """
    fee_model = FeeModelSettings()  # Binance Futures USDT-M defaults

    # Signal generates +0.10% gross return on 1h timeframe
    gross_pct = 0.10
    gross_log = math.log(1 + gross_pct / 100)

    # Compute holding period for 1h timeframe (24 bars)
    horizon_bars = get_horizon_bars("1h")
    holding_periods_8h = compute_holding_periods_8h("1h", horizon_bars)

    # Apply realistic fees
    net_log = apply_fee_model(gross_log, fee_model, holding_periods_8h)
    net_pct = (math.exp(net_log) - 1) * 100

    # Under flat 0.1% fees, this would be profitable: 0.10% - 0.10% = 0%
    # Under realistic fees, it should be a loss
    assert net_pct < 0, \
        f"Signal with +0.10% gross should be unprofitable after realistic fees, got {net_pct:.3f}%"

    # Verify the loss is meaningful
    assert net_pct < -0.01, \
        f"Expected meaningful loss after fees, got {net_pct:.3f}%"


# ============================================================================
# CLI Fee Mode Tests
# ============================================================================

def test_create_fee_model_futures_usdt():
    """Test create_fee_model with futures_usdt mode (both scripts)."""
    # Test DPO training script
    fm_dpo = create_fee_model_dpo("futures_usdt")
    assert isinstance(fm_dpo, FeeModelSettings)
    assert fm_dpo.maker_fee_pct == 0.02
    assert fm_dpo.taker_fee_pct == 0.05
    assert fm_dpo.bnb_discount_enabled == True
    assert fm_dpo.bnb_discount_pct == 10.0
    assert fm_dpo.funding_rate_pct == 0.01
    assert fm_dpo.include_funding == True

    # Test dataset generation script
    fm_dataset = create_fee_model_dataset("futures_usdt")
    assert isinstance(fm_dataset, FeeModelSettings)
    assert fm_dataset.maker_fee_pct == 0.02
    assert fm_dataset.taker_fee_pct == 0.05


def test_create_fee_model_spot():
    """Test create_fee_model with spot mode (both scripts)."""
    # Test DPO training script
    fm_dpo = create_fee_model_dpo("spot")
    assert isinstance(fm_dpo, FeeModelSettings)
    assert fm_dpo.maker_fee_pct == 0.10
    assert fm_dpo.taker_fee_pct == 0.10
    assert fm_dpo.bnb_discount_enabled == True
    assert fm_dpo.bnb_discount_pct == 25.0  # Spot gets 25% BNB discount
    assert fm_dpo.funding_rate_pct == 0.0
    assert fm_dpo.include_funding == False
    assert fm_dpo.slippage_pct == 0.01

    # Test dataset generation script
    fm_dataset = create_fee_model_dataset("spot")
    assert isinstance(fm_dataset, FeeModelSettings)
    assert fm_dataset.maker_fee_pct == 0.10
    assert fm_dataset.taker_fee_pct == 0.10
    assert fm_dataset.bnb_discount_pct == 25.0


def test_create_fee_model_none():
    """Test create_fee_model with none mode (both scripts)."""
    # Test DPO training script
    fm_dpo = create_fee_model_dpo("none")
    assert fm_dpo is None

    # Test dataset generation script
    fm_dataset = create_fee_model_dataset("none")
    assert fm_dataset is None


def test_create_fee_model_invalid_mode():
    """Test create_fee_model with invalid mode raises error."""
    with pytest.raises(ValueError, match="Invalid fee mode"):
        create_fee_model_dpo("invalid")

    with pytest.raises(ValueError, match="Invalid fee mode"):
        create_fee_model_dataset("invalid")


def test_spot_vs_futures_fee_difference():
    """Test that spot and futures modes have meaningfully different cost structures."""
    fm_spot = create_fee_model_dpo("spot")
    fm_futures = create_fee_model_dpo("futures_usdt")

    # Verify spot doesn't include funding (cost is constant)
    cost_spot_0 = fm_spot.round_trip_cost_pct(0)
    cost_spot_10 = fm_spot.round_trip_cost_pct(10)
    assert cost_spot_0 == cost_spot_10, \
        "Spot cost should not vary with holding period"

    # Verify futures cost increases with holding period
    cost_futures_0 = fm_futures.round_trip_cost_pct(0)
    cost_futures_10 = fm_futures.round_trip_cost_pct(10)
    assert cost_futures_10 > cost_futures_0, \
        "Futures cost should increase with holding period"

    # For very short holding periods (0 funding), futures is cheaper than spot
    # because futures has lower base fees (0.02+0.05=0.07 vs 0.10+0.10=0.20)
    assert cost_futures_0 < cost_spot_0, \
        f"Futures with no funding ({cost_futures_0}%) should be cheaper than Spot ({cost_spot_0}%)"

    # For very long holding periods, futures becomes more expensive due to funding
    cost_futures_100 = fm_futures.round_trip_cost_pct(100)
    assert cost_futures_100 > cost_spot_0, \
        f"Futures with 100 periods ({cost_futures_100}%) should be more expensive than Spot ({cost_spot_0}%)"
