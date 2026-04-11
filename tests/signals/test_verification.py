"""Tests for signal verification module."""

import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from signals.verification import (
    VerifiedResult,
    VerificationStats,
    get_verification_horizon_bars,
    load_unverified_signals,
    verify_signal,
    save_verified_result,
    load_verified_results,
    compute_verification_stats,
    check_training_trigger,
    mark_training_triggered,
    export_for_training,
    format_daily_summary,
    VERIFIED_RESULTS_PATH,
    TRAINING_TRIGGER_PATH,
    MIN_SIGNALS_FOR_TRAINING,
)
from config.fee_model import FeeModelSettings


@pytest.fixture
def clean_paths(tmp_path, monkeypatch):
    """Use temporary files for testing."""
    signal_log_path = tmp_path / "signal_log.jsonl"
    verified_path = tmp_path / "verified_results.jsonl"
    trigger_path = tmp_path / "training_trigger.json"

    monkeypatch.setattr("signals.verification.SIGNAL_LOG_PATH", signal_log_path)
    monkeypatch.setattr("signals.verification.VERIFIED_RESULTS_PATH", verified_path)
    monkeypatch.setattr("signals.verification.TRAINING_TRIGGER_PATH", trigger_path)
    monkeypatch.setattr("signals.signal_logger.SIGNAL_LOG_PATH", signal_log_path)

    return signal_log_path, verified_path, trigger_path


@pytest.fixture
def synthetic_signals():
    """Create synthetic signal data."""
    now = datetime.now(timezone.utc)
    # Make signals old enough to be past verification horizon
    # 1h timeframe needs 24 hours, 4h timeframe needs 48 hours (12 bars * 4h)
    old_time_1h = now - timedelta(hours=48)  # 48h ago, well past 24h horizon
    old_time_4h = now - timedelta(hours=72)  # 72h ago, well past 48h horizon

    return [
        {
            "timestamp": old_time_1h.isoformat(),
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "direction": "LONG",
            "final_direction": "LONG",
            "confidence": 0.85,
            "market_regime": "neutral",
            "current_price": 42000.0,
            "executed": True,
        },
        {
            "timestamp": (old_time_1h + timedelta(hours=1)).isoformat(),
            "symbol": "ETH/USDT",
            "timeframe": "1h",
            "direction": "SHORT",
            "final_direction": "SHORT",
            "confidence": 0.75,
            "market_regime": "risk_off",
            "current_price": 2500.0,
            "executed": True,
        },
        {
            "timestamp": old_time_4h.isoformat(),  # Needs to be older for 4h horizon
            "symbol": "BTC/USDT",
            "timeframe": "4h",
            "direction": "FLAT",
            "final_direction": "FLAT",
            "confidence": 0.6,
            "market_regime": "neutral",
            "current_price": 42500.0,
            "executed": False,
        },
    ]


@pytest.fixture
def mock_ohlcv_df():
    """Create mock OHLCV DataFrame with upward price movement."""
    return pd.DataFrame({
        "timestamp": [
            1704000000000 + i * 3600000 for i in range(30)
        ],
        "open": [42000.0 + i * 50 for i in range(30)],
        "high": [42100.0 + i * 50 for i in range(30)],
        "low": [41900.0 + i * 50 for i in range(30)],
        "close": [42050.0 + i * 50 for i in range(30)],  # +1.19% over 24 bars
        "volume": [100 + i for i in range(30)],
    })


class TestGetVerificationHorizonBars:
    """Tests for verification horizon lookup."""

    def test_1h_returns_24_bars(self):
        """1h timeframe uses 24 bar horizon (24 hours)."""
        assert get_verification_horizon_bars("1h") == 24

    def test_4h_returns_12_bars(self):
        """4h timeframe uses 12 bar horizon (48 hours)."""
        assert get_verification_horizon_bars("4h") == 12

    def test_1d_returns_5_bars(self):
        """1d timeframe uses 5 bar horizon (5 days)."""
        assert get_verification_horizon_bars("1d") == 5

    def test_unknown_defaults_to_24(self):
        """Unknown timeframe defaults to 24 bars."""
        assert get_verification_horizon_bars("unknown") == 24


class TestLoadUnverifiedSignals:
    """Tests for loading unverified signals."""

    def test_empty_log_returns_empty(self, clean_paths):
        """Empty signal log returns empty list."""
        signals = load_unverified_signals()
        assert signals == []

    def test_loads_signals_past_horizon(self, clean_paths, synthetic_signals):
        """Loads signals that are past verification horizon."""
        signal_log_path, _, _ = clean_paths

        # Write synthetic signals
        with open(signal_log_path, "w") as f:
            for s in synthetic_signals:
                f.write(json.dumps(s) + "\n")

        signals = load_unverified_signals()
        assert len(signals) == 3

    def test_excludes_already_verified(self, clean_paths, synthetic_signals):
        """Excludes signals that already have verified results."""
        signal_log_path, verified_path, _ = clean_paths

        # Write synthetic signals
        with open(signal_log_path, "w") as f:
            for s in synthetic_signals:
                f.write(json.dumps(s) + "\n")

        # Mark first signal as verified
        verified = {
            "signal_timestamp": synthetic_signals[0]["timestamp"],
            "symbol": synthetic_signals[0]["symbol"],
        }
        with open(verified_path, "w") as f:
            f.write(json.dumps(verified) + "\n")

        signals = load_unverified_signals()
        assert len(signals) == 2

    def test_excludes_signals_not_past_horizon(self, clean_paths):
        """Excludes signals that haven't reached verification horizon."""
        signal_log_path, _, _ = clean_paths

        # Create a recent signal (not past horizon)
        recent = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "direction": "LONG",
            "confidence": 0.8,
        }

        with open(signal_log_path, "w") as f:
            f.write(json.dumps(recent) + "\n")

        signals = load_unverified_signals()
        assert len(signals) == 0


class TestVerifySignal:
    """Tests for verify_signal function."""

    @pytest.fixture
    def mock_market_data(self, mock_ohlcv_df):
        """Create mock market data service."""
        mock = AsyncMock()
        mock.fetch_ohlcv = AsyncMock(return_value=mock_ohlcv_df)
        return mock

    @pytest.mark.asyncio
    async def test_correct_long_prediction(self, synthetic_signals, mock_market_data):
        """Verifies correct LONG prediction when price rises."""
        signal = synthetic_signals[0]  # LONG at 42000
        signal["current_price"] = 42000.0

        result = await verify_signal(
            signal=signal,
            market_data_service=mock_market_data,
        )

        assert result is not None
        assert result.predicted_direction == "LONG"
        assert result.entry_price == 42000.0
        assert result.gross_return_pct > 0  # Price went up

    @pytest.mark.asyncio
    async def test_incorrect_short_prediction(self, synthetic_signals, mock_market_data):
        """Verifies incorrect SHORT prediction when price rises."""
        signal = synthetic_signals[1]  # SHORT
        signal["current_price"] = 42000.0

        result = await verify_signal(
            signal=signal,
            market_data_service=mock_market_data,
        )

        assert result is not None
        assert result.predicted_direction == "SHORT"
        # If price went up, SHORT prediction is incorrect
        if result.gross_return_pct > 0:
            assert result.correct is False or result.actual_direction != "SHORT"

    @pytest.mark.asyncio
    async def test_flat_prediction_handling(self, synthetic_signals, mock_market_data):
        """Verifies FLAT prediction handling."""
        signal = synthetic_signals[2]  # FLAT
        signal["current_price"] = 42000.0

        result = await verify_signal(
            signal=signal,
            market_data_service=mock_market_data,
        )

        assert result is not None
        assert result.predicted_direction == "FLAT"

    @pytest.mark.asyncio
    async def test_insufficient_data_returns_none(self, synthetic_signals):
        """Returns None when insufficient OHLCV data available."""
        mock_market = AsyncMock()
        mock_market.fetch_ohlcv = AsyncMock(
            return_value=pd.DataFrame({
                "timestamp": [1704000000000],
                "open": [42000.0],
                "high": [42100.0],
                "low": [41900.0],
                "close": [42050.0],
                "volume": [100],
            })
        )

        result = await verify_signal(
            signal=synthetic_signals[0],
            market_data_service=mock_market,
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_uses_fee_model_for_net_return(self, synthetic_signals, mock_market_data):
        """Verifies fee model is used for net return calculation."""
        signal = synthetic_signals[0]
        signal["current_price"] = 42000.0

        fee_model = FeeModelSettings()
        result = await verify_signal(
            signal=signal,
            market_data_service=mock_market_data,
            fee_model=fee_model,
        )

        assert result is not None
        # Net return should be less than gross return (fees subtracted)
        assert result.net_return_pct < result.gross_return_pct


class TestSaveAndLoadVerifiedResults:
    """Tests for saving and loading verified results."""

    @pytest.fixture
    def sample_result(self):
        """Create sample verified result."""
        return VerifiedResult(
            signal_timestamp=datetime.now(timezone.utc).isoformat(),
            symbol="BTC/USDT",
            timeframe="1h",
            predicted_direction="LONG",
            signal_confidence=0.85,
            market_regime="neutral",
            entry_price=42000.0,
            exit_price=42500.0,
            gross_return_pct=1.19,
            net_return_pct=1.10,
            actual_direction="LONG",
            correct=True,
            verified_at=datetime.now(timezone.utc).isoformat(),
            horizon_bars=24,
        )

    def test_save_creates_file(self, clean_paths, sample_result):
        """Saving creates the verified results file."""
        _, verified_path, _ = clean_paths

        assert not verified_path.exists()
        save_verified_result(sample_result)
        assert verified_path.exists()

    def test_save_appends_to_file(self, clean_paths, sample_result):
        """Saving appends to existing file."""
        _, verified_path, _ = clean_paths

        save_verified_result(sample_result)
        save_verified_result(sample_result)

        lines = verified_path.read_text().strip().split("\n")
        assert len(lines) == 2

    def test_load_returns_all_results(self, clean_paths, sample_result):
        """Loading returns all saved results."""
        save_verified_result(sample_result)
        save_verified_result(sample_result)

        results = load_verified_results()
        assert len(results) == 2

    def test_load_empty_file_returns_empty(self, clean_paths):
        """Loading non-existent file returns empty list."""
        results = load_verified_results()
        assert results == []


class TestComputeVerificationStats:
    """Tests for computing verification statistics."""

    @pytest.fixture
    def mixed_results(self):
        """Create mixed verification results for stats testing."""
        now = datetime.now(timezone.utc)
        return [
            # Correct LONG predictions
            {
                "signal_timestamp": now.isoformat(),
                "symbol": "BTC/USDT",
                "timeframe": "1h",
                "predicted_direction": "LONG",
                "actual_direction": "LONG",
                "correct": True,
                "signal_confidence": 0.9,
                "market_regime": "neutral",
                "net_return_pct": 1.5,
                "gross_return_pct": 1.6,
                "verified_at": now.isoformat(),
            },
            {
                "signal_timestamp": now.isoformat(),
                "symbol": "ETH/USDT",
                "timeframe": "1h",
                "predicted_direction": "LONG",
                "actual_direction": "LONG",
                "correct": True,
                "signal_confidence": 0.8,
                "market_regime": "risk_on",
                "net_return_pct": 0.8,
                "gross_return_pct": 0.9,
                "verified_at": now.isoformat(),
            },
            # Incorrect LONG (false bullish)
            {
                "signal_timestamp": now.isoformat(),
                "symbol": "BTC/USDT",
                "timeframe": "1h",
                "predicted_direction": "LONG",
                "actual_direction": "SHORT",
                "correct": False,
                "signal_confidence": 0.7,
                "market_regime": "neutral",
                "net_return_pct": -0.5,
                "gross_return_pct": -0.4,
                "verified_at": now.isoformat(),
            },
            # Correct SHORT
            {
                "signal_timestamp": now.isoformat(),
                "symbol": "SOL/USDT",
                "timeframe": "1h",
                "predicted_direction": "SHORT",
                "actual_direction": "SHORT",
                "correct": True,
                "signal_confidence": 0.85,
                "market_regime": "risk_off",
                "net_return_pct": 1.2,
                "gross_return_pct": 1.3,
                "verified_at": now.isoformat(),
            },
            # Incorrect SHORT (false bearish)
            {
                "signal_timestamp": now.isoformat(),
                "symbol": "ETH/USDT",
                "timeframe": "1h",
                "predicted_direction": "SHORT",
                "actual_direction": "LONG",
                "correct": False,
                "signal_confidence": 0.65,
                "market_regime": "risk_off",
                "net_return_pct": -0.8,
                "gross_return_pct": -0.7,
                "verified_at": now.isoformat(),
            },
        ]

    def test_empty_results_returns_zeros(self, clean_paths):
        """Empty results returns zero stats."""
        stats = compute_verification_stats([])

        assert stats.total_verified == 0
        assert stats.accuracy_pct == 0.0
        assert stats.ic == 0.0
        assert stats.sharpe_ratio == 0.0

    def test_computes_accuracy_correctly(self, clean_paths, mixed_results):
        """Computes accuracy percentage correctly."""
        stats = compute_verification_stats(mixed_results)

        # 3 correct out of 5
        assert stats.total_verified == 5
        assert stats.correct_count == 3
        assert stats.accuracy_pct == 60.0

    def test_computes_false_signal_rates(self, clean_paths, mixed_results):
        """Computes false bullish and bearish rates."""
        stats = compute_verification_stats(mixed_results)

        # 1 false bullish out of 3 LONG predictions = 33.33%
        assert pytest.approx(stats.false_bullish_rate, rel=0.01) == 33.33

        # 1 false bearish out of 2 SHORT predictions = 50%
        assert stats.false_bearish_rate == 50.0

    def test_computes_regime_stratified_accuracy(self, clean_paths, mixed_results):
        """Computes accuracy by regime."""
        stats = compute_verification_stats(mixed_results)

        assert "neutral" in stats.accuracy_by_regime
        assert "risk_on" in stats.accuracy_by_regime
        assert "risk_off" in stats.accuracy_by_regime

        # neutral: 1 correct out of 2
        assert stats.accuracy_by_regime["neutral"]["accuracy_pct"] == 50.0

        # risk_on: 1 correct out of 1
        assert stats.accuracy_by_regime["risk_on"]["accuracy_pct"] == 100.0

        # risk_off: 1 correct out of 2
        assert stats.accuracy_by_regime["risk_off"]["accuracy_pct"] == 50.0

    def test_computes_net_return_totals(self, clean_paths, mixed_results):
        """Computes total and average net returns."""
        stats = compute_verification_stats(mixed_results)

        # 1.5 + 0.8 - 0.5 + 1.2 - 0.8 = 2.2
        assert pytest.approx(stats.total_net_return_pct, abs=0.01) == 2.2
        assert pytest.approx(stats.avg_net_return_pct, abs=0.01) == 0.44


class TestTrainingTrigger:
    """Tests for training trigger functionality."""

    def test_check_trigger_below_threshold(self, clean_paths):
        """Check trigger returns not ready below threshold."""
        _, verified_path, _ = clean_paths

        # Create 50 verified results (below 200 threshold)
        with open(verified_path, "w") as f:
            for i in range(50):
                f.write(json.dumps({
                    "signal_timestamp": datetime.now(timezone.utc).isoformat(),
                    "symbol": "BTC/USDT",
                    "verified_at": datetime.now(timezone.utc).isoformat(),
                }) + "\n")

        status = check_training_trigger()

        assert status["ready"] is False
        assert status["signals_since_last_training"] == 50

    def test_check_trigger_above_threshold(self, clean_paths):
        """Check trigger returns ready above threshold."""
        _, verified_path, _ = clean_paths

        # Create 250 verified results (above 200 threshold)
        with open(verified_path, "w") as f:
            for i in range(250):
                f.write(json.dumps({
                    "signal_timestamp": datetime.now(timezone.utc).isoformat(),
                    "symbol": "BTC/USDT",
                    "verified_at": datetime.now(timezone.utc).isoformat(),
                }) + "\n")

        status = check_training_trigger()

        assert status["ready"] is True
        assert status["signals_since_last_training"] == 250

    def test_mark_training_triggered_resets_counter(self, clean_paths):
        """Marking training triggered resets the counter."""
        _, verified_path, trigger_path = clean_paths

        # Create 250 verified results
        now = datetime.now(timezone.utc)
        with open(verified_path, "w") as f:
            for i in range(250):
                # All verified before "now"
                ts = (now - timedelta(hours=1)).isoformat()
                f.write(json.dumps({
                    "signal_timestamp": ts,
                    "symbol": "BTC/USDT",
                    "verified_at": ts,
                }) + "\n")

        # Mark training triggered
        mark_training_triggered()

        # Check trigger should now show 0 signals since last training
        status = check_training_trigger()
        assert status["signals_since_last_training"] == 0


class TestExportForTraining:
    """Tests for exporting verified results as training data."""

    def test_export_creates_file(self, clean_paths, tmp_path):
        """Export creates output file."""
        _, verified_path, _ = clean_paths

        # Create verified results
        with open(verified_path, "w") as f:
            f.write(json.dumps({
                "signal_timestamp": datetime.now(timezone.utc).isoformat(),
                "symbol": "BTC/USDT",
                "timeframe": "1h",
                "actual_direction": "LONG",
                "gross_return_pct": 1.5,
                "net_return_pct": 1.4,
                "market_regime": "neutral",
                "signal_confidence": 0.85,
                "verified_at": datetime.now(timezone.utc).isoformat(),
            }) + "\n")

        output_path = tmp_path / "training_data.jsonl"
        count = export_for_training(output_path)

        assert count == 1
        assert output_path.exists()

    def test_export_filters_by_confidence(self, clean_paths, tmp_path):
        """Export filters by minimum confidence."""
        _, verified_path, _ = clean_paths

        # Create results with different confidences
        with open(verified_path, "w") as f:
            for conf in [0.5, 0.6, 0.7, 0.8, 0.9]:
                f.write(json.dumps({
                    "signal_timestamp": datetime.now(timezone.utc).isoformat(),
                    "symbol": "BTC/USDT",
                    "timeframe": "1h",
                    "actual_direction": "LONG",
                    "gross_return_pct": 1.0,
                    "net_return_pct": 0.9,
                    "market_regime": "neutral",
                    "signal_confidence": conf,
                    "verified_at": datetime.now(timezone.utc).isoformat(),
                }) + "\n")

        output_path = tmp_path / "training_data.jsonl"
        count = export_for_training(output_path, min_confidence=0.7)

        assert count == 3  # Only 0.7, 0.8, 0.9

    def test_export_respects_training_trigger(self, clean_paths, tmp_path):
        """Export only includes signals since last training."""
        _, verified_path, trigger_path = clean_paths

        now = datetime.now(timezone.utc)
        old_time = now - timedelta(hours=48)

        # Create old results (before training trigger)
        with open(verified_path, "w") as f:
            for i in range(5):
                f.write(json.dumps({
                    "signal_timestamp": old_time.isoformat(),
                    "symbol": "BTC/USDT",
                    "timeframe": "1h",
                    "actual_direction": "LONG",
                    "gross_return_pct": 1.0,
                    "net_return_pct": 0.9,
                    "market_regime": "neutral",
                    "signal_confidence": 0.8,
                    "verified_at": old_time.isoformat(),
                }) + "\n")

        # Mark training triggered (after the old results)
        trigger_data = {
            "last_training_timestamp": (now - timedelta(hours=24)).isoformat(),
        }
        with open(trigger_path, "w") as f:
            json.dump(trigger_data, f)

        # Add new results (after trigger)
        with open(verified_path, "a") as f:
            for i in range(3):
                f.write(json.dumps({
                    "signal_timestamp": now.isoformat(),
                    "symbol": "ETH/USDT",
                    "timeframe": "1h",
                    "actual_direction": "SHORT",
                    "gross_return_pct": -0.5,
                    "net_return_pct": -0.6,
                    "market_regime": "risk_off",
                    "signal_confidence": 0.75,
                    "verified_at": now.isoformat(),
                }) + "\n")

        output_path = tmp_path / "training_data.jsonl"
        count = export_for_training(output_path)

        # Should only export 3 new results
        assert count == 3


class TestFormatDailySummary:
    """Tests for formatting daily summary."""

    def test_formats_complete_stats(self):
        """Formats complete statistics summary."""
        stats = VerificationStats(
            total_verified=100,
            correct_count=65,
            accuracy_pct=65.0,
            ic=0.15,
            ic_pvalue=0.01,
            sharpe_ratio=1.5,
            accuracy_by_regime={
                "neutral": {"total": 50, "correct": 30, "accuracy_pct": 60.0},
                "risk_off": {"total": 50, "correct": 35, "accuracy_pct": 70.0},
            },
            false_bullish_rate=20.0,
            false_bearish_rate=15.0,
            total_net_return_pct=5.5,
            avg_net_return_pct=0.055,
            ready_for_training=False,
            signals_since_last_training=100,
        )

        summary = format_daily_summary(stats)

        assert "Total Verified: 100" in summary
        assert "65.0%" in summary
        assert "IC (Info Coefficient): 0.1500" in summary
        assert "Sharpe Ratio: 1.50" in summary
        assert "False Bullish: 20.0%" in summary
        assert "False Bearish: 15.0%" in summary
        assert "Ready for Training: NO" in summary

    def test_handles_empty_stats(self):
        """Handles empty statistics gracefully."""
        stats = VerificationStats(
            total_verified=0,
            correct_count=0,
            accuracy_pct=0.0,
            ic=0.0,
            ic_pvalue=1.0,
            sharpe_ratio=0.0,
            accuracy_by_regime={},
            false_bullish_rate=0.0,
            false_bearish_rate=0.0,
            total_net_return_pct=0.0,
            avg_net_return_pct=0.0,
            ready_for_training=False,
            signals_since_last_training=0,
        )

        summary = format_daily_summary(stats)

        assert "Total Verified: 0" in summary
