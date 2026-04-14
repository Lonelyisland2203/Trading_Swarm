"""Tests for signal models and direction mapping."""

from datetime import datetime, timezone


from signals.signal_models import (
    Signal,
    SignalLogEntry,
    AccuracyRecord,
    map_generator_to_signal,
    get_timeframe_duration_ms,
    get_verification_horizon_bars,
)


class TestMapGeneratorToSignal:
    """Tests for direction mapping."""

    def test_higher_maps_to_long(self):
        """HIGHER from PREDICT_DIRECTION maps to LONG."""
        assert map_generator_to_signal("HIGHER") == "LONG"
        assert map_generator_to_signal("higher") == "LONG"
        assert map_generator_to_signal("Higher") == "LONG"

    def test_lower_maps_to_short(self):
        """LOWER from PREDICT_DIRECTION maps to SHORT."""
        assert map_generator_to_signal("LOWER") == "SHORT"
        assert map_generator_to_signal("lower") == "SHORT"

    def test_increasing_maps_to_long(self):
        """INCREASING from ASSESS_MOMENTUM maps to LONG."""
        assert map_generator_to_signal("INCREASING") == "LONG"
        assert map_generator_to_signal("increasing") == "LONG"

    def test_decreasing_maps_to_short(self):
        """DECREASING from ASSESS_MOMENTUM maps to SHORT."""
        assert map_generator_to_signal("DECREASING") == "SHORT"

    def test_unknown_maps_to_flat(self):
        """Unknown directions map to FLAT."""
        assert map_generator_to_signal("SIDEWAYS") == "FLAT"
        assert map_generator_to_signal("UNKNOWN") == "FLAT"
        assert map_generator_to_signal("") == "FLAT"

    def test_flat_and_neutral_map_to_flat(self):
        """FLAT and NEUTRAL explicitly map to FLAT."""
        assert map_generator_to_signal("FLAT") == "FLAT"
        assert map_generator_to_signal("NEUTRAL") == "FLAT"


class TestSignal:
    """Tests for Signal dataclass."""

    def test_signal_creation(self):
        """Signal can be created with required fields."""
        signal = Signal(
            symbol="BTC/USDT",
            timeframe="1h",
            direction="LONG",
            confidence=0.85,
            reasoning="Strong momentum",
            persona="MOMENTUM",
            timestamp=datetime.now(timezone.utc),
            market_regime="neutral",
        )

        assert signal.symbol == "BTC/USDT"
        assert signal.direction == "LONG"
        assert signal.confidence == 0.85

    def test_signal_defaults(self):
        """Signal has correct defaults for optional fields."""
        signal = Signal(
            symbol="ETH/USDT",
            timeframe="4h",
            direction="SHORT",
            confidence=0.7,
            reasoning="Mean reversion",
            persona="MEAN_REVERSION",
            timestamp=datetime.now(timezone.utc),
            market_regime="risk_off",
        )

        assert signal.critic_score is None
        assert signal.critic_recommendation is None
        assert signal.critic_override is False
        assert signal.final_direction is None

    def test_signal_to_dict(self):
        """Signal can be converted to dictionary."""
        ts = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        signal = Signal(
            symbol="BTC/USDT",
            timeframe="1h",
            direction="LONG",
            confidence=0.8,
            reasoning="Test",
            persona="MOMENTUM",
            timestamp=ts,
            market_regime="neutral",
        )

        data = signal.to_dict()

        assert data["symbol"] == "BTC/USDT"
        assert data["timestamp"] == "2024-01-15T12:00:00+00:00"
        assert isinstance(data, dict)


class TestSignalLogEntry:
    """Tests for SignalLogEntry dataclass."""

    def test_signal_log_entry_creation(self):
        """SignalLogEntry can be created."""
        entry = SignalLogEntry(
            timestamp="2024-01-15T12:00:00Z",
            symbol="BTC/USDT",
            timeframe="1h",
            direction="LONG",
            confidence=0.85,
            persona="MOMENTUM",
            market_regime="neutral",
            reasoning="Test reasoning",
            current_price=42000.0,
            critic_score=0.75,
            critic_recommendation="ACCEPT",
            critic_override=False,
            final_direction="LONG",
            executed=True,
            trade_decision_reason="Signal passed all checks",
        )

        assert entry.symbol == "BTC/USDT"
        assert entry.executed is True

    def test_signal_log_entry_to_dict(self):
        """SignalLogEntry can be converted to dictionary."""
        entry = SignalLogEntry(
            timestamp="2024-01-15T12:00:00Z",
            symbol="ETH/USDT",
            timeframe="4h",
            direction="SHORT",
            confidence=0.6,
            persona="CONTRARIAN",
            market_regime="risk_on",
            reasoning="Overbought",
            current_price=2500.0,
            critic_score=None,
            critic_recommendation=None,
            critic_override=False,
            final_direction="SHORT",
            executed=False,
            trade_decision_reason="Low confidence",
        )

        data = entry.to_dict()
        assert data["symbol"] == "ETH/USDT"
        assert data["critic_score"] is None


class TestAccuracyRecord:
    """Tests for AccuracyRecord dataclass."""

    def test_accuracy_record_creation(self):
        """AccuracyRecord can be created."""
        record = AccuracyRecord(
            signal_timestamp="2024-01-15T12:00:00Z",
            symbol="BTC/USDT",
            timeframe="1h",
            predicted_direction="LONG",
            actual_direction="LONG",
            correct=True,
            signal_confidence=0.85,
            entry_price=42000.0,
            exit_price=42500.0,
            actual_return_pct=1.19,
            verified_at="2024-01-15T13:00:00Z",
        )

        assert record.correct is True
        assert record.actual_return_pct == 1.19


class TestTimeframeFunctions:
    """Tests for timeframe utility functions."""

    def test_get_timeframe_duration_ms(self):
        """Timeframe durations are correct."""
        assert get_timeframe_duration_ms("1m") == 60_000
        assert get_timeframe_duration_ms("5m") == 300_000
        assert get_timeframe_duration_ms("15m") == 900_000
        assert get_timeframe_duration_ms("1h") == 3_600_000
        assert get_timeframe_duration_ms("4h") == 14_400_000
        assert get_timeframe_duration_ms("1d") == 86_400_000

    def test_get_timeframe_duration_unknown_defaults_to_1h(self):
        """Unknown timeframe defaults to 1h."""
        assert get_timeframe_duration_ms("2h") == 3_600_000
        assert get_timeframe_duration_ms("unknown") == 3_600_000

    def test_get_verification_horizon_bars(self):
        """Verification horizon is 1 bar for all timeframes."""
        assert get_verification_horizon_bars("1m") == 1
        assert get_verification_horizon_bars("1h") == 1
        assert get_verification_horizon_bars("1d") == 1
