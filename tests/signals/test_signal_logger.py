"""Tests for signal logger."""

import json
from datetime import datetime, timezone, timedelta
from threading import Thread

import pytest

from signals.signal_logger import (
    log_signal,
    read_signal_log,
    get_signal_count,
    get_signals_since,
)
from signals.signal_models import Signal


@pytest.fixture
def test_signal():
    """Create a test signal."""
    return Signal(
        symbol="BTC/USDT",
        timeframe="1h",
        direction="LONG",
        confidence=0.85,
        reasoning="Strong momentum indicators",
        persona="MOMENTUM",
        timestamp=datetime.now(timezone.utc),
        market_regime="neutral",
        current_price=42000.0,
        rsi=55.0,
        macd=0.05,
        macd_signal=0.03,
        bb_position=0.65,
        critic_score=0.75,
        critic_recommendation="ACCEPT",
        critic_override=False,
        final_direction="LONG",
    )


@pytest.fixture
def clean_log_file(tmp_path, monkeypatch):
    """Use a temporary log file."""
    log_path = tmp_path / "signal_log.jsonl"
    monkeypatch.setattr("signals.signal_logger.SIGNAL_LOG_PATH", log_path)
    return log_path


class TestLogSignal:
    """Tests for log_signal function."""

    def test_log_signal_creates_file(self, test_signal, clean_log_file):
        """Logging creates the file if it doesn't exist."""
        assert not clean_log_file.exists()

        log_signal(test_signal, executed=False, trade_reason="Test")

        assert clean_log_file.exists()

    def test_log_signal_appends_jsonl(self, test_signal, clean_log_file):
        """Each log entry is appended as a JSON line."""
        log_signal(test_signal, executed=False, trade_reason="First")
        log_signal(test_signal, executed=True, trade_reason="Second")

        lines = clean_log_file.read_text().strip().split("\n")
        assert len(lines) == 2

        # Each line should be valid JSON
        for line in lines:
            data = json.loads(line)
            assert data["symbol"] == "BTC/USDT"

    def test_log_signal_includes_all_fields(self, test_signal, clean_log_file):
        """Log entry includes all expected fields."""
        log_signal(test_signal, executed=True, trade_reason="All checks passed")

        content = clean_log_file.read_text().strip()
        data = json.loads(content)

        assert data["symbol"] == "BTC/USDT"
        assert data["timeframe"] == "1h"
        assert data["direction"] == "LONG"
        assert data["confidence"] == 0.85
        assert data["persona"] == "MOMENTUM"
        assert data["market_regime"] == "neutral"
        assert data["current_price"] == 42000.0
        assert data["critic_score"] == 0.75
        assert data["critic_recommendation"] == "ACCEPT"
        assert data["critic_override"] is False
        assert data["final_direction"] == "LONG"
        assert data["executed"] is True
        assert data["trade_decision_reason"] == "All checks passed"

    def test_log_signal_handles_none_critic_fields(self, clean_log_file):
        """Log entry handles None critic fields."""
        signal = Signal(
            symbol="ETH/USDT",
            timeframe="4h",
            direction="SHORT",
            confidence=0.6,
            reasoning="Test",
            persona="CONTRARIAN",
            timestamp=datetime.now(timezone.utc),
            market_regime="risk_off",
        )

        log_signal(signal, executed=False, trade_reason="No critic")

        content = clean_log_file.read_text().strip()
        data = json.loads(content)

        assert data["critic_score"] is None
        assert data["critic_recommendation"] is None

    def test_log_signal_thread_safe(self, test_signal, clean_log_file):
        """Multiple threads can log concurrently without corruption."""
        num_threads = 10
        logs_per_thread = 5
        threads = []

        def log_multiple():
            for i in range(logs_per_thread):
                log_signal(test_signal, executed=False, trade_reason=f"Thread log {i}")

        for _ in range(num_threads):
            t = Thread(target=log_multiple)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Verify all logs were written
        lines = clean_log_file.read_text().strip().split("\n")
        assert len(lines) == num_threads * logs_per_thread

        # Verify each line is valid JSON
        for line in lines:
            data = json.loads(line)
            assert "symbol" in data


class TestReadSignalLog:
    """Tests for read_signal_log function."""

    def test_read_empty_log(self, clean_log_file):
        """Reading non-existent log returns empty list."""
        entries = read_signal_log()
        assert entries == []

    def test_read_log_entries(self, test_signal, clean_log_file):
        """Reading log returns all entries."""
        log_signal(test_signal, executed=False, trade_reason="First")
        log_signal(test_signal, executed=True, trade_reason="Second")

        entries = read_signal_log()

        assert len(entries) == 2
        # Most recent first
        assert entries[0]["trade_decision_reason"] == "Second"
        assert entries[1]["trade_decision_reason"] == "First"

    def test_read_log_with_limit(self, test_signal, clean_log_file):
        """Reading log with limit returns limited entries."""
        for i in range(5):
            log_signal(test_signal, executed=False, trade_reason=f"Entry {i}")

        entries = read_signal_log(limit=3)

        assert len(entries) == 3


class TestGetSignalCount:
    """Tests for get_signal_count function."""

    def test_count_empty_log(self, clean_log_file):
        """Count of non-existent log is 0."""
        count = get_signal_count()
        assert count == 0

    def test_count_entries(self, test_signal, clean_log_file):
        """Count returns correct number of entries."""
        log_signal(test_signal, executed=False, trade_reason="1")
        log_signal(test_signal, executed=False, trade_reason="2")
        log_signal(test_signal, executed=False, trade_reason="3")

        count = get_signal_count()
        assert count == 3


class TestGetSignalsSince:
    """Tests for get_signals_since function."""

    def test_signals_since_filters_correctly(self, clean_log_file):
        """Only returns signals since the specified time."""
        now = datetime.now(timezone.utc)
        old_time = now - timedelta(hours=2)
        recent_time = now - timedelta(minutes=30)

        # Create signals at different times
        old_signal = Signal(
            symbol="BTC/USDT",
            timeframe="1h",
            direction="LONG",
            confidence=0.7,
            reasoning="Old",
            persona="MOMENTUM",
            timestamp=old_time,
            market_regime="neutral",
        )

        recent_signal = Signal(
            symbol="ETH/USDT",
            timeframe="1h",
            direction="SHORT",
            confidence=0.8,
            reasoning="Recent",
            persona="CONTRARIAN",
            timestamp=recent_time,
            market_regime="neutral",
        )

        log_signal(old_signal, executed=False, trade_reason="Old")
        log_signal(recent_signal, executed=False, trade_reason="Recent")

        # Get signals from the last hour
        cutoff = now - timedelta(hours=1)
        entries = get_signals_since(cutoff)

        assert len(entries) == 1
        assert entries[0]["symbol"] == "ETH/USDT"
