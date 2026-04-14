"""Tests for accuracy tracker."""

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pandas as pd
import pytest

from signals.accuracy_tracker import (
    queue_for_verification,
    process_pending_verifications,
    get_accuracy_summary,
    get_recent_accuracy,
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
        reasoning="Test",
        persona="MOMENTUM",
        timestamp=datetime.now(timezone.utc),
        market_regime="neutral",
        final_direction="LONG",
    )


@pytest.fixture
def clean_paths(tmp_path, monkeypatch):
    """Use temporary files for testing."""
    accuracy_path = tmp_path / "accuracy.jsonl"
    pending_path = tmp_path / "pending_verification.jsonl"
    monkeypatch.setattr("signals.accuracy_tracker.ACCURACY_LOG_PATH", accuracy_path)
    monkeypatch.setattr("signals.accuracy_tracker.PENDING_PATH", pending_path)
    return accuracy_path, pending_path


class TestQueueForVerification:
    """Tests for queue_for_verification function."""

    def test_queue_creates_file(self, test_signal, clean_paths):
        """Queuing creates the pending file if it doesn't exist."""
        _, pending_path = clean_paths
        assert not pending_path.exists()

        queue_for_verification(test_signal, entry_price=42000.0)

        assert pending_path.exists()

    def test_queue_appends_record(self, test_signal, clean_paths):
        """Queuing appends a record to the pending file."""
        _, pending_path = clean_paths

        queue_for_verification(test_signal, entry_price=42000.0)
        queue_for_verification(test_signal, entry_price=42500.0)

        lines = pending_path.read_text().strip().split("\n")
        assert len(lines) == 2

    def test_queue_record_contains_required_fields(self, test_signal, clean_paths):
        """Queued record contains all required fields."""
        _, pending_path = clean_paths

        queue_for_verification(test_signal, entry_price=42000.0)

        content = pending_path.read_text().strip()
        data = json.loads(content)

        assert "signal_timestamp" in data
        assert data["symbol"] == "BTC/USDT"
        assert data["timeframe"] == "1h"
        assert data["predicted_direction"] == "LONG"
        assert data["signal_confidence"] == 0.85
        assert data["entry_price"] == 42000.0
        assert "verify_after_ms" in data

    def test_queue_uses_final_direction(self, clean_paths):
        """Queued record uses final_direction (after critic override)."""
        signal = Signal(
            symbol="ETH/USDT",
            timeframe="1h",
            direction="LONG",  # Original
            confidence=0.7,
            reasoning="Test",
            persona="MOMENTUM",
            timestamp=datetime.now(timezone.utc),
            market_regime="neutral",
            final_direction="FLAT",  # After override
        )

        _, pending_path = clean_paths
        queue_for_verification(signal, entry_price=2500.0)

        content = pending_path.read_text().strip()
        data = json.loads(content)

        assert data["predicted_direction"] == "FLAT"


class TestProcessPendingVerifications:
    """Tests for process_pending_verifications function."""

    @pytest.fixture
    def mock_market_data(self):
        """Create mock market data service."""
        mock = AsyncMock()
        mock.fetch_ohlcv = AsyncMock(
            return_value=pd.DataFrame(
                {
                    "timestamp": [1704000000000, 1704003600000],
                    "open": [42000.0, 42100.0],
                    "high": [42200.0, 42300.0],
                    "low": [41900.0, 42000.0],
                    "close": [42100.0, 42200.0],
                    "volume": [100, 150],
                }
            )
        )
        return mock

    @pytest.mark.asyncio
    async def test_process_empty_returns_empty(self, clean_paths, mock_market_data):
        """Processing with no pending returns empty list."""
        records = await process_pending_verifications(mock_market_data)
        assert records == []

    @pytest.mark.asyncio
    async def test_process_not_ready_yet(self, clean_paths, mock_market_data):
        """Pending records not ready for verification are kept."""
        _, pending_path = clean_paths

        # Create a pending record for the future
        future_verify = int(datetime.now(timezone.utc).timestamp() * 1000) + 3600000
        pending = {
            "signal_timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "predicted_direction": "LONG",
            "signal_confidence": 0.85,
            "entry_price": 42000.0,
            "verify_after_ms": future_verify,
        }
        pending_path.write_text(json.dumps(pending) + "\n")

        records = await process_pending_verifications(mock_market_data)

        assert records == []
        # Record should still be in pending
        assert pending_path.exists()
        remaining = pending_path.read_text().strip()
        assert len(remaining) > 0

    @pytest.mark.asyncio
    async def test_process_ready_record(self, clean_paths, mock_market_data):
        """Ready records are verified and logged."""
        accuracy_path, pending_path = clean_paths

        # Create a pending record that's ready (verify_after_ms in the past)
        past_verify = int(datetime.now(timezone.utc).timestamp() * 1000) - 1000
        pending = {
            "signal_timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "predicted_direction": "LONG",
            "signal_confidence": 0.85,
            "entry_price": 42000.0,
            "verify_after_ms": past_verify,
        }
        pending_path.write_text(json.dumps(pending) + "\n")

        records = await process_pending_verifications(mock_market_data)

        # Should return verified records
        assert len(records) == 1
        assert records[0].predicted_direction == "LONG"

        # Accuracy log should contain the record
        assert accuracy_path.exists()

        # Pending should be empty
        remaining = pending_path.read_text().strip()
        assert remaining == ""


class TestGetAccuracySummary:
    """Tests for get_accuracy_summary function."""

    def test_summary_empty_log(self, clean_paths):
        """Summary of empty log returns zeros."""
        summary = get_accuracy_summary()

        assert summary["total"] == 0
        assert summary["correct"] == 0
        assert summary["accuracy_pct"] == 0.0
        assert summary["by_direction"] == {}

    def test_summary_with_records(self, clean_paths):
        """Summary calculates correct statistics."""
        accuracy_path, _ = clean_paths

        # Write some accuracy records
        records = [
            {"predicted_direction": "LONG", "correct": True},
            {"predicted_direction": "LONG", "correct": True},
            {"predicted_direction": "LONG", "correct": False},
            {"predicted_direction": "SHORT", "correct": True},
            {"predicted_direction": "SHORT", "correct": False},
        ]

        with open(accuracy_path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        summary = get_accuracy_summary()

        assert summary["total"] == 5
        assert summary["correct"] == 3
        assert summary["accuracy_pct"] == 60.0

        assert summary["by_direction"]["LONG"]["total"] == 3
        assert summary["by_direction"]["LONG"]["correct"] == 2
        assert summary["by_direction"]["LONG"]["accuracy_pct"] == pytest.approx(66.67, rel=0.01)

        assert summary["by_direction"]["SHORT"]["total"] == 2
        assert summary["by_direction"]["SHORT"]["correct"] == 1
        assert summary["by_direction"]["SHORT"]["accuracy_pct"] == 50.0


class TestGetRecentAccuracy:
    """Tests for get_recent_accuracy function."""

    def test_recent_accuracy_empty_log(self, clean_paths):
        """Recent accuracy of empty log returns zeros."""
        summary = get_recent_accuracy(n=100)

        assert summary["total"] == 0
        assert summary["correct"] == 0
        assert summary["accuracy_pct"] == 0.0

    def test_recent_accuracy_limited(self, clean_paths):
        """Recent accuracy only considers last N records."""
        accuracy_path, _ = clean_paths

        # Write 10 records: first 5 wrong, last 5 correct
        records = []
        for i in range(5):
            records.append({"predicted_direction": "LONG", "correct": False})
        for i in range(5):
            records.append({"predicted_direction": "LONG", "correct": True})

        with open(accuracy_path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        # Get last 5 (all correct)
        summary = get_recent_accuracy(n=5)

        assert summary["total"] == 5
        assert summary["correct"] == 5
        assert summary["accuracy_pct"] == 100.0

        # Get all 10 (50% accuracy)
        summary = get_recent_accuracy(n=100)

        assert summary["total"] == 10
        assert summary["correct"] == 5
        assert summary["accuracy_pct"] == 50.0
