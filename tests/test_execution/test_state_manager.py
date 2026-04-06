"""
Tests for StateManager in execution.state_manager.

Tests daily statistics persistence, order logging, and kill switch mechanism.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from execution.models import DailyStats
from execution.state_manager import StateManager


@pytest.fixture
def state_dir():
    """Create a temporary directory for state files."""
    with TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def state_manager(state_dir):
    """Create StateManager with temporary directory."""
    return StateManager(state_dir)


class TestStateManager:
    """Test suite for StateManager."""

    def test_init_creates_directory(self):
        """Test that StateManager creates state directory if missing."""
        with TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "nested" / "state"
            assert not state_path.exists()

            manager = StateManager(state_path)
            assert state_path.exists()
            assert state_path.is_dir()

    def test_check_kill_switch_no_file(self, state_manager):
        """Test that kill switch is inactive when no STOP file exists."""
        assert not state_manager.is_kill_switch_active()

    def test_check_kill_switch_with_file(self, state_manager):
        """Test that kill switch is active when STOP file is present."""
        state_manager.activate_kill_switch("Test reason")
        assert state_manager.is_kill_switch_active()

    def test_get_daily_stats_creates_new(self, state_manager):
        """Test that get_daily_stats creates new stats if none exist."""
        stats = state_manager.get_daily_stats(starting_balance=10000.0)

        assert stats.date == datetime.now().strftime("%Y-%m-%d")
        assert stats.trade_count == 0
        assert stats.realized_pnl == 0.0
        assert stats.starting_balance == 10000.0
        assert stats.last_order_timestamp is None

    def test_get_daily_stats_loads_existing(self, state_manager):
        """Test that get_daily_stats loads existing stats for today."""
        # Create and update stats
        stats = state_manager.get_daily_stats(starting_balance=10000.0)
        stats.trade_count = 5
        stats.realized_pnl = 150.0
        state_manager.update_daily_stats(stats)

        # Load stats again
        loaded_stats = state_manager.get_daily_stats(starting_balance=10000.0)
        assert loaded_stats.trade_count == 5
        assert loaded_stats.realized_pnl == 150.0
        assert loaded_stats.starting_balance == 10000.0

    def test_get_daily_stats_resets_for_new_day(self, state_manager, state_dir):
        """Test that get_daily_stats resets stats when date changes."""
        # Create stats for yesterday
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        old_stats = DailyStats(
            date=yesterday,
            trade_count=10,
            realized_pnl=-200.0,
            starting_balance=10000.0
        )

        # Manually write yesterday's stats
        stats_file = state_dir / "daily_stats.json"
        with open(stats_file, "w") as f:
            json.dump(old_stats.model_dump(), f)

        # Load stats - should reset for today
        new_stats = state_manager.get_daily_stats(starting_balance=12000.0)
        assert new_stats.date == datetime.now().strftime("%Y-%m-%d")
        assert new_stats.trade_count == 0
        assert new_stats.realized_pnl == 0.0
        assert new_stats.starting_balance == 12000.0

    def test_update_daily_stats(self, state_manager, state_dir):
        """Test that update_daily_stats persists stats correctly."""
        stats = state_manager.get_daily_stats(starting_balance=10000.0)
        stats.trade_count = 3
        stats.realized_pnl = 75.5
        stats.last_order_timestamp = datetime.now()

        state_manager.update_daily_stats(stats)

        # Verify file was written
        stats_file = state_dir / "daily_stats.json"
        assert stats_file.exists()

        # Load and verify contents
        with open(stats_file) as f:
            data = json.load(f)

        assert data["trade_count"] == 3
        assert data["realized_pnl"] == 75.5
        assert data["starting_balance"] == 10000.0

    def test_log_order(self, state_manager, state_dir):
        """Test that log_order appends order to log file."""
        order_data = {
            "order_id": "test123",
            "symbol": "BTCUSDT",
            "side": "buy",
            "amount": 0.001,
            "price": 50000.0
        }

        state_manager.log_order(order_data)

        # Verify file was created
        log_file = state_dir / "order_log.jsonl"
        assert log_file.exists()

        # Verify contents
        with open(log_file) as f:
            logged = json.loads(f.readline())

        assert logged["order_id"] == "test123"
        assert logged["symbol"] == "BTCUSDT"
        assert "logged_at" in logged

    def test_log_order_appends(self, state_manager, state_dir):
        """Test that multiple orders are appended to log."""
        order1 = {"order_id": "order1", "symbol": "BTCUSDT"}
        order2 = {"order_id": "order2", "symbol": "ETHUSDT"}

        state_manager.log_order(order1)
        state_manager.log_order(order2)

        # Verify both orders are in log
        log_file = state_dir / "order_log.jsonl"
        with open(log_file) as f:
            lines = f.readlines()

        assert len(lines) == 2
        assert json.loads(lines[0])["order_id"] == "order1"
        assert json.loads(lines[1])["order_id"] == "order2"

    def test_get_order_history(self, state_manager):
        """Test that get_order_history returns orders in reverse chronological order."""
        # Log multiple orders
        for i in range(5):
            state_manager.log_order({"order_id": f"order{i}", "symbol": "BTCUSDT"})

        # Get history
        history = state_manager.get_order_history(limit=3)

        assert len(history) == 3
        # Most recent first (reverse order)
        assert history[0]["order_id"] == "order4"
        assert history[1]["order_id"] == "order3"
        assert history[2]["order_id"] == "order2"

    def test_activate_kill_switch(self, state_manager, state_dir):
        """Test that activate_kill_switch creates STOP file with reason."""
        state_manager.activate_kill_switch("Daily loss limit exceeded")

        stop_file = state_dir / "STOP"
        assert stop_file.exists()

        # Verify contents
        with open(stop_file) as f:
            data = json.load(f)

        assert data["reason"] == "Daily loss limit exceeded"
        assert "activated_at" in data

    def test_deactivate_kill_switch(self, state_manager):
        """Test that deactivate_kill_switch removes STOP file."""
        # Activate first
        state_manager.activate_kill_switch("Test")
        assert state_manager.is_kill_switch_active()

        # Deactivate
        state_manager.deactivate_kill_switch()
        assert not state_manager.is_kill_switch_active()
