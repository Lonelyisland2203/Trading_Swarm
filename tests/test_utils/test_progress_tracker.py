"""Tests for progress tracker with ETA estimation."""

import json
import time
import pytest

from utils.progress_tracker import ProgressTracker, ProgressStats


class TestProgressTracker:
    """Test progress tracker initialization and basic operations."""

    def test_init(self):
        """Test tracker initialization."""
        tracker = ProgressTracker(total_contexts=100, window_size=10)

        assert tracker.total == 100
        assert tracker.completed == 0
        assert tracker.failed == 0
        assert tracker.window_size == 10
        assert len(tracker.recent_times) == 0

    def test_update_success(self):
        """Test updating with successful context."""
        tracker = ProgressTracker(total_contexts=100)

        tracker.update("ctx_1", duration_sec=60.0, success=True)

        assert tracker.completed == 1
        assert tracker.failed == 0
        assert len(tracker.recent_times) == 1
        assert "ctx_1" in tracker.context_times

    def test_update_failure(self):
        """Test updating with failed context."""
        tracker = ProgressTracker(total_contexts=100)

        tracker.update("ctx_1", duration_sec=60.0, success=False)

        assert tracker.completed == 0
        assert tracker.failed == 1
        assert len(tracker.recent_times) == 0
        assert "ctx_1" in tracker.failed_contexts

    def test_update_multiple(self):
        """Test updating with multiple contexts."""
        tracker = ProgressTracker(total_contexts=100)

        for i in range(10):
            tracker.update(f"ctx_{i}", duration_sec=50.0 + i, success=True)

        assert tracker.completed == 10
        assert len(tracker.recent_times) == 10

    def test_rolling_window_size(self):
        """Test rolling window maintains max size."""
        tracker = ProgressTracker(total_contexts=100, window_size=5)

        for i in range(10):
            tracker.update(f"ctx_{i}", duration_sec=float(i), success=True)

        # Should only keep last 5
        assert len(tracker.recent_times) == 5
        assert list(tracker.recent_times) == [5.0, 6.0, 7.0, 8.0, 9.0]


class TestETACalculation:
    """Test ETA estimation."""

    def test_eta_no_data(self):
        """Test ETA returns inf when no data yet."""
        tracker = ProgressTracker(total_contexts=100)

        eta = tracker.get_eta_seconds()
        assert eta == float("inf")

    def test_eta_with_data(self):
        """Test ETA calculation with data."""
        tracker = ProgressTracker(total_contexts=100)

        # Add 10 contexts at 60 seconds each
        for i in range(10):
            tracker.update(f"ctx_{i}", duration_sec=60.0, success=True)

        eta = tracker.get_eta_seconds()

        # Remaining: 90 contexts * 60 sec = 5400 seconds
        assert eta == pytest.approx(5400.0, rel=0.01)

    def test_eta_rolling_average(self):
        """Test ETA uses rolling average."""
        tracker = ProgressTracker(total_contexts=100, window_size=5)

        # First 5 contexts: 50 seconds each
        for i in range(5):
            tracker.update(f"ctx_{i}", duration_sec=50.0, success=True)

        # Next 5 contexts: 70 seconds each (should replace first 5 in window)
        for i in range(5, 10):
            tracker.update(f"ctx_{i}", duration_sec=70.0, success=True)

        eta = tracker.get_eta_seconds()

        # Remaining: 90 contexts * 70 sec (average of last 5)
        assert eta == pytest.approx(6300.0, rel=0.01)

    def test_average_time(self):
        """Test average time calculation."""
        tracker = ProgressTracker(total_contexts=100)

        tracker.update("ctx_1", duration_sec=50.0, success=True)
        tracker.update("ctx_2", duration_sec=70.0, success=True)

        avg = tracker.get_average_time()
        assert avg == pytest.approx(60.0, rel=0.01)

    def test_average_time_no_data(self):
        """Test average time returns 0.0 when no data."""
        tracker = ProgressTracker(total_contexts=100)

        avg = tracker.get_average_time()
        assert avg == 0.0


class TestSuccessRate:
    """Test success rate calculation."""

    def test_success_rate_all_success(self):
        """Test success rate with all successes."""
        tracker = ProgressTracker(total_contexts=100)

        for i in range(10):
            tracker.update(f"ctx_{i}", duration_sec=60.0, success=True)

        assert tracker.get_success_rate() == 1.0

    def test_success_rate_all_failures(self):
        """Test success rate with all failures."""
        tracker = ProgressTracker(total_contexts=100)

        for i in range(10):
            tracker.update(f"ctx_{i}", duration_sec=60.0, success=False)

        assert tracker.get_success_rate() == 0.0

    def test_success_rate_mixed(self):
        """Test success rate with mixed results."""
        tracker = ProgressTracker(total_contexts=100)

        # 7 successes, 3 failures
        for i in range(7):
            tracker.update(f"ctx_{i}", duration_sec=60.0, success=True)
        for i in range(7, 10):
            tracker.update(f"ctx_{i}", duration_sec=60.0, success=False)

        assert tracker.get_success_rate() == pytest.approx(0.7, rel=0.01)

    def test_success_rate_no_data(self):
        """Test success rate returns 0.0 when no data."""
        tracker = ProgressTracker(total_contexts=100)

        assert tracker.get_success_rate() == 0.0


class TestStats:
    """Test statistics gathering."""

    def test_get_stats(self):
        """Test getting comprehensive stats."""
        tracker = ProgressTracker(total_contexts=100)

        # Add some data
        for i in range(10):
            tracker.update(f"ctx_{i}", duration_sec=60.0, success=True)

        stats = tracker.get_stats()

        assert isinstance(stats, ProgressStats)
        assert stats.total == 100
        assert stats.completed == 10
        assert stats.failed == 0
        assert stats.success_rate == 1.0
        assert stats.avg_time_per_context == 60.0
        assert stats.elapsed_seconds > 0

    def test_elapsed_time(self):
        """Test elapsed time tracking."""
        tracker = ProgressTracker(total_contexts=100)

        time.sleep(0.1)  # Small delay

        elapsed = tracker.get_elapsed_seconds()
        assert elapsed >= 0.1
        assert elapsed < 1.0  # Sanity check


class TestPersistence:
    """Test save/load state."""

    def test_save_state(self, tmp_path):
        """Test saving tracker state to JSON."""
        tracker = ProgressTracker(total_contexts=100)

        # Add some data
        for i in range(10):
            tracker.update(f"ctx_{i}", duration_sec=60.0, success=(i < 8))

        state_file = tmp_path / "progress.json"
        tracker.save_state(state_file)

        assert state_file.exists()

        # Verify JSON structure
        with open(state_file) as f:
            state = json.load(f)

        assert state["total"] == 100
        assert state["completed"] == 8
        assert state["failed"] == 2
        assert len(state["recent_times"]) == 8
        assert len(state["failed_contexts"]) == 2

    def test_load_state(self, tmp_path):
        """Test loading tracker state from JSON."""
        # Create and save tracker
        tracker1 = ProgressTracker(total_contexts=100, window_size=5)

        for i in range(10):
            tracker1.update(f"ctx_{i}", duration_sec=50.0 + i, success=True)

        state_file = tmp_path / "progress.json"
        tracker1.save_state(state_file)

        # Load into new tracker
        tracker2 = ProgressTracker.load_state(state_file)

        assert tracker2 is not None
        assert tracker2.total == 100
        assert tracker2.completed == 10
        assert tracker2.failed == 0
        assert len(tracker2.recent_times) == 5  # Window size

    def test_load_nonexistent_file(self, tmp_path):
        """Test loading from nonexistent file returns None."""
        state_file = tmp_path / "nonexistent.json"

        tracker = ProgressTracker.load_state(state_file)
        assert tracker is None

    def test_save_load_roundtrip(self, tmp_path):
        """Test save/load preserves state."""
        tracker1 = ProgressTracker(total_contexts=100)

        # Add varied data
        for i in range(10):
            tracker1.update(
                f"ctx_{i}",
                duration_sec=50.0 + i * 5,
                success=(i % 3 != 0),  # Fail every 3rd
            )

        state_file = tmp_path / "progress.json"
        tracker1.save_state(state_file)

        tracker2 = ProgressTracker.load_state(state_file)

        assert tracker2.total == tracker1.total
        assert tracker2.completed == tracker1.completed
        assert tracker2.failed == tracker1.failed
        assert len(tracker2.recent_times) == len(tracker1.recent_times)


class TestSummary:
    """Test summary generation."""

    def test_get_summary(self):
        """Test getting comprehensive summary."""
        tracker = ProgressTracker(total_contexts=100)

        for i in range(10):
            tracker.update(f"ctx_{i}", duration_sec=60.0, success=True)

        summary = tracker.get_summary()

        assert summary["total_contexts"] == 100
        assert summary["completed"] == 10
        assert summary["failed"] == 0
        assert summary["success_rate"] == 1.0
        assert summary["avg_time_per_context"] == 60.0
        assert summary["contexts_per_hour"] == pytest.approx(60.0, rel=0.01)
        assert "elapsed_minutes" in summary
        assert "eta_minutes" in summary

    def test_summary_with_no_data(self):
        """Test summary with no completed contexts."""
        tracker = ProgressTracker(total_contexts=100)

        summary = tracker.get_summary()

        assert summary["completed"] == 0
        assert summary["contexts_per_hour"] == 0.0
        assert summary["eta_minutes"] is None  # inf ETA
