"""
Progress tracking with ETA estimation for long-running jobs.

Tracks completed/failed jobs with rolling average for ETA calculation,
and supports JSON export/import for resume capability.
"""

import json
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from loguru import logger


@dataclass
class ProgressStats:
    """Statistics collected during progress tracking."""

    total: int
    completed: int
    failed: int
    elapsed_seconds: float
    eta_seconds: float
    success_rate: float
    avg_time_per_context: float


class ProgressTracker:
    """
    Track generation progress with moving average ETA.

    Features:
    - Rolling average of last N context processing times
    - ETA calculation (remaining contexts × avg time)
    - Success/failure rate tracking
    - Export statistics to JSON for resume
    """

    def __init__(self, total_contexts: int, window_size: int = 10):
        """
        Initialize progress tracker.

        Args:
            total_contexts: Total number of contexts to process
            window_size: Size of rolling average window
        """
        self.total = total_contexts
        self.completed = 0
        self.failed = 0
        self.window_size = window_size

        # Rolling window of recent processing times (seconds)
        self.recent_times: deque[float] = deque(maxlen=window_size)

        # Context tracking
        self.context_times: dict[str, float] = {}
        self.failed_contexts: set[str] = set()

        # Timing
        self.start_time = time.time()
        self.last_update = self.start_time

    def update(self, context_id: str, duration_sec: float, success: bool) -> None:
        """
        Update tracker with completed context.

        Args:
            context_id: Unique context identifier
            duration_sec: Processing time in seconds
            success: Whether processing succeeded
        """
        if success:
            self.completed += 1
            self.recent_times.append(duration_sec)
            self.context_times[context_id] = duration_sec
        else:
            self.failed += 1
            self.failed_contexts.add(context_id)

        self.last_update = time.time()

    def get_eta_seconds(self) -> float:
        """
        Calculate ETA using rolling average.

        Returns:
            Estimated seconds remaining, or inf if no data yet
        """
        if not self.recent_times:
            return float("inf")

        avg_time = sum(self.recent_times) / len(self.recent_times)
        remaining = self.total - self.completed - self.failed
        return remaining * avg_time

    def get_average_time(self) -> float:
        """
        Get average time per context from rolling window.

        Returns:
            Average seconds per context, or 0.0 if no data
        """
        if not self.recent_times:
            return 0.0

        return sum(self.recent_times) / len(self.recent_times)

    def get_success_rate(self) -> float:
        """
        Calculate success rate.

        Returns:
            Success rate (0.0 to 1.0)
        """
        total_processed = self.completed + self.failed
        if total_processed == 0:
            return 0.0

        return self.completed / total_processed

    def get_elapsed_seconds(self) -> float:
        """Get elapsed time since start."""
        return time.time() - self.start_time

    def get_stats(self) -> ProgressStats:
        """
        Get current statistics.

        Returns:
            ProgressStats with current state
        """
        return ProgressStats(
            total=self.total,
            completed=self.completed,
            failed=self.failed,
            elapsed_seconds=self.get_elapsed_seconds(),
            eta_seconds=self.get_eta_seconds(),
            success_rate=self.get_success_rate(),
            avg_time_per_context=self.get_average_time(),
        )

    def print_status(self, prefix: str = "Progress") -> None:
        """
        Print human-readable progress to console.

        Args:
            prefix: Status line prefix
        """
        stats = self.get_stats()
        eta_min = stats.eta_seconds / 60 if stats.eta_seconds != float("inf") else 0.0
        elapsed_min = stats.elapsed_seconds / 60

        # Build status line
        status = (
            f"{prefix}: {self.completed}/{self.total} contexts "
            f"({self.failed} failed) | "
            f"Elapsed: {elapsed_min:.1f}m | "
            f"ETA: {eta_min:.1f}m | "
            f"Success: {stats.success_rate:.1%}"
        )

        print(status)
        logger.info(
            "Progress update",
            completed=self.completed,
            failed=self.failed,
            total=self.total,
            eta_min=eta_min,
            success_rate=stats.success_rate,
        )

    def save_state(self, path: Path) -> None:
        """
        Save tracker state to JSON for resume.

        Args:
            path: Path to save state JSON
        """
        state = {
            "total": self.total,
            "completed": self.completed,
            "failed": self.failed,
            "window_size": self.window_size,
            "recent_times": list(self.recent_times),
            "context_times": self.context_times,
            "failed_contexts": list(self.failed_contexts),
            "elapsed_seconds": self.get_elapsed_seconds(),
            "stats": asdict(self.get_stats()),
        }

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                json.dump(state, f, indent=2)

            logger.debug("Progress state saved", path=str(path))

        except Exception as e:
            logger.error("Failed to save progress state", error=str(e))

    @classmethod
    def load_state(cls, path: Path) -> Optional["ProgressTracker"]:
        """
        Load tracker state from JSON.

        Args:
            path: Path to state JSON

        Returns:
            Restored ProgressTracker, or None if load fails
        """
        if not path.exists():
            logger.warning("Progress state file not found", path=str(path))
            return None

        try:
            with open(path) as f:
                state = json.load(f)

            # Create tracker
            tracker = cls(
                total_contexts=state["total"],
                window_size=state.get("window_size", 10),
            )

            # Restore state
            tracker.completed = state["completed"]
            tracker.failed = state["failed"]
            tracker.recent_times = deque(
                state.get("recent_times", []),
                maxlen=tracker.window_size,
            )
            tracker.context_times = state.get("context_times", {})
            tracker.failed_contexts = set(state.get("failed_contexts", []))

            # Adjust start time to maintain elapsed time
            elapsed = state.get("elapsed_seconds", 0.0)
            tracker.start_time = time.time() - elapsed

            logger.info(
                "Progress state loaded",
                completed=tracker.completed,
                failed=tracker.failed,
                elapsed_min=elapsed / 60,
            )

            return tracker

        except Exception as e:
            logger.error("Failed to load progress state", error=str(e))
            return None

    def get_summary(self) -> dict:
        """
        Get comprehensive summary for reporting.

        Returns:
            Dictionary with all statistics and timing info
        """
        stats = self.get_stats()

        return {
            "total_contexts": self.total,
            "completed": self.completed,
            "failed": self.failed,
            "success_rate": stats.success_rate,
            "elapsed_seconds": stats.elapsed_seconds,
            "elapsed_minutes": stats.elapsed_seconds / 60,
            "elapsed_hours": stats.elapsed_seconds / 3600,
            "eta_seconds": stats.eta_seconds,
            "eta_minutes": stats.eta_seconds / 60 if stats.eta_seconds != float("inf") else None,
            "avg_time_per_context": stats.avg_time_per_context,
            "contexts_per_hour": (
                3600 / stats.avg_time_per_context if stats.avg_time_per_context > 0 else 0.0
            ),
        }
