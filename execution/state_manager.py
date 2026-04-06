"""
State management for the execution layer.

Provides:
- Daily statistics persistence with automatic day rollover
- Append-only order logging in JSONL format
- Kill switch mechanism for emergency trading halt
"""

import json
from datetime import datetime
from pathlib import Path

from loguru import logger

from execution.models import DailyStats


class StateManager:
    """
    Manages execution layer state persistence.

    Handles daily statistics tracking, order history logging, and kill switch
    mechanism for emergency trading halt. All state is persisted to disk for
    crash recovery and debugging.
    """

    def __init__(self, state_dir: Path) -> None:
        """
        Initialize StateManager with state directory.

        Creates state directory if it doesn't exist and sets up file paths for:
        - daily_stats.json: Daily trading statistics
        - order_log.jsonl: Append-only order history
        - STOP: Kill switch file

        Args:
            state_dir: Directory for state file storage
        """
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)

        # Set up file paths
        self.daily_stats_file = self.state_dir / "daily_stats.json"
        self.order_log_file = self.state_dir / "order_log.jsonl"
        self.kill_switch_file = self.state_dir / "STOP"

        logger.debug(f"StateManager initialized with state_dir={state_dir}")

    def is_kill_switch_active(self) -> bool:
        """
        Check if kill switch is active.

        Returns:
            True if STOP file exists, False otherwise
        """
        return self.kill_switch_file.exists()

    def activate_kill_switch(self, reason: str) -> None:
        """
        Activate kill switch by creating STOP file.

        Creates STOP file with timestamp and reason. This signals to the execution
        loop to halt all trading immediately.

        Args:
            reason: Explanation for why kill switch was activated
        """
        data = {
            "activated_at": datetime.now().isoformat(),
            "reason": reason
        }

        with open(self.kill_switch_file, "w") as f:
            json.dump(data, f, indent=2)

        logger.warning(f"Kill switch ACTIVATED: {reason}")

    def deactivate_kill_switch(self) -> None:
        """
        Deactivate kill switch by removing STOP file.

        Safe to call even if STOP file doesn't exist.
        """
        if self.kill_switch_file.exists():
            self.kill_switch_file.unlink()
            logger.warning("Kill switch DEACTIVATED")

    def get_daily_stats(self, starting_balance: float) -> DailyStats:
        """
        Load or create daily statistics.

        Returns existing stats if from today, resets stats if from previous day,
        or creates new stats if none exist.

        Args:
            starting_balance: Starting account balance for today

        Returns:
            DailyStats instance for today
        """
        today = datetime.now().strftime("%Y-%m-%d")

        # Try to load existing stats
        if self.daily_stats_file.exists():
            try:
                with open(self.daily_stats_file) as f:
                    data = json.load(f)

                stats = DailyStats(**data)

                # Check if stats are from today
                if stats.date == today:
                    logger.debug(f"Loaded existing daily stats for {today}")
                    return stats
                else:
                    logger.info(
                        f"Daily stats are from {stats.date}, resetting for {today}"
                    )

            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Failed to load daily stats: {e}, creating new stats")

        # Create new stats for today
        stats = DailyStats(
            date=today,
            starting_balance=starting_balance
        )

        # Persist immediately
        self.update_daily_stats(stats)

        logger.info(f"Created new daily stats for {today}")
        return stats

    def update_daily_stats(self, stats: DailyStats) -> None:
        """
        Persist daily statistics to disk.

        Args:
            stats: DailyStats instance to save
        """
        with open(self.daily_stats_file, "w") as f:
            json.dump(stats.model_dump(mode="json"), f, indent=2)

        logger.debug(
            f"Updated daily stats: {stats.trade_count} trades, "
            f"P&L: {stats.realized_pnl:.2f}"
        )

    def log_order(self, order_data: dict) -> None:
        """
        Append order to order log.

        Adds logged_at timestamp to order data and appends to JSONL file.

        Args:
            order_data: Dictionary containing order details
        """
        # Add timestamp
        log_entry = {
            **order_data,
            "logged_at": datetime.now().isoformat()
        }

        # Append to JSONL file
        with open(self.order_log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        logger.debug(f"Logged order: {order_data.get('order_id', 'unknown')}")

    def get_order_history(self, limit: int = 100) -> list[dict]:
        """
        Read order history from log.

        Returns most recent orders first (reverse chronological order).

        Args:
            limit: Maximum number of orders to return (default: 100)

        Returns:
            List of order dictionaries, most recent first
        """
        if not self.order_log_file.exists():
            return []

        orders = []
        with open(self.order_log_file) as f:
            for line in f:
                try:
                    orders.append(json.loads(line.strip()))
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse order log line: {e}")

        # Return most recent first
        return list(reversed(orders))[:limit]
