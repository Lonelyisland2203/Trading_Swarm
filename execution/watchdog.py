"""
Independent watchdog process for execution monitoring.

COMPLETELY INDEPENDENT from signal_loop and LangGraph:
- Not imported by signal_loop
- Runs as separate process via systemd/supervisor
- Polls positions every 30s
- Enforces safety limits independently

Features:
- Max 2% daily loss → flatten all positions
- Position age > 48h → alert
- STOP file → immediate flatten and exit
- Heartbeat to dashboard/health_status.json

CLI:
    python execution/watchdog.py --exchange hyperliquid [--dry-run]
"""

import argparse
import asyncio
import json
import signal
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from loguru import logger

from execution.models import Position


class OrphanPositionDetector:
    """Detect positions with no corresponding signal in log."""

    def __init__(
        self,
        signal_log_path: Path,
        exchange_client,
    ) -> None:
        """
        Initialize OrphanPositionDetector.

        Args:
            signal_log_path: Path to signal_log.jsonl
            exchange_client: Exchange client (router or adapter)
        """
        self._signal_log_path = signal_log_path
        self._exchange_client = exchange_client

    async def detect(self) -> List[Position]:
        """
        Detect orphan positions.

        Returns:
            List of positions with no matching signal
        """
        positions = await self._exchange_client.get_positions()
        if not positions:
            return []

        # Load recent signals
        signaled_symbols = self._load_recent_signals()

        orphans = []
        for pos in positions:
            if pos.symbol not in signaled_symbols:
                orphans.append(pos)

        return orphans

    def _load_recent_signals(self) -> set:
        """Load symbols from recent signal log entries."""
        symbols = set()

        if not self._signal_log_path.exists():
            return symbols

        try:
            with open(self._signal_log_path) as f:
                for line in f:
                    if line.strip():
                        try:
                            entry = json.loads(line)
                            if entry.get("executed"):
                                symbols.add(entry.get("symbol", ""))
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            logger.warning(f"Error loading signal log: {e}")

        return symbols

    async def detect_and_alert(self) -> None:
        """Detect orphans and log alerts."""
        orphans = await self.detect()
        for pos in orphans:
            logger.warning(f"ORPHAN POSITION DETECTED: {pos.symbol} ({pos.side} {pos.amount})")


class DailyLossBreaker:
    """Flatten when daily loss exceeds threshold."""

    def __init__(
        self,
        exchange_client,
        max_daily_loss_pct: float = 2.0,
        starting_balance: float = 0.0,
        dry_run: bool = False,
    ) -> None:
        """
        Initialize DailyLossBreaker.

        Args:
            exchange_client: Exchange client
            max_daily_loss_pct: Maximum daily loss percentage
            starting_balance: Starting balance for the day
            dry_run: If True, detect but don't flatten
        """
        self._exchange_client = exchange_client
        self._max_loss_pct = max_daily_loss_pct
        self._starting_balance = starting_balance
        self._dry_run = dry_run

    async def check_and_flatten(self) -> bool:
        """
        Check daily loss and flatten if exceeded.

        Returns:
            True if circuit breaker was triggered
        """
        balance = await self._exchange_client.get_balance()
        current = balance.get("total", 0.0)

        if self._starting_balance <= 0:
            return False

        loss_pct = (self._starting_balance - current) / self._starting_balance * 100

        if loss_pct >= self._max_loss_pct:
            logger.critical(
                f"DAILY LOSS BREAKER TRIGGERED: {loss_pct:.2f}% (limit: {self._max_loss_pct:.2f}%)"
            )

            if not self._dry_run:
                await self._exchange_client.flatten_all()
                logger.info("All positions flattened")

            return True

        return False


class HealthMonitor:
    """Write heartbeat to health_status.json."""

    def __init__(
        self,
        exchange_client,
        health_file: Path,
        starting_balance: float = 0.0,
    ) -> None:
        """
        Initialize HealthMonitor.

        Args:
            exchange_client: Exchange client
            health_file: Path to health_status.json
            starting_balance: Starting balance for PnL calculation
        """
        self._exchange_client = exchange_client
        self._health_file = health_file
        self._starting_balance = starting_balance

    async def update(self) -> None:
        """Update health status file."""
        try:
            positions = await self._exchange_client.get_positions()
            balance = await self._exchange_client.get_balance()

            current = balance.get("total", 0.0)
            daily_pnl = current - self._starting_balance

            status = {
                "last_seen": datetime.now().isoformat(),
                "positions_count": len(positions),
                "daily_pnl": daily_pnl,
                "current_balance": current,
                "status": "healthy",
            }

            # Ensure directory exists
            self._health_file.parent.mkdir(parents=True, exist_ok=True)

            with open(self._health_file, "w") as f:
                json.dump(status, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to update health status: {e}")
            # Write error status
            try:
                status = {
                    "last_seen": datetime.now().isoformat(),
                    "status": "error",
                    "error": str(e),
                }
                with open(self._health_file, "w") as f:
                    json.dump(status, f, indent=2)
            except Exception:
                pass


class Watchdog:
    """
    Main watchdog class.

    Coordinates all monitoring functions:
    - STOP file detection
    - Daily loss breaker
    - Orphan position detection
    - Position age monitoring
    - Health heartbeat
    """

    POLL_INTERVAL = 30  # seconds

    def __init__(
        self,
        exchange_client,
        state_dir: str,
        dashboard_dir: str,
        signal_log_path: Path,
        position_tracker_file: Optional[Path] = None,
        max_position_age_hours: int = 48,
        max_daily_loss_pct: float = 2.0,
        dry_run: bool = False,
    ) -> None:
        """
        Initialize Watchdog.

        Args:
            exchange_client: Exchange client (router or adapter)
            state_dir: Directory for state files (STOP file location)
            dashboard_dir: Directory for dashboard files
            signal_log_path: Path to signal log
            position_tracker_file: Path to position age tracker
            max_position_age_hours: Maximum position age before alert
            max_daily_loss_pct: Maximum daily loss percentage
            dry_run: If True, detect but don't act
        """
        self._exchange_client = exchange_client
        self._state_dir = Path(state_dir)
        self._dashboard_dir = Path(dashboard_dir)
        self._signal_log_path = signal_log_path
        self._max_position_age_hours = max_position_age_hours
        self._dry_run = dry_run

        # Position tracker
        self._position_tracker_file = position_tracker_file or (
            self._state_dir / "position_tracker.json"
        )

        # Get starting balance
        self._starting_balance = 0.0

        # Initialize components
        self._orphan_detector = OrphanPositionDetector(
            signal_log_path=signal_log_path,
            exchange_client=exchange_client,
        )

        self._loss_breaker = DailyLossBreaker(
            exchange_client=exchange_client,
            max_daily_loss_pct=max_daily_loss_pct,
            starting_balance=0.0,  # Will be set on start
            dry_run=dry_run,
        )

        self._health_monitor = HealthMonitor(
            exchange_client=exchange_client,
            health_file=self._dashboard_dir / "health_status.json",
            starting_balance=0.0,  # Will be set on start
        )

        self._running = False

    async def _initialize_balance(self) -> None:
        """Initialize starting balance."""
        try:
            balance = await self._exchange_client.get_balance()
            self._starting_balance = balance.get("total", 0.0)
            self._loss_breaker._starting_balance = self._starting_balance
            self._health_monitor._starting_balance = self._starting_balance
            logger.info(f"Starting balance: {self._starting_balance:.2f}")
        except Exception as e:
            logger.error(f"Failed to get starting balance: {e}")

    async def check_stop_file(self) -> bool:
        """
        Check for STOP file and flatten if present.

        Returns:
            True if STOP file found (should exit)
        """
        stop_file = self._state_dir / "STOP"

        if stop_file.exists():
            logger.critical("STOP file detected - flattening all positions")

            if not self._dry_run:
                await self._exchange_client.flatten_all()

            return True

        return False

    async def check_position_ages(self) -> None:
        """Check and alert on old positions."""
        # Load position tracker
        tracker = {}
        if self._position_tracker_file.exists():
            try:
                with open(self._position_tracker_file) as f:
                    tracker = json.load(f)
            except Exception:
                tracker = {}

        positions = await self._exchange_client.get_positions()
        now = datetime.now()

        for pos in positions:
            symbol = pos.symbol

            if symbol not in tracker:
                # First time seeing this position
                tracker[symbol] = {
                    "first_seen": now.isoformat(),
                }
            else:
                first_seen = datetime.fromisoformat(tracker[symbol]["first_seen"])
                age_hours = (now - first_seen).total_seconds() / 3600

                if age_hours > self._max_position_age_hours:
                    logger.warning(
                        f"POSITION AGE ALERT: {symbol} open for {age_hours:.1f}h "
                        f"(limit: {self._max_position_age_hours}h)"
                    )

        # Remove closed positions from tracker
        open_symbols = {pos.symbol for pos in positions}
        tracker = {k: v for k, v in tracker.items() if k in open_symbols}

        # Save tracker
        try:
            with open(self._position_tracker_file, "w") as f:
                json.dump(tracker, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save position tracker: {e}")

    async def run_once(self) -> bool:
        """
        Run one watchdog cycle.

        Returns:
            True if should exit (STOP file or critical error)
        """
        # Check STOP file first
        if await self.check_stop_file():
            return True

        # Check daily loss
        if await self._loss_breaker.check_and_flatten():
            return True  # Exit after flatten

        # Check orphan positions
        await self._orphan_detector.detect_and_alert()

        # Check position ages
        await self.check_position_ages()

        # Update health status
        await self._health_monitor.update()

        return False

    async def run(self) -> None:
        """Run watchdog loop until stopped."""
        self._running = True

        # Initialize
        await self._initialize_balance()

        logger.info(f"Watchdog started (poll_interval={self.POLL_INTERVAL}s)")

        while self._running:
            try:
                should_exit = await self.run_once()
                if should_exit:
                    logger.info("Watchdog exiting")
                    break

            except Exception as e:
                logger.error(f"Watchdog error: {e}")

            await asyncio.sleep(self.POLL_INTERVAL)

    def stop(self) -> None:
        """Stop the watchdog loop."""
        self._running = False


def run_watchdog(
    exchange: str = "hyperliquid",
    private_key: Optional[str] = None,
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None,
    testnet: bool = True,
    dry_run: bool = False,
    state_dir: str = "execution",
    dashboard_dir: str = "dashboard",
    signal_log: str = "signals/signal_log.jsonl",
) -> None:
    """
    Main entry point for watchdog.

    Args:
        exchange: Exchange to use
        private_key: Hyperliquid private key
        api_key: Binance API key
        api_secret: Binance API secret
        testnet: Use testnet
        dry_run: Detect but don't act
        state_dir: State directory
        dashboard_dir: Dashboard directory
        signal_log: Signal log path
    """
    import os

    os.environ["EXCHANGE"] = exchange

    from execution.exchange_router import ExchangeRouter

    router = ExchangeRouter(
        private_key=private_key,
        api_key=api_key,
        api_secret=api_secret,
        testnet=testnet,
        state_dir=state_dir,
    )

    watchdog = Watchdog(
        exchange_client=router,
        state_dir=state_dir,
        dashboard_dir=dashboard_dir,
        signal_log_path=Path(signal_log),
        dry_run=dry_run,
    )

    # Handle signals
    def signal_handler(sig, frame):
        logger.info("Received shutdown signal")
        watchdog.stop()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Run
    asyncio.run(watchdog.run())


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Independent execution watchdog process")
    parser.add_argument(
        "--exchange",
        choices=["hyperliquid", "binance"],
        default="hyperliquid",
        help="Exchange to use",
    )
    parser.add_argument(
        "--private-key",
        help="Hyperliquid private key (or HYPERLIQUID_PRIVATE_KEY env var)",
    )
    parser.add_argument(
        "--api-key",
        help="Binance API key (or BINANCE_API_KEY env var)",
    )
    parser.add_argument(
        "--api-secret",
        help="Binance API secret (or BINANCE_API_SECRET env var)",
    )
    parser.add_argument(
        "--testnet",
        action="store_true",
        default=True,
        help="Use testnet (default: True)",
    )
    parser.add_argument(
        "--mainnet",
        action="store_true",
        help="Use mainnet",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Detect issues but don't act",
    )
    parser.add_argument(
        "--state-dir",
        default="execution",
        help="State directory",
    )
    parser.add_argument(
        "--dashboard-dir",
        default="dashboard",
        help="Dashboard directory",
    )
    parser.add_argument(
        "--signal-log",
        default="signals/signal_log.jsonl",
        help="Signal log path",
    )

    args = parser.parse_args()

    # Get credentials from env if not provided
    import os

    private_key = args.private_key or os.environ.get("HYPERLIQUID_PRIVATE_KEY")
    api_key = args.api_key or os.environ.get("BINANCE_API_KEY")
    api_secret = args.api_secret or os.environ.get("BINANCE_API_SECRET")

    testnet = not args.mainnet

    run_watchdog(
        exchange=args.exchange,
        private_key=private_key,
        api_key=api_key,
        api_secret=api_secret,
        testnet=testnet,
        dry_run=args.dry_run,
        state_dir=args.state_dir,
        dashboard_dir=args.dashboard_dir,
        signal_log=args.signal_log,
    )


if __name__ == "__main__":
    main()
