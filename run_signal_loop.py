#!/usr/bin/env python3
"""
Production signal loop CLI.

Runs the signal generation loop for live trading:
1. Fetches market data
2. Computes indicators
3. Generates signals using qwen3:8b with GRPO adapter
4. Validates signals using deepseek-r1:14b critic
5. Optionally executes trades via Binance

Usage:
    # Dry run (no execution) with default symbols
    python run_signal_loop.py --dry-run

    # Single cycle test with specific symbol
    python run_signal_loop.py --once --symbols BTC/USDT --timeframe 1h

    # Production mode with execution (requires ALLOW_LIVE_TRADING=true)
    ALLOW_LIVE_TRADING=true python run_signal_loop.py --execute

Safety:
    - Kill switch: Create 'execution/state/STOP' file to halt trading
    - Testnet by default
    - --execute requires ALLOW_LIVE_TRADING=true environment variable
"""

import argparse
import asyncio
import os
import sys
from datetime import datetime, timezone

from loguru import logger

from config.settings import settings
from signals.signal_loop import run_loop
from signals.accuracy_tracker import get_accuracy_summary


def print_banner(
    symbols: list[str],
    timeframe: str,
    execute: bool,
    dry_run: bool,
    once: bool,
    min_confidence: float,
):
    """Print startup banner."""
    mode = "LIVE" if execute else "DRY RUN" if dry_run else "SIGNAL ONLY"
    testnet = "TESTNET" if settings.execution.testnet else "MAINNET"

    print()
    print("=" * 70)
    print("  TRADING SWARM - SIGNAL LOOP")
    print("=" * 70)
    print(f"  Mode:           {mode} ({testnet})")
    print(f"  Timeframe:      {timeframe}")
    print(f"  Symbols:        {len(symbols)} ({', '.join(symbols[:3])}{'...' if len(symbols) > 3 else ''})")
    print(f"  Min Confidence: {min_confidence:.0%}")
    print(f"  Generator:      {settings.ollama.generator_model}")
    print(f"  Critic:         {settings.ollama.critic_model}")
    print(f"  Single Cycle:   {'Yes' if once else 'No'}")
    print("=" * 70)

    # Print accuracy summary if available
    accuracy = get_accuracy_summary()
    if accuracy["total"] > 0:
        print(f"  Historical Accuracy: {accuracy['correct']}/{accuracy['total']} ({accuracy['accuracy_pct']:.1f}%)")
        print("=" * 70)

    print()

    if execute:
        print("  WARNING: LIVE TRADING ENABLED")
        print("  Orders will be sent to the exchange!")
        print()

    print(f"  Started: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Production signal loop for live trading",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Dry run with default symbols
    python run_signal_loop.py --dry-run

    # Test single cycle with specific symbol
    python run_signal_loop.py --once --symbols BTC/USDT --timeframe 1h

    # Production mode (requires ALLOW_LIVE_TRADING=true)
    ALLOW_LIVE_TRADING=true python run_signal_loop.py --execute
        """,
    )

    parser.add_argument(
        "--symbols",
        type=str,
        default=None,
        help="Comma-separated symbols (default: from settings)",
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default="1h",
        choices=["1m", "5m", "15m", "1h", "4h", "1d"],
        help="Signal timeframe (default: 1h)",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually send orders (requires ALLOW_LIVE_TRADING=true)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate signals but don't execute",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run one cycle and exit (for testing)",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.6,
        help="Minimum confidence for execution (default: 0.6)",
    )

    return parser.parse_args()


async def main():
    """Main entry point."""
    args = parse_args()

    # Parse symbols
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",")]
    else:
        symbols = settings.market_data.symbols

    # Validate execution mode
    if args.execute and args.dry_run:
        logger.error("Cannot use --execute and --dry-run together")
        sys.exit(1)

    # Check ALLOW_LIVE_TRADING for execute mode
    execution_client = None
    if args.execute:
        allow_live = os.getenv("ALLOW_LIVE_TRADING", "").lower()
        if allow_live != "true":
            logger.error(
                "ALLOW_LIVE_TRADING must be 'true' for --execute mode. "
                "Set environment variable: ALLOW_LIVE_TRADING=true"
            )
            sys.exit(1)

        # Import and create execution client
        try:
            from execution.binance_client import BinanceExecutionClient

            execution_client = BinanceExecutionClient(
                execution_settings=settings.execution,
                fee_model_settings=settings.fee_model,
            )
            logger.info("Execution client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize execution client: {e}")
            sys.exit(1)

    # Print startup banner
    print_banner(
        symbols=symbols,
        timeframe=args.timeframe,
        execute=args.execute,
        dry_run=args.dry_run,
        once=args.once,
        min_confidence=args.min_confidence,
    )

    # Run the loop
    try:
        await run_loop(
            symbols=symbols,
            timeframe=args.timeframe,
            execute=args.execute and not args.dry_run,
            min_confidence=args.min_confidence,
            once=args.once,
            execution_client=execution_client,
        )
    except KeyboardInterrupt:
        logger.info("Received interrupt, shutting down gracefully")
    except Exception as e:
        logger.error(f"Signal loop failed: {e}")
        raise


if __name__ == "__main__":
    # Configure logging
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
        level="INFO",
    )
    logger.add(
        "signals/loop.log",
        rotation="1 day",
        retention="7 days",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        level="DEBUG",
    )

    asyncio.run(main())
