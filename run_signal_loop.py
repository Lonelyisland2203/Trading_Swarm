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
    use_graph: bool = True,
):
    """Print startup banner."""
    mode = "LIVE" if execute else "DRY RUN" if dry_run else "SIGNAL ONLY"
    testnet = "TESTNET" if settings.execution.testnet else "MAINNET"
    pipeline = "LANGGRAPH" if use_graph else "LEGACY"

    print()
    print("=" * 70)
    print("  TRADING SWARM - SIGNAL LOOP")
    print("=" * 70)
    print(f"  Mode:           {mode} ({testnet})")
    print(f"  Pipeline:       {pipeline}")
    print(f"  Timeframe:      {timeframe}")
    print(
        f"  Symbols:        {len(symbols)} ({', '.join(symbols[:3])}{'...' if len(symbols) > 3 else ''})"
    )
    print(f"  Min Confidence: {min_confidence:.0%}")
    print(f"  Generator:      {settings.ollama.generator_model}")
    print(f"  Critic:         {settings.ollama.critic_model}")
    print(f"  Single Cycle:   {'Yes' if once else 'No'}")
    print("=" * 70)

    # Print accuracy summary if available
    accuracy = get_accuracy_summary()
    if accuracy["total"] > 0:
        print(
            f"  Historical Accuracy: {accuracy['correct']}/{accuracy['total']} ({accuracy['accuracy_pct']:.1f}%)"
        )
        print("=" * 70)

    print()

    if execute:
        print("  WARNING: LIVE TRADING ENABLED")
        print("  Orders will be sent to the exchange!")
        print()

    print(f"  Started: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print()


async def run_graph_loop(
    symbols: list[str],
    timeframe: str,
    dry_run: bool = True,
    once: bool = False,
) -> None:
    """
    Run trading graph loop for all symbols.

    Uses LangGraph-based TradingGraph for each symbol.
    Runs on schedule aligned to timeframe bar closes.

    Args:
        symbols: List of trading pairs
        timeframe: Timeframe string
        dry_run: If True, skip execution
        once: Run single cycle and exit
    """
    from orchestration.trading_graph import TradingGraph
    from signals.preflight import check_stop_file, run_preflight_checks
    from signals.signal_models import get_timeframe_duration_ms

    # Calculate bar duration
    bar_duration_ms = get_timeframe_duration_ms(timeframe)
    bar_duration_s = bar_duration_ms / 1000

    logger.info(
        "Starting graph loop",
        symbols=symbols,
        timeframe=timeframe,
        dry_run=dry_run,
        once=once,
    )

    graph = TradingGraph()

    while True:
        # Check STOP file first
        if check_stop_file():
            logger.warning("STOP file detected, halting graph loop")
            break

        cycle_start = datetime.now(timezone.utc)

        # Run preflight checks
        preflight = run_preflight_checks()
        if not preflight.passed:
            logger.warning(
                "Preflight failed, waiting to retry",
                reason=preflight.reason,
            )
            if once:
                logger.error("Preflight failed in --once mode, exiting")
                break
            await asyncio.sleep(60)  # Wait 1 minute and retry
            continue

        # Run graph for each symbol
        results = []
        for symbol in symbols:
            if check_stop_file():
                logger.warning("STOP file detected mid-cycle, halting")
                break

            logger.info("Processing symbol", symbol=symbol, timeframe=timeframe)

            result = await graph.run(
                symbol=symbol,
                timeframe=timeframe,
                dry_run=dry_run,
            )
            results.append(result)

            # Print summary
            if result.synthesis_output:
                direction = result.synthesis_output.direction
                position = result.synthesis_output.position_size_fraction
                errors = len(result.errors)
                print(f"  {symbol}: {direction} ({position:.0%}) {'[ERRORS]' if errors else ''}")

        logger.info(
            "Cycle complete",
            symbols_processed=len(results),
            duration_s=(datetime.now(timezone.utc) - cycle_start).total_seconds(),
        )

        if once:
            logger.info("Single cycle complete, exiting")
            break

        # Calculate sleep until next bar
        cycle_duration = (datetime.now(timezone.utc) - cycle_start).total_seconds()
        sleep_duration = max(0, bar_duration_s - cycle_duration)

        if sleep_duration > 0:
            logger.info(f"Sleeping {sleep_duration:.0f}s until next cycle")
            await asyncio.sleep(sleep_duration)


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
    parser.add_argument(
        "--graph",
        action="store_true",
        default=True,
        dest="use_graph",
        help="Use LangGraph-based pipeline (default: True)",
    )
    parser.add_argument(
        "--no-graph",
        action="store_false",
        dest="use_graph",
        help="Use legacy signal loop (deprecated)",
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
        use_graph=args.use_graph,
    )

    # Run the appropriate pipeline
    try:
        if args.use_graph:
            # LangGraph-based pipeline (Session 17S)
            await run_graph_loop(
                symbols=symbols,
                timeframe=args.timeframe,
                dry_run=args.dry_run or not args.execute,
                once=args.once,
            )
        else:
            # Legacy signal loop (deprecated)
            logger.warning("Using legacy signal loop (--no-graph). Consider migrating to --graph.")
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
