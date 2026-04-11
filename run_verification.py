#!/usr/bin/env python3
"""
run_verification.py — Verify signal outcomes and close the feedback loop.

Reads unverified signals from signal_log.jsonl, fetches actual outcomes,
computes fee-adjusted returns, and logs verified results.

This creates the self-improvement loop:
signals → verification → new training data → GRPO retraining → better signals

Usage:
    # Run once and exit
    python run_verification.py --once

    # Run continuously every 4 hours
    python run_verification.py

    # Export verified signals for training
    python run_verification.py --export data/verified_training.jsonl

    # Show stats only
    python run_verification.py --stats
"""

import argparse
import asyncio
import sys
from datetime import datetime, timezone

from loguru import logger

from config.fee_model import FeeModelSettings
from data.market_data import MarketDataService
from signals.verification import (
    check_training_trigger,
    compute_verification_stats,
    export_for_training,
    format_daily_summary,
    load_unverified_signals,
    save_verified_result,
    verify_signal,
)


# Default schedule: every 4 hours
DEFAULT_INTERVAL_HOURS = 4


def configure_logging() -> None:
    """Configure loguru for verification output."""
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
        level="INFO",
    )


async def run_verification_cycle() -> int:
    """
    Run a single verification cycle.

    Returns:
        Number of signals verified
    """
    unverified = load_unverified_signals()

    if not unverified:
        logger.info("No signals ready for verification")
        return 0

    logger.info(f"Found {len(unverified)} signals ready for verification")

    fee_model = FeeModelSettings()
    verified_count = 0

    async with MarketDataService() as market_data:
        for signal in unverified:
            try:
                result = await verify_signal(
                    signal=signal,
                    market_data_service=market_data,
                    fee_model=fee_model,
                )

                if result:
                    save_verified_result(result)
                    verified_count += 1

                    status = "CORRECT" if result.correct else "INCORRECT"
                    logger.info(
                        f"Verified {result.symbol}: {result.predicted_direction} → "
                        f"{result.actual_direction} ({status}) "
                        f"Net: {result.net_return_pct:+.2f}%"
                    )

            except Exception as e:
                logger.error(f"Error verifying signal: {e}")
                continue

    logger.info(f"Verified {verified_count}/{len(unverified)} signals")

    return verified_count


async def run_verification_loop(
    interval_hours: float = DEFAULT_INTERVAL_HOURS,
    once: bool = False,
) -> None:
    """
    Run verification loop.

    Args:
        interval_hours: Hours between verification cycles
        once: If True, run once and exit
    """
    while True:
        try:
            logger.info("Starting verification cycle")
            verified = await run_verification_cycle()

            # Print stats after each cycle
            stats = compute_verification_stats()
            print("\n" + format_daily_summary(stats) + "\n")

            # Check training trigger
            trigger_status = check_training_trigger()
            if trigger_status["ready"]:
                logger.warning(
                    f"TRAINING TRIGGER: {trigger_status['message']}"
                )

            if once:
                logger.info("Once mode - exiting")
                break

            # Wait for next cycle
            logger.info(f"Next verification cycle in {interval_hours} hours")
            await asyncio.sleep(interval_hours * 3600)

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
            break
        except Exception as e:
            logger.error(f"Verification cycle error: {e}")
            if once:
                raise
            # Wait before retry
            await asyncio.sleep(60)


def print_stats() -> None:
    """Print current verification statistics."""
    stats = compute_verification_stats()
    print(format_daily_summary(stats))


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Verify signal outcomes and close the feedback loop.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--once",
        action="store_true",
        help="Run one verification cycle and exit",
    )

    parser.add_argument(
        "--interval",
        type=float,
        default=DEFAULT_INTERVAL_HOURS,
        help=f"Hours between verification cycles (default: {DEFAULT_INTERVAL_HOURS})",
    )

    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show current verification statistics and exit",
    )

    parser.add_argument(
        "--export",
        type=str,
        metavar="PATH",
        help="Export verified signals for training to PATH and exit",
    )

    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help="Minimum confidence for export (default: 0.0)",
    )

    parser.add_argument(
        "--check-trigger",
        action="store_true",
        help="Check if training trigger threshold reached and exit",
    )

    args = parser.parse_args()

    configure_logging()

    # Print banner
    print("\n" + "=" * 60)
    print("SIGNAL VERIFICATION")
    print(f"Started: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print("=" * 60 + "\n")

    # Handle stats-only mode
    if args.stats:
        print_stats()
        return

    # Handle export mode
    if args.export:
        count = export_for_training(args.export, min_confidence=args.min_confidence)
        print(f"Exported {count} verified signals to {args.export}")
        return

    # Handle check-trigger mode
    if args.check_trigger:
        status = check_training_trigger()
        print(status["message"])
        sys.exit(0 if status["ready"] else 1)

    # Run verification loop
    try:
        asyncio.run(
            run_verification_loop(
                interval_hours=args.interval,
                once=args.once,
            )
        )
    except KeyboardInterrupt:
        logger.info("Verification stopped")


if __name__ == "__main__":
    main()
