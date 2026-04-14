#!/usr/bin/env python3
"""
GRPO Training Data Generator — Generate training data from historical market data.

Creates grpo_training_data.jsonl with market snapshots and verified outcomes.
This is a pure data pipeline - no model loading, no VRAM needed.

Each example contains:
- market_snapshot: formatted prompt (indicators, multi-TF context)
- actual_direction: "LONG"/"SHORT"/"FLAT" based on fee-adjusted net return
- gross_return_pct: raw return before fees (fees applied during reward computation)
- timestamp_ms: bar close timestamp for temporal ordering

Critical constraints:
- Temporal safety: market snapshot only contains data available at bar close time
- Look-ahead for outcome is ONLY used for labeling, never included in prompt
- end_ts=as_of on all data fetches
- Uses walk_forward.py splits for train/test separation

Usage:
    # Generate data for all default symbols and timeframes
    python -m training.grpo_data_generator --output data/grpo_training_data.jsonl

    # Custom symbols and timeframes
    python -m training.grpo_data_generator --symbols BTC/USDT,ETH/USDT --timeframes 1h,4h

    # Limited run for testing
    python -m training.grpo_data_generator --limit 100 --output data/grpo_test.jsonl

    # Specify date range
    python -m training.grpo_data_generator --start-date 2024-01-01 --end-date 2024-06-01
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

from loguru import logger
from tqdm import tqdm

from config.fee_model import FeeModelSettings
from data.historical_windows import (
    HistoricalWindow,
    fetch_window_data,
    timeframe_to_milliseconds,
)
from data.market_data import MarketDataService
from data.market_snapshot import build_market_snapshot
from training.grpo_data import GRPOTrainingExample

# Windows async event loop policy
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


# Default configuration
DEFAULT_SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "ADA/USDT", "DOT/USDT"]
DEFAULT_TIMEFRAMES = ["1h", "4h"]
LOOKBACK_BARS = 100  # Bars for indicators computation
STRIDE_BARS = 24  # How many bars between each window

# Timeframe-adaptive verification horizons (in bars)
# Longer timeframes need fewer bars for the same holding period
VERIFICATION_HORIZONS = {
    "1m": 60,  # 60 minutes = 1 hour
    "5m": 24,  # 2 hours
    "15m": 16,  # 4 hours
    "1h": 24,  # 24 hours
    "4h": 12,  # 48 hours
    "1d": 5,  # 5 days
}


def get_verification_horizon(timeframe: str) -> int:
    """Get timeframe-adaptive verification horizon in bars."""
    return VERIFICATION_HORIZONS.get(timeframe, 24)


def classify_direction(
    gross_return_pct: float,
    fee_model: FeeModelSettings,
    holding_periods_8h: float,
) -> str:
    """
    Classify direction based on fee-adjusted return.

    Args:
        gross_return_pct: Gross return as percentage (e.g., 0.15 for +0.15%)
        fee_model: Fee model for threshold calculation
        holding_periods_8h: Holding period in 8-hour funding periods

    Returns:
        "LONG" if profitable after fees, "SHORT" if loss after fees, "FLAT" if within threshold
    """
    # Get minimum profitable return (break-even threshold)
    min_profitable = fee_model.minimum_profitable_return_pct(holding_periods_8h)

    # Classify based on threshold
    if gross_return_pct > min_profitable:
        return "LONG"
    elif gross_return_pct < -min_profitable:
        return "SHORT"
    else:
        return "FLAT"


def example_to_dict(example: GRPOTrainingExample) -> dict[str, Any]:
    """Convert GRPOTrainingExample to JSON-serializable dict."""
    return {
        "market_snapshot": example.market_snapshot,
        "actual_direction": example.actual_direction,
        "gross_return_pct": example.gross_return_pct,
        "timestamp_ms": example.timestamp_ms,
    }


def load_completed_timestamps(output_file: Path) -> set[int]:
    """Load already-processed timestamps from existing JSONL."""
    from utils.jsonl import iter_jsonl

    return {ex["timestamp_ms"] for ex in iter_jsonl(output_file) if "timestamp_ms" in ex}


async def generate_grpo_dataset(
    output_file: Path,
    symbols: list[str],
    timeframes: list[str],
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    limit: int | None = None,
    resume: bool = True,
) -> int:
    """
    Generate GRPO training dataset from historical market data.

    Args:
        output_file: Path to output JSONL file
        symbols: List of trading pairs
        timeframes: List of timeframes
        start_date: Optional start date for data
        end_date: Optional end date for data
        limit: Maximum examples to generate (None for all)
        resume: Skip already-processed examples

    Returns:
        Number of examples generated
    """
    fee_model = FeeModelSettings()

    # Load completed timestamps for resume
    completed_timestamps = load_completed_timestamps(output_file) if resume else set()
    if completed_timestamps:
        logger.info(f"Resuming: {len(completed_timestamps)} examples already processed")

    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    generated_count = 0
    skipped_count = 0
    error_count = 0

    # Convert date bounds to timestamps
    start_ts = int(start_date.timestamp() * 1000) if start_date else None
    end_ts = int(end_date.timestamp() * 1000) if end_date else None

    async with MarketDataService() as market_data:
        # Calculate total windows for progress bar
        total_estimate = len(symbols) * len(timeframes) * 200  # Rough estimate
        pbar = tqdm(
            total=min(limit, total_estimate) if limit else total_estimate,
            desc="Generating GRPO examples",
            unit="ex",
        )

        for symbol in symbols:
            for timeframe in timeframes:
                try:
                    # Get latest available data to determine window end points
                    latest_df = await market_data.fetch_ohlcv(symbol, timeframe, lookback_bars=10)
                    if latest_df.empty:
                        logger.warning(f"No data for {symbol}/{timeframe}")
                        continue

                    current_ts = int(latest_df["timestamp"].iloc[-1])

                    # Apply end date bound
                    if end_ts and current_ts > end_ts:
                        current_ts = end_ts

                    bar_ms = timeframe_to_milliseconds(timeframe)
                    stride_ms = STRIDE_BARS * bar_ms
                    verification_horizon = get_verification_horizon(timeframe)
                    horizon_ms = verification_horizon * bar_ms

                    # Walk backward through time
                    window_idx = 0
                    while True:
                        if limit and generated_count >= limit:
                            break

                        # Calculate window end timestamp
                        # Need to leave room for verification horizon
                        window_end_ms = current_ts - (window_idx * stride_ms) - horizon_ms

                        # Stop if we've gone before start date
                        if start_ts and window_end_ms < start_ts:
                            break

                        # Stop if we've gone too far back (data may not exist)
                        if window_idx > 500:  # Safety limit
                            break

                        # Skip if already processed
                        if window_end_ms in completed_timestamps:
                            skipped_count += 1
                            window_idx += 1
                            pbar.update(1)
                            continue

                        try:
                            # Fetch historical window data (point-in-time safe)
                            window = HistoricalWindow(
                                symbol=symbol,
                                timeframe=timeframe,
                                end_timestamp_ms=window_end_ms,
                                lookback_bars=LOOKBACK_BARS,
                                stride_index=window_idx,
                            )

                            snapshot_df = await fetch_window_data(
                                market_data, window, min_completeness=0.90
                            )

                            if snapshot_df is None or len(snapshot_df) < LOOKBACK_BARS * 0.9:
                                logger.debug(
                                    f"Insufficient data for window {window_idx} "
                                    f"({symbol}/{timeframe})"
                                )
                                error_count += 1
                                window_idx += 1
                                pbar.update(1)
                                continue

                            # Fetch FORWARD data for verified outcome
                            # This data is ONLY for labeling, never in the prompt
                            forward_end_ms = window_end_ms + horizon_ms

                            # Fetch forward data with temporal anchoring
                            forward_df = await market_data.get_ohlcv_as_of(
                                symbol=symbol,
                                timeframe=timeframe,
                                as_of=forward_end_ms,
                                lookback_bars=verification_horizon + 5,
                            )

                            # Filter to forward period only
                            if forward_df is not None and not forward_df.empty:
                                forward_df = forward_df[
                                    (forward_df["timestamp"] > window_end_ms)
                                    & (forward_df["timestamp"] <= forward_end_ms)
                                ]

                            if (
                                forward_df is None
                                or forward_df.empty
                                or len(forward_df) < verification_horizon * 0.8
                            ):
                                logger.debug(f"Insufficient forward data for window {window_idx}")
                                error_count += 1
                                window_idx += 1
                                pbar.update(1)
                                continue

                            # Calculate verified outcome
                            # Entry: next bar open after snapshot
                            entry_price = float(forward_df["open"].iloc[0])
                            # Exit: horizon bar close
                            exit_price = float(forward_df["close"].iloc[-1])

                            # Gross return percentage
                            gross_return_pct = ((exit_price / entry_price) - 1) * 100

                            # Calculate holding periods for fee model
                            holding_bars = len(forward_df)
                            holding_ms = holding_bars * bar_ms
                            holding_periods_8h = holding_ms / (8 * 3600 * 1000)

                            # Classify direction using fee threshold
                            actual_direction = classify_direction(
                                gross_return_pct, fee_model, holding_periods_8h
                            )

                            # Build market snapshot (only historical data!)
                            market_snapshot = build_market_snapshot(
                                df=snapshot_df,
                                symbol=symbol,
                                timeframe=timeframe,
                                fee_model=fee_model,
                            )

                            # Create training example
                            example = GRPOTrainingExample(
                                market_snapshot=market_snapshot,
                                actual_direction=actual_direction,
                                gross_return_pct=gross_return_pct,
                                timestamp_ms=window_end_ms,
                            )

                            # Append to JSONL
                            with open(output_file, "a") as f:
                                f.write(json.dumps(example_to_dict(example)) + "\n")

                            generated_count += 1
                            pbar.update(1)
                            pbar.set_postfix(
                                gen=generated_count,
                                skip=skipped_count,
                                err=error_count,
                            )

                        except Exception as e:
                            logger.error(
                                f"Error processing window {window_idx} ({symbol}/{timeframe}): {e}"
                            )
                            error_count += 1
                            pbar.update(1)

                        window_idx += 1

                    if limit and generated_count >= limit:
                        break

                except Exception as e:
                    logger.error(f"Error processing {symbol}/{timeframe}: {e}")
                    continue

            if limit and generated_count >= limit:
                break

        pbar.close()

    logger.info(
        "GRPO data generation complete",
        generated=generated_count,
        skipped=skipped_count,
        errors=error_count,
    )

    return generated_count


def parse_symbols(symbols_str: str) -> list[str]:
    """Parse comma-separated symbols string."""
    return [s.strip() for s in symbols_str.split(",") if s.strip()]


def parse_timeframes(timeframes_str: str) -> list[str]:
    """Parse comma-separated timeframes string."""
    return [t.strip() for t in timeframes_str.split(",") if t.strip()]


def parse_date(date_str: str) -> datetime:
    """Parse date string in YYYY-MM-DD format."""
    return datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=UTC)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate GRPO training data from historical market data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/grpo_training_data.jsonl"),
        help="Output JSONL file path",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default=",".join(DEFAULT_SYMBOLS),
        help=f"Comma-separated symbols (default: {','.join(DEFAULT_SYMBOLS)})",
    )
    parser.add_argument(
        "--timeframes",
        type=str,
        default=",".join(DEFAULT_TIMEFRAMES),
        help=f"Comma-separated timeframes (default: {','.join(DEFAULT_TIMEFRAMES)})",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Start date in YYYY-MM-DD format",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date in YYYY-MM-DD format",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of examples to generate (default: no limit)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume from existing output file (default: True)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_false",
        dest="resume",
        help="Start fresh, overwriting existing output",
    )

    args = parser.parse_args()

    # Parse arguments
    symbols = parse_symbols(args.symbols)
    timeframes = parse_timeframes(args.timeframes)
    start_date = parse_date(args.start_date) if args.start_date else None
    end_date = parse_date(args.end_date) if args.end_date else None

    # Validate timeframes
    valid_timeframes = set(VERIFICATION_HORIZONS.keys())
    for tf in timeframes:
        if tf not in valid_timeframes:
            logger.error(f"Invalid timeframe: {tf}. Valid: {valid_timeframes}")
            sys.exit(1)

    # Configure logging
    from utils.logging import configure_cli_logging

    configure_cli_logging(include_name=False)

    logger.info(
        "Starting GRPO data generation",
        symbols=symbols,
        timeframes=timeframes,
        start_date=start_date,
        end_date=end_date,
        limit=args.limit,
    )

    # Run generation
    try:
        count = asyncio.run(
            generate_grpo_dataset(
                output_file=args.output,
                symbols=symbols,
                timeframes=timeframes,
                start_date=start_date,
                end_date=end_date,
                limit=args.limit,
                resume=args.resume,
            )
        )
        logger.info(f"Generated {count} GRPO examples to {args.output}")
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Generation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
