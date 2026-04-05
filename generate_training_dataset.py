#!/usr/bin/env python3
"""
Comprehensive training dataset generation with parallelization.

Generates training data across:
- N symbols
- M timeframes
- K historical windows
- T task types (3: PREDICT_DIRECTION, ASSESS_MOMENTUM, IDENTIFY_SUPPORT_RESISTANCE)
- 5 personas (fixed)

Respects VRAM constraint: models never loaded simultaneously.
Supports resume from interruption.

Usage:
    # Full run
    python generate_training_dataset.py \
        --symbols BTC/USDT,ETH/USDT,SOL/USDT,ADA/USDT,DOT/USDT \
        --timeframes 1h,4h \
        --windows 15 \
        --stride 100 \
        --output outputs/dataset_v1

    # Resume interrupted run
    python generate_training_dataset.py \
        --resume outputs/dataset_v1/examples.jsonl

    # Quick test
    python generate_training_dataset.py --quick-test
"""

import argparse
import asyncio
import json
import sys
import time
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, UTC
from pathlib import Path
from typing import Optional

from loguru import logger

from config.settings import settings
from data.historical_windows import (
    HistoricalWindow,
    calculate_window_timestamps,
    fetch_window_data,
)
from data.inference_queue import InferenceJob, InferenceQueue
from data.market_data import MarketDataService
from data.prompt_builder import PromptBuilder, TaskConfig, TaskType, TASK_CONFIGS
from data.regime_filter import RegimeClassifier
from utils.progress_tracker import ProgressTracker

# Windows async event loop policy for compatibility
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


@dataclass
class DatasetConfig:
    """Configuration for dataset generation."""

    symbols: list[str]
    timeframes: list[str]
    window_count: int
    window_stride_bars: int
    lookback_bars: int
    task_types: list[TaskType]
    output_dir: Path
    resume_from: Optional[Path] = None


async def phase1_prepare_contexts(config: DatasetConfig) -> list[InferenceJob]:
    """
    Phase 1: PARALLEL data preparation.

    Workflow:
    1. Fetch latest OHLCV for all symbol/timeframe pairs (parallel)
    2. Calculate historical window timestamps for each pair
    3. Fetch OHLCV for each window (parallel, skip on insufficient data)
    4. Compute indicators and classify regime for each window
    5. Build prompts for all task types per window
    6. Create InferenceJob for each (symbol, timeframe, window, task)

    Returns:
        List of InferenceJobs ready for sequential processing
    """
    logger.info(
        "Phase 1: Preparing contexts",
        symbols=len(config.symbols),
        timeframes=len(config.timeframes),
        windows=config.window_count,
        task_types=len(config.task_types),
    )

    jobs = []
    regime_classifier = RegimeClassifier()
    prompt_builder = PromptBuilder()

    async with MarketDataService() as service:
        # Parallel: Fetch latest data for all symbol/timeframe pairs
        logger.info("Fetching latest data for all symbol/timeframe pairs...")
        latest_data_tasks = [
            service.fetch_ohlcv(symbol, timeframe, lookback_bars=10)
            for symbol in config.symbols
            for timeframe in config.timeframes
        ]
        latest_data_results = await asyncio.gather(
            *latest_data_tasks, return_exceptions=True
        )

        # Build (symbol, timeframe, latest_timestamp) tuples
        symbol_timeframe_pairs = []
        for (symbol, timeframe), result in zip(
            [(s, t) for s in config.symbols for t in config.timeframes],
            latest_data_results,
        ):
            if isinstance(result, Exception):
                logger.warning(
                    "Failed to fetch latest data",
                    symbol=symbol,
                    timeframe=timeframe,
                    error=str(result),
                )
                continue

            latest_timestamp_ms = int(result["timestamp"].iloc[-1])
            symbol_timeframe_pairs.append((symbol, timeframe, latest_timestamp_ms))

        logger.info(
            f"Successfully fetched data for {len(symbol_timeframe_pairs)} symbol/timeframe pairs"
        )

        # Calculate all historical window timestamps
        logger.info("Calculating historical window timestamps...")
        all_windows = []
        for symbol, timeframe, latest_ts in symbol_timeframe_pairs:
            window_timestamps = calculate_window_timestamps(
                current_timestamp_ms=latest_ts,
                timeframe=timeframe,
                window_count=config.window_count,
                stride_bars=config.window_stride_bars,
            )

            for stride_idx, end_ts in enumerate(window_timestamps):
                window = HistoricalWindow(
                    symbol=symbol,
                    timeframe=timeframe,
                    end_timestamp_ms=end_ts,
                    lookback_bars=config.lookback_bars,
                    stride_index=stride_idx,
                )
                all_windows.append(window)

        logger.info(f"Calculated {len(all_windows)} historical windows")

        # Parallel: Fetch OHLCV for all windows
        logger.info("Fetching OHLCV data for all windows...")
        fetch_tasks = [fetch_window_data(service, window) for window in all_windows]
        window_data_results = await asyncio.gather(*fetch_tasks, return_exceptions=True)

        # Build InferenceJobs for windows with sufficient data
        logger.info("Building inference jobs...")
        skipped_windows = 0
        for window, df_result in zip(all_windows, window_data_results):
            if isinstance(df_result, Exception) or df_result is None:
                skipped_windows += 1
                logger.debug(
                    "Skipping window due to insufficient data",
                    symbol=window.symbol,
                    timeframe=window.timeframe,
                    stride=window.stride_index,
                )
                continue

            df = df_result

            # Classify regime
            regime_series = regime_classifier.classify_regime(df["close"])
            market_regime = regime_series.iloc[-1]

            # Generate tasks for this context (one per task type)
            for task_config in [tc for tc in TASK_CONFIGS if tc.task_type in config.task_types]:
                try:
                    task_prompt = prompt_builder.build_prompt(
                        task=task_config,
                        df=df,
                        symbol=window.symbol,
                        timeframe=window.timeframe,
                        market_regime=market_regime,
                    )
                except ValueError as e:
                    logger.warning(
                        "Failed to build prompt",
                        symbol=window.symbol,
                        timeframe=window.timeframe,
                        task_type=task_config.task_type.value,
                        error=str(e),
                    )
                    continue

                # Create unique context_id for this (symbol, timeframe, timestamp, task)
                # This groups 5 personas together for DPO pairing
                context_id = f"{window.symbol}_{window.timeframe}_{window.end_timestamp_ms}_{task_config.task_type.value}"

                job = InferenceJob(
                    job_id=f"{context_id}_{uuid.uuid4().hex[:8]}",
                    context_id=context_id,
                    symbol=window.symbol,
                    timeframe=window.timeframe,
                    timestamp_ms=window.end_timestamp_ms,
                    ohlcv_df=df,
                    market_regime=market_regime,
                    task_type=task_config.task_type,
                    task_prompt=task_prompt,
                )

                jobs.append(job)

    logger.info(
        "Phase 1 complete",
        total_jobs=len(jobs),
        skipped_windows=skipped_windows,
        contexts=len(set(j.context_id for j in jobs)),
    )
    return jobs


async def phase2_run_inference(
    jobs: list[InferenceJob],
    config: DatasetConfig,
    tracker: ProgressTracker,
) -> None:
    """
    Phase 2: SEQUENTIAL model inference through queue.

    Process jobs one at a time through GPU:
    - Load generator → generate 5 personas → unload
    - Load critic → critique signals → unload
    - Save examples incrementally to JSONL
    - Update progress tracker
    - Skip completed contexts on resume
    """
    logger.info("Phase 2: Running inference", total_jobs=len(jobs))

    output_file = config.output_dir / "examples.jsonl"

    queue = InferenceQueue(output_file, resume=bool(config.resume_from))

    def progress_callback(context_id: str, duration_sec: float, success: bool) -> None:
        """Progress callback for tracker updates."""
        tracker.update(context_id, duration_sec, success)

        # Print progress every 5 contexts
        if tracker.completed % 5 == 0:
            stats = tracker.get_stats()
            logger.info(
                "Progress update",
                completed=stats["completed"],
                total=stats["total"],
                failed=stats["failed"],
                success_rate=f"{stats['success_rate']:.1%}",
                eta_min=stats["eta_seconds"] / 60,
            )

    await queue.process_all(jobs, progress_callback=progress_callback)

    logger.info(
        "Phase 2 complete",
        completed=tracker.completed,
        failed=tracker.failed,
    )


async def phase3_postprocess(config: DatasetConfig) -> dict:
    """
    Phase 3: PARALLEL post-processing.

    Workflow:
    1. Load all examples from JSONL
    2. Validate schema compatibility
    3. Compute statistics by:
       - Market regime (RISK_ON, NEUTRAL, RISK_OFF)
       - Persona (5 types)
       - Task type (3 types)
       - Acceptance rate
    4. Verify context_id grouping (5 personas per context)
    5. Export summary to JSON

    Returns:
        Statistics dictionary
    """
    logger.info("Phase 3: Post-processing")

    examples_file = config.output_dir / "examples.jsonl"
    summary_file = config.output_dir / "summary.json"

    if not examples_file.exists():
        logger.error("Examples file not found", file=str(examples_file))
        return {"error": "Examples file not found"}

    # Load all examples
    logger.info("Loading examples from JSONL...")
    examples = []
    with open(examples_file) as f:
        for line_num, line in enumerate(f, 1):
            try:
                examples.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning(
                    "Skipping malformed line",
                    line_num=line_num,
                    error=str(e),
                )

    logger.info(f"Loaded {len(examples)} examples")

    # Compute statistics
    stats = {
        "generation_date": datetime.now(UTC).isoformat(),
        "config": {
            "symbols": config.symbols,
            "timeframes": config.timeframes,
            "window_count": config.window_count,
            "window_stride_bars": config.window_stride_bars,
            "task_types": [t.value for t in config.task_types],
        },
        "total_examples": len(examples),
        "examples_by_regime": dict(Counter(ex["market_regime"] for ex in examples)),
        "examples_by_persona": dict(Counter(ex["persona"] for ex in examples)),
        "examples_by_task": dict(
            Counter(
                ex.get("generator_signal", {}).get("task_type", "unknown")
                for ex in examples
            )
        ),
        "acceptance_rate": (
            sum(1 for ex in examples if ex.get("was_accepted", False)) / len(examples)
            if examples
            else 0.0
        ),
        "unique_contexts": len(set(ex["context_id"] for ex in examples)),
    }

    # Validate context grouping
    context_counts = Counter(ex["context_id"] for ex in examples)
    incomplete_contexts = {
        ctx: count for ctx, count in context_counts.items() if count != 5
    }
    if incomplete_contexts:
        logger.warning(
            f"Found {len(incomplete_contexts)} incomplete contexts (not 5 personas)",
            sample=list(incomplete_contexts.items())[:5],
        )
        stats["incomplete_contexts"] = len(incomplete_contexts)
        stats["incomplete_context_sample"] = dict(
            list(incomplete_contexts.items())[:10]
        )

    # Save summary
    with open(summary_file, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(
        "Phase 3 complete",
        total_examples=len(examples),
        unique_contexts=stats["unique_contexts"],
        acceptance_rate=f"{stats['acceptance_rate']:.1%}",
    )

    return stats


async def main_workflow(config: DatasetConfig) -> None:
    """Execute 3-phase workflow."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.time()

    logger.info("=" * 60)
    logger.info("STARTING COMPREHENSIVE DATASET GENERATION")
    logger.info("=" * 60)
    logger.info(
        "Configuration",
        symbols=config.symbols,
        timeframes=config.timeframes,
        windows=config.window_count,
        stride=config.window_stride_bars,
        task_types=[t.value for t in config.task_types],
        output_dir=str(config.output_dir),
        resume=bool(config.resume_from),
    )

    # Phase 1: Prepare (parallel)
    phase1_start = time.time()
    jobs = await phase1_prepare_contexts(config)
    phase1_time = time.time() - phase1_start
    logger.info(f"Phase 1 completed in {phase1_time / 60:.1f} minutes")

    if not jobs:
        logger.error("No jobs generated in Phase 1. Exiting.")
        return

    # Initialize progress tracker
    tracker = ProgressTracker(total_contexts=len(jobs))

    # Phase 2: Inference (sequential)
    phase2_start = time.time()
    await phase2_run_inference(jobs, config, tracker)
    phase2_time = time.time() - phase2_start
    logger.info(f"Phase 2 completed in {phase2_time / 60:.1f} minutes")

    # Save final progress state
    progress_file = config.output_dir / "progress.json"
    tracker.save_state(progress_file)

    # Phase 3: Post-process (parallel)
    phase3_start = time.time()
    stats = await phase3_postprocess(config)
    phase3_time = time.time() - phase3_start
    logger.info(f"Phase 3 completed in {phase3_time / 60:.1f} minutes")

    # Final summary
    elapsed_hours = (time.time() - start_time) / 3600
    logger.info("=" * 60)
    logger.info("DATASET GENERATION COMPLETE")
    logger.info("=" * 60)
    logger.info(
        "Summary",
        total_time_hours=f"{elapsed_hours:.2f}",
        total_examples=stats.get("total_examples", 0),
        unique_contexts=stats.get("unique_contexts", 0),
        acceptance_rate=f"{stats.get('acceptance_rate', 0):.1%}",
        output_dir=str(config.output_dir),
    )


def parse_args() -> DatasetConfig:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate comprehensive training dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test (1 symbol, 1 timeframe, 3 windows)
  python generate_training_dataset.py --quick-test

  # Full run with specific symbols
  python generate_training_dataset.py \\
      --symbols BTC/USDT,ETH/USDT,SOL/USDT \\
      --timeframes 1h,4h \\
      --windows 15 \\
      --output outputs/dataset_v1

  # Resume interrupted run
  python generate_training_dataset.py \\
      --resume outputs/dataset_v1/examples.jsonl
        """,
    )

    parser.add_argument(
        "--symbols",
        type=str,
        default="BTC/USDT,ETH/USDT,SOL/USDT,ADA/USDT,DOT/USDT,AVAX/USDT,MATIC/USDT,LINK/USDT,UNI/USDT,ATOM/USDT",
        help="Comma-separated trading pairs",
    )
    parser.add_argument(
        "--timeframes",
        type=str,
        default="1m,5m,15m,1h,4h,1d",
        help="Comma-separated timeframes",
    )
    parser.add_argument(
        "--windows",
        type=int,
        default=15,
        help="Number of historical windows per symbol/timeframe",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=100,
        help="Window stride in bars (how far back each window steps)",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=100,
        help="Lookback bars per window",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/dataset"),
        help="Output directory for generated data",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume from existing JSONL file (path to examples.jsonl)",
    )
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Quick test mode: 1 symbol, 1 timeframe, 3 windows",
    )

    args = parser.parse_args()

    # Quick test mode overrides
    if args.quick_test:
        symbols = ["BTC/USDT"]
        timeframes = ["1h"]
        window_count = 3
        output_dir = Path("outputs/dataset_test")
        resume_from = None
        logger.info("Quick test mode enabled")
    else:
        symbols = args.symbols.split(",")
        timeframes = args.timeframes.split(",")
        window_count = args.windows
        output_dir = args.output
        resume_from = args.resume

    # All three task types
    task_types = [
        TaskType.PREDICT_DIRECTION,
        TaskType.ASSESS_MOMENTUM,
        TaskType.IDENTIFY_SUPPORT_RESISTANCE,
    ]

    return DatasetConfig(
        symbols=symbols,
        timeframes=timeframes,
        window_count=window_count,
        window_stride_bars=args.stride,
        lookback_bars=args.lookback,
        task_types=task_types,
        output_dir=output_dir,
        resume_from=resume_from,
    )


if __name__ == "__main__":
    # Configure logging
    logger.remove()  # Remove default handler
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level="INFO",
    )

    config = parse_args()

    try:
        asyncio.run(main_workflow(config))
    except KeyboardInterrupt:
        logger.warning("Generation interrupted by user (Ctrl+C)")
        logger.info(
            "Progress saved. Resume with: python generate_training_dataset.py --resume "
            + str(config.output_dir / "examples.jsonl")
        )
        sys.exit(0)
    except Exception as e:
        logger.exception("Fatal error during generation")
        sys.exit(1)
