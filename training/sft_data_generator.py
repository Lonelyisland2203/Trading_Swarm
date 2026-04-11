#!/usr/bin/env python3
"""
SFT Data Generator — Reverse reasoning distillation from deepseek-r1:14b.

Generates structured reasoning traces from market snapshots with verified outcomes.
The critic model (deepseek-r1:14b) explains WHY a price moved, producing training
data for the generator model's SFT stage.

Pre-flight order: Data → Temporal → VRAM → Lock → Load

Usage:
    # Full run (2000+ examples)
    python -m training.sft_data_generator --output data/sft_training_data.jsonl

    # Test run
    python -m training.sft_data_generator --limit 10 --output data/sft_test.jsonl

    # Resume from existing JSONL
    python -m training.sft_data_generator --resume data/sft_training_data.jsonl
"""

import argparse
import asyncio
import json
import os
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger
from tqdm import tqdm

from config.fee_model import FeeModelSettings
from config.settings import settings
from data.historical_windows import HistoricalWindow, fetch_window_data
from data.indicators import compute_all_indicators
from data.market_data import MarketDataService
from data.regime_filter import RegimeClassifier
from swarm.ollama_client import OllamaClient
from training.process_lock import acquire_training_lock, check_can_train
from verifier.outcome import determine_direction

# Windows async event loop policy
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


# Critic model for reasoning distillation
CRITIC_MODEL = "deepseek-r1:14b"

# Generation options for deterministic output
GENERATION_OPTIONS = {
    "temperature": 0.0,  # Deterministic for caching
    "top_p": 1.0,
    "num_predict": 1024,
}


@dataclass
class SFTExample:
    """Single SFT training example with input and reasoning trace."""

    example_id: str
    created_at: str
    symbol: str
    timeframe: str
    timestamp_ms: int
    market_snapshot: str  # Formatted market context (input)
    verified_outcome: str  # "HIGHER" | "LOWER" | "FLAT"
    net_return_pct: float  # Fee-adjusted return
    reasoning_trace: str  # Structured reasoning (target output)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


def build_distillation_prompt(
    market_snapshot: str,
    outcome: str,
    net_return_pct: float,
    fee_model: FeeModelSettings,
) -> str:
    """
    Build prompt for reasoning distillation.

    The critic model receives the market snapshot and VERIFIED outcome,
    then explains WHY the price moved in that direction.

    Args:
        market_snapshot: Formatted market context with indicators
        outcome: Verified outcome ("HIGHER" | "LOWER" | "FLAT")
        net_return_pct: Net return after fees
        fee_model: Fee model for context

    Returns:
        Distillation prompt for deepseek-r1:14b
    """
    direction_text = {
        "HIGHER": "went UP",
        "LOWER": "went DOWN",
        "FLAT": "remained FLAT",
    }[outcome]

    # Calculate break-even threshold for context
    break_even = fee_model.minimum_profitable_return_pct(holding_periods_8h=1.0)

    return f"""You are analyzing a cryptocurrency market snapshot AFTER knowing the outcome.
The price {direction_text} with a fee-adjusted net return of {net_return_pct:.3f}%.
Transaction costs (entry + exit + funding + slippage) total approximately {break_even:.3f}%.

Your task: Explain WHY this price movement occurred based on the technical indicators.
Write as if you were predicting this outcome BEFORE it happened.

{market_snapshot}

Provide your analysis in EXACTLY this format:

THESIS: [One sentence directional call with conviction level]

EVIDENCE:
- [Indicator 1]: [Specific value] → [What this signals]
- [Indicator 2]: [Specific value] → [What this signals]
- [Indicator 3]: [Specific value] → [What this signals]
- [Indicator 4]: [Specific value] → [What this signals] (if applicable)
- [Indicator 5]: [Specific value] → [What this signals] (if applicable)

RISK: [Primary risk that could invalidate the thesis]

DECISION: {"LONG" if outcome == "HIGHER" else "SHORT" if outcome == "LOWER" else "FLAT"} | Confidence: [1-5]

Important:
- Cite SPECIFIC indicator values from the snapshot above
- Explain the causal relationship between indicators and price movement
- Acknowledge the transaction cost hurdle ({break_even:.3f}% break-even)
- Be concise but precise"""


def build_market_snapshot(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    fee_model: FeeModelSettings,
) -> str:
    """
    Build market snapshot with indicators for distillation prompt.

    Args:
        df: OHLCV DataFrame with sufficient history
        symbol: Trading pair
        timeframe: Candle timeframe
        fee_model: Fee model for cost context

    Returns:
        Formatted market snapshot string
    """
    # Compute all indicators
    indicators = compute_all_indicators(df, include_volume=True, include_structure=True)

    # Extract key values
    current_price = float(df["close"].iloc[-1])
    rsi = indicators.get("rsi", 50.0)
    macd = indicators.get("macd", 0.0)
    macd_signal = indicators.get("macd_signal", 0.0)
    bb_position = indicators.get("bb_position", 0.5)
    atr = indicators.get("atr", 0.0)
    adx = indicators.get("adx", 0.0)
    obv_slope = indicators.get("obv_slope", 0.0)

    # Format recent price action
    recent = df.tail(10)
    price_lines = []
    for _, row in recent.iterrows():
        timestamp = pd.to_datetime(row["timestamp"], unit="ms")
        change = ((row["close"] - row["open"]) / row["open"]) * 100
        direction = "↑" if change > 0 else "↓"
        price_lines.append(
            f"{timestamp.strftime('%Y-%m-%d %H:%M')} | "
            f"O: ${row['open']:.2f} H: ${row['high']:.2f} "
            f"L: ${row['low']:.2f} C: ${row['close']:.2f} | "
            f"{direction} {change:+.2f}%"
        )

    price_summary = "\n".join(price_lines)

    # Calculate fee context
    holding_periods = 1.0  # 1 funding period estimate
    round_trip_cost = fee_model.round_trip_cost_pct(holding_periods)
    min_profitable = fee_model.minimum_profitable_return_pct(holding_periods)

    # Classify regime
    classifier = RegimeClassifier()
    regime = classifier.classify(df)

    return f"""## Market Data
Symbol: {symbol}
Timeframe: {timeframe}
Current Price: ${current_price:.4f}
Market Regime: {regime.name}

## Technical Indicators
RSI(14): {rsi:.2f}
MACD: {macd:.4f} (Signal: {macd_signal:.4f})
BB Position: {bb_position:.2f} (0.0=lower band, 0.5=middle, 1.0=upper band)
ATR(14): {atr:.4f}
ADX(14): {adx:.2f}
OBV Slope: {obv_slope:.4f}

## Recent Price Action (last 10 bars)
{price_summary}

## Execution Context
Exchange: Binance Futures USDT-M
Estimated round-trip cost: {round_trip_cost:.3f}%
Minimum profitable move: {min_profitable:.3f}%"""


def parse_reasoning_trace(response: str) -> str | None:
    """
    Validate and clean reasoning trace from model response.

    Returns None if the response doesn't contain required sections.
    """
    required_sections = ["THESIS:", "EVIDENCE:", "RISK:", "DECISION:"]

    for section in required_sections:
        if section not in response:
            logger.warning(f"Missing required section: {section}")
            return None

    # Clean up the response (remove any preamble before THESIS)
    if "THESIS:" in response:
        thesis_idx = response.index("THESIS:")
        response = response[thesis_idx:]

    return response.strip()


async def preflight_checks() -> bool:
    """
    Run pre-flight checks: Data → Temporal → VRAM → Lock → Load.

    Returns True if all checks pass, False otherwise.
    """
    logger.info("Running pre-flight checks...")

    # 1. DATA — Check Ollama is accessible
    try:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{settings.ollama.base_url}/api/tags",
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                if resp.status != 200:
                    logger.error("Ollama not responding", status=resp.status)
                    return False
                data = await resp.json()
                models = [m["name"] for m in data.get("models", [])]
                if CRITIC_MODEL not in models and not any(CRITIC_MODEL in m for m in models):
                    logger.error(f"Model {CRITIC_MODEL} not found. Run: ollama pull {CRITIC_MODEL}")
                    return False
        logger.info("✓ DATA: Ollama accessible, model available")
    except Exception as e:
        logger.error(f"Ollama connection failed: {e}")
        return False

    # 2. TEMPORAL — Verify we're not using live data (this is training)
    # Training data generator uses historical windows with end_ts anchoring
    logger.info("✓ TEMPORAL: Historical windows use end_ts anchoring")

    # 3. VRAM — Check GPU memory
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            used, total = map(int, result.stdout.strip().split(", "))
            if used > 2000:  # More than 2GB used
                logger.warning(f"VRAM usage high: {used}/{total} MB. Consider clearing.")
            else:
                logger.info(f"✓ VRAM: {used}/{total} MB (idle)")
        else:
            logger.warning("nvidia-smi failed, skipping VRAM check")
    except Exception as e:
        logger.warning(f"VRAM check failed: {e}")

    # 4. LOCK — Check process lock availability
    can_train, reason = check_can_train()
    if not can_train:
        logger.error(f"Cannot acquire training lock: {reason}")
        return False
    logger.info("✓ LOCK: Training lock available")

    # 5. LOAD — Verify OLLAMA_KEEP_ALIVE=0
    keep_alive = os.environ.get("OLLAMA_KEEP_ALIVE", "")
    if keep_alive != "0":
        logger.warning(f"OLLAMA_KEEP_ALIVE={keep_alive!r}, should be '0'. Setting it now.")
        os.environ["OLLAMA_KEEP_ALIVE"] = "0"
    logger.info("✓ LOAD: OLLAMA_KEEP_ALIVE=0 enforced")

    # Check for STOP file
    stop_file = Path("execution/state/STOP")
    if stop_file.exists():
        logger.error("STOP file exists at execution/state/STOP — refusing to proceed")
        return False

    logger.info("All pre-flight checks passed")
    return True


def load_completed_ids(output_file: Path) -> set[str]:
    """Load already-processed example IDs from existing JSONL."""
    completed = set()
    if output_file.exists():
        with open(output_file) as f:
            for line in f:
                try:
                    ex = json.loads(line)
                    completed.add(ex["example_id"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return completed


async def generate_sft_dataset(
    output_file: Path,
    limit: int | None = None,
    resume: bool = True,
) -> int:
    """
    Generate SFT training dataset with reasoning traces.

    Args:
        output_file: Path to output JSONL file
        limit: Maximum examples to generate (None for all)
        resume: Skip already-processed examples

    Returns:
        Number of examples generated
    """
    # Pre-flight checks
    if not await preflight_checks():
        raise RuntimeError("Pre-flight checks failed")

    # Initialize components
    fee_model = FeeModelSettings()

    # Load completed IDs for resume
    completed_ids = load_completed_ids(output_file) if resume else set()
    if completed_ids:
        logger.info(f"Resuming: {len(completed_ids)} examples already processed")

    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Define symbols and parameters for data generation
    symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "ADA/USDT", "DOT/USDT"]
    timeframes = ["1h", "4h"]
    lookback_bars = 100
    stride_bars = 50  # How far back to walk for each window
    windows_per_symbol = 100  # ~100 windows per symbol/timeframe combo

    # Calculate total examples
    total_possible = len(symbols) * len(timeframes) * windows_per_symbol
    target_count = min(limit, total_possible) if limit else total_possible

    generated_count = 0
    skipped_count = 0
    error_count = 0

    # Acquire training lock for the entire generation process
    with acquire_training_lock():
        logger.info("Training lock acquired")

        async with MarketDataService() as market_data:
            async with OllamaClient() as ollama:
                # Progress bar
                pbar = tqdm(
                    total=target_count,
                    desc="Generating SFT examples",
                    unit="ex",
                )

                for symbol in symbols:
                    for timeframe in timeframes:
                        # Fetch latest timestamp for this pair
                        try:
                            latest_df = await market_data.fetch_ohlcv(
                                symbol, timeframe, lookback_bars=10
                            )
                            if latest_df.empty:
                                logger.warning(f"No data for {symbol}/{timeframe}")
                                continue
                            current_ts = int(latest_df["timestamp"].iloc[-1])
                        except Exception as e:
                            logger.error(f"Failed to fetch latest data: {e}")
                            continue

                        # Calculate bar duration for stride
                        bar_ms = market_data._timeframe_to_ms(timeframe)

                        for window_idx in range(windows_per_symbol):
                            if limit and generated_count >= limit:
                                break

                            # Calculate window end timestamp
                            # Walk back from current, using stride
                            # Need at least 2*lookback for snapshot + forward data
                            offset_ms = (window_idx + 1) * stride_bars * bar_ms
                            window_end_ms = current_ts - offset_ms

                            # Create unique example ID
                            example_id = f"{symbol.replace('/', '_')}_{timeframe}_{window_end_ms}"

                            # Skip if already processed
                            if example_id in completed_ids:
                                skipped_count += 1
                                pbar.update(1)
                                continue

                            try:
                                # Fetch historical window data (point-in-time safe)
                                window = HistoricalWindow(
                                    symbol=symbol,
                                    timeframe=timeframe,
                                    end_timestamp_ms=window_end_ms,
                                    lookback_bars=lookback_bars,
                                    stride_index=window_idx,
                                )

                                snapshot_df = await fetch_window_data(market_data, window)
                                if snapshot_df is None or len(snapshot_df) < lookback_bars * 0.9:
                                    logger.debug(f"Insufficient data for window {window_idx}")
                                    error_count += 1
                                    pbar.update(1)
                                    continue

                                # Fetch FORWARD data (for verified outcome)
                                # Need data AFTER window_end_ms
                                forward_bars = 24  # 24 bars forward (e.g., 24h for 1h TF)
                                forward_end_ms = window_end_ms + forward_bars * bar_ms

                                forward_df = await market_data.fetch_ohlcv(
                                    symbol, timeframe, lookback_bars=forward_bars + 5
                                )
                                # Filter to forward period only
                                forward_df = forward_df[
                                    (forward_df["timestamp"] > window_end_ms) &
                                    (forward_df["timestamp"] <= forward_end_ms)
                                ]

                                if forward_df.empty or len(forward_df) < forward_bars * 0.8:
                                    logger.debug(f"Insufficient forward data for window {window_idx}")
                                    error_count += 1
                                    pbar.update(1)
                                    continue

                                # Calculate verified outcome
                                entry_price = float(snapshot_df["close"].iloc[-1])
                                exit_price = float(forward_df["close"].iloc[-1])
                                gross_return_pct = ((exit_price / entry_price) - 1) * 100

                                # Apply fee model
                                holding_periods = len(forward_df) * bar_ms / (8 * 3600 * 1000)
                                net_return_pct = fee_model.net_return(gross_return_pct, holding_periods)

                                # Determine direction from NET return (after fees)
                                outcome = determine_direction(net_return_pct / 100)

                                # Build market snapshot
                                market_snapshot = build_market_snapshot(
                                    df=snapshot_df,
                                    symbol=symbol,
                                    timeframe=timeframe,
                                    fee_model=fee_model,
                                )

                                # Build distillation prompt
                                distillation_prompt = build_distillation_prompt(
                                    market_snapshot=market_snapshot,
                                    outcome=outcome,
                                    net_return_pct=net_return_pct,
                                    fee_model=fee_model,
                                )

                                # Call critic model for reasoning trace
                                response = await ollama.generate(
                                    model=CRITIC_MODEL,
                                    prompt=distillation_prompt,
                                    options=GENERATION_OPTIONS,
                                )

                                reasoning_text = response.get("response", "")
                                reasoning_trace = parse_reasoning_trace(reasoning_text)

                                if reasoning_trace is None:
                                    logger.warning(f"Invalid reasoning trace for {example_id}")
                                    error_count += 1
                                    pbar.update(1)
                                    continue

                                # Create SFT example
                                example = SFTExample(
                                    example_id=example_id,
                                    created_at=datetime.now(UTC).isoformat(),
                                    symbol=symbol,
                                    timeframe=timeframe,
                                    timestamp_ms=window_end_ms,
                                    market_snapshot=market_snapshot,
                                    verified_outcome=outcome,
                                    net_return_pct=net_return_pct,
                                    reasoning_trace=reasoning_trace,
                                )

                                # Append to JSONL (incremental save)
                                with open(output_file, "a") as f:
                                    f.write(json.dumps(example.to_dict()) + "\n")

                                generated_count += 1
                                pbar.update(1)
                                pbar.set_postfix(
                                    gen=generated_count,
                                    skip=skipped_count,
                                    err=error_count,
                                )

                            except Exception as e:
                                logger.error(f"Error processing window {window_idx}: {e}")
                                error_count += 1
                                pbar.update(1)
                                continue

                        if limit and generated_count >= limit:
                            break
                    if limit and generated_count >= limit:
                        break

                # Explicitly unload model after generation
                await ollama.unload_current()
                pbar.close()

    logger.info(
        "SFT generation complete",
        generated=generated_count,
        skipped=skipped_count,
        errors=error_count,
    )

    return generated_count


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate SFT training data with reasoning traces",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/sft_training_data.jsonl"),
        help="Output JSONL file path",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of examples to generate (default: 2000+)",
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

    # Configure logging
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    )

    # Run generation
    try:
        count = asyncio.run(
            generate_sft_dataset(
                output_file=args.output,
                limit=args.limit,
                resume=args.resume,
            )
        )
        logger.info(f"Generated {count} SFT examples to {args.output}")
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Generation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
