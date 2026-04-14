"""
Deferred accuracy verification for signals.

Signals cannot be verified immediately - we must wait for the next bar to close.
This module queues signals for later verification and tracks accuracy.
"""

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from signals.signal_models import (
    AccuracyRecord,
    get_timeframe_duration_ms,
    get_verification_horizon_bars,
)

if TYPE_CHECKING:
    from signals.signal_models import Signal


ACCURACY_LOG_PATH = Path("signals/accuracy.jsonl")
PENDING_PATH = Path("signals/pending_verification.jsonl")

# Fee threshold for determining "meaningful" move (matches fee model)
FEE_THRESHOLD_PCT = 0.06  # ~0.06% round-trip for Binance


def queue_for_verification(signal: "Signal", entry_price: float) -> None:
    """
    Queue signal for future accuracy verification.

    The signal will be verified after the next bar closes.

    Args:
        signal: Signal to queue
        entry_price: Price at signal generation time
    """
    PENDING_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Calculate when to verify (after next bar closes)
    horizon_bars = get_verification_horizon_bars(signal.timeframe)
    bar_duration_ms = get_timeframe_duration_ms(signal.timeframe)
    verify_after_ms = int(signal.timestamp.timestamp() * 1000) + (horizon_bars * bar_duration_ms)

    pending = {
        "signal_timestamp": signal.timestamp.isoformat(),
        "symbol": signal.symbol,
        "timeframe": signal.timeframe,
        "predicted_direction": signal.final_direction or signal.direction,
        "signal_confidence": signal.confidence,
        "entry_price": entry_price,
        "verify_after_ms": verify_after_ms,
    }

    with open(PENDING_PATH, "a") as f:
        f.write(json.dumps(pending) + "\n")

    logger.debug(
        "Signal queued for verification",
        symbol=signal.symbol,
        verify_after_ms=verify_after_ms,
    )


async def process_pending_verifications(market_data_service) -> list[AccuracyRecord]:
    """
    Check pending verifications and log accuracy.

    Processes any signals whose verification time has passed,
    fetches current price, and logs accuracy results.

    Args:
        market_data_service: MarketDataService instance for fetching prices

    Returns:
        List of AccuracyRecord objects for verified signals
    """
    if not PENDING_PATH.exists():
        return []

    # Read pending records
    pending_records = []
    with open(PENDING_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    pending_records.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning("Skipping invalid JSON line in pending verification")

    if not pending_records:
        return []

    current_time_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    remaining = []
    verified = []

    for record in pending_records:
        if current_time_ms < record["verify_after_ms"]:
            # Not yet time to verify
            remaining.append(record)
            continue

        # Time to verify
        try:
            accuracy = await _verify_signal_accuracy(record, market_data_service)
            _log_accuracy(accuracy)
            verified.append(accuracy)

            logger.info(
                "Signal accuracy verified",
                symbol=record["symbol"],
                predicted=record["predicted_direction"],
                actual=accuracy.actual_direction,
                correct=accuracy.correct,
            )

        except Exception as e:
            logger.error(
                "Verification failed",
                symbol=record["symbol"],
                error=str(e),
            )
            # Keep for retry
            remaining.append(record)

    # Rewrite pending file with remaining records
    with open(PENDING_PATH, "w") as f:
        for r in remaining:
            f.write(json.dumps(r) + "\n")

    return verified


async def _verify_signal_accuracy(
    pending: dict,
    market_data_service,
) -> AccuracyRecord:
    """
    Fetch outcome and compare to prediction.

    Args:
        pending: Pending verification record
        market_data_service: MarketDataService for fetching prices

    Returns:
        AccuracyRecord with verification results
    """
    # Fetch latest bar
    df = await market_data_service.fetch_ohlcv(
        pending["symbol"],
        pending["timeframe"],
        lookback_bars=2,
    )

    # Use the latest close price
    exit_price = float(df["close"].iloc[-1])
    entry_price = pending["entry_price"]

    # Calculate actual return
    actual_return_pct = ((exit_price / entry_price) - 1) * 100

    # Determine actual direction based on fee threshold
    if actual_return_pct > FEE_THRESHOLD_PCT:
        actual_direction = "LONG"
    elif actual_return_pct < -FEE_THRESHOLD_PCT:
        actual_direction = "SHORT"
    else:
        actual_direction = "FLAT"

    # Check if prediction was correct
    predicted = pending["predicted_direction"]
    correct = predicted == actual_direction

    return AccuracyRecord(
        signal_timestamp=pending["signal_timestamp"],
        symbol=pending["symbol"],
        timeframe=pending["timeframe"],
        predicted_direction=predicted,
        actual_direction=actual_direction,
        correct=correct,
        signal_confidence=pending["signal_confidence"],
        entry_price=entry_price,
        exit_price=exit_price,
        actual_return_pct=actual_return_pct,
        verified_at=datetime.now(timezone.utc).isoformat(),
    )


def _log_accuracy(record: AccuracyRecord) -> None:
    """
    Log accuracy record to JSONL file.

    Args:
        record: AccuracyRecord to log
    """
    ACCURACY_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(ACCURACY_LOG_PATH, "a") as f:
        f.write(json.dumps(asdict(record)) + "\n")


def get_accuracy_summary() -> dict:
    """
    Calculate accuracy summary from logged records.

    Returns:
        Summary dict with total, correct, accuracy_pct, by_direction stats
    """
    if not ACCURACY_LOG_PATH.exists():
        return {
            "total": 0,
            "correct": 0,
            "accuracy_pct": 0.0,
            "by_direction": {},
        }

    records = []
    with open(ACCURACY_LOG_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if not records:
        return {
            "total": 0,
            "correct": 0,
            "accuracy_pct": 0.0,
            "by_direction": {},
        }

    total = len(records)
    correct = sum(1 for r in records if r.get("correct", False))
    accuracy_pct = (correct / total * 100) if total > 0 else 0.0

    # Breakdown by direction
    by_direction = {}
    for r in records:
        predicted = r.get("predicted_direction", "UNKNOWN")
        if predicted not in by_direction:
            by_direction[predicted] = {"total": 0, "correct": 0}
        by_direction[predicted]["total"] += 1
        if r.get("correct", False):
            by_direction[predicted]["correct"] += 1

    # Add accuracy percentages
    for direction in by_direction:
        t = by_direction[direction]["total"]
        c = by_direction[direction]["correct"]
        by_direction[direction]["accuracy_pct"] = (c / t * 100) if t > 0 else 0.0

    return {
        "total": total,
        "correct": correct,
        "accuracy_pct": accuracy_pct,
        "by_direction": by_direction,
    }


def get_recent_accuracy(n: int = 100) -> dict:
    """
    Calculate accuracy for the most recent N signals.

    Args:
        n: Number of recent signals to consider

    Returns:
        Summary dict with accuracy for recent signals
    """
    if not ACCURACY_LOG_PATH.exists():
        return {
            "total": 0,
            "correct": 0,
            "accuracy_pct": 0.0,
        }

    records = []
    with open(ACCURACY_LOG_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    # Take most recent n
    recent = records[-n:] if len(records) > n else records

    total = len(recent)
    correct = sum(1 for r in recent if r.get("correct", False))
    accuracy_pct = (correct / total * 100) if total > 0 else 0.0

    return {
        "total": total,
        "correct": correct,
        "accuracy_pct": accuracy_pct,
    }
