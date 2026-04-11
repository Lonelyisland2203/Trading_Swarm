"""
Thread-safe JSONL logging for signals.

Appends signal records to signals/signal_log.jsonl with proper locking.
"""

import json
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from signals.signal_models import Signal


SIGNAL_LOG_PATH = Path("signals/signal_log.jsonl")

# Thread-safe write lock
_write_lock = Lock()


def log_signal(
    signal: "Signal",
    executed: bool,
    trade_reason: str | None = None,
) -> None:
    """
    Append signal to log file (thread-safe).

    Creates the signals directory if it doesn't exist.
    Appends a single JSON line per signal.

    Args:
        signal: Signal object to log
        executed: Whether the signal was executed
        trade_reason: Reason for trade decision (accept/reject)
    """
    SIGNAL_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    entry = {
        "timestamp": signal.timestamp.isoformat(),
        "symbol": signal.symbol,
        "timeframe": signal.timeframe,
        "direction": signal.direction,
        "confidence": signal.confidence,
        "persona": signal.persona,
        "market_regime": signal.market_regime,
        "reasoning": signal.reasoning,
        "current_price": signal.current_price,
        # Critic fields
        "critic_score": signal.critic_score,
        "critic_recommendation": signal.critic_recommendation,
        "critic_override": signal.critic_override,
        # Final outcome
        "final_direction": signal.final_direction or signal.direction,
        "executed": executed,
        "trade_decision_reason": trade_reason,
    }

    with _write_lock:
        with open(SIGNAL_LOG_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")

    logger.debug(
        "Signal logged",
        symbol=signal.symbol,
        direction=signal.final_direction or signal.direction,
        executed=executed,
    )


def read_signal_log(limit: int | None = None) -> list[dict]:
    """
    Read signal log entries.

    Args:
        limit: Maximum number of entries to return (most recent first)

    Returns:
        List of signal log entries
    """
    if not SIGNAL_LOG_PATH.exists():
        return []

    entries = []
    with open(SIGNAL_LOG_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning("Skipping invalid JSON line in signal log")

    # Return most recent first
    entries.reverse()

    if limit:
        entries = entries[:limit]

    return entries


def get_signal_count() -> int:
    """
    Get total number of signals in log.

    Returns:
        Number of signal entries
    """
    if not SIGNAL_LOG_PATH.exists():
        return 0

    count = 0
    with open(SIGNAL_LOG_PATH) as f:
        for line in f:
            if line.strip():
                count += 1

    return count


def get_signals_since(since: datetime) -> list[dict]:
    """
    Get signals logged since a specific time.

    Args:
        since: Timestamp to filter from

    Returns:
        List of signal entries logged since the specified time
    """
    entries = read_signal_log()

    filtered = []
    for entry in entries:
        try:
            ts = datetime.fromisoformat(entry["timestamp"])
            if ts >= since:
                filtered.append(entry)
        except (KeyError, ValueError):
            continue

    return filtered
