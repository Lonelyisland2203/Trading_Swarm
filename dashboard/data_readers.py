"""
Centralized data access for dashboard.

All functions handle missing files gracefully (return empty lists/defaults).
"""

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


# Default paths (can be patched in tests)
SIGNAL_LOG_PATH = Path("signals/signal_log.jsonl")
ORDER_LOG_PATH = Path("execution/order_log.jsonl")
AUTORESEARCH_RESULTS_PATH = Path("autoresearch/results.tsv")
HEALTH_STATUS_PATH = Path("dashboard/health_status.json")


def read_signal_log(limit: int = 50) -> list[dict[str, Any]]:
    """
    Read signal log entries (most recent first).

    Args:
        limit: Maximum entries to return (default 50)

    Returns:
        List of signal log entries, empty if file missing
    """
    if not SIGNAL_LOG_PATH.exists():
        return []

    entries = []
    try:
        with open(SIGNAL_LOG_PATH) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    except OSError:
        return []

    # Most recent first
    entries.reverse()
    return entries[:limit]


def read_order_log(limit: int = 100) -> list[dict[str, Any]]:
    """
    Read order log entries.

    Args:
        limit: Maximum entries to return (default 100)

    Returns:
        List of order log entries, empty if file missing
    """
    if not ORDER_LOG_PATH.exists():
        return []

    entries = []
    try:
        with open(ORDER_LOG_PATH) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    except OSError:
        return []

    return entries[-limit:] if limit else entries


def read_autoresearch_results() -> list[dict[str, Any]]:
    """
    Parse autoresearch/results.tsv into JSON array.

    Returns:
        List of experiment results, empty if file missing
    """
    if not AUTORESEARCH_RESULTS_PATH.exists():
        return []

    results = []
    try:
        with open(AUTORESEARCH_RESULTS_PATH, newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                # Convert numeric fields
                for key in ["sharpe_net", "ic", "brier", "accuracy", "false_bullish_rate"]:
                    if key in row and row[key]:
                        try:
                            row[key] = float(row[key])
                        except ValueError:
                            pass
                results.append(row)
    except OSError:
        return []

    return results


def read_health_status() -> dict[str, Any]:
    """
    Read watchdog health status.

    Returns:
        Health status dict with defaults if file missing
    """
    default = {
        "watchdog_heartbeat": None,
        "positions_count": 0,
        "daily_pnl_usd": 0.0,
        "max_daily_loss_triggered": False,
        "status": "unknown",
    }

    if not HEALTH_STATUS_PATH.exists():
        return default

    try:
        with open(HEALTH_STATUS_PATH) as f:
            data = json.load(f)
            return {**default, **data}
    except (OSError, json.JSONDecodeError):
        return default


def compute_equity_curve(orders: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Compute cumulative P&L from order log.

    Args:
        orders: List of order entries with 'timestamp' and 'pnl' fields

    Returns:
        List of {timestamp, cumulative_pnl}
    """
    if not orders:
        return []

    curve = []
    cumulative = 0.0

    for order in orders:
        pnl = order.get("pnl", 0.0)
        if pnl is None:
            pnl = 0.0
        cumulative += pnl

        curve.append(
            {
                "timestamp": order.get("timestamp", ""),
                "cumulative_pnl": cumulative,
            }
        )

    return curve


def compute_rolling_sharpe(
    equity_curve: list[dict[str, Any]], window: int = 30
) -> list[dict[str, Any]]:
    """
    Compute rolling Sharpe ratio from equity curve.

    Args:
        equity_curve: List of {timestamp, cumulative_pnl}
        window: Rolling window size (default 30)

    Returns:
        List of {timestamp, sharpe} for points with full window
    """
    if len(equity_curve) < window:
        return []

    # Extract cumulative P&L values
    pnls = [point["cumulative_pnl"] for point in equity_curve]

    # Compute returns (differences)
    returns = np.diff(pnls)

    result = []
    for i in range(window - 1, len(returns)):
        window_returns = returns[i - window + 1 : i + 1]

        mean_ret = np.mean(window_returns)
        std_ret = np.std(window_returns)

        # Sharpe = mean / std (annualized would multiply by sqrt(252))
        # Simple daily Sharpe here
        sharpe = mean_ret / std_ret if std_ret > 0 else 0.0

        # Index into equity_curve: returns[i] corresponds to equity_curve[i+1]
        result.append(
            {
                "timestamp": equity_curve[i + 1]["timestamp"],
                "sharpe": float(sharpe),
            }
        )

    return result


def compute_drawdown(equity_curve: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Compute drawdown from peak at each point.

    Args:
        equity_curve: List of {timestamp, cumulative_pnl}

    Returns:
        List of {timestamp, drawdown_pct}
    """
    if not equity_curve:
        return []

    result = []
    peak = float("-inf")

    for point in equity_curve:
        value = point["cumulative_pnl"]

        if value > peak:
            peak = value

        if peak > 0:
            drawdown_pct = (peak - value) / peak
        else:
            drawdown_pct = 0.0

        result.append(
            {
                "timestamp": point["timestamp"],
                "drawdown_pct": drawdown_pct,
            }
        )

    return result


def compute_win_rate(orders: list[dict[str, Any]]) -> float:
    """
    Compute win rate from order log.

    Args:
        orders: List of order entries with 'pnl' field

    Returns:
        Win rate as decimal (0.0-1.0)
    """
    if not orders:
        return 0.0

    wins = sum(1 for o in orders if (o.get("pnl") or 0) > 0)
    total = len([o for o in orders if o.get("pnl") is not None])

    return wins / total if total > 0 else 0.0


def compute_daily_pnl(orders: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Aggregate P&L by day.

    Args:
        orders: List of order entries

    Returns:
        List of {date, pnl} sorted by date
    """
    if not orders:
        return []

    daily: dict[str, float] = {}

    for order in orders:
        ts = order.get("timestamp", "")
        pnl = order.get("pnl", 0.0) or 0.0

        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            date_str = dt.strftime("%Y-%m-%d")
        except (ValueError, AttributeError):
            continue

        daily[date_str] = daily.get(date_str, 0.0) + pnl

    return [{"date": k, "pnl": v} for k, v in sorted(daily.items())]
