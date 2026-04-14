"""
Signal verification — closes the feedback loop by verifying signal outcomes.

Reads unverified signals from signal_log.jsonl, fetches actual outcomes,
computes fee-adjusted returns, and logs verified results to verified_results.jsonl.

This creates the self-improvement loop:
signals → verification → new training data → GRPO retraining → better signals
"""

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger
from scipy import stats

from config.fee_model import FeeModelSettings
from signals.signal_logger import SIGNAL_LOG_PATH
from verifier.constants import HORIZON_BARS, compute_holding_periods_8h

if TYPE_CHECKING:
    from data.market_data import MarketDataService


# Paths
VERIFIED_RESULTS_PATH = Path("signals/verified_results.jsonl")
TRAINING_TRIGGER_PATH = Path("signals/training_trigger.json")

# Thread-safe write lock
_write_lock = Lock()

# Training threshold
MIN_SIGNALS_FOR_TRAINING = 200


@dataclass(slots=True, frozen=True)
class VerifiedResult:
    """
    Verified signal outcome with fee-adjusted metrics.
    """

    # Signal identification
    signal_timestamp: str  # ISO format
    symbol: str
    timeframe: str

    # Prediction
    predicted_direction: str  # LONG/SHORT/FLAT
    signal_confidence: float
    market_regime: str

    # Outcome
    entry_price: float
    exit_price: float
    gross_return_pct: float
    net_return_pct: float
    actual_direction: str  # LONG/SHORT/FLAT based on net return

    # Verification
    correct: bool
    verified_at: str  # ISO format
    horizon_bars: int

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass(slots=True)
class VerificationStats:
    """
    Aggregated verification statistics.
    """

    total_verified: int
    correct_count: int
    accuracy_pct: float

    # IC (Information Coefficient) - correlation between confidence and returns
    ic: float
    ic_pvalue: float

    # Sharpe ratio of predictions
    sharpe_ratio: float

    # Regime-stratified accuracy
    accuracy_by_regime: dict[str, dict]

    # False signal rates
    false_bullish_rate: float  # Predicted LONG, was SHORT
    false_bearish_rate: float  # Predicted SHORT, was LONG

    # Net P&L
    total_net_return_pct: float
    avg_net_return_pct: float

    # Training trigger
    ready_for_training: bool
    signals_since_last_training: int


def get_verification_horizon_bars(timeframe: str) -> int:
    """
    Get timeframe-adaptive verification horizon in bars.

    Uses the same horizons as training data generation:
    - 1h → 24 bars (24 hours)
    - 4h → 12 bars (48 hours)
    - 1d → 5 bars (5 days)

    Args:
        timeframe: Timeframe string

    Returns:
        Number of bars to wait before verification
    """
    return HORIZON_BARS.get(timeframe, 24)


def load_unverified_signals() -> list[dict]:
    """
    Load signals that haven't been verified yet.

    A signal is unverified if:
    1. It exists in signal_log.jsonl
    2. It doesn't exist in verified_results.jsonl (by timestamp+symbol key)
    3. Its verification horizon has passed

    Returns:
        List of unverified signal entries ready for verification
    """
    if not SIGNAL_LOG_PATH.exists():
        return []

    # Load all signals
    signals = []
    with open(SIGNAL_LOG_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    signals.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if not signals:
        return []

    # Load already verified signals (create set of keys for O(1) lookup)
    verified_keys = set()
    if VERIFIED_RESULTS_PATH.exists():
        with open(VERIFIED_RESULTS_PATH) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entry = json.loads(line)
                        key = (entry["signal_timestamp"], entry["symbol"])
                        verified_keys.add(key)
                    except (json.JSONDecodeError, KeyError):
                        continue

    # Filter to unverified signals past their horizon
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    unverified = []

    for signal in signals:
        try:
            key = (signal["timestamp"], signal["symbol"])
            if key in verified_keys:
                continue

            # Check if verification horizon has passed
            signal_ts = datetime.fromisoformat(signal["timestamp"])
            signal_ms = int(signal_ts.timestamp() * 1000)

            timeframe = signal.get("timeframe", "1h")
            horizon_bars = get_verification_horizon_bars(timeframe)

            # Convert timeframe to milliseconds per bar
            tf_ms = _timeframe_to_ms(timeframe)
            horizon_ms = horizon_bars * tf_ms

            verify_after_ms = signal_ms + horizon_ms

            if now_ms >= verify_after_ms:
                unverified.append(signal)

        except (KeyError, ValueError) as e:
            logger.warning(f"Skipping malformed signal: {e}")
            continue

    return unverified


def _timeframe_to_ms(timeframe: str) -> int:
    """Convert timeframe string to milliseconds."""
    durations = {
        "1m": 60_000,
        "5m": 300_000,
        "15m": 900_000,
        "1h": 3_600_000,
        "4h": 14_400_000,
        "1d": 86_400_000,
    }
    return durations.get(timeframe, 3_600_000)


async def verify_signal(
    signal: dict,
    market_data_service: "MarketDataService",
    fee_model: FeeModelSettings | None = None,
) -> VerifiedResult | None:
    """
    Verify a single signal by fetching actual outcome data.

    Args:
        signal: Signal entry from signal_log.jsonl
        market_data_service: Service for fetching OHLCV data
        fee_model: Fee model for computing net returns

    Returns:
        VerifiedResult if verification successful, None otherwise
    """
    if fee_model is None:
        fee_model = FeeModelSettings()

    try:
        symbol = signal["symbol"]
        timeframe = signal.get("timeframe", "1h")
        signal_ts = datetime.fromisoformat(signal["timestamp"])
        entry_price = signal.get("current_price", 0.0)
        predicted_direction = signal.get("final_direction", signal.get("direction", "FLAT"))
        confidence = signal.get("confidence", 0.5)
        market_regime = signal.get("market_regime", "neutral")

        horizon_bars = get_verification_horizon_bars(timeframe)

        # Fetch OHLCV data for verification window
        signal_ms = int(signal_ts.timestamp() * 1000)
        tf_ms = _timeframe_to_ms(timeframe)

        # Calculate forward horizon end for point-in-time safe fetch
        # We need bars AFTER signal_ms, up to horizon_bars forward
        forward_end_ms = signal_ms + (horizon_bars * tf_ms)

        # Use get_ohlcv_as_of for temporal safety (filters by bar close time)
        df = await market_data_service.get_ohlcv_as_of(
            symbol=symbol,
            timeframe=timeframe,
            as_of=forward_end_ms,
            lookback_bars=horizon_bars + 5,
        )

        # Filter to only forward bars (after signal timestamp)
        if df is not None and not df.empty:
            df = df[df["timestamp"] > signal_ms].reset_index(drop=True)

        if df is None or len(df) < horizon_bars:
            logger.warning(
                f"Insufficient data for verification: {symbol} {timeframe}, "
                f"got {len(df) if df is not None else 0} bars, need {horizon_bars}"
            )
            return None

        # Get exit price (close of the horizon bar)
        exit_price = float(df.iloc[horizon_bars - 1]["close"])

        if entry_price <= 0:
            # Use first bar's close as entry if not provided
            entry_price = float(df.iloc[0]["close"])

        # Compute gross return
        gross_return_pct = ((exit_price - entry_price) / entry_price) * 100

        # Compute holding period for fee calculation
        holding_periods_8h = compute_holding_periods_8h(timeframe, horizon_bars)

        # Compute net return after fees
        net_return_pct = fee_model.net_return(gross_return_pct, holding_periods_8h)

        # Classify actual direction based on net return
        fee_threshold = fee_model.minimum_profitable_return_pct(holding_periods_8h)
        if net_return_pct > fee_threshold:
            actual_direction = "LONG"
        elif net_return_pct < -fee_threshold:
            actual_direction = "SHORT"
        else:
            actual_direction = "FLAT"

        # Determine correctness
        if predicted_direction == "FLAT":
            # FLAT is correct if actual is also FLAT or move was small
            correct = actual_direction == "FLAT" or abs(net_return_pct) < fee_threshold
        else:
            correct = predicted_direction == actual_direction

        return VerifiedResult(
            signal_timestamp=signal["timestamp"],
            symbol=symbol,
            timeframe=timeframe,
            predicted_direction=predicted_direction,
            signal_confidence=confidence,
            market_regime=market_regime,
            entry_price=entry_price,
            exit_price=exit_price,
            gross_return_pct=gross_return_pct,
            net_return_pct=net_return_pct,
            actual_direction=actual_direction,
            correct=correct,
            verified_at=datetime.now(timezone.utc).isoformat(),
            horizon_bars=horizon_bars,
        )

    except Exception as e:
        logger.error(f"Error verifying signal: {e}")
        return None


def save_verified_result(result: VerifiedResult) -> None:
    """
    Append verified result to JSONL file (thread-safe).

    Args:
        result: Verified result to save
    """
    VERIFIED_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    with _write_lock:
        with open(VERIFIED_RESULTS_PATH, "a") as f:
            f.write(json.dumps(result.to_dict()) + "\n")


def load_verified_results() -> list[dict]:
    """
    Load all verified results.

    Returns:
        List of verified result dictionaries
    """
    if not VERIFIED_RESULTS_PATH.exists():
        return []

    results = []
    with open(VERIFIED_RESULTS_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    return results


def compute_verification_stats(
    results: list[dict] | None = None,
) -> VerificationStats:
    """
    Compute aggregated verification statistics.

    Args:
        results: Optional list of verified results. If None, loads from file.

    Returns:
        VerificationStats with all metrics
    """
    if results is None:
        results = load_verified_results()

    if not results:
        return VerificationStats(
            total_verified=0,
            correct_count=0,
            accuracy_pct=0.0,
            ic=0.0,
            ic_pvalue=1.0,
            sharpe_ratio=0.0,
            accuracy_by_regime={},
            false_bullish_rate=0.0,
            false_bearish_rate=0.0,
            total_net_return_pct=0.0,
            avg_net_return_pct=0.0,
            ready_for_training=False,
            signals_since_last_training=0,
        )

    total = len(results)
    correct = sum(1 for r in results if r.get("correct", False))
    accuracy_pct = (correct / total) * 100 if total > 0 else 0.0

    # IC: correlation between confidence and actual returns (sign-adjusted)
    confidences = []
    adjusted_returns = []
    net_returns = []

    for r in results:
        conf = r.get("signal_confidence", 0.5)
        net_ret = r.get("net_return_pct", 0.0)
        pred_dir = r.get("predicted_direction", "FLAT")

        net_returns.append(net_ret)

        # Sign-adjust return based on predicted direction
        if pred_dir == "LONG":
            adjusted_returns.append(net_ret)
        elif pred_dir == "SHORT":
            adjusted_returns.append(-net_ret)
        else:
            adjusted_returns.append(0.0)

        confidences.append(conf)

    # Compute IC using Spearman correlation
    ic = 0.0
    ic_pvalue = 1.0
    if len(confidences) >= 3 and np.std(adjusted_returns) > 0:
        try:
            ic, ic_pvalue = stats.spearmanr(confidences, adjusted_returns)
            if np.isnan(ic):
                ic = 0.0
        except Exception:
            pass

    # Compute Sharpe ratio
    sharpe_ratio = 0.0
    if net_returns and np.std(net_returns) > 0:
        sharpe_ratio = np.mean(net_returns) / np.std(net_returns) * np.sqrt(252)

    # Regime-stratified accuracy
    regime_stats: dict[str, dict] = {}
    for r in results:
        regime = r.get("market_regime", "neutral")
        if regime not in regime_stats:
            regime_stats[regime] = {"total": 0, "correct": 0}
        regime_stats[regime]["total"] += 1
        if r.get("correct", False):
            regime_stats[regime]["correct"] += 1

    accuracy_by_regime = {}
    for regime, stats_dict in regime_stats.items():
        total_regime = stats_dict["total"]
        correct_regime = stats_dict["correct"]
        accuracy_by_regime[regime] = {
            "total": total_regime,
            "correct": correct_regime,
            "accuracy_pct": (correct_regime / total_regime) * 100 if total_regime > 0 else 0.0,
        }

    # False signal rates
    false_bullish = 0  # Predicted LONG, was SHORT
    false_bearish = 0  # Predicted SHORT, was LONG
    total_long = 0
    total_short = 0

    for r in results:
        pred = r.get("predicted_direction", "FLAT")
        actual = r.get("actual_direction", "FLAT")

        if pred == "LONG":
            total_long += 1
            if actual == "SHORT":
                false_bullish += 1
        elif pred == "SHORT":
            total_short += 1
            if actual == "LONG":
                false_bearish += 1

    false_bullish_rate = (false_bullish / total_long) * 100 if total_long > 0 else 0.0
    false_bearish_rate = (false_bearish / total_short) * 100 if total_short > 0 else 0.0

    # Net P&L
    total_net_return = sum(net_returns)
    avg_net_return = np.mean(net_returns) if net_returns else 0.0

    # Training trigger check
    signals_since_last = _get_signals_since_last_training()
    ready_for_training = signals_since_last >= MIN_SIGNALS_FOR_TRAINING

    return VerificationStats(
        total_verified=total,
        correct_count=correct,
        accuracy_pct=accuracy_pct,
        ic=ic,
        ic_pvalue=ic_pvalue,
        sharpe_ratio=sharpe_ratio,
        accuracy_by_regime=accuracy_by_regime,
        false_bullish_rate=false_bullish_rate,
        false_bearish_rate=false_bearish_rate,
        total_net_return_pct=total_net_return,
        avg_net_return_pct=avg_net_return,
        ready_for_training=ready_for_training,
        signals_since_last_training=signals_since_last,
    )


def _get_signals_since_last_training() -> int:
    """Get count of verified signals since last training trigger."""
    results = load_verified_results()

    # Check for last training timestamp
    if TRAINING_TRIGGER_PATH.exists():
        try:
            with open(TRAINING_TRIGGER_PATH) as f:
                trigger_data = json.load(f)
                last_training_ts = trigger_data.get("last_training_timestamp")
                if last_training_ts:
                    last_ts = datetime.fromisoformat(last_training_ts)
                    count = 0
                    for r in results:
                        verified_at = datetime.fromisoformat(r["verified_at"])
                        if verified_at > last_ts:
                            count += 1
                    return count
        except (json.JSONDecodeError, KeyError, ValueError):
            pass

    # No trigger file, count all results
    return len(results)


def check_training_trigger() -> dict:
    """
    Check if enough verified signals have accumulated for retraining.

    Returns:
        Dict with trigger status and counts
    """
    signals_since_last = _get_signals_since_last_training()
    ready = signals_since_last >= MIN_SIGNALS_FOR_TRAINING

    return {
        "ready": ready,
        "signals_since_last_training": signals_since_last,
        "threshold": MIN_SIGNALS_FOR_TRAINING,
        "message": (
            f"Ready for GRPO retraining! {signals_since_last} verified signals available."
            if ready
            else f"Need {MIN_SIGNALS_FOR_TRAINING - signals_since_last} more verified signals for retraining."
        ),
    }


def mark_training_triggered() -> None:
    """
    Mark that training was triggered at current time.

    Called after successful GRPO training to reset the counter.
    """
    TRAINING_TRIGGER_PATH.parent.mkdir(parents=True, exist_ok=True)

    trigger_data = {
        "last_training_timestamp": datetime.now(timezone.utc).isoformat(),
        "triggered_at": datetime.now(timezone.utc).isoformat(),
    }

    with open(TRAINING_TRIGGER_PATH, "w") as f:
        json.dump(trigger_data, f, indent=2)

    logger.info("Training trigger marked", timestamp=trigger_data["triggered_at"])


def export_for_training(
    output_path: Path | str,
    min_confidence: float = 0.0,
) -> int:
    """
    Export verified results as training data for GRPO.

    Only exports signals since last training trigger.

    Args:
        output_path: Path to output JSONL file
        min_confidence: Minimum signal confidence to include

    Returns:
        Number of examples exported
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = load_verified_results()

    # Filter by last training timestamp
    if TRAINING_TRIGGER_PATH.exists():
        try:
            with open(TRAINING_TRIGGER_PATH) as f:
                trigger_data = json.load(f)
                last_ts = trigger_data.get("last_training_timestamp")
                if last_ts:
                    last_dt = datetime.fromisoformat(last_ts)
                    results = [
                        r for r in results if datetime.fromisoformat(r["verified_at"]) > last_dt
                    ]
        except (json.JSONDecodeError, KeyError, ValueError):
            pass

    # Filter by confidence
    results = [r for r in results if r.get("signal_confidence", 0) >= min_confidence]

    # Export as training format
    count = 0
    with open(output_path, "w") as f:
        for r in results:
            training_example = {
                "symbol": r["symbol"],
                "timeframe": r["timeframe"],
                "timestamp_ms": int(
                    datetime.fromisoformat(r["signal_timestamp"]).timestamp() * 1000
                ),
                "actual_direction": r["actual_direction"],
                "gross_return_pct": r["gross_return_pct"],
                "net_return_pct": r["net_return_pct"],
                "market_regime": r["market_regime"],
                "confidence": r["signal_confidence"],
            }
            f.write(json.dumps(training_example) + "\n")
            count += 1

    logger.info(f"Exported {count} verified signals for training", output=str(output_path))
    return count


def format_daily_summary(stats: VerificationStats) -> str:
    """
    Format verification stats as a daily summary string.

    Args:
        stats: Computed verification statistics

    Returns:
        Formatted summary string
    """
    lines = [
        "=" * 60,
        "SIGNAL VERIFICATION SUMMARY",
        "=" * 60,
        "",
        f"Total Verified: {stats.total_verified}",
        f"Correct: {stats.correct_count} ({stats.accuracy_pct:.1f}%)",
        "",
        "--- Performance Metrics ---",
        f"IC (Info Coefficient): {stats.ic:.4f} (p={stats.ic_pvalue:.4f})",
        f"Sharpe Ratio: {stats.sharpe_ratio:.2f}",
        f"Total Net Return: {stats.total_net_return_pct:.2f}%",
        f"Avg Net Return: {stats.avg_net_return_pct:.4f}%",
        "",
        "--- False Signal Rates ---",
        f"False Bullish: {stats.false_bullish_rate:.1f}%",
        f"False Bearish: {stats.false_bearish_rate:.1f}%",
        "",
        "--- Regime-Stratified Accuracy ---",
    ]

    for regime, regime_data in stats.accuracy_by_regime.items():
        lines.append(
            f"  {regime}: {regime_data['correct']}/{regime_data['total']} "
            f"({regime_data['accuracy_pct']:.1f}%)"
        )

    lines.extend(
        [
            "",
            "--- Training Status ---",
            f"Signals Since Last Training: {stats.signals_since_last_training}",
            f"Training Threshold: {MIN_SIGNALS_FOR_TRAINING}",
            f"Ready for Training: {'YES' if stats.ready_for_training else 'NO'}",
            "",
            "=" * 60,
        ]
    )

    return "\n".join(lines)
