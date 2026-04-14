#!/usr/bin/env python3
"""
Autonomous hyperparameter search loop for XGBoost signal optimization.

Karpathy-pattern autoresearch: one file to edit (xgboost_config.py),
one eval script (xgboost_eval.py, read-only), one metric to optimize (sharpe_net).

Usage:
    # Run indefinitely until Ctrl+C or STOP file
    python run_autoresearch.py

    # Run at most 10 experiments
    python run_autoresearch.py --max-experiments 10

    # Run for at most 8 hours (overnight)
    python run_autoresearch.py --time-budget-hours 8

    # Dry run (don't actually modify config)
    python run_autoresearch.py --dry-run

    # Optimize different metric
    python run_autoresearch.py --metric ic
"""

import argparse
import csv
import json
import re
import signal as sys_signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

from loguru import logger


# =============================================================================
# Configuration
# =============================================================================

# File paths
XGBOOST_CONFIG_PATH = Path("signals/xgboost_config.py")
RESULTS_TSV_PATH = Path("autoresearch/results.tsv")
STOP_FILE_PATH = Path("STOP")

# Improvement thresholds
IMPROVEMENT_THRESHOLDS = {
    "sharpe_net": 0.02,
    "ic": 0.005,
    "brier": -0.005,  # Lower is better
}

# Parameter search space for XGBoost config
PARAM_SEARCH_SPACE: dict[str, list[Any]] = {
    # Hyperparameters
    "n_estimators": [50, 75, 100, 125, 150, 200],
    "max_depth": [3, 4, 5, 6, 7, 8],
    "learning_rate": [0.05, 0.08, 0.1, 0.12, 0.15, 0.2],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    # Class weights
    "false_bullish_penalty": [1.2, 1.3, 1.5, 1.7, 2.0, 2.5],
    "false_bearish_penalty": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    # Walk-forward
    "n_folds": [3, 4, 5, 6, 7],
    "gap_bars": [12, 18, 24, 30, 36],
}

# Mapping from param names to config patterns
PARAM_PATTERNS: dict[str, tuple[str, str]] = {
    # (extract_pattern, replacement_pattern)
    "n_estimators": (
        r'"n_estimators":\s*(\d+)',
        '"n_estimators": {value}',
    ),
    "max_depth": (
        r'"max_depth":\s*(\d+)',
        '"max_depth": {value}',
    ),
    "learning_rate": (
        r'"learning_rate":\s*([\d.]+)',
        '"learning_rate": {value}',
    ),
    "subsample": (
        r'"subsample":\s*([\d.]+)',
        '"subsample": {value}',
    ),
    "colsample_bytree": (
        r'"colsample_bytree":\s*([\d.]+)',
        '"colsample_bytree": {value}',
    ),
    "false_bullish_penalty": (
        r'"false_bullish_penalty":\s*([\d.]+)',
        '"false_bullish_penalty": {value}',
    ),
    "false_bearish_penalty": (
        r'"false_bearish_penalty":\s*([\d.]+)',
        '"false_bearish_penalty": {value}',
    ),
    "n_folds": (
        r"n_folds:\s*int\s*=\s*(\d+)",
        "n_folds: int = {value}",
    ),
    "gap_bars": (
        r"gap_bars:\s*int\s*=\s*(\d+)",
        "gap_bars: int = {value}",
    ),
}

# Parameter exploration order (prioritized)
PARAM_ORDER = [
    "max_depth",
    "learning_rate",
    "n_estimators",
    "subsample",
    "colsample_bytree",
    "false_bullish_penalty",
    "false_bearish_penalty",
    "n_folds",
    "gap_bars",
]


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class EvalResult:
    """Evaluation result from xgboost_eval.py."""

    sharpe_net: float
    ic: float
    ic_pvalue: float
    brier: float
    directional_accuracy: float
    false_bullish_rate: float
    false_bearish_rate: float
    num_examples: int
    config_hash: str
    shap_top_5: dict[str, float] = field(default_factory=dict)


@dataclass
class ExperimentResult:
    """Result of a single experiment."""

    timestamp: str
    experiment_id: int
    param_changed: str
    old_value: Any
    new_value: Any
    sharpe_net: float
    ic: float
    brier: float
    accuracy: float
    false_bullish_rate: float
    kept: bool


@dataclass
class ParameterState:
    """Tracks the state of parameter exploration."""

    param_index: int = 0
    value_indices: dict[str, int] = field(default_factory=dict)
    directions: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for param in PARAM_ORDER:
            if param not in self.value_indices:
                self.value_indices[param] = -1
            if param not in self.directions:
                self.directions[param] = 1


# =============================================================================
# Config Manipulation
# =============================================================================


def read_current_config_values() -> dict[str, Any]:
    """Parse current values from xgboost_config.py."""
    if not XGBOOST_CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config file not found: {XGBOOST_CONFIG_PATH}")

    content = XGBOOST_CONFIG_PATH.read_text()
    values: dict[str, Any] = {}

    for param, (pattern, _) in PARAM_PATTERNS.items():
        match = re.search(pattern, content)
        if match:
            raw_value = match.group(1)
            # Convert to appropriate type
            if param in ["n_estimators", "max_depth", "n_folds", "gap_bars"]:
                values[param] = int(raw_value)
            else:
                values[param] = float(raw_value)

    return values


def modify_config_parameter(param: str, new_value: Any) -> str:
    """
    Modify a single parameter in xgboost_config.py.

    Returns:
        Old value as string
    """
    content = XGBOOST_CONFIG_PATH.read_text()

    if param not in PARAM_PATTERNS:
        raise ValueError(f"Unknown parameter: {param}")

    extract_pattern, replace_template = PARAM_PATTERNS[param]

    # Extract old value
    match = re.search(extract_pattern, content)
    old_value = match.group(1) if match else "unknown"

    # Format new value
    if isinstance(new_value, float) and new_value == int(new_value):
        value_str = str(int(new_value))
    else:
        value_str = str(new_value)

    # Create replacement
    replacement = replace_template.format(value=value_str)

    # Do the replacement
    new_content, count = re.subn(extract_pattern, replacement, content, count=1)

    if count == 0:
        raise ValueError(f"Could not find {param} in config file")

    XGBOOST_CONFIG_PATH.write_text(new_content)

    return old_value


# =============================================================================
# Results TSV
# =============================================================================


def read_results_tsv() -> list[ExperimentResult]:
    """Read experiment results from TSV file."""
    results: list[ExperimentResult] = []

    if not RESULTS_TSV_PATH.exists():
        return results

    with open(RESULTS_TSV_PATH, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            try:
                results.append(
                    ExperimentResult(
                        timestamp=row["timestamp"],
                        experiment_id=int(row["experiment_id"]),
                        param_changed=row["change_description"],
                        old_value=row.get("old_value", ""),
                        new_value=row.get("new_value", ""),
                        sharpe_net=float(row["sharpe_net"]),
                        ic=float(row["ic"]),
                        brier=float(row["brier"]),
                        accuracy=float(row["accuracy"]),
                        false_bullish_rate=float(row["false_bullish_rate"]),
                        kept=row["kept_or_reverted"].lower() == "kept",
                    )
                )
            except (KeyError, ValueError) as e:
                logger.warning(f"Skipping malformed row: {e}")

    return results


def append_result(result: ExperimentResult) -> None:
    """Append a single result to the TSV file."""
    RESULTS_TSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    file_exists = RESULTS_TSV_PATH.exists()

    fieldnames = [
        "experiment_id",
        "timestamp",
        "change_description",
        "sharpe_net",
        "ic",
        "brier",
        "accuracy",
        "false_bullish_rate",
        "kept_or_reverted",
    ]

    with open(RESULTS_TSV_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        if not file_exists:
            writer.writeheader()
        writer.writerow(
            {
                "experiment_id": result.experiment_id,
                "timestamp": result.timestamp,
                "change_description": f"{result.param_changed}: {result.old_value}→{result.new_value}",
                "sharpe_net": f"{result.sharpe_net:.6f}",
                "ic": f"{result.ic:.6f}",
                "brier": f"{result.brier:.6f}",
                "accuracy": f"{result.accuracy:.4f}",
                "false_bullish_rate": f"{result.false_bullish_rate:.4f}",
                "kept_or_reverted": "kept" if result.kept else "reverted",
            }
        )


def get_best_metric(results: list[ExperimentResult], metric: str) -> float:
    """Get the best metric value from kept experiments."""
    kept_results = [r for r in results if r.kept]
    if not kept_results:
        return -float("inf") if metric != "brier" else float("inf")

    if metric == "sharpe_net":
        return max(r.sharpe_net for r in kept_results)
    elif metric == "ic":
        return max(r.ic for r in kept_results)
    elif metric == "brier":
        return min(r.brier for r in kept_results)
    else:
        return -float("inf")


# =============================================================================
# Parameter Selection
# =============================================================================


def choose_next_parameter(
    state: ParameterState,
    current_values: dict[str, Any],
    results: list[ExperimentResult],
) -> tuple[str, Any, Any]:
    """
    Choose the next parameter to modify using round-robin with direction tracking.

    Returns:
        Tuple of (param_name, old_value, new_value)
    """
    # Update direction based on last result
    if results:
        last_result = results[-1]
        param = last_result.param_changed.split(":")[0].strip()
        if param in state.directions:
            if not last_result.kept:
                state.directions[param] *= -1

    # Find current value index for each parameter
    for param in PARAM_ORDER:
        if param in current_values:
            search_space = PARAM_SEARCH_SPACE[param]
            current_val = current_values[param]
            try:
                state.value_indices[param] = min(
                    range(len(search_space)),
                    key=lambda i: abs(search_space[i] - current_val),
                )
            except (TypeError, ValueError):
                state.value_indices[param] = 0

    # Try parameters in round-robin order
    attempts = 0
    while attempts < len(PARAM_ORDER):
        param = PARAM_ORDER[state.param_index]
        state.param_index = (state.param_index + 1) % len(PARAM_ORDER)

        search_space = PARAM_SEARCH_SPACE[param]
        current_idx = state.value_indices.get(param, 0)
        direction = state.directions.get(param, 1)

        # Try to find a new value
        new_idx = current_idx + direction
        if 0 <= new_idx < len(search_space):
            old_value = current_values.get(param, search_space[current_idx])
            new_value = search_space[new_idx]
            if new_value != old_value:
                state.value_indices[param] = new_idx
                return param, old_value, new_value

        # Try opposite direction
        new_idx = current_idx - direction
        if 0 <= new_idx < len(search_space):
            old_value = current_values.get(param, search_space[current_idx])
            new_value = search_space[new_idx]
            if new_value != old_value:
                state.value_indices[param] = new_idx
                state.directions[param] = -direction
                return param, old_value, new_value

        attempts += 1

    raise RuntimeError("Exhausted all parameter search space")


# =============================================================================
# Git Operations
# =============================================================================


def git_commit(param: str, old_value: Any, new_value: Any, metric_delta: float) -> None:
    """Create a git commit for the parameter change."""
    message = f"autoresearch: {param} {old_value}→{new_value} (sharpe_net {metric_delta:+.4f})"
    subprocess.run(
        ["git", "add", str(XGBOOST_CONFIG_PATH)],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "commit", "-m", message],
        check=True,
        capture_output=True,
    )
    logger.info(f"Committed: {message}")


def git_revert() -> None:
    """Revert changes to config file."""
    subprocess.run(
        ["git", "checkout", str(XGBOOST_CONFIG_PATH)],
        check=True,
        capture_output=True,
    )
    logger.info("Reverted config changes")


# =============================================================================
# Evaluation
# =============================================================================


def run_evaluation(dry_run: bool = False) -> EvalResult:
    """
    Run xgboost_eval.py and parse results.

    Args:
        dry_run: If True, return mock values

    Returns:
        EvalResult with all metrics
    """
    if dry_run:
        import random

        return EvalResult(
            sharpe_net=random.uniform(0.5, 1.5),
            ic=random.uniform(0.02, 0.10),
            ic_pvalue=random.uniform(0.001, 0.05),
            brier=random.uniform(0.20, 0.30),
            directional_accuracy=random.uniform(0.50, 0.60),
            false_bullish_rate=random.uniform(0.10, 0.30),
            false_bearish_rate=random.uniform(0.10, 0.30),
            num_examples=1000,
            config_hash="mock",
        )

    try:
        result = subprocess.run(
            [sys.executable, "evaluation/xgboost_eval.py", "--output-json"],
            capture_output=True,
            text=True,
            timeout=30 * 60,
            check=True,
        )

        data = json.loads(result.stdout)
        return EvalResult(**data)

    except subprocess.TimeoutExpired:
        logger.error("Evaluation timed out")
        raise
    except subprocess.CalledProcessError as e:
        logger.error(f"Evaluation failed: {e.stderr}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse eval output: {e}")
        raise


def check_stop_file() -> bool:
    """Check if STOP file exists."""
    return STOP_FILE_PATH.exists()


# =============================================================================
# Main Loop
# =============================================================================


class AutoresearchLoop:
    """Main autoresearch loop controller."""

    def __init__(
        self,
        max_experiments: int | None = None,
        time_budget_hours: float | None = None,
        dry_run: bool = False,
        metric: str = "sharpe_net",
    ):
        self.max_experiments = max_experiments
        self.time_budget_hours = time_budget_hours
        self.dry_run = dry_run
        self.metric = metric
        self.start_time = time.time()
        self.interrupted = False
        self.state = ParameterState()
        self.results: list[ExperimentResult] = []

    def should_stop(self) -> tuple[bool, str]:
        """Check if the loop should stop."""
        if self.interrupted:
            return True, "User interrupt (Ctrl+C)"

        if check_stop_file():
            return True, "STOP file exists"

        if self.max_experiments is not None:
            if len(self.results) >= self.max_experiments:
                return True, f"Reached max experiments ({self.max_experiments})"

        if self.time_budget_hours is not None:
            elapsed_hours = (time.time() - self.start_time) / 3600
            if elapsed_hours >= self.time_budget_hours:
                return True, f"Time budget exhausted ({self.time_budget_hours}h)"

        return False, ""

    def get_metric_value(self, result: EvalResult) -> float:
        """Get the primary metric value from eval result."""
        if self.metric == "sharpe_net":
            return result.sharpe_net
        elif self.metric == "ic":
            return result.ic
        elif self.metric == "brier":
            return result.brier
        else:
            return result.sharpe_net

    def is_improvement(self, new_value: float, best_value: float) -> bool:
        """Check if new value is an improvement over best."""
        threshold = IMPROVEMENT_THRESHOLDS.get(self.metric, 0.02)
        if self.metric == "brier":
            # Lower is better
            return new_value < best_value + threshold
        else:
            # Higher is better
            return new_value > best_value + threshold

    def run_single_experiment(self) -> ExperimentResult | None:
        """Run a single experiment iteration."""
        # Load previous results
        self.results = read_results_tsv()
        best_metric = get_best_metric(self.results, self.metric)

        # Get current config values
        current_values = read_current_config_values()

        # Choose next parameter to modify
        try:
            param, old_value, new_value = choose_next_parameter(
                self.state, current_values, self.results
            )
        except RuntimeError as e:
            logger.warning(f"Cannot choose next parameter: {e}")
            return None

        logger.info(f"Experiment {len(self.results) + 1}: {param} = {old_value} → {new_value}")

        # Modify config
        modify_config_parameter(param, new_value)

        # Evaluate
        try:
            eval_result = run_evaluation(dry_run=self.dry_run)
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            git_revert()
            return None

        metric_value = self.get_metric_value(eval_result)
        metric_delta = metric_value - (
            best_metric if best_metric != -float("inf") else metric_value
        )

        # Decide keep/revert
        improved = self.is_improvement(metric_value, best_metric)
        kept = improved

        if improved:
            logger.info(f"KEPT: {self.metric} improved {best_metric:.4f} → {metric_value:.4f}")
            if not self.dry_run:
                git_commit(param, old_value, new_value, metric_delta)
        else:
            logger.info(
                f"REVERTED: {self.metric} did not improve ({metric_value:.4f} <= {best_metric:.4f})"
            )
            git_revert()

        # Log result
        result = ExperimentResult(
            timestamp=datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S"),
            experiment_id=len(self.results) + 1,
            param_changed=param,
            old_value=old_value,
            new_value=new_value,
            sharpe_net=eval_result.sharpe_net,
            ic=eval_result.ic,
            brier=eval_result.brier,
            accuracy=eval_result.directional_accuracy,
            false_bullish_rate=eval_result.false_bullish_rate,
            kept=kept,
        )

        append_result(result)
        self.results.append(result)

        return result

    def run(self) -> None:
        """Run the main autoresearch loop."""
        logger.info("Starting XGBoost autoresearch loop")
        logger.info(f"Primary metric: {self.metric}")
        logger.info(f"Max experiments: {self.max_experiments or 'unlimited'}")
        logger.info(f"Time budget: {self.time_budget_hours or 'unlimited'} hours")
        logger.info(f"Dry run: {self.dry_run}")

        # Set up signal handler for graceful shutdown
        def signal_handler(sig: int, frame: Any) -> None:
            self.interrupted = True
            logger.info("Interrupt received, finishing current experiment...")

        sys_signal.signal(sys_signal.SIGINT, signal_handler)

        while True:
            stop, reason = self.should_stop()
            if stop:
                logger.info(f"Stopping: {reason}")
                break

            try:
                result = self.run_single_experiment()
                if result is None:
                    logger.warning("Experiment returned None, continuing...")
            except Exception as e:
                logger.exception(f"Experiment failed: {e}")

        # Print summary
        print()
        print(format_results_table(self.results))


def format_results_table(results: list[ExperimentResult]) -> str:
    """Format results as a summary table."""
    if not results:
        return "No experiments recorded yet."

    lines = []
    lines.append("=" * 100)
    lines.append("AUTORESEARCH EXPERIMENT SUMMARY")
    lines.append("=" * 100)
    lines.append(
        f"{'ID':<4} {'Timestamp':<20} {'Change':<30} "
        f"{'Sharpe':>8} {'IC':>8} {'Brier':>8} {'Status':<10}"
    )
    lines.append("-" * 100)

    for r in results:
        status = "KEPT" if r.kept else "REVERTED"
        change_desc = f"{r.param_changed}: {r.old_value}→{r.new_value}"
        if len(change_desc) > 28:
            change_desc = change_desc[:25] + "..."
        lines.append(
            f"{r.experiment_id:<4} {r.timestamp:<20} {change_desc:<30} "
            f"{r.sharpe_net:>8.4f} {r.ic:>8.4f} {r.brier:>8.4f} {status:<10}"
        )

    lines.append("-" * 100)

    # Summary stats
    kept = [r for r in results if r.kept]
    reverted = [r for r in results if not r.kept]
    lines.append(f"Total: {len(results)} experiments, {len(kept)} kept, {len(reverted)} reverted")

    if kept:
        best_sharpe = max(r.sharpe_net for r in kept)
        best_ic = max(r.ic for r in kept)
        lines.append(f"Best Sharpe: {best_sharpe:.4f}, Best IC: {best_ic:.4f}")

    lines.append("=" * 100)

    return "\n".join(lines)


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Autonomous XGBoost hyperparameter search (Karpathy autoresearch pattern)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run indefinitely until Ctrl+C or STOP file
    python run_autoresearch.py

    # Run at most 10 experiments
    python run_autoresearch.py --max-experiments 10

    # Run for at most 8 hours (overnight)
    python run_autoresearch.py --time-budget-hours 8

    # Optimize different metric
    python run_autoresearch.py --metric ic

    # Dry run (show what would be tried without modifying)
    python run_autoresearch.py --dry-run
        """,
    )

    parser.add_argument(
        "--max-experiments",
        "-n",
        type=int,
        default=None,
        help="Maximum number of experiments to run (default: unlimited)",
    )
    parser.add_argument(
        "--time-budget-hours",
        "-t",
        type=float,
        default=None,
        help="Maximum time to run in hours (default: unlimited)",
    )
    parser.add_argument(
        "--metric",
        "-m",
        type=str,
        default="sharpe_net",
        choices=["sharpe_net", "ic", "brier"],
        help="Primary metric to optimize (default: sharpe_net)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without modifying config",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    loop = AutoresearchLoop(
        max_experiments=args.max_experiments,
        time_budget_hours=args.time_budget_hours,
        dry_run=args.dry_run,
        metric=args.metric,
    )

    loop.run()


if __name__ == "__main__":
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level="INFO",
    )

    try:
        main()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(0)
    except Exception:
        logger.exception("Fatal error in autoresearch")
        sys.exit(1)
