#!/usr/bin/env python3
"""
Autonomous hyperparameter search loop for GRPO training.

Inspired by karpathy/autoresearch - iteratively explores hyperparameter space
using a simple heuristic to decide what to try next. Uses git commits to track
changes and reverts unsuccessful experiments.

Usage:
    # Run indefinitely until Ctrl+C
    python run_autoresearch.py

    # Run at most 10 experiments
    python run_autoresearch.py --max-experiments 10

    # Run for at most 2 hours
    python run_autoresearch.py --time-budget-hours 2

    # Dry run (don't actually train, just show what would be tried)
    python run_autoresearch.py --dry-run
"""

import argparse
import csv
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

from loguru import logger


# Parameter search space definitions
PARAM_SEARCH_SPACE: dict[str, list[Any]] = {
    "G": [2, 4, 6, 8],
    "beta": [0.01, 0.02, 0.04, 0.08, 0.12],
    "epsilon": [0.1, 0.15, 0.2, 0.3],
    "lr": [1e-6, 2e-6, 5e-6, 1e-5, 2e-5],
    "lora_r": [8, 16, 32, 64],
    "lora_alpha": [16, 32, 64, 128],
    "false_bullish_penalty": [1.0, 1.2, 1.5, 2.0, 2.5],
    "false_bearish_penalty": [0.5, 0.6, 0.8, 1.0, 1.2],
}

# Mapping from our parameter names to config field paths
PARAM_TO_CONFIG_FIELD: dict[str, str] = {
    "G": "group_size",
    "beta": "kl_penalty_beta",
    "epsilon": "clip_epsilon",
    "lr": "learning_rate",
    "lora_r": "lora.rank",
    "lora_alpha": "lora.alpha",
    "false_bullish_penalty": "reward.false_bullish_penalty",
    "false_bearish_penalty": "reward.false_bearish_penalty",
}

# Parameter exploration order (can be adjusted based on importance)
PARAM_ORDER = [
    "lr",
    "beta",
    "epsilon",
    "G",
    "lora_r",
    "lora_alpha",
    "false_bullish_penalty",
    "false_bearish_penalty",
]

# Paths
GRPO_CONFIG_PATH = Path("training/grpo_config.py")
RESULTS_TSV_PATH = Path("results.tsv")
DEFAULT_TRAINING_STEPS = 1000
DEFAULT_TRAINING_EXAMPLES = 2000


@dataclass
class ExperimentResult:
    """Result of a single experiment."""

    timestamp: str
    experiment_id: int
    param_changed: str
    old_value: Any
    new_value: Any
    ic: float
    brier: float
    kept: bool


@dataclass
class ParameterState:
    """Tracks the state of parameter exploration."""

    # Current round-robin index
    param_index: int = 0

    # For each parameter, tracks which value index we're at
    value_indices: dict[str, int] = field(default_factory=dict)

    # Tracks which parameters have improved IC recently (responsiveness)
    improvement_counts: dict[str, int] = field(default_factory=dict)

    # Direction of exploration (+1 or -1 for each param)
    directions: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Initialize defaults
        for param in PARAM_ORDER:
            if param not in self.value_indices:
                self.value_indices[param] = -1  # Will be set from current config
            if param not in self.improvement_counts:
                self.improvement_counts[param] = 0
            if param not in self.directions:
                self.directions[param] = 1  # Start by exploring upward


def read_current_config_values() -> dict[str, Any]:
    """
    Parse current values from grpo_config.py.

    Returns:
        Dictionary mapping parameter names to current values.
    """
    if not GRPO_CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config file not found: {GRPO_CONFIG_PATH}")

    content = GRPO_CONFIG_PATH.read_text()
    values: dict[str, Any] = {}

    # Parse each parameter using regex
    patterns = {
        "G": r"group_size:\s*int\s*=\s*(\d+)",
        "beta": r"kl_penalty_beta:\s*float\s*=\s*([\d.e-]+)",
        "epsilon": r"clip_epsilon:\s*float\s*=\s*([\d.]+)",
        "lr": r"learning_rate:\s*float\s*=\s*([\d.e-]+)",
        "lora_r": r"rank:\s*int\s*=\s*(\d+)",
        "lora_alpha": r"alpha:\s*int\s*=\s*(\d+)",
        "false_bullish_penalty": r"false_bullish_penalty:\s*float\s*=\s*([\d.]+)",
        "false_bearish_penalty": r"false_bearish_penalty:\s*float\s*=\s*([\d.]+)",
    }

    for param, pattern in patterns.items():
        match = re.search(pattern, content)
        if match:
            raw_value = match.group(1)
            # Convert to appropriate type
            if param in ["G", "lora_r", "lora_alpha"]:
                values[param] = int(raw_value)
            else:
                values[param] = float(raw_value)
        else:
            logger.warning(f"Could not parse {param} from config")

    return values


def modify_config_parameter(param: str, new_value: Any) -> str:
    """
    Modify a single parameter in grpo_config.py.

    Args:
        param: Parameter name (our naming scheme)
        new_value: New value to set

    Returns:
        Old value as string for commit message
    """
    content = GRPO_CONFIG_PATH.read_text()
    old_content = content

    # Format value for Python code
    if isinstance(new_value, float) and new_value < 0.001:
        value_str = f"{new_value:.0e}"
    elif isinstance(new_value, float):
        value_str = f"{new_value}"
    else:
        value_str = str(new_value)

    # Patterns for extraction (to get old value) and replacement
    extract_patterns = {
        "G": r"group_size:\s*int\s*=\s*(\d+)",
        "beta": r"kl_penalty_beta:\s*float\s*=\s*([\d.e-]+)",
        "epsilon": r"clip_epsilon:\s*float\s*=\s*([\d.]+)",
        "lr": r"learning_rate:\s*float\s*=\s*([\d.e-]+)",
        "lora_r": r"rank:\s*int\s*=\s*(\d+)",
        "lora_alpha": r"alpha:\s*int\s*=\s*(\d+)",
        "false_bullish_penalty": r"false_bullish_penalty:\s*float\s*=\s*([\d.]+)",
        "false_bearish_penalty": r"false_bearish_penalty:\s*float\s*=\s*([\d.]+)",
    }

    replace_patterns = {
        "G": (r"(group_size:\s*int\s*=\s*)\d+", rf"\g<1>{value_str}"),
        "beta": (r"(kl_penalty_beta:\s*float\s*=\s*)[\d.e-]+", rf"\g<1>{value_str}"),
        "epsilon": (r"(clip_epsilon:\s*float\s*=\s*)[\d.]+", rf"\g<1>{value_str}"),
        "lr": (r"(learning_rate:\s*float\s*=\s*)[\d.e-]+", rf"\g<1>{value_str}"),
        "lora_r": (r"(rank:\s*int\s*=\s*)\d+", rf"\g<1>{value_str}"),
        "lora_alpha": (r"(alpha:\s*int\s*=\s*)\d+", rf"\g<1>{value_str}"),
        "false_bullish_penalty": (
            r"(false_bullish_penalty:\s*float\s*=\s*)[\d.]+",
            rf"\g<1>{value_str}",
        ),
        "false_bearish_penalty": (
            r"(false_bearish_penalty:\s*float\s*=\s*)[\d.]+",
            rf"\g<1>{value_str}",
        ),
    }

    if param not in replace_patterns:
        raise ValueError(f"Unknown parameter: {param}")

    # Extract old value first
    extract_match = re.search(extract_patterns[param], old_content)
    old_value = extract_match.group(1) if extract_match else "unknown"

    # Do the replacement
    pattern, replacement = replace_patterns[param]
    new_content, count = re.subn(pattern, replacement, content)

    if count == 0:
        raise ValueError(f"Could not find {param} in config file")

    if count > 1:
        # For lora_r and lora_alpha, we might match multiple times
        # Only replace the first occurrence (in GRPOLoRAConfig)
        if param in ["lora_r", "lora_alpha"]:
            new_content = re.sub(pattern, replacement, content, count=1)
        else:
            raise ValueError(f"Multiple matches for {param} in config file")

    GRPO_CONFIG_PATH.write_text(new_content)

    return old_value


def read_results_tsv() -> list[ExperimentResult]:
    """Read experiment results from TSV file."""
    results: list[ExperimentResult] = []

    if not RESULTS_TSV_PATH.exists():
        return results

    with open(RESULTS_TSV_PATH, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            results.append(
                ExperimentResult(
                    timestamp=row["timestamp"],
                    experiment_id=int(row["experiment_id"]),
                    param_changed=row["param_changed"],
                    old_value=row["old_value"],
                    new_value=row["new_value"],
                    ic=float(row["ic"]),
                    brier=float(row["brier"]),
                    kept=row["kept"].lower() == "true",
                )
            )

    return results


def write_results_tsv(results: list[ExperimentResult]) -> None:
    """Write experiment results to TSV file."""
    fieldnames = [
        "timestamp",
        "experiment_id",
        "param_changed",
        "old_value",
        "new_value",
        "ic",
        "brier",
        "kept",
    ]

    with open(RESULTS_TSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "timestamp": result.timestamp,
                    "experiment_id": result.experiment_id,
                    "param_changed": result.param_changed,
                    "old_value": result.old_value,
                    "new_value": result.new_value,
                    "ic": f"{result.ic:.6f}",
                    "brier": f"{result.brier:.6f}",
                    "kept": str(result.kept),
                }
            )


def append_result(result: ExperimentResult) -> None:
    """Append a single result to the TSV file."""
    file_exists = RESULTS_TSV_PATH.exists()

    fieldnames = [
        "timestamp",
        "experiment_id",
        "param_changed",
        "old_value",
        "new_value",
        "ic",
        "brier",
        "kept",
    ]

    with open(RESULTS_TSV_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        if not file_exists:
            writer.writeheader()
        writer.writerow(
            {
                "timestamp": result.timestamp,
                "experiment_id": result.experiment_id,
                "param_changed": result.param_changed,
                "old_value": result.old_value,
                "new_value": result.new_value,
                "ic": f"{result.ic:.6f}",
                "brier": f"{result.brier:.6f}",
                "kept": str(result.kept),
            }
        )


def get_best_ic(results: list[ExperimentResult]) -> float:
    """Get the best IC from kept experiments."""
    kept_results = [r for r in results if r.kept]
    if not kept_results:
        return -float("inf")
    return max(r.ic for r in kept_results)


def choose_next_parameter(
    state: ParameterState,
    current_values: dict[str, Any],
    results: list[ExperimentResult],
) -> tuple[str, Any, Any]:
    """
    Choose the next parameter to modify using round-robin with simple heuristic.

    Heuristic:
    - Cycle through parameters in PARAM_ORDER
    - For each parameter, try the next value in its search space
    - If the last change to this param improved IC, continue in same direction
    - If it worsened, try the opposite direction
    - Skip parameters that have exhausted their search space

    Args:
        state: Current parameter exploration state
        current_values: Current config values
        results: Previous experiment results

    Returns:
        Tuple of (param_name, old_value, new_value)
    """
    # Update state based on recent results
    if results:
        last_result = results[-1]
        param = last_result.param_changed
        if param in state.directions:
            if last_result.kept:
                # Success - continue in same direction, boost responsiveness
                state.improvement_counts[param] = state.improvement_counts.get(param, 0) + 1
            else:
                # Failure - reverse direction
                state.directions[param] *= -1

    # Find current value index for each parameter
    for param in PARAM_ORDER:
        if param in current_values:
            search_space = PARAM_SEARCH_SPACE[param]
            current_val = current_values[param]
            # Find closest match in search space
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
            if new_value != old_value:  # Ensure we're actually changing something
                state.value_indices[param] = new_idx
                return param, old_value, new_value

        # Try opposite direction
        new_idx = current_idx - direction
        if 0 <= new_idx < len(search_space):
            old_value = current_values.get(param, search_space[current_idx])
            new_value = search_space[new_idx]
            if new_value != old_value:
                state.value_indices[param] = new_idx
                state.directions[param] = -direction  # Update direction
                return param, old_value, new_value

        attempts += 1

    raise RuntimeError("Exhausted all parameter search space")


def git_commit(param: str, value: Any) -> None:
    """Create a git commit for the parameter change."""
    message = f"autoresearch: {param}={value}"
    subprocess.run(
        ["git", "add", str(GRPO_CONFIG_PATH)],
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
    """Revert the last commit."""
    subprocess.run(
        ["git", "reset", "--hard", "HEAD~1"],
        check=True,
        capture_output=True,
    )
    logger.info("Reverted to previous commit")


def run_training(
    max_steps: int = DEFAULT_TRAINING_STEPS,
    max_examples: int = DEFAULT_TRAINING_EXAMPLES,
    dry_run: bool = False,
) -> bool:
    """
    Run short GRPO training.

    Args:
        max_steps: Maximum training steps
        max_examples: Maximum training examples to use
        dry_run: If True, skip actual training

    Returns:
        True if training succeeded
    """
    if dry_run:
        logger.info(f"[DRY RUN] Would train for {max_steps} steps on {max_examples} examples")
        return True

    try:
        result = subprocess.run(
            [
                sys.executable,
                "run_grpo_training.py",
                f"--max-steps={max_steps}",
                f"--limit={max_examples}",
            ],
            capture_output=True,
            text=True,
            timeout=60 * 60,  # 1 hour timeout
        )
        if result.returncode != 0:
            logger.error(f"Training failed: {result.stderr}")
            return False
        return True
    except subprocess.TimeoutExpired:
        logger.error("Training timed out")
        return False
    except Exception as e:
        logger.error(f"Training error: {e}")
        return False


def run_evaluation(dry_run: bool = False) -> tuple[float, float]:
    """
    Run evaluation on held-out test window.

    Args:
        dry_run: If True, return mock values

    Returns:
        Tuple of (IC, Brier score)
    """
    if dry_run:
        # Return mock values for dry run
        import random

        return random.uniform(0.02, 0.10), random.uniform(0.20, 0.30)

    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "training.evaluate_candidate",
                "--adapter",
                "adapters/grpo_latest",
                "--adapter-type",
                "grpo",
                "--data",
                "data/grpo_test_data.jsonl",
                "--output-json",
                "/tmp/eval_result.json",
            ],
            capture_output=True,
            text=True,
            timeout=30 * 60,  # 30 minute timeout
            check=True,  # Raise exception on non-zero exit
        )

        # Parse results from JSON output
        import json

        with open("/tmp/eval_result.json") as f:
            eval_data = json.load(f)

        return eval_data.get("ic", 0.0), eval_data.get("brier_score", 1.0)

    except Exception as e:
        logger.error(f"Evaluation error: {e}")
        return 0.0, 1.0


def format_results_table(results: list[ExperimentResult]) -> str:
    """Format results as a summary table."""
    if not results:
        return "No experiments recorded yet."

    lines = []
    lines.append("=" * 90)
    lines.append("EXPERIMENT SUMMARY")
    lines.append("=" * 90)
    lines.append(
        f"{'ID':<4} {'Timestamp':<20} {'Parameter':<25} {'Old':<10} {'New':<10} {'IC':<8} {'Brier':<8} {'Status':<10}"
    )
    lines.append("-" * 90)

    for r in results:
        status = "KEPT ✓" if r.kept else "REVERTED ✗"
        lines.append(
            f"{r.experiment_id:<4} {r.timestamp:<20} {r.param_changed:<25} "
            f"{str(r.old_value):<10} {str(r.new_value):<10} "
            f"{r.ic:>7.4f} {r.brier:>7.4f} {status:<10}"
        )

    lines.append("-" * 90)

    # Summary stats
    kept = [r for r in results if r.kept]
    reverted = [r for r in results if not r.kept]
    lines.append(f"Total: {len(results)} experiments, {len(kept)} kept, {len(reverted)} reverted")

    if kept:
        best_ic = max(r.ic for r in kept)
        lines.append(f"Best IC: {best_ic:.4f}")

    lines.append("=" * 90)

    return "\n".join(lines)


class AutoresearchLoop:
    """Main autoresearch loop controller."""

    def __init__(
        self,
        max_experiments: int | None = None,
        time_budget_hours: float | None = None,
        dry_run: bool = False,
        training_steps: int = DEFAULT_TRAINING_STEPS,
        training_examples: int = DEFAULT_TRAINING_EXAMPLES,
    ):
        self.max_experiments = max_experiments
        self.time_budget_hours = time_budget_hours
        self.dry_run = dry_run
        self.training_steps = training_steps
        self.training_examples = training_examples
        self.start_time = time.time()
        self.interrupted = False
        self.state = ParameterState()
        self.results: list[ExperimentResult] = []

    def should_stop(self) -> tuple[bool, str]:
        """Check if the loop should stop."""
        if self.interrupted:
            return True, "User interrupt (Ctrl+C)"

        if self.max_experiments is not None:
            if len(self.results) >= self.max_experiments:
                return True, f"Reached max experiments ({self.max_experiments})"

        if self.time_budget_hours is not None:
            elapsed_hours = (time.time() - self.start_time) / 3600
            if elapsed_hours >= self.time_budget_hours:
                return True, f"Time budget exhausted ({self.time_budget_hours}h)"

        return False, ""

    def run_single_experiment(self) -> ExperimentResult | None:
        """Run a single experiment iteration."""
        # Load previous results
        self.results = read_results_tsv()
        best_ic = get_best_ic(self.results)

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

        logger.info(f"Experiment {len(self.results) + 1}: {param} = {old_value} -> {new_value}")

        # Modify config
        modify_config_parameter(param, new_value)

        # Git commit
        if not self.dry_run:
            git_commit(param, new_value)

        # Run training
        training_success = run_training(
            max_steps=self.training_steps,
            max_examples=self.training_examples,
            dry_run=self.dry_run,
        )

        if not training_success:
            logger.error("Training failed, reverting")
            if not self.dry_run:
                git_revert()
            return None

        # Evaluate
        ic, brier = run_evaluation(dry_run=self.dry_run)

        # Decide keep/revert
        improved = ic > best_ic
        kept = improved

        if improved:
            logger.info(f"KEPT ✓ IC improved: {best_ic:.4f} -> {ic:.4f}")
        else:
            logger.info(f"REVERTED ✗ IC did not improve: {ic:.4f} <= {best_ic:.4f}")
            if not self.dry_run:
                git_revert()

        # Log result
        result = ExperimentResult(
            timestamp=datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S"),
            experiment_id=len(self.results) + 1,
            param_changed=param,
            old_value=old_value,
            new_value=new_value,
            ic=ic,
            brier=brier,
            kept=kept,
        )

        append_result(result)
        self.results.append(result)

        return result

    def run(self) -> None:
        """Run the main autoresearch loop."""
        logger.info("Starting autoresearch loop")
        logger.info(f"Max experiments: {self.max_experiments or 'unlimited'}")
        logger.info(f"Time budget: {self.time_budget_hours or 'unlimited'} hours")
        logger.info(f"Dry run: {self.dry_run}")

        # Set up signal handler for graceful shutdown
        def signal_handler(sig: int, frame: Any) -> None:
            self.interrupted = True
            logger.info("Interrupt received, finishing current experiment...")

        signal.signal(signal.SIGINT, signal_handler)

        while True:
            stop, reason = self.should_stop()
            if stop:
                logger.info(f"Stopping: {reason}")
                break

            try:
                result = self.run_single_experiment()
                if result is None:
                    logger.warning("Experiment returned None, continuing...")
                    continue
            except Exception as e:
                logger.exception(f"Experiment failed: {e}")
                continue

        # Print summary
        print()
        print(format_results_table(self.results))


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Autonomous hyperparameter search for GRPO training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run indefinitely until Ctrl+C
  python run_autoresearch.py

  # Run at most 10 experiments
  python run_autoresearch.py --max-experiments 10

  # Run for at most 2 hours
  python run_autoresearch.py --time-budget-hours 2

  # Dry run (show what would be tried without training)
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
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually training",
    )
    parser.add_argument(
        "--training-steps",
        type=int,
        default=DEFAULT_TRAINING_STEPS,
        help=f"Training steps per experiment (default: {DEFAULT_TRAINING_STEPS})",
    )
    parser.add_argument(
        "--training-examples",
        type=int,
        default=DEFAULT_TRAINING_EXAMPLES,
        help=f"Training examples to use (default: {DEFAULT_TRAINING_EXAMPLES})",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    loop = AutoresearchLoop(
        max_experiments=args.max_experiments,
        time_budget_hours=args.time_budget_hours,
        dry_run=args.dry_run,
        training_steps=args.training_steps,
        training_examples=args.training_examples,
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
