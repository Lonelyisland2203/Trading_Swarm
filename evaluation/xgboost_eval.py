#!/usr/bin/env python3
"""
XGBoost model evaluation script for autoresearch loop.

THIS FILE IS READ-ONLY DURING AUTORESEARCH RUNS.
DO NOT MODIFY THIS FILE DURING EXPERIMENTS.

Loads data, reads config from signals/xgboost_config.py, runs walk-forward CV,
computes metrics, prints JSON to stdout, and appends to autoresearch/results.tsv.

Usage:
    python evaluation/xgboost_eval.py
    python evaluation/xgboost_eval.py --data data/grpo_training_data.jsonl
    python evaluation/xgboost_eval.py --output-json
"""

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

try:
    import shap
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    xgb = None
    shap = None

from config.fee_model import FeeModelSettings
from eval.metrics import compute_information_coefficient, compute_sharpe_ratio
from signals.xgboost_config import (
    CLASS_WEIGHTS,
    FEATURE_LIST,
    WALK_FORWARD_CONFIG,
    XGB_PARAMS,
)


# Paths
DEFAULT_DATA_PATH = Path("data/grpo_training_data.jsonl")
RESULTS_TSV_PATH = Path("autoresearch/results.tsv")
BASELINE_METRICS_PATH = Path("evaluation/baseline_metrics.json")


@dataclass
class EvalResult:
    """Complete evaluation result with all metrics."""

    # Primary metric
    sharpe_net: float

    # Secondary metrics
    ic: float
    ic_pvalue: float
    brier: float
    directional_accuracy: float

    # Asymmetric error rates
    false_bullish_rate: float
    false_bearish_rate: float

    # Metadata
    num_examples: int
    config_hash: str

    # SHAP importance (top 5)
    shap_top_5: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return asdict(self)


def extract_features_from_snapshot(snapshot: str) -> dict[str, float | None]:
    """
    Extract numerical features from a market snapshot string.

    Parses indicator values from the formatted prompt text.
    Same logic as evaluation/xgboost_baseline.py for consistency.

    Args:
        snapshot: Market snapshot string from GRPO training data

    Returns:
        Dictionary of feature name -> value (None if not found)
    """
    features: dict[str, float | None] = {f: None for f in FEATURE_LIST}

    # Parse RSI
    if "RSI(14):" in snapshot:
        try:
            val = snapshot.split("RSI(14):")[1].split()[0].strip()
            if val != "N/A":
                features["rsi"] = float(val)
        except (IndexError, ValueError):
            pass

    # Parse MACD Line
    if "MACD Line:" in snapshot:
        try:
            line = snapshot.split("MACD Line:")[1].split("|")[0].strip()
            if line != "N/A":
                features["macd_line"] = float(line)
        except (IndexError, ValueError):
            pass

    # Parse MACD Signal
    if "Signal:" in snapshot:
        try:
            macd_section = snapshot.split("MACD Line:")[1].split("\n")[0]
            if "Signal:" in macd_section:
                signal = macd_section.split("Signal:")[1].strip()
                if signal != "N/A":
                    features["macd_signal"] = float(signal)
        except (IndexError, ValueError):
            pass

    # Parse MACD Histogram
    if "MACD Histogram:" in snapshot:
        try:
            val = snapshot.split("MACD Histogram:")[1].split()[0].strip()
            if val != "N/A":
                features["macd_histogram"] = float(val)
        except (IndexError, ValueError):
            pass

    # Parse Donchian channels
    if "Donchian(20):" in snapshot:
        try:
            donchian_line = snapshot.split("Donchian(20):")[1].split("\n")[0]
            parts = donchian_line.split("|")
            for part in parts:
                if "Upper" in part:
                    val = part.split("$")[1].strip().split()[0]
                    if val != "N/A":
                        features["donchian_upper"] = float(val)
                elif "Mid" in part:
                    val = part.split("$")[1].strip().split()[0]
                    if val != "N/A":
                        features["donchian_middle"] = float(val)
                elif "Lower" in part:
                    val = part.split("$")[1].strip().split()[0]
                    if val != "N/A":
                        features["donchian_lower"] = float(val)
        except (IndexError, ValueError):
            pass

    # Parse KAMA
    if "KAMA(10):" in snapshot:
        try:
            val = snapshot.split("KAMA(10):")[1].split("$")[1].split()[0].strip()
            if val != "N/A":
                features["kama"] = float(val)
        except (IndexError, ValueError):
            pass

    # Parse OBV
    if "OBV:" in snapshot:
        try:
            val = snapshot.split("OBV:")[1].split()[0].strip()
            if val != "N/A":
                features["obv"] = float(val)
        except (IndexError, ValueError):
            pass

    # Parse CMF
    if "CMF(20):" in snapshot:
        try:
            val = snapshot.split("CMF(20):")[1].split()[0].strip()
            if val != "N/A":
                features["cmf"] = float(val)
        except (IndexError, ValueError):
            pass

    # Parse MFI
    if "MFI(14):" in snapshot:
        try:
            val = snapshot.split("MFI(14):")[1].split()[0].strip()
            if val != "N/A":
                features["mfi"] = float(val)
        except (IndexError, ValueError):
            pass

    # Parse VWAP
    if "VWAP:" in snapshot:
        try:
            val = snapshot.split("VWAP:")[1].split("$")[1].split()[0].strip()
            if val != "N/A":
                features["vwap"] = float(val)
        except (IndexError, ValueError):
            pass

    # Parse ATR Normalized
    if "ATR Normalized:" in snapshot:
        try:
            val = snapshot.split("ATR Normalized:")[1].split("%")[0].strip()
            if val != "N/A":
                features["atr_normalized"] = float(val)
        except (IndexError, ValueError):
            pass

    # Parse BB Width
    if "BB Width(20):" in snapshot:
        try:
            val = snapshot.split("BB Width(20):")[1].split("%")[0].strip()
            if val != "N/A":
                features["bb_width"] = float(val)
        except (IndexError, ValueError):
            pass

    # Parse Keltner Width
    if "Keltner Width:" in snapshot:
        try:
            val = snapshot.split("Keltner Width:")[1].split("%")[0].strip()
            if val != "N/A":
                features["keltner_width"] = float(val)
        except (IndexError, ValueError):
            pass

    # Parse Donchian Width
    if "Donchian Width:" in snapshot:
        try:
            val = snapshot.split("Donchian Width:")[1].split("%")[0].strip()
            if val != "N/A":
                features["donchian_width"] = float(val)
        except (IndexError, ValueError):
            pass

    # Parse Market Structure
    if "Open FVG Count:" in snapshot:
        try:
            val = snapshot.split("Open FVG Count:")[1].split()[0].strip()
            if val != "N/A":
                features["open_fvg_count"] = float(val)
        except (IndexError, ValueError):
            pass

    if "Nearest Bullish FVG:" in snapshot:
        try:
            val = snapshot.split("Nearest Bullish FVG:")[1].split("%")[0].strip()
            if val != "N/A":
                features["nearest_bullish_fvg_pct"] = float(val)
        except (IndexError, ValueError):
            pass

    if "Nearest Bearish FVG:" in snapshot:
        try:
            val = snapshot.split("Nearest Bearish FVG:")[1].split("%")[0].strip()
            if val != "N/A":
                features["nearest_bearish_fvg_pct"] = float(val)
        except (IndexError, ValueError):
            pass

    if "Nearest Swing High:" in snapshot:
        try:
            val = snapshot.split("Nearest Swing High:")[1].split("%")[0].strip()
            if val != "N/A":
                features["nearest_swing_high_pct"] = float(val)
        except (IndexError, ValueError):
            pass

    if "Nearest Swing Low:" in snapshot:
        try:
            val = snapshot.split("Nearest Swing Low:")[1].split("%")[0].strip()
            if val != "N/A":
                features["nearest_swing_low_pct"] = float(val)
        except (IndexError, ValueError):
            pass

    return features


def load_training_data(
    data_path: Path,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load training data from JSONL and extract features.

    Uses point-in-time safe data from GRPO training pipeline.
    Timestamp ordering ensures temporal safety in walk-forward CV.

    Args:
        data_path: Path to GRPO training data JSONL

    Returns:
        Tuple of (features_df, targets, returns, timestamps, directions)
    """
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    fee_model = FeeModelSettings()

    examples = []
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            examples.append(json.loads(line))

    logger.info(f"Loaded {len(examples)} examples from {data_path}")

    # Extract features and targets
    feature_rows = []
    targets = []
    returns = []
    timestamps = []
    directions = []

    for ex in examples:
        snapshot = ex["market_snapshot"]
        features = extract_features_from_snapshot(snapshot)
        feature_rows.append(features)

        # Target: binary up/down based on fee-adjusted return
        direction = ex["actual_direction"]

        # Binary target: 1 if LONG (price went up enough), 0 otherwise
        target = 1 if direction == "LONG" else 0
        targets.append(target)

        # Store direction for false bullish/bearish analysis
        directions.append(direction)

        # Store fee-adjusted return for Sharpe calculation
        gross_return = ex["gross_return_pct"]
        net_return = fee_model.net_return(gross_return, holding_periods_8h=1.0)
        returns.append(net_return)

        timestamps.append(ex["timestamp_ms"])

    features_df = pd.DataFrame(feature_rows)

    # Impute missing values with median
    for col in features_df.columns:
        if features_df[col].isna().any():
            median_val = features_df[col].median()
            if pd.isna(median_val):
                median_val = 0.0
            features_df[col] = features_df[col].fillna(median_val)

    return (
        features_df,
        np.array(targets),
        np.array(returns),
        np.array(timestamps),
        np.array(directions),
    )


def create_walk_forward_folds(
    timestamps: np.ndarray,
    n_folds: int | None = None,
    train_ratio: float | None = None,
    gap_bars: int | None = None,
) -> list[dict[str, Any]]:
    """
    Create walk-forward cross-validation folds.

    Uses WALK_FORWARD_CONFIG from xgboost_config.py as defaults.
    Ensures temporal ordering: train data always before test data.

    Args:
        timestamps: Array of timestamps in milliseconds
        n_folds: Number of folds (uses config default if None)
        train_ratio: Fraction for training (uses config default if None)
        gap_bars: Gap between train/test (uses config default if None)

    Returns:
        List of fold dicts with train_indices, test_indices
    """
    # Use config defaults
    n_folds = n_folds or WALK_FORWARD_CONFIG.n_folds
    train_ratio = train_ratio or WALK_FORWARD_CONFIG.train_ratio
    gap_bars = gap_bars or WALK_FORWARD_CONFIG.gap_bars

    n_samples = len(timestamps)
    sorted_indices = np.argsort(timestamps)
    sorted_timestamps = timestamps[sorted_indices]

    folds = []
    fold_size = n_samples // n_folds

    for i in range(n_folds):
        fold_end = (i + 1) * fold_size
        if i == n_folds - 1:
            fold_end = n_samples

        test_size = max(int(fold_size * (1 - train_ratio)), WALK_FORWARD_CONFIG.min_test_size)
        test_start = fold_end - test_size
        train_end = test_start - gap_bars

        if train_end <= WALK_FORWARD_CONFIG.min_train_size:
            continue

        train_indices = sorted_indices[:train_end]
        test_indices = sorted_indices[test_start:fold_end]

        if len(train_indices) < WALK_FORWARD_CONFIG.min_train_size:
            continue
        if len(test_indices) < WALK_FORWARD_CONFIG.min_test_size:
            continue

        folds.append(
            {
                "train_indices": train_indices,
                "test_indices": test_indices,
                "train_start_ts": int(sorted_timestamps[0]),
                "train_end_ts": int(sorted_timestamps[train_end - 1]),
                "test_start_ts": int(sorted_timestamps[test_start]),
                "test_end_ts": int(sorted_timestamps[fold_end - 1]),
            }
        )

    return folds


def get_config_hash() -> str:
    """
    Generate a hash of current XGBoost config for tracking.

    Returns:
        Short hash string of config parameters
    """
    import hashlib

    config_str = json.dumps(
        {
            "xgb_params": XGB_PARAMS,
            "class_weights": CLASS_WEIGHTS,
            "walk_forward": {
                "n_folds": WALK_FORWARD_CONFIG.n_folds,
                "train_ratio": WALK_FORWARD_CONFIG.train_ratio,
                "gap_bars": WALK_FORWARD_CONFIG.gap_bars,
            },
        },
        sort_keys=True,
    )
    return hashlib.sha256(config_str.encode()).hexdigest()[:8]


def compute_shap_importance(
    model: Any,
    X_test: np.ndarray,
    feature_names: list[str],
    top_n: int = 5,
) -> dict[str, float]:
    """
    Compute SHAP feature importance (top N).

    Args:
        model: Trained XGBoost model
        X_test: Test features
        feature_names: Feature names
        top_n: Number of top features to return

    Returns:
        Dict of feature name -> mean |SHAP| value
    """
    if not XGBOOST_AVAILABLE or shap is None:
        return {}

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        # Handle binary classification
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

        mean_abs_shap = np.abs(shap_values).mean(axis=0)

        importance = {}
        for i, name in enumerate(feature_names):
            importance[name] = float(mean_abs_shap[i])

        # Sort and take top N
        sorted_imp = sorted(importance.items(), key=lambda x: -x[1])
        return dict(sorted_imp[:top_n])
    except Exception as e:
        logger.warning(f"SHAP computation failed: {e}")
        return {}


def run_evaluation(data_path: Path) -> EvalResult:
    """
    Run full walk-forward evaluation.

    Uses current config from signals/xgboost_config.py.

    Args:
        data_path: Path to training data JSONL

    Returns:
        EvalResult with all metrics
    """
    if not XGBOOST_AVAILABLE:
        raise ImportError("xgboost not installed. Run: pip install xgboost")

    # Load data
    features_df, targets, returns, timestamps, directions = load_training_data(data_path)

    logger.info(f"Features shape: {features_df.shape}")
    logger.info(f"Target distribution: {np.bincount(targets)}")

    # Create folds
    folds = create_walk_forward_folds(timestamps)
    if not folds:
        raise ValueError("No valid folds created")

    logger.info(f"Running walk-forward CV with {len(folds)} folds")

    # Accumulate predictions across folds
    all_probs = []
    all_preds = []
    all_targets = []
    all_returns = []
    all_directions = []
    last_model = None

    X = features_df.values
    feature_names = list(features_df.columns)

    for i, fold in enumerate(folds):
        X_train = X[fold["train_indices"]]
        y_train = targets[fold["train_indices"]]
        X_test = X[fold["test_indices"]]
        y_test = targets[fold["test_indices"]]
        returns_test = returns[fold["test_indices"]]
        directions_test = directions[fold["test_indices"]]

        logger.debug(f"Fold {i + 1}: train={len(X_train)}, test={len(X_test)}")

        # Train model with current config
        model = xgb.XGBClassifier(**XGB_PARAMS)
        model.fit(X_train, y_train)

        probs = model.predict_proba(X_test)[:, 1]
        preds = model.predict(X_test)

        all_probs.extend(probs)
        all_preds.extend(preds)
        all_targets.extend(y_test)
        all_returns.extend(returns_test)
        all_directions.extend(directions_test)

        last_model = model

    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_returns = np.array(all_returns)
    all_directions = np.array(all_directions)

    # Compute metrics

    # IC: correlation between predicted probability and actual returns
    ic, ic_pvalue = compute_information_coefficient(all_probs, all_returns)

    # Brier score
    brier = float(np.mean((all_probs - all_targets) ** 2))

    # Directional accuracy
    accuracy = float(np.mean(all_preds == all_targets))

    # Sharpe ratio on strategy returns (fee-adjusted)
    # Strategy: long when pred=1, else flat
    strategy_returns = np.where(all_preds == 1, all_returns, 0.0)
    sharpe_net = compute_sharpe_ratio(strategy_returns, annualization_factor=365)

    # False bullish rate: predicted LONG (1) but actual was SHORT or FLAT
    false_bullish_mask = (all_preds == 1) & (all_directions != "LONG")
    false_bullish_rate = float(np.sum(false_bullish_mask) / max(np.sum(all_preds == 1), 1))

    # False bearish rate: predicted SHORT (0) but actual was LONG
    false_bearish_mask = (all_preds == 0) & (all_directions == "LONG")
    false_bearish_rate = float(np.sum(false_bearish_mask) / max(np.sum(all_preds == 0), 1))

    # SHAP importance (from last fold's model)
    X_last_test = X[folds[-1]["test_indices"]]
    shap_top_5 = compute_shap_importance(last_model, X_last_test, feature_names, top_n=5)

    return EvalResult(
        sharpe_net=sharpe_net,
        ic=ic,
        ic_pvalue=ic_pvalue,
        brier=brier,
        directional_accuracy=accuracy,
        false_bullish_rate=false_bullish_rate,
        false_bearish_rate=false_bearish_rate,
        num_examples=len(all_targets),
        config_hash=get_config_hash(),
        shap_top_5=shap_top_5,
    )


def append_to_results_tsv(
    result: EvalResult,
    change_description: str = "manual_eval",
    kept: bool = True,
) -> None:
    """
    Append evaluation result to autoresearch/results.tsv.

    Args:
        result: Evaluation result
        change_description: Description of config change
        kept: Whether this config was kept (vs reverted)
    """
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

    # Get next experiment ID
    experiment_id = 0
    if file_exists:
        with open(RESULTS_TSV_PATH) as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                try:
                    experiment_id = max(experiment_id, int(row["experiment_id"]))
                except (KeyError, ValueError):
                    pass
        experiment_id += 1

    with open(RESULTS_TSV_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        if not file_exists:
            writer.writeheader()
        writer.writerow(
            {
                "experiment_id": experiment_id,
                "timestamp": datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S"),
                "change_description": change_description,
                "sharpe_net": f"{result.sharpe_net:.6f}",
                "ic": f"{result.ic:.6f}",
                "brier": f"{result.brier:.6f}",
                "accuracy": f"{result.directional_accuracy:.4f}",
                "false_bullish_rate": f"{result.false_bullish_rate:.4f}",
                "kept_or_reverted": "kept" if kept else "reverted",
            }
        )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate XGBoost model with current config (read-only script)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script is READ-ONLY during autoresearch runs.
It reads config from signals/xgboost_config.py and evaluates model performance.

Examples:
    python evaluation/xgboost_eval.py
    python evaluation/xgboost_eval.py --output-json
    python evaluation/xgboost_eval.py --data data/custom_data.jsonl
        """,
    )

    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to training data JSONL",
    )
    parser.add_argument(
        "--output-json",
        action="store_true",
        help="Print JSON results to stdout",
    )
    parser.add_argument(
        "--append-tsv",
        action="store_true",
        help="Append results to autoresearch/results.tsv",
    )
    parser.add_argument(
        "--change-description",
        type=str,
        default="manual_eval",
        help="Description of config change (for TSV logging)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose logging",
    )

    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()

    # Configure logging
    logger.remove()
    log_level = "DEBUG" if args.verbose else "INFO"
    logger.add(
        sys.stderr,
        level=log_level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    )

    if not XGBOOST_AVAILABLE:
        logger.error("xgboost and shap must be installed")
        logger.error("Run: pip install xgboost shap")
        sys.exit(1)

    # Run evaluation
    logger.info(f"Evaluating with data from {args.data}")
    result = run_evaluation(args.data)

    # Print results
    if args.output_json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print()
        print("=" * 60)
        print("XGBOOST EVALUATION RESULTS")
        print("=" * 60)
        print(f"Sharpe (net):         {result.sharpe_net:>10.4f}  [PRIMARY METRIC]")
        print(f"IC:                   {result.ic:>10.4f}")
        print(f"IC p-value:           {result.ic_pvalue:>10.4f}")
        print(f"Brier score:          {result.brier:>10.4f}")
        print(f"Directional accuracy: {result.directional_accuracy:>9.2%}")
        print(f"False bullish rate:   {result.false_bullish_rate:>9.2%}")
        print(f"False bearish rate:   {result.false_bearish_rate:>9.2%}")
        print(f"Num examples:         {result.num_examples:>10}")
        print(f"Config hash:          {result.config_hash}")
        print("-" * 60)
        print("SHAP Top 5 Features:")
        for feat, imp in result.shap_top_5.items():
            print(f"  {feat:<30} {imp:.4f}")
        print("=" * 60)

    # Append to TSV if requested
    if args.append_tsv:
        append_to_results_tsv(result, args.change_description, kept=True)
        logger.info(f"Appended results to {RESULTS_TSV_PATH}")


if __name__ == "__main__":
    main()
