"""
XGBoost and LightGBM Baseline Models for LLM Comparison.

Trains gradient boosting models on the same numerical features the LLM sees
in its prompts, providing a baseline for comparison.

Features:
- All 17 indicator scalar summaries from compute_all_indicators()
- funding_rate and open_interest (when available)

Target:
- Binary classification (price up/down after fee-adjusted net return)

Evaluation:
- Walk-forward cross-validation (same temporal splits as LLM)
- Metrics: IC, Brier, Sharpe (fee-adjusted), directional accuracy
- SHAP feature importance

Usage:
    python -m evaluation.xgboost_baseline --data data/grpo_training_data.jsonl
    python -m evaluation.xgboost_baseline --data data/grpo_training_data.jsonl --compare
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

try:
    import lightgbm as lgb
    import shap
    import xgboost as xgb

    GBDT_AVAILABLE = True
except ImportError:
    GBDT_AVAILABLE = False

from config.fee_model import FeeModelSettings
from eval.metrics import compute_information_coefficient, compute_sharpe_ratio


# Feature names corresponding to the 17 indicators + extras
INDICATOR_FEATURES = [
    # Price/Trend (7)
    "rsi",
    "macd_line",
    "macd_signal",
    "macd_histogram",
    "donchian_upper",
    "donchian_middle",
    "donchian_lower",
    "kama",
    # Volume (4)
    "obv",
    "cmf",
    "mfi",
    "vwap",
    # Volatility (4)
    "atr_normalized",
    "bb_width",
    "keltner_width",
    "donchian_width",
    # Market Structure (5)
    "open_fvg_count",
    "nearest_bullish_fvg_pct",
    "nearest_bearish_fvg_pct",
    "nearest_swing_high_pct",
    "nearest_swing_low_pct",
]

# Additional features
EXTRA_FEATURES = [
    "funding_rate",
    "open_interest",
]

ALL_FEATURES = INDICATOR_FEATURES + EXTRA_FEATURES


@dataclass(frozen=True)
class BaselineEvaluation:
    """Evaluation results for a baseline model."""

    model_type: str  # "xgboost" or "lightgbm"
    ic: float
    ic_pvalue: float
    brier_score: float
    sharpe_ratio: float
    directional_accuracy: float
    num_examples: int
    feature_importance: dict[str, float] = field(default_factory=dict)
    shap_importance: dict[str, float] = field(default_factory=dict)
    ic_by_regime: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "model_type": self.model_type,
            "ic": self.ic,
            "ic_pvalue": self.ic_pvalue,
            "brier_score": self.brier_score,
            "sharpe_ratio": self.sharpe_ratio,
            "directional_accuracy": self.directional_accuracy,
            "num_examples": self.num_examples,
            "feature_importance": self.feature_importance,
            "shap_importance": self.shap_importance,
            "ic_by_regime": self.ic_by_regime,
        }


@dataclass
class WalkForwardFold:
    """Single walk-forward fold with train/test indices."""

    train_indices: np.ndarray
    test_indices: np.ndarray
    train_start_ts: int
    train_end_ts: int
    test_start_ts: int
    test_end_ts: int


def extract_features_from_snapshot(snapshot: str) -> dict[str, float | None]:
    """
    Extract numerical features from a market snapshot string.

    Parses the indicator values from the formatted prompt text.

    Args:
        snapshot: Market snapshot string from GRPO training data

    Returns:
        Dictionary of feature name -> value (None if not found)
    """
    features: dict[str, float | None] = {f: None for f in ALL_FEATURES}

    # Parse RSI
    if "RSI(14):" in snapshot:
        try:
            val = snapshot.split("RSI(14):")[1].split()[0].strip()
            if val != "N/A":
                features["rsi"] = float(val)
        except (IndexError, ValueError):
            pass

    # Parse MACD Line and Signal
    if "MACD Line:" in snapshot:
        try:
            line = snapshot.split("MACD Line:")[1].split("|")[0].strip()
            if line != "N/A":
                features["macd_line"] = float(line)
        except (IndexError, ValueError):
            pass

    if "Signal:" in snapshot:
        try:
            # Find the Signal after MACD Line
            macd_section = snapshot.split("MACD Line:")[1].split("\n")[0]
            if "Signal:" in macd_section:
                signal = macd_section.split("Signal:")[1].strip()
                if signal != "N/A":
                    features["macd_signal"] = float(signal)
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

    # funding_rate and open_interest are not in current snapshots
    # They'll remain None and be imputed

    return features


def create_walk_forward_folds(
    timestamps: np.ndarray,
    n_folds: int = 5,
    train_ratio: float = 0.7,
    gap_bars: int = 168,  # 1 week for 1h data - prevents serial correlation leakage
) -> list[WalkForwardFold]:
    """
    Create walk-forward cross-validation folds.

    Ensures temporal ordering: train data always before test data.

    Args:
        timestamps: Array of timestamps in milliseconds
        n_folds: Number of folds
        train_ratio: Fraction of data for training in each fold
        gap_bars: Gap between train and test to avoid leakage

    Returns:
        List of WalkForwardFold objects
    """
    n_samples = len(timestamps)
    sorted_indices = np.argsort(timestamps)
    sorted_timestamps = timestamps[sorted_indices]

    folds = []
    fold_size = n_samples // n_folds

    for i in range(n_folds):
        # Expanding window: use all data up to fold boundary for training
        fold_end = (i + 1) * fold_size
        if i == n_folds - 1:
            fold_end = n_samples

        # Test set is the last portion of this fold
        test_size = max(int(fold_size * (1 - train_ratio)), 30)
        test_start = fold_end - test_size
        train_end = test_start - gap_bars

        if train_end <= 0:
            continue

        train_indices = sorted_indices[:train_end]
        test_indices = sorted_indices[test_start:fold_end]

        if len(train_indices) < 50 or len(test_indices) < 20:
            continue

        folds.append(
            WalkForwardFold(
                train_indices=train_indices,
                test_indices=test_indices,
                train_start_ts=int(sorted_timestamps[0]),
                train_end_ts=int(sorted_timestamps[train_end - 1]),
                test_start_ts=int(sorted_timestamps[test_start]),
                test_end_ts=int(sorted_timestamps[fold_end - 1]),
            )
        )

    return folds


def load_training_data(
    data_path: Path,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load training data from JSONL and extract features.

    Args:
        data_path: Path to GRPO training data JSONL

    Returns:
        Tuple of (features_df, targets, returns, timestamps)
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

    for ex in examples:
        snapshot = ex["market_snapshot"]
        features = extract_features_from_snapshot(snapshot)
        feature_rows.append(features)

        # Target: binary up/down based on fee-adjusted return
        gross_return = ex["gross_return_pct"]
        direction = ex["actual_direction"]

        # Binary target: 1 if LONG (price went up enough), 0 otherwise
        target = 1 if direction == "LONG" else 0
        targets.append(target)

        # Store fee-adjusted return for Sharpe calculation
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
    )


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, Any]:
    """
    Train XGBoost classifier with default hyperparameters.

    Args:
        X_train: Training features
        y_train: Training targets
        X_test: Test features

    Returns:
        Tuple of (predicted_probabilities, predicted_classes, model)
    """
    if not GBDT_AVAILABLE:
        raise ImportError("xgboost not installed")

    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss",
        verbosity=0,
    )

    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]
    preds = model.predict(X_test)

    return probs, preds, model


def train_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, Any]:
    """
    Train LightGBM classifier with default hyperparameters.

    Args:
        X_train: Training features
        y_train: Training targets
        X_test: Test features

    Returns:
        Tuple of (predicted_probabilities, predicted_classes, model)
    """
    if not GBDT_AVAILABLE:
        raise ImportError("lightgbm not installed")

    model = lgb.LGBMClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    )

    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]
    preds = model.predict(X_test)

    return probs, preds, model


def compute_shap_importance(
    model: Any,
    X_test: np.ndarray,
    feature_names: list[str],
) -> dict[str, float]:
    """
    Compute SHAP feature importance.

    Args:
        model: Trained model (XGBoost or LightGBM)
        X_test: Test features
        feature_names: List of feature names

    Returns:
        Dictionary of feature name -> mean absolute SHAP value
    """
    if not GBDT_AVAILABLE:
        return {}

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        # Handle binary classification (may be list or array)
        if isinstance(shap_values, list):
            # For binary classification, take class 1
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

        # Mean absolute SHAP value per feature
        mean_abs_shap = np.abs(shap_values).mean(axis=0)

        importance = {}
        for i, name in enumerate(feature_names):
            importance[name] = float(mean_abs_shap[i])

        # Sort by importance
        return dict(sorted(importance.items(), key=lambda x: -x[1]))
    except Exception as e:
        logger.warning(f"SHAP computation failed: {e}")
        return {}


def evaluate_baseline(
    model_type: str,
    features_df: pd.DataFrame,
    targets: np.ndarray,
    returns: np.ndarray,
    timestamps: np.ndarray,
    n_folds: int = 5,
    gap_bars: int | None = None,
) -> BaselineEvaluation:
    """
    Evaluate baseline model with walk-forward cross-validation.

    Args:
        model_type: "xgboost" or "lightgbm"
        features_df: Feature DataFrame
        targets: Binary targets
        returns: Fee-adjusted returns
        timestamps: Timestamps in milliseconds
        n_folds: Number of CV folds
        gap_bars: Gap between train/test (default: 168 for 1h data = 1 week)

    Returns:
        BaselineEvaluation with aggregated metrics
    """
    folds = create_walk_forward_folds(
        timestamps, n_folds=n_folds, gap_bars=gap_bars if gap_bars is not None else 168
    )

    if not folds:
        raise ValueError("No valid folds created")

    logger.info(f"Evaluating {model_type} with {len(folds)} walk-forward folds")

    all_probs = []
    all_preds = []
    all_targets = []
    all_returns = []
    feature_importances: list[dict[str, float]] = []
    shap_importances: list[dict[str, float]] = []

    X = features_df.values
    feature_names = list(features_df.columns)

    for i, fold in enumerate(folds):
        X_train = X[fold.train_indices]
        y_train = targets[fold.train_indices]
        X_test = X[fold.test_indices]
        y_test = targets[fold.test_indices]
        returns_test = returns[fold.test_indices]

        logger.debug(
            f"Fold {i + 1}: train={len(fold.train_indices)}, test={len(fold.test_indices)}"
        )

        if model_type == "xgboost":
            probs, preds, model = train_xgboost(X_train, y_train, X_test)
            # XGBoost feature importance
            fi = dict(zip(feature_names, model.feature_importances_))
        else:
            probs, preds, model = train_lightgbm(X_train, y_train, X_test)
            # LightGBM feature importance
            fi = dict(zip(feature_names, model.feature_importances_))

        feature_importances.append(fi)

        # SHAP importance (only for last fold to save time)
        if i == len(folds) - 1:
            shap_imp = compute_shap_importance(model, X_test, feature_names)
            shap_importances.append(shap_imp)

        all_probs.extend(probs)
        all_preds.extend(preds)
        all_targets.extend(y_test)
        all_returns.extend(returns_test)

    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_returns = np.array(all_returns)

    # Compute metrics
    # IC: correlation between predicted probability and actual returns
    ic, ic_pvalue = compute_information_coefficient(all_probs, all_returns)

    # Brier score
    brier_score = float(np.mean((all_probs - all_targets) ** 2))

    # Directional accuracy
    directional_accuracy = float(np.mean(all_preds == all_targets))

    # Sharpe ratio on strategy returns (long if pred=1, else flat)
    strategy_returns = np.where(all_preds == 1, all_returns, 0.0)
    sharpe_ratio = compute_sharpe_ratio(strategy_returns, annualization_factor=365)

    # Aggregate feature importance
    avg_fi = {}
    for name in feature_names:
        vals = [fi.get(name, 0.0) for fi in feature_importances]
        avg_fi[name] = float(np.mean(vals))
    avg_fi = dict(sorted(avg_fi.items(), key=lambda x: -x[1]))

    # Use last SHAP importance
    avg_shap = shap_importances[-1] if shap_importances else {}

    return BaselineEvaluation(
        model_type=model_type,
        ic=ic,
        ic_pvalue=ic_pvalue,
        brier_score=brier_score,
        sharpe_ratio=sharpe_ratio,
        directional_accuracy=directional_accuracy,
        num_examples=len(all_targets),
        feature_importance=avg_fi,
        shap_importance=avg_shap,
    )


def load_adapter_evaluation(adapter_path: Path) -> dict[str, Any] | None:
    """
    Load adapter evaluation results from JSON.

    Args:
        adapter_path: Path to adapter directory

    Returns:
        Evaluation dict or None if not found
    """
    eval_path = adapter_path / "evaluation.json"
    if not eval_path.exists():
        return None

    with open(eval_path) as f:
        return json.load(f)


def find_best_adapter(
    adapters_dir: Path,
    adapter_type: str,
) -> tuple[Path | None, dict[str, Any] | None]:
    """
    Find the best adapter of a given type based on IC.

    Args:
        adapters_dir: Directory containing adapter subdirectories
        adapter_type: "dpo" or "grpo"

    Returns:
        Tuple of (best_path, best_evaluation) or (None, None)
    """
    if not adapters_dir.exists():
        return None, None

    best_path = None
    best_eval = None
    best_ic = -float("inf")

    for adapter_dir in adapters_dir.iterdir():
        if not adapter_dir.is_dir():
            continue

        # Check adapter type from name
        if adapter_type not in adapter_dir.name.lower():
            continue

        evaluation = load_adapter_evaluation(adapter_dir)
        if evaluation is None:
            continue

        ic = evaluation.get("ic", -float("inf"))
        if ic > best_ic:
            best_ic = ic
            best_path = adapter_dir
            best_eval = evaluation

    return best_path, best_eval


def format_comparison_table(
    xgb_eval: BaselineEvaluation,
    lgb_eval: BaselineEvaluation,
    grpo_eval: dict[str, Any] | None,
    dpo_eval: dict[str, Any] | None,
) -> str:
    """
    Format comparison table between models.

    Args:
        xgb_eval: XGBoost evaluation
        lgb_eval: LightGBM evaluation
        grpo_eval: Best GRPO adapter evaluation (or None)
        dpo_eval: Best DPO adapter evaluation (or None)

    Returns:
        Formatted comparison table
    """
    lines = []
    lines.append("=" * 90)
    lines.append("MODEL COMPARISON: XGBoost vs LightGBM vs LLM Adapters")
    lines.append("=" * 90)

    # Header
    cols = ["Metric", "XGBoost", "LightGBM"]
    if grpo_eval:
        cols.append("GRPO")
    if dpo_eval:
        cols.append("DPO")

    header = f"{'Metric':<25}"
    header += f"{'XGBoost':>15}"
    header += f"{'LightGBM':>15}"
    if grpo_eval:
        header += f"{'GRPO':>15}"
    if dpo_eval:
        header += f"{'DPO':>15}"
    lines.append(header)
    lines.append("-" * 90)

    # IC
    row = f"{'IC':<25}"
    row += f"{xgb_eval.ic:>15.4f}"
    row += f"{lgb_eval.ic:>15.4f}"
    if grpo_eval:
        row += f"{grpo_eval.get('ic', 0):>15.4f}"
    if dpo_eval:
        row += f"{dpo_eval.get('ic', 0):>15.4f}"
    lines.append(row)

    # p-value
    row = f"{'IC p-value':<25}"
    row += f"{xgb_eval.ic_pvalue:>15.4f}"
    row += f"{lgb_eval.ic_pvalue:>15.4f}"
    if grpo_eval:
        row += f"{grpo_eval.get('ic_pvalue', 1):>15.4f}"
    if dpo_eval:
        row += f"{dpo_eval.get('ic_pvalue', 1):>15.4f}"
    lines.append(row)

    # Brier
    row = f"{'Brier Score':<25}"
    row += f"{xgb_eval.brier_score:>15.4f}"
    row += f"{lgb_eval.brier_score:>15.4f}"
    if grpo_eval:
        row += f"{grpo_eval.get('brier_score', 0):>15.4f}"
    if dpo_eval:
        row += f"{dpo_eval.get('brier_score', 0):>15.4f}"
    lines.append(row)

    # Sharpe
    row = f"{'Sharpe Ratio':<25}"
    row += f"{xgb_eval.sharpe_ratio:>15.2f}"
    row += f"{lgb_eval.sharpe_ratio:>15.2f}"
    if grpo_eval:
        row += f"{'N/A':>15}"
    if dpo_eval:
        row += f"{'N/A':>15}"
    lines.append(row)

    # Directional Accuracy
    row = f"{'Directional Accuracy':<25}"
    row += f"{xgb_eval.directional_accuracy:>14.2%}"
    row += f"{lgb_eval.directional_accuracy:>14.2%}"
    if grpo_eval:
        row += f"{'N/A':>15}"
    if dpo_eval:
        row += f"{'N/A':>15}"
    lines.append(row)

    # Num examples
    row = f"{'Num Examples':<25}"
    row += f"{xgb_eval.num_examples:>15}"
    row += f"{lgb_eval.num_examples:>15}"
    if grpo_eval:
        row += f"{grpo_eval.get('num_examples', 0):>15}"
    if dpo_eval:
        row += f"{dpo_eval.get('num_examples', 0):>15}"
    lines.append(row)

    lines.append("=" * 90)

    # Winner analysis
    lines.append("\nANALYSIS:")

    # Compare XGBoost vs LightGBM
    if xgb_eval.ic > lgb_eval.ic:
        lines.append(f"  - XGBoost IC ({xgb_eval.ic:.4f}) > LightGBM IC ({lgb_eval.ic:.4f})")
    else:
        lines.append(f"  - LightGBM IC ({lgb_eval.ic:.4f}) >= XGBoost IC ({xgb_eval.ic:.4f})")

    best_baseline_ic = max(xgb_eval.ic, lgb_eval.ic)
    best_baseline = "XGBoost" if xgb_eval.ic > lgb_eval.ic else "LightGBM"

    # Compare vs LLM
    if grpo_eval:
        grpo_ic = grpo_eval.get("ic", 0)
        if grpo_ic > best_baseline_ic:
            lines.append(
                f"  - GRPO adapter ({grpo_ic:.4f}) BEATS {best_baseline} ({best_baseline_ic:.4f})"
            )
            lines.append("    => LLM reasoning adds value over numerical features alone")
        else:
            lines.append(
                f"  - {best_baseline} ({best_baseline_ic:.4f}) >= GRPO adapter ({grpo_ic:.4f})"
            )
            lines.append("    => Consider shifting LLM to sentiment/context overlay")

    if dpo_eval:
        dpo_ic = dpo_eval.get("ic", 0)
        if dpo_ic > best_baseline_ic:
            lines.append(
                f"  - DPO adapter ({dpo_ic:.4f}) BEATS {best_baseline} ({best_baseline_ic:.4f})"
            )
        else:
            lines.append(
                f"  - {best_baseline} ({best_baseline_ic:.4f}) >= DPO adapter ({dpo_ic:.4f})"
            )

    lines.append("")
    return "\n".join(lines)


def format_feature_importance_table(
    xgb_eval: BaselineEvaluation,
    lgb_eval: BaselineEvaluation,
    top_n: int = 10,
) -> str:
    """
    Format feature importance comparison table.

    Args:
        xgb_eval: XGBoost evaluation
        lgb_eval: LightGBM evaluation
        top_n: Number of top features to show

    Returns:
        Formatted feature importance table
    """
    lines = []
    lines.append("=" * 70)
    lines.append("FEATURE IMPORTANCE (Top 10)")
    lines.append("=" * 70)

    # XGBoost
    lines.append("\nXGBoost (by gain):")
    lines.append(f"{'Feature':<30} {'Importance':>15}")
    lines.append("-" * 50)
    for i, (feat, imp) in enumerate(xgb_eval.feature_importance.items()):
        if i >= top_n:
            break
        lines.append(f"{feat:<30} {imp:>15.4f}")

    # LightGBM
    lines.append("\nLightGBM (by gain):")
    lines.append(f"{'Feature':<30} {'Importance':>15}")
    lines.append("-" * 50)
    for i, (feat, imp) in enumerate(lgb_eval.feature_importance.items()):
        if i >= top_n:
            break
        lines.append(f"{feat:<30} {imp:>15.4f}")

    # SHAP
    if xgb_eval.shap_importance:
        lines.append("\nXGBoost SHAP (mean |SHAP|):")
        lines.append(f"{'Feature':<30} {'SHAP Value':>15}")
        lines.append("-" * 50)
        for i, (feat, imp) in enumerate(xgb_eval.shap_importance.items()):
            if i >= top_n:
                break
            lines.append(f"{feat:<30} {imp:>15.4f}")

    lines.append("=" * 70)

    # Recommendations
    lines.append("\nRECOMMENDATIONS:")

    # Find features with low importance
    low_imp_xgb = set()
    if xgb_eval.feature_importance:
        sorted_feats = list(xgb_eval.feature_importance.keys())
        if len(sorted_feats) > 5:
            low_imp_xgb = set(sorted_feats[-5:])

    low_imp_lgb = set()
    if lgb_eval.feature_importance:
        sorted_feats = list(lgb_eval.feature_importance.keys())
        if len(sorted_feats) > 5:
            low_imp_lgb = set(sorted_feats[-5:])

    # Features low in both
    consistently_low = low_imp_xgb & low_imp_lgb
    if consistently_low:
        lines.append(f"  - Consider dropping from prompts: {', '.join(consistently_low)}")

    # Top features
    if xgb_eval.shap_importance:
        top_shap = list(xgb_eval.shap_importance.keys())[:3]
        lines.append(f"  - Most predictive features (SHAP): {', '.join(top_shap)}")

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate XGBoost/LightGBM baselines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/grpo_training_data.jsonl"),
        help="Path to training data JSONL",
    )
    parser.add_argument(
        "--adapters-dir",
        type=Path,
        default=Path("adapters"),
        help="Directory containing adapter subdirectories",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation/xgboost_results.json"),
        help="Output JSON file for results",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="Number of walk-forward CV folds",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Include comparison with LLM adapters",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose logging",
    )

    args = parser.parse_args()

    # Configure logging
    logger.remove()
    log_level = "DEBUG" if args.verbose else "INFO"
    logger.add(
        sys.stderr,
        level=log_level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    )

    if not GBDT_AVAILABLE:
        logger.error("xgboost, lightgbm, and shap must be installed")
        logger.error("Run: pip install xgboost lightgbm shap")
        sys.exit(1)

    # Load data
    logger.info(f"Loading data from {args.data}")
    features_df, targets, returns, timestamps = load_training_data(args.data)

    logger.info(f"Features shape: {features_df.shape}")
    logger.info(f"Target distribution: {np.bincount(targets)}")

    # Train and evaluate XGBoost
    logger.info("Training XGBoost baseline...")
    xgb_eval = evaluate_baseline(
        "xgboost",
        features_df,
        targets,
        returns,
        timestamps,
        n_folds=args.n_folds,
    )
    logger.info(f"XGBoost IC: {xgb_eval.ic:.4f}, Accuracy: {xgb_eval.directional_accuracy:.2%}")

    # Train and evaluate LightGBM
    logger.info("Training LightGBM baseline...")
    lgb_eval = evaluate_baseline(
        "lightgbm",
        features_df,
        targets,
        returns,
        timestamps,
        n_folds=args.n_folds,
    )
    logger.info(f"LightGBM IC: {lgb_eval.ic:.4f}, Accuracy: {lgb_eval.directional_accuracy:.2%}")

    # Load LLM adapter evaluations if requested
    grpo_eval = None
    dpo_eval = None

    if args.compare and args.adapters_dir.exists():
        logger.info(f"Looking for adapter evaluations in {args.adapters_dir}")

        _, grpo_eval = find_best_adapter(args.adapters_dir, "grpo")
        if grpo_eval:
            logger.info(f"Found GRPO evaluation: IC={grpo_eval.get('ic', 0):.4f}")

        _, dpo_eval = find_best_adapter(args.adapters_dir, "dpo")
        if dpo_eval:
            logger.info(f"Found DPO evaluation: IC={dpo_eval.get('ic', 0):.4f}")

    # Print comparison table
    print()
    print(format_comparison_table(xgb_eval, lgb_eval, grpo_eval, dpo_eval))
    print(format_feature_importance_table(xgb_eval, lgb_eval))

    # Save results
    results = {
        "xgboost": xgb_eval.to_dict(),
        "lightgbm": lgb_eval.to_dict(),
        "grpo_adapter": grpo_eval,
        "dpo_adapter": dpo_eval,
        "data_path": str(args.data),
        "n_folds": args.n_folds,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
