"""
XGBoost signal generator configuration.

Centralizes all hyperparameters for XGBoost model used in production signals.
This is the ONLY file autoresearch edits for hyperparameter tuning.

Extracted from evaluation/xgboost_baseline.py for production use.
"""

from dataclasses import dataclass
from typing import Any

from config.fee_model import FeeModelSettings


# =============================================================================
# Feature Configuration (17 indicators + 2 extras)
# =============================================================================

# Core 17 technical indicators (same as LLM sees in prompts)
INDICATOR_FEATURES: list[str] = [
    # Price/Trend (8)
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

# Additional features (when available)
EXTRA_FEATURES: list[str] = [
    "funding_rate",
    "open_interest",
]

# All features combined
FEATURE_LIST: list[str] = INDICATOR_FEATURES + EXTRA_FEATURES


# =============================================================================
# XGBoost Hyperparameters
# =============================================================================

XGB_PARAMS: dict[str, Any] = {
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "use_label_encoder": False,
    "eval_metric": "logloss",
    "verbosity": 0,
    # Scale pos weight derived from class weights
    # Calculated as: negative_class_weight / positive_class_weight
    "scale_pos_weight": 1.0,
}


# =============================================================================
# Class Weights (Asymmetric Penalties)
# =============================================================================
# False bullish is costlier than false bearish (asymmetric risk)
# These align with GRPO reward asymmetry: false bullish -1.5x, false bearish -0.8x

CLASS_WEIGHTS: dict[str, float] = {
    "LONG": 1.0,  # Base weight for LONG predictions
    "SHORT": 1.0,  # Base weight for SHORT predictions
    "FLAT": 0.5,  # Lower weight for FLAT (less decisive)
    # Penalty multipliers for incorrect predictions
    "false_bullish_penalty": 1.5,  # Predicted LONG, was SHORT/FLAT
    "false_bearish_penalty": 0.8,  # Predicted SHORT, was LONG/FLAT
}


# =============================================================================
# Walk-Forward Cross-Validation Configuration
# =============================================================================


@dataclass(frozen=True)
class WalkForwardConfig:
    """Walk-forward validation parameters."""

    n_folds: int = 5
    train_ratio: float = 0.7
    gap_bars: int = 24  # Gap between train/test to avoid leakage
    min_train_size: int = 50
    min_test_size: int = 20


WALK_FORWARD_CONFIG = WalkForwardConfig()


# =============================================================================
# Label Threshold Configuration
# =============================================================================


@dataclass(frozen=True)
class LabelThresholdConfig:
    """
    Fee-adjusted thresholds for LONG/SHORT/FLAT classification.

    Uses FeeModelSettings to compute minimum profitable return.
    """

    # Default holding period in 8-hour funding periods
    default_holding_periods_8h: float = 1.0

    def get_threshold_pct(self, holding_periods_8h: float | None = None) -> float:
        """
        Get fee-adjusted threshold for direction classification.

        Returns are classified as:
        - LONG: net_return > threshold
        - SHORT: net_return < -threshold
        - FLAT: otherwise

        Args:
            holding_periods_8h: Holding period in 8h units (default: 1.0)

        Returns:
            Threshold as percentage
        """
        periods = holding_periods_8h or self.default_holding_periods_8h
        fee_model = FeeModelSettings()
        return fee_model.minimum_profitable_return_pct(periods)


LABEL_THRESHOLD = LabelThresholdConfig()


# =============================================================================
# Retraining Configuration
# =============================================================================

# Minimum verified signals before triggering retrain
MIN_SIGNALS_FOR_RETRAIN: int = 200

# Model persistence paths
MODEL_CHECKPOINT_DIR: str = "models/xgboost/"
MODEL_FILENAME: str = "xgboost_signal_model.json"


# =============================================================================
# Production Probability Thresholds (from signal-layer.md)
# =============================================================================

PROBABILITY_THRESHOLDS: dict[str, float] = {
    "flat_threshold": 0.55,  # XGBoost prob < 0.55 → FLAT
    "half_position": 0.55,  # prob >= 0.55 + conflicting regime → half
    "full_position": 0.65,  # prob >= 0.65 + LLM confirms → full
}
