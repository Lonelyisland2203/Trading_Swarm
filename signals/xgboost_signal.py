"""
XGBoost-based signal generator for production use.

Generates trading signals using XGBoost classifier trained on technical indicators.
Uses point-in-time safe data fetching via get_ohlcv_as_of().

Contract (from signal-layer.md):
- Input: 17 indicators as feature vector (same compute_ functions)
- Output: {direction: LONG|SHORT|FLAT, probability: float, features: dict}
- Retrain: walk-forward, triggered every 200 verified signals or by autoresearch
"""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from loguru import logger

from data.indicators import compute_all_indicators
from signals.preflight import run_preflight_checks
from signals.signal_models import SignalDirection
from signals.verification import load_verified_results
from signals.xgboost_config import (
    FEATURE_LIST,
    INDICATOR_FEATURES,
    MIN_SIGNALS_FOR_RETRAIN,
    MODEL_CHECKPOINT_DIR,
    MODEL_FILENAME,
    WALK_FORWARD_CONFIG,
    XGB_PARAMS,
)

if TYPE_CHECKING:
    from data.market_data import MarketDataService

try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    xgb = None


# Default lookback for feature computation
DEFAULT_LOOKBACK_BARS = 100


@dataclass(slots=True)
class XGBoostSignal:
    """
    XGBoost-generated trading signal.

    Attributes:
        symbol: Trading pair (e.g., "BTC/USDT")
        timeframe: Timeframe string (e.g., "1h", "4h")
        direction: Signal direction (LONG/SHORT/FLAT)
        probability: Model probability for bullish outcome (0-1)
        confidence: Confidence score (abs(prob - 0.5) * 2)
        features: Dictionary of feature values used
        timestamp: Signal generation timestamp
        model_version: Optional model version identifier
    """

    symbol: str
    timeframe: str
    direction: SignalDirection
    probability: float
    confidence: float
    features: dict[str, float | None]
    timestamp: datetime
    model_version: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data


def clip_probability(prob: float) -> float:
    """
    Clip probability to valid [0, 1] range.

    Args:
        prob: Raw probability value

    Returns:
        Clipped probability in [0, 1]
    """
    return max(0.0, min(1.0, prob))


def map_probability_to_direction(probability: float) -> SignalDirection:
    """
    Map probability to trading direction.

    Thresholds:
    - prob >= 0.55 → LONG
    - prob <= 0.45 → SHORT
    - otherwise → FLAT

    Args:
        probability: Model probability (0-1)

    Returns:
        Signal direction (LONG/SHORT/FLAT)
    """
    # Use explicit thresholds to avoid floating point precision issues
    long_threshold = 0.55
    short_threshold = 0.45

    if probability >= long_threshold:
        return "LONG"
    elif probability <= short_threshold:
        return "SHORT"
    else:
        return "FLAT"


def extract_features_from_ohlcv(df: pd.DataFrame) -> dict[str, float | None]:
    """
    Extract feature dictionary from OHLCV DataFrame.

    Uses compute_all_indicators() to compute all 17 technical indicators,
    then extracts scalar values for the latest bar.

    Args:
        df: OHLCV DataFrame with columns [timestamp, open, high, low, close, volume]

    Returns:
        Dictionary of feature name -> value (None if not computable)
    """
    if df.empty:
        return {f: None for f in FEATURE_LIST}

    # Compute all indicators
    indicators = compute_all_indicators(df, include_volume=True, include_structure=True)

    # Extract feature values
    features: dict[str, float | None] = {}

    for feature_name in INDICATOR_FEATURES:
        value = indicators.get(feature_name)
        if value is not None and not (isinstance(value, float) and np.isnan(value)):
            features[feature_name] = (
                float(value) if isinstance(value, (int, float, np.number)) else None
            )
        else:
            features[feature_name] = None

    # Extra features (funding_rate, open_interest) - not yet implemented
    features["funding_rate"] = None
    features["open_interest"] = None

    return features


def get_feature_names() -> list[str]:
    """
    Get ordered list of feature names.

    Returns:
        List of feature names in consistent order
    """
    return FEATURE_LIST.copy()


def create_xgboost_model() -> Any:
    """
    Create XGBoost classifier with config parameters.

    Returns:
        XGBClassifier instance configured from XGB_PARAMS
    """
    if not XGBOOST_AVAILABLE:
        raise ImportError("xgboost not installed. Run: pip install xgboost")

    return xgb.XGBClassifier(**XGB_PARAMS)


def load_model(model_path: Path | None = None) -> Any | None:
    """
    Load trained XGBoost model from checkpoint.

    Args:
        model_path: Path to model file. If None, uses default path.

    Returns:
        Loaded model or None if not found
    """
    if not XGBOOST_AVAILABLE:
        logger.warning("XGBoost not available")
        return None

    if model_path is None:
        model_path = Path(MODEL_CHECKPOINT_DIR) / MODEL_FILENAME

    if not model_path.exists():
        logger.warning(f"Model not found at {model_path}")
        return None

    try:
        model = xgb.XGBClassifier()
        model.load_model(model_path)
        logger.info(f"Loaded XGBoost model from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None


def save_model(model: Any, model_path: Path | None = None) -> None:
    """
    Save trained XGBoost model to checkpoint.

    Args:
        model: Trained XGBClassifier
        model_path: Path to save model. If None, uses default path.
    """
    if model_path is None:
        model_path = Path(MODEL_CHECKPOINT_DIR) / MODEL_FILENAME

    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(model_path)
    logger.info(f"Saved XGBoost model to {model_path}")


def create_walk_forward_splits(
    timestamps: np.ndarray,
    n_folds: int = 5,
    train_ratio: float = 0.7,
    gap_bars: int = 24,
) -> list[dict[str, np.ndarray]]:
    """
    Create walk-forward cross-validation splits.

    Ensures temporal ordering: train data always before validation data.

    Args:
        timestamps: Array of timestamps in milliseconds
        n_folds: Number of folds
        train_ratio: Fraction of fold data for training
        gap_bars: Gap between train and validation to avoid leakage

    Returns:
        List of dicts with 'train_indices' and 'val_indices' arrays
    """
    n_samples = len(timestamps)
    sorted_indices = np.argsort(timestamps)

    splits = []
    fold_size = n_samples // n_folds

    for i in range(n_folds):
        # Expanding window: use all data up to fold boundary for training
        fold_end = (i + 1) * fold_size
        if i == n_folds - 1:
            fold_end = n_samples

        # Validation set is the last portion of this fold
        val_size = max(int(fold_size * (1 - train_ratio)), WALK_FORWARD_CONFIG.min_test_size)
        val_start = fold_end - val_size
        train_end = val_start - gap_bars

        if train_end <= WALK_FORWARD_CONFIG.min_train_size:
            continue

        train_indices = sorted_indices[:train_end]
        val_indices = sorted_indices[val_start:fold_end]

        if len(train_indices) < WALK_FORWARD_CONFIG.min_train_size:
            continue
        if len(val_indices) < WALK_FORWARD_CONFIG.min_test_size:
            continue

        splits.append(
            {
                "train_indices": train_indices,
                "val_indices": val_indices,
            }
        )

    return splits


def check_retrain_trigger(signals_count: int) -> bool:
    """
    Check if retrain should be triggered based on signal count.

    Args:
        signals_count: Number of verified signals since last training

    Returns:
        True if retrain should be triggered
    """
    return signals_count >= MIN_SIGNALS_FOR_RETRAIN


def get_retrain_threshold() -> int:
    """
    Get retrain threshold from config.

    Returns:
        Number of signals required to trigger retrain
    """
    return MIN_SIGNALS_FOR_RETRAIN


def should_trigger_retrain() -> bool:
    """
    Check if retrain should be triggered based on verified results.

    Integrates with verification.py to count verified signals.

    Returns:
        True if retrain should be triggered
    """
    try:
        results = load_verified_results()
        return check_retrain_trigger(len(results))
    except Exception as e:
        logger.warning(f"Failed to check retrain trigger: {e}")
        return False


def features_to_array(features: dict[str, float | None]) -> np.ndarray:
    """
    Convert feature dictionary to numpy array for model input.

    Missing values are imputed with 0 (will be handled by model).

    Args:
        features: Feature dictionary

    Returns:
        Numpy array of feature values
    """
    values = []
    for feature_name in FEATURE_LIST:
        value = features.get(feature_name)
        values.append(value if value is not None else 0.0)
    return np.array(values).reshape(1, -1)


async def generate_xgboost_signal(
    symbol: str,
    timeframe: str,
    as_of: int,
    market_data_service: "MarketDataService",
    lookback_bars: int = DEFAULT_LOOKBACK_BARS,
    skip_preflight: bool = False,
) -> XGBoostSignal | None:
    """
    Generate XGBoost trading signal for a symbol.

    Uses point-in-time safe data fetching via get_ohlcv_as_of().

    Args:
        symbol: Trading pair (e.g., "BTC/USDT")
        timeframe: Timeframe string (e.g., "1h", "4h")
        as_of: Point-in-time timestamp in milliseconds
        market_data_service: Market data service for fetching OHLCV
        lookback_bars: Number of bars to fetch for indicator computation
        skip_preflight: Skip preflight checks (for testing)

    Returns:
        XGBoostSignal if successful, None if preflight fails or model unavailable
    """
    # Run preflight checks
    if not skip_preflight:
        preflight_result = run_preflight_checks()
        if not preflight_result.passed:
            logger.warning(f"Preflight failed: {preflight_result.reason}")
            return None

    # Load model
    model = load_model()
    if model is None:
        logger.warning("No trained model available")
        return None

    # Fetch point-in-time safe OHLCV data
    try:
        df = await market_data_service.get_ohlcv_as_of(
            symbol=symbol,
            timeframe=timeframe,
            as_of=as_of,
            lookback_bars=lookback_bars,
        )
    except Exception as e:
        logger.error(f"Failed to fetch OHLCV data: {e}")
        return None

    if df is None or df.empty:
        logger.warning(f"No data available for {symbol} {timeframe}")
        return None

    # Extract features
    features = extract_features_from_ohlcv(df)

    # Convert to array for prediction
    X = features_to_array(features)

    # Get probability
    try:
        proba = model.predict_proba(X)[0, 1]  # Probability of class 1 (bullish)
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return None

    # Clip and map to direction
    probability = clip_probability(float(proba))
    direction = map_probability_to_direction(probability)

    # Compute confidence as distance from 0.5
    confidence = abs(probability - 0.5) * 2.0

    return XGBoostSignal(
        symbol=symbol,
        timeframe=timeframe,
        direction=direction,
        probability=probability,
        confidence=confidence,
        features=features,
        timestamp=datetime.now(timezone.utc),
    )
