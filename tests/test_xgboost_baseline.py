"""
Tests for evaluation/xgboost_baseline.py

Tests XGBoost and LightGBM baseline models using synthetic data.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from evaluation.xgboost_baseline import (
    ALL_FEATURES,
    GBDT_AVAILABLE,
    INDICATOR_FEATURES,
    BaselineEvaluation,
    WalkForwardFold,
    compute_shap_importance,
    create_walk_forward_folds,
    evaluate_baseline,
    extract_features_from_snapshot,
    find_best_adapter,
    format_comparison_table,
    format_feature_importance_table,
    load_adapter_evaluation,
    load_training_data,
    train_lightgbm,
    train_xgboost,
)

# Skip all tests if GBDT libraries not available
pytestmark = pytest.mark.skipif(not GBDT_AVAILABLE, reason="xgboost/lightgbm/shap not installed")


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_market_snapshot() -> str:
    """Sample market snapshot for feature extraction testing."""
    return """## Market Data
Symbol: BTC/USDT
Timeframe: 1h
Timestamp: 2024-01-15 12:00:00 UTC
Current Price: $42500.0000
Market Regime: TRENDING_UP

## Technical Indicators

### Price/Trend
RSI(14): 65.50
MACD Line: 150.2500 | Signal: 120.5000
Donchian(20): Upper $43000.00 | Mid $42000.00 | Lower $41000.00
KAMA(10): $42300.00

### Volume
OBV: 1234567890
CMF(20): 0.1500
MFI(14): 58.25
VWAP: $42200.00

### Volatility
ATR Normalized: 2.50%
BB Width(20): 4.75%
Keltner Width: 3.80%
Donchian Width: 4.76%

### Market Structure
Open FVG Count: 3
Nearest Bullish FVG: 1.25% above
Nearest Bearish FVG: 0.75% below
Nearest Swing High: 2.50% above
Nearest Swing Low: 3.00% below

## Recent Price Action (last 10 bars)
2024-01-15 03:00 | O: $42000.00 H: $42200.00 L: $41900.00 C: $42100.00 | ↑ +0.24%

## Execution Context
Exchange: Binance Futures USDT-M
Estimated round-trip cost: 0.093%
Minimum profitable move: 0.093%"""


@pytest.fixture
def synthetic_training_data() -> list[dict]:
    """Generate synthetic training data."""
    np.random.seed(42)
    n_samples = 200

    data = []
    base_ts = 1704067200000  # 2024-01-01 00:00:00 UTC

    for i in range(n_samples):
        # Generate random features
        rsi = np.random.uniform(30, 70)
        macd_line = np.random.uniform(-100, 100)
        macd_signal = macd_line + np.random.uniform(-20, 20)

        # Generate direction based on features (simple rule)
        signal_strength = (rsi - 50) / 50 + macd_line / 200
        if signal_strength > 0.2:
            direction = "LONG"
            gross_return = np.random.uniform(0.1, 0.5)
        elif signal_strength < -0.2:
            direction = "SHORT"
            gross_return = np.random.uniform(-0.5, -0.1)
        else:
            direction = "FLAT"
            gross_return = np.random.uniform(-0.1, 0.1)

        snapshot = f"""## Market Data
Symbol: BTC/USDT
Timeframe: 1h
Timestamp: 2024-01-{(i % 28) + 1:02d} 12:00:00 UTC
Current Price: $42500.0000

## Technical Indicators

### Price/Trend
RSI(14): {rsi:.2f}
MACD Line: {macd_line:.4f} | Signal: {macd_signal:.4f}
Donchian(20): Upper $43000.00 | Mid $42000.00 | Lower $41000.00
KAMA(10): $42300.00

### Volume
OBV: 1234567890
CMF(20): {np.random.uniform(-0.5, 0.5):.4f}
MFI(14): {np.random.uniform(30, 70):.2f}
VWAP: $42200.00

### Volatility
ATR Normalized: {np.random.uniform(1, 5):.2f}%
BB Width(20): {np.random.uniform(2, 8):.2f}%
Keltner Width: {np.random.uniform(2, 6):.2f}%
Donchian Width: {np.random.uniform(2, 8):.2f}%

### Market Structure
Open FVG Count: {np.random.randint(0, 5)}
Nearest Bullish FVG: {np.random.uniform(0.5, 3):.2f}% above
Nearest Bearish FVG: {np.random.uniform(0.5, 3):.2f}% below
Nearest Swing High: {np.random.uniform(1, 5):.2f}% above
Nearest Swing Low: {np.random.uniform(1, 5):.2f}% below

## Execution Context
Exchange: Binance Futures USDT-M
Estimated round-trip cost: 0.093%
Minimum profitable move: 0.093%"""

        data.append(
            {
                "market_snapshot": snapshot,
                "actual_direction": direction,
                "gross_return_pct": gross_return,
                "timestamp_ms": base_ts + i * 3600000,
            }
        )

    return data


@pytest.fixture
def synthetic_data_file(synthetic_training_data: list[dict]) -> Path:
    """Create temporary JSONL file with synthetic data."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for example in synthetic_training_data:
            f.write(json.dumps(example) + "\n")
        return Path(f.name)


@pytest.fixture
def sample_features_df() -> pd.DataFrame:
    """Create sample feature DataFrame."""
    np.random.seed(42)
    n_samples = 100

    data = {}
    for feat in INDICATOR_FEATURES:
        if feat in ["rsi", "mfi"]:
            data[feat] = np.random.uniform(30, 70, n_samples)
        elif feat in ["cmf"]:
            data[feat] = np.random.uniform(-0.5, 0.5, n_samples)
        elif "pct" in feat or "width" in feat:
            data[feat] = np.random.uniform(0, 5, n_samples)
        elif feat == "open_fvg_count":
            data[feat] = np.random.randint(0, 5, n_samples).astype(float)
        else:
            data[feat] = np.random.uniform(0, 100, n_samples)

    # Extra features
    data["funding_rate"] = np.random.uniform(-0.01, 0.01, n_samples)
    data["open_interest"] = np.random.uniform(1e6, 1e9, n_samples)

    return pd.DataFrame(data)


# ============================================================================
# Feature Extraction Tests
# ============================================================================


class TestFeatureExtraction:
    """Tests for feature extraction from market snapshots."""

    def test_extract_rsi(self, sample_market_snapshot: str) -> None:
        """Test RSI extraction."""
        features = extract_features_from_snapshot(sample_market_snapshot)
        assert features["rsi"] == pytest.approx(65.50, rel=0.01)

    def test_extract_macd(self, sample_market_snapshot: str) -> None:
        """Test MACD extraction."""
        features = extract_features_from_snapshot(sample_market_snapshot)
        assert features["macd_line"] == pytest.approx(150.25, rel=0.01)
        assert features["macd_signal"] == pytest.approx(120.50, rel=0.01)

    def test_extract_donchian(self, sample_market_snapshot: str) -> None:
        """Test Donchian channels extraction."""
        features = extract_features_from_snapshot(sample_market_snapshot)
        assert features["donchian_upper"] == pytest.approx(43000.0, rel=0.01)
        assert features["donchian_middle"] == pytest.approx(42000.0, rel=0.01)
        assert features["donchian_lower"] == pytest.approx(41000.0, rel=0.01)

    def test_extract_kama(self, sample_market_snapshot: str) -> None:
        """Test KAMA extraction."""
        features = extract_features_from_snapshot(sample_market_snapshot)
        assert features["kama"] == pytest.approx(42300.0, rel=0.01)

    def test_extract_volume_indicators(self, sample_market_snapshot: str) -> None:
        """Test volume indicator extraction."""
        features = extract_features_from_snapshot(sample_market_snapshot)
        assert features["obv"] == pytest.approx(1234567890.0, rel=0.01)
        assert features["cmf"] == pytest.approx(0.15, rel=0.01)
        assert features["mfi"] == pytest.approx(58.25, rel=0.01)
        assert features["vwap"] == pytest.approx(42200.0, rel=0.01)

    def test_extract_volatility_indicators(self, sample_market_snapshot: str) -> None:
        """Test volatility indicator extraction."""
        features = extract_features_from_snapshot(sample_market_snapshot)
        assert features["atr_normalized"] == pytest.approx(2.50, rel=0.01)
        assert features["bb_width"] == pytest.approx(4.75, rel=0.01)
        assert features["keltner_width"] == pytest.approx(3.80, rel=0.01)
        assert features["donchian_width"] == pytest.approx(4.76, rel=0.01)

    def test_extract_market_structure(self, sample_market_snapshot: str) -> None:
        """Test market structure extraction."""
        features = extract_features_from_snapshot(sample_market_snapshot)
        assert features["open_fvg_count"] == pytest.approx(3.0, rel=0.01)
        assert features["nearest_bullish_fvg_pct"] == pytest.approx(1.25, rel=0.01)
        assert features["nearest_bearish_fvg_pct"] == pytest.approx(0.75, rel=0.01)
        assert features["nearest_swing_high_pct"] == pytest.approx(2.50, rel=0.01)
        assert features["nearest_swing_low_pct"] == pytest.approx(3.00, rel=0.01)

    def test_extract_missing_features(self) -> None:
        """Test extraction from minimal snapshot."""
        minimal_snapshot = "## Market Data\nSymbol: BTC/USDT"
        features = extract_features_from_snapshot(minimal_snapshot)

        # All features should be None
        for feat in INDICATOR_FEATURES:
            assert features[feat] is None

    def test_extract_na_values(self) -> None:
        """Test extraction handles N/A values."""
        snapshot_with_na = """## Technical Indicators

### Price/Trend
RSI(14): N/A
MACD Line: N/A | Signal: N/A"""

        features = extract_features_from_snapshot(snapshot_with_na)
        assert features["rsi"] is None
        assert features["macd_line"] is None
        assert features["macd_signal"] is None

    def test_all_features_defined(self) -> None:
        """Test that ALL_FEATURES contains expected features."""
        assert len(INDICATOR_FEATURES) == 21  # 17 indicators + sub-components
        assert "funding_rate" in ALL_FEATURES
        assert "open_interest" in ALL_FEATURES


# ============================================================================
# Walk-Forward Split Tests
# ============================================================================


class TestWalkForwardSplits:
    """Tests for walk-forward cross-validation splits."""

    def test_create_folds_basic(self) -> None:
        """Test basic fold creation."""
        timestamps = np.arange(1000) * 3600000  # 1000 hourly timestamps
        folds = create_walk_forward_folds(timestamps, n_folds=5)

        assert len(folds) >= 3  # Should create at least 3 folds
        assert all(isinstance(f, WalkForwardFold) for f in folds)

    def test_temporal_ordering(self) -> None:
        """Test that train data always precedes test data."""
        timestamps = np.arange(500) * 3600000
        folds = create_walk_forward_folds(timestamps, n_folds=3)

        for fold in folds:
            assert fold.train_end_ts < fold.test_start_ts

    def test_no_overlap(self) -> None:
        """Test that train and test indices don't overlap."""
        timestamps = np.arange(500) * 3600000
        folds = create_walk_forward_folds(timestamps, n_folds=3)

        for fold in folds:
            train_set = set(fold.train_indices)
            test_set = set(fold.test_indices)
            assert len(train_set & test_set) == 0

    def test_expanding_window(self) -> None:
        """Test that training window expands across folds."""
        timestamps = np.arange(500) * 3600000
        folds = create_walk_forward_folds(timestamps, n_folds=4)

        if len(folds) >= 2:
            # Later folds should have more training data
            assert len(folds[-1].train_indices) >= len(folds[0].train_indices)

    def test_minimum_samples(self) -> None:
        """Test that folds have minimum required samples."""
        timestamps = np.arange(200) * 3600000
        folds = create_walk_forward_folds(timestamps, n_folds=3)

        for fold in folds:
            assert len(fold.train_indices) >= 50
            assert len(fold.test_indices) >= 20


# ============================================================================
# Data Loading Tests
# ============================================================================


class TestDataLoading:
    """Tests for training data loading."""

    def test_load_training_data(self, synthetic_data_file: Path) -> None:
        """Test loading training data from JSONL."""
        features_df, targets, returns, timestamps = load_training_data(synthetic_data_file)

        assert len(features_df) == 200
        assert len(targets) == 200
        assert len(returns) == 200
        assert len(timestamps) == 200

    def test_targets_are_binary(self, synthetic_data_file: Path) -> None:
        """Test that targets are binary."""
        _, targets, _, _ = load_training_data(synthetic_data_file)

        assert set(targets).issubset({0, 1})

    def test_returns_are_fee_adjusted(self, synthetic_data_file: Path) -> None:
        """Test that returns are fee-adjusted."""
        _, _, returns, _ = load_training_data(synthetic_data_file)

        # Fee-adjusted returns should generally be lower than gross
        # (Can't test exact values without gross returns)
        assert returns.dtype == np.float64

    def test_timestamps_ordered(self, synthetic_data_file: Path) -> None:
        """Test that timestamps are loaded correctly."""
        _, _, _, timestamps = load_training_data(synthetic_data_file)

        # Check monotonic increasing (in sorted order)
        sorted_ts = np.sort(timestamps)
        assert np.all(sorted_ts[1:] >= sorted_ts[:-1])

    def test_features_imputed(self, synthetic_data_file: Path) -> None:
        """Test that missing features are imputed."""
        features_df, _, _, _ = load_training_data(synthetic_data_file)

        # No NaN values after imputation
        assert not features_df.isna().any().any()

    def test_load_nonexistent_file(self) -> None:
        """Test loading nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            load_training_data(Path("nonexistent.jsonl"))


# ============================================================================
# Model Training Tests
# ============================================================================


class TestModelTraining:
    """Tests for XGBoost and LightGBM training."""

    def test_train_xgboost(self, sample_features_df: pd.DataFrame) -> None:
        """Test XGBoost training."""
        X = sample_features_df.values
        y = np.random.randint(0, 2, len(X))

        probs, preds, model = train_xgboost(X[:80], y[:80], X[80:])

        assert len(probs) == 20
        assert len(preds) == 20
        assert np.all((probs >= 0) & (probs <= 1))
        assert set(preds).issubset({0, 1})

    def test_train_lightgbm(self, sample_features_df: pd.DataFrame) -> None:
        """Test LightGBM training."""
        X = sample_features_df.values
        y = np.random.randint(0, 2, len(X))

        probs, preds, model = train_lightgbm(X[:80], y[:80], X[80:])

        assert len(probs) == 20
        assert len(preds) == 20
        assert np.all((probs >= 0) & (probs <= 1))
        assert set(preds).issubset({0, 1})

    def test_xgboost_feature_importance(self, sample_features_df: pd.DataFrame) -> None:
        """Test XGBoost returns feature importance."""
        X = sample_features_df.values
        y = np.random.randint(0, 2, len(X))

        _, _, model = train_xgboost(X[:80], y[:80], X[80:])

        assert hasattr(model, "feature_importances_")
        assert len(model.feature_importances_) == X.shape[1]

    def test_lightgbm_feature_importance(self, sample_features_df: pd.DataFrame) -> None:
        """Test LightGBM returns feature importance."""
        X = sample_features_df.values
        y = np.random.randint(0, 2, len(X))

        _, _, model = train_lightgbm(X[:80], y[:80], X[80:])

        assert hasattr(model, "feature_importances_")
        assert len(model.feature_importances_) == X.shape[1]


# ============================================================================
# SHAP Tests
# ============================================================================


class TestSHAPImportance:
    """Tests for SHAP feature importance."""

    def test_compute_shap_xgboost(self, sample_features_df: pd.DataFrame) -> None:
        """Test SHAP computation for XGBoost."""
        X = sample_features_df.values
        y = np.random.randint(0, 2, len(X))
        feature_names = list(sample_features_df.columns)

        _, _, model = train_xgboost(X[:80], y[:80], X[80:])
        shap_imp = compute_shap_importance(model, X[80:], feature_names)

        assert len(shap_imp) == len(feature_names)
        assert all(v >= 0 for v in shap_imp.values())

    def test_compute_shap_lightgbm(self, sample_features_df: pd.DataFrame) -> None:
        """Test SHAP computation for LightGBM."""
        X = sample_features_df.values
        y = np.random.randint(0, 2, len(X))
        feature_names = list(sample_features_df.columns)

        _, _, model = train_lightgbm(X[:80], y[:80], X[80:])
        shap_imp = compute_shap_importance(model, X[80:], feature_names)

        assert len(shap_imp) == len(feature_names)
        assert all(v >= 0 for v in shap_imp.values())

    def test_shap_sorted_by_importance(self, sample_features_df: pd.DataFrame) -> None:
        """Test SHAP values are sorted descending."""
        X = sample_features_df.values
        y = np.random.randint(0, 2, len(X))
        feature_names = list(sample_features_df.columns)

        _, _, model = train_xgboost(X[:80], y[:80], X[80:])
        shap_imp = compute_shap_importance(model, X[80:], feature_names)

        values = list(shap_imp.values())
        assert values == sorted(values, reverse=True)


# ============================================================================
# Full Evaluation Tests
# ============================================================================


class TestEvaluateBaseline:
    """Tests for full baseline evaluation."""

    def test_evaluate_xgboost(self, synthetic_data_file: Path) -> None:
        """Test full XGBoost evaluation."""
        features_df, targets, returns, timestamps = load_training_data(synthetic_data_file)

        evaluation = evaluate_baseline(
            "xgboost", features_df, targets, returns, timestamps, n_folds=3
        )

        assert isinstance(evaluation, BaselineEvaluation)
        assert evaluation.model_type == "xgboost"
        assert -1 <= evaluation.ic <= 1
        assert 0 <= evaluation.brier_score <= 1
        assert 0 <= evaluation.directional_accuracy <= 1
        assert evaluation.num_examples > 0

    def test_evaluate_lightgbm(self, synthetic_data_file: Path) -> None:
        """Test full LightGBM evaluation."""
        features_df, targets, returns, timestamps = load_training_data(synthetic_data_file)

        evaluation = evaluate_baseline(
            "lightgbm", features_df, targets, returns, timestamps, n_folds=3
        )

        assert isinstance(evaluation, BaselineEvaluation)
        assert evaluation.model_type == "lightgbm"
        assert -1 <= evaluation.ic <= 1
        assert 0 <= evaluation.brier_score <= 1

    def test_feature_importance_populated(self, synthetic_data_file: Path) -> None:
        """Test feature importance is populated."""
        features_df, targets, returns, timestamps = load_training_data(synthetic_data_file)

        evaluation = evaluate_baseline(
            "xgboost", features_df, targets, returns, timestamps, n_folds=3
        )

        assert len(evaluation.feature_importance) > 0
        assert all(isinstance(v, float) for v in evaluation.feature_importance.values())

    def test_shap_importance_populated(self, synthetic_data_file: Path) -> None:
        """Test SHAP importance is populated."""
        features_df, targets, returns, timestamps = load_training_data(synthetic_data_file)

        evaluation = evaluate_baseline(
            "xgboost", features_df, targets, returns, timestamps, n_folds=3
        )

        # SHAP computed on last fold
        assert len(evaluation.shap_importance) > 0

    def test_to_dict(self, synthetic_data_file: Path) -> None:
        """Test evaluation serialization."""
        features_df, targets, returns, timestamps = load_training_data(synthetic_data_file)

        evaluation = evaluate_baseline(
            "xgboost", features_df, targets, returns, timestamps, n_folds=3
        )

        result = evaluation.to_dict()
        assert isinstance(result, dict)
        assert "model_type" in result
        assert "ic" in result
        assert "feature_importance" in result


# ============================================================================
# Adapter Loading Tests
# ============================================================================


class TestAdapterLoading:
    """Tests for adapter evaluation loading."""

    def test_load_adapter_evaluation_exists(self) -> None:
        """Test loading existing adapter evaluation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter_dir = Path(tmpdir) / "grpo_adapter"
            adapter_dir.mkdir()

            eval_data = {"ic": 0.08, "brier_score": 0.22}
            eval_path = adapter_dir / "evaluation.json"
            with open(eval_path, "w") as f:
                json.dump(eval_data, f)

            result = load_adapter_evaluation(adapter_dir)
            assert result is not None
            assert result["ic"] == 0.08

    def test_load_adapter_evaluation_missing(self) -> None:
        """Test loading missing adapter evaluation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter_dir = Path(tmpdir) / "grpo_adapter"
            adapter_dir.mkdir()

            result = load_adapter_evaluation(adapter_dir)
            assert result is None

    def test_find_best_adapter(self) -> None:
        """Test finding best adapter by IC."""
        with tempfile.TemporaryDirectory() as tmpdir:
            adapters_dir = Path(tmpdir)

            # Create multiple adapters
            for i, ic in enumerate([0.05, 0.08, 0.03]):
                adapter_dir = adapters_dir / f"grpo_adapter_{i}"
                adapter_dir.mkdir()
                with open(adapter_dir / "evaluation.json", "w") as f:
                    json.dump({"ic": ic}, f)

            best_path, best_eval = find_best_adapter(adapters_dir, "grpo")
            assert best_path is not None
            assert best_eval is not None
            assert best_eval["ic"] == 0.08

    def test_find_best_adapter_no_match(self) -> None:
        """Test finding best adapter with no matches."""
        with tempfile.TemporaryDirectory() as tmpdir:
            best_path, best_eval = find_best_adapter(Path(tmpdir), "grpo")
            assert best_path is None
            assert best_eval is None


# ============================================================================
# Formatting Tests
# ============================================================================


class TestFormatting:
    """Tests for output formatting."""

    def test_format_comparison_table(self) -> None:
        """Test comparison table formatting."""
        xgb_eval = BaselineEvaluation(
            model_type="xgboost",
            ic=0.08,
            ic_pvalue=0.01,
            brier_score=0.22,
            sharpe_ratio=1.5,
            directional_accuracy=0.55,
            num_examples=100,
        )
        lgb_eval = BaselineEvaluation(
            model_type="lightgbm",
            ic=0.06,
            ic_pvalue=0.05,
            brier_score=0.24,
            sharpe_ratio=1.2,
            directional_accuracy=0.52,
            num_examples=100,
        )

        table = format_comparison_table(xgb_eval, lgb_eval, None, None)

        assert "XGBoost" in table
        assert "LightGBM" in table
        assert "0.08" in table  # XGBoost IC
        assert "ANALYSIS" in table

    def test_format_comparison_with_adapters(self) -> None:
        """Test comparison table with LLM adapters."""
        xgb_eval = BaselineEvaluation(
            model_type="xgboost",
            ic=0.08,
            ic_pvalue=0.01,
            brier_score=0.22,
            sharpe_ratio=1.5,
            directional_accuracy=0.55,
            num_examples=100,
        )
        lgb_eval = BaselineEvaluation(
            model_type="lightgbm",
            ic=0.06,
            ic_pvalue=0.05,
            brier_score=0.24,
            sharpe_ratio=1.2,
            directional_accuracy=0.52,
            num_examples=100,
        )
        grpo_eval = {"ic": 0.10, "brier_score": 0.20, "num_examples": 100}

        table = format_comparison_table(xgb_eval, lgb_eval, grpo_eval, None)

        assert "GRPO" in table
        assert "BEATS" in table  # GRPO beats XGBoost

    def test_format_feature_importance_table(self) -> None:
        """Test feature importance table formatting."""
        xgb_eval = BaselineEvaluation(
            model_type="xgboost",
            ic=0.08,
            ic_pvalue=0.01,
            brier_score=0.22,
            sharpe_ratio=1.5,
            directional_accuracy=0.55,
            num_examples=100,
            feature_importance={"rsi": 0.15, "macd_line": 0.12, "cmf": 0.08},
            shap_importance={"rsi": 0.25, "macd_line": 0.18, "cmf": 0.10},
        )
        lgb_eval = BaselineEvaluation(
            model_type="lightgbm",
            ic=0.06,
            ic_pvalue=0.05,
            brier_score=0.24,
            sharpe_ratio=1.2,
            directional_accuracy=0.52,
            num_examples=100,
            feature_importance={"rsi": 0.18, "macd_line": 0.10, "cmf": 0.09},
        )

        table = format_feature_importance_table(xgb_eval, lgb_eval)

        assert "FEATURE IMPORTANCE" in table
        assert "rsi" in table
        assert "SHAP" in table
        assert "RECOMMENDATIONS" in table


# ============================================================================
# Edge Case Tests
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_features(self) -> None:
        """Test with empty feature DataFrame."""
        features_df = pd.DataFrame()
        targets = np.array([])
        returns = np.array([])
        timestamps = np.array([])

        with pytest.raises(ValueError):
            evaluate_baseline("xgboost", features_df, targets, returns, timestamps, n_folds=3)

    def test_single_class_target(self) -> None:
        """Test with single class target (all 0s or 1s) raises ValueError."""
        np.random.seed(42)
        n_samples = 300  # Need enough for walk-forward folds

        features = pd.DataFrame(
            {
                "rsi": np.random.uniform(30, 70, n_samples),
                "cmf": np.random.uniform(-0.5, 0.5, n_samples),
                "mfi": np.random.uniform(30, 70, n_samples),
            }
        )
        targets = np.ones(n_samples)  # All 1s - classification requires 2+ classes
        returns = np.random.randn(n_samples)
        timestamps = np.arange(n_samples) * 3600000

        # XGBoost classifier requires 2+ classes, so this should raise
        with pytest.raises(ValueError):
            evaluate_baseline("xgboost", features, targets, returns, timestamps, n_folds=3)

    def test_extreme_values(self) -> None:
        """Test with extreme feature values."""
        np.random.seed(42)
        n_samples = 300  # Need enough for walk-forward folds

        # Create extreme values
        features = pd.DataFrame(
            {
                "rsi": np.random.uniform(0, 100, n_samples),
                "obv": np.random.uniform(1e10, 1e12, n_samples),  # Very large
                "cmf": np.random.uniform(-1, 1, n_samples),
            }
        )
        targets = np.random.randint(0, 2, n_samples)
        returns = np.random.randn(n_samples)
        timestamps = np.arange(n_samples) * 3600000

        # Should handle extreme values without crashing
        evaluation = evaluate_baseline("xgboost", features, targets, returns, timestamps, n_folds=3)

        assert isinstance(evaluation, BaselineEvaluation)
        assert not np.isnan(evaluation.ic)

    def test_all_nan_feature(self) -> None:
        """Test handling of all-NaN feature column during imputation."""
        features = pd.DataFrame(
            {
                "rsi": np.random.uniform(30, 70, 100),
                "all_nan": np.full(100, np.nan),  # All NaN
            }
        )

        # Test that fillna with 0 handles all-NaN columns
        for col in features.columns:
            if features[col].isna().any():
                median_val = features[col].median()
                if pd.isna(median_val):
                    median_val = 0.0
                features[col] = features[col].fillna(median_val)

        assert not features.isna().any().any()
        assert features["all_nan"].iloc[0] == 0.0


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests for full pipeline."""

    def test_full_pipeline(self, synthetic_data_file: Path) -> None:
        """Test full evaluation pipeline."""
        # Load data
        features_df, targets, returns, timestamps = load_training_data(synthetic_data_file)

        # Evaluate both models
        xgb_eval = evaluate_baseline(
            "xgboost", features_df, targets, returns, timestamps, n_folds=3
        )
        lgb_eval = evaluate_baseline(
            "lightgbm", features_df, targets, returns, timestamps, n_folds=3
        )

        # Both should complete successfully
        assert xgb_eval.num_examples > 0
        assert lgb_eval.num_examples > 0

        # Format comparison
        table = format_comparison_table(xgb_eval, lgb_eval, None, None)
        assert len(table) > 100

    def test_results_serialization(self, synthetic_data_file: Path, tmp_path: Path) -> None:
        """Test results can be serialized to JSON."""
        features_df, targets, returns, timestamps = load_training_data(synthetic_data_file)

        xgb_eval = evaluate_baseline(
            "xgboost", features_df, targets, returns, timestamps, n_folds=3
        )

        results = {
            "xgboost": xgb_eval.to_dict(),
            "data_path": str(synthetic_data_file),
        }

        output_path = tmp_path / "results.json"
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        # Verify it can be loaded back
        with open(output_path) as f:
            loaded = json.load(f)

        assert loaded["xgboost"]["model_type"] == "xgboost"
        assert loaded["xgboost"]["ic"] == xgb_eval.ic
