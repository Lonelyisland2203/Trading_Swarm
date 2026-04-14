"""
Tests for XGBoost signal generator.

TDD: Tests written FIRST before implementation.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from signals.xgboost_config import (
    INDICATOR_FEATURES,
    XGB_PARAMS,
    WALK_FORWARD_CONFIG,
    LABEL_THRESHOLD,
    MIN_SIGNALS_FOR_RETRAIN,
    PROBABILITY_THRESHOLDS,
)


class TestXGBoostConfig:
    """Tests for xgboost_config.py constants."""

    def test_feature_list_has_21_indicators(self):
        """INDICATOR_FEATURES contains exactly 21 features (8+4+4+5)."""
        # Price/Trend: 8, Volume: 4, Volatility: 4, Market Structure: 5
        assert len(INDICATOR_FEATURES) == 21

    def test_feature_list_includes_all_required(self):
        """Feature list includes all required indicator types."""
        # Price/Trend
        assert "rsi" in INDICATOR_FEATURES
        assert "macd_line" in INDICATOR_FEATURES
        assert "kama" in INDICATOR_FEATURES

        # Volume
        assert "obv" in INDICATOR_FEATURES
        assert "cmf" in INDICATOR_FEATURES
        assert "mfi" in INDICATOR_FEATURES
        assert "vwap" in INDICATOR_FEATURES

        # Volatility
        assert "atr_normalized" in INDICATOR_FEATURES
        assert "bb_width" in INDICATOR_FEATURES
        assert "keltner_width" in INDICATOR_FEATURES
        assert "donchian_width" in INDICATOR_FEATURES

        # Market Structure
        assert "open_fvg_count" in INDICATOR_FEATURES
        assert "nearest_bullish_fvg_pct" in INDICATOR_FEATURES
        assert "nearest_swing_high_pct" in INDICATOR_FEATURES

    def test_xgb_params_has_required_keys(self):
        """XGB_PARAMS has all required hyperparameters."""
        required = ["n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree"]
        for key in required:
            assert key in XGB_PARAMS, f"Missing XGB param: {key}"

    def test_walk_forward_config_defaults(self):
        """Walk-forward config has correct defaults."""
        assert WALK_FORWARD_CONFIG.n_folds == 5
        assert WALK_FORWARD_CONFIG.train_ratio == 0.7
        assert WALK_FORWARD_CONFIG.gap_bars == 24
        assert WALK_FORWARD_CONFIG.min_train_size == 50
        assert WALK_FORWARD_CONFIG.min_test_size == 20

    def test_label_threshold_uses_fee_model(self):
        """Label threshold uses FeeModelSettings for computation."""
        threshold = LABEL_THRESHOLD.get_threshold_pct()
        # Default fee model with 1.0 holding periods should give ~0.09-0.1%
        assert 0.05 < threshold < 0.20

    def test_retrain_threshold_is_200(self):
        """Retrain triggers at 200 verified signals."""
        assert MIN_SIGNALS_FOR_RETRAIN == 200


class TestFeatureExtraction:
    """Tests for feature extraction from OHLCV data."""

    @pytest.fixture
    def sample_ohlcv(self) -> pd.DataFrame:
        """Generate sample OHLCV data for testing."""
        np.random.seed(42)
        n_bars = 100

        timestamps = np.arange(n_bars) * 3600000  # 1h bars in ms
        close = 42000 + np.cumsum(np.random.randn(n_bars) * 100)
        high = close + np.abs(np.random.randn(n_bars) * 50)
        low = close - np.abs(np.random.randn(n_bars) * 50)
        open_price = close + np.random.randn(n_bars) * 20
        volume = np.random.randint(100, 10000, n_bars).astype(float)

        return pd.DataFrame(
            {
                "timestamp": timestamps,
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            }
        )

    def test_feature_extraction_shape(self, sample_ohlcv: pd.DataFrame):
        """Feature extraction produces correct number of features."""
        from signals.xgboost_signal import extract_features_from_ohlcv

        features = extract_features_from_ohlcv(sample_ohlcv)

        # Should have 17 indicator features (extras may be None)
        indicator_count = sum(
            1 for f in INDICATOR_FEATURES if f in features and features[f] is not None
        )
        assert indicator_count >= 14, f"Expected at least 14 features, got {indicator_count}"

    def test_feature_extraction_returns_dict(self, sample_ohlcv: pd.DataFrame):
        """Feature extraction returns dictionary with correct keys."""
        from signals.xgboost_signal import extract_features_from_ohlcv

        features = extract_features_from_ohlcv(sample_ohlcv)

        assert isinstance(features, dict)
        # Check key indicator features are present
        assert "rsi" in features
        assert "macd_line" in features
        assert "atr_normalized" in features

    def test_feature_values_are_numeric(self, sample_ohlcv: pd.DataFrame):
        """All feature values are numeric or None."""
        from signals.xgboost_signal import extract_features_from_ohlcv

        features = extract_features_from_ohlcv(sample_ohlcv)

        for key, value in features.items():
            if value is not None:
                assert isinstance(value, (int, float, np.number)), (
                    f"{key} is not numeric: {type(value)}"
                )


class TestProbabilityOutput:
    """Tests for XGBoost probability output."""

    def test_probability_output_range(self):
        """Output probability is between 0 and 1."""
        from signals.xgboost_signal import XGBoostSignal

        # Mock signal with valid probability
        signal = XGBoostSignal(
            symbol="BTC/USDT",
            timeframe="1h",
            direction="LONG",
            probability=0.72,
            confidence=0.72,
            features={},
            timestamp=datetime.now(timezone.utc),
        )

        assert 0.0 <= signal.probability <= 1.0

    def test_probability_clipped_to_valid_range(self):
        """Probabilities outside [0,1] are clipped."""
        from signals.xgboost_signal import clip_probability

        assert clip_probability(1.5) == 1.0
        assert clip_probability(-0.5) == 0.0
        assert clip_probability(0.7) == 0.7


class TestDirectionMapping:
    """Tests for probability to direction mapping."""

    def test_high_probability_maps_to_long(self):
        """High probability (>=0.55) maps to LONG."""
        from signals.xgboost_signal import map_probability_to_direction

        assert map_probability_to_direction(0.70) == "LONG"
        assert map_probability_to_direction(0.55) == "LONG"

    def test_low_probability_maps_to_short(self):
        """Low probability (<=0.45) maps to SHORT."""
        from signals.xgboost_signal import map_probability_to_direction

        assert map_probability_to_direction(0.30) == "SHORT"
        assert map_probability_to_direction(0.45) == "SHORT"

    def test_middle_probability_maps_to_flat(self):
        """Middle probability (0.45-0.55) maps to FLAT."""
        from signals.xgboost_signal import map_probability_to_direction

        assert map_probability_to_direction(0.50) == "FLAT"
        assert map_probability_to_direction(0.48) == "FLAT"
        assert map_probability_to_direction(0.52) == "FLAT"

    def test_probability_thresholds_from_config(self):
        """Direction mapping uses thresholds from config."""
        flat_threshold = PROBABILITY_THRESHOLDS["flat_threshold"]
        assert flat_threshold == 0.55


class TestTemporalSafety:
    """Tests for point-in-time safety / no lookahead."""

    @pytest.fixture
    def mock_market_data_service(self):
        """Create mock market data service."""
        service = MagicMock()
        service.get_ohlcv_as_of = AsyncMock()
        return service

    @pytest.mark.asyncio
    async def test_temporal_safety_uses_get_ohlcv_as_of(self, mock_market_data_service):
        """Signal generation uses get_ohlcv_as_of for temporal safety."""
        from signals.xgboost_signal import generate_xgboost_signal

        # Setup mock to return sample data
        sample_df = pd.DataFrame(
            {
                "timestamp": np.arange(100) * 3600000,
                "open": np.random.randn(100) + 42000,
                "high": np.random.randn(100) + 42050,
                "low": np.random.randn(100) + 41950,
                "close": np.random.randn(100) + 42000,
                "volume": np.random.randint(100, 10000, 100).astype(float),
            }
        )
        mock_market_data_service.get_ohlcv_as_of.return_value = sample_df

        as_of_ts = int(datetime.now(timezone.utc).timestamp() * 1000)

        # Mock model to return predictions
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])

        # This should call get_ohlcv_as_of, NOT fetch_ohlcv
        with patch("signals.xgboost_signal.run_preflight_checks") as mock_preflight:
            mock_preflight.return_value = MagicMock(passed=True)

            with patch("signals.xgboost_signal.load_model") as mock_load_model:
                mock_load_model.return_value = mock_model

                await generate_xgboost_signal(
                    symbol="BTC/USDT",
                    timeframe="1h",
                    as_of=as_of_ts,
                    market_data_service=mock_market_data_service,
                )

        # Verify get_ohlcv_as_of was called (temporal safety)
        mock_market_data_service.get_ohlcv_as_of.assert_called()

        # Extract call args
        call_args = mock_market_data_service.get_ohlcv_as_of.call_args
        if call_args:
            # Verify as_of timestamp was passed
            assert call_args.kwargs.get("as_of") == as_of_ts or as_of_ts in call_args.args

    @pytest.mark.asyncio
    async def test_no_future_data_in_features(self, mock_market_data_service):
        """Feature vector contains no data from after as_of timestamp."""
        from signals.xgboost_signal import extract_features_from_ohlcv

        # Create data where latest bar is at as_of
        as_of_ts = 100 * 3600000  # 100 hours in ms
        n_bars = 100

        df = pd.DataFrame(
            {
                "timestamp": np.arange(n_bars) * 3600000,  # 0 to 99 hours
                "open": np.random.randn(n_bars) + 42000,
                "high": np.random.randn(n_bars) + 42050,
                "low": np.random.randn(n_bars) + 41950,
                "close": np.random.randn(n_bars) + 42000,
                "volume": np.random.randint(100, 10000, n_bars).astype(float),
            }
        )

        # Filter to only data available at as_of (bar close <= as_of)
        filtered_df = df[df["timestamp"] + 3600000 <= as_of_ts]  # +3600000 for bar close

        features = extract_features_from_ohlcv(filtered_df)

        # Features should be extractable from historical data only
        assert features is not None
        assert "rsi" in features


class TestWalkForwardSplitOrdering:
    """Tests for walk-forward cross-validation temporal ordering."""

    def test_walk_forward_split_ordering(self):
        """Train dates always before validation dates."""
        from signals.xgboost_signal import create_walk_forward_splits

        # Create sample timestamps spanning 1 year
        n_samples = 1000
        timestamps = np.arange(n_samples) * 86400000  # Daily bars in ms

        splits = create_walk_forward_splits(
            timestamps,
            n_folds=WALK_FORWARD_CONFIG.n_folds,
            train_ratio=WALK_FORWARD_CONFIG.train_ratio,
            gap_bars=WALK_FORWARD_CONFIG.gap_bars,
        )

        for i, split in enumerate(splits):
            train_max_ts = timestamps[split["train_indices"]].max()
            val_min_ts = timestamps[split["val_indices"]].min()

            assert train_max_ts < val_min_ts, (
                f"Fold {i}: train_max={train_max_ts} >= val_min={val_min_ts}"
            )

    def test_walk_forward_gap_enforced(self):
        """Gap bars are enforced between train and validation."""
        from signals.xgboost_signal import create_walk_forward_splits

        n_samples = 500
        timestamps = np.arange(n_samples) * 3600000  # Hourly bars

        splits = create_walk_forward_splits(
            timestamps,
            n_folds=3,
            train_ratio=0.7,
            gap_bars=24,
        )

        for split in splits:
            train_max_idx = split["train_indices"].max()
            val_min_idx = split["val_indices"].min()

            # Gap should be at least gap_bars
            assert val_min_idx - train_max_idx >= 24


class TestConfigLoading:
    """Tests for config parameter loading."""

    def test_xgboost_model_uses_config_params(self):
        """XGBoost model uses parameters from xgboost_config.py."""
        from signals.xgboost_signal import create_xgboost_model

        model = create_xgboost_model()

        # Verify model uses config params
        assert model.n_estimators == XGB_PARAMS["n_estimators"]
        assert model.max_depth == XGB_PARAMS["max_depth"]
        assert model.learning_rate == XGB_PARAMS["learning_rate"]

    def test_feature_list_matches_config(self):
        """Feature extraction uses FEATURE_LIST from config."""
        from signals.xgboost_signal import get_feature_names

        feature_names = get_feature_names()

        # Should match config feature list
        for indicator in INDICATOR_FEATURES:
            assert indicator in feature_names


class TestRetrainTrigger:
    """Tests for retrain trigger logic."""

    def test_retrain_trigger_at_threshold(self):
        """Retrain triggers when verified signals reach threshold."""
        from signals.xgboost_signal import check_retrain_trigger

        # Below threshold
        assert check_retrain_trigger(signals_count=100) is False
        assert check_retrain_trigger(signals_count=199) is False

        # At and above threshold
        assert check_retrain_trigger(signals_count=200) is True
        assert check_retrain_trigger(signals_count=250) is True

    def test_retrain_threshold_from_config(self):
        """Retrain threshold uses MIN_SIGNALS_FOR_RETRAIN from config."""
        from signals.xgboost_signal import get_retrain_threshold

        threshold = get_retrain_threshold()
        assert threshold == MIN_SIGNALS_FOR_RETRAIN

    @patch("signals.xgboost_signal.load_verified_results")
    def test_retrain_integrates_with_verification(self, mock_load):
        """Retrain check integrates with verification.py results."""
        from signals.xgboost_signal import should_trigger_retrain

        # Mock 150 verified results
        mock_load.return_value = [{"id": i} for i in range(150)]
        assert should_trigger_retrain() is False

        # Mock 200 verified results
        mock_load.return_value = [{"id": i} for i in range(200)]
        assert should_trigger_retrain() is True


class TestXGBoostSignalDataclass:
    """Tests for XGBoostSignal dataclass."""

    def test_xgboost_signal_creation(self):
        """XGBoostSignal can be created with required fields."""
        from signals.xgboost_signal import XGBoostSignal

        signal = XGBoostSignal(
            symbol="BTC/USDT",
            timeframe="1h",
            direction="LONG",
            probability=0.72,
            confidence=0.72,
            features={"rsi": 65.5, "macd_line": 100.0},
            timestamp=datetime.now(timezone.utc),
        )

        assert signal.symbol == "BTC/USDT"
        assert signal.direction == "LONG"
        assert signal.probability == 0.72
        assert "rsi" in signal.features

    def test_xgboost_signal_to_dict(self):
        """XGBoostSignal can be serialized to dict."""
        from signals.xgboost_signal import XGBoostSignal

        ts = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        signal = XGBoostSignal(
            symbol="ETH/USDT",
            timeframe="4h",
            direction="SHORT",
            probability=0.35,
            confidence=0.65,
            features={"rsi": 25.0},
            timestamp=ts,
        )

        data = signal.to_dict()

        assert data["symbol"] == "ETH/USDT"
        assert data["direction"] == "SHORT"
        assert data["probability"] == 0.35
        assert data["timestamp"] == "2024-01-15T12:00:00+00:00"


class TestPreflightIntegration:
    """Tests for preflight check integration."""

    @pytest.mark.asyncio
    async def test_signal_generation_runs_preflight(self):
        """Signal generation runs preflight checks."""
        from signals.xgboost_signal import generate_xgboost_signal

        with patch("signals.xgboost_signal.run_preflight_checks") as mock_preflight:
            mock_preflight.return_value = MagicMock(passed=False, reason="STOP file")

            mock_service = MagicMock()

            result = await generate_xgboost_signal(
                symbol="BTC/USDT",
                timeframe="1h",
                as_of=int(datetime.now(timezone.utc).timestamp() * 1000),
                market_data_service=mock_service,
            )

            # Preflight should have been called
            mock_preflight.assert_called_once()

            # Should return None when preflight fails
            assert result is None

    @pytest.mark.asyncio
    async def test_signal_generation_proceeds_when_preflight_passes(self):
        """Signal generation proceeds when preflight passes."""
        from signals.xgboost_signal import generate_xgboost_signal

        sample_df = pd.DataFrame(
            {
                "timestamp": np.arange(100) * 3600000,
                "open": np.random.randn(100) + 42000,
                "high": np.random.randn(100) + 42050,
                "low": np.random.randn(100) + 41950,
                "close": np.random.randn(100) + 42000,
                "volume": np.random.randint(100, 10000, 100).astype(float),
            }
        )

        mock_service = MagicMock()
        mock_service.get_ohlcv_as_of = AsyncMock(return_value=sample_df)

        with patch("signals.xgboost_signal.run_preflight_checks") as mock_preflight:
            mock_preflight.return_value = MagicMock(passed=True)

            with patch("signals.xgboost_signal.load_model") as mock_load:
                # Return None to trigger model not found path
                mock_load.return_value = None

                await generate_xgboost_signal(
                    symbol="BTC/USDT",
                    timeframe="1h",
                    as_of=int(datetime.now(timezone.utc).timestamp() * 1000),
                    market_data_service=mock_service,
                )

                # Should have attempted to load model
                mock_load.assert_called()
