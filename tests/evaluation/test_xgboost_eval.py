"""
Tests for the XGBoost evaluation script.

These tests verify temporal safety, metric computation, and results format.
"""

import json
from unittest.mock import patch

import numpy as np
import pytest


class TestExtractFeaturesFromSnapshot:
    """Tests for feature extraction from market snapshots."""

    def test_extracts_rsi(self):
        """Should extract RSI value."""
        from evaluation.xgboost_eval import extract_features_from_snapshot

        snapshot = "RSI(14): 45.5 | Some other data"
        features = extract_features_from_snapshot(snapshot)

        assert features["rsi"] == pytest.approx(45.5)

    def test_extracts_macd_line(self):
        """Should extract MACD Line value."""
        from evaluation.xgboost_eval import extract_features_from_snapshot

        snapshot = "MACD Line: 0.0025 | Signal: 0.0020"
        features = extract_features_from_snapshot(snapshot)

        assert features["macd_line"] == pytest.approx(0.0025)

    def test_handles_na_values(self):
        """Should return None for N/A values."""
        from evaluation.xgboost_eval import extract_features_from_snapshot

        snapshot = "RSI(14): N/A | CMF(20): N/A"
        features = extract_features_from_snapshot(snapshot)

        assert features["rsi"] is None
        assert features["cmf"] is None

    def test_handles_missing_indicators(self):
        """Should return None for missing indicators."""
        from evaluation.xgboost_eval import extract_features_from_snapshot

        snapshot = "Some unrelated text without indicators"
        features = extract_features_from_snapshot(snapshot)

        assert features["rsi"] is None
        assert features["macd_line"] is None


class TestWalkForwardFolds:
    """Tests for walk-forward cross-validation fold creation."""

    def test_creates_folds(self):
        """Should create the configured number of folds."""
        from evaluation.xgboost_eval import create_walk_forward_folds

        timestamps = np.arange(0, 10000, 10)  # 1000 timestamps

        folds = create_walk_forward_folds(timestamps, n_folds=5)

        assert len(folds) > 0
        assert len(folds) <= 5

    def test_train_before_test(self):
        """Train data must always be temporally before test data."""
        from evaluation.xgboost_eval import create_walk_forward_folds

        timestamps = np.arange(0, 10000, 10)

        folds = create_walk_forward_folds(timestamps)

        for fold in folds:
            assert fold["train_end_ts"] < fold["test_start_ts"], "Train must end before test starts"

    def test_no_overlap(self):
        """Train and test indices should not overlap."""
        from evaluation.xgboost_eval import create_walk_forward_folds

        timestamps = np.arange(0, 10000, 10)

        folds = create_walk_forward_folds(timestamps)

        for fold in folds:
            train_set = set(fold["train_indices"])
            test_set = set(fold["test_indices"])

            assert train_set.isdisjoint(test_set), "Train and test should not overlap"

    def test_uses_config_defaults(self):
        """Should use WALK_FORWARD_CONFIG defaults when not specified."""
        from evaluation.xgboost_eval import create_walk_forward_folds
        from signals.xgboost_config import WALK_FORWARD_CONFIG

        timestamps = np.arange(0, 10000, 10)

        # Call without specifying parameters
        folds = create_walk_forward_folds(timestamps)

        # Should have at most n_folds
        assert len(folds) <= WALK_FORWARD_CONFIG.n_folds


class TestConfigHash:
    """Tests for config hash generation."""

    def test_hash_is_deterministic(self):
        """Same config should produce same hash."""
        from evaluation.xgboost_eval import get_config_hash

        hash1 = get_config_hash()
        hash2 = get_config_hash()

        assert hash1 == hash2

    def test_hash_is_short(self):
        """Hash should be a short string (8 chars)."""
        from evaluation.xgboost_eval import get_config_hash

        hash_val = get_config_hash()

        assert isinstance(hash_val, str)
        assert len(hash_val) == 8


class TestEvalResult:
    """Tests for EvalResult dataclass."""

    def test_to_dict(self):
        """to_dict should produce JSON-serializable output."""
        from evaluation.xgboost_eval import EvalResult

        result = EvalResult(
            sharpe_net=1.5,
            ic=0.08,
            ic_pvalue=0.01,
            brier=0.22,
            directional_accuracy=0.55,
            false_bullish_rate=0.15,
            false_bearish_rate=0.20,
            num_examples=1000,
            config_hash="abc12345",
            shap_top_5={"rsi": 0.5, "macd_line": 0.3},
        )

        d = result.to_dict()

        # Should be JSON serializable
        json_str = json.dumps(d)
        assert json_str is not None

        # Should contain all fields
        assert d["sharpe_net"] == 1.5
        assert d["ic"] == 0.08
        assert d["shap_top_5"] == {"rsi": 0.5, "macd_line": 0.3}


class TestResultsTsvAppend:
    """Tests for appending results to TSV."""

    def test_appends_with_correct_format(self, tmp_path):
        """Should append results in correct TSV format."""
        from evaluation.xgboost_eval import EvalResult, append_to_results_tsv

        import evaluation.xgboost_eval as eval_module

        results_path = tmp_path / "results.tsv"

        # Patch the RESULTS_TSV_PATH
        with patch.object(eval_module, "RESULTS_TSV_PATH", results_path):
            result = EvalResult(
                sharpe_net=1.5,
                ic=0.08,
                ic_pvalue=0.01,
                brier=0.22,
                directional_accuracy=0.55,
                false_bullish_rate=0.15,
                false_bearish_rate=0.20,
                num_examples=1000,
                config_hash="abc12345",
            )

            append_to_results_tsv(result, change_description="test_change", kept=True)

        # Verify file was created
        assert results_path.exists()

        # Verify content
        content = results_path.read_text()
        lines = content.strip().split("\n")

        assert len(lines) == 2  # Header + 1 row
        assert "experiment_id" in lines[0]
        assert "sharpe_net" in lines[0]
        assert "test_change" in lines[1]

    def test_increments_experiment_id(self, tmp_path):
        """Should increment experiment_id for each append."""
        from evaluation.xgboost_eval import EvalResult, append_to_results_tsv

        import evaluation.xgboost_eval as eval_module

        results_path = tmp_path / "results.tsv"

        with patch.object(eval_module, "RESULTS_TSV_PATH", results_path):
            result = EvalResult(
                sharpe_net=1.0,
                ic=0.05,
                ic_pvalue=0.05,
                brier=0.25,
                directional_accuracy=0.52,
                false_bullish_rate=0.18,
                false_bearish_rate=0.22,
                num_examples=500,
                config_hash="xyz789",
            )

            append_to_results_tsv(result, "first", kept=True)
            append_to_results_tsv(result, "second", kept=False)

        # Verify content
        content = results_path.read_text()
        lines = content.strip().split("\n")

        assert len(lines) == 3  # Header + 2 rows

        # Parse and check experiment IDs
        import csv

        with open(results_path) as f:
            reader = csv.DictReader(f, delimiter="\t")
            rows = list(reader)

        assert rows[0]["experiment_id"] == "0"
        assert rows[1]["experiment_id"] == "1"


class TestTemporalSafety:
    """Tests specifically for temporal safety guarantees."""

    def test_load_training_data_preserves_timestamp_order(self, tmp_path):
        """Data loading should preserve timestamp information for ordering."""
        from evaluation.xgboost_eval import load_training_data

        # Create test data file
        data_file = tmp_path / "test_data.jsonl"

        test_examples = [
            {
                "market_snapshot": "RSI(14): 45.0",
                "actual_direction": "LONG",
                "gross_return_pct": 0.5,
                "timestamp_ms": 1000,
            },
            {
                "market_snapshot": "RSI(14): 55.0",
                "actual_direction": "SHORT",
                "gross_return_pct": -0.3,
                "timestamp_ms": 2000,
            },
            {
                "market_snapshot": "RSI(14): 50.0",
                "actual_direction": "FLAT",
                "gross_return_pct": 0.1,
                "timestamp_ms": 1500,
            },
        ]

        with open(data_file, "w") as f:
            for ex in test_examples:
                f.write(json.dumps(ex) + "\n")

        features_df, targets, returns, timestamps, directions = load_training_data(data_file)

        # Timestamps should be preserved
        assert len(timestamps) == 3
        assert set(timestamps) == {1000, 1500, 2000}

    def test_folds_respect_gap_bars(self):
        """Walk-forward folds should have gap between train and test."""
        from evaluation.xgboost_eval import create_walk_forward_folds
        from signals.xgboost_config import WALK_FORWARD_CONFIG

        timestamps = np.arange(0, 10000, 10)

        folds = create_walk_forward_folds(timestamps, gap_bars=WALK_FORWARD_CONFIG.gap_bars)

        for fold in folds:
            # There should be a gap between train end and test start
            train_indices = fold["train_indices"]
            test_indices = fold["test_indices"]

            train_end_idx = max(train_indices)
            test_start_idx = min(test_indices)

            gap = test_start_idx - train_end_idx
            assert gap >= WALK_FORWARD_CONFIG.gap_bars, (
                f"Gap {gap} is less than required {WALK_FORWARD_CONFIG.gap_bars}"
            )


class TestMetricComputation:
    """Tests for metric computation."""

    def test_false_bullish_rate_computation(self):
        """False bullish rate should be computed correctly."""
        import numpy as np

        # Simulate predictions and actuals
        predictions = np.array([1, 1, 1, 0, 0])  # 3 predicted LONG
        directions = np.array(["LONG", "SHORT", "FLAT", "LONG", "SHORT"])

        # False bullish = predicted LONG but actual was not LONG
        false_bullish_mask = (predictions == 1) & (directions != "LONG")
        false_bullish_rate = np.sum(false_bullish_mask) / np.sum(predictions == 1)

        # 2 out of 3 LONG predictions were wrong
        assert false_bullish_rate == pytest.approx(2 / 3)

    def test_false_bearish_rate_computation(self):
        """False bearish rate should be computed correctly."""
        import numpy as np

        # Simulate predictions and actuals
        predictions = np.array([1, 0, 0, 0, 0])  # 4 predicted SHORT/FLAT
        directions = np.array(["LONG", "LONG", "SHORT", "FLAT", "LONG"])

        # False bearish = predicted SHORT (0) but actual was LONG
        false_bearish_mask = (predictions == 0) & (directions == "LONG")
        false_bearish_rate = np.sum(false_bearish_mask) / np.sum(predictions == 0)

        # 2 out of 4 SHORT predictions were wrong (actual was LONG)
        assert false_bearish_rate == pytest.approx(2 / 4)
