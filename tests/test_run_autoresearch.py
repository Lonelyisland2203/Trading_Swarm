"""
Tests for the XGBoost autoresearch hyperparameter search loop.

Tests cover:
- Temporal safety in evaluation
- Config isolation (only xgboost_config.py modified)
- Results TSV format
- Git revert on regression
- STOP file handling
- Improvement threshold enforcement
"""

import csv
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from run_autoresearch import (
    IMPROVEMENT_THRESHOLDS,
    PARAM_ORDER,
    PARAM_SEARCH_SPACE,
    RESULTS_TSV_PATH,
    XGBOOST_CONFIG_PATH,
    AutoresearchLoop,
    EvalResult,
    ExperimentResult,
    ParameterState,
    append_result,
    check_stop_file,
    choose_next_parameter,
    format_results_table,
    get_best_metric,
    modify_config_parameter,
    read_results_tsv,
)


# =============================================================================
# Test: Temporal Safety in Evaluation
# =============================================================================


class TestEvalTemporalSafety:
    """Verify xgboost_eval.py uses get_ohlcv_as_of correctly."""

    def test_eval_uses_temporal_safe_data(self):
        """The eval script should use point-in-time safe data loading."""
        # Read the eval script and verify it uses temporal safety patterns
        eval_path = Path("evaluation/xgboost_eval.py")
        assert eval_path.exists(), "xgboost_eval.py must exist"

        content = eval_path.read_text()

        # Should use walk-forward CV (temporal ordering)
        assert "walk_forward" in content.lower() or "walkforward" in content.lower()

        # Should use timestamp ordering
        assert "timestamp" in content.lower()

        # Should not have any obvious lookahead patterns
        assert "future" not in content.lower() or "future" in "future_" not in content

    def test_walk_forward_ensures_train_before_test(self):
        """Walk-forward folds must ensure train data is always before test data."""
        from evaluation.xgboost_eval import create_walk_forward_folds

        import numpy as np

        # Create timestamps in order
        timestamps = np.arange(0, 10000, 10)

        folds = create_walk_forward_folds(timestamps)

        assert len(folds) > 0, "Should create at least one fold"

        for fold in folds:
            train_end_ts = fold["train_end_ts"]
            test_start_ts = fold["test_start_ts"]

            # Train must end before test starts
            assert train_end_ts < test_start_ts, (
                f"Train end ({train_end_ts}) must be before test start ({test_start_ts})"
            )


# =============================================================================
# Test: Config Isolation
# =============================================================================


class TestConfigIsolation:
    """Verify only xgboost_config.py changes between experiments."""

    def test_only_xgboost_config_in_path(self):
        """XGBOOST_CONFIG_PATH should point to signals/xgboost_config.py."""
        assert XGBOOST_CONFIG_PATH == Path("signals/xgboost_config.py")

    def test_modify_only_touches_config_file(self, tmp_path, monkeypatch):
        """modify_config_parameter should only modify xgboost_config.py."""
        import run_autoresearch

        # Create a temp config file
        config_content = '''"""XGBoost config."""
XGB_PARAMS: dict = {
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
}
'''
        config_path = tmp_path / "xgboost_config.py"
        config_path.write_text(config_content)

        # Create another file that should NOT be modified
        other_file = tmp_path / "fee_model.py"
        other_file.write_text("FEE = 0.001\n")
        original_fee_content = other_file.read_text()

        monkeypatch.setattr(run_autoresearch, "XGBOOST_CONFIG_PATH", config_path)

        # Modify a parameter
        modify_config_parameter("max_depth", 8)

        # Verify config was modified
        new_config = config_path.read_text()
        assert '"max_depth": 8' in new_config

        # Verify other file was NOT modified
        assert other_file.read_text() == original_fee_content

    def test_all_modifiable_params_in_search_space(self):
        """All parameters in PARAM_ORDER should be in PARAM_SEARCH_SPACE."""
        for param in PARAM_ORDER:
            assert param in PARAM_SEARCH_SPACE, f"Parameter {param} missing from search space"


# =============================================================================
# Test: Results TSV Format
# =============================================================================


class TestResultsTsvFormat:
    """Verify results.tsv columns match expected schema."""

    EXPECTED_COLUMNS = [
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

    def test_results_tsv_has_correct_header(self):
        """results.tsv should have the expected column headers."""
        if RESULTS_TSV_PATH.exists():
            with open(RESULTS_TSV_PATH) as f:
                reader = csv.DictReader(f, delimiter="\t")
                for expected_col in self.EXPECTED_COLUMNS:
                    assert expected_col in reader.fieldnames, f"Missing column: {expected_col}"

    def test_append_result_creates_correct_format(self, tmp_path, monkeypatch):
        """append_result should create correctly formatted TSV row."""
        import run_autoresearch

        results_path = tmp_path / "results.tsv"
        monkeypatch.setattr(run_autoresearch, "RESULTS_TSV_PATH", results_path)

        result = ExperimentResult(
            timestamp="2026-04-14 10:00:00",
            experiment_id=1,
            param_changed="max_depth",
            old_value=6,
            new_value=7,
            sharpe_net=0.95,
            ic=0.08,
            brier=0.22,
            accuracy=0.55,
            false_bullish_rate=0.15,
            kept=True,
        )

        append_result(result)

        # Read and verify
        with open(results_path) as f:
            reader = csv.DictReader(f, delimiter="\t")
            rows = list(reader)

        assert len(rows) == 1
        row = rows[0]

        assert row["experiment_id"] == "1"
        assert "max_depth" in row["change_description"]
        assert row["kept_or_reverted"] == "kept"
        assert float(row["sharpe_net"]) == pytest.approx(0.95, rel=1e-4)
        assert float(row["ic"]) == pytest.approx(0.08, rel=1e-4)

    def test_read_results_parses_correctly(self, tmp_path, monkeypatch):
        """read_results_tsv should parse all columns correctly."""
        import run_autoresearch

        results_path = tmp_path / "results.tsv"
        results_path.write_text(
            "experiment_id\ttimestamp\tchange_description\tsharpe_net\tic\tbrier\taccuracy\tfalse_bullish_rate\tkept_or_reverted\n"
            "1\t2026-04-14 10:00:00\tmax_depth: 6→7\t0.950000\t0.080000\t0.220000\t0.5500\t0.1500\tkept\n"
            "2\t2026-04-14 11:00:00\tlearning_rate: 0.1→0.05\t0.850000\t0.060000\t0.250000\t0.5200\t0.1800\treverted\n"
        )

        monkeypatch.setattr(run_autoresearch, "RESULTS_TSV_PATH", results_path)

        results = read_results_tsv()

        assert len(results) == 2
        assert results[0].experiment_id == 1
        assert results[0].kept is True
        assert results[0].sharpe_net == pytest.approx(0.95, rel=1e-4)
        assert results[1].kept is False


# =============================================================================
# Test: Git Revert on Regression
# =============================================================================


class TestRevertOnRegression:
    """Verify git revert when Sharpe_net drops."""

    @patch("run_autoresearch.git_revert")
    @patch("run_autoresearch.git_commit")
    @patch("run_autoresearch.run_evaluation")
    @patch("run_autoresearch.modify_config_parameter")
    @patch("run_autoresearch.read_current_config_values")
    @patch("run_autoresearch.read_results_tsv")
    @patch("run_autoresearch.append_result")
    def test_reverts_when_sharpe_drops(
        self,
        mock_append,
        mock_read_results,
        mock_read_config,
        mock_modify,
        mock_eval,
        mock_commit,
        mock_revert,
    ):
        """Should revert when sharpe_net does not improve."""
        # Previous best sharpe_net = 1.0
        mock_read_results.return_value = [
            ExperimentResult(
                "",
                1,
                "max_depth",
                5,
                6,
                sharpe_net=1.0,
                ic=0.08,
                brier=0.22,
                accuracy=0.55,
                false_bullish_rate=0.15,
                kept=True,
            )
        ]

        mock_read_config.return_value = {
            "max_depth": 6,
            "learning_rate": 0.1,
            "n_estimators": 100,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "false_bullish_penalty": 1.5,
            "false_bearish_penalty": 0.8,
            "n_folds": 5,
            "gap_bars": 24,
        }
        mock_modify.return_value = "6"

        # New experiment has worse sharpe_net
        mock_eval.return_value = EvalResult(
            sharpe_net=0.85,  # Worse than 1.0
            ic=0.07,
            ic_pvalue=0.01,
            brier=0.23,
            directional_accuracy=0.54,
            false_bullish_rate=0.16,
            false_bearish_rate=0.20,
            num_examples=1000,
            config_hash="abc123",
        )

        loop = AutoresearchLoop(dry_run=False)
        result = loop.run_single_experiment()

        assert result is not None
        assert result.kept is False
        mock_revert.assert_called_once()

    @patch("run_autoresearch.git_revert")
    @patch("run_autoresearch.git_commit")
    @patch("run_autoresearch.run_evaluation")
    @patch("run_autoresearch.modify_config_parameter")
    @patch("run_autoresearch.read_current_config_values")
    @patch("run_autoresearch.read_results_tsv")
    @patch("run_autoresearch.append_result")
    def test_keeps_when_sharpe_improves(
        self,
        mock_append,
        mock_read_results,
        mock_read_config,
        mock_modify,
        mock_eval,
        mock_commit,
        mock_revert,
    ):
        """Should keep when sharpe_net improves sufficiently."""
        # Previous best sharpe_net = 1.0
        mock_read_results.return_value = [
            ExperimentResult(
                "",
                1,
                "max_depth",
                5,
                6,
                sharpe_net=1.0,
                ic=0.08,
                brier=0.22,
                accuracy=0.55,
                false_bullish_rate=0.15,
                kept=True,
            )
        ]

        mock_read_config.return_value = {
            "max_depth": 6,
            "learning_rate": 0.1,
            "n_estimators": 100,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "false_bullish_penalty": 1.5,
            "false_bearish_penalty": 0.8,
            "n_folds": 5,
            "gap_bars": 24,
        }
        mock_modify.return_value = "6"

        # New experiment has better sharpe_net (improvement > 0.02)
        mock_eval.return_value = EvalResult(
            sharpe_net=1.05,  # Better than 1.0 + threshold
            ic=0.09,
            ic_pvalue=0.005,
            brier=0.20,
            directional_accuracy=0.57,
            false_bullish_rate=0.12,
            false_bearish_rate=0.18,
            num_examples=1000,
            config_hash="abc123",
        )

        loop = AutoresearchLoop(dry_run=False)
        result = loop.run_single_experiment()

        assert result is not None
        assert result.kept is True
        mock_commit.assert_called_once()
        mock_revert.assert_not_called()


# =============================================================================
# Test: STOP File
# =============================================================================


class TestStopFile:
    """Verify loop halts when STOP file exists."""

    def test_check_stop_file_returns_true_when_exists(self, tmp_path, monkeypatch):
        """check_stop_file should return True when STOP file exists."""
        import run_autoresearch

        stop_path = tmp_path / "STOP"
        stop_path.write_text("")

        monkeypatch.setattr(run_autoresearch, "STOP_FILE_PATH", stop_path)

        assert check_stop_file() is True

    def test_check_stop_file_returns_false_when_not_exists(self, tmp_path, monkeypatch):
        """check_stop_file should return False when STOP file does not exist."""
        import run_autoresearch

        stop_path = tmp_path / "STOP"
        # Do NOT create the file

        monkeypatch.setattr(run_autoresearch, "STOP_FILE_PATH", stop_path)

        assert check_stop_file() is False

    def test_loop_should_stop_when_stop_file_exists(self, tmp_path, monkeypatch):
        """AutoresearchLoop.should_stop should return True when STOP file exists."""
        import run_autoresearch

        stop_path = tmp_path / "STOP"
        stop_path.write_text("")

        monkeypatch.setattr(run_autoresearch, "STOP_FILE_PATH", stop_path)

        loop = AutoresearchLoop()
        should_stop, reason = loop.should_stop()

        assert should_stop is True
        assert "STOP" in reason


# =============================================================================
# Test: Improvement Threshold
# =============================================================================


class TestImprovementThreshold:
    """Verify 0.02 threshold for sharpe_net is enforced."""

    def test_improvement_threshold_is_0_02(self):
        """Sharpe_net improvement threshold should be 0.02."""
        assert IMPROVEMENT_THRESHOLDS["sharpe_net"] == 0.02

    def test_loop_uses_correct_threshold(self):
        """AutoresearchLoop.is_improvement should use 0.02 threshold."""
        loop = AutoresearchLoop(metric="sharpe_net")

        # Not enough improvement (0.01 < 0.02)
        assert loop.is_improvement(new_value=1.01, best_value=1.0) is False

        # Exactly at threshold (0.02 == 0.02)
        assert loop.is_improvement(new_value=1.02, best_value=1.0) is False

        # Just above threshold (0.021 > 0.02)
        assert loop.is_improvement(new_value=1.021, best_value=1.0) is True

        # Clear improvement
        assert loop.is_improvement(new_value=1.10, best_value=1.0) is True

    def test_brier_uses_inverted_threshold(self):
        """Brier score (lower is better) should use inverted threshold."""
        loop = AutoresearchLoop(metric="brier")

        # Brier improved (dropped from 0.25 to 0.24)
        # Threshold is -0.005, so needs to be < 0.25 - 0.005 = 0.245
        assert loop.is_improvement(new_value=0.24, best_value=0.25) is True

        # Not enough improvement
        assert loop.is_improvement(new_value=0.248, best_value=0.25) is False


# =============================================================================
# Test: Parameter Selection
# =============================================================================


class TestChooseNextParameter:
    """Tests for the parameter selection heuristic."""

    def test_round_robin_selection(self):
        """Should cycle through parameters in order."""
        state = ParameterState()
        current_values = {
            "max_depth": 6,
            "learning_rate": 0.1,
            "n_estimators": 100,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "false_bullish_penalty": 1.5,
            "false_bearish_penalty": 0.8,
            "n_folds": 5,
            "gap_bars": 24,
        }

        # First selection should be first parameter
        param1, _, _ = choose_next_parameter(state, current_values, [])
        assert param1 == PARAM_ORDER[0]

    def test_new_value_differs_from_old(self):
        """Should always return a different value than current."""
        state = ParameterState()
        current_values = {
            "max_depth": 6,
            "learning_rate": 0.1,
            "n_estimators": 100,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "false_bullish_penalty": 1.5,
            "false_bearish_penalty": 0.8,
            "n_folds": 5,
            "gap_bars": 24,
        }

        for _ in range(15):
            param, old_value, new_value = choose_next_parameter(state, current_values, [])
            assert new_value != old_value
            current_values[param] = new_value


# =============================================================================
# Test: Results Table Formatting
# =============================================================================


class TestFormatResultsTable:
    """Tests for the results table formatter."""

    def test_empty_results(self):
        """Should handle empty results."""
        table = format_results_table([])
        assert "no experiments" in table.lower()

    def test_formats_kept_and_reverted(self):
        """Should show kept and reverted status correctly."""
        results = [
            ExperimentResult(
                "2026-04-14 10:00:00",
                1,
                "max_depth",
                6,
                7,
                sharpe_net=1.0,
                ic=0.08,
                brier=0.22,
                accuracy=0.55,
                false_bullish_rate=0.15,
                kept=True,
            ),
            ExperimentResult(
                "2026-04-14 11:00:00",
                2,
                "learning_rate",
                0.1,
                0.05,
                sharpe_net=0.85,
                ic=0.06,
                brier=0.25,
                accuracy=0.52,
                false_bullish_rate=0.18,
                kept=False,
            ),
        ]

        table = format_results_table(results)

        assert "KEPT" in table
        assert "REVERTED" in table
        assert "max_depth" in table


# =============================================================================
# Test: Search Space Consistency
# =============================================================================


class TestSearchSpaceConsistency:
    """Tests to verify search space configuration."""

    def test_all_params_in_search_space(self):
        """All parameters in PARAM_ORDER should be in search space."""
        for param in PARAM_ORDER:
            assert param in PARAM_SEARCH_SPACE

    def test_search_space_values_sorted(self):
        """Search space values should be sorted for numeric params."""
        for param, values in PARAM_SEARCH_SPACE.items():
            assert values == sorted(values), f"{param} values are not sorted"

    def test_search_space_has_multiple_values(self):
        """Each parameter should have multiple values to try."""
        for param, values in PARAM_SEARCH_SPACE.items():
            assert len(values) >= 2, f"{param} needs at least 2 values"


# =============================================================================
# Test: AutoresearchLoop Stopping Conditions
# =============================================================================


class TestAutoresearchLoopStopping:
    """Tests for AutoresearchLoop stopping conditions."""

    def test_should_stop_max_experiments(self):
        """Should stop when max experiments reached."""
        loop = AutoresearchLoop(max_experiments=5)
        loop.results = [MagicMock() for _ in range(5)]

        should_stop, reason = loop.should_stop()
        assert should_stop is True
        assert "max experiments" in reason.lower()

    def test_should_stop_time_budget(self):
        """Should stop when time budget exhausted."""
        loop = AutoresearchLoop(time_budget_hours=0.001)  # ~3.6 seconds
        loop.start_time = 0  # Started at epoch

        should_stop, reason = loop.should_stop()
        assert should_stop is True
        assert "time budget" in reason.lower()

    def test_should_stop_interrupt(self):
        """Should stop when interrupted."""
        loop = AutoresearchLoop()
        loop.interrupted = True

        should_stop, reason = loop.should_stop()
        assert should_stop is True
        assert "interrupt" in reason.lower()

    def test_should_not_stop_initially(self):
        """Should not stop when just started."""
        with patch("run_autoresearch.check_stop_file", return_value=False):
            loop = AutoresearchLoop(max_experiments=10, time_budget_hours=1)
            should_stop, _ = loop.should_stop()
            assert should_stop is False


# =============================================================================
# Test: Metric Selection
# =============================================================================


class TestMetricSelection:
    """Tests for metric selection via --metric flag."""

    def test_get_metric_value_sharpe(self):
        """Should extract sharpe_net when metric is sharpe_net."""
        loop = AutoresearchLoop(metric="sharpe_net")
        result = EvalResult(
            sharpe_net=1.5,
            ic=0.08,
            ic_pvalue=0.01,
            brier=0.22,
            directional_accuracy=0.55,
            false_bullish_rate=0.15,
            false_bearish_rate=0.20,
            num_examples=1000,
            config_hash="abc",
        )
        assert loop.get_metric_value(result) == 1.5

    def test_get_metric_value_ic(self):
        """Should extract ic when metric is ic."""
        loop = AutoresearchLoop(metric="ic")
        result = EvalResult(
            sharpe_net=1.5,
            ic=0.08,
            ic_pvalue=0.01,
            brier=0.22,
            directional_accuracy=0.55,
            false_bullish_rate=0.15,
            false_bearish_rate=0.20,
            num_examples=1000,
            config_hash="abc",
        )
        assert loop.get_metric_value(result) == 0.08

    def test_get_metric_value_brier(self):
        """Should extract brier when metric is brier."""
        loop = AutoresearchLoop(metric="brier")
        result = EvalResult(
            sharpe_net=1.5,
            ic=0.08,
            ic_pvalue=0.01,
            brier=0.22,
            directional_accuracy=0.55,
            false_bullish_rate=0.15,
            false_bearish_rate=0.20,
            num_examples=1000,
            config_hash="abc",
        )
        assert loop.get_metric_value(result) == 0.22


# =============================================================================
# Test: Get Best Metric
# =============================================================================


class TestGetBestMetric:
    """Tests for get_best_metric function."""

    def test_get_best_sharpe_from_kept(self):
        """Should return best sharpe_net from kept results only."""
        results = [
            ExperimentResult(
                "",
                1,
                "lr",
                0,
                0,
                sharpe_net=0.8,
                ic=0.06,
                brier=0.22,
                accuracy=0.54,
                false_bullish_rate=0.15,
                kept=False,
            ),
            ExperimentResult(
                "",
                2,
                "lr",
                0,
                0,
                sharpe_net=1.2,
                ic=0.08,
                brier=0.20,
                accuracy=0.56,
                false_bullish_rate=0.12,
                kept=True,
            ),
            ExperimentResult(
                "",
                3,
                "lr",
                0,
                0,
                sharpe_net=1.5,
                ic=0.09,
                brier=0.19,
                accuracy=0.57,
                false_bullish_rate=0.10,
                kept=False,
            ),
            ExperimentResult(
                "",
                4,
                "lr",
                0,
                0,
                sharpe_net=1.0,
                ic=0.07,
                brier=0.21,
                accuracy=0.55,
                false_bullish_rate=0.13,
                kept=True,
            ),
        ]

        best = get_best_metric(results, "sharpe_net")
        assert best == 1.2  # Best among kept results

    def test_get_best_empty_results(self):
        """Should return -inf for empty results (or inf for brier)."""
        assert get_best_metric([], "sharpe_net") == -float("inf")
        assert get_best_metric([], "ic") == -float("inf")
        assert get_best_metric([], "brier") == float("inf")

    def test_get_best_no_kept_results(self):
        """Should return -inf if no kept results."""
        results = [
            ExperimentResult(
                "",
                1,
                "lr",
                0,
                0,
                sharpe_net=1.5,
                ic=0.09,
                brier=0.19,
                accuracy=0.57,
                false_bullish_rate=0.10,
                kept=False,
            ),
        ]
        assert get_best_metric(results, "sharpe_net") == -float("inf")
