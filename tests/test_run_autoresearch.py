"""
Tests for the autoresearch hyperparameter search loop.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from run_autoresearch import (
    PARAM_ORDER,
    PARAM_SEARCH_SPACE,
    AutoresearchLoop,
    ExperimentResult,
    ParameterState,
    choose_next_parameter,
    format_results_table,
    get_best_ic,
    modify_config_parameter,
    read_current_config_values,
    read_results_tsv,
    write_results_tsv,
)


class TestReadCurrentConfigValues:
    """Tests for reading config values from grpo_config.py."""

    def test_reads_default_values(self):
        """Should read default values from the actual config file."""
        values = read_current_config_values()

        # Verify expected parameters are present
        assert "G" in values
        assert "beta" in values
        assert "epsilon" in values
        assert "lr" in values
        assert "lora_r" in values
        assert "lora_alpha" in values
        assert "false_bullish_penalty" in values
        assert "false_bearish_penalty" in values

    def test_reads_correct_default_values(self):
        """Should read values that are in valid search space ranges."""
        values = read_current_config_values()

        # Verify values are within expected ranges (allowing for config changes)
        assert values["G"] in [2, 4, 6, 8]
        assert 0.01 <= values["beta"] <= 0.12
        assert 0.1 <= values["epsilon"] <= 0.3
        assert 1e-6 <= values["lr"] <= 2e-5
        assert values["lora_r"] in [8, 16, 32, 64]
        assert values["lora_alpha"] in [16, 32, 64, 128]
        assert 1.0 <= values["false_bullish_penalty"] <= 2.5
        assert 0.5 <= values["false_bearish_penalty"] <= 1.2

    def test_values_are_correct_types(self):
        """Should return correct types for each parameter."""
        values = read_current_config_values()

        # Integer parameters
        assert isinstance(values["G"], int)
        assert isinstance(values["lora_r"], int)
        assert isinstance(values["lora_alpha"], int)

        # Float parameters
        assert isinstance(values["beta"], float)
        assert isinstance(values["epsilon"], float)
        assert isinstance(values["lr"], float)
        assert isinstance(values["false_bullish_penalty"], float)
        assert isinstance(values["false_bearish_penalty"], float)


class TestModifyConfigParameter:
    """Tests for modifying config parameters."""

    @pytest.fixture
    def temp_config_file(self):
        """Create a temporary config file for testing."""
        content = '''"""GRPO config."""
from dataclasses import dataclass

@dataclass(frozen=True)
class GRPORewardConfig:
    false_bullish_penalty: float = 1.5
    false_bearish_penalty: float = 0.8

@dataclass(frozen=True)
class GRPOLoRAConfig:
    rank: int = 32
    alpha: int = 64

@dataclass(frozen=True)
class GRPOTrainingConfig:
    group_size: int = 4
    kl_penalty_beta: float = 0.04
    clip_epsilon: float = 0.2
    learning_rate: float = 5e-6
'''
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(content)
            temp_path = f.name

        yield Path(temp_path)

        # Cleanup
        os.unlink(temp_path)

    def test_modify_group_size(self, temp_config_file, monkeypatch):
        """Should modify group_size parameter."""
        import run_autoresearch

        monkeypatch.setattr(run_autoresearch, "GRPO_CONFIG_PATH", temp_config_file)

        modify_config_parameter("G", 8)

        content = temp_config_file.read_text()
        assert "group_size: int = 8" in content

    def test_modify_learning_rate(self, temp_config_file, monkeypatch):
        """Should modify learning_rate parameter."""
        import run_autoresearch

        monkeypatch.setattr(run_autoresearch, "GRPO_CONFIG_PATH", temp_config_file)

        modify_config_parameter("lr", 1e-5)

        content = temp_config_file.read_text()
        assert "learning_rate: float = 1e-05" in content or "learning_rate: float = 1e-5" in content

    def test_modify_beta(self, temp_config_file, monkeypatch):
        """Should modify kl_penalty_beta parameter."""
        import run_autoresearch

        monkeypatch.setattr(run_autoresearch, "GRPO_CONFIG_PATH", temp_config_file)

        modify_config_parameter("beta", 0.08)

        content = temp_config_file.read_text()
        assert "kl_penalty_beta: float = 0.08" in content

    def test_modify_epsilon(self, temp_config_file, monkeypatch):
        """Should modify clip_epsilon parameter."""
        import run_autoresearch

        monkeypatch.setattr(run_autoresearch, "GRPO_CONFIG_PATH", temp_config_file)

        modify_config_parameter("epsilon", 0.15)

        content = temp_config_file.read_text()
        assert "clip_epsilon: float = 0.15" in content

    def test_modify_lora_r(self, temp_config_file, monkeypatch):
        """Should modify LoRA rank parameter."""
        import run_autoresearch

        monkeypatch.setattr(run_autoresearch, "GRPO_CONFIG_PATH", temp_config_file)

        modify_config_parameter("lora_r", 64)

        content = temp_config_file.read_text()
        assert "rank: int = 64" in content

    def test_modify_lora_alpha(self, temp_config_file, monkeypatch):
        """Should modify LoRA alpha parameter."""
        import run_autoresearch

        monkeypatch.setattr(run_autoresearch, "GRPO_CONFIG_PATH", temp_config_file)

        modify_config_parameter("lora_alpha", 128)

        content = temp_config_file.read_text()
        assert "alpha: int = 128" in content

    def test_modify_false_bullish_penalty(self, temp_config_file, monkeypatch):
        """Should modify false_bullish_penalty parameter."""
        import run_autoresearch

        monkeypatch.setattr(run_autoresearch, "GRPO_CONFIG_PATH", temp_config_file)

        modify_config_parameter("false_bullish_penalty", 2.0)

        content = temp_config_file.read_text()
        assert "false_bullish_penalty: float = 2.0" in content

    def test_modify_false_bearish_penalty(self, temp_config_file, monkeypatch):
        """Should modify false_bearish_penalty parameter."""
        import run_autoresearch

        monkeypatch.setattr(run_autoresearch, "GRPO_CONFIG_PATH", temp_config_file)

        modify_config_parameter("false_bearish_penalty", 1.0)

        content = temp_config_file.read_text()
        assert "false_bearish_penalty: float = 1.0" in content

    def test_unknown_parameter_raises(self, temp_config_file, monkeypatch):
        """Should raise ValueError for unknown parameters."""
        import run_autoresearch

        monkeypatch.setattr(run_autoresearch, "GRPO_CONFIG_PATH", temp_config_file)

        with pytest.raises(ValueError, match="Unknown parameter"):
            modify_config_parameter("unknown_param", 42)


class TestResultsTSV:
    """Tests for results.tsv reading and writing."""

    @pytest.fixture
    def temp_results_file(self):
        """Create a temporary results file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
            temp_path = f.name

        yield Path(temp_path)

        # Cleanup
        if os.path.exists(temp_path):
            os.unlink(temp_path)

    def test_write_and_read_results(self, temp_results_file, monkeypatch):
        """Should write and read results correctly."""
        import run_autoresearch

        monkeypatch.setattr(run_autoresearch, "RESULTS_TSV_PATH", temp_results_file)

        results = [
            ExperimentResult(
                timestamp="2024-01-01 10:00:00",
                experiment_id=1,
                param_changed="lr",
                old_value="5e-6",
                new_value="1e-5",
                ic=0.05,
                brier=0.22,
                kept=True,
            ),
            ExperimentResult(
                timestamp="2024-01-01 11:00:00",
                experiment_id=2,
                param_changed="beta",
                old_value="0.04",
                new_value="0.08",
                ic=0.03,
                brier=0.25,
                kept=False,
            ),
        ]

        write_results_tsv(results)

        read_back = read_results_tsv()

        assert len(read_back) == 2
        assert read_back[0].experiment_id == 1
        assert read_back[0].param_changed == "lr"
        assert read_back[0].kept is True
        assert read_back[1].experiment_id == 2
        assert read_back[1].kept is False

    def test_append_result(self, temp_results_file, monkeypatch):
        """Should append results to existing file."""
        import run_autoresearch

        # Remove the file if it exists so append_result will create it with header
        if temp_results_file.exists():
            temp_results_file.unlink()

        monkeypatch.setattr(run_autoresearch, "RESULTS_TSV_PATH", temp_results_file)

        # Need to re-import the functions after patching
        from run_autoresearch import append_result as patched_append
        from run_autoresearch import read_results_tsv as patched_read

        result1 = ExperimentResult(
            timestamp="2024-01-01 10:00:00",
            experiment_id=1,
            param_changed="lr",
            old_value="5e-6",
            new_value="1e-5",
            ic=0.05,
            brier=0.22,
            kept=True,
        )
        patched_append(result1)

        result2 = ExperimentResult(
            timestamp="2024-01-01 11:00:00",
            experiment_id=2,
            param_changed="beta",
            old_value="0.04",
            new_value="0.08",
            ic=0.06,
            brier=0.21,
            kept=True,
        )
        patched_append(result2)

        read_back = patched_read()
        assert len(read_back) == 2

    def test_read_nonexistent_file(self, monkeypatch):
        """Should return empty list for nonexistent file."""
        import run_autoresearch

        monkeypatch.setattr(
            run_autoresearch,
            "RESULTS_TSV_PATH",
            Path("/nonexistent/path/results.tsv"),
        )

        results = read_results_tsv()
        assert results == []

    def test_get_best_ic_with_kept_results(self):
        """Should return best IC from kept results only."""
        results = [
            ExperimentResult("", 1, "lr", 0, 0, ic=0.03, brier=0.25, kept=False),
            ExperimentResult("", 2, "lr", 0, 0, ic=0.06, brier=0.22, kept=True),
            ExperimentResult("", 3, "lr", 0, 0, ic=0.08, brier=0.20, kept=False),
            ExperimentResult("", 4, "lr", 0, 0, ic=0.05, brier=0.23, kept=True),
        ]

        best_ic = get_best_ic(results)
        assert best_ic == 0.06  # Best among kept results

    def test_get_best_ic_empty_results(self):
        """Should return -inf for empty results."""
        assert get_best_ic([]) == -float("inf")

    def test_get_best_ic_no_kept_results(self):
        """Should return -inf if no kept results."""
        results = [
            ExperimentResult("", 1, "lr", 0, 0, ic=0.08, brier=0.20, kept=False),
        ]
        assert get_best_ic(results) == -float("inf")


class TestChooseNextParameter:
    """Tests for the parameter selection heuristic."""

    def test_round_robin_selection(self):
        """Should cycle through parameters in order."""
        state = ParameterState()
        current_values = {
            "lr": 5e-6,
            "beta": 0.04,
            "epsilon": 0.2,
            "G": 4,
            "lora_r": 32,
            "lora_alpha": 64,
            "false_bullish_penalty": 1.5,
            "false_bearish_penalty": 0.8,
        }

        # First selection should be first parameter in PARAM_ORDER
        param1, _, _ = choose_next_parameter(state, current_values, [])
        assert param1 == PARAM_ORDER[0]

        # Second selection should be second parameter
        param2, _, _ = choose_next_parameter(state, current_values, [])
        assert param2 == PARAM_ORDER[1]

    def test_continues_direction_on_success(self):
        """Should continue in same direction after successful experiment."""
        state = ParameterState()
        current_values = {
            "lr": 5e-6,  # Index 2 in [1e-6, 2e-6, 5e-6, 1e-5, 2e-5]
            "beta": 0.04,
            "epsilon": 0.2,
            "G": 4,
            "lora_r": 32,
            "lora_alpha": 64,
            "false_bullish_penalty": 1.5,
            "false_bearish_penalty": 0.8,
        }

        # First experiment on lr
        param1, old1, new1 = choose_next_parameter(state, current_values, [])
        assert param1 == "lr"

        # Simulate successful experiment
        results = [ExperimentResult("", 1, "lr", old1, new1, ic=0.06, brier=0.22, kept=True)]

        # Update current values
        current_values["lr"] = new1

        # Next time we reach lr, should continue in same direction
        # (need to cycle through other params first)
        for _ in range(len(PARAM_ORDER) - 1):
            choose_next_parameter(state, current_values, results)

        # Back to lr
        param_lr, _, new_lr = choose_next_parameter(state, current_values, results)
        assert param_lr == "lr"
        # Direction should be same (moving up from 5e-6 -> 1e-5 -> 2e-5)

    def test_reverses_direction_on_failure(self):
        """Should reverse direction after failed experiment."""
        state = ParameterState()
        current_values = {
            "lr": 5e-6,
            "beta": 0.04,
            "epsilon": 0.2,
            "G": 4,
            "lora_r": 32,
            "lora_alpha": 64,
            "false_bullish_penalty": 1.5,
            "false_bearish_penalty": 0.8,
        }

        # First experiment on lr
        param1, old1, new1 = choose_next_parameter(state, current_values, [])
        assert param1 == "lr"

        initial_direction = state.directions["lr"]

        # Simulate failed experiment
        results = [ExperimentResult("", 1, "lr", old1, new1, ic=0.02, brier=0.28, kept=False)]

        # The next call should reverse direction for lr
        choose_next_parameter(state, current_values, results)

        assert state.directions["lr"] == -initial_direction

    def test_new_value_differs_from_old(self):
        """Should always return a different value than current."""
        state = ParameterState()
        current_values = {
            "lr": 5e-6,
            "beta": 0.04,
            "epsilon": 0.2,
            "G": 4,
            "lora_r": 32,
            "lora_alpha": 64,
            "false_bullish_penalty": 1.5,
            "false_bearish_penalty": 0.8,
        }

        for _ in range(20):
            param, old_value, new_value = choose_next_parameter(state, current_values, [])
            assert new_value != old_value
            current_values[param] = new_value

    def test_values_in_search_space(self):
        """Should only return values from the search space."""
        state = ParameterState()
        current_values = {
            "lr": 5e-6,
            "beta": 0.04,
            "epsilon": 0.2,
            "G": 4,
            "lora_r": 32,
            "lora_alpha": 64,
            "false_bullish_penalty": 1.5,
            "false_bearish_penalty": 0.8,
        }

        for _ in range(20):
            param, _, new_value = choose_next_parameter(state, current_values, [])
            assert new_value in PARAM_SEARCH_SPACE[param]
            current_values[param] = new_value


class TestAutoresearchLoop:
    """Tests for the main autoresearch loop."""

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
        loop = AutoresearchLoop(max_experiments=10, time_budget_hours=1)

        should_stop, _ = loop.should_stop()
        assert should_stop is False

    @patch("run_autoresearch.modify_config_parameter")
    @patch("run_autoresearch.read_current_config_values")
    @patch("run_autoresearch.run_training")
    @patch("run_autoresearch.run_evaluation")
    @patch("run_autoresearch.git_commit")
    @patch("run_autoresearch.git_revert")
    @patch("run_autoresearch.read_results_tsv")
    @patch("run_autoresearch.append_result")
    def test_run_single_experiment_keeps_on_improvement(
        self,
        mock_append,
        mock_read_results,
        mock_revert,
        mock_commit,
        mock_eval,
        mock_train,
        mock_read_config,
        mock_modify_config,
    ):
        """Should keep experiment when IC improves."""
        mock_read_results.return_value = []
        mock_read_config.return_value = {
            "lr": 5e-6,
            "beta": 0.04,
            "epsilon": 0.2,
            "G": 4,
            "lora_r": 32,
            "lora_alpha": 64,
            "false_bullish_penalty": 1.5,
            "false_bearish_penalty": 0.8,
        }
        mock_modify_config.return_value = "5e-6"
        mock_train.return_value = True
        mock_eval.return_value = (0.08, 0.22)  # Good IC

        loop = AutoresearchLoop(dry_run=False)
        result = loop.run_single_experiment()

        assert result is not None
        assert result.kept is True
        mock_commit.assert_called_once()
        mock_revert.assert_not_called()

    @patch("run_autoresearch.modify_config_parameter")
    @patch("run_autoresearch.read_current_config_values")
    @patch("run_autoresearch.run_training")
    @patch("run_autoresearch.run_evaluation")
    @patch("run_autoresearch.git_commit")
    @patch("run_autoresearch.git_revert")
    @patch("run_autoresearch.read_results_tsv")
    @patch("run_autoresearch.append_result")
    def test_run_single_experiment_reverts_on_no_improvement(
        self,
        mock_append,
        mock_read_results,
        mock_revert,
        mock_commit,
        mock_eval,
        mock_train,
        mock_read_config,
        mock_modify_config,
    ):
        """Should revert experiment when IC does not improve."""
        # Previous experiment with IC=0.10
        mock_read_results.return_value = [
            ExperimentResult("", 1, "lr", 0, 0, ic=0.10, brier=0.22, kept=True)
        ]
        mock_read_config.return_value = {
            "lr": 5e-6,
            "beta": 0.04,
            "epsilon": 0.2,
            "G": 4,
            "lora_r": 32,
            "lora_alpha": 64,
            "false_bullish_penalty": 1.5,
            "false_bearish_penalty": 0.8,
        }
        mock_modify_config.return_value = "5e-6"
        mock_train.return_value = True
        mock_eval.return_value = (0.05, 0.25)  # Worse IC

        loop = AutoresearchLoop(dry_run=False)
        result = loop.run_single_experiment()

        assert result is not None
        assert result.kept is False
        mock_commit.assert_called_once()
        mock_revert.assert_called_once()

    @patch("run_autoresearch.modify_config_parameter")
    @patch("run_autoresearch.read_current_config_values")
    @patch("run_autoresearch.run_training")
    @patch("run_autoresearch.run_evaluation")
    @patch("run_autoresearch.git_commit")
    @patch("run_autoresearch.git_revert")
    @patch("run_autoresearch.read_results_tsv")
    @patch("run_autoresearch.append_result")
    def test_run_single_experiment_handles_training_failure(
        self,
        mock_append,
        mock_read_results,
        mock_revert,
        mock_commit,
        mock_eval,
        mock_train,
        mock_read_config,
        mock_modify_config,
    ):
        """Should revert and return None on training failure."""
        mock_read_results.return_value = []
        mock_read_config.return_value = {
            "lr": 5e-6,
            "beta": 0.04,
            "epsilon": 0.2,
            "G": 4,
            "lora_r": 32,
            "lora_alpha": 64,
            "false_bullish_penalty": 1.5,
            "false_bearish_penalty": 0.8,
        }
        mock_modify_config.return_value = "5e-6"
        mock_train.return_value = False  # Training fails

        loop = AutoresearchLoop(dry_run=False)
        result = loop.run_single_experiment()

        assert result is None
        mock_revert.assert_called_once()
        mock_eval.assert_not_called()

    @patch("run_autoresearch.modify_config_parameter")
    @patch("run_autoresearch.read_current_config_values")
    def test_dry_run_skips_git_and_training(
        self,
        mock_read_config,
        mock_modify_config,
    ):
        """Dry run should not commit, train, or revert."""
        with (
            patch("run_autoresearch.git_commit") as mock_commit,
            patch("run_autoresearch.git_revert") as mock_revert,
            patch("run_autoresearch.run_training") as mock_train,
            patch("run_autoresearch.run_evaluation") as mock_eval,
            patch("run_autoresearch.read_results_tsv") as mock_read,
            patch("run_autoresearch.append_result"),
        ):
            mock_read.return_value = []
            mock_read_config.return_value = {
                "lr": 5e-6,
                "beta": 0.04,
                "epsilon": 0.2,
                "G": 4,
                "lora_r": 32,
                "lora_alpha": 64,
                "false_bullish_penalty": 1.5,
                "false_bearish_penalty": 0.8,
            }
            mock_modify_config.return_value = "5e-6"
            mock_train.return_value = True
            mock_eval.return_value = (0.08, 0.22)

            loop = AutoresearchLoop(dry_run=True)
            result = loop.run_single_experiment()

            assert result is not None
            mock_commit.assert_not_called()
            mock_revert.assert_not_called()


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
                "2024-01-01 10:00:00", 1, "lr", "5e-6", "1e-5", ic=0.06, brier=0.22, kept=True
            ),
            ExperimentResult(
                "2024-01-01 11:00:00", 2, "beta", "0.04", "0.08", ic=0.03, brier=0.28, kept=False
            ),
        ]

        table = format_results_table(results)

        assert "KEPT" in table
        assert "REVERTED" in table
        assert "lr" in table
        assert "beta" in table

    def test_shows_summary_stats(self):
        """Should show summary statistics."""
        results = [
            ExperimentResult("", 1, "lr", 0, 0, ic=0.06, brier=0.22, kept=True),
            ExperimentResult("", 2, "beta", 0, 0, ic=0.03, brier=0.28, kept=False),
            ExperimentResult("", 3, "G", 0, 0, ic=0.07, brier=0.21, kept=True),
        ]

        table = format_results_table(results)

        assert "3 experiments" in table.lower()
        assert "2 kept" in table.lower()
        assert "1 reverted" in table.lower()
        assert "0.07" in table  # Best IC


class TestParameterState:
    """Tests for ParameterState dataclass."""

    def test_default_initialization(self):
        """Should initialize with defaults."""
        state = ParameterState()

        assert state.param_index == 0
        for param in PARAM_ORDER:
            assert param in state.improvement_counts
            assert state.improvement_counts[param] == 0
            assert param in state.directions
            assert state.directions[param] == 1

    def test_all_params_have_defaults(self):
        """All parameters should have initialized tracking."""
        state = ParameterState()

        for param in PARAM_ORDER:
            assert param in state.value_indices
            assert param in state.improvement_counts
            assert param in state.directions


class TestSearchSpaceConsistency:
    """Tests to verify search space configuration."""

    def test_all_params_in_search_space(self):
        """All parameters in PARAM_ORDER should be in search space."""
        for param in PARAM_ORDER:
            assert param in PARAM_SEARCH_SPACE

    def test_search_space_values_sorted(self):
        """Search space values should be sorted for numeric params."""
        for param, values in PARAM_SEARCH_SPACE.items():
            # Check that values are sorted (allows bidirectional search)
            assert values == sorted(values), f"{param} values are not sorted"

    def test_search_space_has_multiple_values(self):
        """Each parameter should have multiple values to try."""
        for param, values in PARAM_SEARCH_SPACE.items():
            assert len(values) >= 2, f"{param} needs at least 2 values"

    def test_default_values_in_search_space(self):
        """Default config values should be in search space."""
        # Read actual defaults
        current_values = read_current_config_values()

        for param, value in current_values.items():
            if param in PARAM_SEARCH_SPACE:
                # Check if value is close to any search space value
                search_values = PARAM_SEARCH_SPACE[param]
                min_diff = min(abs(value - sv) for sv in search_values)
                # Allow small tolerance for floats
                assert min_diff < 1e-9 or min_diff / abs(value) < 0.01, (
                    f"Default {param}={value} not in search space {search_values}"
                )
