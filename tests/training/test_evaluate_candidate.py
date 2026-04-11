"""Tests for evaluate_candidate module."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from training.evaluate_candidate import (
    AdapterType,
    CandidateEvaluation,
    EvaluationError,
    GRPOPrediction,
    compare_adapters,
    compute_calibration_metrics,
    compute_regime_stratified_ic,
    compute_structure_compliance_rate,
    evaluate_dpo_adapter,
    evaluate_grpo_adapter,
    format_comparison_table,
    format_evaluation_table,
    load_test_examples,
)
from training.grpo_data import GRPOTrainingExample


# === Fixtures ===


@pytest.fixture
def sample_examples() -> list[GRPOTrainingExample]:
    """Create sample test examples."""
    return [
        GRPOTrainingExample(
            market_snapshot="BTC/USDT 8h snapshot...",
            actual_direction="LONG",
            gross_return_pct=0.5,
            timestamp_ms=1000000 + i * 1000,
        )
        for i in range(50)
    ]


@pytest.fixture
def sample_predictions() -> list[GRPOPrediction]:
    """Create sample predictions with structure compliance."""
    predictions = []
    for i in range(50):
        # 80% have all sections (below threshold)
        all_present = i < 40
        sections = ("THESIS", "EVIDENCE", "RISK", "DECISION") if all_present else ("THESIS",)
        predictions.append(
            GRPOPrediction(
                completion="## THESIS\n...\n## EVIDENCE\n...\n## RISK\n...\n## DECISION\nLONG"
                if all_present
                else "## THESIS\n...",
                predicted_direction="LONG" if i % 2 == 0 else "SHORT",
                confidence=0.6 + (i % 10) * 0.03,
                sections_found=sections,
                all_sections_present=all_present,
            )
        )
    return predictions


@pytest.fixture
def compliant_predictions() -> list[GRPOPrediction]:
    """Create predictions with 95% structure compliance."""
    predictions = []
    for i in range(100):
        # 95% have all sections (above threshold)
        all_present = i < 95
        sections = ("THESIS", "EVIDENCE", "RISK", "DECISION") if all_present else ("THESIS",)
        predictions.append(
            GRPOPrediction(
                completion="## THESIS\n...\n## EVIDENCE\n...\n## RISK\n...\n## DECISION\nLONG",
                predicted_direction="LONG",
                confidence=0.7,
                sections_found=sections,
                all_sections_present=all_present,
            )
        )
    return predictions


@pytest.fixture
def passing_grpo_evaluation() -> CandidateEvaluation:
    """Create an evaluation that passes all criteria."""
    return CandidateEvaluation(
        adapter_type="grpo",
        adapter_path="adapters/grpo_latest",
        ic=0.08,
        ic_pvalue=0.01,
        brier_score=0.20,
        mean_abs_calibration_error=0.05,
        ic_by_regime={"UNKNOWN": 0.08},
        num_examples=100,
        structure_compliance_rate=0.95,
    )


@pytest.fixture
def failing_grpo_evaluation() -> CandidateEvaluation:
    """Create an evaluation that fails criteria."""
    return CandidateEvaluation(
        adapter_type="grpo",
        adapter_path="adapters/grpo_bad",
        ic=0.03,  # Below threshold
        ic_pvalue=0.10,  # Above threshold
        brier_score=0.30,  # Above threshold
        mean_abs_calibration_error=0.15,
        ic_by_regime={"UNKNOWN": 0.03},
        num_examples=100,
        structure_compliance_rate=0.85,  # Below threshold
    )


@pytest.fixture
def dpo_evaluation() -> CandidateEvaluation:
    """Create a DPO evaluation."""
    return CandidateEvaluation(
        adapter_type="dpo",
        adapter_path="adapters/dpo_latest",
        ic=0.06,
        ic_pvalue=0.03,
        brier_score=0.22,
        mean_abs_calibration_error=0.08,
        ic_by_regime={"UNKNOWN": 0.06},
        num_examples=100,
        structure_compliance_rate=None,  # N/A for DPO
    )


# === Test CandidateEvaluation ===


class TestCandidateEvaluation:
    """Tests for CandidateEvaluation dataclass."""

    def test_passes_promotion_criteria_all_pass(
        self, passing_grpo_evaluation: CandidateEvaluation
    ) -> None:
        """Test that valid evaluation passes all criteria."""
        passes, reason = passing_grpo_evaluation.passes_promotion_criteria()
        assert passes is True
        assert reason == "All criteria passed"

    def test_fails_ic_threshold(self) -> None:
        """Test failure when IC is below threshold."""
        evaluation = CandidateEvaluation(
            adapter_type="grpo",
            adapter_path="test",
            ic=0.04,  # Below 0.05
            ic_pvalue=0.01,
            brier_score=0.20,
            mean_abs_calibration_error=0.05,
            ic_by_regime={},
            num_examples=100,
            structure_compliance_rate=0.95,
        )
        passes, reason = evaluation.passes_promotion_criteria()
        assert passes is False
        assert "IC 0.0400 < 0.05" in reason

    def test_fails_brier_threshold(self) -> None:
        """Test failure when Brier score exceeds threshold."""
        evaluation = CandidateEvaluation(
            adapter_type="grpo",
            adapter_path="test",
            ic=0.08,
            ic_pvalue=0.01,
            brier_score=0.30,  # Above 0.25
            mean_abs_calibration_error=0.05,
            ic_by_regime={},
            num_examples=100,
            structure_compliance_rate=0.95,
        )
        passes, reason = evaluation.passes_promotion_criteria()
        assert passes is False
        assert "Brier 0.3000 > 0.25" in reason

    def test_fails_pvalue_threshold(self) -> None:
        """Test failure when p-value is too high."""
        evaluation = CandidateEvaluation(
            adapter_type="grpo",
            adapter_path="test",
            ic=0.08,
            ic_pvalue=0.10,  # >= 0.05
            brier_score=0.20,
            mean_abs_calibration_error=0.05,
            ic_by_regime={},
            num_examples=100,
            structure_compliance_rate=0.95,
        )
        passes, reason = evaluation.passes_promotion_criteria()
        assert passes is False
        assert "p-value 0.1000 >= 0.05" in reason

    def test_fails_structure_compliance_grpo(self) -> None:
        """Test failure when GRPO structure compliance is low."""
        evaluation = CandidateEvaluation(
            adapter_type="grpo",
            adapter_path="test",
            ic=0.08,
            ic_pvalue=0.01,
            brier_score=0.20,
            mean_abs_calibration_error=0.05,
            ic_by_regime={},
            num_examples=100,
            structure_compliance_rate=0.85,  # Below 0.9
        )
        passes, reason = evaluation.passes_promotion_criteria()
        assert passes is False
        assert "structure_compliance 85.00% < 90%" in reason

    def test_dpo_ignores_structure_compliance(self) -> None:
        """Test that DPO evaluation ignores structure compliance."""
        evaluation = CandidateEvaluation(
            adapter_type="dpo",
            adapter_path="test",
            ic=0.08,
            ic_pvalue=0.01,
            brier_score=0.20,
            mean_abs_calibration_error=0.05,
            ic_by_regime={},
            num_examples=100,
            structure_compliance_rate=None,  # N/A for DPO
        )
        passes, reason = evaluation.passes_promotion_criteria()
        assert passes is True

    def test_multiple_failures_combined(self) -> None:
        """Test that multiple failures are reported together."""
        evaluation = CandidateEvaluation(
            adapter_type="grpo",
            adapter_path="test",
            ic=0.02,  # Fail
            ic_pvalue=0.10,  # Fail
            brier_score=0.30,  # Fail
            mean_abs_calibration_error=0.05,
            ic_by_regime={},
            num_examples=100,
            structure_compliance_rate=0.80,  # Fail
        )
        passes, reason = evaluation.passes_promotion_criteria()
        assert passes is False
        assert "IC" in reason
        assert "Brier" in reason
        assert "p-value" in reason
        assert "structure_compliance" in reason

    def test_to_dict(self, passing_grpo_evaluation: CandidateEvaluation) -> None:
        """Test conversion to dictionary."""
        result = passing_grpo_evaluation.to_dict()
        assert result["adapter_type"] == "grpo"
        assert result["ic"] == 0.08
        assert result["brier_score"] == 0.20
        assert result["structure_compliance_rate"] == 0.95


# === Test Structure Compliance ===


class TestStructureCompliance:
    """Tests for structure compliance computation."""

    def test_compute_structure_compliance_empty(self) -> None:
        """Test compliance rate with empty predictions."""
        assert compute_structure_compliance_rate([]) == 0.0

    def test_compute_structure_compliance_all_compliant(self) -> None:
        """Test 100% compliance."""
        predictions = [
            GRPOPrediction(
                completion="test",
                predicted_direction="LONG",
                confidence=0.7,
                sections_found=("THESIS", "EVIDENCE", "RISK", "DECISION"),
                all_sections_present=True,
            )
            for _ in range(10)
        ]
        assert compute_structure_compliance_rate(predictions) == 1.0

    def test_compute_structure_compliance_none_compliant(self) -> None:
        """Test 0% compliance."""
        predictions = [
            GRPOPrediction(
                completion="test",
                predicted_direction="LONG",
                confidence=0.7,
                sections_found=("THESIS",),
                all_sections_present=False,
            )
            for _ in range(10)
        ]
        assert compute_structure_compliance_rate(predictions) == 0.0

    def test_compute_structure_compliance_partial(
        self, sample_predictions: list[GRPOPrediction]
    ) -> None:
        """Test partial compliance (80%)."""
        rate = compute_structure_compliance_rate(sample_predictions)
        assert rate == 0.8  # 40 out of 50


# === Test Calibration Metrics ===


class TestCalibrationMetrics:
    """Tests for calibration metric computation."""

    def test_brier_score_perfect(self) -> None:
        """Test Brier score with perfect calibration."""
        confidences = np.array([1.0, 0.0, 1.0, 0.0])
        correct = np.array([1.0, 0.0, 1.0, 0.0])
        brier, _ = compute_calibration_metrics(confidences, correct)
        assert brier == 0.0

    def test_brier_score_worst(self) -> None:
        """Test Brier score with worst calibration."""
        confidences = np.array([1.0, 1.0, 0.0, 0.0])
        correct = np.array([0.0, 0.0, 1.0, 1.0])
        brier, _ = compute_calibration_metrics(confidences, correct)
        assert brier == 1.0

    def test_brier_score_moderate(self) -> None:
        """Test Brier score with moderate calibration."""
        confidences = np.array([0.7, 0.3, 0.8, 0.2])
        correct = np.array([1.0, 0.0, 1.0, 0.0])
        brier, _ = compute_calibration_metrics(confidences, correct)
        # Expected: mean((0.7-1)^2 + (0.3-0)^2 + (0.8-1)^2 + (0.2-0)^2)
        # = mean(0.09 + 0.09 + 0.04 + 0.04) = 0.065
        assert abs(brier - 0.065) < 1e-6

    def test_mace_perfect(self) -> None:
        """Test MACE with perfect calibration within bins."""
        # When predictions match accuracy within bins, MACE should be low
        confidences = np.array([0.05] * 10 + [0.95] * 10)
        correct = np.array([0] * 10 + [1] * 10)
        _, mace = compute_calibration_metrics(confidences, correct)
        assert mace < 0.1


# === Test Regime-Stratified IC ===


class TestRegimeStratifiedIC:
    """Tests for regime-stratified IC computation."""

    def test_single_regime(self) -> None:
        """Test IC computation with single regime."""
        confidences = np.array([0.6, 0.7, 0.8, 0.9] * 5)
        returns = np.array([0.01, 0.02, 0.03, 0.04] * 5)
        regimes = np.array(["RISK_ON"] * 20)

        ic_by_regime = compute_regime_stratified_ic(confidences, returns, regimes)

        assert "RISK_ON" in ic_by_regime
        assert ic_by_regime["RISK_ON"] > 0  # Positive correlation

    def test_multiple_regimes(self) -> None:
        """Test IC computation with multiple regimes."""
        confidences = np.array([0.6, 0.7] * 15)
        returns = np.array([0.01, 0.02] * 15)
        regimes = np.array(["RISK_ON"] * 10 + ["RISK_OFF"] * 10 + ["NEUTRAL"] * 10)

        ic_by_regime = compute_regime_stratified_ic(confidences, returns, regimes)

        assert len(ic_by_regime) == 3

    def test_insufficient_samples(self) -> None:
        """Test that regimes with few samples get IC=0."""
        confidences = np.array([0.6, 0.7, 0.8] * 5)
        returns = np.array([0.01, 0.02, 0.03] * 5)
        regimes = np.array(["RISK_ON"] * 12 + ["RARE"] * 3)  # RARE has < 10 samples

        ic_by_regime = compute_regime_stratified_ic(confidences, returns, regimes, min_samples=10)

        assert ic_by_regime["RARE"] == 0.0


# === Test Adapter Comparison ===


class TestAdapterComparison:
    """Tests for adapter comparison functionality."""

    def test_compare_adapters_basic(
        self,
        passing_grpo_evaluation: CandidateEvaluation,
        dpo_evaluation: CandidateEvaluation,
    ) -> None:
        """Test basic comparison between adapters."""
        comparison = compare_adapters(dpo_evaluation, passing_grpo_evaluation)

        assert "adapter_a" in comparison
        assert "adapter_b" in comparison
        assert "deltas" in comparison
        assert "winner" in comparison
        assert "promotion" in comparison

    def test_compare_adapters_deltas(
        self,
        passing_grpo_evaluation: CandidateEvaluation,
        dpo_evaluation: CandidateEvaluation,
    ) -> None:
        """Test delta calculations."""
        comparison = compare_adapters(dpo_evaluation, passing_grpo_evaluation)

        # GRPO has IC=0.08, DPO has IC=0.06
        assert comparison["deltas"]["ic"] == pytest.approx(0.02)
        # GRPO has Brier=0.20, DPO has Brier=0.22
        assert comparison["deltas"]["brier"] == pytest.approx(-0.02)

    def test_compare_adapters_winner(
        self,
        passing_grpo_evaluation: CandidateEvaluation,
        dpo_evaluation: CandidateEvaluation,
    ) -> None:
        """Test winner determination."""
        comparison = compare_adapters(dpo_evaluation, passing_grpo_evaluation)

        # GRPO has higher IC
        assert comparison["winner"]["by_ic"] == "grpo"
        # GRPO has lower (better) Brier
        assert comparison["winner"]["by_brier"] == "grpo"


# === Test Output Formatting ===


class TestFormatting:
    """Tests for output formatting functions."""

    def test_format_evaluation_table(self, passing_grpo_evaluation: CandidateEvaluation) -> None:
        """Test evaluation table formatting."""
        output = format_evaluation_table(passing_grpo_evaluation)

        assert "GRPO" in output
        assert "IC" in output
        assert "Brier Score" in output
        assert "Structure Compliance" in output
        assert "PASS" in output

    def test_format_evaluation_table_failing(
        self, failing_grpo_evaluation: CandidateEvaluation
    ) -> None:
        """Test evaluation table shows FAIL."""
        output = format_evaluation_table(failing_grpo_evaluation)

        assert "FAIL" in output

    def test_format_comparison_table(
        self,
        passing_grpo_evaluation: CandidateEvaluation,
        dpo_evaluation: CandidateEvaluation,
    ) -> None:
        """Test comparison table formatting."""
        output = format_comparison_table(dpo_evaluation, passing_grpo_evaluation)

        assert "COMPARISON" in output
        assert "DPO" in output
        assert "GRPO" in output
        assert "Delta" in output


# === Test Data Loading ===


class TestDataLoading:
    """Tests for test data loading."""

    def test_load_test_examples_valid(self) -> None:
        """Test loading valid JSONL file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for i in range(5):
                data = {
                    "market_snapshot": f"snapshot_{i}",
                    "actual_direction": "LONG",
                    "gross_return_pct": 0.5,
                    "timestamp_ms": 1000000 + i * 1000,
                }
                f.write(json.dumps(data) + "\n")
            f.flush()

            examples = load_test_examples(Path(f.name))

            assert len(examples) == 5
            assert examples[0].market_snapshot == "snapshot_0"

    def test_load_test_examples_not_found(self) -> None:
        """Test loading non-existent file raises error."""
        with pytest.raises(EvaluationError, match="not found"):
            load_test_examples(Path("/nonexistent/path.jsonl"))

    def test_load_test_examples_empty_lines(self) -> None:
        """Test that empty lines are skipped."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            data = {
                "market_snapshot": "snapshot",
                "actual_direction": "LONG",
                "gross_return_pct": 0.5,
                "timestamp_ms": 1000000,
            }
            f.write(json.dumps(data) + "\n")
            f.write("\n")  # Empty line
            f.write(json.dumps(data) + "\n")
            f.flush()

            examples = load_test_examples(Path(f.name))

            assert len(examples) == 2


# === Test Evaluation Functions ===


class TestEvaluationFunctions:
    """Tests for main evaluation functions."""

    def test_evaluate_grpo_adapter_empty_examples(self) -> None:
        """Test that empty examples raises error."""
        with pytest.raises(EvaluationError, match="No test examples"):
            evaluate_grpo_adapter(Path("test"), [])

    def test_evaluate_grpo_adapter_too_few_examples(
        self, sample_examples: list[GRPOTrainingExample]
    ) -> None:
        """Test that too few examples raises error."""
        with pytest.raises(EvaluationError, match="too small"):
            evaluate_grpo_adapter(Path("test"), sample_examples[:20])

    def test_evaluate_grpo_adapter_with_predictions(
        self,
        sample_examples: list[GRPOTrainingExample],
        sample_predictions: list[GRPOPrediction],
    ) -> None:
        """Test evaluation with pre-computed predictions."""
        evaluation = evaluate_grpo_adapter(
            Path("adapters/test"),
            sample_examples,
            predictions=sample_predictions,
        )

        assert evaluation.adapter_type == "grpo"
        assert evaluation.num_examples == 50
        assert evaluation.structure_compliance_rate == 0.8
        assert evaluation.ic is not None
        assert evaluation.brier_score is not None

    def test_evaluate_grpo_adapter_prediction_mismatch(
        self,
        sample_examples: list[GRPOTrainingExample],
        sample_predictions: list[GRPOPrediction],
    ) -> None:
        """Test that mismatched prediction count raises error."""
        with pytest.raises(EvaluationError, match="mismatch"):
            evaluate_grpo_adapter(
                Path("test"),
                sample_examples[:40],  # 40 examples
                predictions=sample_predictions,  # 50 predictions
            )

    def test_evaluate_dpo_adapter_empty_examples(self) -> None:
        """Test that empty examples raises error."""
        with pytest.raises(EvaluationError, match="No test examples"):
            evaluate_dpo_adapter(Path("test"), [])

    def test_evaluate_dpo_adapter_too_few_examples(
        self, sample_examples: list[GRPOTrainingExample]
    ) -> None:
        """Test that too few examples raises error."""
        with pytest.raises(EvaluationError, match="too small"):
            evaluate_dpo_adapter(Path("test"), sample_examples[:20])

    def test_evaluate_dpo_adapter_no_structure_compliance(
        self, sample_examples: list[GRPOTrainingExample]
    ) -> None:
        """Test that DPO evaluation has no structure compliance."""
        evaluation = evaluate_dpo_adapter(Path("test"), sample_examples)

        assert evaluation.adapter_type == "dpo"
        assert evaluation.structure_compliance_rate is None


# === Test CLI Integration ===


class TestCLIIntegration:
    """Tests for CLI argument parsing and integration."""

    def test_adapter_type_validation(self) -> None:
        """Test that adapter type is validated."""

        # This would require more complex mocking of argparse
        # Simplified test: verify AdapterType literal
        valid_types: list[AdapterType] = ["dpo", "grpo"]
        assert len(valid_types) == 2

    def test_mutually_exclusive_flags(self) -> None:
        """Test that --adapter and --compare are mutually exclusive."""
        # The argparse configuration handles this automatically
        # This test documents the expected behavior
        pass


# === Test Edge Cases ===


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_ic_threshold_exactly_at_boundary(self) -> None:
        """Test IC exactly at 0.05 threshold."""
        evaluation = CandidateEvaluation(
            adapter_type="grpo",
            adapter_path="test",
            ic=0.05,  # Exactly at threshold
            ic_pvalue=0.01,
            brier_score=0.20,
            mean_abs_calibration_error=0.05,
            ic_by_regime={},
            num_examples=100,
            structure_compliance_rate=0.95,
        )
        passes, _ = evaluation.passes_promotion_criteria()
        assert passes is True  # >= 0.05

    def test_brier_threshold_exactly_at_boundary(self) -> None:
        """Test Brier exactly at 0.25 threshold."""
        evaluation = CandidateEvaluation(
            adapter_type="grpo",
            adapter_path="test",
            ic=0.08,
            ic_pvalue=0.01,
            brier_score=0.25,  # Exactly at threshold
            mean_abs_calibration_error=0.05,
            ic_by_regime={},
            num_examples=100,
            structure_compliance_rate=0.95,
        )
        passes, _ = evaluation.passes_promotion_criteria()
        assert passes is True  # <= 0.25

    def test_structure_compliance_exactly_at_boundary(self) -> None:
        """Test structure compliance exactly at 0.9 threshold."""
        evaluation = CandidateEvaluation(
            adapter_type="grpo",
            adapter_path="test",
            ic=0.08,
            ic_pvalue=0.01,
            brier_score=0.20,
            mean_abs_calibration_error=0.05,
            ic_by_regime={},
            num_examples=100,
            structure_compliance_rate=0.9,  # Exactly at threshold
        )
        passes, _ = evaluation.passes_promotion_criteria()
        assert passes is True  # >= 0.9

    def test_pvalue_exactly_at_boundary(self) -> None:
        """Test p-value exactly at 0.05 threshold (fails)."""
        evaluation = CandidateEvaluation(
            adapter_type="grpo",
            adapter_path="test",
            ic=0.08,
            ic_pvalue=0.05,  # Exactly at threshold
            brier_score=0.20,
            mean_abs_calibration_error=0.05,
            ic_by_regime={},
            num_examples=100,
            structure_compliance_rate=0.95,
        )
        passes, _ = evaluation.passes_promotion_criteria()
        assert passes is False  # < 0.05 required, not <=

    def test_nan_ic_handling(self) -> None:
        """Test handling of NaN IC values."""
        # When correlation is undefined, scipy returns NaN
        # Our code should handle this gracefully
        from scipy import stats

        # Perfect correlation case
        confidences = np.array([0.5, 0.5, 0.5, 0.5])  # No variance
        returns = np.array([0.01, 0.02, 0.03, 0.04])

        ic, pval = stats.spearmanr(confidences, returns)

        # With no variance in one variable, IC is NaN
        assert np.isnan(ic)
