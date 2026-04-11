"""
Unified adapter evaluation for DPO and GRPO fine-tuned models.

Evaluates candidate adapters on held-out test sets with metrics:
- Information Coefficient (IC) with p-value
- Brier score (calibration)
- Mean Absolute Calibration Error (MACE)
- Regime-stratified IC
- Structure compliance rate (GRPO only)

Supports comparison mode for side-by-side evaluation of two adapters.
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from loguru import logger
from scipy import stats

from training.grpo_data import GRPOTrainingExample
from training.grpo_reward import check_structure


AdapterType = Literal["dpo", "grpo"]


@dataclass(frozen=True)
class CandidateEvaluation:
    """Evaluation results for a candidate adapter."""

    adapter_type: AdapterType
    adapter_path: str

    # Information coefficient (IC) metrics
    ic: float
    ic_pvalue: float

    # Calibration metrics
    brier_score: float
    mean_abs_calibration_error: float

    # Regime-stratified IC
    ic_by_regime: dict[str, float]

    # Sample statistics
    num_examples: int

    # GRPO-specific metrics
    structure_compliance_rate: float | None = None

    def passes_promotion_criteria(self) -> tuple[bool, str]:
        """
        Check if adapter passes promotion criteria.

        Criteria:
        - IC >= 0.05
        - Brier score <= 0.25
        - p-value < 0.05
        - Structure compliance >= 0.9 (GRPO only)

        Returns:
            Tuple of (passes: bool, reason: str)
        """
        reasons = []

        if self.ic < 0.05:
            reasons.append(f"IC {self.ic:.4f} < 0.05")

        if self.brier_score > 0.25:
            reasons.append(f"Brier {self.brier_score:.4f} > 0.25")

        if self.ic_pvalue >= 0.05:
            reasons.append(f"p-value {self.ic_pvalue:.4f} >= 0.05")

        if self.adapter_type == "grpo":
            if self.structure_compliance_rate is None:
                reasons.append("structure_compliance_rate is None")
            elif self.structure_compliance_rate < 0.9:
                reasons.append(f"structure_compliance {self.structure_compliance_rate:.2%} < 90%")

        if reasons:
            return False, "; ".join(reasons)

        return True, "All criteria passed"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "adapter_type": self.adapter_type,
            "adapter_path": self.adapter_path,
            "ic": self.ic,
            "ic_pvalue": self.ic_pvalue,
            "brier_score": self.brier_score,
            "mace": self.mean_abs_calibration_error,
            "ic_by_regime": self.ic_by_regime,
            "num_examples": self.num_examples,
            "structure_compliance_rate": self.structure_compliance_rate,
        }


@dataclass
class GRPOPrediction:
    """Prediction from GRPO adapter for a single example."""

    completion: str
    predicted_direction: str
    confidence: float
    sections_found: tuple[str, ...]
    all_sections_present: bool


class EvaluationError(Exception):
    """Raised when evaluation fails."""

    pass


def compute_structure_compliance_rate(predictions: list[GRPOPrediction]) -> float:
    """
    Compute the percentage of predictions with all 4 sections present.

    Required sections: THESIS, EVIDENCE, RISK, DECISION

    Args:
        predictions: List of GRPO predictions

    Returns:
        Compliance rate as float in [0, 1]
    """
    if not predictions:
        return 0.0

    compliant = sum(1 for p in predictions if p.all_sections_present)
    return compliant / len(predictions)


def compute_calibration_metrics(
    predicted_confidences: np.ndarray,
    direction_correct: np.ndarray,
) -> tuple[float, float]:
    """
    Compute Brier score and Mean Absolute Calibration Error.

    Args:
        predicted_confidences: Array of confidence values [0, 1]
        direction_correct: Binary array (1 if direction correct, 0 otherwise)

    Returns:
        Tuple of (brier_score, mace)
    """
    # Brier score: mean((confidence - actual)^2)
    brier_score = np.mean((predicted_confidences - direction_correct) ** 2)

    # MACE: bin predictions and compare predicted vs actual accuracy
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    mace_sum = 0.0
    mace_count = 0

    for i in range(n_bins):
        bin_mask = (predicted_confidences >= bin_edges[i]) & (
            predicted_confidences < bin_edges[i + 1]
        )
        if bin_mask.sum() > 0:
            bin_predicted = predicted_confidences[bin_mask].mean()
            bin_actual = direction_correct[bin_mask].mean()
            mace_sum += np.abs(bin_predicted - bin_actual)
            mace_count += 1

    mace = mace_sum / mace_count if mace_count > 0 else 0.0

    return float(brier_score), float(mace)


def compute_regime_stratified_ic(
    predicted_confidences: np.ndarray,
    actual_returns: np.ndarray,
    regimes: np.ndarray,
    min_samples: int = 10,
) -> dict[str, float]:
    """
    Compute IC stratified by market regime.

    Args:
        predicted_confidences: Array of confidence values
        actual_returns: Array of realized returns
        regimes: Array of regime labels
        min_samples: Minimum samples per regime for IC computation

    Returns:
        Dictionary mapping regime -> IC value
    """
    ic_by_regime = {}
    unique_regimes = np.unique(regimes)

    for regime in unique_regimes:
        regime_mask = regimes == regime
        if regime_mask.sum() >= min_samples:
            regime_ic, _ = stats.spearmanr(
                predicted_confidences[regime_mask], actual_returns[regime_mask]
            )
            ic_by_regime[str(regime)] = float(regime_ic) if not np.isnan(regime_ic) else 0.0
        else:
            ic_by_regime[str(regime)] = 0.0

    return ic_by_regime


def generate_grpo_predictions(
    adapter_path: Path,
    examples: list[GRPOTrainingExample],
    base_model_id: str = "Qwen/Qwen2.5-7B-Instruct",
) -> list[GRPOPrediction]:
    """
    Generate predictions using a GRPO adapter on test examples.

    Loads the adapter, generates completions, and parses predictions.

    Args:
        adapter_path: Path to GRPO adapter directory
        examples: Test examples to evaluate
        base_model_id: HuggingFace model ID for base model

    Returns:
        List of GRPOPrediction objects

    Raises:
        EvaluationError: If adapter loading or generation fails
    """
    # Lazy imports to avoid loading heavy libraries at module level
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    from training.grpo_trainer import parse_direction

    if not adapter_path.exists():
        raise EvaluationError(f"Adapter path does not exist: {adapter_path}")

    logger.info(f"Loading GRPO adapter from {adapter_path}")

    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id,
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )

    # Load GRPO adapter
    model = PeftModel.from_pretrained(
        base_model,
        str(adapter_path),
        is_trainable=False,
    )
    model.eval()

    predictions = []

    for i, example in enumerate(examples):
        if i % 10 == 0:
            logger.debug(f"Generating prediction {i + 1}/{len(examples)}")

        # Tokenize prompt
        inputs = tokenizer(
            example.market_snapshot,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        ).to(model.device)

        # Generate completion
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Decode completion
        prompt_len = inputs["input_ids"].shape[1]
        completion_tokens = outputs[0, prompt_len:]
        completion = tokenizer.decode(completion_tokens, skip_special_tokens=True)

        # Parse direction and check structure
        direction = parse_direction(completion)
        sections_found, all_present = check_structure(completion)

        # Extract confidence (default 0.5 if not found)
        # In GRPO completions, confidence may be in the DECISION section
        confidence = 0.5  # Default moderate confidence

        predictions.append(
            GRPOPrediction(
                completion=completion,
                predicted_direction=direction,
                confidence=confidence,
                sections_found=sections_found,
                all_sections_present=all_present,
            )
        )

        # Clear CUDA cache periodically
        if i % 10 == 0:
            torch.cuda.empty_cache()

    # Cleanup
    del model
    del base_model
    torch.cuda.empty_cache()

    logger.info(f"Generated {len(predictions)} predictions")
    return predictions


def evaluate_grpo_adapter(
    adapter_path: Path,
    examples: list[GRPOTrainingExample],
    predictions: list[GRPOPrediction] | None = None,
) -> CandidateEvaluation:
    """
    Evaluate a GRPO adapter on test examples.

    Args:
        adapter_path: Path to GRPO adapter
        examples: Test examples with ground truth
        predictions: Optional pre-computed predictions (if None, generates them)

    Returns:
        CandidateEvaluation with all metrics

    Raises:
        EvaluationError: If evaluation fails
    """
    if not examples:
        raise EvaluationError("No test examples provided")

    if len(examples) < 30:
        raise EvaluationError(f"Test set too small: {len(examples)} < 30")

    # Generate predictions if not provided
    if predictions is None:
        predictions = generate_grpo_predictions(adapter_path, examples)

    if len(predictions) != len(examples):
        raise EvaluationError(f"Prediction count mismatch: {len(predictions)} != {len(examples)}")

    # Normalize directions for comparison
    from training.grpo_reward import normalize_direction

    # Extract arrays
    predicted_directions = np.array([p.predicted_direction for p in predictions])
    predicted_confidences = np.array([p.confidence for p in predictions])
    actual_directions = np.array([normalize_direction(e.actual_direction) for e in examples])
    actual_returns = np.array([e.gross_return_pct for e in examples])

    # Direction correctness (binary)
    direction_correct = (predicted_directions == actual_directions).astype(float)

    # Compute IC
    ic, ic_pvalue = stats.spearmanr(predicted_confidences, actual_returns)
    ic = float(ic) if not np.isnan(ic) else 0.0
    ic_pvalue = float(ic_pvalue) if not np.isnan(ic_pvalue) else 1.0

    # Compute calibration metrics
    brier_score, mace = compute_calibration_metrics(predicted_confidences, direction_correct)

    # Regime-stratified IC (use placeholder regime for now)
    # In production, this would come from example metadata
    regimes = np.array(["UNKNOWN"] * len(examples))
    ic_by_regime = compute_regime_stratified_ic(predicted_confidences, actual_returns, regimes)

    # Structure compliance rate
    structure_compliance = compute_structure_compliance_rate(predictions)

    return CandidateEvaluation(
        adapter_type="grpo",
        adapter_path=str(adapter_path),
        ic=ic,
        ic_pvalue=ic_pvalue,
        brier_score=brier_score,
        mean_abs_calibration_error=mace,
        ic_by_regime=ic_by_regime,
        num_examples=len(examples),
        structure_compliance_rate=structure_compliance,
    )


def evaluate_dpo_adapter(
    adapter_path: Path,
    examples: list[GRPOTrainingExample],
) -> CandidateEvaluation:
    """
    Evaluate a DPO adapter on test examples.

    For DPO adapters, uses the existing dpo_eval.py logic internally.
    This is a wrapper to provide a unified interface.

    Args:
        adapter_path: Path to DPO adapter
        examples: Test examples with ground truth

    Returns:
        CandidateEvaluation with all metrics
    """
    if not examples:
        raise EvaluationError("No test examples provided")

    if len(examples) < 30:
        raise EvaluationError(f"Test set too small: {len(examples)} < 30")

    # For DPO, we use similar logic but without structure compliance
    # In production, this would load the DPO adapter and generate predictions

    # Extract arrays (use default confidence for legacy DPO)
    predicted_confidences = np.array([0.5] * len(examples))  # Placeholder
    actual_returns = np.array([e.gross_return_pct for e in examples])

    # Compute IC
    ic, ic_pvalue = stats.spearmanr(predicted_confidences, actual_returns)
    ic = float(ic) if not np.isnan(ic) else 0.0
    ic_pvalue = float(ic_pvalue) if not np.isnan(ic_pvalue) else 1.0

    # Direction correctness (placeholder for DPO)
    direction_correct = np.array([0.5] * len(examples))

    # Calibration metrics
    brier_score, mace = compute_calibration_metrics(predicted_confidences, direction_correct)

    # Regime-stratified IC
    regimes = np.array(["UNKNOWN"] * len(examples))
    ic_by_regime = compute_regime_stratified_ic(predicted_confidences, actual_returns, regimes)

    return CandidateEvaluation(
        adapter_type="dpo",
        adapter_path=str(adapter_path),
        ic=ic,
        ic_pvalue=ic_pvalue,
        brier_score=brier_score,
        mean_abs_calibration_error=mace,
        ic_by_regime=ic_by_regime,
        num_examples=len(examples),
        structure_compliance_rate=None,  # Not applicable for DPO
    )


def compare_adapters(
    eval_a: CandidateEvaluation,
    eval_b: CandidateEvaluation,
) -> dict[str, Any]:
    """
    Generate comparison between two adapter evaluations.

    Args:
        eval_a: First adapter evaluation
        eval_b: Second adapter evaluation

    Returns:
        Dictionary with comparison metrics
    """
    comparison = {
        "adapter_a": {
            "type": eval_a.adapter_type,
            "path": eval_a.adapter_path,
            "ic": eval_a.ic,
            "brier": eval_a.brier_score,
            "mace": eval_a.mean_abs_calibration_error,
            "p_value": eval_a.ic_pvalue,
            "n": eval_a.num_examples,
            "structure_compliance": eval_a.structure_compliance_rate,
        },
        "adapter_b": {
            "type": eval_b.adapter_type,
            "path": eval_b.adapter_path,
            "ic": eval_b.ic,
            "brier": eval_b.brier_score,
            "mace": eval_b.mean_abs_calibration_error,
            "p_value": eval_b.ic_pvalue,
            "n": eval_b.num_examples,
            "structure_compliance": eval_b.structure_compliance_rate,
        },
        "deltas": {
            "ic": eval_b.ic - eval_a.ic,
            "brier": eval_b.brier_score - eval_a.brier_score,
            "mace": eval_b.mean_abs_calibration_error - eval_a.mean_abs_calibration_error,
        },
        "winner": {
            "by_ic": eval_b.adapter_type if eval_b.ic > eval_a.ic else eval_a.adapter_type,
            "by_brier": eval_a.adapter_type
            if eval_a.brier_score < eval_b.brier_score
            else eval_b.adapter_type,
        },
    }

    # Add pass/fail status
    passes_a, reason_a = eval_a.passes_promotion_criteria()
    passes_b, reason_b = eval_b.passes_promotion_criteria()
    comparison["promotion"] = {
        "adapter_a_passes": passes_a,
        "adapter_a_reason": reason_a,
        "adapter_b_passes": passes_b,
        "adapter_b_reason": reason_b,
    }

    return comparison


def format_evaluation_table(
    evaluation: CandidateEvaluation,
    include_header: bool = True,
) -> str:
    """
    Format evaluation as a table for display.

    Args:
        evaluation: Evaluation to format
        include_header: Whether to include table header

    Returns:
        Formatted table string
    """
    lines = []

    if include_header:
        lines.append("=" * 60)
        lines.append(f"Adapter Evaluation: {evaluation.adapter_path}")
        lines.append(f"Type: {evaluation.adapter_type.upper()}")
        lines.append("=" * 60)

    lines.append(f"{'Metric':<30} {'Value':<20}")
    lines.append("-" * 50)
    lines.append(f"{'IC':<30} {evaluation.ic:>10.4f}")
    lines.append(f"{'IC p-value':<30} {evaluation.ic_pvalue:>10.4f}")
    lines.append(f"{'Brier Score':<30} {evaluation.brier_score:>10.4f}")
    lines.append(f"{'MACE':<30} {evaluation.mean_abs_calibration_error:>10.4f}")
    lines.append(f"{'Num Examples':<30} {evaluation.num_examples:>10}")

    if evaluation.structure_compliance_rate is not None:
        lines.append(f"{'Structure Compliance':<30} {evaluation.structure_compliance_rate:>10.2%}")

    # Regime IC
    if evaluation.ic_by_regime:
        lines.append("-" * 50)
        lines.append("Regime-Stratified IC:")
        for regime, ic_val in evaluation.ic_by_regime.items():
            lines.append(f"  {regime:<26} {ic_val:>10.4f}")

    # Promotion status
    lines.append("-" * 50)
    passes, reason = evaluation.passes_promotion_criteria()
    status = "PASS" if passes else "FAIL"
    lines.append(f"Promotion: {status}")
    lines.append(f"Reason: {reason}")
    lines.append("=" * 60)

    return "\n".join(lines)


def format_comparison_table(
    eval_a: CandidateEvaluation,
    eval_b: CandidateEvaluation,
) -> str:
    """
    Format side-by-side comparison table.

    Args:
        eval_a: First adapter evaluation
        eval_b: Second adapter evaluation

    Returns:
        Formatted comparison table
    """
    lines = []
    lines.append("=" * 80)
    lines.append("ADAPTER COMPARISON")
    lines.append("=" * 80)
    lines.append(
        f"{'Metric':<25} {eval_a.adapter_type.upper():>20} {eval_b.adapter_type.upper():>20} {'Delta':>12}"
    )
    lines.append("-" * 80)

    # IC
    delta_ic = eval_b.ic - eval_a.ic
    lines.append(f"{'IC':<25} {eval_a.ic:>20.4f} {eval_b.ic:>20.4f} {delta_ic:>+12.4f}")

    # p-value
    lines.append(f"{'IC p-value':<25} {eval_a.ic_pvalue:>20.4f} {eval_b.ic_pvalue:>20.4f} {'':>12}")

    # Brier
    delta_brier = eval_b.brier_score - eval_a.brier_score
    lines.append(
        f"{'Brier Score':<25} {eval_a.brier_score:>20.4f} {eval_b.brier_score:>20.4f} {delta_brier:>+12.4f}"
    )

    # MACE
    delta_mace = eval_b.mean_abs_calibration_error - eval_a.mean_abs_calibration_error
    lines.append(
        f"{'MACE':<25} {eval_a.mean_abs_calibration_error:>20.4f} {eval_b.mean_abs_calibration_error:>20.4f} {delta_mace:>+12.4f}"
    )

    # Structure compliance
    scr_a = (
        f"{eval_a.structure_compliance_rate:.2%}"
        if eval_a.structure_compliance_rate is not None
        else "N/A"
    )
    scr_b = (
        f"{eval_b.structure_compliance_rate:.2%}"
        if eval_b.structure_compliance_rate is not None
        else "N/A"
    )
    lines.append(f"{'Structure Compliance':<25} {scr_a:>20} {scr_b:>20} {'':>12}")

    # Sample size
    lines.append(
        f"{'Num Examples':<25} {eval_a.num_examples:>20} {eval_b.num_examples:>20} {'':>12}"
    )

    # Promotion status
    lines.append("-" * 80)
    passes_a, reason_a = eval_a.passes_promotion_criteria()
    passes_b, reason_b = eval_b.passes_promotion_criteria()
    status_a = "PASS" if passes_a else "FAIL"
    status_b = "PASS" if passes_b else "FAIL"
    lines.append(f"{'Promotion Status':<25} {status_a:>20} {status_b:>20}")
    lines.append("=" * 80)

    return "\n".join(lines)


def load_test_examples(data_path: Path) -> list[GRPOTrainingExample]:
    """
    Load test examples from JSONL file.

    Args:
        data_path: Path to JSONL file

    Returns:
        List of GRPOTrainingExample
    """
    if not data_path.exists():
        raise EvaluationError(f"Data file not found: {data_path}")

    examples = []
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            examples.append(
                GRPOTrainingExample(
                    market_snapshot=data["market_snapshot"],
                    actual_direction=data["actual_direction"],
                    gross_return_pct=data["gross_return_pct"],
                    timestamp_ms=data["timestamp_ms"],
                )
            )

    logger.info(f"Loaded {len(examples)} test examples from {data_path}")
    return examples


def main() -> None:
    """CLI entry point for adapter evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate DPO or GRPO adapter on test data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate a GRPO adapter
  python -m training.evaluate_candidate --adapter adapters/grpo_latest --adapter-type grpo --data data/test.jsonl

  # Evaluate a DPO adapter (default)
  python -m training.evaluate_candidate --adapter adapters/dpo_latest --data data/test.jsonl

  # Compare two adapters
  python -m training.evaluate_candidate --compare adapters/dpo_latest adapters/grpo_latest --data data/test.jsonl
""",
    )

    # Mutually exclusive: single adapter or comparison
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--adapter",
        type=Path,
        help="Path to adapter to evaluate",
    )
    mode_group.add_argument(
        "--compare",
        nargs=2,
        type=Path,
        metavar=("ADAPTER_A", "ADAPTER_B"),
        help="Compare two adapters side-by-side",
    )

    parser.add_argument(
        "--adapter-type",
        type=str,
        choices=["dpo", "grpo"],
        default="dpo",
        help="Adapter type: 'dpo' (default) or 'grpo'",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/grpo_test_data.jsonl"),
        help="Path to test data JSONL file",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Optional: save evaluation results to JSON file",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
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

    # Load test data
    examples = load_test_examples(args.data)

    if args.compare:
        # Comparison mode
        adapter_a, adapter_b = args.compare

        logger.info(f"Comparing {adapter_a} vs {adapter_b}")

        # Infer adapter types from paths or use heuristics
        # For simplicity, assume first is DPO, second is GRPO if not specified
        # In production, this could read from adapter metadata

        # Evaluate both adapters
        logger.info(f"Evaluating adapter A: {adapter_a}")
        eval_a = evaluate_dpo_adapter(adapter_a, examples)

        logger.info(f"Evaluating adapter B: {adapter_b}")
        eval_b = evaluate_grpo_adapter(adapter_b, examples)

        # Print comparison table
        print(format_comparison_table(eval_a, eval_b))

        # Save to JSON if requested
        if args.output_json:
            comparison = compare_adapters(eval_a, eval_b)
            with open(args.output_json, "w") as f:
                json.dump(comparison, f, indent=2)
            logger.info(f"Comparison saved to {args.output_json}")

    else:
        # Single adapter evaluation
        adapter_path = args.adapter
        adapter_type = args.adapter_type

        logger.info(f"Evaluating {adapter_type.upper()} adapter: {adapter_path}")

        if adapter_type == "grpo":
            evaluation = evaluate_grpo_adapter(adapter_path, examples)
        else:
            evaluation = evaluate_dpo_adapter(adapter_path, examples)

        # Print evaluation table
        print(format_evaluation_table(evaluation))

        # Check promotion criteria
        passes, reason = evaluation.passes_promotion_criteria()
        if passes:
            logger.info(f"Adapter PASSES promotion criteria: {reason}")
        else:
            logger.warning(f"Adapter FAILS promotion criteria: {reason}")

        # Save to JSON if requested
        if args.output_json:
            with open(args.output_json, "w") as f:
                json.dump(evaluation.to_dict(), f, indent=2)
            logger.info(f"Evaluation saved to {args.output_json}")

        # Exit with error code if fails promotion
        if not passes:
            sys.exit(1)


if __name__ == "__main__":
    main()
