#!/usr/bin/env python3
"""
End-to-end GRPO training pipeline.

Orchestrates the full GRPO training pipeline:
1. SFT data generation (calls sft_data_generator.py)
2. SFT training (calls sft_trainer.py)
3. GRPO training (calls grpo_trainer.py)
4. Evaluation (calls evaluate_candidate.py)
5. Promotion gate

Usage:
    # Dry run (print plan without executing)
    python run_grpo_training.py --dry-run

    # Full training run
    python run_grpo_training.py

    # Limit SFT data generation (for testing)
    python run_grpo_training.py --limit 50

    # Force regenerate SFT data
    python run_grpo_training.py --regenerate

    # Force retrain SFT adapter
    python run_grpo_training.py --retrain-sft

    # Override max steps
    python run_grpo_training.py --max-steps 1000
"""

import argparse
import json
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

from loguru import logger


# Default paths
DEFAULT_SFT_DATA_PATH = Path("data/sft_training_data.jsonl")
DEFAULT_SFT_ADAPTER_PATH = Path("adapters/sft_base")
DEFAULT_GRPO_DATA_PATH = Path("data/grpo_training_data.jsonl")


@dataclass
class PhaseResult:
    """Result of a pipeline phase."""

    phase: str
    success: bool
    duration_seconds: float
    message: str
    artifacts: dict[str, Any]


@dataclass
class PipelineConfig:
    """Configuration for the GRPO training pipeline."""

    # SFT data generation
    sft_data_path: Path
    sft_data_limit: int | None
    regenerate_sft_data: bool

    # SFT training
    sft_adapter_path: Path
    retrain_sft: bool

    # GRPO training
    grpo_data_path: Path
    max_steps: int | None
    grpo_config_path: Path | None

    # General
    dry_run: bool


def print_banner(phase_name: str, description: str) -> None:
    """Print a phase banner."""
    print()
    print("=" * 70)
    print(f"PHASE: {phase_name}")
    print(f"       {description}")
    print("=" * 70)
    print()


def phase_sft_data_generation(config: PipelineConfig) -> PhaseResult:
    """
    Phase 1: SFT data generation.

    Generates SFT training data with reasoning traces from the critic model.
    Skips if data exists and --regenerate not passed.
    """
    print_banner("SFT Data Generation", "Generate reasoning traces from deepseek-r1:14b")

    start_time = time.time()

    # Check if we can skip
    if config.sft_data_path.exists() and not config.regenerate_sft_data:
        # Count existing examples
        with open(config.sft_data_path) as f:
            count = sum(1 for line in f if line.strip())
        logger.info(
            f"SFT data exists with {count} examples, skipping generation "
            f"(use --regenerate to force)"
        )
        return PhaseResult(
            phase="sft_data_generation",
            success=True,
            duration_seconds=time.time() - start_time,
            message=f"Skipped: {count} examples already exist",
            artifacts={"sft_data_path": str(config.sft_data_path), "count": count},
        )

    if config.dry_run:
        logger.info("[DRY RUN] Would generate SFT data")
        return PhaseResult(
            phase="sft_data_generation",
            success=True,
            duration_seconds=0.0,
            message="[DRY RUN] Would generate SFT data",
            artifacts={},
        )

    # Import and run
    import asyncio
    from training.sft_data_generator import generate_sft_dataset

    try:
        count = asyncio.run(
            generate_sft_dataset(
                output_file=config.sft_data_path,
                limit=config.sft_data_limit,
                resume=not config.regenerate_sft_data,
            )
        )
        return PhaseResult(
            phase="sft_data_generation",
            success=True,
            duration_seconds=time.time() - start_time,
            message=f"Generated {count} SFT examples",
            artifacts={"sft_data_path": str(config.sft_data_path), "count": count},
        )
    except Exception as e:
        logger.exception(f"SFT data generation failed: {e}")
        return PhaseResult(
            phase="sft_data_generation",
            success=False,
            duration_seconds=time.time() - start_time,
            message=f"Failed: {e}",
            artifacts={},
        )


def phase_sft_training(config: PipelineConfig) -> PhaseResult:
    """
    Phase 2: SFT training.

    Fine-tunes qwen3:8b on SFT data with LoRA.
    Skips if adapter exists and --retrain-sft not passed.
    """
    print_banner("SFT Training", "Fine-tune qwen3:8b on reasoning traces")

    start_time = time.time()

    # Check if we can skip
    if config.sft_adapter_path.exists() and not config.retrain_sft:
        logger.info(
            f"SFT adapter exists at {config.sft_adapter_path}, skipping training "
            f"(use --retrain-sft to force)"
        )
        return PhaseResult(
            phase="sft_training",
            success=True,
            duration_seconds=time.time() - start_time,
            message="Skipped: adapter already exists",
            artifacts={"adapter_path": str(config.sft_adapter_path)},
        )

    if config.dry_run:
        logger.info("[DRY RUN] Would train SFT adapter")
        return PhaseResult(
            phase="sft_training",
            success=True,
            duration_seconds=0.0,
            message="[DRY RUN] Would train SFT adapter",
            artifacts={},
        )

    # Import and run
    from training.sft_trainer import train_sft

    try:
        result = train_sft(
            data_path=config.sft_data_path,
            output_dir=config.sft_adapter_path.parent,
            run_eval=False,  # We'll do evaluation in phase 4
        )

        if not result.success:
            return PhaseResult(
                phase="sft_training",
                success=False,
                duration_seconds=time.time() - start_time,
                message=f"Failed: {result.error}",
                artifacts={},
            )

        return PhaseResult(
            phase="sft_training",
            success=True,
            duration_seconds=time.time() - start_time,
            message=f"Training complete: loss={result.training_loss:.4f}",
            artifacts={
                "adapter_path": str(result.adapter_path),
                "training_loss": result.training_loss,
                "validation_loss": result.validation_loss,
                "epochs": result.epochs_completed,
            },
        )
    except Exception as e:
        logger.exception(f"SFT training failed: {e}")
        return PhaseResult(
            phase="sft_training",
            success=False,
            duration_seconds=time.time() - start_time,
            message=f"Failed: {e}",
            artifacts={},
        )


def phase_grpo_training(config: PipelineConfig) -> PhaseResult:
    """
    Phase 3: GRPO training.

    Runs GRPO RL training on the SFT adapter.
    Always runs (no skip option).
    """
    print_banner("GRPO Training", "Group Relative Policy Optimization on SFT adapter")

    start_time = time.time()

    if config.dry_run:
        logger.info("[DRY RUN] Would run GRPO training")
        return PhaseResult(
            phase="grpo_training",
            success=True,
            duration_seconds=0.0,
            message="[DRY RUN] Would run GRPO training",
            artifacts={},
        )

    # Check for GRPO training data
    if not config.grpo_data_path.exists():
        return PhaseResult(
            phase="grpo_training",
            success=False,
            duration_seconds=time.time() - start_time,
            message=f"GRPO training data not found: {config.grpo_data_path}",
            artifacts={},
        )

    # Load training data
    from training.grpo_data import GRPOTrainingExample

    examples = []
    with open(config.grpo_data_path) as f:
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

    if not examples:
        return PhaseResult(
            phase="grpo_training",
            success=False,
            duration_seconds=time.time() - start_time,
            message="No GRPO training examples found",
            artifacts={},
        )

    logger.info(f"Loaded {len(examples)} GRPO training examples")

    # Load config with overrides
    from training.grpo_config import load_grpo_config

    overrides: dict[str, Any] = {}
    if config.max_steps is not None:
        overrides["max_steps"] = config.max_steps

    grpo_config = load_grpo_config(overrides) if overrides else None

    # Run training
    from training.grpo_trainer import GRPOTrainer

    try:
        trainer = GRPOTrainer(config=grpo_config)
        result = trainer.train(examples)

        if not result.success:
            return PhaseResult(
                phase="grpo_training",
                success=False,
                duration_seconds=time.time() - start_time,
                message=f"Failed: {result.error}",
                artifacts={},
            )

        return PhaseResult(
            phase="grpo_training",
            success=True,
            duration_seconds=time.time() - start_time,
            message=f"Training complete: {result.steps_completed} steps",
            artifacts={
                "adapter_path": str(result.adapter_path),
                "steps_completed": result.steps_completed,
                **result.final_metrics,
            },
        )
    except Exception as e:
        logger.exception(f"GRPO training failed: {e}")
        return PhaseResult(
            phase="grpo_training",
            success=False,
            duration_seconds=time.time() - start_time,
            message=f"Failed: {e}",
            artifacts={},
        )


def phase_evaluation(
    config: PipelineConfig,
    adapter_path: Path,
) -> PhaseResult:
    """
    Phase 4: Evaluation.

    Evaluates the GRPO adapter on held-out test data.
    """
    print_banner("Evaluation", "Evaluate GRPO adapter on test data")

    start_time = time.time()

    if config.dry_run:
        logger.info("[DRY RUN] Would evaluate GRPO adapter")
        return PhaseResult(
            phase="evaluation",
            success=True,
            duration_seconds=0.0,
            message="[DRY RUN] Would evaluate GRPO adapter",
            artifacts={},
        )

    # Load test data
    test_data_path = Path("data/grpo_test_data.jsonl")
    if not test_data_path.exists():
        logger.warning(f"Test data not found at {test_data_path}, using training data split")
        test_data_path = config.grpo_data_path

    from training.evaluate_candidate import (
        evaluate_grpo_adapter,
        load_test_examples,
        format_evaluation_table,
        EvaluationError,
    )

    try:
        examples = load_test_examples(test_data_path)

        # Use last 20% as test set if no separate test file
        if test_data_path == config.grpo_data_path:
            test_size = max(30, len(examples) // 5)
            examples = examples[-test_size:]
            logger.info(f"Using last {len(examples)} examples as test set")

        evaluation = evaluate_grpo_adapter(adapter_path, examples)

        # Print evaluation table
        print(format_evaluation_table(evaluation))

        passes, reason = evaluation.passes_promotion_criteria()

        return PhaseResult(
            phase="evaluation",
            success=True,
            duration_seconds=time.time() - start_time,
            message=f"IC={evaluation.ic:.4f}, Brier={evaluation.brier_score:.4f}",
            artifacts={
                "ic": evaluation.ic,
                "ic_pvalue": evaluation.ic_pvalue,
                "brier_score": evaluation.brier_score,
                "mace": evaluation.mean_abs_calibration_error,
                "structure_compliance": evaluation.structure_compliance_rate,
                "num_examples": evaluation.num_examples,
                "passes_promotion": passes,
                "promotion_reason": reason,
            },
        )
    except EvaluationError as e:
        logger.error(f"Evaluation failed: {e}")
        return PhaseResult(
            phase="evaluation",
            success=False,
            duration_seconds=time.time() - start_time,
            message=f"Failed: {e}",
            artifacts={},
        )
    except Exception as e:
        logger.exception(f"Evaluation failed: {e}")
        return PhaseResult(
            phase="evaluation",
            success=False,
            duration_seconds=time.time() - start_time,
            message=f"Failed: {e}",
            artifacts={},
        )


def phase_promotion(
    config: PipelineConfig,
    adapter_path: Path,
    evaluation_artifacts: dict[str, Any],
) -> PhaseResult:
    """
    Phase 5: Promotion gate.

    If all metrics pass, rename adapter with .promoted suffix.
    """
    print_banner("Promotion Gate", "Check metrics and promote if passed")

    start_time = time.time()

    if config.dry_run:
        logger.info("[DRY RUN] Would check promotion criteria")
        return PhaseResult(
            phase="promotion",
            success=True,
            duration_seconds=0.0,
            message="[DRY RUN] Would check promotion criteria",
            artifacts={},
        )

    passes = evaluation_artifacts.get("passes_promotion", False)
    reason = evaluation_artifacts.get("promotion_reason", "Unknown")

    if not passes:
        print()
        print("=" * 70)
        print("PROMOTION FAILED")
        print("=" * 70)
        print()
        print("Failed criteria:")
        for criterion in reason.split("; "):
            print(f"  - {criterion}")
        print()

        return PhaseResult(
            phase="promotion",
            success=False,
            duration_seconds=time.time() - start_time,
            message=f"Failed: {reason}",
            artifacts={"promoted": False, "reason": reason},
        )

    # Promote: rename with .promoted suffix
    promoted_path = adapter_path.with_suffix(".promoted")

    # If promoted path exists, remove it first
    if promoted_path.exists():
        shutil.rmtree(promoted_path)

    # Copy adapter to promoted path
    shutil.copytree(adapter_path, promoted_path)

    # Add promotion metadata
    metadata_path = promoted_path / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
    else:
        metadata = {}

    metadata["promoted_at"] = datetime.now(UTC).isoformat()
    metadata["promotion_metrics"] = {
        "ic": evaluation_artifacts.get("ic"),
        "brier_score": evaluation_artifacts.get("brier_score"),
        "structure_compliance": evaluation_artifacts.get("structure_compliance"),
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print()
    print("=" * 70)
    print("PROMOTION SUCCESSFUL")
    print("=" * 70)
    print()
    print(f"Adapter promoted to: {promoted_path}")
    print()
    print("Metrics:")
    print(f"  IC:                   {evaluation_artifacts.get('ic', 'N/A'):.4f}")
    print(f"  Brier Score:          {evaluation_artifacts.get('brier_score', 'N/A'):.4f}")
    print(f"  Structure Compliance: {evaluation_artifacts.get('structure_compliance', 'N/A'):.2%}")
    print()

    return PhaseResult(
        phase="promotion",
        success=True,
        duration_seconds=time.time() - start_time,
        message=f"Promoted to {promoted_path}",
        artifacts={"promoted": True, "promoted_path": str(promoted_path)},
    )


def print_plan(config: PipelineConfig) -> None:
    """Print the execution plan without running."""
    print()
    print("=" * 70)
    print("GRPO TRAINING PIPELINE - EXECUTION PLAN")
    print("=" * 70)
    print()

    phases = [
        (
            "1. SFT Data Generation",
            f"Skip: {config.sft_data_path.exists() and not config.regenerate_sft_data}",
            f"Path: {config.sft_data_path}",
            f"Limit: {config.sft_data_limit or 'None (full run)'}",
        ),
        (
            "2. SFT Training",
            f"Skip: {config.sft_adapter_path.exists() and not config.retrain_sft}",
            f"Data: {config.sft_data_path}",
            f"Output: {config.sft_adapter_path}",
        ),
        (
            "3. GRPO Training",
            "Skip: Never (always runs)",
            f"Data: {config.grpo_data_path}",
            f"Max steps: {config.max_steps or 'Default (5000)'}",
        ),
        (
            "4. Evaluation",
            "Skip: Never",
            "Adapter: Output from Phase 3",
            "Test data: data/grpo_test_data.jsonl",
        ),
        (
            "5. Promotion Gate",
            "Skip: Never",
            "Criteria: IC>=0.05, Brier<=0.25, p<0.05, structure>=0.9",
            "Action: Rename with .promoted suffix if passed",
        ),
    ]

    for phase in phases:
        print(f"{phase[0]}")
        for detail in phase[1:]:
            print(f"   {detail}")
        print()

    print("=" * 70)
    print()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="End-to-end GRPO training pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run (print plan)
  python run_grpo_training.py --dry-run

  # Full training run
  python run_grpo_training.py

  # Limit SFT examples (for testing)
  python run_grpo_training.py --limit 50

  # Force regenerate SFT data
  python run_grpo_training.py --regenerate

  # Force retrain SFT adapter
  python run_grpo_training.py --retrain-sft

  # Override GRPO max steps
  python run_grpo_training.py --max-steps 1000

  # Use custom config
  python run_grpo_training.py --config path/to/grpo_config.py
        """,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit SFT data generation to N examples (for testing)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override GRPO max_steps",
    )
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Force regenerate SFT data even if it exists",
    )
    parser.add_argument(
        "--retrain-sft",
        action="store_true",
        help="Force retrain SFT adapter even if it exists",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the execution plan without running",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to custom GRPO config file",
    )
    parser.add_argument(
        "--sft-data-path",
        type=Path,
        default=DEFAULT_SFT_DATA_PATH,
        help=f"Path to SFT training data (default: {DEFAULT_SFT_DATA_PATH})",
    )
    parser.add_argument(
        "--sft-adapter-path",
        type=Path,
        default=DEFAULT_SFT_ADAPTER_PATH,
        help=f"Path to SFT adapter (default: {DEFAULT_SFT_ADAPTER_PATH})",
    )
    parser.add_argument(
        "--grpo-data-path",
        type=Path,
        default=DEFAULT_GRPO_DATA_PATH,
        help=f"Path to GRPO training data (default: {DEFAULT_GRPO_DATA_PATH})",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Build configuration
    config = PipelineConfig(
        sft_data_path=args.sft_data_path,
        sft_data_limit=args.limit,
        regenerate_sft_data=args.regenerate,
        sft_adapter_path=args.sft_adapter_path,
        retrain_sft=args.retrain_sft,
        grpo_data_path=args.grpo_data_path,
        max_steps=args.max_steps,
        grpo_config_path=args.config,
        dry_run=args.dry_run,
    )

    # Print plan if dry run
    if config.dry_run:
        print_plan(config)
        print("[DRY RUN] Execution plan printed above. Use without --dry-run to execute.")
        return

    start_time = time.time()

    print()
    print("=" * 70)
    print("GRPO TRAINING PIPELINE")
    print("=" * 70)
    print(f"Started: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print()

    results: list[PhaseResult] = []

    # Phase 1: SFT Data Generation
    result = phase_sft_data_generation(config)
    results.append(result)
    if not result.success:
        logger.error(f"Pipeline failed at Phase 1: {result.message}")
        sys.exit(1)

    # Phase 2: SFT Training
    result = phase_sft_training(config)
    results.append(result)
    if not result.success:
        logger.error(f"Pipeline failed at Phase 2: {result.message}")
        sys.exit(1)

    # Phase 3: GRPO Training
    result = phase_grpo_training(config)
    results.append(result)
    if not result.success:
        logger.error(f"Pipeline failed at Phase 3: {result.message}")
        sys.exit(1)

    adapter_path = Path(result.artifacts.get("adapter_path", "adapters/grpo_latest"))

    # Phase 4: Evaluation
    result = phase_evaluation(config, adapter_path)
    results.append(result)
    if not result.success:
        logger.error(f"Pipeline failed at Phase 4: {result.message}")
        sys.exit(1)

    evaluation_artifacts = result.artifacts

    # Phase 5: Promotion
    result = phase_promotion(config, adapter_path, evaluation_artifacts)
    results.append(result)

    # Print summary
    total_time = time.time() - start_time

    print()
    print("=" * 70)
    print("PIPELINE SUMMARY")
    print("=" * 70)
    print()
    print(f"{'Phase':<25} {'Status':<10} {'Duration':<15} {'Message'}")
    print("-" * 70)

    for r in results:
        status = "PASS" if r.success else "FAIL"
        duration = f"{r.duration_seconds:.1f}s"
        message = r.message[:40] + "..." if len(r.message) > 40 else r.message
        print(f"{r.phase:<25} {status:<10} {duration:<15} {message}")

    print("-" * 70)
    print(f"Total time: {total_time / 60:.1f} minutes")
    print()

    # Exit with appropriate code
    if not result.success:
        sys.exit(1)


if __name__ == "__main__":
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level="INFO",
    )

    try:
        main()
    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user (Ctrl+C)")
        sys.exit(0)
    except Exception:
        logger.exception("Fatal error in GRPO pipeline")
        sys.exit(1)
