#!/usr/bin/env python3
"""
Evaluate a candidate DPO adapter on the held-out test set and promote if criteria are met.

Fast path (recommended): pass --test-data pointing to the test_eval_data.jsonl saved
by run_dpo_training.py during the training run. This skips expensive market data re-fetching.

Slow path (fallback): omit --test-data and provide --dataset instead. Re-runs phases 1-4
on the full dataset (all market data must be re-fetched from exchange).

Usage (fast):
    python evaluate_candidate.py \\
        --test-data outputs/dpo_run_20260409_180000/test_eval_data.jsonl \\
        --candidate models/adapters/qwen3-8b-dpo/adapter-DPO-1775725178964.candidate

Usage (slow fallback):
    python evaluate_candidate.py \\
        --dataset outputs/combined_examples.jsonl \\
        --candidate models/adapters/qwen3-8b-dpo/adapter-DPO-1775725178964.candidate

Promotion criteria (no baseline - first run):
    - IC >= 0.05 (min_oos_ic)
    - Brier score <= 0.25 (max_brier_score)
    - p-value < 0.05
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from loguru import logger

from config.fee_model import FeeModelSettings
from config.settings import settings
from swarm.training_capture import TrainingExample, load_examples_from_jsonl
from training.dpo_export import construct_preference_pairs
from training.dpo_eval import evaluate_adapter
from training.dpo_trainer import promote_adapter
from training.reward_engine import compute_reward, ComputedReward
from training.walk_forward import (
    create_walk_forward_splits,
    validate_temporal_split,
)
from verifier.engine import verify_batch
from verifier.outcome import VerifiedOutcome

# Windows async event loop policy for compatibility
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


def _extract_direction(sig: dict) -> str | None:
    """Extract direction from signal, handling top-level or signal_data nesting."""
    return sig.get("direction") or sig.get("signal_data", {}).get("direction")


# --------------------------------------------------------------------------- #
# Fast path: load pre-saved test eval data
# --------------------------------------------------------------------------- #


def load_test_eval_data(
    test_data_path: Path,
) -> tuple[list[TrainingExample], list[VerifiedOutcome], list[ComputedReward]]:
    """
    Load pre-saved test evaluation data from a training run.

    Reconstructs lightweight TrainingExample, VerifiedOutcome, and ComputedReward
    objects from the compact JSONL format saved by run_dpo_training.py.

    Args:
        test_data_path: Path to test_eval_data.jsonl

    Returns:
        Tuple of (examples, outcomes, rewards)
    """
    examples: list[TrainingExample] = []
    outcomes: list[VerifiedOutcome] = []
    rewards: list[ComputedReward] = []

    with open(test_data_path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning("Skipping malformed line", line=line_num, error=str(e))
                continue

            # Reconstruct TrainingExample (only fields needed by evaluate_adapter)
            ex = TrainingExample(
                example_id=rec["example_id"],
                context_id=rec.get("context_id", ""),
                symbol=rec.get("symbol", ""),
                timeframe=rec.get("timeframe", ""),
                timestamp_ms=rec.get("timestamp_ms", 0),
                market_regime=rec.get("market_regime", ""),
                persona=rec.get("persona", ""),
                task_prompt=rec.get("task_prompt", ""),
                generator_signal=rec.get("generator_signal", {}),
            )

            # Reconstruct VerifiedOutcome
            outcome = VerifiedOutcome(
                example_id=rec["example_id"],
                actual_direction=rec["actual_direction"],
                realized_return=rec["realized_return"],
                max_adverse_excursion=rec.get("max_adverse_excursion"),
                net_return=rec["net_return"],
                entry_price=rec.get("entry_price", 0.0),
                exit_price=rec.get("exit_price", 0.0),
                bars_held=0,
            )

            # Reconstruct ComputedReward (minimal — only final_reward used in diagnostics)
            from datetime import datetime, UTC

            reward = ComputedReward(
                final_reward=rec["final_reward"],
                return_reward=rec.get("return_reward", 0.0),
                directional_reward=rec.get("directional_reward", 0.0),
                mae_reward=rec.get("mae_reward", 0.0),
                return_weight=settings.reward.return_weight,
                directional_weight=settings.reward.directional_weight,
                mae_weight=settings.reward.mae_weight,
                return_scale=1.0,
                mae_scale=1.0,
                net_return=rec.get("net_return", 0.0),
                realized_return=rec.get("realized_return", 0.0),
                mae=rec.get("max_adverse_excursion"),
                predicted_direction="UNKNOWN",
                actual_direction=rec["actual_direction"],
                confidence=0.5,
                components_used=3,
                computation_timestamp=datetime.now(UTC).isoformat(),
                market_regime=rec.get("market_regime", ""),
            )

            examples.append(ex)
            outcomes.append(outcome)
            rewards.append(reward)

    logger.info(
        "Test eval data loaded",
        path=str(test_data_path),
        examples=len(examples),
    )
    return examples, outcomes, rewards


# --------------------------------------------------------------------------- #
# Slow path: re-run pipeline phases 1-4
# --------------------------------------------------------------------------- #


def phase1_load(jsonl_path: Path) -> list[TrainingExample]:
    logger.info("Phase 1: Loading examples", path=str(jsonl_path))
    all_examples = load_examples_from_jsonl(jsonl_path)
    if not all_examples:
        logger.error("No examples loaded")
        sys.exit(1)

    verifiable = [
        ex
        for ex in all_examples
        if _extract_direction(ex.generator_signal) in ("HIGHER", "LOWER", "FLAT")
    ]
    logger.info(
        "Phase 1 complete",
        loaded=len(all_examples),
        verifiable=len(verifiable),
    )
    return verifiable


async def _run_verify(
    examples: list[TrainingExample], fee_model: FeeModelSettings
) -> list[VerifiedOutcome]:
    from data.market_data import MarketDataService

    async with MarketDataService() as svc:
        return await verify_batch(examples, svc, fee_model=fee_model)


def phase2_verify(
    examples: list[TrainingExample],
    fee_model: FeeModelSettings = FeeModelSettings(),
) -> list[tuple[TrainingExample, VerifiedOutcome]]:
    logger.info("Phase 2: Verifying examples (re-fetches market data)", count=len(examples))
    outcomes = asyncio.run(_run_verify(examples, fee_model))
    outcome_by_id = {o.example_id: o for o in outcomes}
    matched = [
        (ex, outcome_by_id[ex.example_id]) for ex in examples if ex.example_id in outcome_by_id
    ]
    logger.info(
        "Phase 2 complete",
        verified=len(matched),
        total=len(examples),
        success_rate=f"{len(matched) / len(examples):.1%}",
    )
    return matched


def phase3_reward(matched: list[tuple[TrainingExample, VerifiedOutcome]]):
    logger.info("Phase 3: Computing rewards", count=len(matched))
    result = [(ex, outcome, compute_reward(outcome, ex)) for ex, outcome in matched]
    logger.info("Phase 3 complete", total=len(result))
    return result


def phase4_pairs(examples_with_rewards, min_delta: float = 0.2):
    logger.info("Phase 4: Constructing preference pairs")
    pairs = construct_preference_pairs(
        examples_with_rewards,
        min_delta=min_delta,
        min_personas_per_context=3,
    )
    logger.info("Phase 4 complete", pairs=len(pairs))
    return pairs


def run_slow_path(
    dataset_path: Path, min_delta: float
) -> tuple[list[TrainingExample], list[VerifiedOutcome], list[ComputedReward]]:
    """Re-run pipeline to reconstruct test set (slow — re-fetches all market data)."""
    examples = phase1_load(dataset_path)
    matched = phase2_verify(examples)
    examples_with_rewards = phase3_reward(matched)
    pairs = phase4_pairs(examples_with_rewards, min_delta=min_delta)

    if not pairs:
        logger.error("No preference pairs constructed — cannot evaluate")
        sys.exit(1)

    # Reconstruct walk-forward split
    split = create_walk_forward_splits(
        pairs,
        train_window=settings.dpo.train_window,
        test_window=settings.dpo.test_window,
        replay_ratio=settings.dpo.replay_ratio,
        replay_buffer_size=settings.dpo.replay_buffer_size,
        min_training_pairs=settings.dpo.min_training_pairs,
    )
    validate_temporal_split(split)

    logger.info(
        "Walk-forward split reconstructed",
        train_pairs=len(split.train_pairs),
        test_pairs=len(split.test_pairs),
        replay_pairs=len(split.replay_pairs),
    )

    # Build lookup maps: example_id -> data
    example_map = {ex.example_id: ex for ex, _, _ in examples_with_rewards}
    outcome_map = {ex.example_id: outcome for ex, outcome, _ in examples_with_rewards}
    reward_map = {ex.example_id: reward for ex, _, reward in examples_with_rewards}

    # Extract test examples from test pairs (chosen + rejected)
    test_examples: list[TrainingExample] = []
    test_outcomes: list[VerifiedOutcome] = []
    test_rewards: list[ComputedReward] = []
    seen_ids: set[str] = set()

    for pair in split.test_pairs:
        for example_id in (pair.chosen_example_id, pair.rejected_example_id):
            if example_id in seen_ids or example_id not in example_map:
                continue
            seen_ids.add(example_id)
            test_examples.append(example_map[example_id])
            test_outcomes.append(outcome_map[example_id])
            test_rewards.append(reward_map[example_id])

    logger.info(
        "Test set extracted",
        test_pairs=len(split.test_pairs),
        test_examples=len(test_examples),
    )

    return test_examples, test_outcomes, test_rewards


# --------------------------------------------------------------------------- #
# Main evaluation
# --------------------------------------------------------------------------- #


def run_evaluation(
    candidate_path: Path,
    test_examples: list[TrainingExample],
    test_outcomes: list[VerifiedOutcome],
    test_rewards: list[ComputedReward],
) -> None:
    if len(test_examples) < 30:
        logger.error("Test set too small for evaluation", count=len(test_examples))
        sys.exit(1)

    # Run evaluation
    logger.info("Running adapter evaluation", test_examples=len(test_examples))
    evaluation = evaluate_adapter(test_examples, test_outcomes, test_rewards)

    # Print evaluation report
    print()
    print("=" * 55)
    print("ADAPTER EVALUATION RESULTS")
    print("=" * 55)
    print(f"Candidate:      {candidate_path.name}")
    print(f"Test examples:  {evaluation.num_examples}")
    print()
    print(f"IC (Spearman):  {evaluation.ic:+.4f}  (p = {evaluation.ic_pvalue:.4f})")
    print(f"Return-wtd IC:  {evaluation.return_weighted_ic:+.4f}")
    print(f"Brier score:    {evaluation.brier_score:.4f}  (lower is better)")
    print(f"MACE:           {evaluation.mean_abs_calibration_error:.4f}")
    print(f"Mean reward:    {evaluation.mean_reward:+.4f}")
    print(f"Std reward:     {evaluation.std_reward:.4f}")
    print()
    if evaluation.ic_by_regime:
        print("IC by regime:")
        for regime, ic_val in sorted(evaluation.ic_by_regime.items()):
            print(f"  {regime:<22} {ic_val:+.4f}")
        print()

    # Promotion decision
    print("=" * 55)
    print("PROMOTION DECISION (absolute criteria, no baseline)")
    print("=" * 55)
    ic_pass = evaluation.ic >= settings.dpo.min_oos_ic
    brier_pass = evaluation.brier_score <= settings.dpo.max_brier_score
    pval_pass = evaluation.ic_pvalue < 0.05
    print(
        f"  IC >= {settings.dpo.min_oos_ic:.4f}:      {'PASS' if ic_pass else 'FAIL'}  ({evaluation.ic:+.4f})"
    )
    print(
        f"  Brier <= {settings.dpo.max_brier_score:.4f}:   {'PASS' if brier_pass else 'FAIL'}  ({evaluation.brier_score:.4f})"
    )
    print(f"  p < 0.05:         {'PASS' if pval_pass else 'FAIL'}  ({evaluation.ic_pvalue:.4f})")
    print()

    promoted, reason, promoted_path = promote_adapter(
        candidate_path=candidate_path,
        evaluation=evaluation,
        baseline_eval=None,  # First DPO run — use absolute criteria
    )

    if promoted:
        print("RESULT: PROMOTED")
        print(f"Reason: {reason}")
        print(f"Path:   {promoted_path}")
    else:
        print("RESULT: NOT PROMOTED")
        print(f"Reason: {reason}")
        print(f"Candidate retained at: {candidate_path}")

    print()

    # Save evaluation JSON next to adapter
    eval_output = candidate_path / "evaluation.json"
    eval_data = {
        "ic": evaluation.ic,
        "ic_pvalue": evaluation.ic_pvalue,
        "return_weighted_ic": evaluation.return_weighted_ic,
        "brier_score": evaluation.brier_score,
        "mean_abs_calibration_error": evaluation.mean_abs_calibration_error,
        "ic_by_regime": evaluation.ic_by_regime,
        "num_examples": evaluation.num_examples,
        "mean_reward": evaluation.mean_reward,
        "std_reward": evaluation.std_reward,
        "promoted": promoted,
        "promotion_reason": reason,
        "promoted_path": str(promoted_path) if promoted_path else None,
    }
    with open(eval_output, "w") as f:
        json.dump(eval_data, f, indent=2)
    logger.info("Evaluation results saved", path=str(eval_output))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate and promote a candidate DPO adapter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Fast path (recommended — no re-fetching market data):
  python evaluate_candidate.py \\
      --test-data outputs/dpo_run_20260409_180000/test_eval_data.jsonl \\
      --candidate models/adapters/qwen3-8b-dpo/adapter-DPO-1775725178964.candidate

Slow fallback (re-runs full verification, ~1 hour):
  python evaluate_candidate.py \\
      --dataset outputs/combined_examples.jsonl \\
      --candidate models/adapters/qwen3-8b-dpo/adapter-DPO-1775725178964.candidate
        """,
    )
    parser.add_argument(
        "--candidate",
        type=Path,
        required=True,
        help="Candidate adapter directory (*.candidate)",
    )
    # Fast path
    parser.add_argument(
        "--test-data",
        type=Path,
        default=None,
        help="Pre-saved test eval data from the training run (fast path, skips re-verification)",
    )
    # Slow fallback
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Combined examples JSONL (slow path — re-runs full verification)",
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=0.2,
        help="Minimum reward delta for preference pairs (must match training run, default: 0.2)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.candidate.exists():
        logger.error("Candidate adapter not found", path=str(args.candidate))
        sys.exit(1)

    if args.test_data is None and args.dataset is None:
        logger.error("Must provide either --test-data (fast) or --dataset (slow fallback)")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("CANDIDATE ADAPTER EVALUATION")
    logger.info("=" * 60)
    logger.info("Candidate", path=str(args.candidate))

    if args.test_data is not None:
        # Fast path: load pre-saved test data
        if not args.test_data.exists():
            logger.error("Test data file not found", path=str(args.test_data))
            sys.exit(1)
        logger.info("Using pre-saved test data (fast path)", path=str(args.test_data))
        test_examples, test_outcomes, test_rewards = load_test_eval_data(args.test_data)
    else:
        # Slow path: re-run full verification pipeline
        logger.warning(
            "No --test-data provided. Re-running full verification (slow — may take 1+ hours)."
        )
        logger.warning(
            "For future runs, save test_eval_data.jsonl during training and use --test-data."
        )
        test_examples, test_outcomes, test_rewards = run_slow_path(args.dataset, args.min_delta)

    run_evaluation(args.candidate, test_examples, test_outcomes, test_rewards)


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
        logger.warning("Evaluation interrupted by user (Ctrl+C)")
        sys.exit(0)
    except Exception:
        logger.exception("Fatal error during evaluation")
        sys.exit(1)
