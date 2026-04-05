#!/usr/bin/env python3
"""
End-to-end DPO training pipeline.

Takes a JSONL file of training examples (output of generate_training_dataset.py),
runs verification, reward computation, preference pair construction, and DPO training.

Usage:
    # Dry run (stops before model load)
    python run_dpo_training.py \\
        --dataset outputs/dataset/examples.jsonl \\
        --dry-run \\
        --save-pairs

    # Full training run
    python run_dpo_training.py \\
        --dataset outputs/dataset/examples.jsonl \\
        --output outputs/dpo_run_v1

    # Force (skip 24h promotion cooldown on first run)
    python run_dpo_training.py \\
        --dataset outputs/dataset/examples.jsonl \\
        --force
"""

import argparse
import asyncio
import json
import sys
import time
from collections import Counter
from datetime import datetime, UTC
from pathlib import Path

from loguru import logger

from swarm.training_capture import TrainingExample, load_examples_from_jsonl
from training.dpo_export import PreferencePair, construct_preference_pairs, export_to_jsonl
from training.reward_engine import ComputedReward, compute_reward
from verifier.engine import verify_batch
from verifier.outcome import VerifiedOutcome

# Windows async event loop policy for compatibility
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


# --------------------------------------------------------------------------- #
# Phase 1 – Load & filter
# --------------------------------------------------------------------------- #

def phase1_load(jsonl_path: Path) -> list[TrainingExample]:
    """Load examples from JSONL and filter to those with verifiable signals."""
    logger.info("Phase 1: Loading examples", path=str(jsonl_path))

    all_examples = load_examples_from_jsonl(jsonl_path)
    if not all_examples:
        logger.error("No examples loaded — check path and file format")
        sys.exit(1)

    # Filter: must have a direction in generator_signal (required for verification)
    verifiable = [
        ex for ex in all_examples
        if ex.generator_signal.get("direction") in ("HIGHER", "LOWER", "FLAT")
    ]
    dropped = len(all_examples) - len(verifiable)

    if dropped:
        logger.warning("Filtered examples without valid direction", dropped=dropped)

    # Log breakdown by symbol and timeframe
    by_symbol = Counter(ex.symbol for ex in verifiable)
    by_tf = Counter(ex.timeframe for ex in verifiable)
    logger.info(
        "Phase 1 complete",
        total_loaded=len(all_examples),
        verifiable=len(verifiable),
        symbols=dict(by_symbol),
        timeframes=dict(by_tf),
    )
    return verifiable


# --------------------------------------------------------------------------- #
# Phase 2 – Verify
# --------------------------------------------------------------------------- #

async def _run_verify(examples: list[TrainingExample]) -> list[VerifiedOutcome]:
    from data.market_data import MarketDataService
    async with MarketDataService() as svc:
        return await verify_batch(examples, svc)


def phase2_verify(
    examples: list[TrainingExample],
) -> list[tuple[TrainingExample, VerifiedOutcome]]:
    """Verify examples against realized market outcomes."""
    logger.info("Phase 2: Verifying examples", count=len(examples))

    outcomes = asyncio.run(_run_verify(examples))

    # Build outcome lookup by example_id
    outcome_by_id: dict[str, VerifiedOutcome] = {o.example_id: o for o in outcomes}

    matched: list[tuple[TrainingExample, VerifiedOutcome]] = []
    unmatched = 0
    for ex in examples:
        outcome = outcome_by_id.get(ex.example_id)
        if outcome is not None:
            matched.append((ex, outcome))
        else:
            unmatched += 1

    success_rate = len(matched) / len(examples) if examples else 0.0
    logger.info(
        "Phase 2 complete",
        verified=len(matched),
        unmatched=unmatched,
        success_rate=f"{success_rate:.1%}",
    )
    return matched


# --------------------------------------------------------------------------- #
# Phase 3 – Reward
# --------------------------------------------------------------------------- #

def phase3_reward(
    matched: list[tuple[TrainingExample, VerifiedOutcome]],
) -> list[tuple[TrainingExample, VerifiedOutcome, ComputedReward]]:
    """Compute rewards for all verified examples."""
    logger.info("Phase 3: Computing rewards", count=len(matched))

    result = []
    for example, outcome in matched:
        reward = compute_reward(outcome, example)
        result.append((example, outcome, reward))

    rewards = [r.final_reward for _, _, r in result]
    if rewards:
        mean_r = sum(rewards) / len(rewards)
        pct_pos = sum(1 for r in rewards if r > 0) / len(rewards)
        logger.info(
            "Phase 3 complete",
            total=len(result),
            mean_reward=f"{mean_r:.3f}",
            pct_positive=f"{pct_pos:.1%}",
        )
    return result


# --------------------------------------------------------------------------- #
# Phase 4 – Preference pairs
# --------------------------------------------------------------------------- #

def phase4_pairs(
    examples_with_rewards: list[tuple[TrainingExample, VerifiedOutcome, ComputedReward]],
    min_delta: float,
    save_pairs: bool,
    output_dir: Path,
) -> list[PreferencePair]:
    """Construct DPO preference pairs and optionally save to JSONL."""
    logger.info("Phase 4: Constructing preference pairs", min_delta=min_delta)

    pairs = construct_preference_pairs(
        examples_with_rewards,
        min_delta=min_delta,
        min_personas_per_context=3,
    )

    if not pairs:
        logger.warning("No preference pairs constructed — check min_delta and context grouping")

    if save_pairs and pairs:
        pairs_path = output_dir / "preference_pairs.jsonl"
        export_to_jsonl(pairs, str(pairs_path))
        logger.info("Preference pairs saved", path=str(pairs_path))

    logger.info("Phase 4 complete", pairs=len(pairs))
    return pairs


# --------------------------------------------------------------------------- #
# Phase 5 – Train
# --------------------------------------------------------------------------- #

def phase5_train(pairs: list[PreferencePair], force: bool) -> None:
    """Run DPO training and exit non-zero on failure."""
    logger.info("Phase 5: Starting DPO training", pairs=len(pairs))

    # Training deps guard — informative error if not installed
    try:
        from training.dpo_trainer import train_dpo
    except ImportError as e:
        logger.error(
            "Training dependencies not installed. Run: pip install -r requirements-training.txt",
            error=str(e),
        )
        sys.exit(1)

    result = train_dpo(pairs, force_training=force)

    # Print result summary
    result_dict = result.to_dict()
    logger.info("Training result", **result_dict)
    print(json.dumps(result_dict, indent=2))

    if not result.success:
        logger.error("DPO training failed", error=result.error)
        sys.exit(1)

    logger.info("DPO training succeeded", adapter=str(result.adapter_path))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="End-to-end DPO training pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run — verify + reward + pairs, no GPU
  python run_dpo_training.py --dataset outputs/dataset/examples.jsonl --dry-run --save-pairs

  # Full run
  python run_dpo_training.py --dataset outputs/dataset/examples.jsonl

  # Skip promotion cooldown (first run)
  python run_dpo_training.py --dataset outputs/dataset/examples.jsonl --force
        """,
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to examples.jsonl produced by generate_training_dataset.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory for pairs + logs (default: outputs/dpo_run_<timestamp>)",
    )
    parser.add_argument(
        "--save-pairs",
        action="store_true",
        help="Write preference_pairs.jsonl before training",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Stop after Phase 4 (preference pair construction), skip model load",
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=0.2,
        help="Minimum reward delta for preference pairs (default: 0.2)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip 24h promotion cooldown check",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve output directory
    if args.output is None:
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"outputs/dpo_run_{timestamp}")
    else:
        output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    logger.info("=" * 60)
    logger.info("STARTING DPO TRAINING PIPELINE")
    logger.info("=" * 60)
    logger.info(
        "Configuration",
        dataset=str(args.dataset),
        output_dir=str(output_dir),
        min_delta=args.min_delta,
        dry_run=args.dry_run,
        save_pairs=args.save_pairs,
        force=args.force,
    )

    # Phase 1: Load
    examples = phase1_load(args.dataset)

    # Phase 2: Verify
    matched = phase2_verify(examples)
    if not matched:
        logger.error("No examples verified — cannot build preference pairs")
        sys.exit(1)

    # Phase 3: Reward
    examples_with_rewards = phase3_reward(matched)

    # Phase 4: Preference pairs
    pairs = phase4_pairs(
        examples_with_rewards,
        min_delta=args.min_delta,
        save_pairs=args.save_pairs,
        output_dir=output_dir,
    )

    if args.dry_run:
        elapsed = time.time() - start_time
        logger.info(
            "Dry run complete — skipping training",
            preference_pairs=len(pairs),
            elapsed_seconds=f"{elapsed:.1f}",
        )
        return

    if not pairs:
        logger.error("No preference pairs — cannot train")
        sys.exit(1)

    # Phase 5: Train
    phase5_train(pairs, force=args.force)

    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info("DPO PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info("Total time", elapsed_minutes=f"{elapsed / 60:.1f}")


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
        logger.exception("Fatal error in DPO pipeline")
        sys.exit(1)
