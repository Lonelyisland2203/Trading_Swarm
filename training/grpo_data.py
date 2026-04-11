"""
GRPO training data types and walk-forward split functions.

Provides GRPOTrainingExample for representing training inputs and
temporal split utilities for walk-forward validation.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from loguru import logger


class TemporalSplitError(Exception):
    """Raised when temporal split validation fails."""

    pass


@dataclass(frozen=True, slots=True)
class GRPOTrainingExample:
    """
    Single training example for GRPO.

    Attributes:
        market_snapshot: Input prompt with market data and indicators
        actual_direction: Ground truth direction ("LONG", "SHORT", "FLAT")
        gross_return_pct: Ground truth return for reward computation
        timestamp_ms: Timestamp for temporal ordering (milliseconds)
    """

    market_snapshot: str
    actual_direction: str
    gross_return_pct: float
    timestamp_ms: int


@dataclass(frozen=True)
class GRPOWalkForwardSplit:
    """Walk-forward train/test split for GRPO training."""

    train_examples: list[GRPOTrainingExample]
    test_examples: list[GRPOTrainingExample]
    replay_examples: list[GRPOTrainingExample]
    train_start_ms: int
    train_end_ms: int
    test_start_ms: int
    test_end_ms: int


def create_grpo_walk_forward_split(
    examples: list[GRPOTrainingExample],
    train_window: int = 500,
    test_window: int = 100,
    replay_ratio: float = 0.15,
    replay_buffer_size: int = 1000,
) -> GRPOWalkForwardSplit:
    """
    Create walk-forward train/test split for GRPO examples.

    Temporal ordering:
        [-------- history --------][---- train ----][-- test --]
                                   ^                ^
                                   train_start      test_start

    Args:
        examples: List of GRPO training examples (will be sorted by timestamp)
        train_window: Number of most recent examples for training (default: 500)
        test_window: Number of holdout examples for testing (default: 100)
        replay_ratio: Fraction of training window to sample from history (default: 0.15)
        replay_buffer_size: Maximum historical examples to sample from (default: 1000)

    Returns:
        GRPOWalkForwardSplit with train/test/replay examples and timestamps

    Raises:
        TemporalSplitError: If insufficient examples or temporal ordering violated
    """
    if not examples:
        raise TemporalSplitError("No examples provided")

    # Sort by timestamp (defensive - should already be sorted)
    sorted_examples = sorted(examples, key=lambda e: e.timestamp_ms)

    total_examples = len(sorted_examples)
    required_examples = train_window + test_window

    if total_examples < required_examples:
        raise TemporalSplitError(
            f"Insufficient examples: have {total_examples}, "
            f"need {required_examples} ({train_window} train + {test_window} test)"
        )

    # Split: most recent test_window for testing, previous train_window for training
    test_examples = sorted_examples[-test_window:]
    train_examples = sorted_examples[-(test_window + train_window) : -test_window]

    # Handle timestamp boundary collision (same timestamp in train and test)
    if (
        train_examples
        and test_examples
        and train_examples[-1].timestamp_ms == test_examples[0].timestamp_ms
    ):
        boundary_ts = train_examples[-1].timestamp_ms
        extra = [e for e in test_examples if e.timestamp_ms == boundary_ts]
        test_examples = [e for e in test_examples if e.timestamp_ms != boundary_ts]
        train_examples = list(train_examples) + extra

        if not test_examples:
            raise TemporalSplitError(
                "All test examples had the same timestamp as training boundary"
            )

    # Historical examples (everything before training window)
    train_ids = {id(e) for e in train_examples} | {id(e) for e in test_examples}
    history_examples = [e for e in sorted_examples if id(e) not in train_ids]

    # Sample replay buffer from history
    replay_examples: list[GRPOTrainingExample] = []
    if history_examples and replay_ratio > 0:
        recent_history = history_examples[-replay_buffer_size:]
        num_replay = int(train_window * replay_ratio)
        num_replay = min(num_replay, len(recent_history))

        if num_replay > 0:
            replay_examples = random.sample(recent_history, num_replay)
            replay_examples = sorted(replay_examples, key=lambda e: e.timestamp_ms)

    # Validate replay examples are all before training
    if replay_examples:
        train_start_ms = train_examples[0].timestamp_ms
        replay_examples = [e for e in replay_examples if e.timestamp_ms < train_start_ms]

    # Extract timestamps
    train_start_ms = train_examples[0].timestamp_ms
    train_end_ms = train_examples[-1].timestamp_ms
    test_start_ms = test_examples[0].timestamp_ms
    test_end_ms = test_examples[-1].timestamp_ms

    # Validate temporal ordering
    if test_start_ms <= train_end_ms:
        raise TemporalSplitError(
            f"Test data overlaps with training data: "
            f"train ends at {train_end_ms}, test starts at {test_start_ms}"
        )

    logger.info(
        "GRPO walk-forward split created",
        train_examples=len(train_examples),
        test_examples=len(test_examples),
        replay_examples=len(replay_examples),
        train_start_ms=train_start_ms,
        test_start_ms=test_start_ms,
    )

    return GRPOWalkForwardSplit(
        train_examples=train_examples,
        test_examples=test_examples,
        replay_examples=replay_examples,
        train_start_ms=train_start_ms,
        train_end_ms=train_end_ms,
        test_start_ms=test_start_ms,
        test_end_ms=test_end_ms,
    )
