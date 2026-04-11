"""
Walk-forward validation for DPO training.

CRITICAL: Training must NEVER use future data. This module ensures strict
temporal splits with replay buffer sampling to prevent catastrophic forgetting.
"""

import random
from dataclasses import dataclass
from typing import List

from loguru import logger

from training.dpo_export import PreferencePair


@dataclass(frozen=True)
class WalkForwardSplit:
    """Walk-forward train/test split with replay buffer."""

    train_pairs: List[PreferencePair]
    test_pairs: List[PreferencePair]
    replay_pairs: List[PreferencePair]
    train_start_ms: int
    train_end_ms: int
    test_start_ms: int
    test_end_ms: int
    total_pairs: int


class TemporalSplitError(Exception):
    """Raised when temporal split validation fails."""

    pass


def create_walk_forward_splits(
    preference_pairs: List[PreferencePair],
    train_window: int = 500,
    test_window: int = 100,
    replay_ratio: float = 0.15,
    replay_buffer_size: int = 1000,
    min_training_pairs: int = 500,
) -> WalkForwardSplit:
    """
    Create walk-forward train/test split with replay buffer.

    Temporal ordering:
        [-------- history --------][---- train ----][-- test --]
                                   ^                ^
                                   train_start      test_start

    Args:
        preference_pairs: List of DPO preference pairs (must be sorted by timestamp)
        train_window: Number of most recent pairs for training (default: 500)
        test_window: Number of holdout pairs for testing (default: 100)
        replay_ratio: Fraction of training set to sample from history (default: 0.15)
        replay_buffer_size: Maximum historical pairs to sample from (default: 1000)
        min_training_pairs: Minimum pairs required in training set (default: 500)

    Returns:
        WalkForwardSplit with train/test/replay pairs and timestamps

    Raises:
        TemporalSplitError: If insufficient pairs or temporal ordering violated

    Example:
        >>> pairs = [...] # 700 preference pairs sorted by timestamp
        >>> split = create_walk_forward_splits(
        ...     pairs,
        ...     train_window=500,
        ...     test_window=100,
        ...     replay_ratio=0.15,
        ... )
        >>> len(split.train_pairs)  # 500 pairs
        500
        >>> len(split.test_pairs)  # 100 pairs
        100
        >>> len(split.replay_pairs)  # 75 pairs (15% of 500)
        75
    """
    if not preference_pairs:
        raise TemporalSplitError("No preference pairs provided")

    # Sort by timestamp (defensive - should already be sorted)
    sorted_pairs = sorted(preference_pairs, key=lambda p: p.timestamp_ms)

    total_pairs = len(sorted_pairs)
    required_pairs = train_window + test_window

    if total_pairs < required_pairs:
        raise TemporalSplitError(
            f"Insufficient pairs: have {total_pairs}, "
            f"need {required_pairs} ({train_window} train + {test_window} test)"
        )

    # Split: most recent test_window pairs for testing, previous train_window for training
    test_pairs = sorted_pairs[-test_window:]
    train_pairs = sorted_pairs[-(test_window + train_window) : -test_window]

    # Ensure no timestamp straddles the train/test boundary.
    # If train[-1] and test[0] share the same timestamp, absorb all pairs at that
    # timestamp into train so the boundary is strictly clean.
    if train_pairs and test_pairs and train_pairs[-1].timestamp_ms == test_pairs[0].timestamp_ms:
        boundary_ts = train_pairs[-1].timestamp_ms
        # Move pairs with boundary_ts from test → train
        extra = [p for p in test_pairs if p.timestamp_ms == boundary_ts]
        test_pairs = [p for p in test_pairs if p.timestamp_ms != boundary_ts]
        train_pairs = list(train_pairs) + extra

    # Historical pairs (everything before training window)
    train_ids = {id(p) for p in train_pairs} | {id(p) for p in test_pairs}
    history_pairs = [p for p in sorted_pairs if id(p) not in train_ids]

    # Sample replay buffer from history
    if history_pairs:
        # Limit to replay_buffer_size most recent historical pairs
        recent_history = history_pairs[-replay_buffer_size:]

        # Sample replay_ratio * train_window pairs
        num_replay = int(train_window * replay_ratio)
        num_replay = min(num_replay, len(recent_history))  # Can't sample more than we have

        if num_replay > 0:
            replay_pairs = random.sample(recent_history, num_replay)
            # Sort replay pairs by timestamp
            replay_pairs = sorted(replay_pairs, key=lambda p: p.timestamp_ms)
        else:
            replay_pairs = []
    else:
        replay_pairs = []

    # Validate temporal ordering
    train_start_ms = train_pairs[0].timestamp_ms
    train_end_ms = train_pairs[-1].timestamp_ms
    test_start_ms = test_pairs[0].timestamp_ms
    test_end_ms = test_pairs[-1].timestamp_ms

    if test_start_ms <= train_end_ms:
        raise TemporalSplitError(
            f"Test data overlaps with training data: "
            f"train ends at {train_end_ms}, test starts at {test_start_ms}"
        )

    # Validate replay pairs are all before training.
    # If replay[-1] shares a timestamp with train[0], drop those replay pairs
    # to ensure a clean boundary (same fix as the train/test boundary above).
    if replay_pairs:
        train_start_ms = train_pairs[0].timestamp_ms
        if replay_pairs[-1].timestamp_ms >= train_start_ms:
            replay_pairs = [p for p in replay_pairs if p.timestamp_ms < train_start_ms]
        if replay_pairs and replay_pairs[-1].timestamp_ms >= train_start_ms:
            raise TemporalSplitError(
                f"Replay data overlaps with training data: "
                f"replay ends at {replay_pairs[-1].timestamp_ms}, train starts at {train_start_ms}"
            )

    # Check minimum training set size (train + replay)
    total_training = len(train_pairs) + len(replay_pairs)
    if total_training < min_training_pairs:
        raise TemporalSplitError(
            f"Insufficient training pairs: have {total_training}, "
            f"need {min_training_pairs} minimum"
        )

    logger.info(
        "Walk-forward split created",
        train_pairs=len(train_pairs),
        test_pairs=len(test_pairs),
        replay_pairs=len(replay_pairs),
        total_training=total_training,
        train_start_ms=train_start_ms,
        test_start_ms=test_start_ms,
    )

    return WalkForwardSplit(
        train_pairs=train_pairs,
        test_pairs=test_pairs,
        replay_pairs=replay_pairs,
        train_start_ms=train_start_ms,
        train_end_ms=train_end_ms,
        test_start_ms=test_start_ms,
        test_end_ms=test_end_ms,
        total_pairs=total_training,
    )


def validate_temporal_split(split: WalkForwardSplit) -> None:
    """
    Validate temporal ordering of walk-forward split.

    Ensures:
    1. Training pairs all before test pairs
    2. Replay pairs all before training pairs
    3. No timestamp overlap between sets

    Args:
        split: WalkForwardSplit to validate

    Raises:
        TemporalSplitError: If temporal ordering is violated
    """
    # Check training before test
    if split.test_start_ms <= split.train_end_ms:
        raise TemporalSplitError(
            f"Test data starts ({split.test_start_ms}) before training ends ({split.train_end_ms})"
        )

    # Check all training pairs within train window
    for pair in split.train_pairs:
        if pair.timestamp_ms < split.train_start_ms or pair.timestamp_ms > split.train_end_ms:
            raise TemporalSplitError(
                f"Training pair timestamp {pair.timestamp_ms} outside train window "
                f"[{split.train_start_ms}, {split.train_end_ms}]"
            )

    # Check all test pairs within test window
    for pair in split.test_pairs:
        if pair.timestamp_ms < split.test_start_ms or pair.timestamp_ms > split.test_end_ms:
            raise TemporalSplitError(
                f"Test pair timestamp {pair.timestamp_ms} outside test window "
                f"[{split.test_start_ms}, {split.test_end_ms}]"
            )

    # Check all replay pairs before training
    if split.replay_pairs:
        for pair in split.replay_pairs:
            if pair.timestamp_ms >= split.train_start_ms:
                raise TemporalSplitError(
                    f"Replay pair timestamp {pair.timestamp_ms} not before training start {split.train_start_ms}"
                )

    logger.debug("Temporal split validation passed", total_pairs=split.total_pairs)


def merge_train_and_replay(split: WalkForwardSplit, shuffle: bool = True) -> List[PreferencePair]:
    """
    Merge training and replay pairs for DPO training.

    Args:
        split: WalkForwardSplit with train and replay pairs
        shuffle: Whether to shuffle merged pairs (default: True)

    Returns:
        List of preference pairs (train + replay), optionally shuffled

    Example:
        >>> split = create_walk_forward_splits(pairs)
        >>> training_data = merge_train_and_replay(split, shuffle=True)
        >>> len(training_data)  # 500 train + 75 replay = 575
        575
    """
    merged = split.train_pairs + split.replay_pairs

    if shuffle:
        # Shuffle to mix replay and new data
        random.shuffle(merged)
        logger.debug("Training data shuffled", total_pairs=len(merged))
    else:
        # Keep temporal ordering
        merged = sorted(merged, key=lambda p: p.timestamp_ms)
        logger.debug("Training data kept in temporal order", total_pairs=len(merged))

    return merged
