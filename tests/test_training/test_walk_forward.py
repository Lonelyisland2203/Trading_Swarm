"""Tests for walk-forward validation."""

import pytest

from training.dpo_export import PreferencePair
from training.walk_forward import (
    TemporalSplitError,
    create_walk_forward_splits,
    merge_train_and_replay,
    validate_temporal_split,
)


@pytest.fixture
def sample_preference_pairs():
    """Create sample preference pairs with sequential timestamps."""

    def _make_pairs(num_pairs: int, start_timestamp_ms: int = 1000000000):
        """Create num_pairs with sequential timestamps (1 hour apart)."""
        pairs = []
        for i in range(num_pairs):
            timestamp_ms = start_timestamp_ms + (i * 3600000)  # +1 hour each
            pair = PreferencePair(
                prompt=f"Prompt {i}",
                context_id=f"ctx-{i}",
                chosen_reasoning=f"Chosen {i}",
                chosen_direction="HIGHER",
                chosen_confidence=0.8,
                chosen_reward=0.8,
                chosen_example_id=f"ex-chosen-{i}",
                chosen_persona="MOMENTUM",
                rejected_reasoning=f"Rejected {i}",
                rejected_direction="LOWER",
                rejected_confidence=0.6,
                rejected_reward=0.3,
                rejected_example_id=f"ex-rejected-{i}",
                rejected_persona="CONTRARIAN",
                reward_delta=0.5,
                symbol="BTC/USDT",
                timestamp_ms=timestamp_ms,
                market_regime="NEUTRAL",
            )
            pairs.append(pair)
        return pairs

    return _make_pairs


class TestCreateWalkForwardSplits:
    """Test walk-forward split creation."""

    def test_basic_split(self, sample_preference_pairs):
        """Test basic train/test split without replay."""
        pairs = sample_preference_pairs(600)

        split = create_walk_forward_splits(
            pairs,
            train_window=500,
            test_window=100,
            replay_ratio=0.0,  # No replay
        )

        assert len(split.train_pairs) == 500
        assert len(split.test_pairs) == 100
        assert len(split.replay_pairs) == 0
        assert split.total_pairs == 500

    def test_split_with_replay(self, sample_preference_pairs):
        """Test split with replay buffer."""
        # 700 pairs total: 100 history, 500 train, 100 test
        pairs = sample_preference_pairs(700)

        split = create_walk_forward_splits(
            pairs,
            train_window=500,
            test_window=100,
            replay_ratio=0.15,  # 15% replay = 75 pairs
        )

        assert len(split.train_pairs) == 500
        assert len(split.test_pairs) == 100
        assert len(split.replay_pairs) == 75  # 15% of 500
        assert split.total_pairs == 575  # 500 + 75

    def test_insufficient_pairs_raises(self, sample_preference_pairs):
        """Test error when insufficient pairs."""
        pairs = sample_preference_pairs(50)  # Not enough

        with pytest.raises(TemporalSplitError, match="Insufficient pairs"):
            create_walk_forward_splits(
                pairs,
                train_window=500,
                test_window=100,
            )

    def test_temporal_ordering(self, sample_preference_pairs):
        """Test temporal ordering is correct."""
        pairs = sample_preference_pairs(700)

        split = create_walk_forward_splits(
            pairs,
            train_window=500,
            test_window=100,
        )

        # Training should end before test starts
        assert split.train_end_ms < split.test_start_ms

        # First training pair timestamp
        assert split.train_pairs[0].timestamp_ms == split.train_start_ms

        # Last training pair timestamp
        assert split.train_pairs[-1].timestamp_ms == split.train_end_ms

        # First test pair timestamp
        assert split.test_pairs[0].timestamp_ms == split.test_start_ms

        # Last test pair timestamp
        assert split.test_pairs[-1].timestamp_ms == split.test_end_ms

    def test_replay_pairs_before_training(self, sample_preference_pairs):
        """Test replay pairs are all before training window."""
        pairs = sample_preference_pairs(700)

        split = create_walk_forward_splits(
            pairs,
            train_window=500,
            test_window=100,
            replay_ratio=0.15,
        )

        # All replay pairs should be before training start
        for pair in split.replay_pairs:
            assert pair.timestamp_ms < split.train_start_ms

    def test_no_history_available(self, sample_preference_pairs):
        """Test when exactly enough pairs (no history for replay)."""
        pairs = sample_preference_pairs(600)  # Exactly 500 train + 100 test

        split = create_walk_forward_splits(
            pairs,
            train_window=500,
            test_window=100,
            replay_ratio=0.15,
        )

        assert len(split.train_pairs) == 500
        assert len(split.test_pairs) == 100
        assert len(split.replay_pairs) == 0  # No history available
        assert split.total_pairs == 500

    def test_limited_history(self, sample_preference_pairs):
        """Test when history is limited (fewer replay pairs than requested)."""
        # 650 pairs: 50 history, 500 train, 100 test
        pairs = sample_preference_pairs(650)

        split = create_walk_forward_splits(
            pairs,
            train_window=500,
            test_window=100,
            replay_ratio=0.15,  # Would like 75, but only 50 available
        )

        assert len(split.replay_pairs) == 50  # Limited by history
        assert split.total_pairs == 550  # 500 + 50

    def test_replay_buffer_size_limit(self, sample_preference_pairs):
        """Test replay_buffer_size limits historical sampling."""
        # 1700 pairs: 1000 history, 500 train, 100 test
        pairs = sample_preference_pairs(1700)

        split = create_walk_forward_splits(
            pairs,
            train_window=500,
            test_window=100,
            replay_ratio=0.15,
            replay_buffer_size=200,  # Only sample from most recent 200
        )

        # Should sample 75 pairs from most recent 200 historical pairs
        assert len(split.replay_pairs) == 75
        assert split.total_pairs == 575

    def test_custom_window_sizes(self, sample_preference_pairs):
        """Test custom train/test window sizes."""
        pairs = sample_preference_pairs(800)

        split = create_walk_forward_splits(
            pairs,
            train_window=600,
            test_window=150,
        )

        assert len(split.train_pairs) == 600
        assert len(split.test_pairs) == 150

    def test_empty_pairs_raises(self):
        """Test error on empty preference pairs."""
        with pytest.raises(TemporalSplitError, match="No preference pairs"):
            create_walk_forward_splits([])

    def test_min_training_pairs_validation(self, sample_preference_pairs):
        """Test minimum training pairs requirement."""
        pairs = sample_preference_pairs(650)

        # Should pass with default min (500)
        split = create_walk_forward_splits(
            pairs,
            train_window=500,
            test_window=100,
            replay_ratio=0.0,
            min_training_pairs=500,
        )
        assert split.total_pairs == 500

        # Should fail with higher minimum
        with pytest.raises(TemporalSplitError, match="Insufficient training pairs"):
            create_walk_forward_splits(
                pairs,
                train_window=500,
                test_window=100,
                replay_ratio=0.0,
                min_training_pairs=600,  # More than available
            )


class TestValidateTemporalSplit:
    """Test temporal split validation."""

    def test_valid_split_passes(self, sample_preference_pairs):
        """Test valid split passes validation."""
        pairs = sample_preference_pairs(700)
        split = create_walk_forward_splits(pairs)

        # Should not raise
        validate_temporal_split(split)

    def test_overlapping_train_test_fails(self, sample_preference_pairs):
        """Test overlapping train/test raises error."""
        pairs = sample_preference_pairs(700)
        split = create_walk_forward_splits(pairs)

        # Manually corrupt split to have overlap
        # Create new split with test_start before train_end
        from training.walk_forward import WalkForwardSplit

        bad_split = WalkForwardSplit(
            train_pairs=split.train_pairs,
            test_pairs=split.test_pairs,
            replay_pairs=split.replay_pairs,
            train_start_ms=split.train_start_ms,
            train_end_ms=split.test_end_ms,  # Corrupt: train ends after test
            test_start_ms=split.test_start_ms,
            test_end_ms=split.test_end_ms,
            total_pairs=split.total_pairs,
        )

        with pytest.raises(TemporalSplitError, match="Test data starts.*before training ends"):
            validate_temporal_split(bad_split)


class TestMergeTrainAndReplay:
    """Test merging train and replay pairs."""

    def test_merge_with_shuffle(self, sample_preference_pairs):
        """Test merging with shuffle."""
        pairs = sample_preference_pairs(700)
        split = create_walk_forward_splits(pairs, replay_ratio=0.15)

        merged = merge_train_and_replay(split, shuffle=True)

        assert len(merged) == 575  # 500 train + 75 replay
        # Check all pairs present
        assert set(merged) == set(split.train_pairs + split.replay_pairs)

    def test_merge_without_shuffle(self, sample_preference_pairs):
        """Test merging without shuffle (temporal order)."""
        pairs = sample_preference_pairs(700)
        split = create_walk_forward_splits(pairs, replay_ratio=0.15)

        merged = merge_train_and_replay(split, shuffle=False)

        assert len(merged) == 575
        # Check temporal ordering preserved
        for i in range(len(merged) - 1):
            assert merged[i].timestamp_ms <= merged[i + 1].timestamp_ms

    def test_merge_with_no_replay(self, sample_preference_pairs):
        """Test merging when no replay pairs."""
        pairs = sample_preference_pairs(600)
        split = create_walk_forward_splits(pairs, replay_ratio=0.0)

        merged = merge_train_and_replay(split)

        assert len(merged) == 500  # Only training pairs
        assert set(merged) == set(split.train_pairs)
