"""Tests for GRPO training data types and walk-forward splits."""

import pytest

from training.grpo_data import (
    GRPOTrainingExample,
    GRPOWalkForwardSplit,
    TemporalSplitError,
    create_grpo_walk_forward_split,
)


class TestGRPOTrainingExample:
    """Tests for GRPOTrainingExample dataclass."""

    def test_create_example(self) -> None:
        """Test creating a basic training example."""
        example = GRPOTrainingExample(
            market_snapshot="BTC/USDT: RSI=30, MACD=-0.5",
            actual_direction="LONG",
            gross_return_pct=0.5,
            timestamp_ms=1700000000000,
        )
        assert example.market_snapshot == "BTC/USDT: RSI=30, MACD=-0.5"
        assert example.actual_direction == "LONG"
        assert example.gross_return_pct == 0.5
        assert example.timestamp_ms == 1700000000000

    def test_example_is_frozen(self) -> None:
        """Test that example is immutable."""
        example = GRPOTrainingExample(
            market_snapshot="test",
            actual_direction="LONG",
            gross_return_pct=0.1,
            timestamp_ms=1000,
        )
        with pytest.raises(AttributeError):
            example.actual_direction = "SHORT"  # type: ignore[misc]


class TestGRPOWalkForwardSplit:
    """Tests for walk-forward temporal splits."""

    @pytest.fixture
    def sample_examples(self) -> list[GRPOTrainingExample]:
        """Create 700 sample examples with sequential timestamps."""
        return [
            GRPOTrainingExample(
                market_snapshot=f"snapshot_{i}",
                actual_direction="LONG" if i % 2 == 0 else "SHORT",
                gross_return_pct=0.1 * (i % 10 - 5),
                timestamp_ms=1700000000000 + i * 1000,
            )
            for i in range(700)
        ]

    def test_basic_split(self, sample_examples: list[GRPOTrainingExample]) -> None:
        """Test basic train/test split."""
        split = create_grpo_walk_forward_split(
            sample_examples,
            train_window=500,
            test_window=100,
            replay_ratio=0.0,
        )
        assert len(split.train_examples) == 500
        assert len(split.test_examples) == 100
        assert len(split.replay_examples) == 0

    def test_temporal_ordering_enforced(
        self, sample_examples: list[GRPOTrainingExample]
    ) -> None:
        """Test that train examples all come before test examples."""
        split = create_grpo_walk_forward_split(
            sample_examples,
            train_window=500,
            test_window=100,
            replay_ratio=0.0,
        )
        train_max_ts = max(e.timestamp_ms for e in split.train_examples)
        test_min_ts = min(e.timestamp_ms for e in split.test_examples)
        assert train_max_ts < test_min_ts

    def test_replay_buffer_sampling(
        self, sample_examples: list[GRPOTrainingExample]
    ) -> None:
        """Test that replay samples 15% from history."""
        split = create_grpo_walk_forward_split(
            sample_examples,
            train_window=500,
            test_window=100,
            replay_ratio=0.15,
        )
        # 15% of 500 = 75 replay examples
        assert len(split.replay_examples) == 75
        # Replay examples should all be before train examples
        replay_max_ts = max(e.timestamp_ms for e in split.replay_examples)
        train_min_ts = min(e.timestamp_ms for e in split.train_examples)
        assert replay_max_ts < train_min_ts

    def test_insufficient_examples_raises(self) -> None:
        """Test that insufficient examples raises TemporalSplitError."""
        small_examples = [
            GRPOTrainingExample(
                market_snapshot=f"snapshot_{i}",
                actual_direction="LONG",
                gross_return_pct=0.1,
                timestamp_ms=1700000000000 + i * 1000,
            )
            for i in range(100)
        ]
        with pytest.raises(TemporalSplitError, match="Insufficient"):
            create_grpo_walk_forward_split(
                small_examples,
                train_window=500,
                test_window=100,
            )

    def test_empty_examples_raises(self) -> None:
        """Test that empty list raises TemporalSplitError."""
        with pytest.raises(TemporalSplitError, match="No examples"):
            create_grpo_walk_forward_split([])
