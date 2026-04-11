"""Tests for GRPO training data types and walk-forward splits."""

import pytest

from training.grpo_data import GRPOTrainingExample


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
