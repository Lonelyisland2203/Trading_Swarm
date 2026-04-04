"""Tests for DPO preference pair construction and export."""

import pytest

from swarm.training_capture import TrainingExample
from training.dpo_export import (
    PreferencePair,
    compute_reward_delta,
    construct_preference_pairs,
    export_to_huggingface_format,
    validate_preference_pair,
)
from training.reward_engine import ComputedReward
from verifier.outcome import VerifiedOutcome


@pytest.fixture
def sample_context_id():
    """Shared context ID for related examples."""
    return "ctx-12345"


@pytest.fixture
def sample_training_example(sample_context_id):
    """Create a sample training example."""
    def _make_example(
        persona: str,
        direction: str,
        confidence: float,
        reasoning: str,
        context_id: str = None,
    ):
        return TrainingExample(
            symbol="BTC/USDT",
            timeframe="1h",
            timestamp_ms=1609459200000,
            market_regime="NEUTRAL",
            persona=persona,
            context_id=context_id if context_id is not None else sample_context_id,
            task_prompt="Analyze BTC/USDT 1h chart and predict direction.",
            generator_signal={
                "direction": direction,
                "confidence": confidence,
                "reasoning": reasoning,
                "persona": persona,
            },
        )
    return _make_example


@pytest.fixture
def sample_verified_outcome():
    """Create a sample verified outcome."""
    def _make_outcome(realized_return: float):
        return VerifiedOutcome(
            example_id="ex-123",
            actual_direction="HIGHER" if realized_return > 0 else "LOWER",
            realized_return=realized_return,
            max_adverse_excursion=-0.01,
            net_return=realized_return - 0.002,  # After fees
            entry_price=100.0,
            exit_price=100.0 * (1 + realized_return),
            bars_held=24,
        )
    return _make_outcome


@pytest.fixture
def sample_computed_reward():
    """Create a sample computed reward."""
    def _make_reward(final_reward: float):
        return ComputedReward(
            final_reward=final_reward,
            return_reward=final_reward,
            directional_reward=0.0,
            mae_reward=0.0,
            return_weight=1.0,
            directional_weight=0.0,
            mae_weight=0.0,
            return_scale=10.0,
            mae_scale=10.0,
            net_return=0.05,
            realized_return=0.052,
            mae=-0.01,
            predicted_direction="HIGHER",
            actual_direction="HIGHER",
            confidence=0.8,
            components_used=1,
            computation_timestamp="2024-01-01T00:00:00Z",
            market_regime="NEUTRAL",
            reward_version="1.0.0",
        )
    return _make_reward


class TestComputeRewardDelta:
    """Test reward delta computation."""

    def test_positive_delta(self):
        """Test delta when chosen > rejected."""
        delta = compute_reward_delta(0.8, 0.3)
        assert delta == pytest.approx(0.5, abs=0.01)

    def test_zero_delta(self):
        """Test delta when rewards are equal."""
        delta = compute_reward_delta(0.5, 0.5)
        assert delta == 0.0

    def test_negative_rewards(self):
        """Test delta with negative rewards."""
        delta = compute_reward_delta(-0.2, -0.7)
        assert delta == pytest.approx(0.5, abs=0.01)


class TestValidatePreferencePair:
    """Test preference pair validation."""

    def test_valid_pair(
        self,
        sample_training_example,
        sample_computed_reward,
    ):
        """Test valid preference pair passes validation."""
        chosen = sample_training_example("MOMENTUM", "HIGHER", 0.8, "Strong uptrend")
        rejected = sample_training_example("CONTRARIAN", "LOWER", 0.6, "Overbought")

        chosen_reward = sample_computed_reward(0.8)
        rejected_reward = sample_computed_reward(0.3)

        is_valid, reason = validate_preference_pair(
            chosen,
            rejected,
            chosen_reward,
            rejected_reward,
            min_delta=0.2,
        )

        assert is_valid
        assert "delta=0.5" in reason

    def test_context_id_mismatch(
        self,
        sample_training_example,
        sample_computed_reward,
    ):
        """Test validation fails on context ID mismatch."""
        chosen = sample_training_example("MOMENTUM", "HIGHER", 0.8, "Uptrend", context_id="ctx-1")
        rejected = sample_training_example("CONTRARIAN", "LOWER", 0.6, "Overbought", context_id="ctx-2")

        chosen_reward = sample_computed_reward(0.8)
        rejected_reward = sample_computed_reward(0.3)

        is_valid, reason = validate_preference_pair(
            chosen,
            rejected,
            chosen_reward,
            rejected_reward,
        )

        assert not is_valid
        assert "Context ID mismatch" in reason

    def test_empty_context_id(
        self,
        sample_training_example,
        sample_computed_reward,
    ):
        """Test validation fails on empty context_id."""
        chosen = sample_training_example("MOMENTUM", "HIGHER", 0.8, "Uptrend", context_id="")
        rejected = sample_training_example("CONTRARIAN", "LOWER", 0.6, "Overbought", context_id="")

        chosen_reward = sample_computed_reward(0.8)
        rejected_reward = sample_computed_reward(0.3)

        is_valid, reason = validate_preference_pair(
            chosen,
            rejected,
            chosen_reward,
            rejected_reward,
        )

        assert not is_valid
        assert "Empty context_id" in reason

    def test_invalid_reward_ordering(
        self,
        sample_training_example,
        sample_computed_reward,
    ):
        """Test validation fails when rejected > chosen."""
        chosen = sample_training_example("MOMENTUM", "HIGHER", 0.8, "Uptrend")
        rejected = sample_training_example("CONTRARIAN", "LOWER", 0.6, "Overbought")

        chosen_reward = sample_computed_reward(0.3)  # Lower!
        rejected_reward = sample_computed_reward(0.8)  # Higher!

        is_valid, reason = validate_preference_pair(
            chosen,
            rejected,
            chosen_reward,
            rejected_reward,
        )

        assert not is_valid
        assert "Invalid reward ordering" in reason

    def test_insufficient_delta(
        self,
        sample_training_example,
        sample_computed_reward,
    ):
        """Test validation fails when delta < min_delta."""
        chosen = sample_training_example("MOMENTUM", "HIGHER", 0.8, "Uptrend")
        rejected = sample_training_example("CONTRARIAN", "LOWER", 0.6, "Overbought")

        chosen_reward = sample_computed_reward(0.51)
        rejected_reward = sample_computed_reward(0.50)  # Delta = 0.01

        is_valid, reason = validate_preference_pair(
            chosen,
            rejected,
            chosen_reward,
            rejected_reward,
            min_delta=0.2,
        )

        assert not is_valid
        assert "below minimum" in reason


class TestConstructPreferencePairs:
    """Test preference pair construction."""

    def test_construct_pairs_from_multi_persona(
        self,
        sample_training_example,
        sample_verified_outcome,
        sample_computed_reward,
        sample_context_id,
    ):
        """Test constructing pairs from 5 personas."""
        # Create 5 examples with different rewards
        # Rewards: 0.8, 0.7, 0.6, 0.5, 0.2 (sorted descending)
        # Pairing: (0.8, 0.2) delta=0.6, (0.7, 0.5) delta=0.2, middle (0.6) unpaired
        examples_with_rewards = [
            (
                sample_training_example("MOMENTUM", "HIGHER", 0.9, "Strong momentum", sample_context_id),
                sample_verified_outcome(0.05),
                sample_computed_reward(0.8),
            ),
            (
                sample_training_example("CONTRARIAN", "LOWER", 0.7, "Overbought", sample_context_id),
                sample_verified_outcome(-0.04),
                sample_computed_reward(0.2),
            ),
            (
                sample_training_example("MEAN_REVERSION", "LOWER", 0.6, "Mean reversion", sample_context_id),
                sample_verified_outcome(-0.01),
                sample_computed_reward(0.5),
            ),
            (
                sample_training_example("BREAKOUT", "HIGHER", 0.8, "Breakout", sample_context_id),
                sample_verified_outcome(0.03),
                sample_computed_reward(0.7),
            ),
            (
                sample_training_example("CONSERVATIVE", "HIGHER", 0.5, "Conservative", sample_context_id),
                sample_verified_outcome(0.01),
                sample_computed_reward(0.6),
            ),
        ]

        pairs = construct_preference_pairs(
            examples_with_rewards,
            min_delta=0.19,  # Slightly lower to avoid floating point issues with exact 0.2
            min_personas_per_context=3,
        )

        # Should create 2 pairs (5 personas // 2)
        # Pair 1: best (0.8) vs worst (0.2), delta=0.6
        # Pair 2: 2nd-best (0.7) vs 2nd-worst (0.5), delta=0.2 (or close due to FP precision)
        assert len(pairs) == 2

        # Check first pair (highest delta)
        assert pairs[0].chosen_persona == "MOMENTUM"
        assert pairs[0].rejected_persona == "CONTRARIAN"
        assert pairs[0].reward_delta == pytest.approx(0.6, abs=0.01)

        # Check second pair
        assert pairs[1].chosen_persona == "BREAKOUT"
        assert pairs[1].rejected_persona == "MEAN_REVERSION"
        assert pairs[1].reward_delta == pytest.approx(0.2, abs=0.01)

    def test_skip_insufficient_persona_diversity(
        self,
        sample_training_example,
        sample_verified_outcome,
        sample_computed_reward,
        sample_context_id,
    ):
        """Test skipping contexts with < min_personas_per_context."""
        # Only 2 personas (below min of 3)
        examples_with_rewards = [
            (
                sample_training_example("MOMENTUM", "HIGHER", 0.9, "Momentum", sample_context_id),
                sample_verified_outcome(0.05),
                sample_computed_reward(0.8),
            ),
            (
                sample_training_example("CONTRARIAN", "LOWER", 0.7, "Contrarian", sample_context_id),
                sample_verified_outcome(-0.03),
                sample_computed_reward(0.3),
            ),
        ]

        pairs = construct_preference_pairs(
            examples_with_rewards,
            min_delta=0.2,
            min_personas_per_context=3,
        )

        # Should skip due to insufficient diversity
        assert len(pairs) == 0

    def test_skip_examples_without_context_id(
        self,
        sample_training_example,
        sample_verified_outcome,
        sample_computed_reward,
    ):
        """Test skipping examples from single-persona workflow (no context_id)."""
        examples_with_rewards = [
            (
                sample_training_example("MOMENTUM", "HIGHER", 0.9, "Momentum", context_id=""),
                sample_verified_outcome(0.05),
                sample_computed_reward(0.8),
            ),
            (
                sample_training_example("CONTRARIAN", "LOWER", 0.7, "Contrarian", context_id=""),
                sample_verified_outcome(-0.03),
                sample_computed_reward(0.3),
            ),
        ]

        pairs = construct_preference_pairs(examples_with_rewards)

        # Should skip all examples without context_id
        assert len(pairs) == 0

    def test_multiple_contexts(
        self,
        sample_training_example,
        sample_verified_outcome,
        sample_computed_reward,
    ):
        """Test handling multiple contexts separately."""
        # Context 1: 3 personas
        ctx1_examples = [
            (
                sample_training_example("MOMENTUM", "HIGHER", 0.9, "M1", "ctx-1"),
                sample_verified_outcome(0.05),
                sample_computed_reward(0.8),
            ),
            (
                sample_training_example("CONTRARIAN", "LOWER", 0.7, "C1", "ctx-1"),
                sample_verified_outcome(-0.03),
                sample_computed_reward(0.3),
            ),
            (
                sample_training_example("BREAKOUT", "HIGHER", 0.8, "B1", "ctx-1"),
                sample_verified_outcome(0.03),
                sample_computed_reward(0.6),
            ),
        ]

        # Context 2: 3 personas
        ctx2_examples = [
            (
                sample_training_example("MOMENTUM", "HIGHER", 0.9, "M2", "ctx-2"),
                sample_verified_outcome(0.04),
                sample_computed_reward(0.7),
            ),
            (
                sample_training_example("CONTRARIAN", "LOWER", 0.6, "C2", "ctx-2"),
                sample_verified_outcome(-0.02),
                sample_computed_reward(0.2),
            ),
            (
                sample_training_example("MEAN_REVERSION", "LOWER", 0.5, "MR2", "ctx-2"),
                sample_verified_outcome(-0.01),
                sample_computed_reward(0.4),
            ),
        ]

        all_examples = ctx1_examples + ctx2_examples

        pairs = construct_preference_pairs(
            all_examples,
            min_delta=0.2,
            min_personas_per_context=3,
        )

        # Should create 1 pair per context = 2 total
        assert len(pairs) == 2

        # Check contexts are different
        contexts = {pair.context_id for pair in pairs}
        assert len(contexts) == 2


class TestExportToHuggingFaceFormat:
    """Test HuggingFace format export."""

    def test_export_basic(self, sample_context_id):
        """Test basic HF format export."""
        pairs = [
            PreferencePair(
                prompt="Analyze BTC/USDT",
                context_id=sample_context_id,
                chosen_reasoning="Strong momentum",
                chosen_direction="HIGHER",
                chosen_confidence=0.8,
                chosen_reward=0.8,
                chosen_example_id="ex-1",
                chosen_persona="MOMENTUM",
                rejected_reasoning="Overbought",
                rejected_direction="LOWER",
                rejected_confidence=0.6,
                rejected_reward=0.3,
                rejected_example_id="ex-2",
                rejected_persona="CONTRARIAN",
                reward_delta=0.5,
                symbol="BTC/USDT",
                timestamp_ms=1609459200000,
                market_regime="NEUTRAL",
            )
        ]

        hf_data = export_to_huggingface_format(pairs)

        assert len(hf_data) == 1
        assert hf_data[0]["prompt"] == "Analyze BTC/USDT"
        assert hf_data[0]["chosen"] == "Strong momentum"
        assert hf_data[0]["rejected"] == "Overbought"

    def test_export_preserves_order(self, sample_context_id):
        """Test that export preserves pair order."""
        pairs = [
            PreferencePair(
                prompt=f"Prompt {i}",
                context_id=f"{sample_context_id}-{i}",
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
                timestamp_ms=1609459200000 + i,
                market_regime="NEUTRAL",
            )
            for i in range(3)
        ]

        hf_data = export_to_huggingface_format(pairs)

        assert len(hf_data) == 3
        for i, item in enumerate(hf_data):
            assert item["prompt"] == f"Prompt {i}"
            assert item["chosen"] == f"Chosen {i}"
            assert item["rejected"] == f"Rejected {i}"
