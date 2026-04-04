"""Tests for reward computation engine with integration scenarios."""

import pytest

from swarm.training_capture import TrainingExample
from training.reward_config import RewardScaling
from training.reward_engine import (
    REWARD_VERSION,
    BatchDiagnostics,
    ComputedReward,
    compute_reward,
    compute_rewards_for_batch,
)
from verifier.outcome import VerifiedOutcome


@pytest.fixture
def sample_scaling():
    """Default scaling for tests."""
    return RewardScaling(return_scale=10.0, mae_scale=10.0)


@pytest.fixture
def profitable_higher_signal():
    """Sample profitable HIGHER signal with all components."""
    example = TrainingExample(
        example_id="test-higher",
        symbol="BTC/USDT",
        timeframe="1h",
        timestamp_ms=1000,
        market_regime="NEUTRAL",
        generator_signal={
            "direction": "HIGHER",
            "confidence": 0.8,
            "reasoning": "Strong uptrend",
        },
    )
    
    outcome = VerifiedOutcome(
        example_id="test-higher",
        actual_direction="HIGHER",  # Correct prediction
        realized_return=0.05,       # 5% gain
        max_adverse_excursion=-0.02,  # 2% drawdown
        net_return=0.048,           # After costs
        entry_price=100.0,
        exit_price=105.0,
        bars_held=24,
    )
    
    return example, outcome


@pytest.fixture
def wrong_lower_signal():
    """Sample wrong LOWER signal."""
    example = TrainingExample(
        example_id="test-wrong",
        symbol="ETH/USDT",
        timeframe="1h",
        timestamp_ms=2000,
        market_regime="NEUTRAL",
        generator_signal={
            "direction": "LOWER",
            "confidence": 0.9,  # High confidence but wrong
            "reasoning": "Overbought",
        },
    )
    
    outcome = VerifiedOutcome(
        example_id="test-wrong",
        actual_direction="HIGHER",  # Wrong prediction
        realized_return=0.03,
        max_adverse_excursion=-0.01,
        net_return=0.028,
        entry_price=200.0,
        exit_price=206.0,
        bars_held=24,
    )
    
    return example, outcome


@pytest.fixture
def flat_outcome_signal():
    """Sample signal with FLAT outcome."""
    example = TrainingExample(
        example_id="test-flat",
        symbol="BTC/USDT",
        timeframe="1h",
        timestamp_ms=3000,
        market_regime="NEUTRAL",
        generator_signal={
            "direction": "HIGHER",
            "confidence": 0.7,
            "reasoning": "Test",
        },
    )
    
    outcome = VerifiedOutcome(
        example_id="test-flat",
        actual_direction="FLAT",  # No directional move
        realized_return=0.0005,   # Tiny move
        max_adverse_excursion=0.0,
        net_return=-0.001,  # Lost on costs
        entry_price=100.0,
        exit_price=100.05,
        bars_held=24,
    )
    
    return example, outcome


class TestComputeReward:
    """Test main reward computation function."""
    
    def test_profitable_correct_signal(self, profitable_higher_signal, sample_scaling):
        """Test reward for profitable, correct signal."""
        example, outcome = profitable_higher_signal
        
        reward = compute_reward(outcome, example, sample_scaling)
        
        # Should have positive final reward
        assert reward.final_reward > 0
        
        # Check components
        assert reward.return_reward > 0  # Positive return
        assert reward.directional_reward > 0  # Correct direction
        assert reward.mae_reward < 0  # MAE is penalty (negative)
        
        # Check metadata
        assert reward.components_used == 3  # All components
        assert reward.reward_version == REWARD_VERSION
        assert reward.actual_direction == "HIGHER"
        assert reward.predicted_direction == "HIGHER"
    
    def test_wrong_direction_high_confidence(self, wrong_lower_signal, sample_scaling):
        """Test reward for wrong prediction with high confidence."""
        example, outcome = wrong_lower_signal
        
        reward = compute_reward(outcome, example, sample_scaling)
        
        # Direction component should be heavily penalized
        assert reward.directional_reward < -0.5  # High confidence wrong
        
        # But return is still positive
        assert reward.return_reward > 0
        
        # Final reward depends on weight balance
        # With default weights (0.5 return, 0.3 dir, 0.2 mae),
        # positive return might outweigh wrong direction
        assert reward.components_used == 3
    
    def test_flat_outcome_skips_directional(self, flat_outcome_signal, sample_scaling):
        """Test that FLAT outcome skips directional component."""
        example, outcome = flat_outcome_signal
        
        reward = compute_reward(outcome, example, sample_scaling)
        
        # Directional component should be zero
        assert reward.directional_reward == 0.0
        
        # Only 2 components used (return + mae)
        assert reward.components_used == 2
        
        # Weights should be renormalized
        # Original: return=0.5, directional=0.3, mae=0.2
        # After skipping directional: return=0.5/(0.5+0.2)=0.714, mae=0.2/(0.5+0.2)=0.286
    
    def test_missing_mae_skips_mae_component(self, sample_scaling):
        """Test that missing MAE skips MAE component."""
        example = TrainingExample(
            example_id="test-no-mae",
            symbol="BTC/USDT",
            timeframe="1h",
            timestamp_ms=4000,
            market_regime="NEUTRAL",
            generator_signal={
                "direction": "HIGHER",
                "confidence": 0.8,
                "reasoning": "Test",
            },
        )
        
        outcome = VerifiedOutcome(
            example_id="test-no-mae",
            actual_direction="HIGHER",
            realized_return=0.03,
            max_adverse_excursion=None,  # Missing MAE
            net_return=0.028,
            entry_price=100.0,
            exit_price=103.0,
            bars_held=24,
        )
        
        reward = compute_reward(outcome, example, sample_scaling)
        
        # MAE component should be zero
        assert reward.mae_reward == 0.0
        assert reward.mae is None
        
        # Only 2 components used (return + directional)
        assert reward.components_used == 2
    
    def test_final_reward_clipped(self, sample_scaling):
        """Test that final reward is clipped to [-1, 1]."""
        # Create extreme positive scenario
        example = TrainingExample(
            example_id="test-extreme",
            symbol="BTC/USDT",
            timeframe="1h",
            timestamp_ms=5000,
            market_regime="NEUTRAL",
            generator_signal={
                "direction": "HIGHER",
                "confidence": 1.0,  # Max confidence
                "reasoning": "Test",
            },
        )
        
        outcome = VerifiedOutcome(
            example_id="test-extreme",
            actual_direction="HIGHER",
            realized_return=0.20,  # 20% return (extreme)
            max_adverse_excursion=0.0,  # No drawdown
            net_return=0.198,
            entry_price=100.0,
            exit_price=120.0,
            bars_held=24,
        )
        
        reward = compute_reward(outcome, example, sample_scaling)
        
        # Final reward should be clipped to 1.0
        assert reward.final_reward <= 1.0
        assert reward.final_reward >= -1.0
    
    def test_stores_raw_values_for_recomputation(self, profitable_higher_signal, sample_scaling):
        """Test that raw input values are stored."""
        example, outcome = profitable_higher_signal
        
        reward = compute_reward(outcome, example, sample_scaling)
        
        # Check raw values stored
        assert reward.net_return == outcome.net_return
        assert reward.realized_return == outcome.realized_return
        assert reward.mae == outcome.max_adverse_excursion
        assert reward.predicted_direction == "HIGHER"
        assert reward.actual_direction == "HIGHER"
        assert reward.confidence == 0.8
    
    def test_stores_weights_and_scaling(self, profitable_higher_signal, sample_scaling):
        """Test that weights and scaling are stored."""
        example, outcome = profitable_higher_signal
        
        reward = compute_reward(outcome, example, sample_scaling)
        
        # Check weights (from config)
        assert reward.return_weight == 0.5
        assert reward.directional_weight == 0.3
        assert reward.mae_weight == 0.2
        
        # Check scaling
        assert reward.return_scale == 10.0
        assert reward.mae_scale == 10.0
    
    def test_computation_timestamp_present(self, profitable_higher_signal, sample_scaling):
        """Test that computation timestamp is recorded."""
        example, outcome = profitable_higher_signal
        
        reward = compute_reward(outcome, example, sample_scaling)
        
        assert reward.computation_timestamp is not None
        assert len(reward.computation_timestamp) > 0
        # Should be ISO format
        assert "T" in reward.computation_timestamp


class TestComputeRewardsForBatch:
    """Test batch reward computation."""
    
    def test_empty_batch(self, sample_scaling):
        """Test that empty batch returns empty result."""
        result = compute_rewards_for_batch([], sample_scaling)
        
        assert len(result.rewards) == 0
        assert result.diagnostics.total_examples == 0
        assert result.diagnostics.mean_reward == 0.0
    
    def test_single_example_batch(self, profitable_higher_signal, sample_scaling):
        """Test batch with single example."""
        example, outcome = profitable_higher_signal
        
        result = compute_rewards_for_batch([(example, outcome)], sample_scaling)
        
        assert len(result.rewards) == 1
        assert result.diagnostics.total_examples == 1
        assert result.diagnostics.std_reward == 0.0  # Only one example
    
    def test_batch_diagnostics_computed(self, profitable_higher_signal, wrong_lower_signal, sample_scaling):
        """Test that batch diagnostics are computed."""
        pairs = [profitable_higher_signal, wrong_lower_signal]
        
        result = compute_rewards_for_batch(pairs, sample_scaling)
        
        assert len(result.rewards) == 2
        
        # Check diagnostics
        diag = result.diagnostics
        assert diag.total_examples == 2
        assert isinstance(diag.mean_reward, float)
        assert isinstance(diag.std_reward, float)
        assert diag.min_reward <= diag.mean_reward <= diag.max_reward
        assert 0.0 <= diag.pct_positive <= 1.0
        assert 0.0 <= diag.pct_clipped <= 1.0
    
    def test_batch_with_mixed_outcomes(
        self,
        profitable_higher_signal,
        wrong_lower_signal,
        flat_outcome_signal,
        sample_scaling,
    ):
        """Test batch with diverse outcomes."""
        pairs = [profitable_higher_signal, wrong_lower_signal, flat_outcome_signal]
        
        result = compute_rewards_for_batch(pairs, sample_scaling)
        
        assert len(result.rewards) == 3
        assert result.diagnostics.total_examples == 3
        
        # Should have mix of positive and negative
        reward_values = [r.final_reward for r in result.rewards]
        assert min(reward_values) < max(reward_values)  # Some variation
    
    def test_batch_diagnostics_pct_positive(self, sample_scaling):
        """Test percentage positive calculation."""
        # Create 3 examples: 2 positive, 1 negative
        pairs = []
        for i, net_return in enumerate([0.05, 0.03, -0.02]):
            example = TrainingExample(
                example_id=f"test-{i}",
                symbol="BTC/USDT",
                timeframe="1h",
                timestamp_ms=1000 + i,
                market_regime="NEUTRAL",
                generator_signal={
                    "direction": "HIGHER",
                    "confidence": 0.8,
                    "reasoning": "Test",
                },
            )
            
            outcome = VerifiedOutcome(
                example_id=f"test-{i}",
                actual_direction="HIGHER",
                realized_return=net_return,
                max_adverse_excursion=0.0,
                net_return=net_return - 0.002,
                entry_price=100.0,
                exit_price=100.0 + net_return * 100,
                bars_held=24,
            )
            
            pairs.append((example, outcome))
        
        result = compute_rewards_for_batch(pairs, sample_scaling)
        
        # Should have 2/3 positive (approximately, depends on components)
        # At minimum, check pct_positive is computed
        assert 0.0 <= result.diagnostics.pct_positive <= 1.0
    
    def test_rewards_are_deterministic(self, profitable_higher_signal, sample_scaling):
        """Test that same input produces same reward."""
        example, outcome = profitable_higher_signal
        
        reward1 = compute_reward(outcome, example, sample_scaling)
        reward2 = compute_reward(outcome, example, sample_scaling)
        
        # Final rewards should be identical (deterministic)
        assert reward1.final_reward == reward2.final_reward
        assert reward1.return_reward == reward2.return_reward
        assert reward1.directional_reward == reward2.directional_reward
        assert reward1.mae_reward == reward2.mae_reward


class TestComputedReward:
    """Test ComputedReward dataclass."""
    
    def test_computed_reward_is_frozen(self, profitable_higher_signal, sample_scaling):
        """Test that ComputedReward is immutable."""
        example, outcome = profitable_higher_signal
        reward = compute_reward(outcome, example, sample_scaling)
        
        with pytest.raises(AttributeError):
            reward.final_reward = 0.5  # type: ignore
    
    def test_computed_reward_has_all_fields(self, profitable_higher_signal, sample_scaling):
        """Test that all expected fields are present."""
        example, outcome = profitable_higher_signal
        reward = compute_reward(outcome, example, sample_scaling)
        
        # Check all expected attributes exist
        assert hasattr(reward, "final_reward")
        assert hasattr(reward, "return_reward")
        assert hasattr(reward, "directional_reward")
        assert hasattr(reward, "mae_reward")
        assert hasattr(reward, "return_weight")
        assert hasattr(reward, "directional_weight")
        assert hasattr(reward, "mae_weight")
        assert hasattr(reward, "return_scale")
        assert hasattr(reward, "mae_scale")
        assert hasattr(reward, "net_return")
        assert hasattr(reward, "realized_return")
        assert hasattr(reward, "mae")
        assert hasattr(reward, "predicted_direction")
        assert hasattr(reward, "actual_direction")
        assert hasattr(reward, "confidence")
        assert hasattr(reward, "components_used")
        assert hasattr(reward, "computation_timestamp")
        assert hasattr(reward, "reward_version")


class TestBatchDiagnostics:
    """Test BatchDiagnostics dataclass."""
    
    def test_batch_diagnostics_is_frozen(self):
        """Test that BatchDiagnostics is immutable."""
        diag = BatchDiagnostics(
            mean_reward=0.5,
            std_reward=0.2,
            min_reward=-0.3,
            max_reward=0.9,
            pct_positive=0.7,
            pct_clipped=0.1,
            total_examples=10,
        )
        
        with pytest.raises(AttributeError):
            diag.mean_reward = 0.3  # type: ignore
