"""Tests verifying that reward engine uses net_return (after fees) instead of realized_return."""

import pytest

from swarm.training_capture import TrainingExample
from training.reward_config import RewardScaling
from training.reward_engine import compute_reward
from verifier.outcome import VerifiedOutcome


class TestRewardUsesNetReturn:
    """Tests confirming the enforcement: rewards use net_return, not realized_return."""

    def test_reward_uses_net_return_not_gross_return(self):
        """
        Test that compute_reward uses net_return (after fees) for return component.

        This is the critical enforcement: the return component should be based on
        net_return (after transaction costs), not realized_return (gross).
        """
        example = TrainingExample(
            example_id="test-net-return",
            symbol="BTC/USDT",
            timeframe="1h",
            timestamp_ms=1000,
            market_regime="NEUTRAL",
            generator_signal={
                "direction": "HIGHER",
                "confidence": 0.8,
                "reasoning": "Test net vs gross return",
            },
        )

        # Create outcome where gross and net are significantly different
        # Gross: +5.0% (realized_return)
        # Net: +4.7% (net_return after fees)
        outcome = VerifiedOutcome(
            example_id="test-net-return",
            actual_direction="HIGHER",
            realized_return=0.050,  # 5.0% GROSS
            max_adverse_excursion=-0.01,
            net_return=0.047,  # 4.7% NET (after ~0.3% fees)
            entry_price=100.0,
            exit_price=105.0,
            bars_held=24,
        )

        scaling = RewardScaling(return_scale=10.0, mae_scale=10.0)
        reward = compute_reward(outcome, example, scaling)

        # Return reward should be based on net_return (0.047), not realized_return (0.050)
        # With scale=10.0: return_reward = 0.047 * 10.0 = 0.47
        expected_return_reward = 0.047 * 10.0
        assert reward.return_reward == pytest.approx(expected_return_reward, abs=0.001)

        # Verify the stored values match
        assert reward.net_return == 0.047
        assert reward.realized_return == 0.050

        # The difference (0.003 = 0.3%) should be evident in the reward
        # If it had used gross return: 0.050 * 10.0 = 0.50
        # But it uses net return:      0.047 * 10.0 = 0.47
        wrong_return_reward = 0.050 * 10.0
        assert reward.return_reward != wrong_return_reward
        assert reward.return_reward < wrong_return_reward

    def test_fees_reduce_reward_when_margin_is_small(self):
        """
        Test that small-margin trades are penalized by fee deduction.

        A signal that's barely profitable before fees but unprofitable after fees
        should get a negative or near-zero return reward.
        """
        example = TrainingExample(
            example_id="test-small-margin",
            symbol="ETH/USDT",
            timeframe="5m",
            timestamp_ms=2000,
            market_regime="NEUTRAL",
            generator_signal={
                "direction": "HIGHER",
                "confidence": 0.6,
                "reasoning": "Small margin trade",
            },
        )

        # Barely profitable before fees (0.15%), unprofitable after (0.065% loss)
        outcome = VerifiedOutcome(
            example_id="test-small-margin",
            actual_direction="HIGHER",
            realized_return=0.0015,  # +0.15% GROSS (looks good)
            max_adverse_excursion=-0.005,
            net_return=-0.00065,  # -0.065% NET (fees killed profit!)
            entry_price=200.0,
            exit_price=200.30,
            bars_held=1,
        )

        scaling = RewardScaling(return_scale=10.0, mae_scale=10.0)
        reward = compute_reward(outcome, example, scaling)

        # Return reward should be NEGATIVE because net_return is negative
        assert reward.return_reward < 0

        # Verify it's using net_return (negative), not realized_return (positive)
        assert reward.net_return < 0
        assert reward.realized_return > 0

        # The return component should reflect the net loss
        expected_return_reward = -0.00065 * 10.0
        assert reward.return_reward == pytest.approx(expected_return_reward, abs=0.001)

    def test_large_fees_impact_on_reward_magnitude(self):
        """
        Test that fee impact is reflected in final reward magnitude.

        Two outcomes with same direction and MAE should have different rewards
        if their fee impact (spread between net_return and realized_return) differs.
        """
        base_example = TrainingExample(
            example_id="test-fee-impact",
            symbol="BTC/USDT",
            timeframe="1h",
            timestamp_ms=3000,
            market_regime="NEUTRAL",
            generator_signal={
                "direction": "HIGHER",
                "confidence": 0.75,
                "reasoning": "Fee impact comparison",
            },
        )

        # Scenario 1: Low-fee trade (e.g., good execution)
        low_fee_outcome = VerifiedOutcome(
            example_id="test-fee-impact-low",
            actual_direction="HIGHER",
            realized_return=0.10,  # 10% GROSS
            max_adverse_excursion=-0.02,
            net_return=0.098,  # 9.8% NET (small 0.2% fee)
            entry_price=100.0,
            exit_price=110.0,
            bars_held=24,
        )

        # Scenario 2: High-fee trade (e.g., poor execution, funding costs)
        high_fee_outcome = VerifiedOutcome(
            example_id="test-fee-impact-high",
            actual_direction="HIGHER",
            realized_return=0.10,  # 10% GROSS (same as Scenario 1)
            max_adverse_excursion=-0.02,  # Same MAE
            net_return=0.085,  # 8.5% NET (1.5% fee)
            entry_price=100.0,
            exit_price=110.0,
            bars_held=24,
        )

        scaling = RewardScaling(return_scale=10.0, mae_scale=10.0)

        # Create identical examples for fair comparison
        example_low = TrainingExample(
            example_id="test-fee-impact-low",
            symbol="BTC/USDT",
            timeframe="1h",
            timestamp_ms=3000,
            market_regime="NEUTRAL",
            generator_signal={
                "direction": "HIGHER",
                "confidence": 0.75,
                "reasoning": "Fee impact comparison",
            },
        )

        example_high = TrainingExample(
            example_id="test-fee-impact-high",
            symbol="BTC/USDT",
            timeframe="1h",
            timestamp_ms=3001,
            market_regime="NEUTRAL",
            generator_signal={
                "direction": "HIGHER",
                "confidence": 0.75,
                "reasoning": "Fee impact comparison",
            },
        )

        reward_low_fee = compute_reward(low_fee_outcome, example_low, scaling)
        reward_high_fee = compute_reward(high_fee_outcome, example_high, scaling)

        # Both have same direction and MAE, so directional and mae rewards are identical
        assert reward_low_fee.directional_reward == reward_high_fee.directional_reward
        assert reward_low_fee.mae_reward == reward_high_fee.mae_reward

        # But return rewards differ due to fee impact
        assert reward_low_fee.return_reward > reward_high_fee.return_reward

        # Since weights are identical and only return differs, final reward reflects this
        assert reward_low_fee.final_reward > reward_high_fee.final_reward

        # Verify we're using net_return, not realized_return
        assert reward_low_fee.net_return == 0.098
        assert reward_high_fee.net_return == 0.085
        assert reward_low_fee.net_return != reward_high_fee.net_return

        # But realized_return is the same, so if we used that, rewards would be identical
        assert reward_low_fee.realized_return == reward_high_fee.realized_return
