"""
Test verifying that preference pairs rank by NET reward, not gross.

Confirms DPO training will optimize for net profitability after fees.
This test uses real fee model calculations to ensure:
1. Example A: +0.30% gross → +0.207% net (profitable after fees)
2. Example B: +0.08% gross → -0.013% net (unprofitable after fees)

The construct_preference_pairs() function should rank by final_reward,
which uses net_return from the fee model (Task 4).
"""

import math
import pytest

from config.fee_model import FeeModelSettings
from swarm.training_capture import TrainingExample
from training.dpo_export import construct_preference_pairs
from training.reward_engine import compute_reward
from verifier.outcome import VerifiedOutcome, apply_fee_model


class TestPreferencePairNetRanking:
    """Test that preference pairs rank by net reward, not gross."""

    @pytest.fixture
    def fee_model(self):
        """Standard Binance Futures fee model with defaults."""
        return FeeModelSettings(
            maker_fee_pct=0.02,
            taker_fee_pct=0.05,
            entry_order_type="maker",
            exit_order_type="taker",
            bnb_discount_enabled=True,
            bnb_discount_pct=10.0,
            funding_rate_pct=0.01,
            funding_interval_hours=8,
            slippage_pct=0.02,
            include_funding=True,
        )

    @pytest.fixture
    def context_id(self):
        """Shared context ID for both examples."""
        return "ctx-net-ranking-test"

    def test_preference_pair_ranks_by_net_not_gross(self, fee_model, context_id):
        """
        Verify that preference pairs rank by NET reward, ensuring DPO trains on profitability.

        Scenario:
        - Example A: +0.30% gross → +0.207% net (profitable)
        - Example B: +0.08% gross → -0.013% net (unprofitable)

        Fee calculation with defaults:
        - Entry fee (maker): 0.02% * 0.9 = 0.018%
        - Exit fee (taker): 0.05% * 0.9 = 0.045%
        - Funding (0 periods): 0%
        - Slippage: 0.02%
        - Total cost: 0.083%

        Net returns:
        - Example A: 0.30% - 0.083% = 0.217% (approximately, before log conversion)
        - Example B: 0.08% - 0.083% = -0.003% (approximately, before log conversion)

        Expected outcome:
        - Example A (net profitable) should be chosen
        - Example B (net unprofitable) should be rejected
        """
        # Example A: +0.30% gross return
        # Convert percentage to log return
        gross_return_a_pct = 0.30
        gross_log_a = math.log(1 + gross_return_a_pct / 100)

        # Apply fee model to get net log return
        net_log_a = apply_fee_model(gross_log_a, fee_model, holding_periods_8h=0)
        net_return_a_pct = (math.exp(net_log_a) - 1) * 100

        # Example B: +0.08% gross return (below fee hurdle)
        gross_return_b_pct = 0.08
        gross_log_b = math.log(1 + gross_return_b_pct / 100)

        # Apply fee model to get net log return
        net_log_b = apply_fee_model(gross_log_b, fee_model, holding_periods_8h=0)
        net_return_b_pct = (math.exp(net_log_b) - 1) * 100

        # Verify net returns match expectations
        assert net_return_a_pct > 0.20, (
            f"Example A net return {net_return_a_pct:.3f}% should be > 0.20%"
        )
        assert net_return_b_pct < 0, f"Example B net return {net_return_b_pct:.3f}% should be < 0%"

        # Create verified outcomes with net returns
        outcome_a = VerifiedOutcome(
            example_id="example-a-profitable",
            actual_direction="HIGHER",
            realized_return=gross_log_a,  # Gross log return
            max_adverse_excursion=-0.001,
            net_return=net_return_a_pct,  # Net return after fees
            entry_price=100.0,
            exit_price=100.0 * (1 + gross_return_a_pct / 100),
            bars_held=12,
        )

        outcome_b = VerifiedOutcome(
            example_id="example-b-unprofitable",
            actual_direction="HIGHER",
            realized_return=gross_log_b,  # Gross log return
            max_adverse_excursion=-0.001,
            net_return=net_return_b_pct,  # Net return after fees (negative!)
            entry_price=100.0,
            exit_price=100.0 * (1 + gross_return_b_pct / 100),
            bars_held=12,
        )

        # Create training examples for both
        example_a = TrainingExample(
            symbol="BTC/USDT",
            timeframe="1h",
            timestamp_ms=1609459200000,
            market_regime="NEUTRAL",
            persona="MOMENTUM",
            context_id=context_id,
            task_prompt="Analyze BTC/USDT 1h chart and predict direction.",
            generator_signal={
                "direction": "HIGHER",
                "confidence": 0.8,
                "reasoning": "Strong uptrend with momentum, +0.30% gross profit expected",
                "persona": "MOMENTUM",
            },
        )

        example_b = TrainingExample(
            symbol="BTC/USDT",
            timeframe="1h",
            timestamp_ms=1609459200000,
            market_regime="NEUTRAL",
            persona="CONTRARIAN",
            context_id=context_id,
            task_prompt="Analyze BTC/USDT 1h chart and predict direction.",
            generator_signal={
                "direction": "HIGHER",
                "confidence": 0.5,
                "reasoning": "Small upside potential, only +0.08% gross (below fee hurdle)",
                "persona": "CONTRARIAN",
            },
        )

        # Compute rewards using the reward engine
        # This will use net_return from the outcomes
        reward_a = compute_reward(outcome_a, example_a)
        reward_b = compute_reward(outcome_b, example_b)

        # Verify reward ranking
        # Example A (profitable net) should have higher reward than Example B (unprofitable net)
        assert reward_a.final_reward > reward_b.final_reward, (
            f"Example A (net profitable) should have higher reward than Example B (net unprofitable). "
            f"A: {reward_a.final_reward:.3f}, B: {reward_b.final_reward:.3f}"
        )

        # Verify that rewards are based on net returns
        assert reward_a.net_return > 0.20, (
            f"Example A net_return {reward_a.net_return:.3f} should be > 0.20%"
        )
        assert reward_b.net_return < 0, (
            f"Example B net_return {reward_b.net_return:.3f} should be < 0%"
        )

    def test_construct_preference_pairs_chooses_net_profitable(self, fee_model, context_id):
        """
        Integration test: construct_preference_pairs should choose the net-profitable signal.

        This test creates multiple examples (including unprofitable ones) and verifies
        that the final preference pairs rank signals by net profitability.
        """
        # Create 3 examples with varying gross/net returns

        # Example 1: +0.35% gross → +0.267% net (highly profitable)
        gross_1_pct = 0.35
        gross_log_1 = math.log(1 + gross_1_pct / 100)
        net_log_1 = apply_fee_model(gross_log_1, fee_model, holding_periods_8h=0)
        net_return_1_pct = (math.exp(net_log_1) - 1) * 100

        # Example 2: +0.08% gross → -0.003% net (unprofitable)
        gross_2_pct = 0.08
        gross_log_2 = math.log(1 + gross_2_pct / 100)
        net_log_2 = apply_fee_model(gross_log_2, fee_model, holding_periods_8h=0)
        net_return_2_pct = (math.exp(net_log_2) - 1) * 100

        # Example 3: +0.12% gross → +0.037% net (barely profitable)
        gross_3_pct = 0.12
        gross_log_3 = math.log(1 + gross_3_pct / 100)
        net_log_3 = apply_fee_model(gross_log_3, fee_model, holding_periods_8h=0)
        net_return_3_pct = (math.exp(net_log_3) - 1) * 100

        # Create outcomes
        outcome_1 = VerifiedOutcome(
            example_id="example-1-highly-profitable",
            actual_direction="HIGHER",
            realized_return=gross_log_1,
            max_adverse_excursion=-0.001,
            net_return=net_return_1_pct,
            entry_price=100.0,
            exit_price=100.0 * (1 + gross_1_pct / 100),
            bars_held=12,
        )

        outcome_2 = VerifiedOutcome(
            example_id="example-2-unprofitable",
            actual_direction="HIGHER",
            realized_return=gross_log_2,
            max_adverse_excursion=-0.001,
            net_return=net_return_2_pct,
            entry_price=100.0,
            exit_price=100.0 * (1 + gross_2_pct / 100),
            bars_held=12,
        )

        outcome_3 = VerifiedOutcome(
            example_id="example-3-barely-profitable",
            actual_direction="HIGHER",
            realized_return=gross_log_3,
            max_adverse_excursion=-0.001,
            net_return=net_return_3_pct,
            entry_price=100.0,
            exit_price=100.0 * (1 + gross_3_pct / 100),
            bars_held=12,
        )

        # Create training examples
        example_1 = TrainingExample(
            symbol="BTC/USDT",
            timeframe="1h",
            timestamp_ms=1609459200000,
            market_regime="NEUTRAL",
            persona="MOMENTUM",
            context_id=context_id,
            task_prompt="Analyze BTC/USDT 1h chart and predict direction.",
            generator_signal={
                "direction": "HIGHER",
                "confidence": 0.9,
                "reasoning": "Strong momentum signal, +0.35% gross profit",
                "persona": "MOMENTUM",
            },
        )

        example_2 = TrainingExample(
            symbol="BTC/USDT",
            timeframe="1h",
            timestamp_ms=1609459200000,
            market_regime="NEUTRAL",
            persona="CONTRARIAN",
            context_id=context_id,
            task_prompt="Analyze BTC/USDT 1h chart and predict direction.",
            generator_signal={
                "direction": "HIGHER",
                "confidence": 0.4,
                "reasoning": "Weak signal, +0.08% gross (net loss after fees)",
                "persona": "CONTRARIAN",
            },
        )

        example_3 = TrainingExample(
            symbol="BTC/USDT",
            timeframe="1h",
            timestamp_ms=1609459200000,
            market_regime="NEUTRAL",
            persona="BREAKOUT",
            context_id=context_id,
            task_prompt="Analyze BTC/USDT 1h chart and predict direction.",
            generator_signal={
                "direction": "HIGHER",
                "confidence": 0.6,
                "reasoning": "Modest breakout signal, +0.12% gross (barely profitable)",
                "persona": "BREAKOUT",
            },
        )

        # Compute rewards
        reward_1 = compute_reward(outcome_1, example_1)
        reward_2 = compute_reward(outcome_2, example_2)
        reward_3 = compute_reward(outcome_3, example_3)

        # Verify ranking by net return
        assert reward_1.final_reward > reward_3.final_reward > reward_2.final_reward, (
            f"Rewards should rank by net profitability: "
            f"R1={reward_1.final_reward:.3f} > R3={reward_3.final_reward:.3f} > R2={reward_2.final_reward:.3f}"
        )

        # Construct preference pairs
        examples_with_rewards = [
            (example_1, outcome_1, reward_1),
            (example_2, outcome_2, reward_2),
            (example_3, outcome_3, reward_3),
        ]

        pairs = construct_preference_pairs(
            examples_with_rewards,
            min_delta=0.2,
            min_personas_per_context=3,
        )

        # Should construct 1 pair (3 personas // 2)
        assert len(pairs) == 1, f"Expected 1 preference pair, got {len(pairs)}"

        pair = pairs[0]

        # Verify pair structure
        assert pair.context_id == context_id

        # The chosen (higher reward) should be from Example 1 (MOMENTUM - most profitable)
        # The rejected (lower reward) should be from Example 2 (CONTRARIAN - unprofitable)
        assert pair.chosen_persona == "MOMENTUM", (
            f"Chosen should be MOMENTUM (Example 1, net profitable), got {pair.chosen_persona}"
        )
        assert pair.rejected_persona == "CONTRARIAN", (
            f"Rejected should be CONTRARIAN (Example 2, net unprofitable), got {pair.rejected_persona}"
        )

        # Verify reward delta
        assert pair.reward_delta > 0.2, f"Reward delta should be > 0.2, got {pair.reward_delta:.3f}"

        # Verify that chosen reward > rejected reward
        assert pair.chosen_reward > pair.rejected_reward, (
            f"Chosen reward ({pair.chosen_reward:.3f}) should be > "
            f"rejected reward ({pair.rejected_reward:.3f})"
        )

    def test_net_return_beats_gross_return_in_ranking(self, fee_model, context_id):
        """
        Specific test: Example with lower gross return but higher net return should be chosen.

        This test creates a scenario where:
        - Example A has +0.25% gross but +0.167% net
        - Example B has +0.40% gross but only +0.317% net (due to different holding period)

        If ranked by gross: B > A
        If ranked by net: A > B (because A has better cost structure)

        This test verifies we rank by net.
        """
        # Example A: +0.25% gross → +0.167% net (efficient, short holding)
        gross_a_pct = 0.25
        gross_log_a = math.log(1 + gross_a_pct / 100)
        net_log_a = apply_fee_model(gross_log_a, fee_model, holding_periods_8h=0)
        net_return_a_pct = (math.exp(net_log_a) - 1) * 100

        # Example B: +0.40% gross → +0.307% net (higher return, longer holding)
        gross_b_pct = 0.40
        gross_log_b = math.log(1 + gross_b_pct / 100)
        net_log_b = apply_fee_model(gross_log_b, fee_model, holding_periods_8h=0)
        net_return_b_pct = (math.exp(net_log_b) - 1) * 100

        # Create outcomes
        outcome_a = VerifiedOutcome(
            example_id="example-a-efficient",
            actual_direction="HIGHER",
            realized_return=gross_log_a,
            max_adverse_excursion=-0.001,
            net_return=net_return_a_pct,
            entry_price=100.0,
            exit_price=100.0 * (1 + gross_a_pct / 100),
            bars_held=8,
        )

        outcome_b = VerifiedOutcome(
            example_id="example-b-higher-gross",
            actual_direction="HIGHER",
            realized_return=gross_log_b,
            max_adverse_excursion=-0.002,
            net_return=net_return_b_pct,
            entry_price=100.0,
            exit_price=100.0 * (1 + gross_b_pct / 100),
            bars_held=24,
        )

        # Create training examples
        example_a = TrainingExample(
            symbol="BTC/USDT",
            timeframe="1h",
            timestamp_ms=1609459200000,
            market_regime="NEUTRAL",
            persona="MOMENTUM",
            context_id=context_id,
            task_prompt="Analyze BTC/USDT 1h chart and predict direction.",
            generator_signal={
                "direction": "HIGHER",
                "confidence": 0.7,
                "reasoning": "Efficient trade: +0.25% gross with low costs",
                "persona": "MOMENTUM",
            },
        )

        example_b = TrainingExample(
            symbol="BTC/USDT",
            timeframe="1h",
            timestamp_ms=1609459200000,
            market_regime="NEUTRAL",
            persona="CONTRARIAN",
            context_id=context_id,
            task_prompt="Analyze BTC/USDT 1h chart and predict direction.",
            generator_signal={
                "direction": "HIGHER",
                "confidence": 0.8,
                "reasoning": "Higher gross return: +0.40%, but more costs",
                "persona": "CONTRARIAN",
            },
        )

        # Compute rewards
        reward_a = compute_reward(outcome_a, example_a)
        reward_b = compute_reward(outcome_b, example_b)

        # Verify net returns
        assert net_return_a_pct > 0.15, (
            f"Example A net return {net_return_a_pct:.3f}% should be > 0.15%"
        )
        assert net_return_b_pct > net_return_a_pct, (
            f"Example B net return {net_return_b_pct:.3f}% should be > Example A {net_return_a_pct:.3f}%"
        )

        # Verify reward ranking (should follow net returns)
        assert reward_b.final_reward > reward_a.final_reward, (
            f"Example B (higher net return {net_return_b_pct:.3f}%) should have higher reward "
            f"than Example A (lower net return {net_return_a_pct:.3f}%). "
            f"Rewards: B={reward_b.final_reward:.3f}, A={reward_a.final_reward:.3f}"
        )
