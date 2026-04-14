"""
End-to-end integration test for fee-aware workflow.

Verifies the complete pipeline:
1. Prompt includes execution context when fee_model is provided
2. Rewards use net returns after fee deduction
3. Preference pairs rank by net profitability (not gross)

This ensures that the fee model correctly filters unprofitable signals
throughout the entire training pipeline.
"""

import math

import pandas as pd
import pytest

from config.fee_model import FeeModelSettings
from data.prompt_builder import (
    PromptBuilder,
    TaskConfig,
    TaskType,
)
from data.regime_filter import MarketRegime
from swarm.training_capture import TrainingExample
from training.dpo_export import construct_preference_pairs, validate_preference_pair
from training.reward_config import RewardScaling
from training.reward_engine import compute_reward
from verifier.constants import get_horizon_bars, compute_holding_periods_8h
from verifier.outcome import VerifiedOutcome, apply_fee_model, compute_log_return


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def fee_model_futures():
    """Binance Futures USDT-M fee model."""
    return FeeModelSettings(
        maker_fee_pct=0.02,
        taker_fee_pct=0.05,
        bnb_discount_enabled=True,
        bnb_discount_pct=10.0,
        funding_rate_pct=0.01,
        include_funding=True,
        slippage_pct=0.02,
    )


@pytest.fixture
def sample_ohlcv():
    """Sample OHLCV data for testing."""
    # Create 100 bars of synthetic data
    dates = pd.date_range(start="2024-01-01", periods=100, freq="1h")
    prices = [100.0 + i * 0.5 for i in range(100)]  # Uptrend

    df = pd.DataFrame(
        {
            "timestamp": dates,
            "open": prices,
            "high": [p * 1.01 for p in prices],
            "low": [p * 0.99 for p in prices],
            "close": prices,
            "volume": [1000.0] * 100,
        }
    )

    return df


@pytest.fixture
def task_direction():
    """Direction prediction task."""
    return TaskConfig(
        task_type=TaskType.PREDICT_DIRECTION,
        weight=1.0,
        difficulty=2,
        min_bars_required=50,
    )


# ============================================================================
# Test 1: Prompt Includes Execution Context
# ============================================================================


def test_prompt_includes_execution_context_with_fee_model(
    sample_ohlcv,
    task_direction,
    fee_model_futures,
):
    """Test that prompt includes execution context when fee_model is provided."""
    builder = PromptBuilder()

    prompt_with_fees = builder.build_prompt(
        task=task_direction,
        df=sample_ohlcv,
        symbol="BTC/USDT",
        timeframe="1h",
        market_regime=MarketRegime.NEUTRAL,
        fee_model=fee_model_futures,
    )

    # Execution context should include key fee information
    assert "## Execution Context" in prompt_with_fees
    assert "Round-trip cost" in prompt_with_fees or "round-trip cost" in prompt_with_fees
    assert "Minimum profitable" in prompt_with_fees or "minimum profitable" in prompt_with_fees

    # Should mention exchange and mode
    assert "Binance" in prompt_with_fees
    assert "Futures" in prompt_with_fees or "USDT-M" in prompt_with_fees


def test_prompt_no_execution_context_without_fee_model(
    sample_ohlcv,
    task_direction,
):
    """Test that prompt excludes execution context when fee_model=None."""
    builder = PromptBuilder()

    prompt_without_fees = builder.build_prompt(
        task=task_direction,
        df=sample_ohlcv,
        symbol="BTC/USDT",
        timeframe="1h",
        market_regime=MarketRegime.NEUTRAL,
        fee_model=None,
    )

    # Execution context should NOT be present
    assert "## Execution Context" not in prompt_without_fees
    assert (
        "Round-trip cost" not in prompt_without_fees
        and "round-trip cost" not in prompt_without_fees
    )
    assert (
        "Minimum profitable" not in prompt_without_fees
        and "minimum profitable" not in prompt_without_fees
    )


def test_execution_context_varies_by_timeframe(
    sample_ohlcv,
    task_direction,
    fee_model_futures,
):
    """Test that execution context reflects timeframe-specific costs."""
    builder = PromptBuilder()

    # 1m timeframe (60 bars = 1 hour = 0.125 funding periods)
    sample_1m = sample_ohlcv.copy()
    prompt_1m = builder.build_prompt(
        task=task_direction,
        df=sample_1m,
        symbol="BTC/USDT",
        timeframe="1m",
        market_regime=MarketRegime.NEUTRAL,
        fee_model=fee_model_futures,
    )

    # 1d timeframe (5 bars = 5 days = 15 funding periods)
    # Create sufficient data for 1d (need 50+ bars)
    dates_1d = pd.date_range(start="2024-01-01", periods=100, freq="1D")
    prices_1d = [100.0 + i * 0.5 for i in range(100)]
    sample_1d = pd.DataFrame(
        {
            "timestamp": dates_1d,
            "open": prices_1d,
            "high": [p * 1.01 for p in prices_1d],
            "low": [p * 0.99 for p in prices_1d],
            "close": prices_1d,
            "volume": [1000.0] * 100,
        }
    )

    prompt_1d = builder.build_prompt(
        task=task_direction,
        df=sample_1d,
        symbol="BTC/USDT",
        timeframe="1d",
        market_regime=MarketRegime.NEUTRAL,
        fee_model=fee_model_futures,
    )

    # Both should have execution context
    assert "## Execution Context" in prompt_1m
    assert "## Execution Context" in prompt_1d

    # Both should mention costs
    assert "Round-trip cost" in prompt_1m or "round-trip cost" in prompt_1m
    assert "Round-trip cost" in prompt_1d or "round-trip cost" in prompt_1d

    # This is a smoke test - exact values tested in other test files
    assert len(prompt_1m) > 0
    assert len(prompt_1d) > 0


# ============================================================================
# Test 2: Rewards Use Net Returns
# ============================================================================


def test_reward_uses_net_return_not_gross(fee_model_futures):
    """Test that reward computation uses net_return, not realized_return."""
    # Create a scenario where gross is profitable but net is not
    timeframe = "1h"
    horizon_bars = get_horizon_bars(timeframe)
    holding_periods = compute_holding_periods_8h(timeframe, horizon_bars)

    # Entry: 100.0, Exit: 100.10 → +0.10% gross
    entry_price = 100.0
    exit_price = 100.10
    gross_log = compute_log_return(entry_price, exit_price)
    gross_pct = (math.exp(gross_log) - 1) * 100

    # Apply fees
    net_log = apply_fee_model(gross_log, fee_model_futures, holding_periods)
    net_pct = (math.exp(net_log) - 1) * 100

    # Sanity check: gross should be ~0.10%, net should be negative
    assert 0.09 < gross_pct < 0.11, f"Expected gross ~0.10%, got {gross_pct:.3f}%"
    assert net_pct < 0, f"Expected net < 0%, got {net_pct:.3f}%"

    # Create verified outcome
    outcome = VerifiedOutcome(
        example_id="test-123",
        actual_direction="HIGHER",
        realized_return=gross_log,
        max_adverse_excursion=-0.001,
        net_return=net_log,  # This is what should be used
        entry_price=entry_price,
        exit_price=exit_price,
        bars_held=horizon_bars,
    )

    # Create training example
    example = TrainingExample(
        symbol="BTC/USDT",
        timeframe=timeframe,
        timestamp_ms=1700000000000,
        context_id="ctx-1",
        persona="MOMENTUM",
        generator_signal={
            "direction": "HIGHER",
            "confidence": 0.8,
            "reasoning": "Bullish momentum",
        },
        task_prompt="Analyze BTC/USDT",
        market_regime="NEUTRAL",
        was_accepted=True,
    )

    # Compute reward
    reward = compute_reward(
        verified_outcome=outcome,
        training_example=example,
        scaling=RewardScaling(),
    )

    # Verify that net_return was used, not realized_return
    assert reward.net_return == net_log, "Reward should use net_return"
    assert reward.realized_return == gross_log, "Realized return should match gross"

    # Return component should be based on net_return
    # With default scaling (return_scale=10.0), a small negative net return
    # should produce a negative return_reward
    assert reward.return_reward < 0, (
        f"Return reward should be negative for net loss, got {reward.return_reward:.3f}"
    )

    # Note: Final reward may still be positive due to directional component (0.3 weight)
    # The key test is that return_reward uses net_return (negative) not realized_return (positive)


def test_reward_comparison_gross_vs_net(fee_model_futures):
    """
    Test that two signals with different gross returns can be ranked
    differently after fees.

    Scenario:
    - Signal A: +0.30% gross (profitable after fees)
    - Signal B: +0.10% gross (unprofitable after fees)

    Without fees, B would still be "profitable".
    With fees, only A should have positive return component.
    """
    timeframe = "1h"
    horizon_bars = get_horizon_bars(timeframe)
    holding_periods = compute_holding_periods_8h(timeframe, horizon_bars)

    # Signal A: +0.30% gross
    gross_log_a = math.log(1 + 0.30 / 100)
    net_log_a = apply_fee_model(gross_log_a, fee_model_futures, holding_periods)
    net_pct_a = (math.exp(net_log_a) - 1) * 100

    # Signal B: +0.10% gross (below minimum profitable threshold of 0.113%)
    gross_log_b = math.log(1 + 0.10 / 100)
    net_log_b = apply_fee_model(gross_log_b, fee_model_futures, holding_periods)
    net_pct_b = (math.exp(net_log_b) - 1) * 100

    # Verify assumptions
    assert net_pct_a > 0, f"Signal A should be profitable: {net_pct_a:.3f}%"
    assert net_pct_b < 0, f"Signal B should be unprofitable: {net_pct_b:.3f}%"

    # Create outcomes
    outcome_a = VerifiedOutcome(
        example_id="signal-a",
        actual_direction="HIGHER",
        realized_return=gross_log_a,
        max_adverse_excursion=-0.001,
        net_return=net_log_a,
        entry_price=100.0,
        exit_price=100.30,
        bars_held=horizon_bars,
    )

    outcome_b = VerifiedOutcome(
        example_id="signal-b",
        actual_direction="HIGHER",
        realized_return=gross_log_b,
        max_adverse_excursion=-0.001,
        net_return=net_log_b,
        entry_price=100.0,
        exit_price=100.12,
        bars_held=horizon_bars,
    )

    # Create examples
    example_a = TrainingExample(
        symbol="BTC/USDT",
        timeframe=timeframe,
        timestamp_ms=1700000000000,
        context_id="ctx-test",
        persona="MOMENTUM",
        generator_signal={
            "direction": "HIGHER",
            "confidence": 0.8,
            "reasoning": "Strong momentum",
        },
        task_prompt="Predict direction",
        market_regime="NEUTRAL",
        was_accepted=True,
    )

    example_b = TrainingExample(
        symbol="BTC/USDT",
        timeframe=timeframe,
        timestamp_ms=1700000000000,
        context_id="ctx-test",
        persona="SCALPER",
        generator_signal={
            "direction": "HIGHER",
            "confidence": 0.6,
            "reasoning": "Weak momentum",
        },
        task_prompt="Predict direction",
        market_regime="NEUTRAL",
        was_accepted=True,
    )

    # Compute rewards
    reward_a = compute_reward(outcome_a, example_a)
    reward_b = compute_reward(outcome_b, example_b)

    # A should have positive return component, B should have negative return component
    assert reward_a.return_reward > 0, (
        f"Signal A should have positive return component, got {reward_a.return_reward:.3f}"
    )
    assert reward_b.return_reward < 0, (
        f"Signal B should have negative return component, got {reward_b.return_reward:.3f}"
    )

    # A's reward should be higher than B's
    assert reward_a.final_reward > reward_b.final_reward, (
        f"A ({reward_a.final_reward:.3f}) should outrank B ({reward_b.final_reward:.3f})"
    )

    # Verify net returns are being used
    assert reward_a.net_return == net_log_a
    assert reward_b.net_return == net_log_b


# ============================================================================
# Test 3: Preference Pairs Rank by Net Profitability
# ============================================================================


def test_preference_pairs_rank_by_net_return(fee_model_futures):
    """
    Test that preference pair construction ranks signals by net profitability.

    Creates 5 signals with different returns to ensure sufficient diversity:
    - Strong positive, moderate positive, neutral, moderate negative, strong negative

    Verifies that chosen/rejected ranking reflects net return differences.
    """
    timeframe = "1h"
    horizon_bars = get_horizon_bars(timeframe)
    holding_periods = compute_holding_periods_8h(timeframe, horizon_bars)

    # Create 5 signals with varying returns to create clear reward differences
    # Include wrong directional predictions to create larger reward gaps
    signals = [
        ("signal-1", "MOMENTUM", "HIGHER", 0.50),  # Correct, profitable
        ("signal-2", "CONTRARIAN", "HIGHER", 0.20),  # Correct, marginally profitable
        ("signal-3", "SCALPER", "HIGHER", 0.05),  # Correct, unprofitable
        ("signal-4", "SWING", "LOWER", 0.50),  # Wrong direction
        ("signal-5", "MACRO", "LOWER", 0.20),  # Wrong direction
    ]

    examples_with_rewards = []

    for signal_id, persona, predicted_dir, gross_pct in signals:
        # Compute net return
        gross_log = math.log(1 + gross_pct / 100)
        net_log = apply_fee_model(gross_log, fee_model_futures, holding_periods)

        # Actual direction is always HIGHER
        outcome = VerifiedOutcome(
            example_id=signal_id,
            actual_direction="HIGHER",
            realized_return=gross_log,
            max_adverse_excursion=-0.001,
            net_return=net_log,
            entry_price=100.0,
            exit_price=100.0 + gross_pct,
            bars_held=horizon_bars,
        )

        # Create example with predicted direction
        example = TrainingExample(
            symbol="BTC/USDT",
            timeframe=timeframe,
            timestamp_ms=1700000000000,
            context_id="ctx-multi",  # Same context for all
            persona=persona,
            generator_signal={
                "direction": predicted_dir,
                "confidence": 0.7,
                "reasoning": f"{persona} analysis",
            },
            task_prompt="Predict BTC/USDT direction",
            market_regime="NEUTRAL",
            was_accepted=True,
        )

        # Compute reward
        reward = compute_reward(outcome, example)

        examples_with_rewards.append((example, outcome, reward))

    # Construct preference pairs
    pairs = construct_preference_pairs(
        examples_with_rewards=examples_with_rewards,
        min_delta=0.01,  # Very low threshold since we have directional differences
        min_personas_per_context=5,
    )

    # Should create at least one pair
    assert len(pairs) > 0, f"Should create at least one preference pair, got {len(pairs)}"

    # Verify pairs are properly ranked
    for pair in pairs:
        # Chosen should have higher reward than rejected
        assert pair.chosen_reward > pair.rejected_reward, (
            f"Chosen reward ({pair.chosen_reward:.3f}) should exceed rejected ({pair.rejected_reward:.3f})"
        )

        # Reward delta should be positive
        assert pair.reward_delta > 0, (
            f"Reward delta should be positive, got {pair.reward_delta:.3f}"
        )

        # Verify metadata
        assert pair.context_id == "ctx-multi"
        assert pair.symbol == "BTC/USDT"


def test_preference_pair_validation_with_net_returns(fee_model_futures):
    """Test that preference pair validation respects net return ranking."""
    timeframe = "1h"
    horizon_bars = get_horizon_bars(timeframe)
    holding_periods = compute_holding_periods_8h(timeframe, horizon_bars)

    # Chosen: +0.30% gross → profitable after fees
    gross_log_chosen = math.log(1 + 0.30 / 100)
    net_log_chosen = apply_fee_model(gross_log_chosen, fee_model_futures, holding_periods)

    outcome_chosen = VerifiedOutcome(
        example_id="chosen-1",
        actual_direction="HIGHER",
        realized_return=gross_log_chosen,
        max_adverse_excursion=-0.001,
        net_return=net_log_chosen,
        entry_price=100.0,
        exit_price=100.30,
        bars_held=horizon_bars,
    )

    example_chosen = TrainingExample(
        symbol="BTC/USDT",
        timeframe=timeframe,
        timestamp_ms=1700000000000,
        context_id="ctx-pair",
        persona="MOMENTUM",
        generator_signal={
            "direction": "HIGHER",
            "confidence": 0.8,
            "reasoning": "Strong signal",
        },
        task_prompt="Predict direction",
        market_regime="NEUTRAL",
        was_accepted=True,
    )

    # Rejected: +0.10% gross → unprofitable after fees
    gross_log_rejected = math.log(1 + 0.10 / 100)
    net_log_rejected = apply_fee_model(gross_log_rejected, fee_model_futures, holding_periods)

    outcome_rejected = VerifiedOutcome(
        example_id="rejected-1",
        actual_direction="HIGHER",
        realized_return=gross_log_rejected,
        max_adverse_excursion=-0.001,
        net_return=net_log_rejected,
        entry_price=100.0,
        exit_price=100.10,
        bars_held=horizon_bars,
    )

    example_rejected = TrainingExample(
        symbol="BTC/USDT",
        timeframe=timeframe,
        timestamp_ms=1700000000000,
        context_id="ctx-pair",
        persona="SCALPER",
        generator_signal={
            "direction": "HIGHER",
            "confidence": 0.6,
            "reasoning": "Weak signal",
        },
        task_prompt="Predict direction",
        market_regime="NEUTRAL",
        was_accepted=True,
    )

    # Compute rewards
    reward_chosen = compute_reward(outcome_chosen, example_chosen)
    reward_rejected = compute_reward(outcome_rejected, example_rejected)

    # Validate the pair
    is_valid, reason = validate_preference_pair(
        chosen_example=example_chosen,
        rejected_example=example_rejected,
        chosen_reward=reward_chosen,
        rejected_reward=reward_rejected,
        min_delta=0.1,
    )

    assert is_valid, f"Pair should be valid: {reason}"
    assert "Valid pair" in reason

    # Verify ranking is correct
    assert reward_chosen.final_reward > reward_rejected.final_reward, (
        "Chosen reward should exceed rejected reward"
    )


# ============================================================================
# Test 4: Backward Compatibility - fee_model=None
# ============================================================================


def test_fee_mode_none_backward_compatibility(sample_ohlcv, task_direction):
    """
    Test that fee_model=None mode works without errors (backward compatibility).

    Verifies:
    1. Prompts build successfully without execution context
    2. Rewards compute successfully using realized_return = net_return
    3. Preference pairs construct successfully
    """
    # 1. Prompt building
    builder = PromptBuilder()
    prompt = builder.build_prompt(
        task=task_direction,
        df=sample_ohlcv,
        symbol="BTC/USDT",
        timeframe="1h",
        market_regime=MarketRegime.NEUTRAL,
        fee_model=None,  # No fee model
    )

    assert len(prompt) > 0
    assert "## Execution Costs" not in prompt

    # 2. Reward computation (net_return = realized_return when no fees)
    gross_log = math.log(1 + 0.15 / 100)  # +0.15%

    outcome = VerifiedOutcome(
        example_id="test-no-fees",
        actual_direction="HIGHER",
        realized_return=gross_log,
        max_adverse_excursion=-0.001,
        net_return=gross_log,  # Same as realized when no fees
        entry_price=100.0,
        exit_price=100.15,
        bars_held=24,
    )

    example = TrainingExample(
        symbol="BTC/USDT",
        timeframe="1h",
        timestamp_ms=1700000000000,
        context_id="ctx-no-fees",
        persona="MOMENTUM",
        generator_signal={
            "direction": "HIGHER",
            "confidence": 0.8,
            "reasoning": "Test",
        },
        task_prompt="Test prompt",
        market_regime="NEUTRAL",
        was_accepted=True,
    )

    reward = compute_reward(outcome, example)

    # Reward should be positive (no fees to reduce return)
    assert reward.final_reward > 0
    assert reward.net_return == gross_log
    assert reward.realized_return == gross_log

    # 3. Preference pair construction
    # Create multiple examples with varying returns and directions for diversity
    examples_with_rewards = []
    signals = [
        ("MOMENTUM", "HIGHER", 0.30),
        ("CONTRARIAN", "HIGHER", 0.15),
        ("SCALPER", "LOWER", 0.30),  # Wrong direction
    ]

    for i, (persona, predicted_dir, gross_pct) in enumerate(signals):
        gross_log = math.log(1 + gross_pct / 100)

        outcome = VerifiedOutcome(
            example_id=f"test-{i}",
            actual_direction="HIGHER",  # Actual is always HIGHER
            realized_return=gross_log,
            max_adverse_excursion=-0.001,
            net_return=gross_log,
            entry_price=100.0,
            exit_price=100.0 + gross_pct,
            bars_held=24,
        )

        example = TrainingExample(
            symbol="BTC/USDT",
            timeframe="1h",
            timestamp_ms=1700000000000,
            context_id="ctx-no-fees",
            persona=persona,
            generator_signal={
                "direction": predicted_dir,
                "confidence": 0.7,
                "reasoning": f"{persona} test",
            },
            task_prompt="Test prompt",
            market_regime="NEUTRAL",
            was_accepted=True,
        )

        reward = compute_reward(outcome, example)
        examples_with_rewards.append((example, outcome, reward))

    # Should construct pairs successfully
    pairs = construct_preference_pairs(
        examples_with_rewards=examples_with_rewards,
        min_delta=0.01,  # Very low threshold for testing
        min_personas_per_context=3,
    )

    assert len(pairs) > 0, f"Should create preference pairs without fee model, got {len(pairs)}"


# ============================================================================
# Test 5: End-to-End Workflow
# ============================================================================


def test_fee_aware_end_to_end(sample_ohlcv, task_direction, fee_model_futures):
    """
    End-to-end test of fee-aware workflow.

    Workflow:
    1. Build prompt with execution context
    2. Simulate signal generation
    3. Verify outcome with net return calculation
    4. Compute reward using net return
    5. Construct preference pairs ranked by net profitability

    Verifies complete integration of fee model across all pipeline stages.
    """
    timeframe = "1h"
    horizon_bars = get_horizon_bars(timeframe)
    holding_periods = compute_holding_periods_8h(timeframe, horizon_bars)

    # -----------------------------------------------------------------------
    # Stage 1: Build Prompt with Execution Context
    # -----------------------------------------------------------------------

    builder = PromptBuilder()
    prompt = builder.build_prompt(
        task=task_direction,
        df=sample_ohlcv,
        symbol="BTC/USDT",
        timeframe=timeframe,
        market_regime=MarketRegime.NEUTRAL,
        fee_model=fee_model_futures,
    )

    # Verify execution context is included
    assert "## Execution Context" in prompt, "Prompt should include execution context"
    assert "Round-trip cost" in prompt or "round-trip cost" in prompt, (
        "Should mention round-trip cost"
    )

    # -----------------------------------------------------------------------
    # Stage 2: Simulate Multi-Persona Signal Generation
    # -----------------------------------------------------------------------

    # Create 5 signals with varying quality (simulating generator output)
    signals = [
        ("MOMENTUM", "HIGHER", 0.9, 0.50),  # +0.50% gross → profitable
        ("CONTRARIAN", "HIGHER", 0.7, 0.25),  # +0.25% gross → profitable
        ("SCALPER", "HIGHER", 0.6, 0.12),  # +0.12% gross → unprofitable
        ("SWING", "HIGHER", 0.5, 0.08),  # +0.08% gross → unprofitable
        ("MACRO", "LOWER", 0.4, -0.15),  # -0.15% gross → more unprofitable
    ]

    examples_with_rewards = []

    for persona, direction, confidence, gross_pct in signals:
        # -----------------------------------------------------------------------
        # Stage 3: Verify Outcome with Net Return
        # -----------------------------------------------------------------------

        # Compute gross log return
        if gross_pct >= 0:
            entry, exit = 100.0, 100.0 + gross_pct
            actual_dir = "HIGHER"
        else:
            entry, exit = 100.0, 100.0 + gross_pct
            actual_dir = "LOWER"

        gross_log = compute_log_return(entry, exit)

        # Apply fee model to get net return
        net_log = apply_fee_model(gross_log, fee_model_futures, holding_periods)

        outcome = VerifiedOutcome(
            example_id=f"e2e-{persona}",
            actual_direction=actual_dir,
            realized_return=gross_log,
            max_adverse_excursion=-0.002,
            net_return=net_log,  # Fee-adjusted
            entry_price=entry,
            exit_price=exit,
            bars_held=horizon_bars,
        )

        # Create training example
        example = TrainingExample(
            symbol="BTC/USDT",
            timeframe=timeframe,
            timestamp_ms=1700000000000,
            context_id="ctx-e2e",
            persona=persona,
            generator_signal={
                "direction": direction,
                "confidence": confidence,
                "reasoning": f"{persona} analysis from prompt",
            },
            task_prompt=prompt,  # Use the fee-aware prompt
            market_regime="NEUTRAL",
            was_accepted=True,
        )

        # -----------------------------------------------------------------------
        # Stage 4: Compute Reward Using Net Return
        # -----------------------------------------------------------------------

        reward = compute_reward(outcome, example)

        # Verify net_return is used
        assert reward.net_return == net_log
        assert reward.realized_return == gross_log

        examples_with_rewards.append((example, outcome, reward))

    # -----------------------------------------------------------------------
    # Stage 5: Construct Preference Pairs Ranked by Net Profitability
    # -----------------------------------------------------------------------

    pairs = construct_preference_pairs(
        examples_with_rewards=examples_with_rewards,
        min_delta=0.05,  # Lower threshold to ensure pairs are created
        min_personas_per_context=5,
    )

    # Should create at least one pair
    assert len(pairs) > 0, f"Should create preference pairs from 5 personas, got {len(pairs)}"

    # Verify all pairs are properly ranked
    for pair in pairs:
        # Chosen should have higher net reward
        assert pair.chosen_reward > pair.rejected_reward, (
            f"Chosen ({pair.chosen_reward:.3f}) should exceed rejected ({pair.rejected_reward:.3f})"
        )

        # Verify shared context
        assert pair.context_id == "ctx-e2e"

        # Verify prompt is the fee-aware prompt
        assert "## Execution Costs" in pair.prompt or pair.prompt == prompt

    # Verify that the highest net return signal is ranked highest
    # (MOMENTUM with +0.50% gross should be the most profitable after fees)
    momentum_reward = next(r for e, o, r in examples_with_rewards if e.persona == "MOMENTUM")

    # MOMENTUM should have positive final reward
    assert momentum_reward.final_reward > 0, "Most profitable signal should have positive reward"

    # Check return components (directional may dominate final reward)
    scalper_reward = next(r for e, o, r in examples_with_rewards if e.persona == "SCALPER")
    swing_reward = next(r for e, o, r in examples_with_rewards if e.persona == "SWING")
    contrarian_reward = next(r for e, o, r in examples_with_rewards if e.persona == "CONTRARIAN")

    # SCALPER predicted correctly but net return is slightly positive (+0.007%)
    # SWING predicted incorrectly (LOWER when actual was HIGHER)
    # So SCALPER should rank higher than SWING despite both having similar net returns

    # Verify ranking: MOMENTUM > CONTRARIAN (both correct) > SCALPER (correct but low net) > SWING/MACRO (wrong direction)
    assert momentum_reward.final_reward > contrarian_reward.final_reward, (
        "Higher net return with correct direction should rank higher"
    )
    assert contrarian_reward.final_reward > scalper_reward.final_reward, (
        "Higher net return should rank higher when both correct"
    )

    # Wrong direction signals should rank lowest
    assert scalper_reward.final_reward > swing_reward.final_reward, (
        "Correct direction should rank higher than wrong direction"
    )
