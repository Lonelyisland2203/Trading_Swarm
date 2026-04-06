"""Tests for swarm workflow orchestrator."""

import pandas as pd
import pytest

from data.regime_filter import MarketRegime
from swarm.orchestrator import (
    ACCEPTANCE_THRESHOLDS,
    compute_final_confidence,
    should_accept_signal,
)


class TestAcceptanceThresholds:
    """Test regime-aware acceptance thresholds."""

    def test_risk_off_highest_threshold(self):
        """Test that RISK_OFF has highest threshold."""
        risk_off = ACCEPTANCE_THRESHOLDS[MarketRegime.RISK_OFF]
        neutral = ACCEPTANCE_THRESHOLDS[MarketRegime.NEUTRAL]
        risk_on = ACCEPTANCE_THRESHOLDS[MarketRegime.RISK_ON]

        assert risk_off > neutral > risk_on

    def test_thresholds_in_valid_range(self):
        """Test that all thresholds are between 0 and 1."""
        for threshold in ACCEPTANCE_THRESHOLDS.values():
            assert 0.0 <= threshold <= 1.0


class TestShouldAcceptSignal:
    """Test signal acceptance logic."""

    def test_accept_high_score_accept_recommendation(self):
        """Test acceptance when score and recommendation align (ACCEPT)."""
        critique = {
            "score": 0.8,
            "recommendation": "ACCEPT",
            "technical_alignment": 0.7,
        }

        accepted, reason = should_accept_signal(critique, MarketRegime.NEUTRAL)

        assert accepted
        assert "0.8" in reason or "0.80" in reason  # Score mentioned in reason

    def test_reject_on_reject_recommendation(self):
        """Test rejection when recommendation is REJECT."""
        critique = {
            "score": 0.8,  # High score doesn't matter
            "recommendation": "REJECT",
            "technical_alignment": 0.7,
        }

        accepted, reason = should_accept_signal(critique, MarketRegime.NEUTRAL)

        assert not accepted
        assert "REJECT" in reason

    def test_reject_on_low_score(self):
        """Test rejection when score below threshold."""
        critique = {
            "score": 0.5,  # Below 0.65 threshold for NEUTRAL
            "recommendation": "ACCEPT",
            "technical_alignment": 0.6,
        }

        accepted, reason = should_accept_signal(critique, MarketRegime.NEUTRAL)

        assert not accepted
        assert "below threshold" in reason.lower()

    def test_reject_on_low_technical_alignment(self):
        """Test rejection when technical_alignment < 0.4."""
        critique = {
            "score": 0.8,  # High score
            "recommendation": "ACCEPT",
            "technical_alignment": 0.3,  # Below 0.4 gate
        }

        accepted, reason = should_accept_signal(critique, MarketRegime.NEUTRAL)

        assert not accepted
        assert "technical alignment" in reason.lower()

    def test_regime_aware_threshold_risk_off(self):
        """Test that RISK_OFF uses higher threshold."""
        critique = {
            "score": 0.70,  # Would pass NEUTRAL (0.65) but not RISK_OFF (0.75)
            "recommendation": "ACCEPT",
            "technical_alignment": 0.6,
        }

        # Should pass NEUTRAL
        accepted_neutral, _ = should_accept_signal(critique, MarketRegime.NEUTRAL)
        assert accepted_neutral

        # Should fail RISK_OFF
        accepted_risk_off, _ = should_accept_signal(critique, MarketRegime.RISK_OFF)
        assert not accepted_risk_off

    def test_regime_aware_threshold_risk_on(self):
        """Test that RISK_ON uses lower threshold."""
        critique = {
            "score": 0.62,  # Would fail NEUTRAL (0.65) but pass RISK_ON (0.60)
            "recommendation": "ACCEPT",
            "technical_alignment": 0.5,
        }

        # Should fail NEUTRAL
        accepted_neutral, _ = should_accept_signal(critique, MarketRegime.NEUTRAL)
        assert not accepted_neutral

        # Should pass RISK_ON
        accepted_risk_on, _ = should_accept_signal(critique, MarketRegime.RISK_ON)
        assert accepted_risk_on

    def test_override_threshold(self):
        """Test that override_threshold parameter works."""
        critique = {
            "score": 0.55,
            "recommendation": "ACCEPT",
            "technical_alignment": 0.6,
        }

        # With override threshold of 0.5, should pass
        accepted, _ = should_accept_signal(critique, MarketRegime.NEUTRAL, override_threshold=0.5)
        assert accepted

        # With override threshold of 0.6, should fail
        rejected, _ = should_accept_signal(critique, MarketRegime.NEUTRAL, override_threshold=0.6)
        assert not rejected


class TestComputeFinalConfidence:
    """Test confidence blending logic."""

    def test_both_high_confidence(self):
        """Test blending when both generator and critic are confident."""
        final = compute_final_confidence(
            generator_confidence=0.8,
            critic_score=0.9,
        )

        # Should be high (weighted toward critic at alpha=0.6)
        expected = 0.6 * 0.9 + 0.4 * 0.8
        assert abs(final - expected) < 0.001

    def test_generator_high_critic_low(self):
        """Test that low critic score reduces final confidence."""
        final = compute_final_confidence(
            generator_confidence=0.9,
            critic_score=0.3,
        )

        # Should be reduced due to low critic score
        assert final < 0.9
        expected = 0.6 * 0.3 + 0.4 * 0.9
        assert abs(final - expected) < 0.001

    def test_generator_low_critic_high(self):
        """Test that high critic score boosts confidence."""
        final = compute_final_confidence(
            generator_confidence=0.4,
            critic_score=0.9,
        )

        # Should be boosted by high critic score
        assert final > 0.4
        expected = 0.6 * 0.9 + 0.4 * 0.4
        assert abs(final - expected) < 0.001

    def test_custom_alpha(self):
        """Test custom alpha parameter."""
        # Alpha = 0.5 means equal weight
        final = compute_final_confidence(
            generator_confidence=0.6,
            critic_score=0.8,
            alpha=0.5,
        )

        expected = 0.5 * 0.8 + 0.5 * 0.6
        assert abs(final - expected) < 0.001

    def test_alpha_zero_uses_only_generator(self):
        """Test alpha=0 means only generator confidence."""
        final = compute_final_confidence(
            generator_confidence=0.7,
            critic_score=0.3,
            alpha=0.0,
        )

        assert abs(final - 0.7) < 0.001

    def test_alpha_one_uses_only_critic(self):
        """Test alpha=1 means only critic score."""
        final = compute_final_confidence(
            generator_confidence=0.3,
            critic_score=0.9,
            alpha=1.0,
        )

        assert abs(final - 0.9) < 0.001


class TestMarketContextBuilding:
    """Test market context serialization."""

    def test_build_market_context_includes_indicators(self):
        """Test that market context includes all required indicators."""
        from swarm.orchestrator import _build_market_context

        # Create sample OHLCV data
        df = pd.DataFrame({
            "timestamp": [1704067200000 + i * 3600000 for i in range(100)],
            "open": [100.0 + i * 0.1 for i in range(100)],
            "high": [102.0 + i * 0.1 for i in range(100)],
            "low": [98.0 + i * 0.1 for i in range(100)],
            "close": [101.0 + i * 0.1 for i in range(100)],
            "volume": [1000.0] * 100,
        })

        context = _build_market_context(df, MarketRegime.NEUTRAL)

        # Check required fields
        assert "current_price" in context
        assert "regime" in context
        assert "rsi" in context
        assert "macd" in context
        assert "macd_signal" in context
        assert "bb_position" in context
        assert "recent_ohlcv" in context

    def test_build_market_context_serializable(self):
        """Test that market context is JSON-serializable."""
        import json
        from swarm.orchestrator import _build_market_context

        df = pd.DataFrame({
            "timestamp": [1704067200000 + i * 3600000 for i in range(100)],
            "open": [100.0] * 100,
            "high": [102.0] * 100,
            "low": [98.0] * 100,
            "close": [101.0] * 100,
            "volume": [1000.0] * 100,
        })

        context = _build_market_context(df, MarketRegime.RISK_OFF)

        # Should be serializable
        json_str = json.dumps(context)
        assert len(json_str) > 0

        # Should be deserializable
        recovered = json.loads(json_str)
        assert recovered["regime"] == "risk_off"  # Enum value is lowercase

    def test_build_market_context_recent_ohlcv_has_5_bars(self):
        """Test that recent_ohlcv includes exactly 5 bars."""
        from swarm.orchestrator import _build_market_context

        df = pd.DataFrame({
            "timestamp": [1704067200000 + i * 3600000 for i in range(100)],
            "open": [100.0] * 100,
            "high": [102.0] * 100,
            "low": [98.0] * 100,
            "close": [101.0] * 100,
            "volume": [1000.0] * 100,
        })

        context = _build_market_context(df, MarketRegime.NEUTRAL)

        # Count lines in recent_ohlcv (should be 5)
        ohlcv_lines = context["recent_ohlcv"].strip().split("\n")
        assert len(ohlcv_lines) == 5


class TestTrainingCapture:
    """Test training example integration."""

    def test_training_example_version(self):
        """Test that training examples have version field."""
        from swarm.training_capture import TrainingExample, TRAINING_EXAMPLE_VERSION

        example = TrainingExample()
        assert example.version == TRAINING_EXAMPLE_VERSION

    def test_training_example_serialization(self):
        """Test that training examples are serializable."""
        import json
        from swarm.training_capture import TrainingExample

        example = TrainingExample(
            symbol="BTC/USDT",
            timeframe="1h",
            timestamp_ms=1704067200000,
            market_regime="NEUTRAL",
        )

        # Should serialize to dict
        data = example.to_dict()
        assert isinstance(data, dict)
        assert data["symbol"] == "BTC/USDT"

        # Should be JSON-serializable
        json_str = json.dumps(data)
        assert len(json_str) > 0

    def test_filter_by_acceptance(self):
        """Test filtering training examples by acceptance status."""
        from swarm.training_capture import TrainingExample, filter_by_acceptance

        examples = [
            TrainingExample(was_accepted=True, symbol="A"),
            TrainingExample(was_accepted=False, symbol="B"),
            TrainingExample(was_accepted=True, symbol="C"),
        ]

        accepted = filter_by_acceptance(examples, accepted_only=True)
        assert len(accepted) == 2
        assert all(ex.was_accepted for ex in accepted)

        rejected = filter_by_acceptance(examples, rejected_only=True)
        assert len(rejected) == 1
        assert not rejected[0].was_accepted

    def test_filter_by_regime(self):
        """Test filtering training examples by regime."""
        from swarm.training_capture import TrainingExample, filter_by_regime

        examples = [
            TrainingExample(market_regime="RISK_OFF"),
            TrainingExample(market_regime="NEUTRAL"),
            TrainingExample(market_regime="RISK_OFF"),
        ]

        risk_off = filter_by_regime(examples, "RISK_OFF")
        assert len(risk_off) == 2
        assert all(ex.market_regime == "RISK_OFF" for ex in risk_off)

    def test_filter_complete(self):
        """Test filtering to examples with ground truth."""
        from swarm.training_capture import TrainingExample, filter_complete

        examples = [
            TrainingExample(actual_direction="HIGHER", realized_return=0.05),
            TrainingExample(actual_direction=None, realized_return=None),
            TrainingExample(actual_direction="LOWER", realized_return=-0.02),
        ]

        complete = filter_complete(examples)
        assert len(complete) == 2
        assert all(ex.actual_direction is not None for ex in complete)
        assert all(ex.realized_return is not None for ex in complete)


@pytest.mark.asyncio
async def test_run_swarm_workflow_accepts_higher_tf_data():
    """Should accept and pass through higher_tf_data to PromptBuilder."""
    from tests.fixtures.timeframe_fixtures import create_test_df_bullish
    from swarm.orchestrator import run_swarm_workflow
    from data.prompt_builder import TaskType

    # Create test data
    df = create_test_df_bullish(bars=100)
    higher_tf_data = {
        "4h": create_test_df_bullish(bars=100),
        "1d": create_test_df_bullish(bars=100),
    }

    state, example = await run_swarm_workflow(
        symbol="BTC/USDT",
        timeframe="1h",
        ohlcv_df=df,
        market_regime=MarketRegime.NEUTRAL,
        task_prompt="Test prompt",
        task_type=TaskType.PREDICT_DIRECTION,
        higher_tf_data=higher_tf_data,
    )

    # Verify higher TF context appears in task_prompt
    assert "Higher Timeframe Context" in state["task_prompt"]


@pytest.mark.asyncio
async def test_run_swarm_workflow_backward_compatible():
    """Should work without higher_tf_data parameter (backward compatibility)."""
    from tests.fixtures.timeframe_fixtures import create_test_df_bullish
    from swarm.orchestrator import run_swarm_workflow
    from data.prompt_builder import TaskType

    df = create_test_df_bullish(bars=100)

    # Call without higher_tf_data
    state, example = await run_swarm_workflow(
        symbol="BTC/USDT",
        timeframe="1h",
        ohlcv_df=df,
        market_regime=MarketRegime.NEUTRAL,
        task_prompt="Test prompt",
        task_type=TaskType.PREDICT_DIRECTION,
    )

    # Should not have higher TF context
    assert "Higher Timeframe Context" not in state["task_prompt"]
