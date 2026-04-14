"""
Tests for synthesis node.

TDD: Tests written FIRST before implementation.

Session 17N: Synthesis node combines XGBoost signal + LLM context + critic veto
into final trading decision with position sizing.

Rules from signal-layer.md:
- XGBoost prob < 0.55 → FLAT (no trade regardless of context)
- XGBoost prob ≥ 0.55 + LLM regime_flag == "conflicting" → half position
- XGBoost prob ≥ 0.65 + LLM confirms → full position
- DeepSeek veto overrides everything → FLAT
"""

from datetime import datetime, timezone


from signals.signal_models import SignalDirection
from signals.xgboost_signal import XGBoostSignal
from signals.llm_context import LLMContext
from signals.synthesis import (
    SynthesisInput,
    SynthesisOutput,
    synthesize,
)


def _make_xgboost_signal(
    probability: float,
    direction: SignalDirection | None = None,
) -> XGBoostSignal:
    """Helper to create XGBoostSignal for testing."""
    if direction is None:
        # Map probability to direction
        if probability >= 0.55:
            direction = "LONG"
        elif probability <= 0.45:
            direction = "SHORT"
        else:
            direction = "FLAT"

    return XGBoostSignal(
        symbol="BTC/USDT",
        timeframe="1h",
        direction=direction,
        probability=probability,
        confidence=abs(probability - 0.5) * 2,
        features={"rsi": 50.0},
        timestamp=datetime.now(timezone.utc),
    )


def _make_llm_context(
    regime_flag: str = "neutral",
    confidence: float = 0.7,
    bullish_factors: list[str] | None = None,
    bearish_factors: list[str] | None = None,
) -> LLMContext:
    """Helper to create LLMContext for testing."""
    return LLMContext(
        bullish_factors=bullish_factors or [],
        bearish_factors=bearish_factors or [],
        regime_flag=regime_flag,
        confidence=confidence,
    )


class TestFlatBelowThreshold:
    """Tests for XGBoost prob < 0.55 → FLAT regardless of context."""

    def test_flat_below_threshold(self):
        """XGBoost prob 0.50 → FLAT, regardless of any LLM context."""
        xgb_signal = _make_xgboost_signal(probability=0.50)
        llm_context = _make_llm_context(regime_flag="confirming", confidence=0.9)

        input_data = SynthesisInput(
            xgboost_signal=xgb_signal,
            llm_context=llm_context,
            critic_veto=False,
        )

        result = synthesize(input_data)

        assert result.direction == "FLAT"
        assert result.position_size_fraction == 0.0

    def test_flat_at_boundary(self):
        """XGBoost prob 0.549 → FLAT (boundary condition)."""
        xgb_signal = _make_xgboost_signal(probability=0.549)
        llm_context = _make_llm_context(regime_flag="confirming", confidence=1.0)

        input_data = SynthesisInput(
            xgboost_signal=xgb_signal,
            llm_context=llm_context,
            critic_veto=False,
        )

        result = synthesize(input_data)

        assert result.direction == "FLAT"
        assert result.position_size_fraction == 0.0

    def test_flat_for_short_below_threshold(self):
        """XGBoost prob 0.40 (SHORT direction) but below 0.55 → still FLAT."""
        xgb_signal = _make_xgboost_signal(probability=0.40, direction="SHORT")
        llm_context = _make_llm_context(regime_flag="confirming")

        input_data = SynthesisInput(
            xgboost_signal=xgb_signal,
            llm_context=llm_context,
            critic_veto=False,
        )

        # For SHORT with prob=0.40: conviction = 1-0.40 = 0.60, which is >= 0.55
        # So this should be tradeable as SHORT
        result = synthesize(input_data)
        assert result.direction == "SHORT"
        assert result.position_size_fraction > 0.0


class TestHalfPositionConflicting:
    """Tests for XGBoost prob ≥ 0.55 + conflicting regime → half position."""

    def test_half_position_conflicting(self):
        """XGBoost prob 0.60, regime_flag='conflicting' → direction matches, position=0.5."""
        xgb_signal = _make_xgboost_signal(probability=0.60)
        llm_context = _make_llm_context(regime_flag="conflicting")

        input_data = SynthesisInput(
            xgboost_signal=xgb_signal,
            llm_context=llm_context,
            critic_veto=False,
        )

        result = synthesize(input_data)

        assert result.direction == "LONG"  # Matches XGBoost
        assert result.position_size_fraction == 0.5

    def test_half_position_short_conflicting(self):
        """XGBoost SHORT with conflicting context → half position."""
        xgb_signal = _make_xgboost_signal(probability=0.35, direction="SHORT")
        llm_context = _make_llm_context(regime_flag="conflicting")

        input_data = SynthesisInput(
            xgboost_signal=xgb_signal,
            llm_context=llm_context,
            critic_veto=False,
        )

        result = synthesize(input_data)

        assert result.direction == "SHORT"
        assert result.position_size_fraction == 0.5


class TestFullPositionConfirming:
    """Tests for XGBoost prob ≥ 0.65 + confirming regime → full position."""

    def test_full_position_confirming(self):
        """XGBoost prob 0.70, regime_flag='confirming' → full position."""
        xgb_signal = _make_xgboost_signal(probability=0.70)
        llm_context = _make_llm_context(regime_flag="confirming")

        input_data = SynthesisInput(
            xgboost_signal=xgb_signal,
            llm_context=llm_context,
            critic_veto=False,
        )

        result = synthesize(input_data)

        assert result.direction == "LONG"
        assert result.position_size_fraction == 1.0

    def test_high_prob_neutral_context_partial(self):
        """XGBoost prob 0.70 with neutral context → some position (not full)."""
        xgb_signal = _make_xgboost_signal(probability=0.70)
        llm_context = _make_llm_context(regime_flag="neutral")

        input_data = SynthesisInput(
            xgboost_signal=xgb_signal,
            llm_context=llm_context,
            critic_veto=False,
        )

        result = synthesize(input_data)

        assert result.direction == "LONG"
        # Neutral regime should give partial position (between 0.5 and 1.0)
        assert 0.5 <= result.position_size_fraction <= 1.0

    def test_moderate_prob_confirming_partial(self):
        """XGBoost prob 0.58 (below 0.65) with confirming → not full position."""
        xgb_signal = _make_xgboost_signal(probability=0.58)
        llm_context = _make_llm_context(regime_flag="confirming")

        input_data = SynthesisInput(
            xgboost_signal=xgb_signal,
            llm_context=llm_context,
            critic_veto=False,
        )

        result = synthesize(input_data)

        assert result.direction == "LONG"
        # prob ≥ 0.55 but < 0.65, so not full position even with confirming
        assert result.position_size_fraction < 1.0


class TestVetoOverridesEverything:
    """Tests for DeepSeek veto → FLAT regardless of other inputs."""

    def test_veto_overrides_everything(self):
        """XGBoost prob 0.90, regime='confirming', but veto=True → FLAT."""
        xgb_signal = _make_xgboost_signal(probability=0.90)
        llm_context = _make_llm_context(regime_flag="confirming", confidence=1.0)

        input_data = SynthesisInput(
            xgboost_signal=xgb_signal,
            llm_context=llm_context,
            critic_veto=True,
        )

        result = synthesize(input_data)

        assert result.direction == "FLAT"
        assert result.position_size_fraction == 0.0

    def test_veto_with_high_conviction_short(self):
        """Veto overrides even high-conviction SHORT signals."""
        xgb_signal = _make_xgboost_signal(probability=0.10, direction="SHORT")
        llm_context = _make_llm_context(regime_flag="confirming")

        input_data = SynthesisInput(
            xgboost_signal=xgb_signal,
            llm_context=llm_context,
            critic_veto=True,
        )

        result = synthesize(input_data)

        assert result.direction == "FLAT"
        assert result.position_size_fraction == 0.0


class TestMissingContextFallback:
    """Tests for missing LLM context → XGBoost signal with reduced position."""

    def test_missing_context_fallback(self):
        """LLM context is None → use XGBoost signal alone with 0.7x position."""
        xgb_signal = _make_xgboost_signal(probability=0.70)

        input_data = SynthesisInput(
            xgboost_signal=xgb_signal,
            llm_context=None,  # Missing context
            critic_veto=False,
        )

        result = synthesize(input_data)

        assert result.direction == "LONG"
        assert result.position_size_fraction == 0.7

    def test_missing_context_short(self):
        """Missing context with SHORT signal → 0.7x position."""
        xgb_signal = _make_xgboost_signal(probability=0.25, direction="SHORT")

        input_data = SynthesisInput(
            xgboost_signal=xgb_signal,
            llm_context=None,
            critic_veto=False,
        )

        result = synthesize(input_data)

        assert result.direction == "SHORT"
        assert result.position_size_fraction == 0.7

    def test_missing_context_below_threshold(self):
        """Missing context but below threshold → still FLAT."""
        xgb_signal = _make_xgboost_signal(probability=0.52)

        input_data = SynthesisInput(
            xgboost_signal=xgb_signal,
            llm_context=None,
            critic_veto=False,
        )

        result = synthesize(input_data)

        assert result.direction == "FLAT"
        assert result.position_size_fraction == 0.0


class TestPositionSizeNeverExceedsOne:
    """Tests ensuring position_size_fraction never > 1.0."""

    def test_position_size_never_exceeds_one(self):
        """No input combination produces position_size_fraction > 1.0."""
        # Test extreme inputs
        extreme_inputs = [
            (0.99, "confirming", False),
            (0.95, "confirming", False),
            (0.80, "confirming", False),
            (0.70, "confirming", False),
            (0.65, "confirming", False),
        ]

        for prob, regime, veto in extreme_inputs:
            xgb_signal = _make_xgboost_signal(probability=prob)
            llm_context = _make_llm_context(regime_flag=regime, confidence=1.0)

            input_data = SynthesisInput(
                xgboost_signal=xgb_signal,
                llm_context=llm_context,
                critic_veto=veto,
            )

            result = synthesize(input_data)

            assert result.position_size_fraction <= 1.0, (
                f"Position size {result.position_size_fraction} > 1.0 "
                f"for prob={prob}, regime={regime}"
            )

    def test_position_size_non_negative(self):
        """Position size is never negative."""
        inputs = [
            (0.10, "conflicting", False),
            (0.30, "conflicting", True),
            (0.50, "neutral", False),
        ]

        for prob, regime, veto in inputs:
            xgb_signal = _make_xgboost_signal(probability=prob)
            llm_context = _make_llm_context(regime_flag=regime)

            input_data = SynthesisInput(
                xgboost_signal=xgb_signal,
                llm_context=llm_context,
                critic_veto=veto,
            )

            result = synthesize(input_data)

            assert result.position_size_fraction >= 0.0


class TestRationaleTextGenerated:
    """Tests for human-readable rationale in output."""

    def test_rationale_text_generated(self):
        """Output includes human-readable rationale string."""
        xgb_signal = _make_xgboost_signal(probability=0.70)
        llm_context = _make_llm_context(regime_flag="confirming")

        input_data = SynthesisInput(
            xgboost_signal=xgb_signal,
            llm_context=llm_context,
            critic_veto=False,
        )

        result = synthesize(input_data)

        assert result.rationale is not None
        assert isinstance(result.rationale, str)
        assert len(result.rationale) > 0

    def test_rationale_mentions_key_factors(self):
        """Rationale mentions XGBoost probability and regime."""
        xgb_signal = _make_xgboost_signal(probability=0.70)
        llm_context = _make_llm_context(regime_flag="conflicting")

        input_data = SynthesisInput(
            xgboost_signal=xgb_signal,
            llm_context=llm_context,
            critic_veto=False,
        )

        result = synthesize(input_data)

        # Rationale should mention probability or regime
        rationale_lower = result.rationale.lower()
        assert "0.70" in result.rationale or "70" in result.rationale or "prob" in rationale_lower
        assert "conflict" in rationale_lower or "half" in rationale_lower

    def test_rationale_mentions_veto(self):
        """Rationale mentions veto when applied."""
        xgb_signal = _make_xgboost_signal(probability=0.90)
        llm_context = _make_llm_context(regime_flag="confirming")

        input_data = SynthesisInput(
            xgboost_signal=xgb_signal,
            llm_context=llm_context,
            critic_veto=True,
        )

        result = synthesize(input_data)

        assert "veto" in result.rationale.lower()


class TestComponentsPreserved:
    """Tests for components dict preserving all inputs for logging."""

    def test_components_dict_has_all_inputs(self):
        """SynthesisOutput.components preserves all inputs."""
        xgb_signal = _make_xgboost_signal(probability=0.70)
        llm_context = _make_llm_context(
            regime_flag="confirming",
            bullish_factors=["high OI", "negative funding"],
        )

        input_data = SynthesisInput(
            xgboost_signal=xgb_signal,
            llm_context=llm_context,
            critic_veto=False,
        )

        result = synthesize(input_data)

        assert "xgboost_signal" in result.components
        assert "llm_context" in result.components
        assert "critic_veto" in result.components

        # XGBoost signal data should be serializable
        xgb_data = result.components["xgboost_signal"]
        assert xgb_data["probability"] == 0.70
        assert xgb_data["direction"] == "LONG"

        # LLM context data should be serializable
        llm_data = result.components["llm_context"]
        assert llm_data["regime_flag"] == "confirming"
        assert "high OI" in llm_data["bullish_factors"]

    def test_components_with_none_context(self):
        """Components handles None LLM context."""
        xgb_signal = _make_xgboost_signal(probability=0.65)

        input_data = SynthesisInput(
            xgboost_signal=xgb_signal,
            llm_context=None,
            critic_veto=False,
        )

        result = synthesize(input_data)

        assert result.components["llm_context"] is None


class TestSynthesisDataclasses:
    """Tests for SynthesisInput and SynthesisOutput dataclasses."""

    def test_synthesis_input_creation(self):
        """SynthesisInput can be created with required fields."""
        xgb_signal = _make_xgboost_signal(probability=0.65)
        llm_context = _make_llm_context()

        input_data = SynthesisInput(
            xgboost_signal=xgb_signal,
            llm_context=llm_context,
            critic_veto=False,
        )

        assert input_data.xgboost_signal == xgb_signal
        assert input_data.llm_context == llm_context
        assert input_data.critic_veto is False

    def test_synthesis_output_creation(self):
        """SynthesisOutput can be created with required fields."""
        output = SynthesisOutput(
            direction="LONG",
            position_size_fraction=0.75,
            rationale="Test rationale",
            components={"test": "data"},
        )

        assert output.direction == "LONG"
        assert output.position_size_fraction == 0.75
        assert output.rationale == "Test rationale"
        assert output.components == {"test": "data"}


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_exactly_at_0_55_threshold(self):
        """XGBoost prob exactly 0.55 → tradeable (boundary condition)."""
        xgb_signal = _make_xgboost_signal(probability=0.55)
        llm_context = _make_llm_context(regime_flag="neutral")

        input_data = SynthesisInput(
            xgboost_signal=xgb_signal,
            llm_context=llm_context,
            critic_veto=False,
        )

        result = synthesize(input_data)

        assert result.direction == "LONG"
        assert result.position_size_fraction > 0.0

    def test_exactly_at_0_65_threshold(self):
        """XGBoost prob exactly 0.65 with confirming → full position."""
        xgb_signal = _make_xgboost_signal(probability=0.65)
        llm_context = _make_llm_context(regime_flag="confirming")

        input_data = SynthesisInput(
            xgboost_signal=xgb_signal,
            llm_context=llm_context,
            critic_veto=False,
        )

        result = synthesize(input_data)

        assert result.direction == "LONG"
        assert result.position_size_fraction == 1.0

    def test_short_direction_high_conviction(self):
        """SHORT direction with high conviction and confirming context."""
        xgb_signal = _make_xgboost_signal(probability=0.15, direction="SHORT")
        llm_context = _make_llm_context(regime_flag="confirming")

        input_data = SynthesisInput(
            xgboost_signal=xgb_signal,
            llm_context=llm_context,
            critic_veto=False,
        )

        result = synthesize(input_data)

        assert result.direction == "SHORT"
        # 1 - 0.15 = 0.85 conviction, >= 0.65, confirming → full position
        assert result.position_size_fraction == 1.0
