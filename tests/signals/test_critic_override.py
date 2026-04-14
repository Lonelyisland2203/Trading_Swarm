"""Tests for DeepSeek veto override logic.

Session 17N: Refactored to test synthesis veto behavior.
Old logic tested thesis quality evaluation.
New logic tests binary APPROVE/VETO from DeepSeek risk filter.
"""

from datetime import datetime, timezone

from signals.synthesis import SynthesisInput, synthesize
from signals.xgboost_signal import XGBoostSignal
from signals.llm_context import LLMContext


def _make_xgboost_signal(probability: float = 0.70) -> XGBoostSignal:
    """Create test XGBoost signal."""
    direction = "LONG" if probability >= 0.55 else ("SHORT" if probability <= 0.45 else "FLAT")
    return XGBoostSignal(
        symbol="BTC/USDT",
        timeframe="1h",
        direction=direction,
        probability=probability,
        confidence=abs(probability - 0.5) * 2,
        features={"rsi": 55.0},
        timestamp=datetime.now(timezone.utc),
    )


def _make_llm_context(regime_flag: str = "confirming") -> LLMContext:
    """Create test LLM context."""
    return LLMContext(
        bullish_factors=["Test factor"],
        bearish_factors=[],
        regime_flag=regime_flag,
        confidence=0.8,
    )


class TestVetoOverrideLogic:
    """Tests for DeepSeek veto override in synthesis."""

    def test_veto_true_produces_flat(self):
        """When critic_veto=True, synthesis always produces FLAT."""
        xgb_signal = _make_xgboost_signal(probability=0.90)
        llm_context = _make_llm_context(regime_flag="confirming")

        result = synthesize(
            SynthesisInput(
                xgboost_signal=xgb_signal,
                llm_context=llm_context,
                critic_veto=True,
            )
        )

        assert result.direction == "FLAT"
        assert result.position_size_fraction == 0.0

    def test_veto_false_allows_trade(self):
        """When critic_veto=False, synthesis allows trade."""
        xgb_signal = _make_xgboost_signal(probability=0.70)
        llm_context = _make_llm_context(regime_flag="confirming")

        result = synthesize(
            SynthesisInput(
                xgboost_signal=xgb_signal,
                llm_context=llm_context,
                critic_veto=False,
            )
        )

        assert result.direction == "LONG"
        assert result.position_size_fraction > 0.0

    def test_veto_overrides_high_conviction_long(self):
        """Veto overrides even very high probability LONG signals."""
        xgb_signal = _make_xgboost_signal(probability=0.95)
        llm_context = _make_llm_context(regime_flag="confirming")

        result = synthesize(
            SynthesisInput(
                xgboost_signal=xgb_signal,
                llm_context=llm_context,
                critic_veto=True,
            )
        )

        assert result.direction == "FLAT"
        assert "veto" in result.rationale.lower()

    def test_veto_overrides_high_conviction_short(self):
        """Veto overrides even high probability SHORT signals."""
        xgb_signal = XGBoostSignal(
            symbol="BTC/USDT",
            timeframe="1h",
            direction="SHORT",
            probability=0.10,  # High conviction short
            confidence=0.80,
            features={"rsi": 30.0},
            timestamp=datetime.now(timezone.utc),
        )
        llm_context = _make_llm_context(regime_flag="confirming")

        result = synthesize(
            SynthesisInput(
                xgboost_signal=xgb_signal,
                llm_context=llm_context,
                critic_veto=True,
            )
        )

        assert result.direction == "FLAT"

    def test_veto_with_conflicting_context(self):
        """Veto still applies even with conflicting context."""
        xgb_signal = _make_xgboost_signal(probability=0.65)
        llm_context = _make_llm_context(regime_flag="conflicting")

        result = synthesize(
            SynthesisInput(
                xgboost_signal=xgb_signal,
                llm_context=llm_context,
                critic_veto=True,
            )
        )

        assert result.direction == "FLAT"

    def test_veto_with_none_context(self):
        """Veto applies even when LLM context is None."""
        xgb_signal = _make_xgboost_signal(probability=0.75)

        result = synthesize(
            SynthesisInput(
                xgboost_signal=xgb_signal,
                llm_context=None,
                critic_veto=True,
            )
        )

        assert result.direction == "FLAT"
        assert result.position_size_fraction == 0.0


class TestNonVetoScenarios:
    """Tests for synthesis when not vetoed."""

    def test_low_prob_flat_not_affected_by_veto_false(self):
        """Low probability signals are FLAT regardless of veto status."""
        xgb_signal = _make_xgboost_signal(probability=0.50)
        llm_context = _make_llm_context(regime_flag="confirming")

        result = synthesize(
            SynthesisInput(
                xgboost_signal=xgb_signal,
                llm_context=llm_context,
                critic_veto=False,
            )
        )

        # FLAT because of low conviction, not veto
        assert result.direction == "FLAT"

    def test_high_prob_confirming_full_position(self):
        """High prob + confirming = full position when not vetoed."""
        xgb_signal = _make_xgboost_signal(probability=0.70)
        llm_context = _make_llm_context(regime_flag="confirming")

        result = synthesize(
            SynthesisInput(
                xgboost_signal=xgb_signal,
                llm_context=llm_context,
                critic_veto=False,
            )
        )

        assert result.direction == "LONG"
        assert result.position_size_fraction == 1.0

    def test_moderate_prob_conflicting_half_position(self):
        """Moderate prob + conflicting = half position when not vetoed."""
        xgb_signal = _make_xgboost_signal(probability=0.60)
        llm_context = _make_llm_context(regime_flag="conflicting")

        result = synthesize(
            SynthesisInput(
                xgboost_signal=xgb_signal,
                llm_context=llm_context,
                critic_veto=False,
            )
        )

        assert result.direction == "LONG"
        assert result.position_size_fraction == 0.5


class TestVetoRationale:
    """Tests for veto rationale in output."""

    def test_veto_rationale_mentions_veto(self):
        """Rationale explicitly mentions veto."""
        xgb_signal = _make_xgboost_signal(probability=0.80)
        llm_context = _make_llm_context(regime_flag="confirming")

        result = synthesize(
            SynthesisInput(
                xgboost_signal=xgb_signal,
                llm_context=llm_context,
                critic_veto=True,
            )
        )

        assert "veto" in result.rationale.lower()

    def test_veto_rationale_mentions_risk(self):
        """Rationale mentions risk filter."""
        xgb_signal = _make_xgboost_signal(probability=0.75)
        llm_context = _make_llm_context(regime_flag="neutral")

        result = synthesize(
            SynthesisInput(
                xgboost_signal=xgb_signal,
                llm_context=llm_context,
                critic_veto=True,
            )
        )

        # Should mention either "veto" or "risk" or "reject"
        rationale_lower = result.rationale.lower()
        assert any(word in rationale_lower for word in ["veto", "risk", "reject"])
