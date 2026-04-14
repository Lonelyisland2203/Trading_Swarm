"""
Synthesis node for combining XGBoost signal + LLM context into trading decision.

Session 17N: This is the critical decision node that combines:
1. XGBoost signal (probability + direction)
2. LLM context (regime_flag, bullish/bearish factors)
3. DeepSeek critic veto

Output: Final direction + position size fraction

Rules from signal-layer.md:
- XGBoost prob < 0.55 → FLAT (no trade regardless of context)
- XGBoost prob ≥ 0.55 + LLM regime_flag == "conflicting" → half position
- XGBoost prob ≥ 0.65 + LLM confirms → full position
- DeepSeek veto overrides everything → FLAT
"""

from dataclasses import dataclass
from typing import Any

from signals.llm_context import LLMContext
from signals.signal_models import SignalDirection
from signals.xgboost_signal import XGBoostSignal


# Thresholds from signal-layer.md
TRADEABLE_THRESHOLD = 0.55  # Min conviction to trade
FULL_POSITION_THRESHOLD = 0.65  # Min conviction for full position with confirming context
MISSING_CONTEXT_FRACTION = 0.7  # Position size when LLM context unavailable


@dataclass(frozen=True)
class SynthesisInput:
    """
    Input to synthesis node.

    Attributes:
        xgboost_signal: XGBoost model output with probability and direction
        llm_context: LLM market context (may be None if generation failed)
        critic_veto: True if DeepSeek critic vetoed the trade
    """

    xgboost_signal: XGBoostSignal
    llm_context: LLMContext | None
    critic_veto: bool


@dataclass(frozen=True)
class SynthesisOutput:
    """
    Output from synthesis node.

    Attributes:
        direction: Final trading direction (LONG/SHORT/FLAT)
        position_size_fraction: Position size as fraction of max (0.0 to 1.0)
        rationale: Human-readable explanation of the decision
        components: Dict with all inputs preserved for logging
    """

    direction: SignalDirection
    position_size_fraction: float
    rationale: str
    components: dict[str, Any]


def _get_conviction(xgb_signal: XGBoostSignal) -> float:
    """
    Get conviction score from XGBoost signal.

    For LONG: conviction = probability
    For SHORT: conviction = 1 - probability
    For FLAT: conviction = 0.5 (neutral)

    Args:
        xgb_signal: XGBoost signal

    Returns:
        Conviction score (0.5 to 1.0)
    """
    prob = xgb_signal.probability

    if xgb_signal.direction == "LONG":
        return prob
    elif xgb_signal.direction == "SHORT":
        return 1.0 - prob
    else:
        return 0.5


def _build_components(
    xgb_signal: XGBoostSignal,
    llm_context: LLMContext | None,
    critic_veto: bool,
) -> dict[str, Any]:
    """
    Build components dict for logging.

    Serializes all inputs to JSON-compatible format.
    """
    return {
        "xgboost_signal": xgb_signal.to_dict(),
        "llm_context": llm_context.to_dict() if llm_context else None,
        "critic_veto": critic_veto,
    }


def synthesize(input_data: SynthesisInput) -> SynthesisOutput:
    """
    Synthesize final trading decision from XGBoost signal and LLM context.

    Rules (from signal-layer.md):
    1. DeepSeek veto overrides everything → FLAT
    2. XGBoost prob < 0.55 (conviction < 0.55) → FLAT
    3. XGBoost prob ≥ 0.55 + LLM regime_flag == "conflicting" → half position
    4. XGBoost prob ≥ 0.65 + LLM confirms → full position
    5. Missing LLM context → XGBoost direction with 0.7x position

    Args:
        input_data: SynthesisInput with XGBoost signal, LLM context, and veto flag

    Returns:
        SynthesisOutput with final direction, position size, and rationale
    """
    xgb_signal = input_data.xgboost_signal
    llm_context = input_data.llm_context
    critic_veto = input_data.critic_veto

    components = _build_components(xgb_signal, llm_context, critic_veto)

    # Rule 1: Veto overrides everything
    if critic_veto:
        return SynthesisOutput(
            direction="FLAT",
            position_size_fraction=0.0,
            rationale="DeepSeek critic veto: trade rejected due to risk filter",
            components=components,
        )

    # Get conviction from XGBoost signal
    conviction = _get_conviction(xgb_signal)

    # Rule 2: Below threshold → FLAT
    if conviction < TRADEABLE_THRESHOLD:
        return SynthesisOutput(
            direction="FLAT",
            position_size_fraction=0.0,
            rationale=f"Conviction {conviction:.2f} below threshold {TRADEABLE_THRESHOLD}",
            components=components,
        )

    # Direction is tradeable, determine position size
    direction = xgb_signal.direction

    # Rule 5: Missing context → 0.7x position
    if llm_context is None:
        return SynthesisOutput(
            direction=direction,
            position_size_fraction=MISSING_CONTEXT_FRACTION,
            rationale=f"XGBoost {direction} (prob={xgb_signal.probability:.2f}), "
            f"LLM context unavailable, using {MISSING_CONTEXT_FRACTION}x position",
            components=components,
        )

    regime_flag = llm_context.regime_flag

    # Rule 3: Conflicting regime → half position
    if regime_flag == "conflicting":
        return SynthesisOutput(
            direction=direction,
            position_size_fraction=0.5,
            rationale=f"XGBoost {direction} (prob={xgb_signal.probability:.2f}) with "
            f"conflicting LLM context, using half position",
            components=components,
        )

    # Rule 4: High conviction + confirming → full position
    if conviction >= FULL_POSITION_THRESHOLD and regime_flag == "confirming":
        return SynthesisOutput(
            direction=direction,
            position_size_fraction=1.0,
            rationale=f"XGBoost {direction} (prob={xgb_signal.probability:.2f}) with "
            f"confirming LLM context, using full position",
            components=components,
        )

    # Moderate conviction or neutral context → scaled position
    # Scale linearly: 0.55 → 0.5, 0.65 → 0.75 for neutral, or 0.65 → 1.0 for confirming
    if regime_flag == "confirming":
        # Confirming but below full threshold: scale from 0.5 at 0.55 to 1.0 at 0.65
        scale = (conviction - TRADEABLE_THRESHOLD) / (FULL_POSITION_THRESHOLD - TRADEABLE_THRESHOLD)
        position_fraction = 0.5 + 0.5 * scale
    elif regime_flag == "neutral":
        # Neutral: scale from 0.5 at 0.55 to 0.75 at 0.65+
        scale = min(
            1.0,
            (conviction - TRADEABLE_THRESHOLD) / (FULL_POSITION_THRESHOLD - TRADEABLE_THRESHOLD),
        )
        position_fraction = 0.5 + 0.25 * scale
    else:
        # Unknown regime flag, treat as neutral
        position_fraction = 0.5

    # Clamp to [0, 1]
    position_fraction = max(0.0, min(1.0, position_fraction))

    return SynthesisOutput(
        direction=direction,
        position_size_fraction=position_fraction,
        rationale=f"XGBoost {direction} (prob={xgb_signal.probability:.2f}) with "
        f"{regime_flag} LLM context, position={position_fraction:.2f}",
        components=components,
    )
