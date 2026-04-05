"""
Signal generator with trading personas and response extraction.

Implements:
- Five trading personas with distinct system prompts
- Regime-informed weighted persona sampling
- Multi-stage JSON extraction with fallback strategies
- Single clarification retry on parse failure
"""

import json
import random
import re
from dataclasses import dataclass
from enum import Enum

from loguru import logger

from data.prompt_builder import TaskType
from data.regime_filter import MarketRegime
from .exceptions import ResponseValidationError
from .ollama_client import OllamaClient

# Thread-local random generator (matches prompt_builder pattern)
_persona_rng = random.Random()


class TradingPersona(Enum):
    """Trading personas with different market approaches."""

    CONTRARIAN = "contrarian"
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    CONSERVATIVE = "conservative"


# Base persona weights (sum to 1.0, matches RewardWeights validation)
BASE_PERSONA_WEIGHTS = {
    TradingPersona.CONTRARIAN: 0.20,
    TradingPersona.MOMENTUM: 0.25,
    TradingPersona.MEAN_REVERSION: 0.20,
    TradingPersona.BREAKOUT: 0.15,
    TradingPersona.CONSERVATIVE: 0.20,
}

# Regime modifiers (multiplicative)
REGIME_MODIFIERS = {
    MarketRegime.RISK_OFF: {
        TradingPersona.CONSERVATIVE: 1.5,  # Boost conservative in risk-off
        TradingPersona.BREAKOUT: 0.5,      # Reduce breakout
    },
    MarketRegime.RISK_ON: {
        TradingPersona.MOMENTUM: 1.3,      # Boost momentum in risk-on
        TradingPersona.MEAN_REVERSION: 0.7,
    },
    # NEUTRAL: no modifier (use base weights)
}

# Persona system prompts
PERSONA_PROMPTS = {
    TradingPersona.CONTRARIAN: """You are a contrarian trader who profits from market overreactions.

Your philosophy:
- Extreme fear creates buying opportunities
- Extreme greed signals caution
- RSI >70 suggests potential reversal lower
- RSI <30 suggests potential reversal higher
- When everyone is bullish, be cautious. When everyone is bearish, look for value.

Focus on identifying when sentiment has gone too far and a reversal is likely.""",

    TradingPersona.MOMENTUM: """You are a momentum trader who rides established trends.

Your philosophy:
- The trend is your friend until it bends
- Price above moving averages = bullish, below = bearish
- MACD crossovers confirm trend changes
- Don't fight the tape - follow strong directional moves
- Let winners run, cut losers quickly

Focus on identifying when trends are strong and likely to continue.""",

    TradingPersona.MEAN_REVERSION: """You are a mean-reversion trader who profits from price returning to average.

Your philosophy:
- All prices revert to their mean over time
- Bollinger Band extremes signal reversion opportunities
- Price near upper band = expect pullback
- Price near lower band = expect bounce
- Markets oscillate around equilibrium

Focus on identifying when price has deviated significantly from average and is likely to revert.""",

    TradingPersona.BREAKOUT: """You are a breakout trader who captures explosive moves after consolidation.

Your philosophy:
- Compression leads to expansion
- Tight Bollinger Bands signal upcoming volatility
- Breakouts from consolidation are powerful
- Volume confirms breakout strength
- Early entry in new trends yields best returns

Focus on identifying when markets are compressed and ready to explode in one direction.""",

    TradingPersona.CONSERVATIVE: """You are a conservative trader who prioritizes capital preservation.

Your philosophy:
- Preservation of capital is the first rule
- Only high-probability setups justify risk
- When in doubt, stay out
- Small losses are acceptable, large losses are not
- Confidence must be backed by clear technical alignment

Focus on identifying only the clearest, highest-conviction setups with strong technical confirmation.""",
}


@dataclass(slots=True, frozen=True)
class GeneratorSignal:
    """
    Validated signal from generator model.

    Supports multiple task types with different schemas:
    - PREDICT_DIRECTION: {direction, confidence}
    - ASSESS_MOMENTUM: {direction, confidence}
    - IDENTIFY_SUPPORT_RESISTANCE: {support_price, support_confidence, resistance_price, resistance_confidence}
    """

    task_type: TaskType
    signal_data: dict  # Task-specific fields (direction/confidence OR support/resistance)
    reasoning: str
    persona: TradingPersona
    raw_response: str  # Preserve original for debugging

    # Convenience properties for backward compatibility with PREDICT_DIRECTION
    @property
    def direction(self) -> str | None:
        """Get direction for direction-based tasks (PREDICT_DIRECTION, ASSESS_MOMENTUM)."""
        return self.signal_data.get("direction")

    @property
    def confidence(self) -> float | None:
        """Get confidence for direction-based tasks (PREDICT_DIRECTION, ASSESS_MOMENTUM)."""
        return self.signal_data.get("confidence")


def sample_persona(
    regime: MarketRegime,
    seed: int | None = None,
) -> TradingPersona:
    """
    Sample persona with regime-informed weighting.

    Uses isolated RNG to avoid global state pollution (matches prompt_builder pattern).

    Args:
        regime: Current market regime
        seed: Random seed for reproducibility

    Returns:
        Sampled trading persona
    """
    rng = random.Random(seed) if seed is not None else _persona_rng

    weights = BASE_PERSONA_WEIGHTS.copy()
    modifiers = REGIME_MODIFIERS.get(regime, {})

    for persona, modifier in modifiers.items():
        weights[persona] *= modifier

    # Normalize
    total = sum(weights.values())
    normalized = {p: w / total for p, w in weights.items()}

    selected = rng.choices(
        list(normalized.keys()),
        weights=list(normalized.values()),
        k=1,
    )[0]

    logger.debug(
        "Persona sampled",
        persona=selected.value,
        regime=regime.value,
        weight=normalized[selected],
    )

    return selected


def _validate_signal_schema(data: dict, task_type: TaskType) -> bool:
    """
    Validate signal data has required fields for task type.

    Note: "reasoning" is optional and defaults to empty string if missing.

    Args:
        data: Parsed JSON data
        task_type: Task type to validate against

    Returns:
        True if schema is valid for task type
    """
    if task_type == TaskType.IDENTIFY_SUPPORT_RESISTANCE:
        required = {
            "support_price",
            "support_confidence",
            "resistance_price",
            "resistance_confidence",
        }
        return required.issubset(data.keys())
    else:  # PREDICT_DIRECTION, ASSESS_MOMENTUM
        required = {"direction", "confidence"}
        return required.issubset(data.keys())


def extract_signal(raw_response: str, persona: TradingPersona, task_type: TaskType) -> GeneratorSignal:
    """
    Extract signal from LLM response with multi-stage fallback.

    Attempts (in order):
    1. Direct JSON parse
    2. Strip markdown fences (```json ... ```)
    3. Extract from thinking tags (<think>...</think>)
    4. Regex extraction of key fields

    Args:
        raw_response: Raw LLM response text
        persona: Persona used for generation
        task_type: Task type for schema validation

    Returns:
        Validated signal

    Raises:
        ResponseValidationError: All extraction attempts failed
    """
    text = raw_response.strip()

    # Attempt 1: Direct JSON parse
    try:
        data = json.loads(text)
        return _validate_and_build_signal(data, persona, task_type, raw_response)
    except json.JSONDecodeError:
        pass

    # Attempt 2: Strip markdown fences
    fence_pattern = r'```(?:json)?\s*(.*?)\s*```'
    match = re.search(fence_pattern, text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(1))
            return _validate_and_build_signal(data, persona, task_type, raw_response)
        except json.JSONDecodeError:
            pass

    # Attempt 3: Extract from thinking tags (DeepSeek artifact)
    think_pattern = r'</think>\s*(.*?)$'
    match = re.search(think_pattern, text, re.DOTALL)
    if match:
        # Recursive call on post-think content
        return extract_signal(match.group(1), persona, task_type)

    # Attempt 4: Regex extraction (last resort, only for direction-based tasks)
    if task_type in (TaskType.PREDICT_DIRECTION, TaskType.ASSESS_MOMENTUM):
        direction_match = re.search(r'"direction"\s*:\s*"(INCREASING|DECREASING|HIGHER|LOWER)"', text, re.IGNORECASE)
        confidence_match = re.search(r'"confidence"\s*:\s*([\d.]+)', text)

        if direction_match and confidence_match:
            logger.warning("Using regex fallback extraction", persona=persona.value, task_type=task_type.value)
            return GeneratorSignal(
                task_type=task_type,
                signal_data={
                    "direction": direction_match.group(1).upper(),
                    "confidence": float(confidence_match.group(1)),
                },
                reasoning="[Extracted via regex fallback]",
                persona=persona,
                raw_response=raw_response,
            )

    # All attempts failed
    raise ResponseValidationError(
        f"Could not extract signal from response (persona={persona.value}, task={task_type.value}): {text[:200]}..."
    )


def _validate_and_build_signal(
    data: dict,
    persona: TradingPersona,
    task_type: TaskType,
    raw: str,
) -> GeneratorSignal:
    """
    Validate JSON schema and build signal based on task type.

    Args:
        data: Parsed JSON data
        persona: Persona used for generation
        task_type: Task type for schema validation
        raw: Raw response text

    Returns:
        Validated signal

    Raises:
        ValueError: Invalid schema
    """
    # Validate schema
    if not _validate_signal_schema(data, task_type):
        raise ValueError(f"Invalid schema for task type {task_type.value}: missing required fields")

    reasoning = str(data.get("reasoning", ""))

    if task_type == TaskType.IDENTIFY_SUPPORT_RESISTANCE:
        # Support/Resistance task
        support_price = float(data["support_price"])
        support_confidence = float(data["support_confidence"])
        resistance_price = float(data["resistance_price"])
        resistance_confidence = float(data["resistance_confidence"])

        # Clamp confidences to valid range
        support_confidence = max(0.0, min(1.0, support_confidence))
        resistance_confidence = max(0.0, min(1.0, resistance_confidence))

        # Validate prices are positive
        if support_price <= 0 or resistance_price <= 0:
            raise ValueError(f"Invalid prices: support={support_price}, resistance={resistance_price}")

        signal_data = {
            "support_price": support_price,
            "support_confidence": support_confidence,
            "resistance_price": resistance_price,
            "resistance_confidence": resistance_confidence,
        }

    else:  # PREDICT_DIRECTION, ASSESS_MOMENTUM
        # Direction-based task
        direction = data.get("direction", "").upper()

        # Handle both formats: HIGHER/LOWER and INCREASING/DECREASING
        if direction not in ("HIGHER", "LOWER", "INCREASING", "DECREASING"):
            raise ValueError(f"Invalid direction: {direction}")

        confidence = float(data.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))  # Clamp to valid range

        signal_data = {
            "direction": direction,
            "confidence": confidence,
        }

    return GeneratorSignal(
        task_type=task_type,
        signal_data=signal_data,
        reasoning=reasoning,
        persona=persona,
        raw_response=raw,
    )


async def generate_signal(
    client: OllamaClient,
    model: str,
    prompt: str,
    regime: MarketRegime,
    task_type: TaskType,
    temperature: float = 0.7,
    seed: int | None = None,
    persona_override: TradingPersona | None = None,
) -> GeneratorSignal | None:
    """
    Generate trading signal with persona-enhanced prompt.

    Includes single clarification retry on parse failure.

    Args:
        client: Ollama client (must be in async context)
        model: Model identifier (e.g., "qwen3:8b")
        prompt: Base prompt from PromptBuilder
        regime: Current market regime
        task_type: Task type for schema validation
        temperature: Generation temperature
        seed: Random seed for reproducibility
        persona_override: Optional specific persona (for cross-persona generation)

    Returns:
        Validated signal, or None if extraction failed after retry

    Example:
        async with OllamaClient() as client:
            signal = await generate_signal(
                client, "qwen3:8b", prompt, MarketRegime.NEUTRAL,
                TaskType.PREDICT_DIRECTION
            )
            if signal and signal.direction:
                print(f"Direction: {signal.direction}, Confidence: {signal.confidence}")
    """
    # Sample persona or use override for cross-persona generation
    persona = persona_override if persona_override is not None else sample_persona(regime, seed=seed)

    # Build persona-enhanced prompt
    persona_prompt = PERSONA_PROMPTS[persona]
    full_prompt = f"{persona_prompt}\n\n{prompt}"

    # Generation options
    # think=False disables qwen3's reasoning step so output goes to `response`
    # (with thinking on, all tokens are consumed by the `thinking` key, leaving response empty)
    options = {
        "temperature": temperature,
        "top_p": 0.9,
        "num_predict": 512,
        "think": False,
    }
    if seed is not None:
        options["seed"] = seed

    # Generate
    logger.info(
        "Generating signal",
        model=model,
        persona=persona.value,
        regime=regime.value,
        task_type=task_type.value,
        temperature=temperature,
    )

    response = await client.generate(model, full_prompt, options)
    raw_text = response["response"]

    # Extract signal
    try:
        signal = extract_signal(raw_text, persona, task_type)
        logger.info(
            "Signal generated",
            task_type=task_type.value,
            persona=persona.value,
            signal_data=signal.signal_data,
        )
        return signal

    except ResponseValidationError as e:
        # Single retry with clarification
        logger.warning("Parse failed, retrying with clarification", error=str(e))

        # Task-specific clarification
        if task_type == TaskType.IDENTIFY_SUPPORT_RESISTANCE:
            clarification = """
Your previous response could not be parsed as JSON.
Please respond with ONLY valid JSON, no markdown or explanation:
{"support_price": float, "support_confidence": 0.0-1.0, "resistance_price": float, "resistance_confidence": 0.0-1.0, "reasoning": "brief explanation"}
"""
        else:  # PREDICT_DIRECTION, ASSESS_MOMENTUM
            clarification = """
Your previous response could not be parsed as JSON.
Please respond with ONLY valid JSON, no markdown or explanation:
{"direction": "HIGHER" | "LOWER" | "INCREASING" | "DECREASING", "confidence": 0.0-1.0, "reasoning": "brief explanation"}
"""

        clarified_prompt = full_prompt + "\n\n" + clarification

        retry_response = await client.generate(model, clarified_prompt, options)
        retry_text = retry_response["response"]

        try:
            signal = extract_signal(retry_text, persona, task_type)
            logger.info("Signal extracted after clarification", task_type=task_type.value)
            return signal

        except ResponseValidationError as retry_error:
            logger.error(
                "Signal extraction failed after retry",
                persona=persona.value,
                task_type=task_type.value,
                original_error=str(e),
                retry_error=str(retry_error),
                raw_response=retry_text[:200],
            )
            return None  # Mark as invalid
