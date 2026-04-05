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
    """Validated signal from generator model."""

    direction: str  # "HIGHER" | "LOWER"
    confidence: float  # 0.0-1.0
    reasoning: str
    persona: TradingPersona
    raw_response: str  # Preserve original for debugging


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


def extract_signal(raw_response: str, persona: TradingPersona) -> GeneratorSignal:
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

    Returns:
        Validated signal

    Raises:
        ResponseValidationError: All extraction attempts failed
    """
    text = raw_response.strip()

    # Attempt 1: Direct JSON parse
    try:
        data = json.loads(text)
        return _validate_and_build_signal(data, persona, raw_response)
    except json.JSONDecodeError:
        pass

    # Attempt 2: Strip markdown fences
    fence_pattern = r'```(?:json)?\s*(.*?)\s*```'
    match = re.search(fence_pattern, text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(1))
            return _validate_and_build_signal(data, persona, raw_response)
        except json.JSONDecodeError:
            pass

    # Attempt 3: Extract from thinking tags (DeepSeek artifact)
    think_pattern = r'</think>\s*(.*?)$'
    match = re.search(think_pattern, text, re.DOTALL)
    if match:
        # Recursive call on post-think content
        return extract_signal(match.group(1), persona)

    # Attempt 4: Regex extraction (last resort)
    direction_match = re.search(r'"direction"\s*:\s*"(HIGHER|LOWER)"', text, re.IGNORECASE)
    confidence_match = re.search(r'"confidence"\s*:\s*([\d.]+)', text)

    if direction_match and confidence_match:
        logger.warning("Using regex fallback extraction", persona=persona.value)
        return GeneratorSignal(
            direction=direction_match.group(1).upper(),
            confidence=float(confidence_match.group(1)),
            reasoning="[Extracted via regex fallback]",
            persona=persona,
            raw_response=raw_response,
        )

    # All attempts failed
    raise ResponseValidationError(
        f"Could not extract signal from response (persona={persona.value}): {text[:200]}..."
    )


def _validate_and_build_signal(
    data: dict,
    persona: TradingPersona,
    raw: str,
) -> GeneratorSignal:
    """
    Validate JSON schema and build signal.

    Args:
        data: Parsed JSON data
        persona: Persona used for generation
        raw: Raw response text

    Returns:
        Validated signal

    Raises:
        ValueError: Invalid schema
    """
    direction = data.get("direction", "").upper()
    if direction not in ("HIGHER", "LOWER"):
        raise ValueError(f"Invalid direction: {direction}")

    confidence = float(data.get("confidence", 0.5))
    confidence = max(0.0, min(1.0, confidence))  # Clamp to valid range

    return GeneratorSignal(
        direction=direction,
        confidence=confidence,
        reasoning=str(data.get("reasoning", "")),
        persona=persona,
        raw_response=raw,
    )


async def generate_signal(
    client: OllamaClient,
    model: str,
    prompt: str,
    regime: MarketRegime,
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
        temperature: Generation temperature
        seed: Random seed for reproducibility
        persona_override: Optional specific persona (for cross-persona generation)

    Returns:
        Validated signal, or None if extraction failed after retry

    Example:
        async with OllamaClient() as client:
            signal = await generate_signal(
                client, "qwen3:8b", prompt, MarketRegime.NEUTRAL
            )
            if signal:
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
        temperature=temperature,
    )

    response = await client.generate(model, full_prompt, options)
    raw_text = response["response"]

    # Extract signal
    try:
        signal = extract_signal(raw_text, persona)
        logger.info(
            "Signal generated",
            direction=signal.direction,
            confidence=signal.confidence,
            persona=persona.value,
        )
        return signal

    except ResponseValidationError as e:
        # Single retry with clarification
        logger.warning("Parse failed, retrying with clarification", error=str(e))

        clarification = """
Your previous response could not be parsed as JSON.
Please respond with ONLY valid JSON, no markdown or explanation:
{"direction": "HIGHER" | "LOWER", "confidence": 0.0-1.0, "reasoning": "brief explanation"}
"""

        clarified_prompt = full_prompt + "\n\n" + clarification

        retry_response = await client.generate(model, clarified_prompt, options)
        retry_text = retry_response["response"]

        try:
            signal = extract_signal(retry_text, persona)
            logger.info("Signal extracted after clarification", direction=signal.direction)
            return signal

        except ResponseValidationError as retry_error:
            logger.error(
                "Signal extraction failed after retry",
                persona=persona.value,
                original_error=str(e),
                retry_error=str(retry_error),
                raw_response=retry_text[:200],
            )
            return None  # Mark as invalid
