"""
LLM context overlay node for market analysis.

Session 17M: Qwen produces CONTEXT only, NEVER direction.
The LLM analyzes funding rates, OI delta, liquidations, and news
to provide qualitative context (bullish/bearish factors, regime flag).

Hard constraint: Output must NEVER contain LONG/SHORT/BUY/SELL.
"""

import json
import re
from dataclasses import dataclass
from typing import Any

import httpx
from loguru import logger

from signals.preflight import enforce_ollama_keep_alive, run_preflight_checks

# Valid regime flag values
VALID_REGIME_FLAGS = frozenset(["confirming", "conflicting", "neutral"])

# Forbidden directional words (case-insensitive, standalone)
FORBIDDEN_WORDS_PATTERN = re.compile(
    r"\b(LONG|SHORT|BUY|SELL)\b",
    re.IGNORECASE,
)

# System prompt that explicitly forbids directional output
SYSTEM_PROMPT = """You are a market context analyst. Analyze the provided market data.
List bullish factors and bearish factors based on the data.
Classify the overall regime as confirming, conflicting, or neutral.
Output valid JSON with keys: bullish_factors, bearish_factors, regime_flag, confidence.

CRITICAL RULES:
- DO NOT predict price direction.
- DO NOT output LONG, SHORT, BUY, or SELL.
- Only describe market conditions, not trading actions.
- regime_flag must be exactly one of: confirming, conflicting, neutral
- confidence must be a float between 0 and 1"""

# Ollama configuration
OLLAMA_MODEL = "qwen3:8b"
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_TIMEOUT = 60.0  # seconds


@dataclass(frozen=True)
class LLMContext:
    """
    Market context from LLM analysis.

    Attributes:
        bullish_factors: List of bullish market factors identified
        bearish_factors: List of bearish market factors identified
        regime_flag: One of 'confirming', 'conflicting', 'neutral'
        confidence: Float 0-1 indicating confidence in the analysis
    """

    bullish_factors: list[str]
    bearish_factors: list[str]
    regime_flag: str
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "bullish_factors": self.bullish_factors,
            "bearish_factors": self.bearish_factors,
            "regime_flag": self.regime_flag,
            "confidence": self.confidence,
        }


def _create_neutral_context(confidence: float = 0.0) -> LLMContext:
    """Create a neutral fallback context."""
    return LLMContext(
        bullish_factors=[],
        bearish_factors=[],
        regime_flag="neutral",
        confidence=confidence,
    )


def _contains_forbidden_words(text: str) -> bool:
    """Check if text contains forbidden directional words."""
    return FORBIDDEN_WORDS_PATTERN.search(text) is not None


def _extract_json_from_text(text: str) -> dict[str, Any] | None:
    """
    Extract JSON object from text that may contain surrounding content.

    Handles LLM outputs like:
    "Here is my analysis: {...json...} That's my view."
    """
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find JSON object in text
    # Look for outermost braces
    start_idx = text.find("{")
    if start_idx == -1:
        return None

    # Find matching closing brace
    brace_count = 0
    for i, char in enumerate(text[start_idx:], start_idx):
        if char == "{":
            brace_count += 1
        elif char == "}":
            brace_count -= 1
            if brace_count == 0:
                try:
                    return json.loads(text[start_idx : i + 1])
                except json.JSONDecodeError:
                    return None

    return None


def _validate_and_parse_response(response_text: str) -> LLMContext | None:
    """
    Validate and parse LLM response into LLMContext.

    Returns None if:
    - JSON is invalid/missing required fields
    - Contains forbidden directional words
    - regime_flag is not valid
    """
    # Check for forbidden words in raw response
    if _contains_forbidden_words(response_text):
        logger.warning("LLM response contained forbidden directional words")
        return None

    # Extract JSON
    data = _extract_json_from_text(response_text)
    if data is None:
        logger.warning("Failed to extract JSON from LLM response")
        return None

    # Validate required fields
    required_fields = ["bullish_factors", "bearish_factors", "regime_flag", "confidence"]
    for field in required_fields:
        if field not in data:
            logger.warning(f"Missing required field: {field}")
            return None

    # Validate types
    if not isinstance(data["bullish_factors"], list):
        return None
    if not isinstance(data["bearish_factors"], list):
        return None
    if not isinstance(data["regime_flag"], str):
        return None

    # Validate regime_flag value
    if data["regime_flag"] not in VALID_REGIME_FLAGS:
        logger.warning(f"Invalid regime_flag: {data['regime_flag']}")
        return None

    # Validate confidence is numeric
    try:
        confidence = float(data["confidence"])
        confidence = max(0.0, min(1.0, confidence))  # Clamp to [0, 1]
    except (TypeError, ValueError):
        return None

    # Check factors for forbidden words
    all_factors = " ".join(data["bullish_factors"] + data["bearish_factors"])
    if _contains_forbidden_words(all_factors):
        logger.warning("LLM factors contained forbidden directional words")
        return None

    return LLMContext(
        bullish_factors=data["bullish_factors"],
        bearish_factors=data["bearish_factors"],
        regime_flag=data["regime_flag"],
        confidence=confidence,
    )


def build_user_prompt(
    funding_rate: float | None,
    oi_delta: float | None,
    liquidation_data: dict[str, Any] | None,
    news_headlines: list[str] | None,
) -> str:
    """
    Build user prompt from market data inputs.

    Args:
        funding_rate: Current funding rate (e.g., -0.01 for -1%)
        oi_delta: Change in open interest as percentage
        liquidation_data: Dict with liquidation info
        news_headlines: List of recent news headlines

    Returns:
        Formatted prompt string for LLM
    """
    sections = []

    if funding_rate is not None:
        pct = funding_rate * 100
        direction = "negative" if funding_rate < 0 else "positive"
        sections.append(f"Funding Rate: {pct:.4f}% ({direction})")

    if oi_delta is not None:
        direction = "increasing" if oi_delta > 0 else "decreasing"
        sections.append(f"Open Interest Delta: {oi_delta:.2f}% ({direction})")

    if liquidation_data is not None:
        liq_parts = []
        if "long_liquidations" in liquidation_data:
            liq_parts.append(f"Long liquidations: ${liquidation_data['long_liquidations']:,.0f}")
        if "short_liquidations" in liquidation_data:
            liq_parts.append(f"Short liquidations: ${liquidation_data['short_liquidations']:,.0f}")
        if "longs" in liquidation_data:
            liq_parts.append(f"Longs: {liquidation_data['longs']}")
        if "shorts" in liquidation_data:
            liq_parts.append(f"Shorts: {liquidation_data['shorts']}")
        if liq_parts:
            sections.append(f"Liquidation Data: {', '.join(liq_parts)}")

    if news_headlines:
        headlines_text = "\n".join(f"- {h}" for h in news_headlines[:5])
        sections.append(f"Recent News Headlines:\n{headlines_text}")

    if not sections:
        return "No market data available. Return neutral context with low confidence."

    return "Analyze the following market data:\n\n" + "\n\n".join(sections)


async def _call_ollama(prompt: str) -> str:
    """
    Call Ollama API with the given prompt.

    Args:
        prompt: User prompt to send to the model

    Returns:
        Raw response text from the model

    Raises:
        Exception: On connection failure or timeout
    """
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "system": SYSTEM_PROMPT,
        "stream": False,
        "options": {
            "temperature": 0.3,  # Lower temperature for more consistent JSON
            "num_predict": 512,
        },
    }

    async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
        response = await client.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=payload,
        )
        response.raise_for_status()
        result = response.json()
        return result.get("response", "")


async def _unload_model() -> None:
    """
    Unload model from VRAM.

    With OLLAMA_KEEP_ALIVE=0, models unload automatically after inference.
    This is a no-op placeholder for explicit unload if needed.
    """
    # OLLAMA_KEEP_ALIVE=0 handles automatic unload
    # Explicit unload not needed but logged for traceability
    logger.debug("Model unload triggered (OLLAMA_KEEP_ALIVE=0 handles this)")


def _calculate_input_confidence_cap(
    funding_rate: float | None,
    oi_delta: float | None,
    liquidation_data: dict[str, Any] | None,
    news_headlines: list[str] | None,
) -> float:
    """
    Calculate confidence cap based on available inputs.

    More data = higher confidence cap.
    """
    available_inputs = sum(
        [
            funding_rate is not None,
            oi_delta is not None,
            liquidation_data is not None,
            news_headlines is not None and len(news_headlines) > 0,
        ]
    )

    if available_inputs == 0:
        return 0.1
    elif available_inputs == 1:
        return 0.4
    elif available_inputs == 2:
        return 0.6
    elif available_inputs == 3:
        return 0.8
    else:
        return 1.0


async def generate_market_context(
    funding_rate: float | None,
    oi_delta: float | None,
    liquidation_data: dict[str, Any] | None,
    news_headlines: list[str] | None,
) -> LLMContext:
    """
    Generate market context using LLM analysis.

    This function:
    1. Runs VRAM preflight checks
    2. Sets OLLAMA_KEEP_ALIVE=0
    3. Calls Ollama with market data
    4. Parses and validates response
    5. Returns neutral fallback on any failure

    Args:
        funding_rate: Current funding rate (e.g., -0.01 for -1%)
        oi_delta: Change in open interest as percentage
        liquidation_data: Dict with liquidation info
        news_headlines: List of recent news headlines

    Returns:
        LLMContext with bullish/bearish factors and regime classification
    """
    # Calculate confidence cap based on available inputs
    confidence_cap = _calculate_input_confidence_cap(
        funding_rate, oi_delta, liquidation_data, news_headlines
    )

    # If no data at all, return neutral immediately
    if confidence_cap <= 0.1:
        logger.info("No market data provided, returning neutral context")
        return _create_neutral_context(confidence=0.1)

    # Special case: funding_rate=None and oi_delta=None but have other data
    if funding_rate is None and oi_delta is None:
        logger.info("Core metrics (funding, OI) missing, returning neutral context")
        return _create_neutral_context(confidence=min(0.3, confidence_cap))

    # Run preflight checks
    preflight_result = run_preflight_checks()
    if not preflight_result.passed:
        logger.warning(f"Preflight failed: {preflight_result.reason}")
        return _create_neutral_context()

    # Enforce OLLAMA_KEEP_ALIVE=0
    enforce_ollama_keep_alive()

    # Build prompt
    user_prompt = build_user_prompt(
        funding_rate=funding_rate,
        oi_delta=oi_delta,
        liquidation_data=liquidation_data,
        news_headlines=news_headlines,
    )

    try:
        # Call Ollama
        response_text = await _call_ollama(user_prompt)
        logger.debug(f"Ollama response: {response_text[:200]}...")

        # Parse and validate
        context = _validate_and_parse_response(response_text)

        if context is None:
            logger.warning("Failed to validate LLM response, using neutral fallback")
            return _create_neutral_context()

        # Apply confidence cap based on available inputs
        capped_confidence = min(context.confidence, confidence_cap)

        return LLMContext(
            bullish_factors=context.bullish_factors,
            bearish_factors=context.bearish_factors,
            regime_flag=context.regime_flag,
            confidence=capped_confidence,
        )

    except Exception as e:
        logger.error(f"Ollama call failed: {e}")
        return _create_neutral_context()

    finally:
        # Ensure model unload
        await _unload_model()
