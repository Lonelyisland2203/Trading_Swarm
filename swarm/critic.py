"""
Signal critic using DeepSeek-R1 for evaluation.

Implements adversarial critique to identify flaws in generator reasoning,
technical alignment, and confidence calibration.
"""

import json
import re
from dataclasses import dataclass

from loguru import logger

from .exceptions import ResponseValidationError
from .ollama_client import OllamaClient

# Critique prompt with adversarial framing
CRITIQUE_TEMPLATE = """You are an objective trading signal evaluator. Assess signal quality fairly and accurately.

Guidelines:
- ACCEPT if reasoning is logical and indicators broadly support the direction
- REJECT only if there are clear contradictions or obvious logical flaws  
- UNCERTAIN if evidence is genuinely mixed

A good signal does not need to be perfect — it just needs to be coherent and grounded in the data.

## Generator Signal
Direction: {direction}
Confidence: {confidence}
Reasoning: {reasoning}
Persona: {persona}

## Original Task
{task_prompt}

## Market Context
Symbol: {symbol}
Current Price: ${current_price:.4f}
Market Regime: {market_regime}

## Technical Indicators
- RSI(14): {rsi:.2f}
- MACD: {macd:.4f} (Signal: {macd_signal:.4f})
- BB Position: {bb_position:.2f} (0.0=lower band, 0.5=middle, 1.0=upper band)

## Recent Price Action (last 5 bars)
{recent_ohlcv}

## Your Task
Evaluate this signal objectively across these dimensions:

1. **Reasoning Quality** - Does the logic logically follow from the data? Are there gaps or contradictions?
2. **Technical Alignment** - Do indicators actually support the prediction? Check for contradictions (e.g., "bullish" but RSI >70 suggests overbought).
3. **Confidence Calibration** - Is the confidence level justified by the strength of evidence? Over-confident or under-confident?
4. **Persona Consistency** - Does the reasoning align with the {persona} trading philosophy?

Be specific in your critique, but do not manufacture flaws that aren't there.

If you cannot find significant flaws after careful analysis, acknowledge the signal is sound.

Respond in JSON format:
{{
  "reasoning_quality": 0.0-1.0,
  "technical_alignment": 0.0-1.0,
  "confidence_calibration": 0.0-1.0,
  "critique": "detailed analysis citing specific issues or strengths",
  "recommendation": "ACCEPT" | "REJECT" | "UNCERTAIN"
}}
"""


@dataclass(slots=True, frozen=True)
class CritiqueResult:
    """
    Critic evaluation of generator signal.

    Scores are independent dimensions:
    - reasoning_quality: Coherence and logic flow
    - technical_alignment: Indicators support prediction
    - confidence_calibration: Confidence justified by evidence

    Overall score is computed property (weighted average).
    """

    reasoning_quality: float
    technical_alignment: float
    confidence_calibration: float
    critique: str
    recommendation: str  # "ACCEPT" | "REJECT" | "UNCERTAIN"
    raw_response: str

    @property
    def score(self) -> float:
        """
        Derived score from sub-dimensions (weighted average).

        Weights:
        - 35% reasoning_quality
        - 40% technical_alignment (most important)
        - 25% confidence_calibration

        Returns:
            Overall quality score (0.0-1.0)
        """
        return (
            0.35 * self.reasoning_quality
            + 0.40 * self.technical_alignment
            + 0.25 * self.confidence_calibration
        )


def validate_critique(result: CritiqueResult) -> bool:
    """
    Sanity checks on critic output.

    Detects common failure modes:
    - Contradiction between scores and recommendation
    - Too-short critique (likely low effort)
    - Invalid score ranges

    Args:
        result: Critique result to validate

    Returns:
        True if critique appears valid, False otherwise
    """
    # Check consistency between overall score and recommendation
    if result.recommendation == "ACCEPT" and result.score < 0.5:
        logger.warning(
            "Critique inconsistency: ACCEPT but low score",
            score=result.score,
            recommendation=result.recommendation,
        )
        return False

    if result.recommendation == "REJECT" and result.score > 0.7:
        logger.warning(
            "Critique inconsistency: REJECT but high score",
            score=result.score,
            recommendation=result.recommendation,
        )
        return False

    # Check critique has substance
    if len(result.critique) < 50:
        logger.warning("Critique too short", length=len(result.critique))
        return False

    # Check all sub-scores are in valid range
    scores = [
        result.reasoning_quality,
        result.technical_alignment,
        result.confidence_calibration,
    ]

    for score in scores:
        if not (0.0 <= score <= 1.0):
            logger.warning("Invalid score range", score=score)
            return False

    return True


def extract_critique(raw_response: str) -> CritiqueResult:
    """
    Extract critique from LLM response with multi-stage fallback.

    Attempts (in order):
    1. Direct JSON parse
    2. Strip markdown fences (```json ... ```)
    3. Regex extraction of key fields

    Args:
        raw_response: Raw LLM response text

    Returns:
        Validated critique

    Raises:
        ResponseValidationError: All extraction attempts failed
    """
    text = raw_response.strip()

    # Attempt 1: Direct JSON parse
    try:
        data = json.loads(text)
        return _validate_and_build_critique(data, raw_response)
    except json.JSONDecodeError:
        pass

    # Attempt 2: Strip markdown fences
    fence_pattern = r'```(?:json)?\s*(.*?)\s*```'
    match = re.search(fence_pattern, text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(1))
            return _validate_and_build_critique(data, raw_response)
        except json.JSONDecodeError:
            pass

    # Attempt 3: Regex extraction (last resort)
    reasoning_quality_match = re.search(r'"reasoning_quality"\s*:\s*([\d.]+)', text)
    technical_alignment_match = re.search(r'"technical_alignment"\s*:\s*([\d.]+)', text)
    confidence_calibration_match = re.search(r'"confidence_calibration"\s*:\s*([\d.]+)', text)
    recommendation_match = re.search(
        r'"recommendation"\s*:\s*"(ACCEPT|REJECT|UNCERTAIN)"',
        text,
        re.IGNORECASE,
    )

    if all([
        reasoning_quality_match,
        technical_alignment_match,
        confidence_calibration_match,
        recommendation_match,
    ]):
        logger.warning("Using regex fallback extraction for critique")
        return CritiqueResult(
            reasoning_quality=float(reasoning_quality_match.group(1)),
            technical_alignment=float(technical_alignment_match.group(1)),
            confidence_calibration=float(confidence_calibration_match.group(1)),
            critique="[Extracted via regex fallback]",
            recommendation=recommendation_match.group(1).upper(),
            raw_response=raw_response,
        )

    # All attempts failed
    raise ResponseValidationError(
        f"Could not extract critique from response: {text[:200]}..."
    )


def _validate_and_build_critique(data: dict, raw: str) -> CritiqueResult:
    """
    Validate JSON schema and build critique result.

    Args:
        data: Parsed JSON data
        raw: Raw response text

    Returns:
        Validated critique

    Raises:
        ValueError: Invalid schema
    """
    # Validate recommendation
    recommendation = data.get("recommendation", "").upper()
    if recommendation not in ("ACCEPT", "REJECT", "UNCERTAIN"):
        raise ValueError(f"Invalid recommendation: {recommendation}")

    # Clamp scores to valid range
    def clamp(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    reasoning_quality = clamp(data.get("reasoning_quality", 0.5))
    technical_alignment = clamp(data.get("technical_alignment", 0.5))
    confidence_calibration = clamp(data.get("confidence_calibration", 0.5))

    return CritiqueResult(
        reasoning_quality=reasoning_quality,
        technical_alignment=technical_alignment,
        confidence_calibration=confidence_calibration,
        critique=str(data.get("critique", "")),
        recommendation=recommendation,
        raw_response=raw,
    )


async def evaluate_signal(
    client: OllamaClient,
    model: str,
    generator_signal: dict,
    market_context: dict,
    task_prompt: str,
    temperature: float = 0.3,
) -> CritiqueResult | None:
    """
    Evaluate generator signal using critic model.

    Args:
        client: Ollama client (must be in async context)
        model: Critic model identifier (e.g., "deepseek-r1:14b")
        generator_signal: Generator signal dict (direction, confidence, reasoning, persona)
        market_context: Market context dict (symbol, price, regime, indicators, recent_ohlcv)
        task_prompt: Original task prompt given to generator
        temperature: Generation temperature (default 0.3 for consistent critique)

    Returns:
        Validated critique, or None if extraction failed

    Example:
        async with OllamaClient() as client:
            critique = await evaluate_signal(
                client, "deepseek-r1:14b", signal, context, prompt
            )
            if critique and critique.score >= 0.7:
                print("Signal accepted")
    """
    # Build critique prompt
    prompt = CRITIQUE_TEMPLATE.format(
        direction=generator_signal.get("direction", "UNKNOWN"),
        confidence=generator_signal.get("confidence", 0.0),
        reasoning=generator_signal.get("reasoning", ""),
        persona=generator_signal.get("persona", "unknown"),
        task_prompt=task_prompt,
        symbol=market_context.get("symbol", ""),
        current_price=market_context.get("current_price", 0.0),
        market_regime=market_context.get("regime", "NEUTRAL"),
        rsi=market_context.get("rsi", 50.0),
        macd=market_context.get("macd", 0.0),
        macd_signal=market_context.get("macd_signal", 0.0),
        bb_position=market_context.get("bb_position", 0.5),
        recent_ohlcv=market_context.get("recent_ohlcv", ""),
    )

    # Generation options — think:False disables CoT reasoning in deepseek-r1,
    # which otherwise routes all output to `thinking` leaving `response` empty
    options = {
        "temperature": temperature,
        "top_p": 0.9,
        "num_predict": 512,
        "think": False,
    }

    logger.info("Evaluating signal", model=model, direction=generator_signal.get("direction"))

    # Generate critique
    response = await client.generate(model, prompt, options)
    raw_text = response["response"]

    # Extract critique
    try:
        critique = extract_critique(raw_text)

        # Validate critique
        if not validate_critique(critique):
            logger.warning("Critique failed validation", recommendation=critique.recommendation)
            # Still return it - validation is advisory
            # Caller can decide whether to use questionable critiques

        logger.info(
            "Critique generated",
            score=critique.score,
            recommendation=critique.recommendation,
            reasoning_quality=critique.reasoning_quality,
            technical_alignment=critique.technical_alignment,
        )

        return critique

    except ResponseValidationError as e:
        logger.error("Critique extraction failed", error=str(e), raw_response=raw_text[:200])
        return None
