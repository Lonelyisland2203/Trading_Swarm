"""
Swarm workflow orchestrator for Generator -> Critic pipeline.

Coordinates signal generation, critique, and training data capture.
Implements VRAM-safe model switching without LangGraph dependency.
"""

from dataclasses import asdict
from typing import TypedDict

import pandas as pd
from loguru import logger

from config.settings import settings
from data.indicators import compute_bb_position, compute_macd, compute_rsi
from data.prompt_builder import TaskType, PromptBuilder, sample_task
from data.regime_filter import MarketRegime
from .critic import evaluate_signal
from .generator import generate_signal, TradingPersona
from .ollama_client import OllamaClient
from .training_capture import TrainingExample

# Regime-aware acceptance thresholds
ACCEPTANCE_THRESHOLDS = {
    MarketRegime.RISK_OFF: 0.60,
    MarketRegime.NEUTRAL: 0.55,
    MarketRegime.RISK_ON: 0.50,
}


class SwarmState(TypedDict):
    """
    Workflow state - must be JSON-serializable for potential future checkpointing.

    Note: DataFrames are NOT stored in state. Use timestamp_ms + symbol
    to reproduce data via get_ohlcv_as_of().
    """

    # Input identifiers (used to fetch data, not store it)
    symbol: str
    timeframe: str
    timestamp_ms: int  # Point-in-time for reproducibility

    # Pre-computed market context (serializable)
    market_context: dict  # Contains: regime, rsi, macd, bb_position, current_price, recent_ohlcv
    task_prompt: str  # The rendered prompt from PromptBuilder

    # Generator outputs
    generator_signal: dict | None  # Serialized GeneratorSignal
    generator_error: str | None

    # Critic outputs
    critique: dict | None  # Serialized CritiqueResult
    critic_error: str | None

    # Final decision
    final_signal: dict | None
    workflow_status: str  # "success" | "generator_failed" | "critic_failed" | "rejected"
    acceptance_reason: str


def _build_market_context(df: pd.DataFrame, regime: MarketRegime) -> dict:
    """
    Build serializable market context from OHLCV data.

    Args:
        df: OHLCV DataFrame
        regime: Market regime

    Returns:
        Dict with indicators and recent price action
    """
    # Compute indicators
    rsi = compute_rsi(df["close"]).iloc[-1]
    macd_line, macd_signal, _ = compute_macd(df["close"])
    bb_pos = compute_bb_position(df["close"]).iloc[-1]

    # Format recent price action (last 5 bars)
    recent = df.tail(5)
    ohlcv_lines = []
    for _, row in recent.iterrows():
        timestamp = pd.to_datetime(row["timestamp"], unit="ms")
        change = ((row["close"] - row["open"]) / row["open"]) * 100
        direction = "[UP]" if change > 0 else "[DOWN]"

        ohlcv_lines.append(
            f"{timestamp.strftime('%Y-%m-%d %H:%M')} | "
            f"O: ${row['open']:.2f} H: ${row['high']:.2f} "
            f"L: ${row['low']:.2f} C: ${row['close']:.2f} | "
            f"{direction} {change:+.2f}%"
        )

    return {
        "symbol": "",  # Filled by caller
        "current_price": float(df["close"].iloc[-1]),
        "regime": regime.value,
        "rsi": float(rsi) if not pd.isna(rsi) else 50.0,
        "macd": float(macd_line.iloc[-1]) if not pd.isna(macd_line.iloc[-1]) else 0.0,
        "macd_signal": float(macd_signal.iloc[-1]) if not pd.isna(macd_signal.iloc[-1]) else 0.0,
        "bb_position": float(bb_pos) if not pd.isna(bb_pos) else 0.5,
        "recent_ohlcv": "\n".join(ohlcv_lines),
    }


def should_accept_signal(
    critique: dict,
    regime: MarketRegime,
    override_threshold: float | None = None,
) -> tuple[bool, str]:
    threshold = override_threshold or ACCEPTANCE_THRESHOLDS[regime]

    # Compute score from sub-dimensions — CritiqueResult.score is a @property
    # and dataclasses.asdict() drops properties, so we cannot rely on "score" key.
    score = (
        0.35 * critique.get("reasoning_quality", 0.0)
        + 0.40 * critique.get("technical_alignment", 0.0)
        + 0.25 * critique.get("confidence_calibration", 0.0)
    )
    recommendation = critique.get("recommendation", "UNCERTAIN")
    technical_alignment = critique.get("technical_alignment", 0.0)

    # Hard rejection ONLY if score is also low — don't trust label alone
    if recommendation == "REJECT" and score < 0.45:
        return False, f"Rejected: low score {score:.2f} and REJECT recommendation"

    # Score-based filtering (primary gate)
    if score < threshold:
        return False, f"Score {score:.2f} below threshold {threshold}"

    # Technical alignment gate
    if technical_alignment < 0.35:  # Slightly relaxed from 0.4
        return False, f"Technical alignment too low: {technical_alignment:.2f}"

    return True, f"Score {score:.2f} above threshold {threshold}, alignment {technical_alignment:.2f}"


async def run_swarm_workflow(
    symbol: str,
    timeframe: str,
    ohlcv_df: pd.DataFrame,
    market_regime: MarketRegime,
    task_prompt: str,
    task_type: TaskType = TaskType.PREDICT_DIRECTION,
    higher_tf_data: dict[str, pd.DataFrame] | None = None,
) -> tuple[SwarmState, TrainingExample]:
    """
    Execute generator -> critic workflow with guaranteed VRAM cleanup.

    VRAM Safety:
    - OllamaClient context manager guarantees cleanup on any exit
    - Explicit unload_current() between model switches
    - keep_alive=0 validated at client initialization

    Args:
        symbol: Trading pair
        timeframe: Candle timeframe
        ohlcv_df: OHLCV DataFrame with sufficient history
        market_regime: Current market regime
        task_prompt: Rendered prompt from PromptBuilder
        task_type: Task type for signal generation
        higher_tf_data: Optional dict of higher timeframe OHLCV DataFrames
                       Keys are timeframe strings (e.g., "4h", "1d")
                       Values are OHLCV DataFrames

    Returns:
        Tuple of (final_state, training_example)

    Example:
        state, example = await run_swarm_workflow(
            "BTC/USDT", "1h", df, MarketRegime.NEUTRAL, prompt
        )
        if state["workflow_status"] == "success":
            print(f"Signal: {state['final_signal']['direction']}")
    """
    # Build market context
    market_context = _build_market_context(ohlcv_df, market_regime)
    market_context["symbol"] = symbol

    # If higher_tf_data provided, rebuild prompt with multi-timeframe context
    if higher_tf_data is not None:
        logger.info(
            "Rebuilding prompt with higher timeframe context",
            timeframe=timeframe,
            higher_tfs=list(higher_tf_data.keys()),
        )

        # Sample task based on available data
        task = sample_task(available_bars=len(ohlcv_df))

        # Build prompt with higher TF context
        builder = PromptBuilder()
        task_prompt = builder.build_prompt(
            task=task,
            df=ohlcv_df,
            symbol=symbol,
            timeframe=timeframe,
            market_regime=market_regime,
            higher_tf_data=higher_tf_data,
        )

        logger.debug(
            "Prompt rebuilt with higher TF context",
            prompt_length=len(task_prompt),
        )

    # Initialize state
    state: SwarmState = {
        "symbol": symbol,
        "timeframe": timeframe,
        "timestamp_ms": int(ohlcv_df["timestamp"].iloc[-1]),
        "market_context": market_context,
        "task_prompt": task_prompt,
        "generator_signal": None,
        "generator_error": None,
        "critique": None,
        "critic_error": None,
        "final_signal": None,
        "workflow_status": "running",
        "acceptance_reason": "",
    }

    # Initialize training example
    training_example = TrainingExample(
        symbol=symbol,
        timeframe=timeframe,
        timestamp_ms=state["timestamp_ms"],
        market_regime=market_regime.value,
        indicators=market_context,
        task_prompt=task_prompt,
    )

    async with OllamaClient() as client:
        try:
            # ===== GENERATOR PHASE =====
            logger.info("Starting generator phase", symbol=symbol, regime=market_regime.value)

            signal = await generate_signal(
                client,
                settings.ollama.generator_model,
                task_prompt,
                market_regime,
                task_type,
                temperature=settings.swarm.generator_temperature,
            )

            if signal is None:
                # generate_signal returns None after exhausting retries
                state["generator_error"] = "Signal extraction failed after retry"
                state["workflow_status"] = "generator_failed"
                training_example.rejection_reason = "Generator extraction failed"
                logger.warning("Generator failed", symbol=symbol)
                return state, training_example

            # Capture generator output
            state["generator_signal"] = asdict(signal)
            training_example.persona = signal.persona.value
            training_example.generator_signal = asdict(signal)
            training_example.generator_raw_response = signal.raw_response
            training_example.full_generator_prompt = task_prompt  # Persona already included

            logger.info(
                "Generator complete",
                direction=signal.direction,
                confidence=signal.confidence,
                persona=signal.persona.value,
            )

            # CRITICAL: Unload generator before loading critic
            await client.unload_current()

            # ===== CRITIC PHASE =====
            logger.info("Starting critic phase", symbol=symbol)

            critique = await evaluate_signal(
                client,
                settings.ollama.critic_model,
                state["generator_signal"],
                market_context,
                task_prompt,
                temperature=settings.swarm.critic_temperature,
            )

            if critique is None:
                # Extraction failed - accept generator signal with flag
                state["critic_error"] = "Critique extraction failed"
                state["workflow_status"] = "critic_failed"
                state["final_signal"] = state["generator_signal"]
                state["acceptance_reason"] = "Accepted by default (critic failed)"
                training_example.was_accepted = True
                training_example.acceptance_reason = "Critic failed - accepted by default"
                logger.warning("Critic failed, accepting by default", symbol=symbol)
                return state, training_example

            # Capture critic output
            state["critique"] = asdict(critique)
            training_example.critique = asdict(critique)
            training_example.critic_raw_response = critique.raw_response
            training_example.critique_prompt = task_prompt  # Simplified - full prompt too long

            logger.info(
                "Critic complete",
                score=critique.score,
                recommendation=critique.recommendation,
            )

            # ===== ACCEPTANCE DECISION =====
            accepted, reason = should_accept_signal(state["critique"], market_regime)

            if accepted:
                state["workflow_status"] = "success"
                state["final_signal"] = state["generator_signal"]
                state["acceptance_reason"] = reason
                training_example.was_accepted = True
                training_example.acceptance_reason = reason
                logger.info("Signal accepted", reason=reason)
            else:
                state["workflow_status"] = "rejected"
                state["final_signal"] = None
                state["acceptance_reason"] = reason
                training_example.was_accepted = False
                training_example.rejection_reason = reason
                logger.info("Signal rejected", reason=reason)

            return state, training_example

        except Exception as e:
            # Unexpected error - log and mark as failed
            logger.error("Workflow exception", error=str(e), symbol=symbol)
            state["workflow_status"] = "error"
            state["generator_error"] = str(e)
            training_example.rejection_reason = f"Workflow error: {str(e)}"
            return state, training_example

        finally:
            # Defensive: ensure unload even if exception in workflow
            # Note: __aexit__ already calls unload_current(), but this is belt-and-suspenders
            await client.unload_current()


def compute_final_confidence(
    generator_confidence: float,
    critic_score: float,
    alpha: float = 0.6,
) -> float:
    """
    Blend generator confidence with critic evaluation.

    High critic score validates generator confidence.
    Low critic score reduces effective confidence.

    Args:
        generator_confidence: Generator's confidence (0.0-1.0)
        critic_score: Critic's overall score (0.0-1.0)
        alpha: Weight on critic score (default 0.6)

    Returns:
        Blended confidence (0.0-1.0)

    Example:
        >>> compute_final_confidence(0.8, 0.9)  # Both high
        0.86  # High final confidence
        >>> compute_final_confidence(0.8, 0.3)  # Critic skeptical
        0.50  # Reduced final confidence
    """
    return alpha * critic_score + (1 - alpha) * generator_confidence


async def run_multi_persona_workflow(
    symbol: str,
    timeframe: str,
    ohlcv_df: pd.DataFrame,
    market_regime: MarketRegime,
    task_prompt: str,
    task_type: TaskType = TaskType.PREDICT_DIRECTION,
) -> tuple[dict, list[TrainingExample]]:
    """
    Generate signals from ALL personas for the same context (DPO training).

    This enables preference pair construction by ranking signals by computed rewards.
    All personas receive the SAME task_prompt and market context, ensuring valid
    same-prompt preference pairs for DPO.

    Workflow per persona:
    1. Generate signal using persona-specific prompt
    2. Critique signal independently
    3. Record acceptance decision
    4. Unload model before next persona

    Graceful degradation:
    - Generator failure for a persona → skip, continue with others
    - Critic failure → accept signal with flag
    - All generators failed → return empty list

    Args:
        symbol: Trading pair (e.g., "BTC/USDT")
        timeframe: Candle timeframe (e.g., "1h")
        ohlcv_df: OHLCV DataFrame with sufficient history
        market_regime: Current market regime
        task_prompt: Rendered prompt from PromptBuilder (same for all personas)

    Returns:
        Tuple of (summary_state, training_examples):
        - summary_state: Workflow summary with counts
        - training_examples: List of TrainingExample (one per successful generation)

    Summary state keys:
        - workflow_status: "success" | "partial_success" | "all_failed"
        - personas_attempted: Number of personas attempted (always 5)
        - signals_generated: Number of signals successfully generated
        - signals_accepted: Number of signals passing acceptance threshold
        - errors: List of error messages per persona

    Example:
        >>> state, examples = await run_multi_persona_workflow(
        ...     "BTC/USDT", "1h", df, MarketRegime.NEUTRAL, prompt
        ... )
        >>> print(f"Generated {len(examples)} signals for DPO training")
        >>> # Rank by rewards to create preference pairs
        >>> sorted_examples = sorted(examples, key=lambda e: compute_reward(e))
    """
    # Build market context once (shared by all personas)
    market_context = _build_market_context(ohlcv_df, market_regime)
    market_context["symbol"] = symbol
    timestamp_ms = int(ohlcv_df["timestamp"].iloc[-1])

    # Generate context ID (shared across all personas for DPO pairing)
    import uuid
    context_id = str(uuid.uuid4())

    # Initialize summary
    summary = {
        "workflow_status": "running",
        "personas_attempted": 0,
        "signals_generated": 0,
        "signals_accepted": 0,
        "errors": [],
    }

    training_examples: list[TrainingExample] = []

    async with OllamaClient() as client:
        # Loop through all personas
        for persona in TradingPersona:
            summary["personas_attempted"] += 1

            logger.info(
                "Generating signal",
                symbol=symbol,
                persona=persona.value,
                regime=market_regime.value,
            )

            try:
                # ===== GENERATOR PHASE =====
                signal = await generate_signal(
                    client,
                    settings.ollama.generator_model,
                    task_prompt,
                    market_regime,
                    task_type,
                    temperature=0.7,
                    persona_override=persona,  # Force this persona
                )

                # Unload generator before critic
                await client.unload_current()

                if signal is None:
                    error_msg = f"{persona.value}: Signal generation failed"
                    summary["errors"].append(error_msg)
                    logger.warning(error_msg)
                    continue

                summary["signals_generated"] += 1

                # Initialize training example for this persona
                training_example = TrainingExample(
                    symbol=symbol,
                    timeframe=timeframe,
                    timestamp_ms=timestamp_ms,
                    market_regime=market_regime.value,
                    indicators=market_context,
                    task_prompt=task_prompt,
                    persona=persona.value,  # Set persona field
                    context_id=context_id,  # Shared across all personas from this context
                    generator_signal=asdict(signal),
                    generator_raw_response=signal.raw_response,
                    full_generator_prompt=task_prompt,  # Base prompt (persona already in signal)
                )

                # ===== CRITIC PHASE =====
                logger.info(
                    "Starting critic",
                    symbol=symbol,
                    persona=persona.value,
                    signal_direction=signal.direction,
                )

                critique = await evaluate_signal(
                    client,
                    settings.ollama.critic_model,
                    asdict(signal),
                    market_context,
                    task_prompt,
                    temperature=0.0,
                )

                # Unload critic before next persona
                await client.unload_current()

                if critique is None:
                    # Critic failed - accept by default with flag
                    training_example.was_accepted = True
                    training_example.acceptance_reason = f"Critic failed for {persona.value} - accepted by default"
                    training_example.critic_error = "Critique extraction failed"
                    logger.warning(
                        "Critic failed, accepting by default",
                        persona=persona.value,
                    )
                else:
                    # Capture critique
                    training_example.critique = asdict(critique)
                    training_example.critic_raw_response = critique.raw_response
                    training_example.critique_prompt = task_prompt

                    # ===== ACCEPTANCE DECISION =====
                    accepted, reason = should_accept_signal(
                        asdict(critique),
                        market_regime,
                    )

                    training_example.was_accepted = accepted
                    if accepted:
                        training_example.acceptance_reason = reason
                        summary["signals_accepted"] += 1
                        logger.info(
                            "Signal accepted",
                            persona=persona.value,
                            reason=reason,
                        )
                    else:
                        training_example.rejection_reason = reason
                        logger.info(
                            "Signal rejected",
                            persona=persona.value,
                            reason=reason,
                        )

                # Add to training examples regardless of acceptance
                # (DPO needs both accepted and rejected for preference pairs)
                training_examples.append(training_example)

            except Exception as e:
                error_msg = f"{persona.value}: Unexpected error - {str(e)}"
                summary["errors"].append(error_msg)
                logger.error(
                    "Persona workflow failed",
                    persona=persona.value,
                    error=str(e),
                )
                # Unload defensively
                await client.unload_current()
                continue

    # Determine final status
    if summary["signals_generated"] == 0:
        summary["workflow_status"] = "all_failed"
    elif summary["signals_generated"] < summary["personas_attempted"]:
        summary["workflow_status"] = "partial_success"
    else:
        summary["workflow_status"] = "success"

    logger.info(
        "Multi-persona workflow complete",
        status=summary["workflow_status"],
        generated=summary["signals_generated"],
        accepted=summary["signals_accepted"],
        total_examples=len(training_examples),
    )

    return summary, training_examples
