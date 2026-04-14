"""
Core async loop for signal generation, critic evaluation, and execution.

This module orchestrates the entire signal generation pipeline:
1. Preflight checks (STOP file, process lock, VRAM)
2. Market data fetching
3. Indicator computation
4. Signal generation (qwen3:8b)
5. Critic evaluation (deepseek-r1:14b)
6. Override logic
7. Signal logging
8. Optional execution
"""

import asyncio
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from loguru import logger

from config.settings import settings
from config.fee_model import FeeModelSettings
from data.indicators import compute_all_indicators, compute_bb_position
from data.market_data import MarketDataService, DataUnavailableError
from data.prompt_builder import TaskType, PromptBuilder, TaskConfig
from data.regime_filter import RegimeClassifier
from execution.models import SignalInput
from signals.accuracy_tracker import queue_for_verification, process_pending_verifications
from signals.preflight import (
    run_preflight_checks,
    check_stop_file,
    enforce_ollama_keep_alive,
)
from signals.signal_logger import log_signal
from signals.signal_models import (
    Signal,
    map_generator_to_signal,
    get_timeframe_duration_ms,
)
from swarm.critic import evaluate_signal, CritiqueResult
from swarm.generator import generate_signal
from swarm.ollama_client import OllamaClient
from training.process_lock import acquire_inference_lock, ProcessLockError

if TYPE_CHECKING:
    from execution.binance_client import BinanceExecutionClient


def should_override_signal(critique: CritiqueResult) -> bool:
    """
    Determine if critic should override signal to FLAT.

    Override conditions:
    1. Recommendation is REJECT AND
    2. Either reasoning_quality < 0.5 OR technical_alignment < 0.5

    Args:
        critique: CritiqueResult from critic evaluation

    Returns:
        True if signal should be overridden to FLAT
    """
    if critique.recommendation != "REJECT":
        return False

    # High-confidence rejection criteria
    if critique.reasoning_quality < 0.5:
        logger.info(
            "Critic override: low reasoning quality",
            reasoning_quality=critique.reasoning_quality,
        )
        return True

    if critique.technical_alignment < 0.5:
        logger.info(
            "Critic override: low technical alignment",
            technical_alignment=critique.technical_alignment,
        )
        return True

    return False


def format_recent_ohlcv(df, n: int = 5) -> str:
    """Format recent OHLCV data for critic prompt."""
    recent = df.tail(n)
    lines = []
    for _, row in recent.iterrows():
        ts = datetime.fromtimestamp(row["timestamp"] / 1000, tz=timezone.utc)
        lines.append(
            f"{ts.strftime('%H:%M')} | O:{row['open']:.2f} H:{row['high']:.2f} "
            f"L:{row['low']:.2f} C:{row['close']:.2f}"
        )
    return "\n".join(lines)


async def generate_signal_for_symbol(
    symbol: str,
    timeframe: str,
    market_data_service: MarketDataService,
    ollama_client: OllamaClient,
    fee_model: FeeModelSettings,
    prompt_builder: PromptBuilder,
) -> tuple[Signal | None, str | None]:
    """
    Generate signal for a single symbol.

    Args:
        symbol: Trading pair
        timeframe: Timeframe
        market_data_service: Market data service
        ollama_client: Ollama client
        fee_model: Fee model settings
        prompt_builder: Prompt builder

    Returns:
        Tuple of (Signal or None, task_prompt or None)
    """
    # 1. Fetch market data
    try:
        df = await market_data_service.fetch_ohlcv(
            symbol, timeframe, lookback_bars=100
        )
    except DataUnavailableError as e:
        logger.error("Failed to fetch data", symbol=symbol, error=str(e))
        return None, None

    if len(df) < 50:
        logger.warning("Insufficient data", symbol=symbol, bars=len(df))
        return None, None

    # 2. Compute indicators
    indicators = compute_all_indicators(df, include_volume=True, include_structure=True)

    # 3. Get market regime
    classifier = RegimeClassifier()
    regime = classifier.get_current_regime(df["close"])

    # 4. Build prompt
    task = TaskConfig(
        task_type=TaskType.PREDICT_DIRECTION,
        weight=1.0,
        difficulty=2,
        min_bars_required=50,
    )

    prompt = prompt_builder.build_prompt(
        task=task,
        df=df,
        symbol=symbol,
        timeframe=timeframe,
        market_regime=regime,
        fee_model=fee_model,
    )

    # 5. Generate signal
    logger.info("Generating signal", symbol=symbol, regime=regime.value)

    generator_signal = await generate_signal(
        client=ollama_client,
        model=settings.ollama.generator_model,
        prompt=prompt,
        regime=regime,
        task_type=TaskType.PREDICT_DIRECTION,
        temperature=0.7,
    )

    if generator_signal is None:
        logger.warning("Signal generation failed", symbol=symbol)
        return None, prompt

    # 6. Map to execution format
    direction = map_generator_to_signal(generator_signal.direction or "FLAT")
    current_price = float(df["close"].iloc[-1])

    signal = Signal(
        symbol=symbol,
        timeframe=timeframe,
        direction=direction,
        confidence=generator_signal.confidence or 0.5,
        reasoning=generator_signal.reasoning,
        persona=generator_signal.persona.value,
        timestamp=datetime.now(timezone.utc),
        market_regime=regime.value,
        current_price=current_price,
        rsi=indicators.get("rsi", 50.0),
        macd=indicators.get("macd_line", 0.0),
        macd_signal=indicators.get("macd_signal", 0.0),
        bb_position=float(compute_bb_position(df["close"]).iloc[-1]),
    )

    return signal, prompt


async def evaluate_with_critic(
    signal: Signal,
    df,
    prompt: str,
    ollama_client: OllamaClient,
) -> Signal:
    """
    Evaluate signal with critic and apply override if needed.

    Args:
        signal: Signal to evaluate
        df: OHLCV DataFrame
        prompt: Original task prompt
        ollama_client: Ollama client

    Returns:
        Signal with critic evaluation results
    """
    # Build market context for critic
    market_context = {
        "symbol": signal.symbol,
        "current_price": signal.current_price,
        "regime": signal.market_regime,
        "rsi": signal.rsi,
        "macd": signal.macd,
        "macd_signal": signal.macd_signal,
        "bb_position": signal.bb_position,
        "recent_ohlcv": format_recent_ohlcv(df, n=5),
    }

    generator_signal_dict = {
        "direction": signal.direction,
        "confidence": signal.confidence,
        "reasoning": signal.reasoning,
        "persona": signal.persona,
    }

    # Evaluate
    critique = await evaluate_signal(
        client=ollama_client,
        model=settings.ollama.critic_model,
        generator_signal=generator_signal_dict,
        market_context=market_context,
        task_prompt=prompt,
        temperature=0.3,
    )

    if critique is None:
        logger.warning("Critique failed, keeping original signal")
        signal.final_direction = signal.direction
        return signal

    # Update signal with critic results
    signal.critic_score = critique.score
    signal.critic_recommendation = critique.recommendation
    signal.critic_reasoning = critique.critique

    # Check for override
    if should_override_signal(critique):
        signal.critic_override = True
        signal.final_direction = "FLAT"
        logger.info(
            "Signal overridden to FLAT by critic",
            original=signal.direction,
            critic_score=critique.score,
        )
    else:
        signal.final_direction = signal.direction

    return signal


def print_signal_summary(signal: Signal, executed: bool, trade_reason: str | None):
    """Print concise signal summary to stdout."""
    override_marker = " [OVERRIDDEN]" if signal.critic_override else ""
    exec_marker = " -> EXECUTED" if executed else ""

    print(f"\n{'='*60}")
    print(f"SIGNAL: {signal.symbol} @ {signal.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"{'='*60}")
    print(f"Direction:  {signal.direction} -> {signal.final_direction}{override_marker}")
    print(f"Confidence: {signal.confidence:.1%}")
    print(f"Persona:    {signal.persona}")
    print(f"Regime:     {signal.market_regime}")
    print(f"Price:      ${signal.current_price:,.2f}")

    if signal.critic_score is not None:
        print(f"Critic:     {signal.critic_recommendation} (score: {signal.critic_score:.2f})")

    if trade_reason:
        print(f"Trade:      {trade_reason}")

    print(f"{'='*60}{exec_marker}")


async def run_cycle(
    symbols: list[str],
    timeframe: str,
    execute: bool,
    min_confidence: float,
    execution_client: "BinanceExecutionClient | None" = None,
) -> list[Signal]:
    """
    Run one complete signal generation cycle for all symbols.

    Args:
        symbols: List of trading pairs
        timeframe: Timeframe
        execute: Whether to execute trades
        min_confidence: Minimum confidence for execution
        execution_client: Optional execution client

    Returns:
        List of generated signals
    """
    signals = []
    fee_model = settings.fee_model
    prompt_builder = PromptBuilder()

    # Enforce OLLAMA_KEEP_ALIVE=0
    enforce_ollama_keep_alive()

    async with MarketDataService() as market_data_service:
        async with OllamaClient() as ollama_client:
            # Process pending accuracy verifications first
            try:
                await process_pending_verifications(market_data_service)
            except Exception as e:
                logger.warning("Accuracy verification failed", error=str(e))

            for symbol in symbols:
                # Check STOP file between symbols
                if check_stop_file():
                    logger.warning("STOP file detected, halting cycle")
                    break

                logger.info("Processing symbol", symbol=symbol, timeframe=timeframe)

                # Generate signal
                signal, prompt = await generate_signal_for_symbol(
                    symbol=symbol,
                    timeframe=timeframe,
                    market_data_service=market_data_service,
                    ollama_client=ollama_client,
                    fee_model=fee_model,
                    prompt_builder=prompt_builder,
                )

                if signal is None:
                    continue

                # Unload generator, load critic
                await ollama_client.unload_current()

                # Re-fetch data for critic context (could cache but keeps it simple)
                try:
                    df = await market_data_service.fetch_ohlcv(
                        symbol, timeframe, lookback_bars=100
                    )
                except DataUnavailableError:
                    signal.final_direction = signal.direction
                    df = None

                # Evaluate with critic if we have data
                if df is not None and prompt is not None:
                    signal = await evaluate_with_critic(
                        signal=signal,
                        df=df,
                        prompt=prompt,
                        ollama_client=ollama_client,
                    )

                    # Unload critic
                    await ollama_client.unload_current()

                # Queue for accuracy verification
                queue_for_verification(signal, signal.current_price)

                # Determine execution
                executed = False
                trade_reason = None

                if execute and execution_client is not None:
                    if signal.final_direction == "FLAT":
                        trade_reason = "Signal is FLAT - no trade"
                    elif signal.confidence < min_confidence:
                        trade_reason = f"Confidence {signal.confidence:.1%} below threshold {min_confidence:.1%}"
                    else:
                        # Build signal input for execution
                        signal_input = SignalInput(
                            symbol=symbol,
                            direction="long" if signal.final_direction == "LONG" else "short",
                            confidence=signal.confidence,
                            expected_return_pct=0.5,  # Conservative estimate
                            stop_loss_pct=2.0,  # Default stop loss
                            take_profit_pct=4.0,  # 2:1 R:R
                            timeframe=timeframe,
                            entry_price=signal.current_price,
                        )

                        try:
                            decision = await execution_client.accept_signal(signal_input)
                            trade_reason = decision.reason

                            if decision.execute:
                                # Place order
                                order = await execution_client.place_market_order(
                                    symbol=decision.symbol,
                                    side=decision.side,
                                    amount=decision.amount,
                                )
                                executed = True
                                trade_reason = f"Order placed: {order.order_id}"
                                logger.info(
                                    "Order executed",
                                    symbol=symbol,
                                    side=decision.side,
                                    amount=decision.amount,
                                    order_id=order.order_id,
                                )
                        except Exception as e:
                            trade_reason = f"Execution error: {str(e)}"
                            logger.error("Execution failed", error=str(e))
                else:
                    trade_reason = "Execution disabled" if not execute else "No execution client"

                # Log signal
                log_signal(signal, executed=executed, trade_reason=trade_reason)

                # Print summary
                print_signal_summary(signal, executed, trade_reason)

                signals.append(signal)

    return signals


async def run_loop(
    symbols: list[str],
    timeframe: str,
    execute: bool,
    min_confidence: float = 0.6,
    once: bool = False,
    execution_client: "BinanceExecutionClient | None" = None,
) -> None:
    """
    Main async loop for signal generation.

    Runs cycles on schedule aligned to timeframe bar closes.

    Args:
        symbols: List of trading pairs
        timeframe: Timeframe
        execute: Whether to execute trades
        min_confidence: Minimum confidence for execution
        once: Run single cycle and exit
        execution_client: Optional execution client
    """
    bar_duration_ms = get_timeframe_duration_ms(timeframe)
    bar_duration_s = bar_duration_ms / 1000

    logger.info(
        "Starting signal loop",
        symbols=symbols,
        timeframe=timeframe,
        execute=execute,
        once=once,
    )

    while True:
        # Check STOP file first
        if check_stop_file():
            logger.warning("STOP file detected, halting signal loop")
            break

        cycle_start = datetime.now(timezone.utc)

        # Run preflight checks
        preflight = run_preflight_checks()
        if not preflight.passed:
            logger.warning(
                "Preflight failed, waiting to retry",
                reason=preflight.reason,
            )
            if once:
                logger.error("Preflight failed in --once mode, exiting")
                break
            await asyncio.sleep(60)  # Wait 1 minute and retry
            continue

        # Acquire inference lock for the cycle
        try:
            with acquire_inference_lock():
                signals = await run_cycle(
                    symbols=symbols,
                    timeframe=timeframe,
                    execute=execute,
                    min_confidence=min_confidence,
                    execution_client=execution_client,
                )

                logger.info(
                    "Cycle complete",
                    signals_generated=len(signals),
                    duration_s=(datetime.now(timezone.utc) - cycle_start).total_seconds(),
                )

        except ProcessLockError as e:
            logger.warning("Process lock unavailable", reason=str(e))
            if once:
                logger.error("Lock failed in --once mode, exiting")
                break
            await asyncio.sleep(60)
            continue

        if once:
            logger.info("Single cycle complete, exiting")
            break

        # Calculate sleep until next bar
        cycle_duration = (datetime.now(timezone.utc) - cycle_start).total_seconds()
        sleep_duration = max(0, bar_duration_s - cycle_duration)

        if sleep_duration > 0:
            logger.info(f"Sleeping {sleep_duration:.0f}s until next cycle")
            await asyncio.sleep(sleep_duration)
