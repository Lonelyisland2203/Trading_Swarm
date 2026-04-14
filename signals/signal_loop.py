"""
Core async loop for signal generation with XGBoost + LLM context synthesis.

Session 17N: Refactored pipeline:
1. Preflight checks (STOP file, process lock, VRAM)
2. XGBoost signal generation (probability + direction)
3. LLM context generation (Qwen: bullish/bearish factors, regime)
4. DeepSeek risk filter (APPROVE/VETO)
5. Synthesis node (combines all inputs into final decision)
6. Signal logging
7. Optional execution

REMOVED: Qwen generator producing LONG/SHORT thesis
REMOVED: DeepSeek critic evaluating thesis quality
ADDED: XGBoost core signal, LLM context overlay, synthesis node
"""

import asyncio
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from loguru import logger

from data.market_data import MarketDataService
from execution.models import SignalInput
from signals.accuracy_tracker import queue_for_verification, process_pending_verifications
from signals.llm_context import generate_market_context, LLMContext
from signals.preflight import (
    run_preflight_checks,
    check_stop_file,
    enforce_ollama_keep_alive,
)
from signals.signal_logger import log_signal
from signals.signal_models import (
    Signal,
    get_timeframe_duration_ms,
)
from signals.synthesis import SynthesisInput, SynthesisOutput, synthesize
from signals.xgboost_signal import generate_xgboost_signal, XGBoostSignal
from training.process_lock import acquire_inference_lock, ProcessLockError

if TYPE_CHECKING:
    from execution.binance_client import BinanceExecutionClient


# DeepSeek risk filter configuration
DEEPSEEK_MODEL = "deepseek-r1:14b"
DEEPSEEK_TIMEOUT = 60.0


async def call_risk_filter(
    xgb_signal: XGBoostSignal,
    llm_context: LLMContext | None,
) -> bool:
    """
    Call DeepSeek risk filter to veto or approve trade.

    Args:
        xgb_signal: XGBoost signal with direction and probability
        llm_context: LLM market context (may be None)

    Returns:
        True if trade is vetoed, False if approved
    """
    import httpx

    # Build context for risk filter
    context_parts = [
        f"XGBoost Signal: {xgb_signal.direction} (probability={xgb_signal.probability:.2f})",
        f"Symbol: {xgb_signal.symbol}, Timeframe: {xgb_signal.timeframe}",
    ]

    if llm_context:
        if llm_context.bullish_factors:
            context_parts.append(f"Bullish factors: {', '.join(llm_context.bullish_factors)}")
        if llm_context.bearish_factors:
            context_parts.append(f"Bearish factors: {', '.join(llm_context.bearish_factors)}")
        context_parts.append(f"Regime: {llm_context.regime_flag}")

    prompt = f"""Given this market context and XGBoost signal, are there any red flags that should veto this trade?

{chr(10).join(context_parts)}

Respond with exactly one word: APPROVE or VETO
Then provide one sentence of reasoning."""

    payload = {
        "model": DEEPSEEK_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 100,
        },
    }

    try:
        async with httpx.AsyncClient(timeout=DEEPSEEK_TIMEOUT) as client:
            response = await client.post(
                "http://localhost:11434/api/generate",
                json=payload,
            )
            response.raise_for_status()
            result = response.json()
            response_text = result.get("response", "").strip().upper()

            # Parse response
            veto = response_text.startswith("VETO")

            if veto:
                logger.info(
                    "Risk filter VETO",
                    symbol=xgb_signal.symbol,
                    response=response_text[:100],
                )
            else:
                logger.debug(
                    "Risk filter APPROVE",
                    symbol=xgb_signal.symbol,
                )

            return veto

    except Exception as e:
        logger.warning(f"Risk filter call failed: {e}, defaulting to APPROVE")
        return False  # Don't veto on failure


def print_signal_summary(
    synthesis_output: SynthesisOutput,
    xgb_signal: XGBoostSignal,
    executed: bool,
    trade_reason: str | None,
):
    """Print concise signal summary to stdout."""
    veto_marker = (
        " [VETOED]"
        if synthesis_output.direction == "FLAT" and synthesis_output.position_size_fraction == 0
        else ""
    )
    exec_marker = " -> EXECUTED" if executed else ""

    print(f"\n{'=' * 60}")
    print(f"SIGNAL: {xgb_signal.symbol} @ {xgb_signal.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"{'=' * 60}")
    print(f"XGBoost:    {xgb_signal.direction} (prob={xgb_signal.probability:.2f})")
    print(
        f"Final:      {synthesis_output.direction} (position={synthesis_output.position_size_fraction:.0%}){veto_marker}"
    )
    print(f"Rationale:  {synthesis_output.rationale}")

    if trade_reason:
        print(f"Trade:      {trade_reason}")

    print(f"{'=' * 60}{exec_marker}")


def synthesis_to_legacy_signal(
    xgb_signal: XGBoostSignal,
    synthesis_output: SynthesisOutput,
    llm_context: LLMContext | None,
    critic_veto: bool,
) -> Signal:
    """
    Convert synthesis output to legacy Signal format for logging compatibility.

    This maintains backward compatibility with existing signal_logger.py and
    accuracy_tracker.py which expect the Signal dataclass.
    """
    return Signal(
        symbol=xgb_signal.symbol,
        timeframe=xgb_signal.timeframe,
        direction=xgb_signal.direction,  # Original XGBoost direction
        confidence=xgb_signal.confidence,
        reasoning=synthesis_output.rationale,
        persona="XGBOOST",  # New source identifier
        timestamp=xgb_signal.timestamp,
        market_regime=llm_context.regime_flag if llm_context else "unknown",
        current_price=0.0,  # Not available from XGBoost signal
        rsi=xgb_signal.features.get("rsi", 50.0) or 50.0,
        macd=xgb_signal.features.get("macd_line", 0.0) or 0.0,
        macd_signal=xgb_signal.features.get("macd_signal", 0.0) or 0.0,
        bb_position=xgb_signal.features.get("bb_position", 0.5) or 0.5,
        critic_score=None,
        critic_recommendation="VETO" if critic_veto else "APPROVE",
        critic_override=critic_veto,
        critic_reasoning=None,
        final_direction=synthesis_output.direction,
    )


async def run_cycle(
    symbols: list[str],
    timeframe: str,
    execute: bool,
    min_confidence: float,
    execution_client: "BinanceExecutionClient | None" = None,
) -> list[Signal]:
    """
    Run one complete signal generation cycle for all symbols.

    New pipeline (Session 17N):
    1. Generate XGBoost signal (probability + direction)
    2. Generate LLM context (bullish/bearish factors, regime)
    3. Call DeepSeek risk filter (APPROVE/VETO)
    4. Synthesize final decision
    5. Log and optionally execute

    Args:
        symbols: List of trading pairs
        timeframe: Timeframe
        execute: Whether to execute trades
        min_confidence: Minimum confidence for execution
        execution_client: Optional execution client

    Returns:
        List of generated signals (legacy format for compatibility)
    """
    signals = []

    # Enforce OLLAMA_KEEP_ALIVE=0
    enforce_ollama_keep_alive()

    async with MarketDataService() as market_data_service:
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

            # Get current timestamp for point-in-time safety
            as_of = int(datetime.now(timezone.utc).timestamp() * 1000)

            # Step 1: Generate XGBoost signal
            xgb_signal = await generate_xgboost_signal(
                symbol=symbol,
                timeframe=timeframe,
                as_of=as_of,
                market_data_service=market_data_service,
            )

            if xgb_signal is None:
                logger.warning("XGBoost signal generation failed", symbol=symbol)
                continue

            logger.info(
                "XGBoost signal generated",
                symbol=symbol,
                direction=xgb_signal.direction,
                probability=xgb_signal.probability,
            )

            # Step 2: Generate LLM context
            # Note: In production, we'd fetch funding rate, OI, etc. from market data
            # For now, using placeholder values
            llm_context: LLMContext | None = None
            try:
                llm_context = await generate_market_context(
                    funding_rate=None,  # TODO: Fetch from market data
                    oi_delta=None,  # TODO: Fetch from market data
                    liquidation_data=None,
                    news_headlines=None,
                )
                logger.debug(
                    "LLM context generated",
                    regime_flag=llm_context.regime_flag,
                    confidence=llm_context.confidence,
                )
            except Exception as e:
                logger.warning(f"LLM context generation failed: {e}")
                llm_context = None

            # Step 3: Call DeepSeek risk filter
            critic_veto = False
            try:
                critic_veto = await call_risk_filter(xgb_signal, llm_context)
            except Exception as e:
                logger.warning(f"Risk filter failed: {e}, proceeding without veto")

            # Step 4: Synthesize final decision
            synthesis_input = SynthesisInput(
                xgboost_signal=xgb_signal,
                llm_context=llm_context,
                critic_veto=critic_veto,
            )

            synthesis_output = synthesize(synthesis_input)

            logger.info(
                "Synthesis complete",
                symbol=symbol,
                final_direction=synthesis_output.direction,
                position_fraction=synthesis_output.position_size_fraction,
            )

            # Queue for accuracy verification
            legacy_signal = synthesis_to_legacy_signal(
                xgb_signal, synthesis_output, llm_context, critic_veto
            )

            # Get current price for verification (from XGBoost features or re-fetch)
            current_price = 0.0
            try:
                df = await market_data_service.fetch_ohlcv(symbol, timeframe, lookback_bars=1)
                if df is not None and len(df) > 0:
                    current_price = float(df["close"].iloc[-1])
                    legacy_signal.current_price = current_price
            except Exception:
                pass

            queue_for_verification(legacy_signal, current_price)

            # Determine execution
            executed = False
            trade_reason = None

            if execute and execution_client is not None:
                if synthesis_output.direction == "FLAT":
                    trade_reason = "Signal is FLAT - no trade"
                elif synthesis_output.position_size_fraction == 0:
                    trade_reason = "Position size is 0 - no trade"
                elif xgb_signal.confidence < min_confidence:
                    trade_reason = f"Confidence {xgb_signal.confidence:.1%} below threshold {min_confidence:.1%}"
                else:
                    # Build signal input for execution
                    signal_input = SignalInput(
                        symbol=symbol,
                        direction="long" if synthesis_output.direction == "LONG" else "short",
                        confidence=xgb_signal.confidence,
                        expected_return_pct=0.5,  # Conservative estimate
                        stop_loss_pct=2.0,  # Default stop loss
                        take_profit_pct=4.0,  # 2:1 R:R
                        timeframe=timeframe,
                        entry_price=current_price,
                    )

                    try:
                        decision = await execution_client.accept_signal(signal_input)
                        trade_reason = decision.reason

                        if decision.execute:
                            # Adjust amount by position_size_fraction
                            adjusted_amount = (
                                decision.amount * synthesis_output.position_size_fraction
                            )

                            # Place order
                            order = await execution_client.place_market_order(
                                symbol=decision.symbol,
                                side=decision.side,
                                amount=adjusted_amount,
                            )
                            executed = True
                            trade_reason = f"Order placed: {order.order_id}"
                            logger.info(
                                "Order executed",
                                symbol=symbol,
                                side=decision.side,
                                amount=adjusted_amount,
                                order_id=order.order_id,
                            )
                    except Exception as e:
                        trade_reason = f"Execution error: {str(e)}"
                        logger.error("Execution failed", error=str(e))
            else:
                trade_reason = "Execution disabled" if not execute else "No execution client"

            # Log signal (includes synthesis components in reasoning)
            log_signal(legacy_signal, executed=executed, trade_reason=trade_reason)

            # Print summary
            print_signal_summary(synthesis_output, xgb_signal, executed, trade_reason)

            signals.append(legacy_signal)

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
