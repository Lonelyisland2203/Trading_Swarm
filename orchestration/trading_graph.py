"""
LangGraph-based trading pipeline orchestration.

Session 17S: Wires all nodes into a StateGraph:
- data_node: fetch OHLCV via get_ohlcv_as_of, compute 17 indicators
- xgboost_node: generate XGBoost signal
- context_node: generate LLM market context (Qwen)
- synthesis_node: synthesize final decision
- execution_node: route to exchange via exchange_router

Error handling: try/except around each node. On error → set state.errors, default to FLAT.
STOP file check at graph entry point.
Log complete TradingState to signals/signal_log.jsonl after each run.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from data.indicators import compute_all_indicators
from data.market_data import MarketDataService
from signals.llm_context import LLMContext, generate_market_context
from signals.preflight import check_stop_file
from signals.signal_loop import call_risk_filter
from signals.synthesis import SynthesisInput, SynthesisOutput, synthesize
from signals.xgboost_signal import (
    XGBoostSignal,
    generate_xgboost_signal,
    DEFAULT_LOOKBACK_BARS,
)

if TYPE_CHECKING:
    from execution.exchange_router import ExchangeRouter


# Path for signal logging
SIGNAL_LOG_PATH = Path("signals/signal_log.jsonl")


@dataclass
class TradingState:
    """
    Complete state for a single trading graph invocation.

    Contains all intermediate results for logging and debugging.

    Attributes:
        symbol: Trading pair (e.g., "BTC/USDT")
        timeframe: Timeframe string (e.g., "1h", "4h")
        dry_run: If True, skip execution
        ohlcv_data: Raw OHLCV DataFrame (not serialized)
        indicators: Dict of computed indicator values
        xgboost_signal: XGBoost model output
        llm_context: LLM market context (may be None if failed)
        critic_veto: True if DeepSeek vetoed the trade
        synthesis_output: Final trading decision
        execution_result: Result from exchange (None if dry_run)
        errors: List of error messages from failed nodes
        timestamp: When the state was created
    """

    symbol: str
    timeframe: str
    dry_run: bool = True
    ohlcv_data: Any = None  # pd.DataFrame, not serialized
    indicators: dict[str, Any] | None = None
    xgboost_signal: XGBoostSignal | None = None
    llm_context: LLMContext | None = None
    critic_veto: bool = False
    synthesis_output: SynthesisOutput | None = None
    execution_result: dict[str, Any] | None = None
    errors: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dictionary."""
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "dry_run": self.dry_run,
            "indicators": self.indicators,
            "xgboost_signal": self.xgboost_signal.to_dict() if self.xgboost_signal else None,
            "llm_context": self.llm_context.to_dict() if self.llm_context else None,
            "critic_veto": self.critic_veto,
            "synthesis_output": {
                "direction": self.synthesis_output.direction,
                "position_size_fraction": self.synthesis_output.position_size_fraction,
                "rationale": self.synthesis_output.rationale,
            }
            if self.synthesis_output
            else None,
            "execution_result": self.execution_result,
            "errors": self.errors,
            "timestamp": self.timestamp.isoformat(),
        }


def _create_flat_synthesis_output(rationale: str) -> SynthesisOutput:
    """Create a FLAT synthesis output for error/stop cases."""
    return SynthesisOutput(
        direction="FLAT",
        position_size_fraction=0.0,
        rationale=rationale,
        components={},
    )


class TradingGraph:
    """
    LangGraph-based trading pipeline.

    Orchestrates the flow: data → XGBoost → LLM context → synthesis → execution.
    Each node handles its own errors, recording them in state.errors.

    Usage:
        graph = TradingGraph()
        result = await graph.run(symbol="BTC/USDT", timeframe="1h", dry_run=True)
    """

    def __init__(self, exchange_router: "ExchangeRouter | None" = None):
        """
        Initialize the trading graph.

        Args:
            exchange_router: Optional exchange router for execution. If None,
                             execution is disabled even when dry_run=False.
        """
        self._exchange_router = exchange_router

    async def _data_node(self, state: TradingState) -> TradingState:
        """
        Fetch OHLCV data and compute indicators.

        Uses get_ohlcv_as_of() for point-in-time safety.
        """
        try:
            as_of = int(datetime.now(timezone.utc).timestamp() * 1000)

            async with MarketDataService() as market_data_service:
                df = await market_data_service.get_ohlcv_as_of(
                    symbol=state.symbol,
                    timeframe=state.timeframe,
                    as_of=as_of,
                    lookback_bars=DEFAULT_LOOKBACK_BARS,
                )

            state.ohlcv_data = df

            # Compute all 17 indicators
            indicators = compute_all_indicators(df, include_volume=True, include_structure=True)

            # Extract scalar values (not Series)
            state.indicators = {
                k: v for k, v in indicators.items() if k != "series" and not k.startswith("raw_")
            }

            logger.info(
                "Data node complete",
                symbol=state.symbol,
                bars=len(df),
                rsi=state.indicators.get("rsi"),
            )

        except Exception as e:
            logger.error(f"Data node failed: {e}")
            state.errors.append(f"Data fetch failed: {str(e)}")

        return state

    async def _xgboost_node(self, state: TradingState) -> TradingState:
        """
        Generate XGBoost signal.

        XGBoost runs on CPU (no VRAM conflict).
        """
        if state.ohlcv_data is None or state.errors:
            # Previous node failed, skip
            logger.warning("Skipping XGBoost node - no data available")
            return state

        try:
            as_of = int(datetime.now(timezone.utc).timestamp() * 1000)

            async with MarketDataService() as market_data_service:
                signal = await generate_xgboost_signal(
                    symbol=state.symbol,
                    timeframe=state.timeframe,
                    as_of=as_of,
                    market_data_service=market_data_service,
                    skip_preflight=True,  # Already checked at graph entry
                )

            state.xgboost_signal = signal

            if signal:
                logger.info(
                    "XGBoost signal generated",
                    symbol=state.symbol,
                    direction=signal.direction,
                    probability=signal.probability,
                )
            else:
                state.errors.append("XGBoost signal generation returned None")
                logger.warning("XGBoost signal is None")

        except Exception as e:
            logger.error(f"XGBoost node failed: {e}")
            state.errors.append(f"XGBoost failed: {str(e)}")

        return state

    async def _context_node(self, state: TradingState) -> TradingState:
        """
        Generate LLM market context using Qwen.

        VRAM: Qwen loads → inference → unloads (OLLAMA_KEEP_ALIVE=0).
        """
        if state.xgboost_signal is None:
            # Previous node failed, skip
            logger.warning("Skipping context node - no XGBoost signal")
            return state

        try:
            # In production, we'd fetch funding rate, OI, etc. from market data
            # For now, using None to get neutral fallback context
            context = await generate_market_context(
                funding_rate=None,
                oi_delta=None,
                liquidation_data=None,
                news_headlines=None,
            )

            state.llm_context = context

            logger.info(
                "LLM context generated",
                regime_flag=context.regime_flag,
                confidence=context.confidence,
            )

        except Exception as e:
            logger.warning(f"Context node failed: {e}, proceeding without LLM context")
            state.llm_context = None  # Synthesis handles None gracefully

        return state

    async def _risk_filter_node(self, state: TradingState) -> TradingState:
        """
        Call DeepSeek risk filter.

        VRAM: DeepSeek loads → inference → unloads (OLLAMA_KEEP_ALIVE=0).
        On failure: treat as no-veto (conservative).
        """
        if state.xgboost_signal is None:
            logger.warning("Skipping risk filter - no XGBoost signal")
            return state

        try:
            veto = await call_risk_filter(state.xgboost_signal, state.llm_context)
            state.critic_veto = veto

            if veto:
                logger.info("Risk filter VETO", symbol=state.symbol)
            else:
                logger.debug("Risk filter APPROVE", symbol=state.symbol)

        except Exception as e:
            logger.warning(f"Risk filter failed: {e}, defaulting to no-veto")
            state.critic_veto = False  # Conservative: don't veto on failure

        return state

    async def _synthesis_node(self, state: TradingState) -> TradingState:
        """
        Synthesize final trading decision.

        Rules (from signal-layer.md):
        - XGBoost prob < 0.55 → FLAT
        - XGBoost prob ≥ 0.55 + conflicting regime → half position
        - XGBoost prob ≥ 0.65 + confirming → full position
        - DeepSeek veto → FLAT
        - Missing LLM context → 0.7x position
        """
        if state.xgboost_signal is None:
            # No signal to synthesize, return FLAT
            state.synthesis_output = _create_flat_synthesis_output("No XGBoost signal available")
            return state

        try:
            synthesis_input = SynthesisInput(
                xgboost_signal=state.xgboost_signal,
                llm_context=state.llm_context,
                critic_veto=state.critic_veto,
            )

            output = synthesize(synthesis_input)
            state.synthesis_output = output

            logger.info(
                "Synthesis complete",
                direction=output.direction,
                position_fraction=output.position_size_fraction,
                rationale=output.rationale[:80],
            )

        except Exception as e:
            logger.error(f"Synthesis node failed: {e}")
            state.errors.append(f"Synthesis failed: {str(e)}")
            state.synthesis_output = _create_flat_synthesis_output(f"Synthesis error: {str(e)}")

        return state

    async def _execution_node(self, state: TradingState) -> TradingState:
        """
        Execute trade via exchange router.

        Skipped if:
        - dry_run is True
        - synthesis_output is FLAT
        - no exchange_router configured
        """
        if state.dry_run:
            logger.info("Dry run mode - skipping execution")
            return state

        if state.synthesis_output is None or state.synthesis_output.direction == "FLAT":
            logger.info("No trade to execute (FLAT)")
            return state

        if self._exchange_router is None:
            logger.warning("No exchange router configured, skipping execution")
            return state

        try:
            # Determine order parameters
            direction = state.synthesis_output.direction
            side = "buy" if direction == "LONG" else "sell"

            # Position sizing (simplified - in production use proper risk management)
            # TODO: Integrate with proper position sizing from risk manager
            base_amount = 0.001  # Minimum BTC amount
            amount = base_amount * state.synthesis_output.position_size_fraction

            # Place market order
            result = await self._exchange_router.place_order(
                symbol=state.symbol,
                side=side,
                amount=amount,
                order_type="market",
            )

            state.execution_result = {
                "order_id": result.order_id if result else None,
                "side": side,
                "amount": amount,
                "symbol": state.symbol,
            }

            logger.info(
                "Order executed",
                order_id=state.execution_result["order_id"],
                side=side,
                amount=amount,
            )

        except Exception as e:
            logger.error(f"Execution node failed: {e}")
            state.errors.append(f"Execution failed: {str(e)}")

        return state

    def _log_state(self, state: TradingState) -> None:
        """Log complete state to signals/signal_log.jsonl."""
        try:
            SIGNAL_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

            with open(SIGNAL_LOG_PATH, "a") as f:
                json.dump(state.to_dict(), f)
                f.write("\n")

            logger.debug("State logged to signal_log.jsonl")

        except Exception as e:
            logger.error(f"Failed to log state: {e}")

    async def run(
        self,
        symbol: str,
        timeframe: str,
        dry_run: bool = True,
    ) -> TradingState:
        """
        Run the complete trading graph.

        Flow: STOP check → data → XGBoost → LLM context → risk filter → synthesis → execution → log

        Args:
            symbol: Trading pair (e.g., "BTC/USDT")
            timeframe: Timeframe string (e.g., "1h", "4h")
            dry_run: If True, skip execution

        Returns:
            TradingState with all intermediate results
        """
        state = TradingState(symbol=symbol, timeframe=timeframe, dry_run=dry_run)

        # Check STOP file first
        if check_stop_file():
            logger.warning("STOP file detected, halting graph")
            state.errors.append("STOP file detected - trading halted")
            state.synthesis_output = _create_flat_synthesis_output(
                "STOP file detected - trading halted"
            )
            self._log_state(state)
            return state

        # Run nodes sequentially
        state = await self._data_node(state)
        state = await self._xgboost_node(state)
        state = await self._context_node(state)
        state = await self._risk_filter_node(state)
        state = await self._synthesis_node(state)
        state = await self._execution_node(state)

        # Log complete state
        self._log_state(state)

        return state


async def run_trading_graph(
    symbol: str,
    timeframe: str,
    dry_run: bool = True,
    exchange_router: "ExchangeRouter | None" = None,
) -> TradingState:
    """
    Convenience function to run the trading graph.

    Args:
        symbol: Trading pair
        timeframe: Timeframe string
        dry_run: If True, skip execution
        exchange_router: Optional exchange router

    Returns:
        TradingState with results
    """
    graph = TradingGraph(exchange_router=exchange_router)
    return await graph.run(symbol=symbol, timeframe=timeframe, dry_run=dry_run)
