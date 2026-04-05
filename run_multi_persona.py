#!/usr/bin/env python3
"""Run multi-persona signal generation workflow."""

import asyncio
import sys

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from data.market_data import MarketDataService
from data.regime_filter import RegimeClassifier
from data.prompt_builder import PromptBuilder, sample_task
from swarm.orchestrator import run_multi_persona_workflow

SYMBOLS = ["BTC/USDT", "ETH/USDT"]
TIMEFRAME = "1h"
LOOKBACK_BARS = 100

async def process_symbol(
    symbol: str,
    service: MarketDataService,
    regime_classifier: RegimeClassifier,
    prompt_builder: PromptBuilder,
) -> list:
    """Process a single symbol end-to-end.
    
    Args:
        symbol: Trading pair e.g. BTC/USDT
        service: Shared market data service
        regime_classifier: Regime classifier instance
        prompt_builder: Prompt builder instance
    Returns:
        List of training examples for this symbol
    """
    print(f"Processing {symbol}...")
    df = await service.fetch_ohlcv(symbol, TIMEFRAME, LOOKBACK_BARS)
    print(f"  [{symbol}] Fetched {len(df)} bars")

    regime_series = regime_classifier.classify_regime(df["close"])
    market_regime = regime_series.iloc[-1]
    print(f"  [{symbol}] Regime: {market_regime.value}")

    task = sample_task(available_bars=len(df))
    task_prompt = prompt_builder.build_prompt(
        task=task,
        df=df,
        symbol=symbol,
        timeframe=TIMEFRAME,
        market_regime=market_regime,
    )

    summary, examples = await run_multi_persona_workflow(
        symbol=symbol,
        timeframe=TIMEFRAME,
        ohlcv_df=df,
        market_regime=market_regime,
        task_prompt=task_prompt,
    )

    print(f"  [{symbol}] Status: {summary['workflow_status']}, "
          f"signals: {summary['signals_generated']}, "
          f"accepted: {summary['signals_accepted']}")

    return examples


async def main():
    print("Starting multi-persona workflow...")
    print("  1. Generate signals from all 5 personas per symbol (parallel)")
    print("  2. Run critic evaluation on each signal")
    print("  3. Save training examples to outputs/")
    print()

    regime_classifier = RegimeClassifier()
    prompt_builder = PromptBuilder()

    async with MarketDataService() as service:
        # ── KEY CHANGE: both symbols run simultaneously ──────────────
        results = await asyncio.gather(*[
            process_symbol(symbol, service, regime_classifier, prompt_builder)
            for symbol in SYMBOLS
        ])

    all_examples = [ex for symbol_examples in results for ex in symbol_examples]

    print("\n" + "=" * 60)
    print(f"WORKFLOW COMPLETE: {len(all_examples)} training examples generated")
    print("=" * 60)

    for ex in all_examples[:3]:
        signal = ex.generator_signal
        print(f"\n{ex.symbol} [{ex.persona}]:")
        print(f"  Direction: {signal['direction']} ({signal['confidence']:.1%})")
        print(f"  Reasoning: {signal['reasoning'][:100]}...")


if __name__ == "__main__":
    asyncio.run(main())