#!/usr/bin/env python3
"""Generate a single trading signal to test the system."""

import asyncio
from swarm.generator import generate_signal, TradingPersona
from swarm.ollama_client import OllamaClient
from data.market_data import MarketDataService
from data.regime_filter import RegimeClassifier
from data.prompt_builder import PromptBuilder, TaskType, TASK_CONFIGS
from config.settings import settings


async def main():
    # 1. Fetch market data
    print("Fetching BTC/USDT 1h data...")
    async with MarketDataService() as market:
        df = await market.fetch_ohlcv("BTC/USDT", "1h", lookback_bars=100)
        print(f"[OK] Fetched {len(df)} bars")

        # 2. Determine regime
        regime_filter = RegimeClassifier()
        regime, _ = regime_filter.get_current_regime(df["close"])
        print(f"[OK] Market regime: {regime}")

        # 3. Build prompt
        prompt_builder = PromptBuilder()
        persona = TradingPersona.MOMENTUM
        task = next(t for t in TASK_CONFIGS if t.task_type == TaskType.PREDICT_DIRECTION)
        prompt = prompt_builder.build_prompt(
            task=task,
            df=df,
            symbol="BTC/USDT",
            timeframe="1h",
            market_regime=regime,
        )
        print(f"[OK] Prompt built ({len(prompt)} chars)")

        # 4. Generate signal
        print("Generating signal (requires Ollama)...")
        async with OllamaClient() as ollama:
            signal = await generate_signal(
                client=ollama,
                model=settings.ollama.generator_model,
                prompt=prompt,
                regime=regime,
                task_type=task.task_type,
                persona_override=persona,
            )

    print("\n" + "=" * 60)
    print("SIGNAL GENERATED:")
    print("=" * 60)
    if signal:
        print(f"Direction:  {signal.signal_data.get('direction', 'N/A')}")
        conf = signal.signal_data.get("confidence", 0)
        print(f"Confidence: {conf:.2%}")
        reasoning = signal.signal_data.get("reasoning") or signal.signal_data.get("rationale", "")
        print(f"Reasoning:  {reasoning[:300]}")
        print(f"Persona:    {signal.persona}")
        print(f"Task:       {signal.task_type}")
        print(f"\nFull signal_data keys: {list(signal.signal_data.keys())}")
    else:
        print("No signal generated.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
