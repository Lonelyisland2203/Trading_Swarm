#!/usr/bin/env python3
"""Generate a single trading signal to test the system."""

import asyncio
from swarm.generator import Generator
from data.market_data import MarketDataClient
from data.regime_filter import RegimeFilter
from data.prompt_builder import PromptBuilder

async def main():
    print("Fetching BTC/USDT 1h data...")

    # 1. Fetch market data
    client = MarketDataClient()
    await client.initialize()
    df = await client.fetch_ohlcv("BTC/USDT", "1h", lookback_bars=100)
    print(f"✓ Fetched {len(df)} bars")

    # 2. Determine regime
    regime_filter = RegimeFilter()
    regime = regime_filter.determine_regime(df)
    print(f"✓ Market regime: {regime}")

    # 3. Build prompt
    prompt_builder = PromptBuilder()
    prompt = prompt_builder.build_generator_prompt(
        symbol="BTC/USDT",
        timeframe="1h",
        market_data=df,
        regime=regime,
        persona="MOMENTUM"
    )
    print(f"✓ Prompt built ({len(prompt)} chars)")

    # 4. Generate signal
    generator = Generator()
    signal = await generator.generate(
        symbol="BTC/USDT",
        timeframe="1h",
        market_data=df,
        regime=regime,
        persona="MOMENTUM",
        context_id="test-001"
    )

    print("\n" + "=" * 60)
    print(f"SIGNAL GENERATED:")
    print("=" * 60)
    print(f"Direction:  {signal['direction']}")
    print(f"Confidence: {signal['confidence']:.2%}")
    print(f"Reasoning:  {signal['reasoning'][:200]}...")
    print(f"Persona:    {signal['persona']}")
    print("=" * 60)

    await client.close()

if __name__ == "__main__":
    asyncio.run(main())
