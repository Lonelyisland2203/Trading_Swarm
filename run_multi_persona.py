#!/usr/bin/env python3
"""Run multi-persona signal generation workflow."""

import asyncio
from swarm.multi_persona import run_multi_persona_workflow

async def main():
    print("Starting multi-persona workflow...")
    print("This will:")
    print("  1. Generate signals from 3 personas (MOMENTUM, CONTRARIAN, SWING)")
    print("  2. Run critic evaluation on each signal")
    print("  3. Save training examples to outputs/")
    print()

    training_examples = await run_multi_persona_workflow(
        symbols=["BTC/USDT", "ETH/USDT"],
        timeframe="1h"
    )

    print("\n" + "=" * 60)
    print(f"WORKFLOW COMPLETE: {len(training_examples)} training examples generated")
    print("=" * 60)

    for ex in training_examples[:3]:  # Show first 3
        signal = ex.generator_signal
        print(f"\n{ex.symbol} [{ex.persona}]:")
        print(f"  Direction: {signal['direction']} ({signal['confidence']:.1%})")
        print(f"  Reasoning: {signal['reasoning'][:100]}...")

if __name__ == "__main__":
    asyncio.run(main())
