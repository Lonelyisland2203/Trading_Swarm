#!/usr/bin/env python3
"""Quick smoke test to verify Trading Swarm is ready to run."""

import sys

print("=" * 60)
print("Trading Swarm — Quick Smoke Test")
print("=" * 60)

# Test 1: Config loads
print("\n[1/5] Testing config loading...")
try:
    from config.settings import settings

    print("✓ Config loaded")
    print(f"  - Generator model: {settings.ollama.generator_model}")
    print(f"  - Critic model: {settings.ollama.critic_model}")
    print(f"  - Exchange: {settings.market_data.exchange}")
    print(f"  - Symbols: {settings.market_data.symbols}")
except Exception as e:
    print(f"✗ Config load failed: {e}")
    sys.exit(1)

# Test 2: Ollama connectivity
print("\n[2/5] Testing Ollama connection...")
try:
    import httpx

    response = httpx.get(f"{settings.ollama.base_url}/api/tags", timeout=5.0)
    if response.status_code == 200:
        models = response.json().get("models", [])
        model_names = [m["name"] for m in models]
        print(f"✓ Ollama is running ({len(models)} models)")

        # Check required models
        required = [settings.ollama.generator_model, settings.ollama.critic_model]
        for model in required:
            if model in model_names:
                print(f"  ✓ {model}")
            else:
                print(f"  ✗ {model} NOT FOUND — run: ollama pull {model}")
    else:
        print(f"✗ Ollama returned status {response.status_code}")
        sys.exit(1)
except Exception as e:
    print(f"✗ Cannot connect to Ollama: {e}")
    print("  → Make sure Ollama is running: ollama serve")
    sys.exit(1)

# Test 3: Market data client
print("\n[3/5] Testing market data client...")
try:
    from data.market_data import MarketDataClient

    client = MarketDataClient()
    print("✓ Market data client initialized")
    print(f"  - Exchange: {client.exchange.name}")
    print(f"  - Rate limit: {client.exchange.rateLimit}ms")
except Exception as e:
    print(f"✗ Market data client failed: {e}")
    sys.exit(1)

# Test 4: Generator
print("\n[4/5] Testing generator...")
try:
    from swarm.generator import Generator

    generator = Generator()
    print("✓ Generator initialized")
    print(f"  - Personas: {settings.swarm.generator_personas}")
except Exception as e:
    print(f"✗ Generator failed: {e}")
    sys.exit(1)

# Test 5: Critic
print("\n[5/5] Testing critic...")
try:
    from swarm.critic import Critic

    critic = Critic()
    print("✓ Critic initialized")
    print(f"  - Critique enabled: {settings.swarm.critique_enabled}")
except Exception as e:
    print(f"✗ Critic failed: {e}")
    sys.exit(1)

# Success!
print("\n" + "=" * 60)
print("✅ ALL CHECKS PASSED — Trading Swarm is ready!")
print("=" * 60)
print("\nNext steps:")
print("  1. Run a test signal generation:")
print(
    "     python -c \"from swarm.workflow import run_single_signal; import asyncio; asyncio.run(run_single_signal('BTC/USDT', '1h'))\""
)
print("\n  2. Run multi-persona workflow:")
print(
    '     python -c "from swarm.multi_persona import run_multi_persona_workflow; import asyncio; asyncio.run(run_multi_persona_workflow())"'
)
print()
