# CLAUDE.md

## Stack
- Python 3.13.7, pip
- Models: qwen3:8b (generator, 4-bit), deepseek-r1:14b (critic, 4-bit) via Ollama
- Training: transformers + PEFT (LoRA). NOT Unsloth.
- Data: CCXT free tier + Binance free API
- Execution: Binance Futures USDT-M via ccxt
- Hardware: RTX 5070 Ti, 16 GB VRAM

## Commands
- Install: `pip install -r requirements.txt`
- Test: `pytest` (530+ tests, run before every commit)
- Lint: `ruff check --fix . && ruff format .`
- Type check: `mypy --strict <module>`

## Hard Constraints — NEVER VIOLATE
- VRAM: Models NEVER loaded simultaneously. OLLAMA_KEEP_ALIVE=0.
- Process isolation: Process A and B never concurrent (process_lock.py, fcntl).
- Pre-flight order: Data → Temporal → VRAM → Lock → Load. Always.
- Temporal safety: get_ohlcv_as_of() filters by bar close; pass end_ts=as_of. No lookahead.
- Fee model: FeeModelSettings is ground truth. All rewards net of fees. Never modify fee_model.py without explicit instruction.
- Execution safety: Testnet default. Live requires ALLOW_LIVE_TRADING=true + typed confirmation. Kill switch via STOP file. Max 2% daily loss. 1x leverage only.
- Asymmetric penalties: False bullish costlier than missed opportunity. Never use symmetric clipping.
- Never cache responses where temperature > 0.
- Never load adapters older than 30 days.
- Never skip SFT stage before RL fine-tuning.

## Anti-Patterns
- No Unsloth (debugging opacity)
- No DPO when verifiable rewards exist (use GRPO)
- No linear approximation for log↔pct fee conversions

## Code Conventions
- Indicator functions: compute_ prefix, return pd.Series
- Reward components: clipped to [-1, 1]
- All new modules need tests in tests/ mirroring source structure
- Type hints on all public functions

@import .claude/context/training-layer.md
@import .claude/context/execution-layer.md
@import .claude/context/data-layer.md

## Current State
Completed through Session 17B.
- training/sft_data_generator.py — reverse reasoning distillation from deepseek-r1:14b, outputs data/sft_training_data.jsonl
- training/sft_trainer.py — fine-tunes qwen3:8b on SFT data, LoRA r=32/alpha=64, saves to adapters/sft_base/
- Tests: tests/training/test_sft_data_generator.py, tests/training/test_sft_trainer.py

## Next Session
Session 17C — GRPO Reward Engine (grpo_reward.py + grpo_config.py)