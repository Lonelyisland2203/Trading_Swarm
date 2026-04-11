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
Completed through Session 17D.
- training/sft_data_generator.py — reverse reasoning distillation from deepseek-r1:14b, outputs data/sft_training_data.jsonl
- training/sft_trainer.py — fine-tunes qwen3:8b on SFT data, LoRA r=32/alpha=64, saves to adapters/sft_base/
- training/grpo_config.py — all GRPO hyperparameters (G=4, β=0.04, ε=0.2, reward weights, asymmetry coefficients)
- training/grpo_reward.py — asymmetric reward matrix (false bullish -1.5×, false bearish -0.8×), structure reward (regex THESIS→EVIDENCE→RISK→DECISION), combined reward with clipping
- training/grpo_data.py — GRPOTrainingExample dataclass, GRPOWalkForwardSplit, temporal split functions
- training/grpo_trainer.py — full GRPO training loop: sequential G=4 generation, KL penalty (β=0.04), PPO clipping (ε=0.2), checkpointing every 500 steps, STOP file handling, CLI entry point
- Tests: 130 GRPO tests (config: 18, data: 9, reward: 47, trainer: 56)

## Next Session
Session 17E — GRPO training data generation pipeline (create grpo_training_data.jsonl from historical market data)