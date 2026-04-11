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
- Test: `pytest` (1400+ tests, run before every commit)
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
Completed through Session 17K.
- training/sft_data_generator.py — reverse reasoning distillation from deepseek-r1:14b, outputs data/sft_training_data.jsonl
- training/sft_trainer.py — fine-tunes qwen3:8b on SFT data, LoRA r=32/alpha=64, saves to adapters/sft_base/
- training/grpo_config.py — all GRPO hyperparameters (G=4, β=0.04, ε=0.2, reward weights, asymmetry coefficients)
- training/grpo_reward.py — asymmetric reward matrix (false bullish -1.5×, false bearish -0.8×), structure reward (regex THESIS→EVIDENCE→RISK→DECISION), combined reward with clipping
- training/grpo_data.py — GRPOTrainingExample dataclass, GRPOWalkForwardSplit, temporal split functions
- training/grpo_trainer.py — full GRPO training loop: sequential G=4 generation, KL penalty (β=0.04), PPO clipping (ε=0.2), checkpointing every 500 steps, STOP file handling, CLI entry point
- training/grpo_data_generator.py — generates grpo_training_data.jsonl from historical market data: fetches OHLCV with temporal safety (get_ohlcv_as_of), computes all 17 indicators, builds market snapshots, looks ahead by timeframe-adaptive horizon (1h→24 bars, 4h→12 bars), classifies direction using fee threshold (LONG/SHORT/FLAT), CLI with --symbols/--timeframes/--start-date/--end-date/--limit flags
- training/evaluate_candidate.py — unified adapter evaluation for DPO/GRPO, metrics (IC, Brier, MACE, regime-stratified IC, structure_compliance_rate), promotion criteria (IC≥0.05, Brier≤0.25, p<0.05, structure≥0.9 for GRPO), --compare mode for side-by-side evaluation
- run_grpo_training.py — end-to-end GRPO pipeline CLI: 5 phases (SFT data gen, SFT train, GRPO train, eval, promotion), skip logic for existing artifacts, --dry-run/--regenerate/--retrain-sft/--limit/--max-steps flags
- run_autoresearch.py — autonomous hyperparameter search loop (karpathy/autoresearch-inspired): round-robin parameter selection, direction tracking, git commit/revert on IC improvement/regression, results.tsv logging, --max-experiments/--time-budget-hours/--dry-run flags
- evaluation/xgboost_baseline.py — XGBoost/LightGBM baseline for LLM comparison: extracts same 17 indicators from market snapshots, walk-forward CV with temporal ordering, metrics (IC, Brier, Sharpe, directional accuracy), SHAP feature importance, comparison table vs GRPO/DPO adapters, CLI with --data/--compare/--n-folds flags
- signals/ — production signal loop package (101 tests):
  - signal_models.py — Signal dataclass, SignalDirection, map_generator_to_signal (HIGHER/LOWER→LONG/SHORT)
  - preflight.py — STOP file check, VRAM check (6GB min), process lock via check_can_infer()
  - signal_logger.py — thread-safe JSONL logging to signals/signal_log.jsonl
  - accuracy_tracker.py — deferred accuracy verification (queue signals, verify after next bar closes)
  - signal_loop.py — async loop: generate_signal_for_symbol(), evaluate_with_critic(), critic override logic (REJECT + reasoning_quality<0.5 OR technical_alignment<0.5 → FLAT)
  - verification.py — closes feedback loop: load unverified signals, fetch outcomes, compute fee-adjusted returns, aggregate stats (IC, Sharpe, regime-stratified accuracy, false bullish/bearish rates), training trigger (≥200 signals), export for GRPO retraining
- run_signal_loop.py — CLI: --symbols, --timeframe, --execute, --dry-run, --once, --min-confidence
- run_verification.py — CLI: --once, --interval, --stats, --export, --check-trigger; runs on schedule (default 4h) or once mode
- Tests: 426 tests (previous: 396, verification: 30)

## Next Session
Session 17L — End-to-end GRPO training dry run (generate data, train SFT, train GRPO, evaluate)