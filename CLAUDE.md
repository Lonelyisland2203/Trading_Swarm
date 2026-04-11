# Trading Swarm - Project Memory

## Project Overview

Autonomous AI trading signal system with self-improvement via DPO fine-tuning.

**Active Models:**
- Generator: qwen3:8b (non-thinking mode, 4-bit quantization)
- Critic: deepseek-r1:14b (native reasoning, 4-bit quantization)

**Hardware Constraints:**
- RTX 5070 Ti (16 GB VRAM) - models NEVER loaded simultaneously
- OLLAMA_KEEP_ALIVE=0 enforced via validator - critical for VRAM management

## Current State

**Completed:**
- Sessions 1-10: Environment, Data, Swarm, Verifier, Reward, Evaluation, DPO Infrastructure, Dataset Generation, End-to-End DPO Workflow
- Session 11a: Realistic Fee Model
- Session 11b: Technical Indicator Expansion (17 indicators, 5 groups, compute_all_indicators aggregation)
- Session 12: Multi-Timeframe Context (4-indicator voting, confluence detection, optional higher_tf_data parameter)
- Session 13: Derivatives Data Integration (funding rates, open interest, adaptive TTL caching, perpetual mapping)
- Session 14: Fee-Aware DPO Training (execution context in prompts, net returns in rewards, fee model integration)
- Session 15: Binance Execution Client (execution module, safety controls, fee-aware sizing)
- Session 16: DPO Evaluation Pipeline (evaluate_candidate.py, test_eval_data saving, cache key fix, nested signal_data in dpo_eval.py)

**Next:** Session 17 - TBD (generate more training data to meet promotion thresholds; current candidate has IC=+6.4% but needs larger test set for p<0.05)

**Active Issues:**
- `test_critic.py::TestCritiquePrompt::test_prompt_has_adversarial_framing` - stale expectation after critic.py modification
- 4 tests in `test_process_lock.py` - Windows/fcntl platform incompatibility

## Architecture Decisions

### Fee-Aware Training (Session 14)
- **Teaching Pattern:** Execution context added to all three prompt templates (DirectionPredictionPrompt, MomentumAssessmentPrompt, SupportResistancePrompt)
- **Execution Context Fields:** Exchange type, trading mode (Futures USDT-M/Spot), round-trip cost %, minimum profitable move %
- **Enforcement:** RewardEngine switched from gross returns (realized_return) to net returns (net_return) for reward computation
- **Net Return Impact:** Signals with positive gross but negative net returns now receive negative rewards, teaching model to account for trading costs
- **Optional Integration:** PromptBuilder.build_prompt() accepts optional fee_model parameter - backward compatible when omitted
- **Fee Model Calculation:** Uses compute_holding_periods_8h() for timeframe-adaptive fee estimation

### Evaluation Pipeline (Session 16)
- **Fast Evaluation (recommended):** `python evaluate_candidate.py --test-data outputs/dpo_run_TIMESTAMP/test_eval_data.jsonl --candidate models/adapters/qwen3-8b-dpo/adapter-DPO-TIMESTAMP.candidate` — loads pre-saved test data, completes in seconds
- **Slow Fallback:** `--dataset outputs/combined_examples.jsonl` — re-runs full verification (~1-2h, re-fetches all market data)
- **Promotion criteria (no baseline):** IC >= 0.05, Brier <= 0.25, p < 0.05
- **test_eval_data.jsonl:** Saved automatically by `run_dpo_training.py` after Phase 4 (before training). Contains flattened (example, outcome, reward) for the test split only.
- **evaluation.json:** Saved inside adapter directory after evaluation runs.
- **Cache key stability:** `fetch_ohlcv()` without explicit `end_ts` now snaps to the nearest bar close boundary (`end_ts = (now_ms // bar_duration_ms) * bar_duration_ms`), so all verification calls within the same bar period share a cache entry.
- **Candidate adapter:** `models/adapters/qwen3-8b-dpo/adapter-DPO-1775725178964.candidate` — IC=+6.4% (not significant, N=198 too small), Brier=0.30 (overconfident). Retained for future comparison.

### Execution Layer (Session 15)
- **Purpose:** Production execution layer for deploying signals to Binance
- **Safety-First Design:** Testnet by default, live trading requires ALLOW_LIVE_TRADING=true
- **Kill Switch:** STOP file in execution/state/ halts all trading immediately
- **Circuit Breakers:**
  - Daily loss limit (default 2%) - auto-activates kill switch at 1.5x
  - Daily trade count limit (default 10)
  - Max open positions (default 3)
  - Order cooldown (default 60s between orders)
  - Max position size (default 2% of portfolio)
- **Fee-Aware Sizing:** Position sizing accounts for fees, respects risk limits
- **Signal Flow:** accept_signal() evaluates and returns decision; does NOT auto-execute
- **State Persistence:** daily_stats.json, order_log.jsonl in execution/state/
- **Integration:** Uses existing FeeModelSettings from config/fee_model.py

### Derivatives Data (Session 13)
- **Adaptive TTL:** Historical (>7d) = 24h, recent (1h-7d) = 2h, live (<1h) = 30min
- **Perpetual Mapping:** Disk-cached market metadata (24h TTL), graceful None on missing perpetuals
- **Capability Checking:** Exchange-level `has['fetchFundingRate']` and `has['fetchOpenInterest']` checks
- **API:** `get_market_context()` orchestrates funding rate + open interest fetching with fallbacks
- **Methods:** `_compute_adaptive_ttl()`, `_load_perpetual_markets()`, `_get_perpetual_symbol()`, `load_markets()`, `fetch_funding_rate()`, `fetch_open_interest()`, `get_market_context()`

### Multi-Timeframe Context (Session 12)
- **Hierarchy:** TIMEFRAME_HIERARCHY = ["1m", "5m", "15m", "1h", "4h", "1d"] - adaptive selection of up to 2 nearest higher TFs
- **4-Indicator Voting:** Ichimoku cloud position, KAMA slope, Donchian channel position, RSI zone -> trend classification (bullish/bearish/neutral)
- **Confluence Detection:** Multi-TF alignment categorized as aligned, mixed, or conflicting
- **Integration:** Optional `higher_tf_data` parameter throughout stack (backward compatible)

### Indicator System (Session 11b)
- **17 indicators across 5 groups:** Momentum (Donchian, Ichimoku, KAMA), Volume (OBV, CMF, MFI, VWAP), Volatility (ATR, BB bandwidth, Keltner, historical vol), Structure (TTM Squeeze, FVG, Swing Points), Crypto (funding_rate, open_interest stubs)
- **Pattern:** All indicators use `compute_` prefix with pd.Series parameters
- **Aggregation:** `compute_all_indicators()` returns dict of all indicator values for PromptBuilder integration

### Configuration Layer
- Nested Pydantic settings with flat env var mapping
- Reward weights validated to sum to 1.0 via `@model_validator`
- DatasetGenerationSettings: window_count=15, stride=100, completeness=0.95, retries=2

### Fee Model Implementation (Session 11a)
- **Purpose:** Binance Futures USDT-M realistic fee model to filter unprofitable signals
- **FeeModelSettings:** Pydantic config with env var support (FEE_MODEL_*)
- **Defaults:** maker 0.02%, taker 0.05%, 10% BNB discount, 0.01% funding/8h, 0.02% slippage
- **Core API:** `round_trip_cost_pct()`, `net_return()`, `minimum_profitable_return_pct()`

### Data Layer
- Async caching: diskcache wrapped in `asyncio.to_thread()`
- Point-in-time safety: `get_ohlcv_as_of()` filters by bar close time; passes `end_ts=as_of` to `fetch_ohlcv()`
- Regime classification: Realized volatility percentiles (no VIX for crypto)
- Task sampling: Weighted by difficulty with isolated RNG
- Historical windows: Configurable stride, >95% completeness threshold
- Multi-timeframe context: Optional higher_tf_data parameter in PromptBuilder.build_prompt()
- Fee-aware prompts: Optional fee_model parameter in PromptBuilder.build_prompt() adds execution context section

### Swarm Layer
- VRAM Management: Semaphore + explicit unload between model switches
- Caching Strategy: Temperature gate - only cache temp=0 generations
- Persona Selection: Regime-informed weighted sampling (5 personas)
- Response Validation: 4-stage JSON extraction with single clarification retry
- Orchestrator: Accepts optional higher_tf_data via _get_task_config_by_type() helper

### Training Layer - DPO
- **Stack:** Direct transformers + PEFT (not Unsloth - better debuggability)
- **Model Source:** HuggingFace download (Ollama cache incompatible with transformers)
- **Process Isolation:** fcntl file locks (exclusive training, shared inference)
- **VRAM Budget:** 9-11 GB for training, 9 GB conservative minimum
- **LoRA Config:** r=32, alpha=64, 7 target modules
- **DPO Hyperparams:** beta=0.1, lr=5e-6, batch_size=1, grad_accum=16
- **Walk-Forward:** 500 train / 100 test pairs per window, temporal ordering
- **Replay Buffer:** 15% old data to prevent catastrophic forgetting
- **Promotion:** IC > 0.02, Brier > 0.01, p < 0.05, N >= 100, 24h cooldown, 3 max/week

### DPO Pipeline (Session 10)
- **5-Phase CLI:** Load -> Verify -> Reward -> Pairs -> Train
- **CLI flags:** `--dataset` (required), `--output`, `--save-pairs`, `--dry-run`, `--min-delta`, `--force`

### Dataset Generation (Session 9)
- **Scale:** 13,500 examples (10 symbols x 6 timeframes x 15 windows x 3 tasks x 5 personas)
- **3-Phase Parallelization:** Phase 1 parallel data prep, Phase 2 sequential VRAM inference, Phase 3 parallel post-processing

### Verifier Layer
- Timeframe-adaptive horizons (1m->60 bars, 1h->24 bars, 1d->5 bars)
- Log returns for additivity and DPO compatibility
- Entry at next bar open (realistic execution)

### Reward Layer
- Clipped linear reward bounded to [-1, 1] for stable DPO gradients
- Three components: return (0.50), directional (0.30), MAE (0.20)
- Net return enforcement: RewardEngine uses net_return (after fees) instead of realized_return (gross)

### Evaluation Layer
- Spearman IC primary, BH-FDR correction for multiple hypothesis testing
- Metrics: IC, Sharpe, Sortino, Calmar, max drawdown, win rate, profit factor

## File Index

### Configuration
- `config/settings.py` - Pydantic settings + DPOTrainingSettings + DatasetGenerationSettings
- `config/fee_model.py` - FeeModelSettings for Binance Futures USDT-M fee calculations

### Execution Layer
- `execution/__init__.py` - Package exports
- `execution/exceptions.py` - ExecutionError hierarchy
- `execution/models.py` - OrderResult, Position, TradeDecision, DailyStats, SignalInput
- `execution/state_manager.py` - Kill switch, daily stats, order logging
- `execution/position_sizing.py` - Fee-aware position sizing
- `execution/binance_client.py` - BinanceExecutionClient main class

### Data Layer
- `data/indicators.py` - 17 indicators + compute_all_indicators() aggregation
- `data/cache_wrapper.py` - AsyncDiskCache with asyncio.to_thread()
- `data/market_data.py` - CCXT client + adaptive TTL + perpetual mapping + derivatives data (funding rates, open interest)
- `data/regime_filter.py` - RegimeClassifier with volatility percentiles
- `data/prompt_builder.py` - Task sampling, compute_all_indicators integration, multi-TF context
- `data/historical_windows.py` - Window walking with completeness validation
- `data/inference_queue.py` - Sequential job processor with JSONL streaming

### Utils
- `utils/progress_tracker.py` - Progress tracking with rolling ETA, JSON state persistence

### Swarm Layer
- `swarm/exceptions.py` - Custom exception hierarchy
- `swarm/ollama_client.py` - VRAM-aware Ollama client with semaphore
- `swarm/generator.py` - Signal generator with 5 personas
- `swarm/critic.py` - Critique generation with deepseek-r1:14b
- `swarm/orchestrator.py` - LangGraph workflow + multi-persona + higher_tf_data support
- `swarm/training_capture.py` - TrainingExample with context_id + `load_examples_from_jsonl()`
- `swarm/adapter_loader.py` - Adapter discovery, validation, Ollama tags

### Training Layer
- `training/reward_config.py` - RewardScaling dataclass
- `training/reward_components.py` - Individual reward functions
- `training/reward_engine.py` - Main reward API with market_regime
- `training/dpo_export.py` - Preference pair construction + export
- `training/process_lock.py` - fcntl file locks
- `training/vram_check.py` - GPU VRAM detection (NVIDIA + Apple Silicon)
- `training/walk_forward.py` - Temporal splits with replay buffer
- `training/dpo_eval.py` - IC, Brier, MACE, promotion logic
- `training/dpo_trainer.py` - DPO training pipeline (transformers + PEFT)

### Scripts
- `generate_training_dataset.py` - Main CLI for dataset generation
- `run_dpo_training.py` - End-to-end DPO pipeline CLI
- `run_multi_persona.py` - Lightweight multi-persona test script (deprecated; use generate_training_dataset.py)
- `evaluate_candidate.py` - Candidate adapter evaluation and promotion (fast path via --test-data, slow fallback via --dataset)

### Verifier Layer
- `verifier/constants.py` - Timeframe constants + `compute_holding_periods_8h()`
- `verifier/outcome.py` - OutcomeData + `apply_fee_model()`
- `verifier/config.py`, `verifier/validator.py`, `verifier/engine.py` - Core verification logic

### Evaluation Layer
- `eval/` - config, metrics, engine

### Tests
- `tests/fixtures/timeframe_fixtures.py` - OHLCV patterns for multi-TF testing
- `tests/test_config.py` - 40 tests
- `tests/test_indicators.py` - 19 tests (original indicators)
- `tests/test_indicators_extended.py` - 63 tests (extended indicators)
- `tests/test_data_layer.py` - 23 tests
- `tests/test_data_layer/test_inference_queue.py` - 16 tests
- `tests/test_prompt_builder_mtf.py` - 37 tests (multi-timeframe context)
- `tests/test_prompt_builder_fee_context.py` - 7 tests (execution context in prompts)
- `tests/test_market_data_derivatives.py` - 23 tests (adaptive TTL, perpetual mapping, derivatives fetching)
- `tests/test_ollama_client.py` - 17 tests
- `tests/test_generator.py` - 20 tests
- `tests/test_critic.py` - 22 tests
- `tests/test_orchestrator.py` - 23 tests (5 pre-existing failures)
- `tests/test_verifier/` - 77 tests
- `tests/test_reward/` - 63 tests
- `tests/test_reward_net_returns.py` - 5 tests (net return enforcement in reward engine)
- `tests/test_eval/` - 49 tests
- `tests/test_training/` - 71 tests + `test_dpo_pipeline.py` (23 tests)
- `tests/test_swarm/test_adapter_loader.py` - 16 tests
- `tests/test_integration/test_fee_model_integration.py` - 6 tests
- `tests/test_fee_aware_integration.py` - 11 tests (end-to-end fee-aware workflow)
- `tests/test_execution/` - 166 tests (exceptions, models, settings, state, sizing, client, safety controls)

## Known Issues & Gotchas

### Active Test Failures
- `test_critic.py::test_prompt_has_adversarial_framing` - stale expectation after critic.py change
- 4 tests in `test_process_lock.py` - fcntl unavailable on Windows

### Runtime Dependencies
- Ollama service: Required only at runtime
- Model downloads: ~14 GB total (qwen3:8b + deepseek-r1:14b)

### Code Patterns
- All diskcache operations wrapped in `asyncio.to_thread()`
- Generator prompts must include `/no_think`
- Context managers for AsyncDiskCache, MarketDataService, OllamaClient
- Custom EnumJSONEncoder for JSON serialization of task types and personas
- **Indicator pattern:** `compute_` prefix, pd.Series params, return pd.Series or dict of Series
- **Multi-TF pattern:** Optional `higher_tf_data` parameter cascaded through orchestrator -> PromptBuilder
- **InferenceQueue init:** Deletes existing output file when `resume=False`; preserves it when `resume=True`
- **GeneratorSignal structure:** Task-specific fields (direction, confidence, etc.) live inside nested `signal_data` dict, NOT at top level
- **Windows console:** Avoid Unicode emoji in print() — cp1252 codec crashes on characters like `⚠️`; use ASCII alternatives
- **fetch_ohlcv cache key:** When `end_ts` is None (default in `verify_example`), snaps to `(now_ms // bar_duration_ms) * bar_duration_ms` so calls within the same bar period share a cache entry. Callers passing explicit `end_ts` (e.g. `get_ohlcv_as_of`) are unaffected.
- **Evaluation data pattern:** `run_dpo_training.py` saves `test_eval_data.jsonl` in the run output dir after Phase 4. Pass to `evaluate_candidate.py --test-data` for fast evaluation without re-verification. Without this file, evaluation re-fetches all market data (~1-2h).

## Working Decisions

- Separate `requirements.txt` (inference) and `requirements-training.txt` (DPO)
- Process A and Process B NEVER run concurrently (enforced by process_lock.py)
- Models accessed by exact tag (e.g., `qwen3:8b`)
- Adapter directory: `models/adapters/adapter-{PERSONA}-{TIMESTAMP}.promoted`
- Batched execution: Generate by timeframe to keep runs <16 hours
- Deferred: M3 (fetch optimization), M4 (type hints), Session 9 tests

---

**Total Tests:** 906 passing (5 pre-existing orchestrator failures excluded)
**Python Version:** 3.13.7
