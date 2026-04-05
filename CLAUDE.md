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
- Session 11 (partial): Fee Model Tasks 1-3 complete (FeeModelSettings with round_trip_cost_pct, net_return, minimum_profitable_return_pct)

**Next:** Session 11 Tasks 4-10 - Integrate fee model into verifier and DPO pipeline

**Active Issues:**
- `test_critic.py::TestCritiquePrompt::test_prompt_has_adversarial_framing` - stale expectation after critic.py modification
- 4 tests in `test_process_lock.py` - Windows/fcntl platform incompatibility

## Architecture Decisions

### Configuration Layer
- Nested Pydantic settings with flat env var mapping
- Reward weights validated to sum to 1.0 via `@model_validator`
- DatasetGenerationSettings: window_count=15, stride=100, completeness=0.95, retries=2
- **FeeModelSettings:** Binance Futures USDT-M fee model with configurable order types
  - Base fees: maker 0.02%, taker 0.05%, 10% BNB discount applied
  - Funding: 0.01% per 8h period (configurable via `include_funding`)
  - Slippage: 0.02% round-trip
  - Methods: `round_trip_cost_pct()`, `net_return()`, `minimum_profitable_return_pct()`
  - Order types: `entry_order_type`, `exit_order_type` (maker/taker)
  - Validates holding_periods_8h >= 0

### Data Layer
- Async caching: diskcache wrapped in `asyncio.to_thread()`
- Point-in-time safety: `get_ohlcv_as_of()` filters by bar close time; passes `end_ts=as_of` to `fetch_ohlcv()` to anchor exchange queries historically (without this, all windows fail)
- Regime classification: Realized volatility percentiles (no VIX for crypto)
- Task sampling: Weighted by difficulty with isolated RNG
- Historical windows: Configurable stride, >95% completeness threshold
- Inference queue: JSONL streaming, context_id tracking for 5-persona completion

### Swarm Layer
- VRAM Management: Semaphore + explicit unload between model switches
- Caching Strategy: Temperature gate - only cache temp=0 generations
- Persona Selection: Regime-informed weighted sampling (5 personas)
- Response Validation: 4-stage JSON extraction with single clarification retry
- Multi-Persona Workflow: `run_multi_persona_workflow()` with `context_id` grouping

### Training Layer - DPO
- **Stack:** Direct transformers + PEFT (not Unsloth - better debuggability)
- **Model Source:** HuggingFace download (Ollama cache incompatible with transformers)
- **Process Isolation:** fcntl file locks (exclusive training, shared inference)
- **VRAM Budget:** 9-11 GB for training, 9 GB conservative minimum
- **LoRA Config:** r=32, alpha=64, 7 target modules
- **DPO Hyperparams:** beta=0.1, lr=5e-6, batch_size=1, grad_accum=16
- **Walk-Forward:** 500 train / 100 test pairs per window, temporal ordering
- **Replay Buffer:** 15% old data to prevent catastrophic forgetting
- **Evaluation Metrics:** IC, return-weighted IC, Brier, MACE, regime-stratified IC
- **Promotion:** IC > 0.02, Brier > 0.01, p < 0.05, N >= 100, 24h cooldown, 3 max/week
- **Pre-flight Order:** Data -> Temporal -> VRAM -> Lock -> Load
- **Adapter Loading:** 30-day max age, graceful fallback to base

### DPO Pipeline (Session 10)
- **5-Phase CLI:** Load -> Verify -> Reward -> Pairs -> Train
- **Phase 1:** Filters examples requiring direction in generator_signal
- **Phase 2:** Async batch verification via MarketDataService
- **Phase 3:** Reward computation per matched pair
- **Phase 4:** Preference pair construction, optional `--save-pairs`
- **Phase 5:** Calls `train_dpo()`, exits 1 on failure
- **CLI flags:** `--dataset` (required), `--output`, `--save-pairs`, `--dry-run`, `--min-delta`, `--force`

### Dataset Generation (Session 9)
- **Scale:** 13,500 examples (10 symbols x 6 timeframes x 15 windows x 3 tasks x 5 personas)
- **3-Phase Parallelization:** Phase 1 parallel data prep (async.gather), Phase 2 sequential VRAM inference, Phase 3 parallel post-processing
- **Resume:** Incremental JSONL saving + context_id tracking + state persistence
- **Task Types:** PREDICT_DIRECTION, ASSESS_MOMENTUM, IDENTIFY_SUPPORT_RESISTANCE
- **Batched Execution:** Run by timeframe to keep individual runs <16 hours
- **CLI:** `generate_training_dataset.py` with quick test mode verified

### Verifier Layer
- Timeframe-adaptive horizons (1m->60 bars, 1h->24 bars, 1d->5 bars)
- Log returns for additivity and DPO compatibility
- Entry at next bar open (realistic execution)

### Reward Layer
- Clipped linear reward bounded to [-1, 1] for stable DPO gradients
- Three components: return (0.50), directional (0.30), MAE (0.20)

### Evaluation Layer
- Spearman IC primary, BH-FDR correction for multiple hypothesis testing
- Metrics: IC, Sharpe, Sortino, Calmar, max drawdown, win rate, profit factor

## File Index

### Configuration
- `config/settings.py` - Pydantic settings + DPOTrainingSettings + DatasetGenerationSettings
- `config/fee_model.py` - FeeModelSettings for Binance Futures USDT-M fee calculations
- `pyproject.toml` - Project metadata, tool configs

### Data Layer
- `data/indicators.py` - Technical indicators (RSI, MACD, BB)
- `data/cache_wrapper.py` - AsyncDiskCache with asyncio.to_thread()
- `data/market_data.py` - CCXT client with context manager
- `data/regime_filter.py` - RegimeClassifier with volatility percentiles
- `data/prompt_builder.py` - Task sampling with isolated RNG
- `data/historical_windows.py` - Window walking with completeness validation
- `data/inference_queue.py` - Sequential job processor with JSONL streaming

### Utils
- `utils/progress_tracker.py` - Progress tracking with rolling ETA, JSON state persistence

### Swarm Layer
- `swarm/exceptions.py` - Custom exception hierarchy
- `swarm/ollama_client.py` - VRAM-aware Ollama client with semaphore
- `swarm/generator.py` - Signal generator with 5 personas
- `swarm/critic.py` - Critique generation with deepseek-r1:14b
- `swarm/orchestrator.py` - LangGraph workflow + multi-persona
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
- `generate_training_dataset.py` - Main CLI for dataset generation (3-phase parallelization)
- `run_dpo_training.py` - End-to-end DPO pipeline CLI (5-phase: load/verify/reward/pairs/train)

### Verifier Layer
- `verifier/` - constants, config, outcome, validator, engine

### Evaluation Layer
- `eval/` - config, metrics, engine

### Tests (448 total)
- `tests/test_config.py` - 40 tests (18 original + 22 FeeModelSettings)
- `tests/test_indicators.py` - 19 tests
- `tests/test_data_layer.py` - 21 tests
- `tests/test_ollama_client.py` - 17 tests
- `tests/test_generator.py` - 20 tests
- `tests/test_critic.py` - 22 tests
- `tests/test_orchestrator.py` - 23 tests
- `tests/test_verifier/` - 64 tests
- `tests/test_reward/` - 63 tests
- `tests/test_eval/` - 49 tests
- `tests/test_training/` - 71 tests + `test_dpo_pipeline.py` (23 tests)
- `tests/test_swarm/test_adapter_loader.py` - 16 tests

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

## Working Decisions

- Separate `requirements.txt` (inference) and `requirements-training.txt` (DPO)
- Process A and Process B NEVER run concurrently (enforced by process_lock.py)
- Models accessed by exact tag (e.g., `qwen3:8b`)
- Adapter directory: `models/adapters/adapter-{PERSONA}-{TIMESTAMP}.promoted`
- Batched execution: Generate by timeframe to keep runs <16 hours
- Deferred: M3 (fetch optimization), M4 (type hints), Session 9 tests

---

**Total Tests:** 448 passing
**Python Version:** 3.13.7
