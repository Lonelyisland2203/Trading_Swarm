# Trading Swarm

Autonomous AI trading signal system with self-improvement capabilities using DPO fine-tuning.

## Architecture

- **Generator**: Qwen3-8B (non-thinking mode, 4-bit quantization)
- **Critic**: DeepSeek-R1-14B (native reasoning, 4-bit quantization)
- **Orchestration**: LangGraph
- **Verification**: Pure pandas backtesting
- **Training**: DPO fine-tuning (separate process)

## Hardware Requirements

- **GPU**: RTX 5070 Ti (16 GB VRAM) or equivalent
- **RAM**: 16 GB minimum
- **Disk**: 500 MB free (models stored by Ollama separately)
- **CUDA**: Required for local GPU acceleration

## Prerequisites

1. **Python 3.11+**
2. **Ollama** installed and running (`ollama serve`)
3. **Models pulled**:
   ```bash
   ollama pull qwen3:8b
   ollama pull deepseek-r1:14b
   ```

## Quick Start

```bash
# 1. Setup environment
make setup

# 2. Activate virtual environment
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate.bat  # Windows

# 3. Configure environment
cp .env.example .env
# Edit .env with your settings

# 4. Verify Ollama models
make verify-ollama

# 5. Run tests
make test

# 6. Start application (Session 4+)
./start.sh  # Linux/Mac
# or
start.bat   # Windows
```

## Project Structure

```
trading-swarm/
├── config/          # Configuration and settings
├── swarm/           # LLM swarm agents (generator, critic)
├── data/            # Market data fetching and preprocessing
├── verifier/        # Backtesting and verification
├── training/        # DPO fine-tuning (separate process)
├── eval/            # Evaluation and metrics
├── tests/           # Test suite
├── models/          # Model weights (gitignored)
├── outputs/         # Results and logs (gitignored)
└── data/cache/      # Market data cache (gitignored)
```

## Development Sessions

This project is implemented in 12 sessions:

- [x] **Session 1**: Environment Setup + Config Layer (current)
- [ ] **Session 2**: Data Layer
- [ ] **Session 3**: Swarm Layer Part 1 (Generator)
- [ ] **Session 4**: Swarm Layer Part 2 (Critic + Orchestrator)
- [ ] **Session 5**: Verifier Layer
- [ ] **Session 6**: Training Layer Part 1 (Dataset)
- [ ] **Session 7**: Training Layer Part 2 (DPO)
- [ ] **Session 8**: Evaluation Layer
- [ ] **Session 9**: Integration Testing
- [ ] **Session 10**: Performance Optimization
- [ ] **Session 11**: Production Hardening
- [ ] **Session 12**: Documentation + Deployment

## Testing

```bash
# Run all tests
make test

# Run with coverage
make test-cov

# Run specific test file
pytest tests/test_config.py -v
```

## Critical Constraints

1. **VRAM Management**: Models NEVER loaded simultaneously (16 GB limit)
2. **Process Isolation**: Inference (Process A) and Training (Process B) run separately
3. **Quantization**: All models use 4-bit quantization
4. **Ollama Keep-Alive**: Set to 0 to force model unloading

## Configuration

Key settings in `.env`:

- `OLLAMA_GENERATOR_MODEL`: Generator model tag (default: qwen3:8b)
- `OLLAMA_CRITIC_MODEL`: Critic model tag (default: deepseek-r1:14b)
- `OLLAMA_KEEP_ALIVE`: Model persistence (must be 0)
- `REWARD_WEIGHT_*`: Reward function weights (must sum to 1.0)

See `.env.example` for complete configuration options.

## License

MIT
