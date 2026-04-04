#!/usr/bin/env bash
# Cross-platform launcher for Trading Swarm

set -e

echo "Trading Swarm - Startup Script"
echo "==============================="
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "ERROR: Virtual environment not found."
    echo "Run 'make setup' first."
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "ERROR: .env file not found."
    echo "Copy .env.example to .env and configure."
    exit 1
fi

# Verify Ollama is running
echo "Checking Ollama service..."
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "ERROR: Ollama service not running."
    echo "Start with: ollama serve"
    exit 1
fi

echo "✓ Ollama service running"
echo ""

# Load environment
export $(grep -v '^#' .env | xargs)

# Run main application (placeholder for Session 4)
echo "Starting Trading Swarm..."
echo ""
python -c "from config.settings import settings; print('Configuration loaded successfully'); print(f'Generator: {settings.ollama.generator_model}'); print(f'Critic: {settings.ollama.critic_model}')"

echo ""
echo "Application ready. (Full orchestrator in Session 4)"
