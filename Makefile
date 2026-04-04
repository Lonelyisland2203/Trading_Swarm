.PHONY: setup install test run lint clean help

# Default target
.DEFAULT_GOAL := help

help: ## Show this help message
	@echo "Trading Swarm - Available Commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

setup: ## Create virtual environment and install dependencies
	python3 -m venv venv
	. venv/bin/activate && pip install --upgrade pip
	. venv/bin/activate && pip install -r requirements.txt
	@echo ""
	@echo "✓ Setup complete. Activate environment with: source venv/bin/activate"
	@echo ""
	@echo "Next steps:"
	@echo "  1. Ensure Ollama is running: ollama serve"
	@echo "  2. Pull models: ollama pull qwen3:8b && ollama pull deepseek-r1:14b"
	@echo "  3. Copy .env.example to .env and configure"
	@echo "  4. Run tests: make test"

install: ## Install dependencies (assumes venv already exists)
	pip install --upgrade pip
	pip install -r requirements.txt

install-training: ## Install training dependencies (SEPARATE environment)
	@echo "WARNING: Installing training dependencies."
	@echo "Ensure Process A (inference) is NOT running."
	@read -p "Continue? [y/N] " confirm && [ "$$confirm" = "y" ] || exit 1
	pip install -r requirements-training.txt

test: ## Run all tests
	pytest tests/ -v

test-config: ## Run configuration tests only
	pytest tests/test_config.py -v

test-cov: ## Run tests with coverage report
	pytest tests/ -v --cov=. --cov-report=html --cov-report=term

lint: ## Run code quality checks
	@echo "Running ruff..."
	@ruff check . || true
	@echo "Running black check..."
	@black --check . || true

format: ## Auto-format code
	black .
	ruff check --fix .

verify-ollama: ## Check Ollama service and models
	@echo "Checking Ollama service..."
	@curl -s http://localhost:11434/api/tags || echo "ERROR: Ollama not running. Start with: ollama serve"
	@echo ""
	@echo "Checking required models..."
	@curl -s http://localhost:11434/api/tags | grep -q "qwen3:8b" && echo "✓ qwen3:8b found" || echo "✗ qwen3:8b missing - run: ollama pull qwen3:8b"
	@curl -s http://localhost:11434/api/tags | grep -q "deepseek-r1:14b" && echo "✓ deepseek-r1:14b found" || echo "✗ deepseek-r1:14b missing - run: ollama pull deepseek-r1:14b"

run: ## Run the main application (placeholder)
	@echo "Not implemented yet - see Session 4 (Orchestrator)"

clean: ## Remove cache and temporary files
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .coverage htmlcov
	rm -rf outputs/* data/cache/* .cache/*

clean-all: clean ## Remove venv and all generated files
	rm -rf venv
