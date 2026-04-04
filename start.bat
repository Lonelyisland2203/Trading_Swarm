@echo off
REM Cross-platform launcher for Trading Swarm (Windows)

echo Trading Swarm - Startup Script
echo ===============================
echo.

REM Check if virtual environment exists
if not exist "venv\" (
    echo ERROR: Virtual environment not found.
    echo Run 'python -m venv venv' and 'pip install -r requirements.txt'
    exit /b 1
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Check if .env exists
if not exist ".env" (
    echo ERROR: .env file not found.
    echo Copy .env.example to .env and configure.
    exit /b 1
)

REM Verify Ollama is running
echo Checking Ollama service...
curl -s http://localhost:11434/api/tags >nul 2>&1
if errorlevel 1 (
    echo ERROR: Ollama service not running.
    echo Start Ollama first.
    exit /b 1
)

echo OK: Ollama service running
echo.

REM Run main application (placeholder for Session 4)
echo Starting Trading Swarm...
echo.
python -c "from config.settings import settings; print('Configuration loaded successfully'); print(f'Generator: {settings.ollama.generator_model}'); print(f'Critic: {settings.ollama.critic_model}')"

echo.
echo Application ready. (Full orchestrator in Session 4)
pause
