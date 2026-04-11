"""Integration tests for signal loop."""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from signals.signal_loop import (
    run_cycle,
    run_loop,
    should_override_signal,
    generate_signal_for_symbol,
    evaluate_with_critic,
    format_recent_ohlcv,
)
from signals.signal_models import Signal
from swarm.critic import CritiqueResult
from swarm.generator import GeneratorSignal, TradingPersona
from data.prompt_builder import TaskType
from data.regime_filter import MarketRegime


@pytest.fixture
def mock_ohlcv_df():
    """Create mock OHLCV DataFrame."""
    n = 100
    return pd.DataFrame({
        "timestamp": range(1704000000000, 1704000000000 + n * 3600000, 3600000),
        "open": [42000.0 + i * 10 for i in range(n)],
        "high": [42100.0 + i * 10 for i in range(n)],
        "low": [41900.0 + i * 10 for i in range(n)],
        "close": [42050.0 + i * 10 for i in range(n)],
        "volume": [100 + i for i in range(n)],
    })


@pytest.fixture
def mock_generator_signal():
    """Create mock generator signal."""
    return GeneratorSignal(
        task_type=TaskType.PREDICT_DIRECTION,
        signal_data={"direction": "HIGHER", "confidence": 0.85},
        reasoning="Strong bullish momentum",
        persona=TradingPersona.MOMENTUM,
        raw_response='{"direction": "HIGHER", "confidence": 0.85}',
    )


@pytest.fixture
def mock_critique_accept():
    """Create mock accepting critique."""
    return CritiqueResult(
        reasoning_quality=0.8,
        technical_alignment=0.75,
        confidence_calibration=0.7,
        critique="Signal looks sound",
        recommendation="ACCEPT",
        raw_response="...",
    )


@pytest.fixture
def mock_critique_reject_override():
    """Create mock rejecting critique that triggers override."""
    return CritiqueResult(
        reasoning_quality=0.4,  # Below 0.5
        technical_alignment=0.45,  # Below 0.5
        confidence_calibration=0.5,
        critique="Reasoning is flawed and indicators don't support",
        recommendation="REJECT",
        raw_response="...",
    )


class TestFormatRecentOhlcv:
    """Tests for format_recent_ohlcv helper."""

    def test_formats_last_n_rows(self, mock_ohlcv_df):
        """Formats last N rows of OHLCV data."""
        result = format_recent_ohlcv(mock_ohlcv_df, n=3)

        lines = result.strip().split("\n")
        assert len(lines) == 3
        assert "O:" in lines[0]
        assert "H:" in lines[0]
        assert "L:" in lines[0]
        assert "C:" in lines[0]


class TestGenerateSignalForSymbol:
    """Tests for generate_signal_for_symbol function."""

    @pytest.fixture
    def mock_dependencies(self, mock_ohlcv_df, mock_generator_signal):
        """Set up all mock dependencies."""
        market_data = AsyncMock()
        market_data.fetch_ohlcv = AsyncMock(return_value=mock_ohlcv_df)

        ollama_client = AsyncMock()

        fee_model = MagicMock()
        fee_model.minimum_profitable_return_pct = MagicMock(return_value=0.06)

        prompt_builder = MagicMock()
        prompt_builder.build_prompt = MagicMock(return_value="Test prompt")

        return market_data, ollama_client, fee_model, prompt_builder

    @pytest.mark.asyncio
    async def test_successful_signal_generation(
        self, mock_dependencies, mock_generator_signal
    ):
        """Successfully generates signal."""
        market_data, ollama_client, fee_model, prompt_builder = mock_dependencies

        with patch("signals.signal_loop.generate_signal", return_value=mock_generator_signal):
            with patch("signals.signal_loop.compute_all_indicators", return_value={"rsi": 55.0}):
                with patch("signals.signal_loop.RegimeClassifier") as mock_regime:
                    mock_regime.return_value.get_current_regime.return_value = MarketRegime.NEUTRAL

                    signal, prompt = await generate_signal_for_symbol(
                        symbol="BTC/USDT",
                        timeframe="1h",
                        market_data_service=market_data,
                        ollama_client=ollama_client,
                        fee_model=fee_model,
                        prompt_builder=prompt_builder,
                    )

                    assert signal is not None
                    assert signal.symbol == "BTC/USDT"
                    assert signal.direction == "LONG"  # HIGHER -> LONG
                    assert signal.confidence == 0.85

    @pytest.mark.asyncio
    async def test_data_unavailable_returns_none(self, mock_dependencies):
        """Returns None when data is unavailable."""
        market_data, ollama_client, fee_model, prompt_builder = mock_dependencies

        from data.market_data import DataUnavailableError
        market_data.fetch_ohlcv = AsyncMock(side_effect=DataUnavailableError("Test"))

        signal, prompt = await generate_signal_for_symbol(
            symbol="BTC/USDT",
            timeframe="1h",
            market_data_service=market_data,
            ollama_client=ollama_client,
            fee_model=fee_model,
            prompt_builder=prompt_builder,
        )

        assert signal is None
        assert prompt is None

    @pytest.mark.asyncio
    async def test_insufficient_data_returns_none(self, mock_dependencies):
        """Returns None when data has too few bars."""
        market_data, ollama_client, fee_model, prompt_builder = mock_dependencies

        # Only 10 bars (need 50+)
        market_data.fetch_ohlcv = AsyncMock(return_value=pd.DataFrame({
            "timestamp": range(10),
            "open": range(10),
            "high": range(10),
            "low": range(10),
            "close": range(10),
            "volume": range(10),
        }))

        signal, prompt = await generate_signal_for_symbol(
            symbol="BTC/USDT",
            timeframe="1h",
            market_data_service=market_data,
            ollama_client=ollama_client,
            fee_model=fee_model,
            prompt_builder=prompt_builder,
        )

        assert signal is None


class TestEvaluateWithCritic:
    """Tests for evaluate_with_critic function."""

    @pytest.fixture
    def test_signal(self):
        """Create test signal."""
        return Signal(
            symbol="BTC/USDT",
            timeframe="1h",
            direction="LONG",
            confidence=0.85,
            reasoning="Test",
            persona="MOMENTUM",
            timestamp=datetime.now(timezone.utc),
            market_regime="neutral",
            current_price=42000.0,
            rsi=55.0,
            macd=0.05,
            macd_signal=0.03,
            bb_position=0.65,
        )

    @pytest.mark.asyncio
    async def test_critic_accepts_signal(
        self, test_signal, mock_ohlcv_df, mock_critique_accept
    ):
        """Critic accepting signal keeps original direction."""
        ollama_client = AsyncMock()

        with patch("signals.signal_loop.evaluate_signal", return_value=mock_critique_accept):
            result = await evaluate_with_critic(
                signal=test_signal,
                df=mock_ohlcv_df,
                prompt="Test prompt",
                ollama_client=ollama_client,
            )

            assert result.final_direction == "LONG"
            assert result.critic_override is False
            assert result.critic_score == mock_critique_accept.score

    @pytest.mark.asyncio
    async def test_critic_rejects_overrides_to_flat(
        self, test_signal, mock_ohlcv_df, mock_critique_reject_override
    ):
        """Critic rejecting with low scores overrides to FLAT."""
        ollama_client = AsyncMock()

        with patch("signals.signal_loop.evaluate_signal", return_value=mock_critique_reject_override):
            result = await evaluate_with_critic(
                signal=test_signal,
                df=mock_ohlcv_df,
                prompt="Test prompt",
                ollama_client=ollama_client,
            )

            assert result.final_direction == "FLAT"
            assert result.critic_override is True

    @pytest.mark.asyncio
    async def test_critic_failure_keeps_original(
        self, test_signal, mock_ohlcv_df
    ):
        """Critic failure keeps original signal."""
        ollama_client = AsyncMock()

        with patch("signals.signal_loop.evaluate_signal", return_value=None):
            result = await evaluate_with_critic(
                signal=test_signal,
                df=mock_ohlcv_df,
                prompt="Test prompt",
                ollama_client=ollama_client,
            )

            assert result.final_direction == "LONG"
            assert result.critic_override is False


class TestRunCycle:
    """Tests for run_cycle function."""

    @pytest.fixture
    def mock_all(self, mock_ohlcv_df, mock_generator_signal, mock_critique_accept, tmp_path, monkeypatch):
        """Set up all mocks for run_cycle."""
        # Mock STOP file path
        stop_path = tmp_path / "STOP"
        monkeypatch.setattr("signals.preflight.STOP_FILE_PATH", stop_path)

        # Mock log paths
        log_path = tmp_path / "signal_log.jsonl"
        pending_path = tmp_path / "pending.jsonl"
        monkeypatch.setattr("signals.signal_logger.SIGNAL_LOG_PATH", log_path)
        monkeypatch.setattr("signals.accuracy_tracker.PENDING_PATH", pending_path)

        return {
            "stop_path": stop_path,
            "log_path": log_path,
        }

    @pytest.mark.asyncio
    async def test_run_cycle_generates_signals(self, mock_all, mock_ohlcv_df, mock_generator_signal, mock_critique_accept):
        """Run cycle generates signals for all symbols."""
        with patch("signals.signal_loop.MarketDataService") as MockMDS:
            mock_mds = AsyncMock()
            mock_mds.fetch_ohlcv = AsyncMock(return_value=mock_ohlcv_df)
            MockMDS.return_value.__aenter__ = AsyncMock(return_value=mock_mds)
            MockMDS.return_value.__aexit__ = AsyncMock()

            with patch("signals.signal_loop.OllamaClient") as MockOllama:
                mock_ollama = AsyncMock()
                mock_ollama.unload_current = AsyncMock()
                MockOllama.return_value.__aenter__ = AsyncMock(return_value=mock_ollama)
                MockOllama.return_value.__aexit__ = AsyncMock()

                with patch("signals.signal_loop.generate_signal", return_value=mock_generator_signal):
                    with patch("signals.signal_loop.evaluate_signal", return_value=mock_critique_accept):
                        with patch("signals.signal_loop.compute_all_indicators", return_value={"rsi": 55.0}):
                            with patch("signals.signal_loop.RegimeClassifier") as MockRegime:
                                MockRegime.return_value.get_current_regime.return_value = MarketRegime.NEUTRAL
                                with patch("signals.signal_loop.compute_bb_position") as mock_bb:
                                    mock_bb.return_value = pd.Series([0.5])

                                    signals = await run_cycle(
                                        symbols=["BTC/USDT"],
                                        timeframe="1h",
                                        execute=False,
                                        min_confidence=0.6,
                                    )

                                    assert len(signals) == 1
                                    assert signals[0].symbol == "BTC/USDT"


class TestRunLoop:
    """Tests for run_loop function."""

    @pytest.mark.asyncio
    async def test_once_mode_runs_single_cycle(self, tmp_path, monkeypatch):
        """Once mode runs a single cycle and exits."""
        # Mock STOP file
        stop_path = tmp_path / "STOP"
        monkeypatch.setattr("signals.preflight.STOP_FILE_PATH", stop_path)

        # Mock preflight
        with patch("signals.signal_loop.run_preflight_checks") as mock_preflight:
            from signals.preflight import PreflightResult
            mock_preflight.return_value = PreflightResult(passed=True, reason="OK")

            with patch("signals.signal_loop.acquire_inference_lock"):
                with patch("signals.signal_loop.run_cycle", return_value=[]) as mock_cycle:
                    await run_loop(
                        symbols=["BTC/USDT"],
                        timeframe="1h",
                        execute=False,
                        once=True,
                    )

                    # run_cycle should be called exactly once
                    mock_cycle.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_file_halts_loop(self, tmp_path, monkeypatch):
        """STOP file halts the loop immediately."""
        stop_path = tmp_path / "STOP"
        stop_path.touch()  # Create STOP file
        monkeypatch.setattr("signals.preflight.STOP_FILE_PATH", stop_path)

        with patch("signals.signal_loop.run_preflight_checks") as mock_preflight:
            with patch("signals.signal_loop.run_cycle") as mock_cycle:
                await run_loop(
                    symbols=["BTC/USDT"],
                    timeframe="1h",
                    execute=False,
                    once=False,  # Would loop forever without STOP file
                )

                # run_cycle should NOT be called
                mock_cycle.assert_not_called()

    @pytest.mark.asyncio
    async def test_preflight_failure_waits_in_once_mode(self, tmp_path, monkeypatch):
        """Preflight failure in once mode exits."""
        stop_path = tmp_path / "STOP"
        monkeypatch.setattr("signals.preflight.STOP_FILE_PATH", stop_path)

        with patch("signals.signal_loop.run_preflight_checks") as mock_preflight:
            from signals.preflight import PreflightResult
            mock_preflight.return_value = PreflightResult(passed=False, reason="VRAM low")

            with patch("signals.signal_loop.run_cycle") as mock_cycle:
                await run_loop(
                    symbols=["BTC/USDT"],
                    timeframe="1h",
                    execute=False,
                    once=True,
                )

                # run_cycle should NOT be called
                mock_cycle.assert_not_called()
