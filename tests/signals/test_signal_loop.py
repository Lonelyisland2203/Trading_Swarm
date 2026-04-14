"""
Integration tests for refactored signal loop.

Session 17N: Tests for new pipeline:
1. XGBoost signal generation
2. LLM context generation
3. DeepSeek risk filter
4. Synthesis node
5. Signal logging and execution
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from signals.signal_loop import (
    run_cycle,
    run_loop,
    call_risk_filter,
    synthesis_to_legacy_signal,
)
from signals.signal_models import Signal
from signals.xgboost_signal import XGBoostSignal
from signals.llm_context import LLMContext
from signals.synthesis import SynthesisOutput


@pytest.fixture
def mock_ohlcv_df():
    """Create mock OHLCV DataFrame."""
    n = 100
    return pd.DataFrame(
        {
            "timestamp": range(1704000000000, 1704000000000 + n * 3600000, 3600000),
            "open": [42000.0 + i * 10 for i in range(n)],
            "high": [42100.0 + i * 10 for i in range(n)],
            "low": [41900.0 + i * 10 for i in range(n)],
            "close": [42050.0 + i * 10 for i in range(n)],
            "volume": [100 + i for i in range(n)],
        }
    )


@pytest.fixture
def mock_xgboost_signal():
    """Create mock XGBoost signal."""
    return XGBoostSignal(
        symbol="BTC/USDT",
        timeframe="1h",
        direction="LONG",
        probability=0.72,
        confidence=0.44,
        features={"rsi": 55.0, "macd_line": 0.05},
        timestamp=datetime.now(timezone.utc),
    )


@pytest.fixture
def mock_llm_context():
    """Create mock LLM context."""
    return LLMContext(
        bullish_factors=["Strong funding rate", "Increasing OI"],
        bearish_factors=["RSI approaching overbought"],
        regime_flag="confirming",
        confidence=0.75,
    )


class TestCallRiskFilter:
    """Tests for DeepSeek risk filter."""

    @pytest.mark.asyncio
    async def test_risk_filter_approve(self, mock_xgboost_signal, mock_llm_context):
        """Risk filter returns False (approve) when Ollama responds APPROVE."""
        with patch("httpx.AsyncClient") as MockClient:
            mock_response = MagicMock()
            mock_response.json.return_value = {"response": "APPROVE - signal looks reasonable"}
            mock_response.raise_for_status = MagicMock()

            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock()
            MockClient.return_value = mock_client

            result = await call_risk_filter(mock_xgboost_signal, mock_llm_context)

            assert result is False  # Not vetoed

    @pytest.mark.asyncio
    async def test_risk_filter_veto(self, mock_xgboost_signal, mock_llm_context):
        """Risk filter returns True (veto) when Ollama responds VETO."""
        with patch("httpx.AsyncClient") as MockClient:
            mock_response = MagicMock()
            mock_response.json.return_value = {"response": "VETO - high risk detected"}
            mock_response.raise_for_status = MagicMock()

            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock()
            MockClient.return_value = mock_client

            result = await call_risk_filter(mock_xgboost_signal, mock_llm_context)

            assert result is True  # Vetoed

    @pytest.mark.asyncio
    async def test_risk_filter_failure_defaults_to_approve(
        self, mock_xgboost_signal, mock_llm_context
    ):
        """Risk filter returns False (approve) on failure."""
        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            # Simulate error on POST request
            mock_client.post = AsyncMock(side_effect=Exception("Connection error"))

            # Set up async context manager properly
            async def mock_aenter(self):
                return mock_client

            async def mock_aexit(self, *args):
                pass

            MockClient.return_value.__aenter__ = mock_aenter
            MockClient.return_value.__aexit__ = mock_aexit

            result = await call_risk_filter(mock_xgboost_signal, mock_llm_context)

            assert result is False  # Default to approve on failure

    @pytest.mark.asyncio
    async def test_risk_filter_with_none_context(self, mock_xgboost_signal):
        """Risk filter works with None LLM context."""
        with patch("httpx.AsyncClient") as MockClient:
            mock_response = MagicMock()
            mock_response.json.return_value = {"response": "APPROVE"}
            mock_response.raise_for_status = MagicMock()

            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock()
            MockClient.return_value = mock_client

            result = await call_risk_filter(mock_xgboost_signal, None)

            assert result is False


class TestSynthesisToLegacySignal:
    """Tests for conversion to legacy Signal format."""

    def test_converts_to_legacy_format(self, mock_xgboost_signal, mock_llm_context):
        """Converts synthesis output to legacy Signal dataclass."""
        synthesis_output = SynthesisOutput(
            direction="LONG",
            position_size_fraction=1.0,
            rationale="XGBoost LONG with confirming context",
            components={},
        )

        signal = synthesis_to_legacy_signal(
            mock_xgboost_signal, synthesis_output, mock_llm_context, False
        )

        assert isinstance(signal, Signal)
        assert signal.symbol == "BTC/USDT"
        assert signal.direction == "LONG"
        assert signal.final_direction == "LONG"
        assert signal.market_regime == "confirming"
        assert signal.persona == "XGBOOST"

    def test_handles_veto(self, mock_xgboost_signal, mock_llm_context):
        """Handles vetoed signal correctly."""
        synthesis_output = SynthesisOutput(
            direction="FLAT",
            position_size_fraction=0.0,
            rationale="DeepSeek critic veto",
            components={},
        )

        signal = synthesis_to_legacy_signal(
            mock_xgboost_signal, synthesis_output, mock_llm_context, True
        )

        assert signal.final_direction == "FLAT"
        assert signal.critic_override is True
        assert signal.critic_recommendation == "VETO"

    def test_handles_none_context(self, mock_xgboost_signal):
        """Handles None LLM context."""
        synthesis_output = SynthesisOutput(
            direction="LONG",
            position_size_fraction=0.7,
            rationale="No context available",
            components={},
        )

        signal = synthesis_to_legacy_signal(mock_xgboost_signal, synthesis_output, None, False)

        assert signal.market_regime == "unknown"


class TestRunCycle:
    """Tests for run_cycle function."""

    @pytest.fixture
    def mock_all(self, tmp_path, monkeypatch):
        """Set up all mocks for run_cycle."""
        # Mock STOP file path
        stop_path = tmp_path / "STOP"
        from utils.stop_file import StopFileChecker

        monkeypatch.setattr("utils.stop_file.default_stop_checker", StopFileChecker(stop_path))

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
    async def test_run_cycle_generates_signals(
        self, mock_all, mock_ohlcv_df, mock_xgboost_signal, mock_llm_context
    ):
        """Run cycle generates signals for all symbols."""
        with patch("signals.signal_loop.MarketDataService") as MockMDS:
            mock_mds = AsyncMock()
            mock_mds.fetch_ohlcv = AsyncMock(return_value=mock_ohlcv_df)
            MockMDS.return_value.__aenter__ = AsyncMock(return_value=mock_mds)
            MockMDS.return_value.__aexit__ = AsyncMock()

            with patch(
                "signals.signal_loop.generate_xgboost_signal", return_value=mock_xgboost_signal
            ):
                with patch(
                    "signals.signal_loop.generate_market_context", return_value=mock_llm_context
                ):
                    with patch("signals.signal_loop.call_risk_filter", return_value=False):
                        signals = await run_cycle(
                            symbols=["BTC/USDT"],
                            timeframe="1h",
                            execute=False,
                            min_confidence=0.6,
                        )

                        assert len(signals) == 1
                        assert signals[0].symbol == "BTC/USDT"

    @pytest.mark.asyncio
    async def test_run_cycle_handles_xgboost_failure(self, mock_all, mock_ohlcv_df):
        """Run cycle handles XGBoost signal generation failure."""
        with patch("signals.signal_loop.MarketDataService") as MockMDS:
            mock_mds = AsyncMock()
            mock_mds.fetch_ohlcv = AsyncMock(return_value=mock_ohlcv_df)
            MockMDS.return_value.__aenter__ = AsyncMock(return_value=mock_mds)
            MockMDS.return_value.__aexit__ = AsyncMock()

            with patch("signals.signal_loop.generate_xgboost_signal", return_value=None):
                signals = await run_cycle(
                    symbols=["BTC/USDT"],
                    timeframe="1h",
                    execute=False,
                    min_confidence=0.6,
                )

                assert len(signals) == 0

    @pytest.mark.asyncio
    async def test_run_cycle_handles_llm_context_failure(
        self, mock_all, mock_ohlcv_df, mock_xgboost_signal
    ):
        """Run cycle handles LLM context generation failure."""
        with patch("signals.signal_loop.MarketDataService") as MockMDS:
            mock_mds = AsyncMock()
            mock_mds.fetch_ohlcv = AsyncMock(return_value=mock_ohlcv_df)
            MockMDS.return_value.__aenter__ = AsyncMock(return_value=mock_mds)
            MockMDS.return_value.__aexit__ = AsyncMock()

            with patch(
                "signals.signal_loop.generate_xgboost_signal", return_value=mock_xgboost_signal
            ):
                with patch(
                    "signals.signal_loop.generate_market_context",
                    side_effect=Exception("LLM error"),
                ):
                    with patch("signals.signal_loop.call_risk_filter", return_value=False):
                        signals = await run_cycle(
                            symbols=["BTC/USDT"],
                            timeframe="1h",
                            execute=False,
                            min_confidence=0.6,
                        )

                        # Should still generate signal with None context
                        assert len(signals) == 1

    @pytest.mark.asyncio
    async def test_run_cycle_veto_produces_flat(
        self, mock_all, mock_ohlcv_df, mock_xgboost_signal, mock_llm_context
    ):
        """Run cycle produces FLAT when risk filter vetoes."""
        with patch("signals.signal_loop.MarketDataService") as MockMDS:
            mock_mds = AsyncMock()
            mock_mds.fetch_ohlcv = AsyncMock(return_value=mock_ohlcv_df)
            MockMDS.return_value.__aenter__ = AsyncMock(return_value=mock_mds)
            MockMDS.return_value.__aexit__ = AsyncMock()

            with patch(
                "signals.signal_loop.generate_xgboost_signal", return_value=mock_xgboost_signal
            ):
                with patch(
                    "signals.signal_loop.generate_market_context", return_value=mock_llm_context
                ):
                    with patch("signals.signal_loop.call_risk_filter", return_value=True):  # Veto
                        signals = await run_cycle(
                            symbols=["BTC/USDT"],
                            timeframe="1h",
                            execute=False,
                            min_confidence=0.6,
                        )

                        assert len(signals) == 1
                        assert signals[0].final_direction == "FLAT"
                        assert signals[0].critic_override is True


class TestRunLoop:
    """Tests for run_loop function."""

    @pytest.mark.asyncio
    async def test_once_mode_runs_single_cycle(self, tmp_path, monkeypatch):
        """Once mode runs a single cycle and exits."""
        # Mock STOP file
        stop_path = tmp_path / "STOP"
        from utils.stop_file import StopFileChecker

        monkeypatch.setattr("utils.stop_file.default_stop_checker", StopFileChecker(stop_path))

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
        from utils.stop_file import StopFileChecker

        monkeypatch.setattr("utils.stop_file.default_stop_checker", StopFileChecker(stop_path))

        with patch("signals.signal_loop.run_preflight_checks"):
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
        from utils.stop_file import StopFileChecker

        monkeypatch.setattr("utils.stop_file.default_stop_checker", StopFileChecker(stop_path))

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


class TestPipelineIntegration:
    """Tests for full pipeline integration."""

    @pytest.fixture
    def integration_mocks(
        self, tmp_path, monkeypatch, mock_ohlcv_df, mock_xgboost_signal, mock_llm_context
    ):
        """Set up all mocks for integration test."""
        stop_path = tmp_path / "STOP"
        log_path = tmp_path / "signal_log.jsonl"
        pending_path = tmp_path / "pending.jsonl"

        from utils.stop_file import StopFileChecker

        monkeypatch.setattr("utils.stop_file.default_stop_checker", StopFileChecker(stop_path))
        monkeypatch.setattr("signals.signal_logger.SIGNAL_LOG_PATH", log_path)
        monkeypatch.setattr("signals.accuracy_tracker.PENDING_PATH", pending_path)

        return {
            "ohlcv": mock_ohlcv_df,
            "xgb_signal": mock_xgboost_signal,
            "llm_context": mock_llm_context,
            "log_path": log_path,
        }

    @pytest.mark.asyncio
    async def test_full_pipeline_with_confirming_context(self, integration_mocks):
        """Full pipeline with confirming context produces LONG signal."""
        with patch("signals.signal_loop.MarketDataService") as MockMDS:
            mock_mds = AsyncMock()
            mock_mds.fetch_ohlcv = AsyncMock(return_value=integration_mocks["ohlcv"])
            MockMDS.return_value.__aenter__ = AsyncMock(return_value=mock_mds)
            MockMDS.return_value.__aexit__ = AsyncMock()

            with patch(
                "signals.signal_loop.generate_xgboost_signal",
                return_value=integration_mocks["xgb_signal"],
            ):
                with patch(
                    "signals.signal_loop.generate_market_context",
                    return_value=integration_mocks["llm_context"],
                ):
                    with patch("signals.signal_loop.call_risk_filter", return_value=False):
                        signals = await run_cycle(
                            symbols=["BTC/USDT"],
                            timeframe="1h",
                            execute=False,
                            min_confidence=0.6,
                        )

                        assert len(signals) == 1
                        signal = signals[0]

                        # XGBoost signal has prob=0.72, confirming context
                        # Should produce LONG with full position
                        assert signal.final_direction == "LONG"
                        assert signal.persona == "XGBOOST"
                        assert signal.market_regime == "confirming"

    @pytest.mark.asyncio
    async def test_pipeline_with_conflicting_context(self, integration_mocks):
        """Pipeline with conflicting context produces half position."""
        conflicting_context = LLMContext(
            bullish_factors=["Some bullish"],
            bearish_factors=["Many bearish factors"],
            regime_flag="conflicting",
            confidence=0.6,
        )

        with patch("signals.signal_loop.MarketDataService") as MockMDS:
            mock_mds = AsyncMock()
            mock_mds.fetch_ohlcv = AsyncMock(return_value=integration_mocks["ohlcv"])
            MockMDS.return_value.__aenter__ = AsyncMock(return_value=mock_mds)
            MockMDS.return_value.__aexit__ = AsyncMock()

            with patch(
                "signals.signal_loop.generate_xgboost_signal",
                return_value=integration_mocks["xgb_signal"],
            ):
                with patch(
                    "signals.signal_loop.generate_market_context", return_value=conflicting_context
                ):
                    with patch("signals.signal_loop.call_risk_filter", return_value=False):
                        signals = await run_cycle(
                            symbols=["BTC/USDT"],
                            timeframe="1h",
                            execute=False,
                            min_confidence=0.6,
                        )

                        # With conflicting context, synthesis should reduce position
                        # The signal logging uses legacy format, but reasoning captures synthesis
                        assert len(signals) == 1
                        assert (
                            "conflict" in signals[0].reasoning.lower()
                            or "half" in signals[0].reasoning.lower()
                        )

    @pytest.mark.asyncio
    async def test_pipeline_logs_to_jsonl(self, integration_mocks):
        """Pipeline logs signals to JSONL file."""
        with patch("signals.signal_loop.MarketDataService") as MockMDS:
            mock_mds = AsyncMock()
            mock_mds.fetch_ohlcv = AsyncMock(return_value=integration_mocks["ohlcv"])
            MockMDS.return_value.__aenter__ = AsyncMock(return_value=mock_mds)
            MockMDS.return_value.__aexit__ = AsyncMock()

            with patch(
                "signals.signal_loop.generate_xgboost_signal",
                return_value=integration_mocks["xgb_signal"],
            ):
                with patch(
                    "signals.signal_loop.generate_market_context",
                    return_value=integration_mocks["llm_context"],
                ):
                    with patch("signals.signal_loop.call_risk_filter", return_value=False):
                        await run_cycle(
                            symbols=["BTC/USDT"],
                            timeframe="1h",
                            execute=False,
                            min_confidence=0.6,
                        )

                        # Check log file exists and has content
                        log_path = integration_mocks["log_path"]
                        assert log_path.exists()
                        content = log_path.read_text()
                        assert "BTC/USDT" in content
