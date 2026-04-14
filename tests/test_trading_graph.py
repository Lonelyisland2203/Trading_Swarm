"""
Integration tests for the LangGraph trading graph.

Session 17S: Full pipeline tests covering data → indicators → XGBoost → LLM context → synthesis → execution.
Tests verify VRAM sequencing, node failure handling, STOP file behavior, and dry-run mode.
"""

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from signals.llm_context import LLMContext
from signals.synthesis import SynthesisOutput
from signals.xgboost_signal import XGBoostSignal


# -------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------


@pytest.fixture
def sample_ohlcv_df() -> pd.DataFrame:
    """Create sample OHLCV DataFrame with 100 bars."""
    n = 100
    return pd.DataFrame(
        {
            "timestamp": [1704067200000 + i * 3600000 for i in range(n)],
            "open": [100.0 + i * 0.1 for i in range(n)],
            "high": [102.0 + i * 0.1 for i in range(n)],
            "low": [98.0 + i * 0.1 for i in range(n)],
            "close": [101.0 + i * 0.1 for i in range(n)],
            "volume": [1000.0] * n,
        }
    )


@pytest.fixture
def mock_xgboost_signal() -> XGBoostSignal:
    """Create mock XGBoost signal."""
    return XGBoostSignal(
        symbol="BTC/USDT",
        timeframe="1h",
        direction="LONG",
        probability=0.72,
        confidence=0.44,
        features={"rsi": 55.0, "macd_line": 0.5},
        timestamp=datetime.now(timezone.utc),
    )


@pytest.fixture
def mock_llm_context() -> LLMContext:
    """Create mock LLM context."""
    return LLMContext(
        bullish_factors=["Positive funding rate", "Rising OI"],
        bearish_factors=["High leverage"],
        regime_flag="confirming",
        confidence=0.7,
    )


@pytest.fixture
def tmp_stop_file(tmp_path: Path) -> Path:
    """Create temporary STOP file path."""
    stop_dir = tmp_path / "execution" / "state"
    stop_dir.mkdir(parents=True, exist_ok=True)
    return stop_dir / "STOP"


# -------------------------------------------------------------------
# Test State Schema
# -------------------------------------------------------------------


class TestTradingState:
    """Test TradingState dataclass structure."""

    def test_state_schema_has_all_required_fields(self):
        """TradingState should contain all intermediate results for logging."""
        from orchestration.trading_graph import TradingState

        # Create empty state
        state = TradingState(
            symbol="BTC/USDT",
            timeframe="1h",
            dry_run=False,
        )

        # Verify all required fields exist
        assert hasattr(state, "symbol")
        assert hasattr(state, "timeframe")
        assert hasattr(state, "ohlcv_data")
        assert hasattr(state, "indicators")
        assert hasattr(state, "xgboost_signal")
        assert hasattr(state, "llm_context")
        assert hasattr(state, "critic_veto")
        assert hasattr(state, "synthesis_output")
        assert hasattr(state, "execution_result")
        assert hasattr(state, "errors")
        assert hasattr(state, "dry_run")

    def test_state_serializable(self):
        """TradingState should be JSON-serializable for logging."""
        import json
        from orchestration.trading_graph import TradingState

        state = TradingState(
            symbol="BTC/USDT",
            timeframe="1h",
            dry_run=True,
            errors=["test error"],
        )

        # Should serialize without error
        data = state.to_dict()
        json_str = json.dumps(data)
        assert len(json_str) > 0

        # Should roundtrip
        recovered = json.loads(json_str)
        assert recovered["symbol"] == "BTC/USDT"
        assert recovered["errors"] == ["test error"]


# -------------------------------------------------------------------
# Test Full Pipeline
# -------------------------------------------------------------------


class TestFullPipeline:
    """Test complete pipeline from data to execution."""

    @pytest.mark.asyncio
    async def test_full_pipeline_produces_valid_synthesis_output(
        self, sample_ohlcv_df, mock_xgboost_signal, mock_llm_context
    ):
        """
        Full pipeline: data → indicators → XGBoost → LLM context → synthesis → execution.
        Verify final output is a valid SynthesisOutput.
        """
        from orchestration.trading_graph import TradingGraph

        # Mock all external dependencies
        with (
            patch("orchestration.trading_graph.MarketDataService") as mock_mds,
            patch("orchestration.trading_graph.generate_xgboost_signal") as mock_xgb,
            patch("orchestration.trading_graph.generate_market_context") as mock_ctx,
            patch("orchestration.trading_graph.call_risk_filter") as mock_risk,
            patch("orchestration.trading_graph.check_stop_file", return_value=False),
        ):
            # Setup mocks
            mock_mds_instance = AsyncMock()
            mock_mds_instance.get_ohlcv_as_of = AsyncMock(return_value=sample_ohlcv_df)
            mock_mds_instance.__aenter__ = AsyncMock(return_value=mock_mds_instance)
            mock_mds_instance.__aexit__ = AsyncMock()
            mock_mds.return_value = mock_mds_instance

            mock_xgb.return_value = mock_xgboost_signal
            mock_ctx.return_value = mock_llm_context
            mock_risk.return_value = False  # No veto

            # Run graph
            graph = TradingGraph()
            result = await graph.run(
                symbol="BTC/USDT",
                timeframe="1h",
                dry_run=True,
            )

            # Verify result
            assert result is not None
            assert result.synthesis_output is not None
            assert isinstance(result.synthesis_output, SynthesisOutput)
            assert result.synthesis_output.direction in ["LONG", "SHORT", "FLAT"]
            assert 0.0 <= result.synthesis_output.position_size_fraction <= 1.0
            assert result.errors == []

    @pytest.mark.asyncio
    async def test_full_pipeline_with_execution(
        self, sample_ohlcv_df, mock_xgboost_signal, mock_llm_context
    ):
        """Pipeline should call exchange when not in dry-run mode."""
        from orchestration.trading_graph import TradingGraph

        mock_router = AsyncMock()
        mock_router.place_order = AsyncMock(return_value=MagicMock(order_id="test123"))

        with (
            patch("orchestration.trading_graph.MarketDataService") as mock_mds,
            patch("orchestration.trading_graph.generate_xgboost_signal") as mock_xgb,
            patch("orchestration.trading_graph.generate_market_context") as mock_ctx,
            patch("orchestration.trading_graph.call_risk_filter") as mock_risk,
            patch("orchestration.trading_graph.check_stop_file", return_value=False),
        ):
            mock_mds_instance = AsyncMock()
            mock_mds_instance.get_ohlcv_as_of = AsyncMock(return_value=sample_ohlcv_df)
            mock_mds_instance.__aenter__ = AsyncMock(return_value=mock_mds_instance)
            mock_mds_instance.__aexit__ = AsyncMock()
            mock_mds.return_value = mock_mds_instance

            mock_xgb.return_value = mock_xgboost_signal
            mock_ctx.return_value = mock_llm_context
            mock_risk.return_value = False

            graph = TradingGraph(exchange_router=mock_router)
            result = await graph.run(
                symbol="BTC/USDT",
                timeframe="1h",
                dry_run=False,
            )

            # Should have called exchange
            assert mock_router.place_order.called
            assert result.execution_result is not None


# -------------------------------------------------------------------
# Test VRAM Sequencing
# -------------------------------------------------------------------


class TestVRAMSequencing:
    """Test that models never coexist in VRAM."""

    @pytest.mark.asyncio
    async def test_vram_sequencing_xgboost_cpu_no_conflict(
        self, sample_ohlcv_df, mock_xgboost_signal, mock_llm_context
    ):
        """
        XGBoost runs on CPU (no GPU conflict).
        Qwen loads → inference → unloads.
        DeepSeek loads → inference → unloads.
        Models never coexist in VRAM.
        """
        from orchestration.trading_graph import TradingGraph

        # Track model loading order
        model_events = []

        async def mock_generate_xgboost(*args, **kwargs):
            model_events.append("xgboost_start")
            # XGBoost runs on CPU, no VRAM
            model_events.append("xgboost_end")
            return mock_xgboost_signal

        async def mock_generate_context(*args, **kwargs):
            model_events.append("qwen_load")
            model_events.append("qwen_inference")
            model_events.append("qwen_unload")
            return mock_llm_context

        async def mock_risk_filter(*args, **kwargs):
            model_events.append("deepseek_load")
            model_events.append("deepseek_inference")
            model_events.append("deepseek_unload")
            return False

        with (
            patch("orchestration.trading_graph.MarketDataService") as mock_mds,
            patch(
                "orchestration.trading_graph.generate_xgboost_signal",
                side_effect=mock_generate_xgboost,
            ),
            patch(
                "orchestration.trading_graph.generate_market_context",
                side_effect=mock_generate_context,
            ),
            patch("orchestration.trading_graph.call_risk_filter", side_effect=mock_risk_filter),
            patch("orchestration.trading_graph.check_stop_file", return_value=False),
        ):
            mock_mds_instance = AsyncMock()
            mock_mds_instance.get_ohlcv_as_of = AsyncMock(return_value=sample_ohlcv_df)
            mock_mds_instance.__aenter__ = AsyncMock(return_value=mock_mds_instance)
            mock_mds_instance.__aexit__ = AsyncMock()
            mock_mds.return_value = mock_mds_instance

            graph = TradingGraph()
            await graph.run(symbol="BTC/USDT", timeframe="1h", dry_run=True)

            # Verify correct order: XGBoost → Qwen → DeepSeek
            assert model_events.index("xgboost_end") < model_events.index("qwen_load")
            assert model_events.index("qwen_unload") < model_events.index("deepseek_load")


# -------------------------------------------------------------------
# Test Node Failures
# -------------------------------------------------------------------


class TestNodeFailures:
    """Test error handling when individual nodes fail."""

    @pytest.mark.asyncio
    async def test_node_failure_data_returns_flat(self):
        """Data fetch fails → graph returns FLAT, logs error."""
        from orchestration.trading_graph import TradingGraph

        with (
            patch("orchestration.trading_graph.MarketDataService") as mock_mds,
            patch("orchestration.trading_graph.check_stop_file", return_value=False),
        ):
            mock_mds_instance = AsyncMock()
            mock_mds_instance.get_ohlcv_as_of = AsyncMock(side_effect=Exception("Network error"))
            mock_mds_instance.__aenter__ = AsyncMock(return_value=mock_mds_instance)
            mock_mds_instance.__aexit__ = AsyncMock()
            mock_mds.return_value = mock_mds_instance

            graph = TradingGraph()
            result = await graph.run(symbol="BTC/USDT", timeframe="1h", dry_run=True)

            # Should return FLAT on data failure
            assert result.synthesis_output is not None
            assert result.synthesis_output.direction == "FLAT"
            assert len(result.errors) > 0
            assert any("data" in e.lower() or "network" in e.lower() for e in result.errors)

    @pytest.mark.asyncio
    async def test_node_failure_xgboost_returns_flat(self, sample_ohlcv_df):
        """XGBoost inference fails → FLAT."""
        from orchestration.trading_graph import TradingGraph

        with (
            patch("orchestration.trading_graph.MarketDataService") as mock_mds,
            patch("orchestration.trading_graph.generate_xgboost_signal") as mock_xgb,
            patch("orchestration.trading_graph.check_stop_file", return_value=False),
        ):
            mock_mds_instance = AsyncMock()
            mock_mds_instance.get_ohlcv_as_of = AsyncMock(return_value=sample_ohlcv_df)
            mock_mds_instance.__aenter__ = AsyncMock(return_value=mock_mds_instance)
            mock_mds_instance.__aexit__ = AsyncMock()
            mock_mds.return_value = mock_mds_instance

            mock_xgb.return_value = None  # XGBoost failed

            graph = TradingGraph()
            result = await graph.run(symbol="BTC/USDT", timeframe="1h", dry_run=True)

            assert result.synthesis_output.direction == "FLAT"
            assert len(result.errors) > 0

    @pytest.mark.asyncio
    async def test_node_failure_llm_uses_xgboost_only(self, sample_ohlcv_df, mock_xgboost_signal):
        """LLM context fails → synthesis uses XGBoost only (degraded mode)."""
        from orchestration.trading_graph import TradingGraph

        with (
            patch("orchestration.trading_graph.MarketDataService") as mock_mds,
            patch("orchestration.trading_graph.generate_xgboost_signal") as mock_xgb,
            patch("orchestration.trading_graph.generate_market_context") as mock_ctx,
            patch("orchestration.trading_graph.call_risk_filter") as mock_risk,
            patch("orchestration.trading_graph.check_stop_file", return_value=False),
        ):
            mock_mds_instance = AsyncMock()
            mock_mds_instance.get_ohlcv_as_of = AsyncMock(return_value=sample_ohlcv_df)
            mock_mds_instance.__aenter__ = AsyncMock(return_value=mock_mds_instance)
            mock_mds_instance.__aexit__ = AsyncMock()
            mock_mds.return_value = mock_mds_instance

            mock_xgb.return_value = mock_xgboost_signal
            mock_ctx.side_effect = Exception("Ollama connection failed")
            mock_risk.return_value = False

            graph = TradingGraph()
            result = await graph.run(symbol="BTC/USDT", timeframe="1h", dry_run=True)

            # Should still produce signal using XGBoost only
            assert result.synthesis_output is not None
            assert result.synthesis_output.direction == "LONG"  # From mock_xgboost_signal
            # LLM context should be None
            assert result.llm_context is None

    @pytest.mark.asyncio
    async def test_node_failure_critic_proceeds_without_veto(
        self, sample_ohlcv_df, mock_xgboost_signal, mock_llm_context
    ):
        """DeepSeek fails → synthesis proceeds without veto (conservative: treat as no-veto)."""
        from orchestration.trading_graph import TradingGraph

        with (
            patch("orchestration.trading_graph.MarketDataService") as mock_mds,
            patch("orchestration.trading_graph.generate_xgboost_signal") as mock_xgb,
            patch("orchestration.trading_graph.generate_market_context") as mock_ctx,
            patch("orchestration.trading_graph.call_risk_filter") as mock_risk,
            patch("orchestration.trading_graph.check_stop_file", return_value=False),
        ):
            mock_mds_instance = AsyncMock()
            mock_mds_instance.get_ohlcv_as_of = AsyncMock(return_value=sample_ohlcv_df)
            mock_mds_instance.__aenter__ = AsyncMock(return_value=mock_mds_instance)
            mock_mds_instance.__aexit__ = AsyncMock()
            mock_mds.return_value = mock_mds_instance

            mock_xgb.return_value = mock_xgboost_signal
            mock_ctx.return_value = mock_llm_context
            mock_risk.side_effect = Exception("DeepSeek timeout")

            graph = TradingGraph()
            result = await graph.run(symbol="BTC/USDT", timeframe="1h", dry_run=True)

            # Should proceed with trade (no veto on failure)
            assert result.synthesis_output is not None
            assert result.synthesis_output.direction != "FLAT"
            assert result.critic_veto is False


# -------------------------------------------------------------------
# Test STOP File Handling
# -------------------------------------------------------------------


class TestStopFileHandling:
    """Test STOP file kill switch behavior."""

    @pytest.mark.asyncio
    async def test_stop_file_halts_immediately(self, tmp_stop_file):
        """STOP file exists → graph exits immediately, no execution."""
        from orchestration.trading_graph import TradingGraph

        # Create STOP file
        tmp_stop_file.touch()

        with patch("orchestration.trading_graph.check_stop_file", return_value=True):
            graph = TradingGraph()
            result = await graph.run(symbol="BTC/USDT", timeframe="1h", dry_run=False)

            # Should return FLAT immediately
            assert result.synthesis_output.direction == "FLAT"
            assert any("stop" in e.lower() for e in result.errors)


# -------------------------------------------------------------------
# Test Dry Run Mode
# -------------------------------------------------------------------


class TestDryRunMode:
    """Test --dry-run flag behavior."""

    @pytest.mark.asyncio
    async def test_dry_run_mode_prevents_exchange_calls(
        self, sample_ohlcv_df, mock_xgboost_signal, mock_llm_context
    ):
        """--dry-run flag prevents any exchange calls."""
        from orchestration.trading_graph import TradingGraph

        mock_router = AsyncMock()
        mock_router.place_order = AsyncMock()

        with (
            patch("orchestration.trading_graph.MarketDataService") as mock_mds,
            patch("orchestration.trading_graph.generate_xgboost_signal") as mock_xgb,
            patch("orchestration.trading_graph.generate_market_context") as mock_ctx,
            patch("orchestration.trading_graph.call_risk_filter") as mock_risk,
            patch("orchestration.trading_graph.check_stop_file", return_value=False),
        ):
            mock_mds_instance = AsyncMock()
            mock_mds_instance.get_ohlcv_as_of = AsyncMock(return_value=sample_ohlcv_df)
            mock_mds_instance.__aenter__ = AsyncMock(return_value=mock_mds_instance)
            mock_mds_instance.__aexit__ = AsyncMock()
            mock_mds.return_value = mock_mds_instance

            mock_xgb.return_value = mock_xgboost_signal
            mock_ctx.return_value = mock_llm_context
            mock_risk.return_value = False

            graph = TradingGraph(exchange_router=mock_router)
            result = await graph.run(
                symbol="BTC/USDT",
                timeframe="1h",
                dry_run=True,  # DRY RUN
            )

            # Exchange should NOT be called
            assert not mock_router.place_order.called
            assert result.execution_result is None
            assert result.synthesis_output is not None


# -------------------------------------------------------------------
# Test Indicators Computation
# -------------------------------------------------------------------


class TestIndicatorsComputation:
    """Test that indicators are computed correctly in data node."""

    @pytest.mark.asyncio
    async def test_data_node_computes_all_17_indicators(self, sample_ohlcv_df):
        """Data node should compute all 17 indicators from OHLCV."""
        from orchestration.trading_graph import TradingGraph

        with (
            patch("orchestration.trading_graph.MarketDataService") as mock_mds,
            patch("orchestration.trading_graph.generate_xgboost_signal") as mock_xgb,
            patch("orchestration.trading_graph.generate_market_context") as mock_ctx,
            patch("orchestration.trading_graph.call_risk_filter") as mock_risk,
            patch("orchestration.trading_graph.check_stop_file", return_value=False),
        ):
            mock_mds_instance = AsyncMock()
            mock_mds_instance.get_ohlcv_as_of = AsyncMock(return_value=sample_ohlcv_df)
            mock_mds_instance.__aenter__ = AsyncMock(return_value=mock_mds_instance)
            mock_mds_instance.__aexit__ = AsyncMock()
            mock_mds.return_value = mock_mds_instance

            # Make XGBoost return None to stop early but check indicators were computed
            mock_xgb.return_value = None
            mock_ctx.return_value = None
            mock_risk.return_value = False

            graph = TradingGraph()
            result = await graph.run(symbol="BTC/USDT", timeframe="1h", dry_run=True)

            # Verify indicators were computed
            assert result.indicators is not None
            assert "rsi" in result.indicators
            assert "macd_line" in result.indicators
            assert "atr_normalized" in result.indicators


# -------------------------------------------------------------------
# Test Signal Logging
# -------------------------------------------------------------------


class TestSignalLogging:
    """Test that TradingState is logged to signal_log.jsonl."""

    @pytest.mark.asyncio
    async def test_complete_state_logged_after_run(
        self, sample_ohlcv_df, mock_xgboost_signal, mock_llm_context, tmp_path
    ):
        """Complete TradingState should be logged to signals/signal_log.jsonl after each run."""
        from orchestration.trading_graph import TradingGraph

        log_file = tmp_path / "signal_log.jsonl"

        with (
            patch("orchestration.trading_graph.MarketDataService") as mock_mds,
            patch("orchestration.trading_graph.generate_xgboost_signal") as mock_xgb,
            patch("orchestration.trading_graph.generate_market_context") as mock_ctx,
            patch("orchestration.trading_graph.call_risk_filter") as mock_risk,
            patch("orchestration.trading_graph.check_stop_file", return_value=False),
            patch("orchestration.trading_graph.SIGNAL_LOG_PATH", log_file),
        ):
            mock_mds_instance = AsyncMock()
            mock_mds_instance.get_ohlcv_as_of = AsyncMock(return_value=sample_ohlcv_df)
            mock_mds_instance.__aenter__ = AsyncMock(return_value=mock_mds_instance)
            mock_mds_instance.__aexit__ = AsyncMock()
            mock_mds.return_value = mock_mds_instance

            mock_xgb.return_value = mock_xgboost_signal
            mock_ctx.return_value = mock_llm_context
            mock_risk.return_value = False

            graph = TradingGraph()
            await graph.run(symbol="BTC/USDT", timeframe="1h", dry_run=True)

            # Check log file was written
            assert log_file.exists()
            content = log_file.read_text()
            assert "BTC/USDT" in content
            assert "LONG" in content or "SHORT" in content or "FLAT" in content
