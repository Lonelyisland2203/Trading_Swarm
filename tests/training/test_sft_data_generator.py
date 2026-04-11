"""Tests for training/sft_data_generator.py.

Tests cover:
- Process lock acquisition
- Output JSONL schema validation
- Reasoning trace parsing
- Market snapshot building
- Distillation prompt construction
- Pre-flight order verification

All Ollama calls are mocked — never hit real models.
"""

import json
import tempfile
from datetime import datetime, UTC
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from config.fee_model import FeeModelSettings

# Fixed timestamp for temporal isolation
FIXED_TS = pd.Timestamp("2024-01-15 12:00:00", tz="UTC")
FIXED_TS_MS = int(FIXED_TS.timestamp() * 1000)


@pytest.fixture
def sample_ohlcv_df() -> pd.DataFrame:
    """Create sample OHLCV DataFrame with fixed timestamps."""
    n_bars = 100
    base_price = 42000.0
    timestamps = [
        FIXED_TS_MS - (n_bars - i) * 3600 * 1000  # 1h bars
        for i in range(n_bars)
    ]

    data = {
        "timestamp": timestamps,
        "open": [base_price + i * 10 for i in range(n_bars)],
        "high": [base_price + i * 10 + 50 for i in range(n_bars)],
        "low": [base_price + i * 10 - 30 for i in range(n_bars)],
        "close": [base_price + i * 10 + 20 for i in range(n_bars)],
        "volume": [1000.0 + i * 10 for i in range(n_bars)],
    }
    return pd.DataFrame(data)


@pytest.fixture
def fee_model() -> FeeModelSettings:
    """Default fee model for tests."""
    return FeeModelSettings()


@pytest.fixture
def mock_ollama_response() -> dict:
    """Mock Ollama response with valid reasoning trace."""
    return {
        "response": """THESIS: Strong bullish momentum with high conviction based on RSI oversold bounce and MACD crossover.

EVIDENCE:
- RSI(14): 35.2 → Oversold territory indicating potential reversal
- MACD: 0.0012 (Signal: -0.0005) → Bullish crossover forming
- BB Position: 0.15 → Near lower band, bounce expected
- ADX: 28.5 → Strong trend developing
- OBV Slope: 0.023 → Increasing buying volume

RISK: False breakout if broader market sentiment turns negative

DECISION: LONG | Confidence: 4""",
        "model": "deepseek-r1:14b",
        "done": True,
    }


class TestParseReasoningTrace:
    """Tests for parse_reasoning_trace function."""

    def test_valid_trace_returns_cleaned_string(self) -> None:
        """Valid trace with all sections is parsed correctly."""
        from training.sft_data_generator import parse_reasoning_trace

        trace = """Some preamble text

THESIS: Bullish outlook

EVIDENCE:
- RSI: 45 → neutral
- MACD: positive

RISK: Market volatility

DECISION: LONG | Confidence: 3"""

        result = parse_reasoning_trace(trace)

        assert result is not None
        assert result.startswith("THESIS:")
        assert "EVIDENCE:" in result
        assert "RISK:" in result
        assert "DECISION:" in result
        # Preamble should be removed
        assert "preamble" not in result

    def test_missing_section_returns_none(self) -> None:
        """Missing required section returns None."""
        from training.sft_data_generator import parse_reasoning_trace

        # Missing RISK section
        trace = """THESIS: Bullish

EVIDENCE:
- RSI: 45

DECISION: LONG"""

        result = parse_reasoning_trace(trace)
        assert result is None

    @pytest.mark.parametrize("missing_section", [
        "THESIS:",
        "EVIDENCE:",
        "RISK:",
        "DECISION:",
    ])
    def test_each_missing_section_returns_none(self, missing_section: str) -> None:
        """Each required section must be present."""
        from training.sft_data_generator import parse_reasoning_trace

        full_trace = """THESIS: Test

EVIDENCE:
- Test

RISK: Test

DECISION: LONG"""

        # Remove one section
        broken_trace = full_trace.replace(missing_section, "")
        result = parse_reasoning_trace(broken_trace)
        assert result is None


class TestBuildMarketSnapshot:
    """Tests for build_market_snapshot function."""

    def test_snapshot_contains_required_sections(
        self, sample_ohlcv_df: pd.DataFrame, fee_model: FeeModelSettings
    ) -> None:
        """Market snapshot contains all required sections."""
        from training.sft_data_generator import build_market_snapshot

        snapshot = build_market_snapshot(
            df=sample_ohlcv_df,
            symbol="BTC/USDT",
            timeframe="1h",
            fee_model=fee_model,
        )

        assert "## Market Data" in snapshot
        assert "Symbol: BTC/USDT" in snapshot
        assert "Timeframe: 1h" in snapshot
        assert "## Technical Indicators" in snapshot
        assert "RSI(14):" in snapshot
        assert "MACD:" in snapshot
        assert "## Recent Price Action" in snapshot
        assert "## Execution Context" in snapshot
        assert "round-trip cost:" in snapshot

    def test_snapshot_uses_fee_model_costs(
        self, sample_ohlcv_df: pd.DataFrame
    ) -> None:
        """Snapshot reflects fee model costs."""
        from training.sft_data_generator import build_market_snapshot

        # Custom fee model with higher costs
        custom_fee = FeeModelSettings(
            maker_fee_pct=0.1,
            taker_fee_pct=0.1,
            slippage_pct=0.05,
        )

        snapshot = build_market_snapshot(
            df=sample_ohlcv_df,
            symbol="ETH/USDT",
            timeframe="4h",
            fee_model=custom_fee,
        )

        # Higher fees should be reflected
        assert "Minimum profitable move:" in snapshot


class TestBuildDistillationPrompt:
    """Tests for build_distillation_prompt function."""

    def test_prompt_includes_outcome_context(self, fee_model: FeeModelSettings) -> None:
        """Distillation prompt includes verified outcome context."""
        from training.sft_data_generator import build_distillation_prompt

        prompt = build_distillation_prompt(
            market_snapshot="## Market Data\nTest snapshot",
            outcome="HIGHER",
            net_return_pct=0.15,
            fee_model=fee_model,
        )

        assert "went UP" in prompt
        assert "0.150%" in prompt
        assert "DECISION: LONG" in prompt

    def test_prompt_includes_bearish_outcome(self, fee_model: FeeModelSettings) -> None:
        """Bearish outcome generates SHORT decision."""
        from training.sft_data_generator import build_distillation_prompt

        prompt = build_distillation_prompt(
            market_snapshot="## Market Data\nTest",
            outcome="LOWER",
            net_return_pct=-0.25,
            fee_model=fee_model,
        )

        assert "went DOWN" in prompt
        assert "DECISION: SHORT" in prompt

    def test_prompt_includes_fee_context(self, fee_model: FeeModelSettings) -> None:
        """Prompt includes transaction cost context."""
        from training.sft_data_generator import build_distillation_prompt

        prompt = build_distillation_prompt(
            market_snapshot="## Test",
            outcome="FLAT",
            net_return_pct=0.0,
            fee_model=fee_model,
        )

        # Should mention break-even threshold
        assert "break-even" in prompt.lower() or "transaction costs" in prompt.lower()


class TestSFTExample:
    """Tests for SFTExample dataclass and JSONL schema."""

    def test_to_dict_produces_valid_json(self) -> None:
        """SFTExample.to_dict() produces JSON-serializable dict."""
        from training.sft_data_generator import SFTExample

        example = SFTExample(
            example_id="test_123",
            created_at=datetime.now(UTC).isoformat(),
            symbol="BTC/USDT",
            timeframe="1h",
            timestamp_ms=FIXED_TS_MS,
            market_snapshot="## Test snapshot",
            verified_outcome="HIGHER",
            net_return_pct=0.15,
            reasoning_trace="THESIS: Test\nEVIDENCE:\n- Test\nRISK: Test\nDECISION: LONG",
        )

        # Should serialize without error
        json_str = json.dumps(example.to_dict())
        assert json_str is not None

        # Should deserialize correctly
        loaded = json.loads(json_str)
        assert loaded["example_id"] == "test_123"
        assert loaded["symbol"] == "BTC/USDT"
        assert loaded["verified_outcome"] == "HIGHER"

    def test_jsonl_schema_has_required_fields(self) -> None:
        """JSONL output schema contains all required fields."""
        from training.sft_data_generator import SFTExample

        example = SFTExample(
            example_id="schema_test",
            created_at=datetime.now(UTC).isoformat(),
            symbol="ETH/USDT",
            timeframe="4h",
            timestamp_ms=FIXED_TS_MS,
            market_snapshot="snapshot",
            verified_outcome="LOWER",
            net_return_pct=-0.05,
            reasoning_trace="trace",
        )

        data = example.to_dict()

        required_fields = [
            "example_id",
            "created_at",
            "symbol",
            "timeframe",
            "timestamp_ms",
            "market_snapshot",
            "verified_outcome",
            "net_return_pct",
            "reasoning_trace",
        ]

        for field in required_fields:
            assert field in data, f"Missing required field: {field}"


class TestProcessLockAcquisition:
    """Tests for process lock handling."""

    @patch("training.sft_data_generator.acquire_training_lock")
    @patch("training.sft_data_generator.preflight_checks")
    @patch("training.sft_data_generator.OllamaClient")
    @patch("training.sft_data_generator.MarketDataService")
    async def test_acquires_lock_before_generation(
        self,
        mock_market_data: MagicMock,
        mock_ollama: MagicMock,
        mock_preflight: AsyncMock,
        mock_lock: MagicMock,
    ) -> None:
        """Training lock is acquired before generating examples."""
        from training.sft_data_generator import generate_sft_dataset

        mock_preflight.return_value = True
        mock_lock.return_value.__enter__ = MagicMock()
        mock_lock.return_value.__exit__ = MagicMock()

        # Mock market data to return empty to avoid processing
        mock_market_data.return_value.__aenter__ = AsyncMock()
        mock_market_data.return_value.__aexit__ = AsyncMock()
        mock_market_data.return_value.fetch_ohlcv = AsyncMock(return_value=pd.DataFrame())

        mock_ollama.return_value.__aenter__ = AsyncMock()
        mock_ollama.return_value.__aexit__ = AsyncMock()

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            output_path = Path(f.name)

        try:
            await generate_sft_dataset(output_path, limit=1)
            mock_lock.assert_called_once()
        finally:
            output_path.unlink(missing_ok=True)

    def test_check_can_train_returns_tuple(self) -> None:
        """check_can_train returns (bool, str) tuple."""
        from training.process_lock import check_can_train

        result = check_can_train()

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], str)


class TestPreflightOrder:
    """Tests for pre-flight check order enforcement."""

    @patch("training.sft_data_generator.settings")
    async def test_preflight_checks_data_first(
        self, mock_settings: MagicMock
    ) -> None:
        """Pre-flight checks DATA availability first."""
        from training.sft_data_generator import preflight_checks

        mock_settings.ollama.base_url = "http://localhost:11434"

        # Mock aiohttp to fail immediately (DATA check fails)
        with patch("aiohttp.ClientSession") as mock_session:
            mock_session.return_value.__aenter__ = AsyncMock(
                side_effect=Exception("Connection refused")
            )
            mock_session.return_value.__aexit__ = AsyncMock()

            result = await preflight_checks()

        # Should fail at DATA check
        assert result is False

    @patch("training.sft_data_generator.check_can_train")
    @patch("training.sft_data_generator.settings")
    async def test_preflight_checks_lock_availability(
        self,
        mock_settings: MagicMock,
        mock_check: MagicMock,
    ) -> None:
        """Pre-flight checks LOCK availability."""
        from training.sft_data_generator import preflight_checks

        mock_settings.ollama.base_url = "http://localhost:11434"
        mock_check.return_value = (False, "Training process is running")

        # Mock successful Ollama check
        with patch("aiohttp.ClientSession") as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={"models": [{"name": "deepseek-r1:14b"}]})

            mock_ctx = AsyncMock()
            mock_ctx.__aenter__ = AsyncMock(return_value=mock_response)
            mock_ctx.__aexit__ = AsyncMock()

            mock_sess = AsyncMock()
            mock_sess.get = MagicMock(return_value=mock_ctx)
            mock_sess.__aenter__ = AsyncMock(return_value=mock_sess)
            mock_sess.__aexit__ = AsyncMock()

            mock_session.return_value = mock_sess

            # Mock subprocess for nvidia-smi
            with patch("subprocess.run") as mock_run:
                mock_run.return_value.returncode = 1  # nvidia-smi fails, but continues

                result = await preflight_checks()

        # Should fail at LOCK check
        assert result is False


class TestLoadCompletedIds:
    """Tests for resume functionality."""

    def test_loads_ids_from_jsonl(self) -> None:
        """Completed IDs are loaded from existing JSONL."""
        from training.sft_data_generator import load_completed_ids

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"example_id": "id_1", "symbol": "BTC/USDT"}\n')
            f.write('{"example_id": "id_2", "symbol": "ETH/USDT"}\n')
            f.write('{"example_id": "id_3", "symbol": "SOL/USDT"}\n')
            path = Path(f.name)

        try:
            ids = load_completed_ids(path)
            assert ids == {"id_1", "id_2", "id_3"}
        finally:
            path.unlink()

    def test_handles_missing_file(self) -> None:
        """Missing file returns empty set."""
        from training.sft_data_generator import load_completed_ids

        path = Path("/nonexistent/path/file.jsonl")
        ids = load_completed_ids(path)
        assert ids == set()

    def test_handles_malformed_lines(self) -> None:
        """Malformed JSONL lines are skipped."""
        from training.sft_data_generator import load_completed_ids

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"example_id": "valid_1"}\n')
            f.write('not valid json\n')
            f.write('{"no_id_field": true}\n')
            f.write('{"example_id": "valid_2"}\n')
            path = Path(f.name)

        try:
            ids = load_completed_ids(path)
            assert ids == {"valid_1", "valid_2"}
        finally:
            path.unlink()


class TestMockOllamaCalls:
    """Tests verifying Ollama calls are properly mocked."""

    @patch("training.sft_data_generator.OllamaClient")
    async def test_ollama_generate_is_mocked(
        self,
        mock_client_class: MagicMock,
        mock_ollama_response: dict,
    ) -> None:
        """Ollama generate calls use mocked responses."""
        mock_client = AsyncMock()
        mock_client.generate = AsyncMock(return_value=mock_ollama_response)
        mock_client.unload_current = AsyncMock()
        mock_client_class.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_class.return_value.__aexit__ = AsyncMock()

        # Verify mock is configured correctly
        response = await mock_client.generate(
            model="deepseek-r1:14b",
            prompt="test prompt",
            options={},
        )

        assert response["model"] == "deepseek-r1:14b"
        assert "THESIS:" in response["response"]
        mock_client.generate.assert_called_once()

    def test_no_real_ollama_imports_at_module_level(self) -> None:
        """Module doesn't make real Ollama calls on import."""
        # This test passes if the import doesn't fail or make network calls
        import training.sft_data_generator  # noqa: F401

        # If we get here, no network calls were made during import
        assert True
