"""
Tests for LLM context overlay node.

TDD: Tests written FIRST before implementation.
Session 17M: Qwen produces context, NEVER direction.
"""

import os
import re
from unittest.mock import MagicMock, patch

import pytest


class TestLLMContextSchema:
    """Tests for LLMContext dataclass schema."""

    def test_output_schema_has_required_fields(self):
        """LLMContext dataclass has all required fields."""
        from signals.llm_context import LLMContext

        # Create instance with all required fields
        ctx = LLMContext(
            bullish_factors=["funding negative", "OI declining"],
            bearish_factors=["liquidations heavy"],
            regime_flag="confirming",
            confidence=0.75,
        )

        # Verify field types
        assert isinstance(ctx.bullish_factors, list)
        assert isinstance(ctx.bearish_factors, list)
        assert isinstance(ctx.regime_flag, str)
        assert isinstance(ctx.confidence, float)

    def test_output_schema_field_values(self):
        """LLMContext fields have correct values."""
        from signals.llm_context import LLMContext

        bullish = ["short squeeze potential", "funding turning negative"]
        bearish = ["high leverage", "whale exits"]

        ctx = LLMContext(
            bullish_factors=bullish,
            bearish_factors=bearish,
            regime_flag="conflicting",
            confidence=0.5,
        )

        assert ctx.bullish_factors == bullish
        assert ctx.bearish_factors == bearish
        assert ctx.regime_flag == "conflicting"
        assert ctx.confidence == 0.5

    def test_regime_flag_valid_values(self):
        """Regime flag must be one of confirming/conflicting/neutral."""
        from signals.llm_context import VALID_REGIME_FLAGS

        assert "confirming" in VALID_REGIME_FLAGS
        assert "conflicting" in VALID_REGIME_FLAGS
        assert "neutral" in VALID_REGIME_FLAGS
        assert len(VALID_REGIME_FLAGS) == 3

    def test_confidence_range(self):
        """Confidence must be float 0-1."""
        from signals.llm_context import LLMContext

        # Valid confidence
        ctx = LLMContext(
            bullish_factors=[],
            bearish_factors=[],
            regime_flag="neutral",
            confidence=0.5,
        )
        assert 0.0 <= ctx.confidence <= 1.0

    def test_llm_context_to_dict(self):
        """LLMContext can be serialized to dict."""
        from signals.llm_context import LLMContext

        ctx = LLMContext(
            bullish_factors=["factor1"],
            bearish_factors=["factor2"],
            regime_flag="neutral",
            confidence=0.6,
        )

        data = ctx.to_dict()

        assert data["bullish_factors"] == ["factor1"]
        assert data["bearish_factors"] == ["factor2"]
        assert data["regime_flag"] == "neutral"
        assert data["confidence"] == 0.6


class TestNoDirectionalOutput:
    """Tests ensuring LLM output NEVER contains direction words."""

    # Forbidden directional words (case-insensitive, as standalone words)
    FORBIDDEN_WORDS = ["LONG", "SHORT", "BUY", "SELL"]
    FORBIDDEN_PATTERN = re.compile(
        r"\b(" + "|".join(FORBIDDEN_WORDS) + r")\b",
        re.IGNORECASE,
    )

    def test_no_directional_output_in_dataclass(self):
        """LLMContext values never contain directional words."""
        from signals.llm_context import LLMContext

        # Valid context without direction
        ctx = LLMContext(
            bullish_factors=["funding rate negative", "OI declining"],
            bearish_factors=["heavy liquidations on longs"],
            regime_flag="conflicting",
            confidence=0.6,
        )

        # Check all string fields
        all_text = " ".join(ctx.bullish_factors + ctx.bearish_factors) + ctx.regime_flag

        # Should NOT match forbidden pattern
        match = self.FORBIDDEN_PATTERN.search(all_text)
        assert match is None, f"Found forbidden word: {match.group() if match else ''}"

    @pytest.mark.asyncio
    async def test_no_directional_output_in_raw_llm_response(self):
        """Raw LLM response string never contains LONG/SHORT/BUY/SELL."""
        from signals.llm_context import generate_market_context

        # Mock Ollama to return a response
        mock_response = {
            "bullish_factors": ["negative funding suggests shorts paying longs"],
            "bearish_factors": ["declining OI may indicate position exits"],
            "regime_flag": "neutral",
            "confidence": 0.5,
        }

        with patch("signals.llm_context.run_preflight_checks") as mock_preflight:
            mock_preflight.return_value = MagicMock(passed=True)

            with patch("signals.llm_context._call_ollama") as mock_ollama:
                # Return valid JSON without directional words
                import json

                mock_ollama.return_value = json.dumps(mock_response)

                ctx = await generate_market_context(
                    funding_rate=-0.01,
                    oi_delta=-5.0,
                    liquidation_data=None,
                    news_headlines=None,
                )

        # Verify output has no directional words
        all_text = " ".join(ctx.bullish_factors + ctx.bearish_factors)
        match = self.FORBIDDEN_PATTERN.search(all_text)
        assert match is None, f"Found forbidden word in output: {match.group() if match else ''}"

    @pytest.mark.asyncio
    async def test_llm_response_with_forbidden_words_filtered(self):
        """LLM response containing directional words gets filtered/rejected."""
        from signals.llm_context import generate_market_context

        # Mock Ollama returning forbidden words (bad behavior we must handle)
        bad_response = {
            "bullish_factors": ["GO LONG here", "BUY signal detected"],
            "bearish_factors": ["SHORT opportunity"],
            "regime_flag": "confirming",
            "confidence": 0.8,
        }

        with patch("signals.llm_context.run_preflight_checks") as mock_preflight:
            mock_preflight.return_value = MagicMock(passed=True)

            with patch("signals.llm_context._call_ollama") as mock_ollama:
                import json

                mock_ollama.return_value = json.dumps(bad_response)

                ctx = await generate_market_context(
                    funding_rate=-0.01,
                    oi_delta=-5.0,
                    liquidation_data=None,
                    news_headlines=None,
                )

        # Should fallback to neutral (directional words = invalid output)
        assert ctx.regime_flag == "neutral"
        assert ctx.confidence == 0.0

    def test_system_prompt_forbids_direction(self):
        """System prompt explicitly forbids directional output."""
        from signals.llm_context import SYSTEM_PROMPT

        # System prompt must contain explicit prohibition
        assert "DO NOT predict price direction" in SYSTEM_PROMPT
        assert "DO NOT output LONG" in SYSTEM_PROMPT or "LONG" in SYSTEM_PROMPT
        assert "DO NOT output" in SYSTEM_PROMPT and "SHORT" in SYSTEM_PROMPT


class TestVRAMPreflight:
    """Tests for VRAM preflight before Ollama load."""

    @pytest.mark.asyncio
    async def test_preflight_called_before_ollama(self):
        """run_preflight_checks() called before Ollama inference."""
        from signals.llm_context import generate_market_context

        call_order = []

        with patch("signals.llm_context.run_preflight_checks") as mock_preflight:
            mock_preflight.return_value = MagicMock(passed=True)
            mock_preflight.side_effect = lambda *args, **kwargs: (
                call_order.append("preflight"),
                MagicMock(passed=True),
            )[1]

            with patch("signals.llm_context._call_ollama") as mock_ollama:
                mock_ollama.side_effect = lambda *args, **kwargs: (
                    call_order.append("ollama"),
                    '{"bullish_factors":[],"bearish_factors":[],"regime_flag":"neutral","confidence":0.5}',
                )[1]

                await generate_market_context(
                    funding_rate=0.01,
                    oi_delta=0.0,
                    liquidation_data=None,
                    news_headlines=None,
                )

        # Preflight must come before Ollama
        assert call_order == ["preflight", "ollama"]

    @pytest.mark.asyncio
    async def test_preflight_failure_skips_ollama(self):
        """When preflight fails, Ollama is never called."""
        from signals.llm_context import generate_market_context

        with patch("signals.llm_context.run_preflight_checks") as mock_preflight:
            mock_preflight.return_value = MagicMock(passed=False, reason="STOP file exists")

            with patch("signals.llm_context._call_ollama") as mock_ollama:
                ctx = await generate_market_context(
                    funding_rate=0.01,
                    oi_delta=0.0,
                    liquidation_data=None,
                    news_headlines=None,
                )

        # Ollama should NOT have been called
        mock_ollama.assert_not_called()

        # Should return neutral fallback
        assert ctx.regime_flag == "neutral"
        assert ctx.confidence == 0.0


class TestStructuredParsing:
    """Tests for JSON extraction with graceful fallback."""

    @pytest.mark.asyncio
    async def test_valid_json_parsed_correctly(self):
        """Valid JSON response is parsed into LLMContext."""
        from signals.llm_context import generate_market_context

        valid_json = """{
            "bullish_factors": ["factor1", "factor2"],
            "bearish_factors": ["factor3"],
            "regime_flag": "confirming",
            "confidence": 0.8
        }"""

        with patch("signals.llm_context.run_preflight_checks") as mock_preflight:
            mock_preflight.return_value = MagicMock(passed=True)

            with patch("signals.llm_context._call_ollama") as mock_ollama:
                mock_ollama.return_value = valid_json

                ctx = await generate_market_context(
                    funding_rate=-0.01,
                    oi_delta=2.0,
                    liquidation_data={"long_liquidations": 1000},
                    news_headlines=["Test headline"],
                )

        assert ctx.bullish_factors == ["factor1", "factor2"]
        assert ctx.bearish_factors == ["factor3"]
        assert ctx.regime_flag == "confirming"
        # With all 4 inputs, confidence cap is 1.0, so LLM's 0.8 is preserved
        assert ctx.confidence == 0.8

    @pytest.mark.asyncio
    async def test_malformed_json_fallback_neutral(self):
        """Malformed JSON falls back to neutral context."""
        from signals.llm_context import generate_market_context

        malformed_json = "This is not JSON at all {broken"

        with patch("signals.llm_context.run_preflight_checks") as mock_preflight:
            mock_preflight.return_value = MagicMock(passed=True)

            with patch("signals.llm_context._call_ollama") as mock_ollama:
                mock_ollama.return_value = malformed_json

                ctx = await generate_market_context(
                    funding_rate=0.01,
                    oi_delta=0.0,
                    liquidation_data=None,
                    news_headlines=None,
                )

        # Should fallback to neutral
        assert ctx.regime_flag == "neutral"
        assert ctx.confidence == 0.0
        assert ctx.bullish_factors == []
        assert ctx.bearish_factors == []

    @pytest.mark.asyncio
    async def test_partial_json_fallback_neutral(self):
        """JSON missing required fields falls back to neutral."""
        from signals.llm_context import generate_market_context

        partial_json = '{"bullish_factors": ["x"]}'  # Missing other fields

        with patch("signals.llm_context.run_preflight_checks") as mock_preflight:
            mock_preflight.return_value = MagicMock(passed=True)

            with patch("signals.llm_context._call_ollama") as mock_ollama:
                mock_ollama.return_value = partial_json

                ctx = await generate_market_context(
                    funding_rate=0.01,
                    oi_delta=0.0,
                    liquidation_data=None,
                    news_headlines=None,
                )

        # Should fallback to neutral (missing required fields)
        assert ctx.regime_flag == "neutral"
        assert ctx.confidence == 0.0

    @pytest.mark.asyncio
    async def test_json_with_extra_text_extracted(self):
        """JSON embedded in extra text is still extracted."""
        from signals.llm_context import generate_market_context

        json_with_text = """Here is my analysis:

        {"bullish_factors": ["A"], "bearish_factors": ["B"], "regime_flag": "neutral", "confidence": 0.6}

        That's my view."""

        with patch("signals.llm_context.run_preflight_checks") as mock_preflight:
            mock_preflight.return_value = MagicMock(passed=True)

            with patch("signals.llm_context._call_ollama") as mock_ollama:
                mock_ollama.return_value = json_with_text

                ctx = await generate_market_context(
                    funding_rate=0.01,
                    oi_delta=0.0,
                    liquidation_data=None,
                    news_headlines=None,
                )

        # Should extract JSON from surrounding text
        assert ctx.bullish_factors == ["A"]
        assert ctx.bearish_factors == ["B"]
        assert ctx.confidence == 0.6

    @pytest.mark.asyncio
    async def test_invalid_regime_flag_fallback(self):
        """Invalid regime flag value falls back to neutral."""
        from signals.llm_context import generate_market_context

        invalid_regime = """{
            "bullish_factors": [],
            "bearish_factors": [],
            "regime_flag": "bullish",
            "confidence": 0.5
        }"""

        with patch("signals.llm_context.run_preflight_checks") as mock_preflight:
            mock_preflight.return_value = MagicMock(passed=True)

            with patch("signals.llm_context._call_ollama") as mock_ollama:
                mock_ollama.return_value = invalid_regime

                ctx = await generate_market_context(
                    funding_rate=0.01,
                    oi_delta=0.0,
                    liquidation_data=None,
                    news_headlines=None,
                )

        # Should fallback to neutral (invalid regime_flag)
        assert ctx.regime_flag == "neutral"


class TestEmptyInputHandling:
    """Tests for handling None/empty inputs."""

    @pytest.mark.asyncio
    async def test_all_none_inputs_returns_neutral(self):
        """All None inputs return neutral context with low confidence."""
        from signals.llm_context import generate_market_context

        with patch("signals.llm_context.run_preflight_checks") as mock_preflight:
            mock_preflight.return_value = MagicMock(passed=True)

            with patch("signals.llm_context._call_ollama") as mock_ollama:
                # Should not even call Ollama with all None inputs
                ctx = await generate_market_context(
                    funding_rate=None,
                    oi_delta=None,
                    liquidation_data=None,
                    news_headlines=None,
                )

                # Ollama should NOT be called when no data provided
                mock_ollama.assert_not_called()

        # Should return neutral with low confidence
        assert ctx.regime_flag == "neutral"
        assert ctx.confidence <= 0.3  # Low confidence when no data

    @pytest.mark.asyncio
    async def test_funding_rate_none_oi_delta_none(self):
        """funding_rate=None and oi_delta=None returns neutral."""
        from signals.llm_context import generate_market_context

        with patch("signals.llm_context.run_preflight_checks") as mock_preflight:
            mock_preflight.return_value = MagicMock(passed=True)

            ctx = await generate_market_context(
                funding_rate=None,
                oi_delta=None,
                liquidation_data={"longs": 100, "shorts": 50},  # Some data
                news_headlines=None,
            )

        # Should still work but with reduced confidence
        assert ctx.regime_flag == "neutral"
        assert ctx.confidence <= 0.5

    @pytest.mark.asyncio
    async def test_partial_inputs_reduced_confidence(self):
        """Partial inputs (some None) reduce confidence."""
        from signals.llm_context import generate_market_context

        with patch("signals.llm_context.run_preflight_checks") as mock_preflight:
            mock_preflight.return_value = MagicMock(passed=True)

            with patch("signals.llm_context._call_ollama") as mock_ollama:
                mock_ollama.return_value = """{
                    "bullish_factors": ["funding negative"],
                    "bearish_factors": [],
                    "regime_flag": "confirming",
                    "confidence": 0.9
                }"""

                ctx = await generate_market_context(
                    funding_rate=-0.01,  # Has data
                    oi_delta=None,  # Missing
                    liquidation_data=None,  # Missing
                    news_headlines=None,  # Missing
                )

        # Confidence should be capped due to missing inputs
        assert ctx.confidence <= 0.6


class TestOllamaCleanup:
    """Tests for OLLAMA_KEEP_ALIVE=0 enforcement."""

    @pytest.mark.asyncio
    async def test_ollama_keep_alive_set_before_inference(self):
        """OLLAMA_KEEP_ALIVE=0 is set before Ollama call."""
        from signals.llm_context import generate_market_context

        with patch("signals.llm_context.run_preflight_checks") as mock_preflight:
            mock_preflight.return_value = MagicMock(passed=True)

            with patch("signals.llm_context.enforce_ollama_keep_alive") as mock_enforce:
                with patch("signals.llm_context._call_ollama") as mock_ollama:
                    mock_ollama.return_value = """{
                        "bullish_factors": [],
                        "bearish_factors": [],
                        "regime_flag": "neutral",
                        "confidence": 0.5
                    }"""

                    await generate_market_context(
                        funding_rate=0.01,
                        oi_delta=0.0,
                        liquidation_data=None,
                        news_headlines=None,
                    )

        # Should have called enforce before Ollama
        mock_enforce.assert_called()

    def test_enforce_ollama_keep_alive_sets_env(self, monkeypatch):
        """enforce_ollama_keep_alive sets OLLAMA_KEEP_ALIVE=0."""
        from signals.preflight import enforce_ollama_keep_alive

        monkeypatch.delenv("OLLAMA_KEEP_ALIVE", raising=False)

        enforce_ollama_keep_alive()

        assert os.environ.get("OLLAMA_KEEP_ALIVE") == "0"

    @pytest.mark.asyncio
    async def test_model_unloaded_after_inference(self):
        """Model is unloaded after inference completes."""
        from signals.llm_context import generate_market_context

        with patch("signals.llm_context.run_preflight_checks") as mock_preflight:
            mock_preflight.return_value = MagicMock(passed=True)

            with patch("signals.llm_context._call_ollama") as mock_ollama:
                mock_ollama.return_value = """{
                    "bullish_factors": [],
                    "bearish_factors": [],
                    "regime_flag": "neutral",
                    "confidence": 0.5
                }"""

                with patch("signals.llm_context._unload_model") as mock_unload:
                    await generate_market_context(
                        funding_rate=0.01,
                        oi_delta=0.0,
                        liquidation_data=None,
                        news_headlines=None,
                    )

        # Should unload after inference
        mock_unload.assert_called()


class TestOllamaIntegration:
    """Tests for Ollama integration (mocked)."""

    @pytest.mark.asyncio
    async def test_ollama_failure_returns_neutral(self):
        """Ollama connection failure returns neutral context."""
        from signals.llm_context import generate_market_context

        with patch("signals.llm_context.run_preflight_checks") as mock_preflight:
            mock_preflight.return_value = MagicMock(passed=True)

            with patch("signals.llm_context._call_ollama") as mock_ollama:
                mock_ollama.side_effect = Exception("Connection refused")

                ctx = await generate_market_context(
                    funding_rate=0.01,
                    oi_delta=2.0,
                    liquidation_data=None,
                    news_headlines=None,
                )

        # Should return neutral fallback on Ollama failure
        assert ctx.regime_flag == "neutral"
        assert ctx.confidence == 0.0

    @pytest.mark.asyncio
    async def test_ollama_timeout_returns_neutral(self):
        """Ollama timeout returns neutral context."""
        from signals.llm_context import generate_market_context

        with patch("signals.llm_context.run_preflight_checks") as mock_preflight:
            mock_preflight.return_value = MagicMock(passed=True)

            with patch("signals.llm_context._call_ollama") as mock_ollama:
                import asyncio

                mock_ollama.side_effect = asyncio.TimeoutError()

                ctx = await generate_market_context(
                    funding_rate=0.01,
                    oi_delta=2.0,
                    liquidation_data=None,
                    news_headlines=None,
                )

        # Should return neutral fallback on timeout
        assert ctx.regime_flag == "neutral"
        assert ctx.confidence == 0.0


class TestPromptConstruction:
    """Tests for market context prompt construction."""

    def test_prompt_includes_funding_rate(self):
        """Prompt includes funding rate when provided."""
        from signals.llm_context import build_user_prompt

        prompt = build_user_prompt(
            funding_rate=-0.01,
            oi_delta=None,
            liquidation_data=None,
            news_headlines=None,
        )

        assert "funding" in prompt.lower()
        assert "-0.01" in prompt or "-1%" in prompt or "negative" in prompt.lower()

    def test_prompt_includes_oi_delta(self):
        """Prompt includes OI delta when provided."""
        from signals.llm_context import build_user_prompt

        prompt = build_user_prompt(
            funding_rate=None,
            oi_delta=5.5,
            liquidation_data=None,
            news_headlines=None,
        )

        assert "oi" in prompt.lower() or "open interest" in prompt.lower()
        assert "5.5" in prompt

    def test_prompt_includes_liquidation_data(self):
        """Prompt includes liquidation data when provided."""
        from signals.llm_context import build_user_prompt

        prompt = build_user_prompt(
            funding_rate=None,
            oi_delta=None,
            liquidation_data={"long_liquidations": 1000000, "short_liquidations": 500000},
            news_headlines=None,
        )

        assert "liquidation" in prompt.lower()

    def test_prompt_includes_news_headlines(self):
        """Prompt includes news headlines when provided."""
        from signals.llm_context import build_user_prompt

        prompt = build_user_prompt(
            funding_rate=None,
            oi_delta=None,
            liquidation_data=None,
            news_headlines=["Bitcoin ETF sees record inflows", "Whale moves 10k BTC"],
        )

        assert "news" in prompt.lower() or "headline" in prompt.lower()
        assert "ETF" in prompt

    def test_prompt_empty_when_all_none(self):
        """Prompt is minimal when all inputs are None."""
        from signals.llm_context import build_user_prompt

        prompt = build_user_prompt(
            funding_rate=None,
            oi_delta=None,
            liquidation_data=None,
            news_headlines=None,
        )

        # Should indicate no data available
        assert "no data" in prompt.lower() or "unavailable" in prompt.lower() or len(prompt) < 100
