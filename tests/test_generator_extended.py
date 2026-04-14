"""Tests for generator with multiple task types."""

import pytest
from unittest.mock import AsyncMock

from data.prompt_builder import TaskType
from data.regime_filter import MarketRegime
from swarm.generator import (
    GeneratorSignal,
    TradingPersona,
    _validate_signal_schema,
    _validate_and_build_signal,
    extract_signal,
    generate_signal,
)
from swarm.exceptions import ResponseValidationError


class TestSignalSchemaValidation:
    """Test schema validation for different task types."""

    def test_validate_predict_direction_schema_valid(self):
        """Test PREDICT_DIRECTION schema validation with valid data."""
        data = {
            "direction": "HIGHER",
            "confidence": 0.75,
            "reasoning": "Strong uptrend",
        }

        assert _validate_signal_schema(data, TaskType.PREDICT_DIRECTION) is True

    def test_validate_assess_momentum_schema_valid(self):
        """Test ASSESS_MOMENTUM schema validation with valid data."""
        data = {
            "direction": "INCREASING",
            "confidence": 0.80,
            "reasoning": "RSI rising",
        }

        assert _validate_signal_schema(data, TaskType.ASSESS_MOMENTUM) is True

    def test_validate_support_resistance_schema_valid(self):
        """Test IDENTIFY_SUPPORT_RESISTANCE schema validation with valid data."""
        data = {
            "support_price": 49000.0,
            "support_confidence": 0.85,
            "resistance_price": 51000.0,
            "resistance_confidence": 0.78,
            "reasoning": "Recent swing points",
        }

        assert _validate_signal_schema(data, TaskType.IDENTIFY_SUPPORT_RESISTANCE) is True

    def test_validate_direction_schema_missing_field(self):
        """Test direction schema validation fails with missing field."""
        data = {
            "direction": "HIGHER",
            # Missing confidence
            "reasoning": "Strong uptrend",
        }

        assert _validate_signal_schema(data, TaskType.PREDICT_DIRECTION) is False

    def test_validate_support_resistance_schema_missing_field(self):
        """Test support/resistance schema validation fails with missing field."""
        data = {
            "support_price": 49000.0,
            "support_confidence": 0.85,
            # Missing resistance fields
            "reasoning": "Recent swing points",
        }

        assert _validate_signal_schema(data, TaskType.IDENTIFY_SUPPORT_RESISTANCE) is False


class TestSignalExtraction:
    """Test signal extraction for different task types."""

    def test_extract_direction_signal_valid_json(self):
        """Test extracting direction signal from valid JSON."""
        response = '{"direction": "HIGHER", "confidence": 0.75, "reasoning": "Strong momentum"}'

        signal = extract_signal(response, TradingPersona.MOMENTUM, TaskType.PREDICT_DIRECTION)

        assert signal.task_type == TaskType.PREDICT_DIRECTION
        assert signal.signal_data["direction"] == "HIGHER"
        assert signal.signal_data["confidence"] == 0.75
        assert signal.reasoning == "Strong momentum"
        assert signal.persona == TradingPersona.MOMENTUM

    def test_extract_momentum_signal_valid_json(self):
        """Test extracting momentum assessment signal from valid JSON."""
        response = '{"direction": "INCREASING", "confidence": 0.82, "reasoning": "MACD crossing up"}'

        signal = extract_signal(response, TradingPersona.MOMENTUM, TaskType.ASSESS_MOMENTUM)

        assert signal.task_type == TaskType.ASSESS_MOMENTUM
        assert signal.signal_data["direction"] == "INCREASING"
        assert signal.signal_data["confidence"] == 0.82
        assert signal.reasoning == "MACD crossing up"

    def test_extract_support_resistance_signal_valid_json(self):
        """Test extracting support/resistance signal from valid JSON."""
        response = '''{
            "support_price": 48500.0,
            "support_confidence": 0.88,
            "resistance_price": 51200.0,
            "resistance_confidence": 0.75,
            "reasoning": "Swing highs at 51200, swing lows at 48500"
        }'''

        signal = extract_signal(response, TradingPersona.CONSERVATIVE, TaskType.IDENTIFY_SUPPORT_RESISTANCE)

        assert signal.task_type == TaskType.IDENTIFY_SUPPORT_RESISTANCE
        assert signal.signal_data["support_price"] == 48500.0
        assert signal.signal_data["support_confidence"] == 0.88
        assert signal.signal_data["resistance_price"] == 51200.0
        assert signal.signal_data["resistance_confidence"] == 0.75
        assert "Swing highs" in signal.reasoning

    def test_extract_signal_with_markdown_fences(self):
        """Test extracting signal from markdown-fenced JSON."""
        response = '''```json
{
    "direction": "LOWER",
    "confidence": 0.65,
    "reasoning": "Breaking support"
}
```'''

        signal = extract_signal(response, TradingPersona.CONTRARIAN, TaskType.PREDICT_DIRECTION)

        assert signal.signal_data["direction"] == "LOWER"
        assert signal.signal_data["confidence"] == 0.65

    def test_extract_signal_regex_fallback_direction(self):
        """Test regex fallback extraction for direction tasks."""
        response = 'Looking at the chart, I think "direction": "HIGHER" with "confidence": 0.70 is appropriate'

        signal = extract_signal(response, TradingPersona.MOMENTUM, TaskType.PREDICT_DIRECTION)

        assert signal.signal_data["direction"] == "HIGHER"
        assert signal.signal_data["confidence"] == 0.70
        assert "[Extracted via regex fallback]" in signal.reasoning

    def test_extract_signal_invalid_raises(self):
        """Test invalid response raises ResponseValidationError."""
        response = "This is not valid JSON at all"

        with pytest.raises(ResponseValidationError, match="Could not extract signal"):
            extract_signal(response, TradingPersona.MOMENTUM, TaskType.PREDICT_DIRECTION)


class TestSignalBuilding:
    """Test signal building and validation."""

    def test_build_direction_signal_clamps_confidence(self):
        """Test confidence is clamped to [0, 1]."""
        data = {
            "direction": "HIGHER",
            "confidence": 1.5,  # Invalid, should be clamped
            "reasoning": "Test",
        }

        signal = _validate_and_build_signal(
            data,
            TradingPersona.MOMENTUM,
            TaskType.PREDICT_DIRECTION,
            "raw",
        )

        assert signal.signal_data["confidence"] == 1.0

    def test_build_support_resistance_validates_positive_prices(self):
        """Test support/resistance prices must be positive."""
        data = {
            "support_price": -100.0,  # Invalid
            "support_confidence": 0.5,
            "resistance_price": 51000.0,
            "resistance_confidence": 0.5,
            "reasoning": "Test",
        }

        with pytest.raises(ValueError, match="Invalid prices"):
            _validate_and_build_signal(
                data,
                TradingPersona.CONSERVATIVE,
                TaskType.IDENTIFY_SUPPORT_RESISTANCE,
                "raw",
            )

    def test_build_direction_signal_validates_direction(self):
        """Test invalid direction raises ValueError."""
        data = {
            "direction": "SIDEWAYS",  # Invalid
            "confidence": 0.5,
            "reasoning": "Test",
        }

        with pytest.raises(ValueError, match="Invalid direction"):
            _validate_and_build_signal(
                data,
                TradingPersona.MOMENTUM,
                TaskType.PREDICT_DIRECTION,
                "raw",
            )


class TestGeneratorSignalProperties:
    """Test GeneratorSignal convenience properties."""

    def test_direction_property_for_direction_task(self):
        """Test direction property returns correct value."""
        signal = GeneratorSignal(
            task_type=TaskType.PREDICT_DIRECTION,
            signal_data={"direction": "HIGHER", "confidence": 0.75},
            reasoning="Test",
            persona=TradingPersona.MOMENTUM,
            raw_response="raw",
        )

        assert signal.direction == "HIGHER"
        assert signal.confidence == 0.75

    def test_direction_property_for_support_resistance_task(self):
        """Test direction property returns None for support/resistance task."""
        signal = GeneratorSignal(
            task_type=TaskType.IDENTIFY_SUPPORT_RESISTANCE,
            signal_data={
                "support_price": 48500.0,
                "support_confidence": 0.88,
                "resistance_price": 51200.0,
                "resistance_confidence": 0.75,
            },
            reasoning="Test",
            persona=TradingPersona.CONSERVATIVE,
            raw_response="raw",
        )

        assert signal.direction is None
        assert signal.confidence is None
        assert signal.signal_data["support_price"] == 48500.0


@pytest.mark.asyncio
class TestGenerateSignalIntegration:
    """Integration tests for generate_signal with different task types."""

    async def test_generate_direction_signal_success(self, monkeypatch):
        """Test successful direction signal generation."""
        # Mock OllamaClient.generate
        mock_response = {
            "response": '{"direction": "HIGHER", "confidence": 0.80, "reasoning": "Strong uptrend"}'
        }

        async def mock_generate(*args, **kwargs):
            return mock_response

        mock_client = AsyncMock()
        mock_client.generate = mock_generate

        signal = await generate_signal(
            mock_client,
            "qwen3:8b",
            "Test prompt",
            MarketRegime.NEUTRAL,
            TaskType.PREDICT_DIRECTION,
        )

        assert signal is not None
        assert signal.task_type == TaskType.PREDICT_DIRECTION
        assert signal.signal_data["direction"] == "HIGHER"
        assert signal.signal_data["confidence"] == 0.80

    async def test_generate_support_resistance_signal_success(self, monkeypatch):
        """Test successful support/resistance signal generation."""
        mock_response = {
            "response": '''{
                "support_price": 49000.0,
                "support_confidence": 0.85,
                "resistance_price": 51000.0,
                "resistance_confidence": 0.78,
                "reasoning": "Recent swing points"
            }'''
        }

        async def mock_generate(*args, **kwargs):
            return mock_response

        mock_client = AsyncMock()
        mock_client.generate = mock_generate

        signal = await generate_signal(
            mock_client,
            "qwen3:8b",
            "Test prompt",
            MarketRegime.NEUTRAL,
            TaskType.IDENTIFY_SUPPORT_RESISTANCE,
        )

        assert signal is not None
        assert signal.task_type == TaskType.IDENTIFY_SUPPORT_RESISTANCE
        assert signal.signal_data["support_price"] == 49000.0
        assert signal.signal_data["resistance_confidence"] == 0.78

    async def test_generate_signal_retry_with_clarification(self, monkeypatch):
        """Test retry with task-specific clarification on parse failure."""
        # First call returns invalid, second call returns valid
        call_count = 0

        async def mock_generate(*args, **kwargs):
            nonlocal call_count
            call_count += 1

            if call_count == 1:
                return {"response": "Invalid response"}
            else:
                return {
                    "response": '{"direction": "LOWER", "confidence": 0.65, "reasoning": "Clarified"}'
                }

        mock_client = AsyncMock()
        mock_client.generate = mock_generate

        signal = await generate_signal(
            mock_client,
            "qwen3:8b",
            "Test prompt",
            MarketRegime.NEUTRAL,
            TaskType.PREDICT_DIRECTION,
        )

        assert signal is not None
        assert call_count == 2
        assert signal.signal_data["direction"] == "LOWER"
