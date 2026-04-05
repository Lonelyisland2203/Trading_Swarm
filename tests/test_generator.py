"""Tests for signal generator with personas and response extraction."""

import json
import pytest

from data.prompt_builder import TaskType
from data.regime_filter import MarketRegime
from swarm.generator import (
    BASE_PERSONA_WEIGHTS,
    REGIME_MODIFIERS,
    TradingPersona,
    extract_signal,
    sample_persona,
)
from swarm.exceptions import ResponseValidationError


class TestPersonaSampling:
    """Test persona sampling with regime weighting."""

    def test_sample_persona_with_seed_reproducible(self):
        """Test that same seed produces same persona."""
        persona1 = sample_persona(MarketRegime.NEUTRAL, seed=42)
        persona2 = sample_persona(MarketRegime.NEUTRAL, seed=42)
        assert persona1 == persona2

    def test_sample_persona_different_seeds(self):
        """Test that different seeds can produce different personas."""
        # Run multiple times to check for variation
        personas = set()
        for seed in range(100):
            personas.add(sample_persona(MarketRegime.NEUTRAL, seed=seed))

        # Should have sampled multiple personas
        assert len(personas) > 1

    def test_regime_modifiers_affect_weights(self):
        """Test that regime modifiers change persona distribution."""
        # Sample many times in RISK_OFF regime
        risk_off_personas = [
            sample_persona(MarketRegime.RISK_OFF, seed=i) for i in range(1000)
        ]

        # Count conservative vs breakout
        conservative_count = sum(1 for p in risk_off_personas if p == TradingPersona.CONSERVATIVE)
        breakout_count = sum(1 for p in risk_off_personas if p == TradingPersona.BREAKOUT)

        # In RISK_OFF, conservative should be boosted (1.5x), breakout reduced (0.5x)
        # So conservative should appear more often than breakout
        assert conservative_count > breakout_count

    def test_base_weights_sum_to_one(self):
        """Test that base persona weights sum to 1.0."""
        total = sum(BASE_PERSONA_WEIGHTS.values())
        assert abs(total - 1.0) < 1e-6

    def test_all_personas_reachable(self):
        """Test that all personas can be sampled."""
        sampled = set()
        for seed in range(1000):
            sampled.add(sample_persona(MarketRegime.NEUTRAL, seed=seed))

        # Should have sampled all 5 personas
        assert len(sampled) == len(TradingPersona)


class TestResponseExtraction:
    """Test multi-stage JSON extraction."""

    def test_extract_direct_json(self):
        """Test extraction from direct JSON."""
        response = '{"direction": "HIGHER", "confidence": 0.75, "reasoning": "Strong momentum"}'
        signal = extract_signal(response, TradingPersona.MOMENTUM, TaskType.PREDICT_DIRECTION)

        assert signal.direction == "HIGHER"
        assert signal.confidence == 0.75
        assert signal.reasoning == "Strong momentum"
        assert signal.persona == TradingPersona.MOMENTUM

    def test_extract_from_markdown_fences(self):
        """Test extraction from JSON in markdown code fences."""
        response = '''Here is my prediction:
```json
{"direction": "LOWER", "confidence": 0.6, "reasoning": "Overbought"}
```
Hope this helps!'''

        signal = extract_signal(response, TradingPersona.CONTRARIAN, TaskType.PREDICT_DIRECTION)

        assert signal.direction == "LOWER"
        assert signal.confidence == 0.6
        assert signal.reasoning == "Overbought"

    def test_extract_from_thinking_tags(self):
        """Test extraction from DeepSeek thinking tags."""
        response = '''<think>Let me analyze this...</think>
{"direction": "HIGHER", "confidence": 0.8, "reasoning": "Breakout signal"}'''

        signal = extract_signal(response, TradingPersona.BREAKOUT, TaskType.PREDICT_DIRECTION)

        assert signal.direction == "HIGHER"
        assert signal.confidence == 0.8

    def test_extract_with_regex_fallback(self):
        """Test regex extraction as last resort."""
        response = '''My analysis shows "direction": "LOWER" with "confidence": 0.55'''

        signal = extract_signal(response, TradingPersona.CONSERVATIVE, TaskType.PREDICT_DIRECTION)

        assert signal.direction == "LOWER"
        assert signal.confidence == 0.55
        assert "extracted via regex fallback" in signal.reasoning.lower()

    def test_extract_invalid_direction_raises_error(self):
        """Test that invalid direction raises ValueError."""
        response = '{"direction": "SIDEWAYS", "confidence": 0.5, "reasoning": "Test"}'

        with pytest.raises(ValueError, match="Invalid direction"):
            extract_signal(response, TradingPersona.CONSERVATIVE, TaskType.PREDICT_DIRECTION)

    def test_extract_confidence_clamped(self):
        """Test that confidence is clamped to [0, 1]."""
        # Too high
        response1 = '{"direction": "HIGHER", "confidence": 1.5, "reasoning": "Test"}'
        signal1 = extract_signal(response1, TradingPersona.MOMENTUM, TaskType.PREDICT_DIRECTION)
        assert signal1.confidence == 1.0

        # Too low
        response2 = '{"direction": "LOWER", "confidence": -0.2, "reasoning": "Test"}'
        signal2 = extract_signal(response2, TradingPersona.CONSERVATIVE, TaskType.PREDICT_DIRECTION)
        assert signal2.confidence == 0.0

    def test_extract_case_insensitive_direction(self):
        """Test that direction matching is case-insensitive."""
        response = '{"direction": "higher", "confidence": 0.7, "reasoning": "Test"}'
        signal = extract_signal(response, TradingPersona.MOMENTUM, TaskType.PREDICT_DIRECTION)
        assert signal.direction == "HIGHER"  # Normalized to uppercase

    def test_extract_unparseable_raises_error(self):
        """Test that unparseable response raises ResponseValidationError."""
        response = "This is not JSON at all and has no direction or confidence"

        with pytest.raises(ResponseValidationError):
            extract_signal(response, TradingPersona.CONSERVATIVE, TaskType.PREDICT_DIRECTION)

    def test_extract_preserves_raw_response(self):
        """Test that raw response is preserved in signal."""
        response = '{"direction": "HIGHER", "confidence": 0.65, "reasoning": "Test"}'
        signal = extract_signal(response, TradingPersona.MOMENTUM, TaskType.PREDICT_DIRECTION)
        assert signal.raw_response == response

    def test_extract_handles_missing_reasoning(self):
        """Test that missing reasoning field is handled gracefully."""
        response = '{"direction": "LOWER", "confidence": 0.5}'
        signal = extract_signal(response, TradingPersona.CONTRARIAN, TaskType.PREDICT_DIRECTION)
        assert signal.reasoning == ""  # Empty string for missing reasoning


class TestPersonaPrompts:
    """Test that persona prompts are well-defined."""

    def test_all_personas_have_prompts(self):
        """Test that all personas have system prompts defined."""
        from swarm.generator import PERSONA_PROMPTS

        for persona in TradingPersona:
            assert persona in PERSONA_PROMPTS
            prompt = PERSONA_PROMPTS[persona]
            assert len(prompt) > 0
            assert isinstance(prompt, str)

    def test_persona_prompts_are_distinct(self):
        """Test that persona prompts are unique."""
        from swarm.generator import PERSONA_PROMPTS

        prompts = list(PERSONA_PROMPTS.values())
        assert len(prompts) == len(set(prompts))  # All unique

    def test_contrarian_prompt_mentions_key_concepts(self):
        """Test that contrarian prompt includes expected concepts."""
        from swarm.generator import PERSONA_PROMPTS

        prompt = PERSONA_PROMPTS[TradingPersona.CONTRARIAN]
        assert "overreaction" in prompt.lower() or "contrary" in prompt.lower()
        assert "rsi" in prompt.lower()

    def test_momentum_prompt_mentions_trend(self):
        """Test that momentum prompt emphasizes trends."""
        from swarm.generator import PERSONA_PROMPTS

        prompt = PERSONA_PROMPTS[TradingPersona.MOMENTUM]
        assert "trend" in prompt.lower()
        assert "macd" in prompt.lower() or "moving average" in prompt.lower()

    def test_conservative_prompt_mentions_risk(self):
        """Test that conservative prompt emphasizes risk management."""
        from swarm.generator import PERSONA_PROMPTS

        prompt = PERSONA_PROMPTS[TradingPersona.CONSERVATIVE]
        assert "capital" in prompt.lower() or "preservation" in prompt.lower()
        assert "risk" in prompt.lower() or "conservative" in prompt.lower()
