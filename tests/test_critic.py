"""Tests for signal critic with response extraction and validation."""

import pytest

from swarm.critic import (
    CritiqueResult,
    extract_critique,
    validate_critique,
)
from swarm.exceptions import ResponseValidationError


class TestCritiqueResultScore:
    """Test derived score computation."""

    def test_score_weighted_average(self):
        """Test that score is computed as weighted average."""
        critique = CritiqueResult(
            reasoning_quality=0.8,
            technical_alignment=0.6,
            confidence_calibration=0.7,
            critique="Test",
            recommendation="ACCEPT",
            raw_response="{}",
        )

        # Score = 0.35 * 0.8 + 0.40 * 0.6 + 0.25 * 0.7
        expected = 0.35 * 0.8 + 0.40 * 0.6 + 0.25 * 0.7
        assert abs(critique.score - expected) < 0.001

    def test_score_all_high(self):
        """Test score when all dimensions are high."""
        critique = CritiqueResult(
            reasoning_quality=1.0,
            technical_alignment=1.0,
            confidence_calibration=1.0,
            critique="Test",
            recommendation="ACCEPT",
            raw_response="{}",
        )

        assert critique.score == 1.0

    def test_score_all_low(self):
        """Test score when all dimensions are low."""
        critique = CritiqueResult(
            reasoning_quality=0.0,
            technical_alignment=0.0,
            confidence_calibration=0.0,
            critique="Test",
            recommendation="REJECT",
            raw_response="{}",
        )

        assert critique.score == 0.0

    def test_technical_alignment_weight(self):
        """Test that technical_alignment has highest weight (40%)."""
        # High technical, low others
        critique1 = CritiqueResult(
            reasoning_quality=0.0,
            technical_alignment=1.0,
            confidence_calibration=0.0,
            critique="Test",
            recommendation="UNCERTAIN",
            raw_response="{}",
        )

        # High others, low technical
        critique2 = CritiqueResult(
            reasoning_quality=1.0,
            technical_alignment=0.0,
            confidence_calibration=1.0,
            critique="Test",
            recommendation="UNCERTAIN",
            raw_response="{}",
        )

        # critique2 should have higher score (0.35 + 0.25 = 0.6 vs 0.4)
        # But critique1's score per unit is higher (0.4 from single dimension)
        assert critique1.score == 0.4  # Only technical: 0.4 * 1.0
        assert critique2.score == 0.6  # reasoning + confidence: 0.35 + 0.25


class TestCritiqueValidation:
    """Test critique validation logic."""

    def test_valid_critique_passes(self):
        """Test that valid critique passes validation."""
        critique = CritiqueResult(
            reasoning_quality=0.7,
            technical_alignment=0.6,
            confidence_calibration=0.8,
            critique="The signal shows strong technical alignment with RSI indicating momentum.",
            recommendation="ACCEPT",
            raw_response="{}",
        )

        assert validate_critique(critique)

    def test_accept_with_low_score_fails(self):
        """Test that ACCEPT + low score fails validation."""
        critique = CritiqueResult(
            reasoning_quality=0.3,
            technical_alignment=0.2,
            confidence_calibration=0.4,
            critique="This is a detailed critique with sufficient length to pass length check.",
            recommendation="ACCEPT",  # Contradicts low score
            raw_response="{}",
        )

        assert not validate_critique(critique)

    def test_reject_with_high_score_fails(self):
        """Test that REJECT + high score fails validation."""
        critique = CritiqueResult(
            reasoning_quality=0.9,
            technical_alignment=0.8,
            confidence_calibration=0.85,
            critique="This is a detailed critique with sufficient length to pass length check.",
            recommendation="REJECT",  # Contradicts high score
            raw_response="{}",
        )

        assert not validate_critique(critique)

    def test_short_critique_fails(self):
        """Test that too-short critique fails validation."""
        critique = CritiqueResult(
            reasoning_quality=0.7,
            technical_alignment=0.6,
            confidence_calibration=0.8,
            critique="Too short",  # Less than 50 chars
            recommendation="ACCEPT",
            raw_response="{}",
        )

        assert not validate_critique(critique)

    def test_invalid_score_range_fails(self):
        """Test that out-of-range scores fail validation."""
        critique = CritiqueResult(
            reasoning_quality=1.5,  # Out of range
            technical_alignment=0.6,
            confidence_calibration=0.8,
            critique="This is a detailed critique with sufficient length to pass length check.",
            recommendation="ACCEPT",
            raw_response="{}",
        )

        assert not validate_critique(critique)


class TestCritiqueExtraction:
    """Test multi-stage critique extraction."""

    def test_extract_direct_json(self):
        """Test extraction from direct JSON."""
        response = '''{
            "reasoning_quality": 0.8,
            "technical_alignment": 0.7,
            "confidence_calibration": 0.6,
            "critique": "Strong analysis with good technical support",
            "recommendation": "ACCEPT"
        }'''

        critique = extract_critique(response)

        assert critique.reasoning_quality == 0.8
        assert critique.technical_alignment == 0.7
        assert critique.confidence_calibration == 0.6
        assert "Strong analysis" in critique.critique
        assert critique.recommendation == "ACCEPT"

    def test_extract_from_markdown_fences(self):
        """Test extraction from JSON in markdown code fences."""
        response = '''Here is my critique:
```json
{
    "reasoning_quality": 0.5,
    "technical_alignment": 0.4,
    "confidence_calibration": 0.6,
    "critique": "Weak technical alignment, RSI contradicts reasoning",
    "recommendation": "REJECT"
}
```
Hope this helps!'''

        critique = extract_critique(response)

        assert critique.reasoning_quality == 0.5
        assert critique.recommendation == "REJECT"

    def test_extract_with_regex_fallback(self):
        """Test regex extraction as last resort."""
        response = '''My analysis:
        "reasoning_quality": 0.7,
        "technical_alignment": 0.8,
        "confidence_calibration": 0.65,
        "recommendation": "ACCEPT"
        '''

        critique = extract_critique(response)

        assert critique.reasoning_quality == 0.7
        assert critique.technical_alignment == 0.8
        assert critique.confidence_calibration == 0.65
        assert critique.recommendation == "ACCEPT"
        assert "regex fallback" in critique.critique.lower()

    def test_extract_invalid_recommendation_raises_error(self):
        """Test that invalid recommendation raises ValueError."""
        response = '''{
            "reasoning_quality": 0.7,
            "technical_alignment": 0.6,
            "confidence_calibration": 0.5,
            "critique": "Test",
            "recommendation": "MAYBE"
        }'''

        with pytest.raises(ValueError, match="Invalid recommendation"):
            extract_critique(response)

    def test_extract_scores_clamped(self):
        """Test that scores are clamped to [0, 1]."""
        # Too high
        response1 = '''{
            "reasoning_quality": 1.5,
            "technical_alignment": 0.6,
            "confidence_calibration": 0.5,
            "critique": "Test",
            "recommendation": "ACCEPT"
        }'''
        critique1 = extract_critique(response1)
        assert critique1.reasoning_quality == 1.0

        # Too low
        response2 = '''{
            "reasoning_quality": 0.5,
            "technical_alignment": -0.2,
            "confidence_calibration": 0.5,
            "critique": "Test",
            "recommendation": "REJECT"
        }'''
        critique2 = extract_critique(response2)
        assert critique2.technical_alignment == 0.0

    def test_extract_case_insensitive_recommendation(self):
        """Test that recommendation matching is case-insensitive."""
        response = '''{
            "reasoning_quality": 0.7,
            "technical_alignment": 0.6,
            "confidence_calibration": 0.5,
            "critique": "Test",
            "recommendation": "accept"
        }'''

        critique = extract_critique(response)
        assert critique.recommendation == "ACCEPT"  # Normalized to uppercase

    def test_extract_unparseable_raises_error(self):
        """Test that unparseable response raises ResponseValidationError."""
        response = "This is not JSON at all and has no critique fields"

        with pytest.raises(ResponseValidationError):
            extract_critique(response)

    def test_extract_preserves_raw_response(self):
        """Test that raw response is preserved in critique."""
        response = '''{
            "reasoning_quality": 0.7,
            "technical_alignment": 0.6,
            "confidence_calibration": 0.5,
            "critique": "Test",
            "recommendation": "UNCERTAIN"
        }'''

        critique = extract_critique(response)
        assert critique.raw_response == response

    def test_extract_handles_missing_critique_field(self):
        """Test that missing critique field is handled gracefully."""
        response = '''{
            "reasoning_quality": 0.7,
            "technical_alignment": 0.6,
            "confidence_calibration": 0.5,
            "recommendation": "ACCEPT"
        }'''

        critique = extract_critique(response)
        assert critique.critique == ""  # Empty string for missing critique


class TestCritiquePrompt:
    """Test critique prompt structure."""

    def test_prompt_has_adversarial_framing(self):
        """Test that critique template uses adversarial language."""
        from swarm.critic import CRITIQUE_TEMPLATE

        template_lower = CRITIQUE_TEMPLATE.lower()

        # Check for adversarial keywords
        assert "find flaws" in template_lower or "skeptical" in template_lower
        assert "actively look for problems" in template_lower or "critical" in template_lower

    def test_prompt_requires_specific_citations(self):
        """Test that prompt requires citing specific issues."""
        from swarm.critic import CRITIQUE_TEMPLATE

        assert "cite" in CRITIQUE_TEMPLATE.lower() or "specific" in CRITIQUE_TEMPLATE.lower()

    def test_prompt_includes_task_context(self):
        """Test that prompt includes original task prompt."""
        from swarm.critic import CRITIQUE_TEMPLATE

        assert "{task_prompt}" in CRITIQUE_TEMPLATE

    def test_prompt_includes_recent_ohlcv(self):
        """Test that prompt includes recent price action."""
        from swarm.critic import CRITIQUE_TEMPLATE

        assert "{recent_ohlcv}" in CRITIQUE_TEMPLATE
