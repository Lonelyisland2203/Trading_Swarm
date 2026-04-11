"""Tests for critic override logic."""

import pytest

from signals.signal_loop import should_override_signal
from swarm.critic import CritiqueResult


class TestShouldOverrideSignal:
    """Tests for critic override decision logic."""

    def test_accept_never_overrides(self):
        """ACCEPT recommendation never overrides."""
        critique = CritiqueResult(
            reasoning_quality=0.3,  # Low, but doesn't matter for ACCEPT
            technical_alignment=0.3,
            confidence_calibration=0.3,
            critique="Signal looks fine",
            recommendation="ACCEPT",
            raw_response="...",
        )

        assert should_override_signal(critique) is False

    def test_uncertain_never_overrides(self):
        """UNCERTAIN recommendation never overrides."""
        critique = CritiqueResult(
            reasoning_quality=0.4,
            technical_alignment=0.4,
            confidence_calibration=0.4,
            critique="Not sure about this one",
            recommendation="UNCERTAIN",
            raw_response="...",
        )

        assert should_override_signal(critique) is False

    def test_reject_with_low_reasoning_quality_overrides(self):
        """REJECT with reasoning_quality < 0.5 overrides to FLAT."""
        critique = CritiqueResult(
            reasoning_quality=0.4,  # Below 0.5
            technical_alignment=0.7,  # Above 0.5
            confidence_calibration=0.6,
            critique="Reasoning is flawed",
            recommendation="REJECT",
            raw_response="...",
        )

        assert should_override_signal(critique) is True

    def test_reject_with_low_technical_alignment_overrides(self):
        """REJECT with technical_alignment < 0.5 overrides to FLAT."""
        critique = CritiqueResult(
            reasoning_quality=0.7,  # Above 0.5
            technical_alignment=0.4,  # Below 0.5
            confidence_calibration=0.6,
            critique="Indicators contradict prediction",
            recommendation="REJECT",
            raw_response="...",
        )

        assert should_override_signal(critique) is True

    def test_reject_with_both_low_scores_overrides(self):
        """REJECT with both scores < 0.5 overrides to FLAT."""
        critique = CritiqueResult(
            reasoning_quality=0.3,
            technical_alignment=0.2,
            confidence_calibration=0.8,  # This one is high
            critique="Multiple issues",
            recommendation="REJECT",
            raw_response="...",
        )

        assert should_override_signal(critique) is True

    def test_reject_with_high_scores_does_not_override(self):
        """REJECT with high scores does NOT override."""
        critique = CritiqueResult(
            reasoning_quality=0.6,  # Above 0.5
            technical_alignment=0.55,  # Above 0.5
            confidence_calibration=0.3,  # Low but not considered
            critique="Minor concerns only",
            recommendation="REJECT",
            raw_response="...",
        )

        assert should_override_signal(critique) is False

    def test_reject_at_threshold_does_not_override(self):
        """REJECT with scores exactly at 0.5 does NOT override."""
        critique = CritiqueResult(
            reasoning_quality=0.5,  # Exactly 0.5 - not < 0.5
            technical_alignment=0.5,
            confidence_calibration=0.5,
            critique="Borderline",
            recommendation="REJECT",
            raw_response="...",
        )

        assert should_override_signal(critique) is False

    def test_reject_just_below_threshold_overrides(self):
        """REJECT with score just below 0.5 overrides."""
        critique = CritiqueResult(
            reasoning_quality=0.49,  # Just below 0.5
            technical_alignment=0.8,
            confidence_calibration=0.7,
            critique="Slightly weak reasoning",
            recommendation="REJECT",
            raw_response="...",
        )

        assert should_override_signal(critique) is True

    def test_confidence_calibration_not_considered(self):
        """Low confidence_calibration alone doesn't trigger override."""
        critique = CritiqueResult(
            reasoning_quality=0.8,
            technical_alignment=0.7,
            confidence_calibration=0.1,  # Very low but not considered
            critique="Over-confident but technically sound",
            recommendation="REJECT",
            raw_response="...",
        )

        assert should_override_signal(critique) is False


class TestCritiqueResultScore:
    """Tests for CritiqueResult.score computed property."""

    def test_score_weighted_correctly(self):
        """Score is computed with correct weights: 35%, 40%, 25%."""
        critique = CritiqueResult(
            reasoning_quality=1.0,
            technical_alignment=1.0,
            confidence_calibration=1.0,
            critique="Perfect",
            recommendation="ACCEPT",
            raw_response="...",
        )

        # 0.35 * 1.0 + 0.40 * 1.0 + 0.25 * 1.0 = 1.0
        assert critique.score == 1.0

    def test_score_zero_inputs(self):
        """Score is 0 when all inputs are 0."""
        critique = CritiqueResult(
            reasoning_quality=0.0,
            technical_alignment=0.0,
            confidence_calibration=0.0,
            critique="Bad",
            recommendation="REJECT",
            raw_response="...",
        )

        assert critique.score == 0.0

    def test_score_mixed_inputs(self):
        """Score is computed correctly for mixed inputs."""
        critique = CritiqueResult(
            reasoning_quality=0.8,
            technical_alignment=0.6,
            confidence_calibration=0.4,
            critique="Mixed",
            recommendation="UNCERTAIN",
            raw_response="...",
        )

        # 0.35 * 0.8 + 0.40 * 0.6 + 0.25 * 0.4 = 0.28 + 0.24 + 0.10 = 0.62
        expected = 0.35 * 0.8 + 0.40 * 0.6 + 0.25 * 0.4
        assert abs(critique.score - expected) < 0.001

    def test_technical_alignment_has_highest_weight(self):
        """Technical alignment has the highest impact on score."""
        # Only technical_alignment = 1.0
        critique1 = CritiqueResult(
            reasoning_quality=0.0,
            technical_alignment=1.0,
            confidence_calibration=0.0,
            critique="Test",
            recommendation="ACCEPT",
            raw_response="...",
        )

        # Only reasoning_quality = 1.0
        critique2 = CritiqueResult(
            reasoning_quality=1.0,
            technical_alignment=0.0,
            confidence_calibration=0.0,
            critique="Test",
            recommendation="ACCEPT",
            raw_response="...",
        )

        # Technical (40%) > Reasoning (35%)
        assert critique1.score > critique2.score
        assert critique1.score == 0.40
        assert critique2.score == 0.35
