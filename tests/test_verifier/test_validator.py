"""Tests for point-in-time validation."""

import pytest

from verifier.validator import (
    validate_forward_data_completeness,
    validate_no_lookahead,
)


class TestValidateNoLookahead:
    """Test point-in-time lookahead detection."""
    
    def test_correct_temporal_ordering_passes(self):
        """Test that correct ordering passes validation."""
        # signal < entry <= forward_start < forward_end
        validate_no_lookahead(
            signal_timestamp_ms=1000,
            entry_timestamp_ms=1001,
            forward_data_start_ms=1001,
            forward_data_end_ms=2000,
        )
        
        # Should complete without assertion error
    
    def test_entry_before_signal_fails(self):
        """Test that entry before signal raises AssertionError."""
        with pytest.raises(AssertionError, match="Entry must be after signal"):
            validate_no_lookahead(
                signal_timestamp_ms=1000,
                entry_timestamp_ms=999,  # BEFORE signal!
                forward_data_start_ms=1001,
                forward_data_end_ms=2000,
            )
    
    def test_entry_equal_to_signal_fails(self):
        """Test that entry equal to signal raises AssertionError."""
        with pytest.raises(AssertionError, match="Entry must be after signal"):
            validate_no_lookahead(
                signal_timestamp_ms=1000,
                entry_timestamp_ms=1000,  # EQUAL to signal!
                forward_data_start_ms=1001,
                forward_data_end_ms=2000,
            )
    
    def test_forward_data_before_entry_fails(self):
        """Test that forward data starting before entry raises AssertionError."""
        with pytest.raises(AssertionError, match="Forward data must start at or after entry"):
            validate_no_lookahead(
                signal_timestamp_ms=1000,
                entry_timestamp_ms=1500,
                forward_data_start_ms=1400,  # BEFORE entry!
                forward_data_end_ms=2000,
            )
    
    def test_forward_data_can_equal_entry(self):
        """Test that forward data starting at entry time is allowed."""
        # This is valid: entry and forward data start at same timestamp
        validate_no_lookahead(
            signal_timestamp_ms=1000,
            entry_timestamp_ms=1001,
            forward_data_start_ms=1001,  # Equal to entry - OK
            forward_data_end_ms=2000,
        )
    
    def test_invalid_forward_data_range_fails(self):
        """Test that forward data with end <= start raises AssertionError."""
        with pytest.raises(AssertionError, match="Forward data range invalid"):
            validate_no_lookahead(
                signal_timestamp_ms=1000,
                entry_timestamp_ms=1001,
                forward_data_start_ms=2000,
                forward_data_end_ms=2000,  # End equals start!
            )
        
        with pytest.raises(AssertionError, match="Forward data range invalid"):
            validate_no_lookahead(
                signal_timestamp_ms=1000,
                entry_timestamp_ms=1001,
                forward_data_start_ms=2000,
                forward_data_end_ms=1500,  # End before start!
            )
    
    def test_assertion_message_includes_timestamps(self):
        """Test that assertion error includes diagnostic timestamps."""
        try:
            validate_no_lookahead(
                signal_timestamp_ms=1000,
                entry_timestamp_ms=999,
                forward_data_start_ms=1001,
                forward_data_end_ms=2000,
            )
            pytest.fail("Should have raised AssertionError")
        except AssertionError as e:
            # Message should include the timestamps
            assert "1000" in str(e)
            assert "999" in str(e)


class TestValidateForwardDataCompleteness:
    """Test forward data completeness validation."""
    
    def test_exact_match_passes(self):
        """Test that exact bar count match passes."""
        result = validate_forward_data_completeness(
            expected_bars=24,
            actual_bars=24,
        )
        assert result is True
    
    def test_within_tolerance_passes(self):
        """Test that difference within tolerance passes."""
        # Default tolerance is 1 bar
        result = validate_forward_data_completeness(
            expected_bars=24,
            actual_bars=23,
            tolerance=1,
        )
        assert result is True
        
        result = validate_forward_data_completeness(
            expected_bars=24,
            actual_bars=25,
            tolerance=1,
        )
        assert result is True
    
    def test_exceeds_tolerance_fails(self):
        """Test that difference exceeding tolerance fails."""
        result = validate_forward_data_completeness(
            expected_bars=24,
            actual_bars=20,  # 4 bars missing, tolerance is 1
            tolerance=1,
        )
        assert result is False
        
        result = validate_forward_data_completeness(
            expected_bars=24,
            actual_bars=28,  # 4 extra, tolerance is 1
            tolerance=1,
        )
        assert result is False
    
    def test_custom_tolerance(self):
        """Test with custom tolerance value."""
        # Tolerance of 5 bars
        result = validate_forward_data_completeness(
            expected_bars=100,
            actual_bars=97,
            tolerance=5,
        )
        assert result is True
        
        result = validate_forward_data_completeness(
            expected_bars=100,
            actual_bars=90,  # 10 missing, exceeds tolerance of 5
            tolerance=5,
        )
        assert result is False
    
    def test_zero_tolerance(self):
        """Test with zero tolerance (exact match required)."""
        result = validate_forward_data_completeness(
            expected_bars=24,
            actual_bars=24,
            tolerance=0,
        )
        assert result is True
        
        result = validate_forward_data_completeness(
            expected_bars=24,
            actual_bars=23,  # Off by 1, but tolerance is 0
            tolerance=0,
        )
        assert result is False
