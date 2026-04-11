"""Tests for GRPO trainer."""

from training.grpo_trainer import parse_direction


class TestParseDirection:
    """Tests for parse_direction function."""

    def test_parse_long_uppercase(self) -> None:
        """Test parsing LONG from uppercase."""
        completion = "## THESIS\nBullish\n## EVIDENCE\nRSI\n## RISK\nVol\n## DECISION\nLONG"
        assert parse_direction(completion) == "LONG"

    def test_parse_long_lowercase(self) -> None:
        """Test parsing long from lowercase."""
        completion = "## DECISION\nlong"
        assert parse_direction(completion) == "LONG"

    def test_parse_short(self) -> None:
        """Test parsing SHORT."""
        completion = "## DECISION\nSHORT"
        assert parse_direction(completion) == "SHORT"

    def test_parse_higher_maps_to_long(self) -> None:
        """Test that HIGHER maps to LONG."""
        completion = "## DECISION\nHIGHER"
        assert parse_direction(completion) == "LONG"

    def test_parse_lower_maps_to_short(self) -> None:
        """Test that LOWER maps to SHORT."""
        completion = "## DECISION\nlower"
        assert parse_direction(completion) == "SHORT"

    def test_parse_flat(self) -> None:
        """Test parsing FLAT."""
        completion = "## DECISION\nFLAT"
        assert parse_direction(completion) == "FLAT"

    def test_parse_neutral_maps_to_flat(self) -> None:
        """Test that NEUTRAL maps to FLAT."""
        completion = "## DECISION\nneutral"
        assert parse_direction(completion) == "FLAT"

    def test_corrupt_completion_defaults_flat(self) -> None:
        """Test that unparseable completion defaults to FLAT."""
        completion = "This is garbage text with no structure"
        assert parse_direction(completion) == "FLAT"

    def test_missing_decision_section_defaults_flat(self) -> None:
        """Test that missing DECISION section defaults to FLAT."""
        completion = "## THESIS\nBullish\n## EVIDENCE\nRSI"
        assert parse_direction(completion) == "FLAT"

    def test_direction_in_middle_of_text(self) -> None:
        """Test parsing direction when surrounded by other text."""
        completion = "## DECISION\nBased on analysis, I recommend going LONG with caution."
        assert parse_direction(completion) == "LONG"
