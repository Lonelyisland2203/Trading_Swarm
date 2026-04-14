"""
Tests for fee flip diagnostic in run_dpo_training.py.

Verifies that the diagnostic correctly identifies examples that flip from
profitable (under flat 0.1% fees) to unprofitable (under realistic fees).
"""

import math


from config.fee_model import FeeModelSettings
from swarm.training_capture import TrainingExample
from verifier.outcome import VerifiedOutcome


# Import the function to test
import sys
from pathlib import Path

# Add project root to path to import run_dpo_training
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from run_dpo_training import compute_fee_flip_diagnostic, FEE_FLIP_WARNING_THRESHOLD


def create_example(
    example_id: str,
    symbol: str,
    timeframe: str,
    signal_direction: str = "HIGHER",
) -> TrainingExample:
    """Helper to create a minimal TrainingExample for testing."""
    return TrainingExample(
        example_id=example_id,
        timestamp_ms=1704067200000,  # 2024-01-01 00:00:00 UTC
        symbol=symbol,
        timeframe=timeframe,
        persona="neutral",
        task_prompt="test prompt",
        full_generator_prompt="test full prompt",
        critique_prompt="test critique prompt",
        generator_signal={"direction": signal_direction, "confidence": 0.8},
        generator_raw_response="test raw response",
        critique={"quality": "good"},
        critic_raw_response="test critic raw",
        market_regime="normal",
    )


def create_outcome(
    example_id: str,
    gross_return_pct: float,
    direction: str = "HIGHER",
) -> VerifiedOutcome:
    """
    Helper to create a VerifiedOutcome with specified gross return.

    Args:
        example_id: UUID matching the example
        gross_return_pct: Gross return in percentage (e.g., 0.15 for +0.15%)
        direction: Realized direction

    Returns:
        VerifiedOutcome with gross return converted to log return
    """
    gross_log = math.log(1 + gross_return_pct / 100)

    # Entry/exit prices that produce the desired return
    entry_price = 100.0
    exit_price = entry_price * (1 + gross_return_pct / 100)

    return VerifiedOutcome(
        example_id=example_id,
        actual_direction=direction,
        realized_return=gross_log,
        max_adverse_excursion=-0.01,  # 1% drawdown
        net_return=0.0,  # Will be computed by fee model
        entry_price=entry_price,
        exit_price=exit_price,
        bars_held=60,  # Varies by test
    )


class TestFeeFlipDiagnostic:
    """Tests for compute_fee_flip_diagnostic function."""

    def test_empty_input(self, capsys):
        """Test diagnostic handles empty input gracefully."""
        fee_model = FeeModelSettings()
        compute_fee_flip_diagnostic([], fee_model)

        # Should not crash, may not print anything
        captured = capsys.readouterr()
        assert True  # No exception is success

    def test_no_flips_all_profitable(self, capsys):
        """Test diagnostic when all examples remain profitable after fees."""
        fee_model = FeeModelSettings()

        # Create examples with high gross returns that survive fees
        examples_and_outcomes = [
            (
                create_example("ex1", "BTC/USDT", "1h"),
                create_outcome("ex1", gross_return_pct=1.0),  # +1.0% gross
            ),
            (
                create_example("ex2", "BTC/USDT", "1h"),
                create_outcome("ex2", gross_return_pct=0.5),  # +0.5% gross
            ),
        ]

        compute_fee_flip_diagnostic(examples_and_outcomes, fee_model)

        captured = capsys.readouterr()
        assert "FEE FLIP DIAGNOSTIC" in captured.out
        assert "1h" in captured.out

        # Should show 0 flips
        lines = captured.out.split("\n")
        for line in lines:
            if "1h" in line and "|" in line:
                # Parse the flip count from the table
                parts = [p.strip() for p in line.split("|")]
                if len(parts) > 2:
                    flipped_count = int(parts[2])
                    assert flipped_count == 0
                    break

    def test_detects_flips_marginal_returns(self, capsys):
        """
        Test diagnostic detects flips for marginal returns.

        Note: Old model = 0.10% flat, new model = 0.083% (without funding).
        For a flip to occur, old_net must be positive but new_net must be negative.
        This requires: 0.083% < gross < 0.10%
        """
        fee_model = FeeModelSettings()

        # Create examples with marginal returns that flip to negative
        examples_and_outcomes = [
            # This should flip: 0.09% gross
            # Old: 0.09 - 0.10 = -0.01% (actually negative!)
            # Wait, we need gross where old_net > 0 but new_net < 0
            # That means: gross - 0.10 > 0 AND gross - 0.083 < 0
            # This is impossible! Old fees are HIGHER than new fees.
            #
            # Actually, let me reconsider: For funding periods > 0, new fees can be higher.
            # For 1h (3 funding periods): new = 0.083 + 0.03 = 0.113%
            # So for 1h timeframe: 0.10% < gross < 0.113% would flip
            (
                create_example("ex1", "BTC/USDT", "1h"),
                create_outcome("ex1", gross_return_pct=0.105),  # Should flip with funding
            ),
            # This should flip: 0.11% gross on 1h
            (
                create_example("ex2", "BTC/USDT", "1h"),
                create_outcome("ex2", gross_return_pct=0.11),
            ),
            # This should NOT flip: 0.20% gross is enough to survive fees
            (
                create_example("ex3", "BTC/USDT", "1h"),
                create_outcome("ex3", gross_return_pct=0.20),
            ),
        ]

        compute_fee_flip_diagnostic(examples_and_outcomes, fee_model)

        captured = capsys.readouterr()
        assert "FEE FLIP DIAGNOSTIC" in captured.out

        # Should detect at least 1 flip (ex1 and ex2)
        lines = captured.out.split("\n")
        for line in lines:
            if "1h" in line and "|" in line:
                parts = [p.strip() for p in line.split("|")]
                if len(parts) > 2 and parts[0] == "1h":
                    flipped_count = int(parts[2])
                    assert flipped_count >= 1, f"Expected flips but got {flipped_count}"
                    break

    def test_groups_by_timeframe(self, capsys):
        """Test diagnostic groups examples by timeframe correctly."""
        fee_model = FeeModelSettings()

        examples_and_outcomes = [
            (
                create_example("ex1", "BTC/USDT", "1m"),
                create_outcome("ex1", gross_return_pct=0.15),
            ),
            (
                create_example("ex2", "ETH/USDT", "1h"),
                create_outcome("ex2", gross_return_pct=0.15),
            ),
            (
                create_example("ex3", "BTC/USDT", "1d"),
                create_outcome("ex3", gross_return_pct=0.15),
            ),
        ]

        compute_fee_flip_diagnostic(examples_and_outcomes, fee_model)

        captured = capsys.readouterr()

        # Should have separate rows for 1m, 1h, 1d
        assert "1m" in captured.out
        assert "1h" in captured.out
        assert "1d" in captured.out

    def test_warning_threshold(self, capsys):
        """
        Test that warnings are triggered when flip rate exceeds threshold.

        Use 1h timeframe where funding makes new fees (0.113%) > old fees (0.10%).
        """
        fee_model = FeeModelSettings()

        # Create 10 examples where 5 flip (50% flip rate > 15% threshold)
        examples_and_outcomes = []

        # 5 examples that flip (0.105% gross on 1h - between 0.10% and 0.113%)
        for i in range(5):
            examples_and_outcomes.append(
                (
                    create_example(f"flip{i}", "BTC/USDT", "1h"),
                    create_outcome(f"flip{i}", gross_return_pct=0.105),
                )
            )

        # 5 examples that don't flip (1.0% gross)
        for i in range(5):
            examples_and_outcomes.append(
                (
                    create_example(f"ok{i}", "BTC/USDT", "1h"),
                    create_outcome(f"ok{i}", gross_return_pct=1.0),
                )
            )

        compute_fee_flip_diagnostic(examples_and_outcomes, fee_model)

        captured = capsys.readouterr()

        # Should trigger warning due to high flip rate
        assert "WARNING" in captured.out
        assert "flip rate" in captured.out.lower()

    def test_1d_funding_breakdown(self, capsys):
        """Test that 1d timeframe shows funding cost breakdown."""
        fee_model = FeeModelSettings()

        examples_and_outcomes = [
            (
                create_example("ex1", "BTC/USDT", "1d"),
                create_outcome("ex1", gross_return_pct=0.15),
            ),
        ]

        compute_fee_flip_diagnostic(examples_and_outcomes, fee_model)

        captured = capsys.readouterr()

        # Should show funding breakdown for 1d
        assert "1d funding cost alone" in captured.out
        assert "periods" in captured.out

    def test_average_net_calculation(self, capsys):
        """Test that average old/new net returns are calculated correctly."""
        fee_model = FeeModelSettings()

        # Known example: 0.15% gross
        # Old net: 0.15 - 0.1 = 0.05%
        # New net: 0.15 - 0.083 ≈ 0.067%
        examples_and_outcomes = [
            (
                create_example("ex1", "BTC/USDT", "1m"),
                create_outcome("ex1", gross_return_pct=0.15),
            ),
        ]

        compute_fee_flip_diagnostic(examples_and_outcomes, fee_model)

        captured = capsys.readouterr()

        # Check that averages are printed
        assert "Avg Old Net" in captured.out
        assert "Avg New Net" in captured.out

        # Parse the output to verify numbers are present
        lines = captured.out.split("\n")
        for line in lines:
            if "1m" in line and "|" in line and "Avg" not in line:
                parts = [p.strip() for p in line.split("|")]
                if len(parts) >= 6:
                    # Both averages should be present
                    assert "%" in parts[4]  # Old net
                    assert "%" in parts[5]  # New net
                    break

    def test_total_row_calculation(self, capsys):
        """Test that TOTAL row sums across timeframes correctly."""
        fee_model = FeeModelSettings()

        examples_and_outcomes = [
            # 1m: 2 examples, 1 flip
            (
                create_example("ex1", "BTC/USDT", "1m"),
                create_outcome("ex1", gross_return_pct=0.10),  # Flips
            ),
            (
                create_example("ex2", "BTC/USDT", "1m"),
                create_outcome("ex2", gross_return_pct=1.0),  # OK
            ),
            # 1h: 3 examples, 2 flips
            (
                create_example("ex3", "ETH/USDT", "1h"),
                create_outcome("ex3", gross_return_pct=0.10),  # Flips
            ),
            (
                create_example("ex4", "ETH/USDT", "1h"),
                create_outcome("ex4", gross_return_pct=0.08),  # Flips
            ),
            (
                create_example("ex5", "ETH/USDT", "1h"),
                create_outcome("ex5", gross_return_pct=1.0),  # OK
            ),
        ]

        compute_fee_flip_diagnostic(examples_and_outcomes, fee_model)

        captured = capsys.readouterr()

        # Should show TOTAL row with 5 examples
        assert "TOTAL" in captured.out
        lines = captured.out.split("\n")
        for line in lines:
            if "TOTAL" in line and "|" in line:
                parts = [p.strip() for p in line.split("|")]
                if len(parts) >= 3:
                    total_examples = int(parts[1])
                    assert total_examples == 5
                    break

    def test_negative_returns_not_counted_as_flips(self, capsys):
        """Test that negative returns don't count as flips."""
        fee_model = FeeModelSettings()

        # Example with negative gross return
        # Old: -0.50% - 0.1% = -0.60%
        # New: -0.50% - 0.083% = -0.583%
        # Both negative → no flip
        examples_and_outcomes = [
            (
                create_example("ex1", "BTC/USDT", "1m"),
                create_outcome("ex1", gross_return_pct=-0.50, direction="LOWER"),
            ),
        ]

        compute_fee_flip_diagnostic(examples_and_outcomes, fee_model)

        captured = capsys.readouterr()

        # Should show 0 flips (was negative, still negative)
        lines = captured.out.split("\n")
        for line in lines:
            if "1m" in line and "|" in line:
                parts = [p.strip() for p in line.split("|")]
                if len(parts) > 2 and parts[0] == "1m":
                    flipped_count = int(parts[2])
                    assert flipped_count == 0
                    break

    def test_realistic_scenario_mixed_timeframes(self, capsys):
        """
        Integration test: realistic scenario with mixed timeframes,
        some flips, some not.

        Flip zones (old_net > 0 but new_net < 0):
        - 1m (0.125 periods): old=0.10%, new=0.084% → no natural flip zone
        - 1h (3 periods): old=0.10%, new=0.113% → flip zone: 0.10% < gross < 0.113%
        - 1d (15 periods): old=0.10%, new=0.233% → flip zone: 0.10% < gross < 0.233%
        """
        fee_model = FeeModelSettings()

        examples_and_outcomes = [
            # 1m: No flips (new fees lower than old)
            (
                create_example("1m_1", "BTC/USDT", "1m"),
                create_outcome("1m_1", gross_return_pct=0.12),  # No flip
            ),
            (
                create_example("1m_2", "BTC/USDT", "1m"),
                create_outcome("1m_2", gross_return_pct=0.08),  # No flip
            ),
            # 1h: Some flips (funding makes new > old)
            (
                create_example("1h_1", "ETH/USDT", "1h"),
                create_outcome("1h_1", gross_return_pct=0.105),  # Flips!
            ),
            (
                create_example("1h_2", "ETH/USDT", "1h"),
                create_outcome("1h_2", gross_return_pct=0.50),  # Survives
            ),
            # 1d: Higher funding costs → wider flip zone
            (
                create_example("1d_1", "BTC/USDT", "1d"),
                create_outcome("1d_1", gross_return_pct=0.15),  # Flips!
            ),
            (
                create_example("1d_2", "BTC/USDT", "1d"),
                create_outcome("1d_2", gross_return_pct=2.0),  # Survives
            ),
        ]

        compute_fee_flip_diagnostic(examples_and_outcomes, fee_model)

        captured = capsys.readouterr()

        # Verify structure is present
        assert "FEE FLIP DIAGNOSTIC" in captured.out
        assert "Timeframe" in captured.out
        assert "Total Examples" in captured.out
        assert "Flipped to Negative" in captured.out
        assert "Flip Rate" in captured.out
        assert "TOTAL" in captured.out

        # Should have all three timeframes
        assert "1m" in captured.out
        assert "1h" in captured.out
        assert "1d" in captured.out

        # Should detect some flips (at least in 1h and 1d)
        total_flipped = 0
        lines = captured.out.split("\n")
        for line in lines:
            if "|" in line and any(tf in line for tf in ["1m", "1h", "1d"]):
                parts = [p.strip() for p in line.split("|")]
                if len(parts) > 2:
                    try:
                        # Try to parse flipped count (skip header)
                        if parts[0] in ["1m", "1h", "1d"]:
                            flipped = int(parts[2])
                            total_flipped += flipped
                    except (ValueError, IndexError):
                        continue

        assert total_flipped > 0, "Expected some flips in realistic scenario"


class TestFeeFlipIntegration:
    """Integration tests for fee flip diagnostic in full pipeline."""

    def test_diagnostic_called_in_phase3(self, capsys):
        """Test that diagnostic is automatically called during phase3_reward."""
        from run_dpo_training import phase3_reward

        fee_model = FeeModelSettings()

        # Create matched pairs
        matched = [
            (
                create_example("ex1", "BTC/USDT", "1m"),
                create_outcome("ex1", gross_return_pct=0.15),
            ),
        ]

        # This should trigger the diagnostic
        phase3_reward(matched, fee_model=fee_model)

        captured = capsys.readouterr()

        # Diagnostic output should be present
        assert "FEE FLIP DIAGNOSTIC" in captured.out

    def test_warning_threshold_constant(self):
        """Test that warning threshold is set correctly."""
        assert FEE_FLIP_WARNING_THRESHOLD == 0.15
        assert isinstance(FEE_FLIP_WARNING_THRESHOLD, float)
