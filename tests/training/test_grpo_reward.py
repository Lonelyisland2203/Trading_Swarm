"""
Tests for GRPO reward computation module.

Tests cover:
1. Asymmetric reward matrix (decision reward)
2. Structure reward (regex validation)
3. Directional accuracy
4. Combined reward computation
5. Group advantage calculation
"""

import pytest

from config.fee_model import FeeModelSettings
from training.grpo_config import GRPORewardConfig
from training.grpo_reward import (
    clip_component,
    normalize_direction,
    compute_decision_reward,
    check_structure,
    compute_structure_reward,
    compute_directional_accuracy,
    compute_net_return,
    compute_grpo_reward,
    compute_group_advantages,
    GRPORewardResult,
)


class TestClipComponent:
    """Tests for clip_component function."""

    def test_value_within_bounds(self):
        """Values within bounds should not be clipped."""
        assert clip_component(0.5) == 0.5
        assert clip_component(-0.5) == -0.5
        assert clip_component(0.0) == 0.0

    def test_value_exceeds_max(self):
        """Values above max should be clipped to max."""
        assert clip_component(1.5) == 1.0
        assert clip_component(2.0) == 1.0
        assert clip_component(100.0) == 1.0

    def test_value_below_min(self):
        """Values below min should be clipped to min."""
        assert clip_component(-1.5) == -1.0
        assert clip_component(-2.0) == -1.0
        assert clip_component(-100.0) == -1.0

    def test_boundary_values(self):
        """Boundary values should be preserved."""
        assert clip_component(1.0) == 1.0
        assert clip_component(-1.0) == -1.0

    def test_custom_bounds(self):
        """Custom min/max bounds should work."""
        assert clip_component(0.8, min_val=-0.5, max_val=0.5) == 0.5
        assert clip_component(-0.8, min_val=-0.5, max_val=0.5) == -0.5


class TestNormalizeDirection:
    """Tests for normalize_direction function."""

    def test_higher_normalizes_to_long(self):
        """HIGHER should normalize to LONG."""
        assert normalize_direction("HIGHER") == "LONG"
        assert normalize_direction("higher") == "LONG"
        assert normalize_direction("  HIGHER  ") == "LONG"

    def test_long_stays_long(self):
        """LONG should stay LONG."""
        assert normalize_direction("LONG") == "LONG"
        assert normalize_direction("long") == "LONG"

    def test_lower_normalizes_to_short(self):
        """LOWER should normalize to SHORT."""
        assert normalize_direction("LOWER") == "SHORT"
        assert normalize_direction("lower") == "SHORT"

    def test_short_stays_short(self):
        """SHORT should stay SHORT."""
        assert normalize_direction("SHORT") == "SHORT"
        assert normalize_direction("short") == "SHORT"

    def test_flat_normalizes(self):
        """FLAT and NEUTRAL should normalize to FLAT."""
        assert normalize_direction("FLAT") == "FLAT"
        assert normalize_direction("NEUTRAL") == "FLAT"
        assert normalize_direction("neutral") == "FLAT"

    def test_unknown_normalizes_to_flat(self):
        """Unknown directions should normalize to FLAT."""
        assert normalize_direction("SIDEWAYS") == "FLAT"
        assert normalize_direction("UNKNOWN") == "FLAT"
        assert normalize_direction("") == "FLAT"


class TestComputeDecisionReward:
    """Tests for compute_decision_reward function."""

    @pytest.fixture
    def config(self):
        """Default reward config."""
        return GRPORewardConfig()

    def test_true_bullish_positive_return(self, config):
        """True bullish with positive return should give positive reward."""
        reward = compute_decision_reward(
            predicted_direction="LONG",
            actual_direction="LONG",
            net_return_pct=0.5,
            config=config,
        )
        assert reward > 0
        assert reward == pytest.approx(0.5, rel=1e-3)  # 1.0 * 0.5

    def test_true_bearish_positive_return(self, config):
        """True bearish should give positive reward."""
        reward = compute_decision_reward(
            predicted_direction="SHORT",
            actual_direction="SHORT",
            net_return_pct=0.5,
            config=config,
        )
        assert reward > 0
        assert reward == pytest.approx(0.5, rel=1e-3)  # 1.0 * |0.5|

    def test_false_bullish_penalized_more_heavily(self, config):
        """False bullish should be penalized 1.5x more than false bearish."""
        # False bullish: predicted LONG, actual SHORT
        false_bullish_reward = compute_decision_reward(
            predicted_direction="LONG",
            actual_direction="SHORT",
            net_return_pct=0.5,
            config=config,
        )

        # False bearish: predicted SHORT, actual LONG
        false_bearish_reward = compute_decision_reward(
            predicted_direction="SHORT",
            actual_direction="LONG",
            net_return_pct=0.5,
            config=config,
        )

        # Both should be negative
        assert false_bullish_reward < 0
        assert false_bearish_reward < 0

        # False bullish should be more negative
        assert false_bullish_reward < false_bearish_reward

        # Check specific values
        # false_bullish: -1.5 * 0.5 = -0.75
        # false_bearish: -0.8 * 0.5 = -0.4
        assert false_bullish_reward == pytest.approx(-0.75, rel=1e-3)
        assert false_bearish_reward == pytest.approx(-0.4, rel=1e-3)

    def test_flat_prediction_zero_reward(self, config):
        """FLAT predictions should get zero reward."""
        reward = compute_decision_reward(
            predicted_direction="FLAT",
            actual_direction="LONG",
            net_return_pct=1.0,
            config=config,
        )
        assert reward == 0.0

    def test_clipping_applies(self, config):
        """Large returns should be clipped to [-1, 1]."""
        # Large positive return with correct prediction
        reward = compute_decision_reward(
            predicted_direction="LONG",
            actual_direction="LONG",
            net_return_pct=5.0,  # Would give reward > 1
            config=config,
        )
        assert reward == 1.0

        # Large negative penalty
        reward = compute_decision_reward(
            predicted_direction="LONG",
            actual_direction="SHORT",
            net_return_pct=2.0,  # -1.5 * 2.0 = -3.0 -> clipped to -1.0
            config=config,
        )
        assert reward == -1.0

    def test_custom_asymmetry_coefficients(self):
        """Custom asymmetry coefficients should be used."""
        config = GRPORewardConfig(
            false_bullish_penalty=2.0,
            false_bearish_penalty=0.5,
        )

        false_bullish = compute_decision_reward(
            predicted_direction="LONG",
            actual_direction="SHORT",
            net_return_pct=0.3,
            config=config,
        )

        # -2.0 * 0.3 = -0.6
        assert false_bullish == pytest.approx(-0.6, rel=1e-3)


class TestCheckStructure:
    """Tests for check_structure function (regex validation)."""

    def test_all_sections_present_in_order(self):
        """All sections in correct order should return True."""
        completion = """
## THESIS
The market is bullish.

## EVIDENCE
- RSI is above 70
- Price broke resistance

## RISK
- Overbought conditions

## DECISION
LONG with high confidence.
"""
        sections, all_present = check_structure(completion)
        assert all_present is True
        assert sections == ("THESIS", "EVIDENCE", "RISK", "DECISION")

    def test_alternative_formatting(self):
        """Different formatting styles should work."""
        # Bold style
        completion = """
**THESIS**
Bullish outlook.

**EVIDENCE**
Strong momentum.

**RISK**
Volatility.

**DECISION**
Go long.
"""
        sections, all_present = check_structure(completion)
        assert all_present is True

        # Colon style
        completion = """
THESIS: Market is up.
EVIDENCE: Indicators positive.
RISK: Some risk exists.
DECISION: Long position.
"""
        sections, all_present = check_structure(completion)
        assert all_present is True

    def test_missing_section(self):
        """Missing sections should return False."""
        completion = """
## THESIS
Bullish.

## EVIDENCE
Strong.

## DECISION
Long.
"""
        sections, all_present = check_structure(completion)
        assert all_present is False
        assert "RISK" not in sections

    def test_wrong_order(self):
        """Sections out of order should return False."""
        completion = """
## EVIDENCE
Some evidence.

## THESIS
The thesis.

## RISK
Some risk.

## DECISION
The decision.
"""
        sections, all_present = check_structure(completion)
        assert all_present is False
        # Only sections that come after the previous one are counted
        assert len(sections) < 4

    def test_empty_string(self):
        """Empty string should return no sections."""
        sections, all_present = check_structure("")
        assert all_present is False
        assert sections == ()

    def test_risks_plural_accepted(self):
        """RISKS (plural) should be accepted."""
        completion = """
## THESIS
Test.

## EVIDENCE
Test.

## RISKS
Test.

## DECISION
Test.
"""
        sections, all_present = check_structure(completion)
        assert all_present is True


class TestComputeStructureReward:
    """Tests for compute_structure_reward function."""

    @pytest.fixture
    def config(self):
        return GRPORewardConfig()

    def test_all_sections_present_gives_reward(self, config):
        """Complete structure should give positive reward."""
        completion = "## THESIS\nT\n## EVIDENCE\nE\n## RISK\nR\n## DECISION\nD"
        reward, sections, all_present = compute_structure_reward(completion, config)
        assert all_present is True
        assert reward == config.structure_reward_value  # 0.2

    def test_incomplete_structure_zero_reward(self, config):
        """Incomplete structure should give zero reward."""
        completion = "## THESIS\nT\n## EVIDENCE\nE\n## DECISION\nD"
        reward, sections, all_present = compute_structure_reward(completion, config)
        assert all_present is False
        assert reward == 0.0


class TestComputeDirectionalAccuracy:
    """Tests for compute_directional_accuracy function."""

    @pytest.fixture
    def config(self):
        return GRPORewardConfig()

    def test_correct_prediction_long(self, config):
        """Correct LONG prediction should give 1.0."""
        reward = compute_directional_accuracy("LONG", "LONG", config)
        assert reward == 1.0

    def test_correct_prediction_short(self, config):
        """Correct SHORT prediction should give 1.0."""
        reward = compute_directional_accuracy("SHORT", "SHORT", config)
        assert reward == 1.0

    def test_incorrect_prediction(self, config):
        """Incorrect prediction should give 0.0."""
        reward = compute_directional_accuracy("LONG", "SHORT", config)
        assert reward == 0.0

    def test_flat_prediction(self, config):
        """FLAT prediction should give 0.0."""
        reward = compute_directional_accuracy("FLAT", "LONG", config)
        assert reward == 0.0

    def test_flat_actual(self, config):
        """FLAT actual should give 0.0."""
        reward = compute_directional_accuracy("LONG", "FLAT", config)
        assert reward == 0.0


class TestComputeNetReturn:
    """Tests for compute_net_return function."""

    def test_with_default_fee_model(self):
        """Should compute net return using default fee model."""
        net = compute_net_return(gross_return_pct=0.5, holding_periods_8h=1)
        # Default fee model: ~0.093% round trip cost
        # Net = 0.5 - ~0.093 ≈ 0.407
        assert net < 0.5
        assert net > 0.3

    def test_with_custom_fee_model(self):
        """Should use custom fee model if provided."""
        fee_model = FeeModelSettings(
            maker_fee_pct=0.01,
            taker_fee_pct=0.02,
            slippage_pct=0.01,
            include_funding=False,
        )
        net = compute_net_return(
            gross_return_pct=0.5,
            holding_periods_8h=1,
            fee_model=fee_model,
        )
        # Lower fees -> higher net return
        assert net > 0.4


class TestComputeGRPOReward:
    """Tests for compute_grpo_reward main function."""

    @pytest.fixture
    def valid_completion(self):
        """Completion with all required sections."""
        return "## THESIS\nT\n## EVIDENCE\nE\n## RISK\nR\n## DECISION\nD"

    @pytest.fixture
    def invalid_completion(self):
        """Completion missing sections."""
        return "Just some text without proper structure."

    def test_correct_prediction_with_structure(self, valid_completion):
        """Correct prediction with valid structure should give high reward."""
        result = compute_grpo_reward(
            completion=valid_completion,
            predicted_direction="LONG",
            actual_direction="LONG",
            gross_return_pct=0.5,
        )

        assert isinstance(result, GRPORewardResult)
        assert result.final_reward > 0
        assert result.decision_reward > 0
        assert result.structure_reward > 0
        assert result.directional_reward == 1.0
        assert result.all_sections_present is True

    def test_incorrect_prediction_without_structure(self, invalid_completion):
        """Wrong prediction without structure should give low reward."""
        result = compute_grpo_reward(
            completion=invalid_completion,
            predicted_direction="LONG",
            actual_direction="SHORT",
            gross_return_pct=0.5,
        )

        assert result.final_reward < 0
        assert result.decision_reward < 0
        assert result.structure_reward == 0.0
        assert result.directional_reward == 0.0
        assert result.all_sections_present is False

    def test_final_reward_clipping(self, valid_completion):
        """Final reward should be clipped to [-1, 1]."""
        # Very large positive return
        result = compute_grpo_reward(
            completion=valid_completion,
            predicted_direction="LONG",
            actual_direction="LONG",
            gross_return_pct=50.0,  # Huge return
        )
        assert result.final_reward <= 1.0

        # Very large negative penalty
        result = compute_grpo_reward(
            completion=valid_completion,
            predicted_direction="LONG",
            actual_direction="SHORT",
            gross_return_pct=50.0,  # Huge loss
        )
        assert result.final_reward >= -1.0

    def test_reward_components_weighted_correctly(self, valid_completion):
        """Reward components should be weighted according to config."""
        config = GRPORewardConfig()

        result = compute_grpo_reward(
            completion=valid_completion,
            predicted_direction="FLAT",  # Zero decision and directional
            actual_direction="LONG",
            gross_return_pct=0.5,
            config=config,
        )

        # With FLAT prediction:
        # - decision_reward = 0
        # - structure_reward = 0.2 (all sections present)
        # - directional_reward = 0 (FLAT prediction)
        # Final = 0.6 * 0 + 0.2 * 0.2 + 0.2 * 0 = 0.04
        assert result.final_reward == pytest.approx(0.04, rel=1e-2)

    def test_custom_config(self, valid_completion):
        """Custom config should be used."""
        config = GRPORewardConfig(
            decision_weight=0.8,
            structure_weight=0.1,
            directional_weight=0.1,
            false_bullish_penalty=2.0,
        )

        result = compute_grpo_reward(
            completion=valid_completion,
            predicted_direction="LONG",
            actual_direction="SHORT",
            gross_return_pct=0.3,
            config=config,
        )

        # Higher decision weight + higher penalty should give more negative reward
        assert result.final_reward < -0.3

    def test_result_contains_all_fields(self, valid_completion):
        """Result should contain all expected fields."""
        result = compute_grpo_reward(
            completion=valid_completion,
            predicted_direction="LONG",
            actual_direction="LONG",
            gross_return_pct=0.5,
        )

        assert result.predicted_direction == "LONG"
        assert result.actual_direction == "LONG"
        assert result.gross_return_pct == 0.5
        assert result.net_return_pct < 0.5  # After fees
        assert result.sections_found == ("THESIS", "EVIDENCE", "RISK", "DECISION")


class TestComputeGroupAdvantages:
    """Tests for compute_group_advantages function."""

    def test_empty_list(self):
        """Empty list should return empty list."""
        advantages = compute_group_advantages([])
        assert advantages == []

    def test_single_element(self):
        """Single element should have advantage of 0."""
        advantages = compute_group_advantages([0.5])
        assert advantages == [0.0]

    def test_advantages_sum_to_zero(self):
        """Advantages should sum to approximately zero."""
        rewards = [0.5, -0.2, 0.3, 0.1]
        advantages = compute_group_advantages(rewards)
        assert len(advantages) == 4
        assert abs(sum(advantages)) < 1e-6

    def test_highest_reward_has_positive_advantage(self):
        """Highest reward should have positive advantage."""
        rewards = [0.1, 0.5, -0.2, 0.0]  # 0.5 is highest
        advantages = compute_group_advantages(rewards)
        max_idx = rewards.index(max(rewards))
        assert advantages[max_idx] > 0

    def test_lowest_reward_has_negative_advantage(self):
        """Lowest reward should have negative advantage."""
        rewards = [0.1, 0.5, -0.2, 0.0]  # -0.2 is lowest
        advantages = compute_group_advantages(rewards)
        min_idx = rewards.index(min(rewards))
        assert advantages[min_idx] < 0

    def test_equal_rewards_zero_advantages(self):
        """Equal rewards should have zero advantages (within epsilon)."""
        rewards = [0.5, 0.5, 0.5, 0.5]
        advantages = compute_group_advantages(rewards)
        for adv in advantages:
            assert abs(adv) < 1e-6

    def test_typical_group_of_four(self):
        """Test typical G=4 scenario."""
        rewards = [0.8, 0.2, -0.3, 0.1]
        advantages = compute_group_advantages(rewards)

        # Mean = (0.8 + 0.2 - 0.3 + 0.1) / 4 = 0.2
        # First element (0.8) should have highest advantage
        assert advantages[0] > advantages[1] > advantages[3] > advantages[2]


class TestAsymmetricPenaltyBehavior:
    """Tests specifically verifying asymmetric penalty behavior."""

    def test_false_bullish_worse_than_false_bearish_same_magnitude(self):
        """
        False bullish should be penalized more than false bearish
        for the same return magnitude.
        """
        config = GRPORewardConfig()

        # Both predictions wrong with same return magnitude
        false_bullish = compute_grpo_reward(
            completion="Just text",
            predicted_direction="LONG",
            actual_direction="SHORT",
            gross_return_pct=0.5,
            config=config,
        )

        false_bearish = compute_grpo_reward(
            completion="Just text",
            predicted_direction="SHORT",
            actual_direction="LONG",
            gross_return_pct=0.5,
            config=config,
        )

        # Both should be negative (wrong predictions)
        assert false_bullish.final_reward < 0
        assert false_bearish.final_reward < 0

        # False bullish should be MORE negative
        assert false_bullish.final_reward < false_bearish.final_reward

        # The decision component should reflect the 1.5 vs 0.8 ratio
        ratio = abs(false_bullish.decision_reward) / abs(false_bearish.decision_reward)
        expected_ratio = config.false_bullish_penalty / config.false_bearish_penalty
        assert ratio == pytest.approx(expected_ratio, rel=0.01)

    def test_asymmetry_affects_final_reward(self):
        """Asymmetric penalties should significantly affect final reward."""
        # Use config with extreme asymmetry
        config = GRPORewardConfig(
            false_bullish_penalty=3.0,  # Very harsh on false bullish
            false_bearish_penalty=0.5,  # Lenient on false bearish
        )

        false_bullish = compute_grpo_reward(
            completion="Just text",
            predicted_direction="LONG",
            actual_direction="SHORT",
            gross_return_pct=0.3,
            config=config,
        )

        false_bearish = compute_grpo_reward(
            completion="Just text",
            predicted_direction="SHORT",
            actual_direction="LONG",
            gross_return_pct=0.3,
            config=config,
        )

        # With 6x ratio (3.0/0.5), false bullish should be much worse
        assert false_bullish.final_reward < false_bearish.final_reward
        assert false_bullish.decision_reward < -0.5  # Strong negative
        assert false_bearish.decision_reward > -0.3  # Mild negative
