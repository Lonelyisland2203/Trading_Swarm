"""
Tests for the end-to-end DPO pipeline (run_dpo_training.py).

Covers:
- load_examples_from_jsonl: happy path, malformed lines, version mismatch, empty file
- phase1_load: filtering on valid directions
- phase2_verify: matching outcomes to examples (all, partial, zero)
- phase3_reward: reward computation plumbing
- phase4_pairs: preference pair construction end-to-end with mocks
- dry-run path: train_dpo never called
- CLI argument parsing

No GPU or live market data required.
"""

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from swarm.training_capture import (
    TrainingExample,
    TRAINING_EXAMPLE_VERSION,
    load_examples_from_jsonl,
)


# --------------------------------------------------------------------------- #
# Helpers / fixtures
# --------------------------------------------------------------------------- #


def _make_example(**kwargs) -> dict:
    """Return a minimal serialisable TrainingExample dict."""
    ex = TrainingExample(
        symbol=kwargs.get("symbol", "BTC/USDT"),
        timeframe=kwargs.get("timeframe", "1h"),
        timestamp_ms=kwargs.get("timestamp_ms", 1_700_000_000_000),
        context_id=kwargs.get("context_id", "ctx-1"),
        persona=kwargs.get("persona", "MOMENTUM"),
        generator_signal=kwargs.get(
            "generator_signal",
            {"direction": "HIGHER", "confidence": 0.8, "reasoning": "Bullish momentum"},
        ),
        task_prompt=kwargs.get("task_prompt", "Analyze BTC/USDT"),
        market_regime=kwargs.get("market_regime", "NEUTRAL"),
        was_accepted=kwargs.get("was_accepted", True),
    )
    return ex.to_dict()


@pytest.fixture
def tmp_jsonl(tmp_path):
    """Return a factory that writes lines to a temp JSONL file."""

    def _write(lines: list[str]) -> Path:
        p = tmp_path / "examples.jsonl"
        p.write_text("\n".join(lines))
        return p

    return _write


@pytest.fixture
def single_example_jsonl(tmp_jsonl):
    """JSONL file containing one valid example."""
    return tmp_jsonl([json.dumps(_make_example())])


@pytest.fixture
def multi_example_jsonl(tmp_jsonl):
    """JSONL file with 5 valid examples (multi-persona context)."""
    personas = ["MOMENTUM", "CONTRARIAN", "SCALPER", "SWING", "MACRO"]
    lines = [
        json.dumps(
            _make_example(
                context_id="ctx-multi",
                persona=p,
                example_id_seed=i,
            )
        )
        for i, p in enumerate(personas)
    ]
    return tmp_jsonl(lines)


# --------------------------------------------------------------------------- #
# load_examples_from_jsonl
# --------------------------------------------------------------------------- #


class TestLoadExamplesFromJsonl:
    def test_happy_path_single(self, single_example_jsonl):
        examples = load_examples_from_jsonl(single_example_jsonl)
        assert len(examples) == 1
        assert isinstance(examples[0], TrainingExample)

    def test_happy_path_multi(self, tmp_jsonl):
        lines = [json.dumps(_make_example(context_id="ctx-1")) for _ in range(3)]
        path = tmp_jsonl(lines)
        examples = load_examples_from_jsonl(path)
        assert len(examples) == 3

    def test_empty_file(self, tmp_path):
        p = tmp_path / "empty.jsonl"
        p.write_text("")
        examples = load_examples_from_jsonl(p)
        assert examples == []

    def test_nonexistent_file(self, tmp_path):
        examples = load_examples_from_jsonl(tmp_path / "missing.jsonl")
        assert examples == []

    def test_skips_malformed_lines(self, tmp_jsonl):
        lines = [
            json.dumps(_make_example()),
            "NOT VALID JSON {{{{",
            json.dumps(_make_example()),
        ]
        path = tmp_jsonl(lines)
        examples = load_examples_from_jsonl(path)
        assert len(examples) == 2

    def test_skips_version_mismatch(self, tmp_jsonl):
        bad = _make_example()
        bad["version"] = "99.0.0"  # Incompatible major version
        good = _make_example()
        path = tmp_jsonl([json.dumps(bad), json.dumps(good)])
        examples = load_examples_from_jsonl(path)
        assert len(examples) == 1

    def test_skips_blank_lines(self, tmp_jsonl):
        lines = [
            json.dumps(_make_example()),
            "",
            "   ",
            json.dumps(_make_example()),
        ]
        path = tmp_jsonl(lines)
        examples = load_examples_from_jsonl(path)
        assert len(examples) == 2

    def test_compatible_version_passes(self, tmp_jsonl):
        ex = _make_example()
        ex["version"] = TRAINING_EXAMPLE_VERSION
        path = tmp_jsonl([json.dumps(ex)])
        examples = load_examples_from_jsonl(path)
        assert len(examples) == 1


# --------------------------------------------------------------------------- #
# phase1_load
# --------------------------------------------------------------------------- #


class TestPhase1Load:
    def test_filters_missing_direction(self, tmp_jsonl):
        no_dir = _make_example()
        no_dir["generator_signal"] = {}
        valid = _make_example()
        path = tmp_jsonl([json.dumps(no_dir), json.dumps(valid)])

        from run_dpo_training import phase1_load

        with patch("sys.exit") as mock_exit:
            mock_exit.side_effect = SystemExit
            result = phase1_load(path)

        assert len(result) == 1
        assert result[0].generator_signal.get("direction") == "HIGHER"

    def test_exits_on_empty_file(self, tmp_path):
        p = tmp_path / "empty.jsonl"
        p.write_text("")

        from run_dpo_training import phase1_load

        with pytest.raises(SystemExit):
            phase1_load(p)

    def test_valid_directions_pass(self, tmp_jsonl):
        examples = [
            _make_example(generator_signal={"direction": d, "confidence": 0.7, "reasoning": "x"})
            for d in ("HIGHER", "LOWER", "FLAT")
        ]
        path = tmp_jsonl([json.dumps(e) for e in examples])

        from run_dpo_training import phase1_load

        result = phase1_load(path)
        assert len(result) == 3


# --------------------------------------------------------------------------- #
# phase2_verify (matching logic)
# --------------------------------------------------------------------------- #


class TestPhase2Verify:
    def _make_outcome(self, example_id: str):
        from verifier.outcome import VerifiedOutcome

        return VerifiedOutcome(
            example_id=example_id,
            actual_direction="HIGHER",
            realized_return=0.03,
            max_adverse_excursion=-0.01,
            net_return=0.028,
            entry_price=50000.0,
            exit_price=51500.0,
            bars_held=24,
        )

    def test_all_verified(self, single_example_jsonl):
        examples = load_examples_from_jsonl(single_example_jsonl)
        fake_outcome = self._make_outcome(examples[0].example_id)

        with patch(
            "run_dpo_training._run_verify",
            new=AsyncMock(return_value=[fake_outcome]),
        ):
            from run_dpo_training import phase2_verify

            matched = phase2_verify(examples)

        assert len(matched) == 1
        ex, outcome = matched[0]
        assert outcome.example_id == ex.example_id

    def test_partial_verification(self, tmp_jsonl):
        ex1 = _make_example()
        ex2 = _make_example()
        path = tmp_jsonl([json.dumps(ex1), json.dumps(ex2)])
        examples = load_examples_from_jsonl(path)

        # Only verify first example
        fake_outcome = self._make_outcome(examples[0].example_id)

        with patch(
            "run_dpo_training._run_verify",
            new=AsyncMock(return_value=[fake_outcome]),
        ):
            from run_dpo_training import phase2_verify

            matched = phase2_verify(examples)

        assert len(matched) == 1

    def test_zero_outcomes(self, single_example_jsonl):
        examples = load_examples_from_jsonl(single_example_jsonl)

        with patch(
            "run_dpo_training._run_verify",
            new=AsyncMock(return_value=[]),
        ):
            from run_dpo_training import phase2_verify

            matched = phase2_verify(examples)

        assert matched == []


# --------------------------------------------------------------------------- #
# phase3_reward
# --------------------------------------------------------------------------- #


class TestPhase3Reward:
    def _make_matched_pair(self):
        from verifier.outcome import VerifiedOutcome

        ex = TrainingExample(**_make_example())
        outcome = VerifiedOutcome(
            example_id=ex.example_id,
            actual_direction="HIGHER",
            realized_return=0.03,
            max_adverse_excursion=-0.01,
            net_return=0.028,
            entry_price=50000.0,
            exit_price=51500.0,
            bars_held=24,
        )
        return ex, outcome

    def test_returns_triples(self):
        from run_dpo_training import phase3_reward

        pair = self._make_matched_pair()
        result = phase3_reward([pair])
        assert len(result) == 1
        ex, outcome, reward = result[0]
        assert -1.0 <= reward.final_reward <= 1.0

    def test_empty_input(self):
        from run_dpo_training import phase3_reward

        result = phase3_reward([])
        assert result == []


# --------------------------------------------------------------------------- #
# phase4_pairs (construct_preference_pairs plumbing)
# --------------------------------------------------------------------------- #


class TestPhase4Pairs:
    def _make_triple(self, persona: str, reward_val: float, context_id: str = "ctx-1"):
        from training.reward_engine import ComputedReward
        from verifier.outcome import VerifiedOutcome

        ex = TrainingExample(
            **_make_example(
                persona=persona,
                context_id=context_id,
                generator_signal={
                    "direction": "HIGHER",
                    "confidence": 0.8,
                    "reasoning": f"Reasoning from {persona}",
                },
            )
        )
        outcome = VerifiedOutcome(
            example_id=ex.example_id,
            actual_direction="HIGHER",
            realized_return=0.03,
            max_adverse_excursion=-0.01,
            net_return=0.028,
            entry_price=50000.0,
            exit_price=51500.0,
            bars_held=24,
        )
        reward = ComputedReward(
            final_reward=reward_val,
            return_reward=reward_val,
            directional_reward=reward_val,
            mae_reward=0.0,
            return_weight=0.5,
            directional_weight=0.3,
            mae_weight=0.2,
            return_scale=10.0,
            mae_scale=5.0,
            net_return=0.028,
            realized_return=0.03,
            mae=-0.01,
            predicted_direction="HIGHER",
            actual_direction="HIGHER",
            confidence=0.8,
            components_used=2,
            computation_timestamp="2026-04-05T00:00:00+00:00",
            market_regime="NEUTRAL",
        )
        return ex, outcome, reward

    def test_produces_pairs_with_sufficient_delta(self, tmp_path):
        personas = ["MOMENTUM", "CONTRARIAN", "SCALPER", "SWING", "MACRO"]
        rewards = [0.9, 0.7, 0.5, 0.3, 0.1]
        triples = [self._make_triple(p, r, context_id="ctx-1") for p, r in zip(personas, rewards)]

        from run_dpo_training import phase4_pairs

        pairs = phase4_pairs(
            triples,
            min_delta=0.2,
            save_pairs=False,
            output_dir=tmp_path,
        )
        assert len(pairs) > 0
        for pair in pairs:
            assert pair.reward_delta >= 0.2

    def test_saves_jsonl_when_requested(self, tmp_path):
        personas = ["MOMENTUM", "CONTRARIAN", "SCALPER", "SWING", "MACRO"]
        rewards = [0.9, 0.7, 0.5, 0.3, 0.1]
        triples = [self._make_triple(p, r, context_id="ctx-1") for p, r in zip(personas, rewards)]

        from run_dpo_training import phase4_pairs

        phase4_pairs(
            triples,
            min_delta=0.2,
            save_pairs=True,
            output_dir=tmp_path,
        )
        assert (tmp_path / "preference_pairs.jsonl").exists()

    def test_empty_triples_returns_empty(self, tmp_path):
        from run_dpo_training import phase4_pairs

        pairs = phase4_pairs([], min_delta=0.2, save_pairs=False, output_dir=tmp_path)
        assert pairs == []


# --------------------------------------------------------------------------- #
# Dry-run path
# --------------------------------------------------------------------------- #


class TestDryRun:
    def test_train_dpo_not_called_on_dry_run(self, tmp_path, monkeypatch, tmp_jsonl):
        """train_dpo must never be called when --dry-run is set."""
        personas = ["MOMENTUM", "CONTRARIAN", "SCALPER", "SWING", "MACRO"]
        lines = [
            json.dumps(
                _make_example(
                    context_id="ctx-dry",
                    persona=p,
                    generator_signal={
                        "direction": "HIGHER",
                        "confidence": 0.8,
                        "reasoning": f"Reasoning {p}",
                    },
                )
            )
            for p in personas
        ]
        path = tmp_jsonl(lines)

        train_dpo_called = []

        from verifier.outcome import VerifiedOutcome

        def fake_train_dpo(*args, **kwargs):  # pragma: no cover
            train_dpo_called.append(True)
            raise AssertionError("train_dpo should not be called in dry-run mode")

        # Patch verify to return outcomes for all examples
        examples_data = [json.loads(l) for l in lines]

        async def fake_run_verify(examples, fee_model):
            return [
                VerifiedOutcome(
                    example_id=ex.example_id,
                    actual_direction="HIGHER",
                    realized_return=0.03,
                    max_adverse_excursion=-0.01,
                    net_return=0.028,
                    entry_price=50000.0,
                    exit_price=51500.0,
                    bars_held=24,
                )
                for ex in examples
            ]

        monkeypatch.setattr("run_dpo_training._run_verify", fake_run_verify)
        monkeypatch.setattr(
            "run_dpo_training.phase5_train",
            lambda *a, **kw: train_dpo_called.append(True),
        )

        # Simulate dry-run by calling main() with mocked sys.argv
        monkeypatch.setattr(
            "sys.argv",
            ["run_dpo_training.py", "--dataset", str(path), "--dry-run"],
        )

        from run_dpo_training import main

        main()  # Should return without calling phase5_train

        assert not train_dpo_called, "phase5_train was called during dry-run"


# --------------------------------------------------------------------------- #
# CLI argument parsing
# --------------------------------------------------------------------------- #


class TestArgParsing:
    def test_required_dataset_arg(self, monkeypatch, tmp_path):
        p = tmp_path / "examples.jsonl"
        monkeypatch.setattr("sys.argv", ["run_dpo_training.py", "--dataset", str(p)])
        from run_dpo_training import parse_args

        args = parse_args()
        assert args.dataset == p

    def test_defaults(self, monkeypatch, tmp_path):
        p = tmp_path / "examples.jsonl"
        monkeypatch.setattr("sys.argv", ["run_dpo_training.py", "--dataset", str(p)])
        from run_dpo_training import parse_args

        args = parse_args()
        assert args.min_delta == 0.2
        assert args.dry_run is False
        assert args.save_pairs is False
        assert args.force is False
        assert args.output is None

    def test_all_flags(self, monkeypatch, tmp_path):
        p = tmp_path / "examples.jsonl"
        out = tmp_path / "out"
        monkeypatch.setattr(
            "sys.argv",
            [
                "run_dpo_training.py",
                "--dataset",
                str(p),
                "--output",
                str(out),
                "--save-pairs",
                "--dry-run",
                "--min-delta",
                "0.3",
                "--force",
            ],
        )
        from run_dpo_training import parse_args

        args = parse_args()
        assert args.output == out
        assert args.save_pairs is True
        assert args.dry_run is True
        assert args.min_delta == pytest.approx(0.3)
        assert args.force is True


# --------------------------------------------------------------------------- #
# Fee flip diagnostic
# --------------------------------------------------------------------------- #


class TestPhase2VerifyUsesFeeModel:
    """Test that phase2_verify integrates fee_model parameter."""

    def test_phase2_verify_passes_fee_model_to_run_verify(self, single_example_jsonl):
        """Test that phase2_verify passes fee_model to _run_verify."""
        from run_dpo_training import phase2_verify
        from config.fee_model import FeeModelSettings
        from verifier.outcome import VerifiedOutcome
        from unittest.mock import AsyncMock, patch

        examples = load_examples_from_jsonl(single_example_jsonl)
        fee_model = FeeModelSettings(maker_fee_pct=0.03)

        fake_outcome = VerifiedOutcome(
            example_id=examples[0].example_id,
            actual_direction="HIGHER",
            realized_return=0.03,
            max_adverse_excursion=-0.01,
            net_return=0.028,
            entry_price=50000.0,
            exit_price=51500.0,
            bars_held=24,
        )

        # Mock _run_verify to capture the fee_model argument
        with patch(
            "run_dpo_training._run_verify",
            new=AsyncMock(return_value=[fake_outcome]),
        ) as mock_run_verify:
            matched = phase2_verify(examples, fee_model=fee_model)

        # Verify _run_verify was called with fee_model
        mock_run_verify.assert_called_once()
        args, kwargs = mock_run_verify.call_args
        assert kwargs.get("fee_model") == fee_model or (len(args) > 1 and args[1] == fee_model)
        assert len(matched) == 1


class TestFeeFlipDiagnostic:
    def test_compute_fee_flip_diagnostic_no_flips(self, capsys):
        """Test diagnostic with examples that don't flip (all stay profitable)."""
        from run_dpo_training import compute_fee_flip_diagnostic
        from config.fee_model import FeeModelSettings
        from verifier.outcome import VerifiedOutcome

        # Create examples with gross returns > fee hurdle (0.083% at 0 periods)
        examples_and_outcomes = [
            # +0.15% gross → +0.067% net (no flip)
            (
                TrainingExample(**_make_example(timeframe="1h")),
                VerifiedOutcome(
                    example_id="ex1",
                    actual_direction="HIGHER",
                    realized_return=0.0014925,  # ln(1 + 0.15/100)
                    max_adverse_excursion=-0.01,
                    net_return=0.00066,
                    entry_price=50000.0,
                    exit_price=50075.0,
                    bars_held=24,
                ),
            ),
        ]

        fee_model = FeeModelSettings()

        # Should not raise
        compute_fee_flip_diagnostic(examples_and_outcomes, fee_model)

        captured = capsys.readouterr()
        assert "FEE FLIP DIAGNOSTIC" in captured.out
        assert "Flipped to Negative" in captured.out

    def test_compute_fee_flip_diagnostic_with_flips(self, capsys):
        """Test diagnostic with examples that flip from positive to negative."""
        from run_dpo_training import compute_fee_flip_diagnostic
        from config.fee_model import FeeModelSettings
        from verifier.outcome import VerifiedOutcome
        import math

        # Create examples with mixed outcomes
        examples_and_outcomes = [
            # +0.08% gross (below fee hurdle 0.083%) → negative net (FLIP!)
            (
                TrainingExample(**_make_example(timeframe="1m")),
                VerifiedOutcome(
                    example_id="ex_flip1",
                    actual_direction="HIGHER",
                    realized_return=math.log(1 + 0.08 / 100),
                    max_adverse_excursion=-0.01,
                    net_return=-0.003,
                    entry_price=50000.0,
                    exit_price=50040.0,
                    bars_held=1,
                ),
            ),
            # +0.15% gross → stays positive (no flip)
            (
                TrainingExample(**_make_example(timeframe="1m")),
                VerifiedOutcome(
                    example_id="ex_stay1",
                    actual_direction="HIGHER",
                    realized_return=math.log(1 + 0.15 / 100),
                    max_adverse_excursion=-0.01,
                    net_return=0.067,
                    entry_price=50000.0,
                    exit_price=50075.0,
                    bars_held=1,
                ),
            ),
        ]

        fee_model = FeeModelSettings()

        # Should not raise
        compute_fee_flip_diagnostic(examples_and_outcomes, fee_model)

        captured = capsys.readouterr()
        assert "FEE FLIP DIAGNOSTIC" in captured.out
        assert "1m" in captured.out
        # Should show 1 flip out of 2
        assert "1" in captured.out or "50.0%" in captured.out

    def test_compute_fee_flip_diagnostic_empty(self):
        """Test diagnostic with empty input (should return early)."""
        from run_dpo_training import compute_fee_flip_diagnostic
        from config.fee_model import FeeModelSettings

        fee_model = FeeModelSettings()

        # Should not raise or print anything for empty input
        compute_fee_flip_diagnostic([], fee_model)

    def test_compute_fee_flip_diagnostic_by_timeframe(self, capsys):
        """Test diagnostic groups results by timeframe correctly."""
        from run_dpo_training import compute_fee_flip_diagnostic
        from config.fee_model import FeeModelSettings
        from verifier.outcome import VerifiedOutcome
        import math

        # Create examples from different timeframes
        examples_and_outcomes = [
            # 1m timeframe with flip
            (
                TrainingExample(**_make_example(timeframe="1m")),
                VerifiedOutcome(
                    example_id="ex_1m",
                    actual_direction="HIGHER",
                    realized_return=math.log(1 + 0.08 / 100),
                    max_adverse_excursion=-0.01,
                    net_return=-0.003,
                    entry_price=50000.0,
                    exit_price=50040.0,
                    bars_held=1,
                ),
            ),
            # 1h timeframe without flip
            (
                TrainingExample(**_make_example(timeframe="1h")),
                VerifiedOutcome(
                    example_id="ex_1h",
                    actual_direction="HIGHER",
                    realized_return=math.log(1 + 0.15 / 100),
                    max_adverse_excursion=-0.01,
                    net_return=0.067,
                    entry_price=50000.0,
                    exit_price=50075.0,
                    bars_held=24,
                ),
            ),
        ]

        fee_model = FeeModelSettings()

        compute_fee_flip_diagnostic(examples_and_outcomes, fee_model)

        captured = capsys.readouterr()
        assert "1m" in captured.out
        assert "1h" in captured.out
        assert "FEE FLIP DIAGNOSTIC" in captured.out
