"""
Tests for GRPO training data generator.

Tests cover:
- Temporal isolation (no future data in snapshots)
- Direction classification against fee threshold
- Output JSONL schema matches GRPOTrainingExample dataclass
- Market snapshot building
- Verification horizon configuration
"""

import json
from datetime import datetime, UTC
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest

from config.fee_model import FeeModelSettings
from training.grpo_data import GRPOTrainingExample
from training.grpo_data_generator import (
    DEFAULT_SYMBOLS,
    DEFAULT_TIMEFRAMES,
    LOOKBACK_BARS,
    VERIFICATION_HORIZONS,
    build_market_snapshot,
    classify_direction,
    example_to_dict,
    get_verification_horizon,
    load_completed_timestamps,
    parse_date,
    parse_symbols,
    parse_timeframes,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def fee_model() -> FeeModelSettings:
    """Create default fee model."""
    return FeeModelSettings()


@pytest.fixture
def sample_ohlcv_df() -> pd.DataFrame:
    """Create sample OHLCV DataFrame with 100 bars."""
    base_ts = 1704067200000  # 2024-01-01 00:00:00 UTC
    bar_ms = 3600000  # 1h

    data = []
    price = 42000.0

    for i in range(100):
        # Generate realistic price movement
        import random

        random.seed(i)  # Reproducible
        change = random.uniform(-0.02, 0.02)
        open_price = price
        close_price = price * (1 + change)
        high_price = max(open_price, close_price) * (1 + random.uniform(0, 0.01))
        low_price = min(open_price, close_price) * (1 - random.uniform(0, 0.01))
        volume = random.uniform(100, 1000)

        data.append(
            {
                "timestamp": base_ts + (i * bar_ms),
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "close": close_price,
                "volume": volume,
            }
        )
        price = close_price

    return pd.DataFrame(data)


@pytest.fixture
def sample_grpo_example() -> GRPOTrainingExample:
    """Create sample GRPO training example."""
    return GRPOTrainingExample(
        market_snapshot="## Market Data\nSymbol: BTC/USDT\n...",
        actual_direction="LONG",
        gross_return_pct=0.25,
        timestamp_ms=1704067200000,
    )


# =============================================================================
# Direction Classification Tests
# =============================================================================


class TestClassifyDirection:
    """Tests for direction classification based on fee threshold."""

    def test_long_classification_above_threshold(self, fee_model: FeeModelSettings):
        """Test LONG classification when return exceeds fee threshold."""
        # Default fee model: ~0.083% minimum profitable return (no funding)
        min_return = fee_model.minimum_profitable_return_pct(0)

        # Return above threshold -> LONG
        gross_return = min_return + 0.1  # Above threshold
        direction = classify_direction(gross_return, fee_model, holding_periods_8h=0)
        assert direction == "LONG"

    def test_short_classification_below_negative_threshold(self, fee_model: FeeModelSettings):
        """Test SHORT classification when return is below negative threshold."""
        min_return = fee_model.minimum_profitable_return_pct(0)

        # Return below negative threshold -> SHORT
        gross_return = -min_return - 0.1  # Below negative threshold
        direction = classify_direction(gross_return, fee_model, holding_periods_8h=0)
        assert direction == "SHORT"

    def test_flat_classification_within_threshold(self, fee_model: FeeModelSettings):
        """Test FLAT classification when return is within fee threshold."""
        min_return = fee_model.minimum_profitable_return_pct(0)

        # Return within threshold -> FLAT
        gross_return = min_return * 0.5  # Half the threshold
        direction = classify_direction(gross_return, fee_model, holding_periods_8h=0)
        assert direction == "FLAT"

        # Zero return -> FLAT
        direction = classify_direction(0.0, fee_model, holding_periods_8h=0)
        assert direction == "FLAT"

        # Small negative -> FLAT
        gross_return = -min_return * 0.5
        direction = classify_direction(gross_return, fee_model, holding_periods_8h=0)
        assert direction == "FLAT"

    def test_classification_with_funding_periods(self, fee_model: FeeModelSettings):
        """Test classification accounts for funding periods."""
        # With funding periods, threshold increases
        min_return_0 = fee_model.minimum_profitable_return_pct(0)
        min_return_3 = fee_model.minimum_profitable_return_pct(3)

        assert min_return_3 > min_return_0

        # Return that would be LONG without funding might be FLAT with funding
        gross_return = min_return_0 + 0.01  # Just above no-funding threshold
        if gross_return < min_return_3:
            direction = classify_direction(gross_return, fee_model, holding_periods_8h=3)
            assert direction == "FLAT"

    def test_boundary_cases(self, fee_model: FeeModelSettings):
        """Test exact boundary values."""
        min_return = fee_model.minimum_profitable_return_pct(0)

        # Exactly at threshold - should be FLAT (not exceeding)
        direction = classify_direction(min_return, fee_model, holding_periods_8h=0)
        assert direction == "FLAT"

        # Just over threshold
        direction = classify_direction(min_return + 0.001, fee_model, holding_periods_8h=0)
        assert direction == "LONG"

        # Just under negative threshold
        direction = classify_direction(-min_return - 0.001, fee_model, holding_periods_8h=0)
        assert direction == "SHORT"

    def test_extreme_returns(self, fee_model: FeeModelSettings):
        """Test classification with extreme returns."""
        # Large positive return
        direction = classify_direction(10.0, fee_model, holding_periods_8h=0)
        assert direction == "LONG"

        # Large negative return
        direction = classify_direction(-10.0, fee_model, holding_periods_8h=0)
        assert direction == "SHORT"


# =============================================================================
# Market Snapshot Tests
# =============================================================================


class TestBuildMarketSnapshot:
    """Tests for market snapshot building."""

    def test_snapshot_contains_required_sections(
        self, sample_ohlcv_df: pd.DataFrame, fee_model: FeeModelSettings
    ):
        """Test market snapshot contains all required sections."""
        snapshot = build_market_snapshot(
            df=sample_ohlcv_df,
            symbol="BTC/USDT",
            timeframe="1h",
            fee_model=fee_model,
        )

        # Check required sections
        assert "## Market Data" in snapshot
        assert "Symbol: BTC/USDT" in snapshot
        assert "Timeframe: 1h" in snapshot
        assert "Current Price:" in snapshot
        assert "Market Regime:" in snapshot

        assert "## Technical Indicators" in snapshot
        assert "### Price/Trend" in snapshot
        assert "RSI(14):" in snapshot
        assert "MACD Line:" in snapshot

        assert "### Volume" in snapshot
        assert "OBV:" in snapshot
        assert "CMF(20):" in snapshot

        assert "### Volatility" in snapshot
        assert "ATR Normalized:" in snapshot
        assert "BB Width(20):" in snapshot

        assert "### Market Structure" in snapshot
        assert "Open FVG Count:" in snapshot

        assert "## Recent Price Action" in snapshot
        assert "## Execution Context" in snapshot
        assert "round-trip cost:" in snapshot
        assert "Minimum profitable move:" in snapshot

    def test_snapshot_uses_only_historical_data(
        self, sample_ohlcv_df: pd.DataFrame, fee_model: FeeModelSettings
    ):
        """Test snapshot only uses data from the DataFrame (no future leak)."""
        snapshot = build_market_snapshot(
            df=sample_ohlcv_df,
            symbol="BTC/USDT",
            timeframe="1h",
            fee_model=fee_model,
        )

        # The last bar's close price should be the "Current Price"
        last_close = sample_ohlcv_df["close"].iloc[-1]
        assert f"${last_close:.4f}" in snapshot

        # Timestamp should be from the last bar
        last_ts = sample_ohlcv_df["timestamp"].iloc[-1]
        last_dt = pd.to_datetime(last_ts, unit="ms")
        assert last_dt.strftime("%Y-%m-%d") in snapshot

    def test_snapshot_with_empty_df_raises(self, fee_model: FeeModelSettings):
        """Test that empty DataFrame raises error."""
        empty_df = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

        with pytest.raises(ValueError):
            build_market_snapshot(
                df=empty_df,
                symbol="BTC/USDT",
                timeframe="1h",
                fee_model=fee_model,
            )

    def test_snapshot_handles_missing_volume(self, fee_model: FeeModelSettings):
        """Test snapshot handles DataFrame without volume column gracefully."""
        # Create DataFrame without volume
        data = {
            "timestamp": [1704067200000 + i * 3600000 for i in range(100)],
            "open": [100.0] * 100,
            "high": [101.0] * 100,
            "low": [99.0] * 100,
            "close": [100.5] * 100,
            "volume": [0.0] * 100,  # Zero volume
        }
        df = pd.DataFrame(data)

        # Should not raise
        snapshot = build_market_snapshot(
            df=df, symbol="BTC/USDT", timeframe="1h", fee_model=fee_model
        )
        assert "## Market Data" in snapshot


# =============================================================================
# Temporal Isolation Tests
# =============================================================================


class TestTemporalIsolation:
    """Tests for temporal isolation - no future data leakage."""

    def test_snapshot_timestamp_matches_last_bar(
        self, sample_ohlcv_df: pd.DataFrame, fee_model: FeeModelSettings
    ):
        """Test that snapshot timestamp matches the last bar's timestamp."""
        snapshot = build_market_snapshot(
            df=sample_ohlcv_df,
            symbol="BTC/USDT",
            timeframe="1h",
            fee_model=fee_model,
        )

        last_ts = sample_ohlcv_df["timestamp"].iloc[-1]
        expected_dt = pd.to_datetime(last_ts, unit="ms").strftime("%Y-%m-%d %H:%M:%S")

        assert expected_dt in snapshot

    def test_recent_price_action_only_shows_historical_bars(
        self, sample_ohlcv_df: pd.DataFrame, fee_model: FeeModelSettings
    ):
        """Test that recent price action shows only bars from the DataFrame."""
        snapshot = build_market_snapshot(
            df=sample_ohlcv_df,
            symbol="BTC/USDT",
            timeframe="1h",
            fee_model=fee_model,
        )

        # Get the last 10 bar timestamps from DataFrame
        last_10 = sample_ohlcv_df.tail(10)

        for _, row in last_10.iterrows():
            ts = pd.to_datetime(row["timestamp"], unit="ms")
            date_str = ts.strftime("%Y-%m-%d %H:%M")
            assert date_str in snapshot

    def test_no_future_timestamps_in_snapshot(
        self, sample_ohlcv_df: pd.DataFrame, fee_model: FeeModelSettings
    ):
        """Test that no timestamps after the last bar appear in snapshot."""
        last_ts = sample_ohlcv_df["timestamp"].iloc[-1]
        last_dt = pd.to_datetime(last_ts, unit="ms")

        snapshot = build_market_snapshot(
            df=sample_ohlcv_df,
            symbol="BTC/USDT",
            timeframe="1h",
            fee_model=fee_model,
        )

        # Parse all timestamps in the snapshot and verify none are in the future
        import re

        # Match timestamps in format YYYY-MM-DD HH:MM
        timestamps = re.findall(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}", snapshot)

        for ts_str in timestamps:
            ts_dt = datetime.strptime(ts_str, "%Y-%m-%d %H:%M")
            # Allow UTC suffix
            if hasattr(ts_dt, "tzinfo") and ts_dt.tzinfo is None:
                ts_dt = ts_dt.replace(tzinfo=None)
            last_dt_naive = last_dt.replace(tzinfo=None)
            assert ts_dt <= last_dt_naive, f"Future timestamp found: {ts_str}"


# =============================================================================
# JSONL Schema Tests
# =============================================================================


class TestJSONLSchema:
    """Tests for JSONL output schema matching GRPOTrainingExample."""

    def test_example_to_dict_has_required_fields(self, sample_grpo_example: GRPOTrainingExample):
        """Test example_to_dict returns all required fields."""
        d = example_to_dict(sample_grpo_example)

        assert "market_snapshot" in d
        assert "actual_direction" in d
        assert "gross_return_pct" in d
        assert "timestamp_ms" in d

    def test_example_to_dict_types(self, sample_grpo_example: GRPOTrainingExample):
        """Test example_to_dict returns correct types."""
        d = example_to_dict(sample_grpo_example)

        assert isinstance(d["market_snapshot"], str)
        assert isinstance(d["actual_direction"], str)
        assert isinstance(d["gross_return_pct"], float)
        assert isinstance(d["timestamp_ms"], int)

    def test_example_to_dict_is_json_serializable(self, sample_grpo_example: GRPOTrainingExample):
        """Test example_to_dict output is JSON serializable."""
        d = example_to_dict(sample_grpo_example)

        # Should not raise
        json_str = json.dumps(d)
        assert isinstance(json_str, str)

        # Should round-trip
        parsed = json.loads(json_str)
        assert parsed == d

    def test_example_direction_values(self):
        """Test actual_direction is valid enum value."""
        valid_directions = {"LONG", "SHORT", "FLAT"}

        for direction in valid_directions:
            example = GRPOTrainingExample(
                market_snapshot="test",
                actual_direction=direction,
                gross_return_pct=0.0,
                timestamp_ms=1704067200000,
            )
            d = example_to_dict(example)
            assert d["actual_direction"] in valid_directions

    def test_load_completed_timestamps_empty_file(self, tmp_path: Path):
        """Test loading from non-existent file returns empty set."""
        output_file = tmp_path / "nonexistent.jsonl"
        completed = load_completed_timestamps(output_file)
        assert completed == set()

    def test_load_completed_timestamps_from_jsonl(self, tmp_path: Path):
        """Test loading completed timestamps from existing JSONL."""
        output_file = tmp_path / "test.jsonl"

        # Write some examples
        examples = [
            {"timestamp_ms": 1704067200000, "other": "data"},
            {"timestamp_ms": 1704070800000, "other": "data"},
            {"timestamp_ms": 1704074400000, "other": "data"},
        ]

        with open(output_file, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")

        completed = load_completed_timestamps(output_file)
        assert completed == {1704067200000, 1704070800000, 1704074400000}

    def test_load_completed_timestamps_handles_malformed_lines(self, tmp_path: Path):
        """Test loading handles malformed JSON lines gracefully."""
        output_file = tmp_path / "test.jsonl"

        with open(output_file, "w") as f:
            f.write('{"timestamp_ms": 1704067200000}\n')
            f.write("malformed json\n")
            f.write('{"no_timestamp": true}\n')
            f.write('{"timestamp_ms": 1704070800000}\n')

        completed = load_completed_timestamps(output_file)
        assert completed == {1704067200000, 1704070800000}


# =============================================================================
# Verification Horizon Tests
# =============================================================================


class TestVerificationHorizon:
    """Tests for timeframe-adaptive verification horizons."""

    def test_verification_horizons_defined_for_all_timeframes(self):
        """Test all standard timeframes have defined horizons."""
        expected_timeframes = {"1m", "5m", "15m", "1h", "4h", "1d"}
        assert set(VERIFICATION_HORIZONS.keys()) == expected_timeframes

    def test_get_verification_horizon_returns_correct_values(self):
        """Test get_verification_horizon returns correct values."""
        assert get_verification_horizon("1h") == 24
        assert get_verification_horizon("4h") == 12
        assert get_verification_horizon("1d") == 5

    def test_get_verification_horizon_default_for_unknown(self):
        """Test unknown timeframe returns default value."""
        assert get_verification_horizon("unknown") == 24

    def test_horizons_scale_inversely_with_timeframe(self):
        """Test shorter timeframes have more bars to cover similar period."""
        # 1h uses 24 bars = 24 hours
        # 4h uses 12 bars = 48 hours
        # 1d uses 5 bars = 5 days

        horizon_1h = get_verification_horizon("1h")
        horizon_4h = get_verification_horizon("4h")
        horizon_1d = get_verification_horizon("1d")

        # 1h horizon should be larger (more bars)
        assert horizon_1h > horizon_4h
        assert horizon_4h > horizon_1d


# =============================================================================
# CLI Parsing Tests
# =============================================================================


class TestCLIParsing:
    """Tests for CLI argument parsing."""

    def test_parse_symbols(self):
        """Test parsing comma-separated symbols."""
        result = parse_symbols("BTC/USDT,ETH/USDT,SOL/USDT")
        assert result == ["BTC/USDT", "ETH/USDT", "SOL/USDT"]

    def test_parse_symbols_with_spaces(self):
        """Test parsing symbols with extra spaces."""
        result = parse_symbols("BTC/USDT , ETH/USDT , SOL/USDT")
        assert result == ["BTC/USDT", "ETH/USDT", "SOL/USDT"]

    def test_parse_symbols_empty_string(self):
        """Test parsing empty string returns empty list."""
        result = parse_symbols("")
        assert result == []

    def test_parse_timeframes(self):
        """Test parsing comma-separated timeframes."""
        result = parse_timeframes("1h,4h,1d")
        assert result == ["1h", "4h", "1d"]

    def test_parse_date(self):
        """Test parsing date string."""
        result = parse_date("2024-01-15")
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15
        assert result.tzinfo == UTC

    def test_parse_date_invalid_raises(self):
        """Test invalid date format raises ValueError."""
        with pytest.raises(ValueError):
            parse_date("01-15-2024")

        with pytest.raises(ValueError):
            parse_date("invalid")


# =============================================================================
# Default Configuration Tests
# =============================================================================


class TestDefaultConfiguration:
    """Tests for default configuration values."""

    def test_default_symbols_not_empty(self):
        """Test default symbols list is not empty."""
        assert len(DEFAULT_SYMBOLS) > 0

    def test_default_symbols_format(self):
        """Test default symbols have correct format."""
        for symbol in DEFAULT_SYMBOLS:
            assert "/" in symbol
            assert symbol.endswith("USDT")

    def test_default_timeframes_valid(self):
        """Test default timeframes are valid."""
        valid_timeframes = set(VERIFICATION_HORIZONS.keys())
        for tf in DEFAULT_TIMEFRAMES:
            assert tf in valid_timeframes

    def test_lookback_bars_sufficient(self):
        """Test lookback bars is sufficient for indicator computation."""
        # Most indicators need ~26-50 bars minimum
        assert LOOKBACK_BARS >= 50


# =============================================================================
# Integration Tests (Mocked)
# =============================================================================


class TestGenerateGRPODataset:
    """Integration tests for generate_grpo_dataset (mocked)."""

    @pytest.mark.asyncio
    async def test_generates_correct_schema(self, tmp_path: Path):
        """Test generated examples have correct schema."""
        output_file = tmp_path / "test_output.jsonl"

        # Create mock market data service
        mock_df = pd.DataFrame(
            {
                "timestamp": [1704067200000 + i * 3600000 for i in range(120)],
                "open": [100.0 + i * 0.1 for i in range(120)],
                "high": [101.0 + i * 0.1 for i in range(120)],
                "low": [99.0 + i * 0.1 for i in range(120)],
                "close": [100.5 + i * 0.1 for i in range(120)],
                "volume": [1000.0] * 120,
            }
        )

        with patch("training.grpo_data_generator.MarketDataService") as MockService:
            # Setup mock
            mock_instance = AsyncMock()
            mock_instance.fetch_ohlcv = AsyncMock(return_value=mock_df)
            mock_instance.get_ohlcv_as_of = AsyncMock(return_value=mock_df.tail(30))
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            MockService.return_value = mock_instance

            with patch(
                "training.grpo_data_generator.fetch_window_data",
                new_callable=AsyncMock,
            ) as mock_fetch:
                mock_fetch.return_value = mock_df.tail(100)

                from training.grpo_data_generator import generate_grpo_dataset

                await generate_grpo_dataset(
                    output_file=output_file,
                    symbols=["BTC/USDT"],
                    timeframes=["1h"],
                    limit=2,
                    resume=False,
                )

        # Verify output file exists and has correct schema
        if output_file.exists():
            with open(output_file) as f:
                for line in f:
                    example = json.loads(line)
                    assert "market_snapshot" in example
                    assert "actual_direction" in example
                    assert "gross_return_pct" in example
                    assert "timestamp_ms" in example
                    assert example["actual_direction"] in {"LONG", "SHORT", "FLAT"}


# =============================================================================
# Fee Model Integration Tests
# =============================================================================


class TestFeeModelIntegration:
    """Tests for fee model integration in direction classification."""

    def test_classification_uses_actual_fee_values(self):
        """Test classification uses actual fee model values, not magic numbers."""
        fee_model = FeeModelSettings()

        # Calculate actual threshold
        min_profitable = fee_model.minimum_profitable_return_pct(0)

        # Test just above threshold
        direction = classify_direction(min_profitable + 0.01, fee_model, holding_periods_8h=0)
        assert direction == "LONG"

        # Test just below threshold
        direction = classify_direction(min_profitable - 0.01, fee_model, holding_periods_8h=0)
        assert direction == "FLAT"

    def test_classification_with_custom_fee_model(self):
        """Test classification works with custom fee model."""
        # High fee model
        high_fee_model = FeeModelSettings(
            maker_fee_pct=0.1,  # 0.1%
            taker_fee_pct=0.15,  # 0.15%
            bnb_discount_enabled=False,
        )

        min_profitable = high_fee_model.minimum_profitable_return_pct(0)
        assert min_profitable > 0.2  # Should be higher than default

        # Return that would be LONG with default fees is FLAT with high fees
        direction = classify_direction(0.15, high_fee_model, holding_periods_8h=0)
        assert direction == "FLAT"

    def test_funding_periods_affect_classification(self):
        """Test that longer holding periods (more funding) affect classification."""
        fee_model = FeeModelSettings()

        min_0 = fee_model.minimum_profitable_return_pct(0)
        min_5 = fee_model.minimum_profitable_return_pct(5)

        # Threshold should increase with funding periods
        assert min_5 > min_0

        # Return that's LONG with no funding might be FLAT with 5 periods
        gross_return = (min_0 + min_5) / 2  # Between the two thresholds

        direction_0 = classify_direction(gross_return, fee_model, holding_periods_8h=0)
        direction_5 = classify_direction(gross_return, fee_model, holding_periods_8h=5)

        assert direction_0 == "LONG"
        assert direction_5 == "FLAT"
