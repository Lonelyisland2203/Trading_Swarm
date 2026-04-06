"""Tests for ExecutionSettings configuration."""

import os
from pathlib import Path

import pytest

from config.settings import AppSettings, ExecutionSettings


class TestExecutionSettings:
    """Test the ExecutionSettings model directly."""

    def test_default_values(self):
        """Verify all default values are set correctly."""
        settings = ExecutionSettings()

        assert settings.testnet is True
        assert settings.max_daily_trades == 10
        assert settings.max_daily_loss_pct == 2.0
        assert settings.max_open_positions == 3
        assert settings.max_position_pct == 0.02
        assert settings.order_cooldown_seconds == 60
        assert settings.min_confidence == 0.6

    def test_testnet_default_is_true(self):
        """Verify testnet defaults to True (safety first)."""
        settings = ExecutionSettings()
        assert settings.testnet is True

    def test_max_position_pct_capped(self):
        """Verify max_position_pct cannot exceed 0.10 (10%)."""
        # Should work at the cap
        settings = ExecutionSettings(max_position_pct=0.10)
        assert settings.max_position_pct == 0.10

        # Should fail above the cap
        with pytest.raises(ValueError):
            ExecutionSettings(max_position_pct=0.11)

    def test_max_daily_loss_pct_minimum(self):
        """Verify max_daily_loss_pct must be at least 0.1%."""
        # Should work at minimum
        settings = ExecutionSettings(max_daily_loss_pct=0.1)
        assert settings.max_daily_loss_pct == 0.1

        # Should fail below minimum
        with pytest.raises(ValueError):
            ExecutionSettings(max_daily_loss_pct=0.05)

    def test_state_dir_default(self):
        """Verify state_dir defaults to Path('execution/state')."""
        settings = ExecutionSettings()
        assert settings.state_dir == Path("execution/state")
        assert isinstance(settings.state_dir, Path)


class TestExecutionSettingsInAppSettings:
    """Test integration with AppSettings."""

    def test_execution_in_app_settings(self):
        """Verify AppSettings has 'execution' attribute with testnet=True."""
        settings = AppSettings()
        assert hasattr(settings, "execution")
        assert isinstance(settings.execution, ExecutionSettings)
        assert settings.execution.testnet is True

    def test_env_var_override(self):
        """Verify environment variables override defaults."""
        # Set env vars
        os.environ["EXECUTION_MAX_DAILY_TRADES"] = "20"
        os.environ["EXECUTION_MAX_POSITION_PCT"] = "0.05"

        try:
            # Create settings with env vars
            settings = AppSettings()

            # Verify overrides
            assert settings.execution.max_daily_trades == 20
            assert settings.execution.max_position_pct == 0.05
        finally:
            # Cleanup
            os.environ.pop("EXECUTION_MAX_DAILY_TRADES", None)
            os.environ.pop("EXECUTION_MAX_POSITION_PCT", None)
