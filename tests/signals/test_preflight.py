"""Tests for preflight checks."""

import os
from unittest.mock import patch

import pytest

from signals.preflight import (
    check_stop_file,
    run_preflight_checks,
    enforce_ollama_keep_alive,
)
from training.vram_check import VRAMStatus
from utils.stop_file import StopFileChecker


class TestCheckStopFile:
    """Tests for STOP file check."""

    def test_stop_file_not_exists(self, tmp_path):
        """Returns False when STOP file doesn't exist."""
        # Create checker with non-existent path
        fake_checker = StopFileChecker(tmp_path / "STOP")

        with patch("utils.stop_file.default_stop_checker", fake_checker):
            assert check_stop_file() is False

    def test_stop_file_exists(self, tmp_path):
        """Returns True when STOP file exists."""
        # Create STOP file
        stop_file = tmp_path / "STOP"
        stop_file.touch()
        fake_checker = StopFileChecker(stop_file)

        with patch("utils.stop_file.default_stop_checker", fake_checker):
            assert check_stop_file() is True


class TestRunPreflightChecks:
    """Tests for run_preflight_checks function."""

    @pytest.fixture
    def mock_can_infer(self):
        """Mock check_can_infer."""
        with patch("signals.preflight.check_can_infer") as mock:
            mock.return_value = (True, "Ready to infer")
            yield mock

    @pytest.fixture
    def mock_vram_check(self):
        """Mock check_vram_availability."""
        with patch("signals.preflight.check_vram_availability") as mock:
            mock.return_value = VRAMStatus(
                total_mb=16384,
                used_mb=4096,
                free_mb=12288,
                gpu_name="Test GPU",
                can_train=True,
                reason="Sufficient VRAM",
            )
            yield mock

    @pytest.fixture
    def no_stop_file(self, tmp_path):
        """Ensure STOP file doesn't exist."""
        fake_checker = StopFileChecker(tmp_path / "STOP")
        with patch("utils.stop_file.default_stop_checker", fake_checker):
            yield tmp_path / "STOP"

    def test_all_checks_pass(self, no_stop_file, mock_can_infer, mock_vram_check):
        """All preflight checks pass."""
        result = run_preflight_checks()

        assert result.passed is True
        assert "passed" in result.reason.lower()
        assert result.vram_status is not None

    def test_stop_file_blocks(self, tmp_path, mock_can_infer, mock_vram_check):
        """STOP file blocks preflight."""
        stop_file = tmp_path / "STOP"
        stop_file.touch()
        fake_checker = StopFileChecker(stop_file)

        with patch("utils.stop_file.default_stop_checker", fake_checker):
            result = run_preflight_checks()

        assert result.passed is False
        assert "STOP" in result.reason

    def test_process_lock_blocks(self, no_stop_file, mock_vram_check):
        """Process lock blocks preflight."""
        with patch("signals.preflight.check_can_infer") as mock:
            mock.return_value = (False, "Training process is running")

            result = run_preflight_checks()

            assert result.passed is False
            assert "lock" in result.reason.lower() or "Training" in result.reason

    def test_vram_insufficient_blocks(self, no_stop_file, mock_can_infer):
        """Insufficient VRAM blocks preflight."""
        with patch("signals.preflight.check_vram_availability") as mock:
            mock.return_value = VRAMStatus(
                total_mb=8192,
                used_mb=7168,
                free_mb=1024,
                gpu_name="Test GPU",
                can_train=False,
                reason="Insufficient VRAM: 1.0 GB free",
            )

            result = run_preflight_checks()

            assert result.passed is False
            assert "VRAM" in result.reason

    def test_skip_vram_check(self, no_stop_file, mock_can_infer):
        """VRAM check can be skipped."""
        result = run_preflight_checks(skip_vram_check=True)

        assert result.passed is True
        assert "skipped" in result.reason.lower()
        assert result.vram_status is None


class TestEnforceOllamaKeepAlive:
    """Tests for OLLAMA_KEEP_ALIVE enforcement."""

    def test_sets_keep_alive_to_zero(self, monkeypatch):
        """Sets OLLAMA_KEEP_ALIVE to 0."""
        monkeypatch.delenv("OLLAMA_KEEP_ALIVE", raising=False)

        enforce_ollama_keep_alive()

        assert os.environ.get("OLLAMA_KEEP_ALIVE") == "0"

    def test_overwrites_existing_value(self, monkeypatch):
        """Overwrites existing non-zero value."""
        monkeypatch.setenv("OLLAMA_KEEP_ALIVE", "300")

        enforce_ollama_keep_alive()

        assert os.environ.get("OLLAMA_KEEP_ALIVE") == "0"

    def test_no_change_if_already_zero(self, monkeypatch):
        """No change needed if already 0."""
        monkeypatch.setenv("OLLAMA_KEEP_ALIVE", "0")

        enforce_ollama_keep_alive()

        assert os.environ.get("OLLAMA_KEEP_ALIVE") == "0"
