"""Tests for VRAM availability checking."""

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from training.vram_check import VRAMStatus, check_vram_availability, log_vram_status


class TestCheckVRAMAvailability:
    """Test VRAM availability checking."""

    @patch("subprocess.run")
    def test_nvidia_gpu_sufficient_vram(self, mock_run):
        """Test NVIDIA GPU with sufficient VRAM."""
        # Mock nvidia-smi output: 16384 MB total, 2048 MB used, 14336 MB free
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="16384, 2048, 14336, NVIDIA GeForce RTX 5070 Ti\n",
        )

        status = check_vram_availability(min_free_gb=9.0)

        assert status.total_mb == 16384
        assert status.used_mb == 2048
        assert status.free_mb == 14336
        assert status.gpu_name == "NVIDIA GeForce RTX 5070 Ti"
        assert status.can_train
        assert "14.0 GB free" in status.reason

    @patch("subprocess.run")
    def test_nvidia_gpu_insufficient_vram(self, mock_run):
        """Test NVIDIA GPU with insufficient VRAM."""
        # Mock nvidia-smi output: 16384 MB total, 12288 MB used, 4096 MB free
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="16384, 12288, 4096, NVIDIA GeForce RTX 5070 Ti\n",
        )

        status = check_vram_availability(min_free_gb=9.0)

        assert status.total_mb == 16384
        assert status.used_mb == 12288
        assert status.free_mb == 4096
        assert status.gpu_name == "NVIDIA GeForce RTX 5070 Ti"
        assert not status.can_train
        assert "Insufficient VRAM" in status.reason
        assert "4.0 GB free" in status.reason

    @patch("subprocess.run")
    def test_nvidia_gpu_custom_threshold(self, mock_run):
        """Test NVIDIA GPU with custom minimum threshold."""
        # Mock nvidia-smi output: 8192 MB free
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="24576, 16384, 8192, NVIDIA RTX 4090\n",
        )

        # Should pass with 6 GB threshold
        status = check_vram_availability(min_free_gb=6.0)
        assert status.can_train

        # Should fail with 10 GB threshold
        status = check_vram_availability(min_free_gb=10.0)
        assert not status.can_train

    @patch("subprocess.run")
    def test_nvidia_smi_not_found(self, mock_run):
        """Test when nvidia-smi is not available."""
        # First call (nvidia-smi) raises FileNotFoundError
        # Second call (sysctl hw.model) returns non-Mac
        # Third call should not happen
        mock_run.side_effect = [
            FileNotFoundError("nvidia-smi not found"),
            MagicMock(returncode=0, stdout="GenericPC\n"),
        ]

        status = check_vram_availability()

        assert status.total_mb == 0
        assert status.free_mb == 0
        assert status.gpu_name == "Unknown"
        assert not status.can_train
        assert "Cannot determine VRAM" in status.reason

    @patch("subprocess.run")
    def test_apple_silicon_sufficient_memory(self, mock_run):
        """Test Apple Silicon with sufficient unified memory."""
        # Mock sysctl outputs for Apple Silicon
        mock_run.side_effect = [
            FileNotFoundError("nvidia-smi not found"),  # No NVIDIA GPU
            MagicMock(returncode=0, stdout="Mac14,2\n"),  # hw.model
            MagicMock(returncode=0, stdout="68719476736\n"),  # hw.memsize (64 GB)
        ]

        status = check_vram_availability(min_free_gb=9.0)

        # 64 GB total -> ~32 GB estimated GPU -> ~24 GB estimated free
        assert status.gpu_name == "Apple Silicon (unified memory)"
        assert status.can_train
        assert "Apple Silicon detected" in status.reason

    @patch("subprocess.run")
    def test_apple_silicon_insufficient_memory(self, mock_run):
        """Test Apple Silicon with insufficient unified memory."""
        # Mock sysctl outputs for Apple Silicon with only 16 GB RAM
        mock_run.side_effect = [
            FileNotFoundError("nvidia-smi not found"),
            MagicMock(returncode=0, stdout="Mac14,2\n"),
            MagicMock(returncode=0, stdout="17179869184\n"),  # hw.memsize (16 GB)
        ]

        status = check_vram_availability(min_free_gb=9.0)

        # 16 GB total -> ~8 GB estimated GPU -> ~6 GB estimated free
        assert status.gpu_name == "Apple Silicon (unified memory)"
        assert not status.can_train
        assert "need 9.0 GB minimum" in status.reason

    @patch("subprocess.run")
    def test_nvidia_smi_timeout(self, mock_run):
        """Test nvidia-smi timeout."""
        mock_run.side_effect = [
            subprocess.TimeoutExpired("nvidia-smi", 5),
            MagicMock(returncode=1, stdout=""),  # sysctl fails
        ]

        status = check_vram_availability()

        assert not status.can_train
        assert "Cannot determine VRAM" in status.reason

    @patch("subprocess.run")
    def test_nvidia_smi_error(self, mock_run):
        """Test nvidia-smi returning error."""
        mock_run.side_effect = [
            MagicMock(returncode=1, stdout="Error: No GPU found\n"),
            MagicMock(returncode=1, stdout=""),
        ]

        status = check_vram_availability()

        assert not status.can_train
        assert "Cannot determine VRAM" in status.reason

    @patch("subprocess.run")
    def test_log_vram_status_success(self, mock_run, caplog):
        """Test log_vram_status with successful detection."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="16384, 2048, 14336, NVIDIA GeForce RTX 5070 Ti\n",
        )

        status = log_vram_status()

        assert status.can_train
        # Check that logging occurred (caplog captures loguru output)

    @patch("subprocess.run")
    def test_log_vram_status_insufficient(self, mock_run, caplog):
        """Test log_vram_status with insufficient VRAM."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="16384, 14336, 2048, NVIDIA GeForce RTX 5070 Ti\n",
        )

        status = log_vram_status()

        assert not status.can_train
        # Warning should be logged

    def test_vram_status_is_frozen(self):
        """Test VRAMStatus is immutable (frozen dataclass)."""
        status = VRAMStatus(
            total_mb=16384,
            used_mb=2048,
            free_mb=14336,
            gpu_name="Test GPU",
            can_train=True,
            reason="Test reason",
        )

        with pytest.raises(AttributeError):
            status.can_train = False  # Should fail - frozen dataclass
