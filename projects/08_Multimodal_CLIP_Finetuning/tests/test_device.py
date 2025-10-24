"""Tests for DeviceManager."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from device_manager import DeviceManager


class TestDeviceManager:
    """Test suite for DeviceManager."""

    def test_initialization(self):
        """Test device manager initialization."""
        dm = DeviceManager(use_mps=True)
        assert dm.use_mps is True

        dm = DeviceManager(use_mps=False)
        assert dm.use_mps is False

    def test_device_property_caching(self):
        """Test that device property is cached."""
        dm = DeviceManager(use_mps=False)

        device1 = dm.device
        device2 = dm.device

        # Should return same instance
        assert device1 is device2

    @pytest.mark.apple_silicon
    def test_apple_silicon_detection_on_apple_silicon(self):
        """Test Apple Silicon detection on actual Apple Silicon hardware.

        This test only runs on Apple Silicon hardware.
        """
        dm = DeviceManager()

        # On Apple Silicon, this should return True
        assert dm.is_apple_silicon is True

    def test_apple_silicon_detection_mocked_non_mac(self):
        """Test Apple Silicon detection on non-macOS systems."""
        with patch("platform.system", return_value="Linux"):
            dm = DeviceManager()
            assert dm.is_apple_silicon is False

    def test_apple_silicon_detection_mocked_intel_mac(self):
        """Test Apple Silicon detection on Intel-based Mac."""
        with (
            patch("platform.system", return_value="Darwin"),
            patch("platform.machine", return_value="x86_64"),
        ):
            dm = DeviceManager()
            assert dm.is_apple_silicon is False

    @pytest.mark.apple_silicon
    def test_mps_availability_on_apple_silicon(self):
        """Test MPS availability check on Apple Silicon.

        This test only runs on Apple Silicon hardware with MPS support.
        """
        dm = DeviceManager()

        # On Apple Silicon with PyTorch MPS, this should return True
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            assert dm.is_mps_available is True
        else:
            # If MPS not available, should gracefully return False
            assert dm.is_mps_available is False

    def test_mps_availability_mocked_unavailable(self):
        """Test MPS availability when PyTorch doesn't support MPS."""
        # Create a mock torch.backends without mps attribute
        mock_backends = MagicMock()
        del mock_backends.mps

        with patch("torch.backends", mock_backends):
            dm = DeviceManager()
            assert dm.is_mps_available is False

    def test_device_selection_cpu_disabled_mps(self):
        """Test device selection when MPS is disabled."""
        dm = DeviceManager(use_mps=False)
        device = dm.device

        assert device.type == "cpu"

    @pytest.mark.apple_silicon
    def test_device_selection_mps_on_apple_silicon(self):
        """Test device selection on Apple Silicon with MPS enabled.

        This test only runs on Apple Silicon hardware.
        """
        dm = DeviceManager(use_mps=True)
        device = dm.device

        # Should select MPS if available, otherwise CPU
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            assert device.type == "mps"
        else:
            assert device.type == "cpu"

    def test_device_selection_fallback_to_cpu(self):
        """Test device selection falls back to CPU when MPS unavailable."""
        # Mock MPS as unavailable
        with patch.object(DeviceManager, "is_mps_available", False):
            dm = DeviceManager(use_mps=True)
            device = dm.device

            assert device.type == "cpu"

    def test_get_memory_info_cpu(self):
        """Test memory info retrieval for CPU device."""
        dm = DeviceManager(use_mps=False)
        memory_info = dm.get_memory_info()

        assert memory_info["device_type"] == "cpu"
        assert "note" in memory_info

    @pytest.mark.apple_silicon
    def test_get_memory_info_mps(self):
        """Test memory info retrieval for MPS device.

        This test only runs on Apple Silicon hardware with MPS.
        """
        dm = DeviceManager(use_mps=True)

        # Only test if MPS is actually available
        if dm.device.type == "mps":
            memory_info = dm.get_memory_info()

            assert memory_info["device_type"] == "mps"
            assert "unified memory" in memory_info["note"]

    def test_log_device_info(self, caplog):
        """Test device info logging."""
        import logging

        caplog.set_level(logging.INFO)

        dm = DeviceManager(use_mps=False)
        dm.log_device_info()

        # Check that device info was logged
        assert any("Device Configuration" in record.message for record in caplog.records)
        assert any("Selected Device" in record.message for record in caplog.records)

    def test_optimize_for_device_cpu(self, caplog):
        """Test device optimization for CPU."""
        import logging

        caplog.set_level(logging.INFO)

        dm = DeviceManager(use_mps=False)
        dm.optimize_for_device()

        # Should log that no optimizations for CPU
        assert any("CPU" in record.message for record in caplog.records)

    @pytest.mark.apple_silicon
    def test_optimize_for_device_mps(self, caplog):
        """Test device optimization for MPS.

        This test only runs on Apple Silicon hardware with MPS.
        """
        import logging

        caplog.set_level(logging.INFO)

        dm = DeviceManager(use_mps=True)

        # Only test if MPS is actually available
        if dm.device.type == "mps":
            dm.optimize_for_device()

            # Should log MPS-specific optimizations
            assert any("MPS" in record.message for record in caplog.records)
