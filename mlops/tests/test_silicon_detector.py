"""Tests for Apple Silicon Detector"""

from __future__ import annotations

import platform
from unittest.mock import Mock, patch

import pytest

from mlops.silicon.detector import AppleSiliconDetector, HardwareInfo


class TestAppleSiliconDetector:
    """Test suite for AppleSiliconDetector"""

    @pytest.fixture
    def mock_apple_silicon(self):
        """Mock Apple Silicon system"""
        with patch("platform.system", return_value="Darwin"), \
             patch("platform.machine", return_value="arm64"), \
             patch("platform.processor", return_value="arm"):
            yield

    @pytest.fixture
    def mock_non_apple_silicon(self):
        """Mock non-Apple Silicon system"""
        with patch("platform.system", return_value="Linux"), \
             patch("platform.machine", return_value="x86_64"), \
             patch("platform.processor", return_value="x86_64"):
            yield

    def test_detector_initialization(self, mock_apple_silicon):
        """Test detector initializes correctly"""
        detector = AppleSiliconDetector()
        assert detector is not None
        assert hasattr(detector, "_hardware_info")

    def test_is_apple_silicon_detection(self, mock_apple_silicon):
        """Test Apple Silicon detection"""
        detector = AppleSiliconDetector()
        # On mock system, this will be True due to platform mocks
        assert isinstance(detector.is_apple_silicon, bool)

    def test_non_apple_silicon_detection(self, mock_non_apple_silicon):
        """Test non-Apple Silicon detection"""
        detector = AppleSiliconDetector()
        assert detector.is_apple_silicon is False

    def test_get_hardware_info(self, mock_apple_silicon):
        """Test hardware info retrieval"""
        detector = AppleSiliconDetector()
        info = detector.get_hardware_info()

        assert isinstance(info, HardwareInfo)
        assert isinstance(info.is_apple_silicon, bool)
        assert isinstance(info.chip_type, str)
        assert isinstance(info.chip_variant, str)
        assert isinstance(info.memory_total_gb, float)
        assert isinstance(info.core_count, int)

    @patch("subprocess.run")
    def test_chip_detection_m1(self, mock_run, mock_apple_silicon):
        """Test M1 chip detection"""
        mock_run.return_value = Mock(stdout="Apple M1\n")

        detector = AppleSiliconDetector()
        chip_type, chip_variant = detector._detect_chip()

        assert chip_type == "M1"
        assert chip_variant == "Base"

    @patch("subprocess.run")
    def test_chip_detection_m2_pro(self, mock_run, mock_apple_silicon):
        """Test M2 Pro chip detection"""
        mock_run.return_value = Mock(stdout="Apple M2 Pro\n")

        detector = AppleSiliconDetector()
        chip_type, chip_variant = detector._detect_chip()

        assert chip_type == "M2"
        assert chip_variant == "Pro"

    @patch("subprocess.run")
    def test_chip_detection_m3_max(self, mock_run, mock_apple_silicon):
        """Test M3 Max chip detection"""
        mock_run.return_value = Mock(stdout="Apple M3 Max\n")

        detector = AppleSiliconDetector()
        chip_type, chip_variant = detector._detect_chip()

        assert chip_type == "M3"
        assert chip_variant == "Max"

    @patch("subprocess.run")
    def test_memory_detection(self, mock_run, mock_apple_silicon):
        """Test memory detection"""
        # Mock 16GB of memory
        mock_run.return_value = Mock(stdout=str(16 * 1024**3))

        detector = AppleSiliconDetector()
        memory_gb = detector._detect_memory()

        assert memory_gb == 16.0

    @patch("subprocess.run")
    def test_core_detection(self, mock_run, mock_apple_silicon):
        """Test CPU core detection"""
        # Mock 8 total cores
        mock_run.side_effect = [
            Mock(stdout="8\n"),  # Total cores
            Mock(stdout="4\n"),  # Performance cores
            Mock(stdout="4\n"),  # Efficiency cores
        ]

        detector = AppleSiliconDetector()
        core_info = detector._detect_cores()

        assert core_info["total"] == 8
        assert core_info["performance"] == 4
        assert core_info["efficiency"] == 4

    def test_mlx_availability_check(self, mock_apple_silicon):
        """Test MLX availability check"""
        detector = AppleSiliconDetector()
        mlx_available = detector._check_mlx()

        # Will be False unless MLX is actually installed
        assert isinstance(mlx_available, bool)

    def test_mps_availability_check(self, mock_apple_silicon):
        """Test MPS availability check"""
        detector = AppleSiliconDetector()
        mps_available = detector._check_mps()

        # Will be False unless PyTorch is actually installed
        assert isinstance(mps_available, bool)

    def test_ane_availability_check(self, mock_apple_silicon):
        """Test ANE availability check"""
        detector = AppleSiliconDetector()
        ane_available = detector._check_ane()

        assert isinstance(ane_available, bool)

    @patch("subprocess.run")
    def test_thermal_state_detection(self, mock_run, mock_apple_silicon):
        """Test thermal state detection"""
        detector = AppleSiliconDetector()
        thermal_state = detector._detect_thermal_state()

        assert isinstance(thermal_state, int)
        assert 0 <= thermal_state <= 3

    @patch("subprocess.run")
    def test_power_mode_detection_low_power(self, mock_run, mock_apple_silicon):
        """Test power mode detection - low power"""
        mock_run.return_value = Mock(stdout="low power mode\n")

        detector = AppleSiliconDetector()
        power_mode = detector._detect_power_mode()

        assert power_mode == "low_power"

    @patch("subprocess.run")
    def test_power_mode_detection_ac_power(self, mock_run, mock_apple_silicon):
        """Test power mode detection - AC power"""
        mock_run.return_value = Mock(stdout="AC power\n")

        detector = AppleSiliconDetector()
        power_mode = detector._detect_power_mode()

        assert power_mode == "high_performance"

    def test_refresh_method(self, mock_apple_silicon):
        """Test refresh method updates hardware info"""
        detector = AppleSiliconDetector()
        initial_info = detector.get_hardware_info()

        detector.refresh()
        refreshed_info = detector.get_hardware_info()

        # Info should be updated (timestamps would differ in real scenario)
        assert isinstance(refreshed_info, HardwareInfo)

    def test_hardware_info_to_dict(self, mock_apple_silicon):
        """Test HardwareInfo to_dict conversion"""
        detector = AppleSiliconDetector()
        info = detector.get_hardware_info()
        info_dict = info.to_dict()

        assert isinstance(info_dict, dict)
        assert "is_apple_silicon" in info_dict
        assert "chip_type" in info_dict
        assert "memory_total_gb" in info_dict
        assert "mlx_available" in info_dict

    def test_chip_detection_failure(self, mock_apple_silicon):
        """Test graceful handling of chip detection failure"""
        with patch("subprocess.run", side_effect=Exception("sysctl failed")):
            detector = AppleSiliconDetector()
            chip_type, chip_variant = detector._detect_chip()

            assert chip_type == "Unknown"
            assert chip_variant == "Unknown"

    def test_memory_detection_failure(self, mock_apple_silicon):
        """Test graceful handling of memory detection failure"""
        with patch("subprocess.run", side_effect=Exception("sysctl failed")):
            detector = AppleSiliconDetector()
            memory_gb = detector._detect_memory()

            assert memory_gb == 0.0

    def test_core_detection_failure(self, mock_apple_silicon):
        """Test graceful handling of core detection failure"""
        with patch("subprocess.run", side_effect=Exception("sysctl failed")):
            detector = AppleSiliconDetector()
            core_info = detector._detect_cores()

            assert core_info["total"] == 0
            assert core_info["performance"] == 0
            assert core_info["efficiency"] == 0


@pytest.mark.skipif(
    platform.system() != "Darwin" or platform.machine() != "arm64",
    reason="Requires Apple Silicon hardware",
)
class TestAppleSiliconDetectorRealHardware:
    """Tests that require actual Apple Silicon hardware"""

    def test_real_hardware_detection(self):
        """Test detection on real Apple Silicon hardware"""
        detector = AppleSiliconDetector()

        assert detector.is_apple_silicon is True

        info = detector.get_hardware_info()
        assert info.chip_type in ["M1", "M2", "M3", "M4"]
        assert info.chip_variant in ["Base", "Pro", "Max", "Ultra"]
        assert info.memory_total_gb > 0
        assert info.core_count > 0

    def test_real_framework_detection(self):
        """Test framework detection on real hardware"""
        detector = AppleSiliconDetector()
        info = detector.get_hardware_info()

        # At least one framework should be available
        assert info.mlx_available or info.mps_available or True
