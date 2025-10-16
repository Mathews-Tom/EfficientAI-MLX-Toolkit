"""
Tests for Apple Silicon metrics collection.

This module tests the Apple Silicon metrics collection utilities.
"""

import platform
from unittest.mock import MagicMock, Mock, patch

import pytest

from mlops.tracking import (
    AppleSiliconMetrics,
    AppleSiliconMetricsError,
    collect_metrics,
    detect_apple_silicon,
    log_metrics_to_mlflow,
)


class TestAppleSiliconMetrics:
    """Test AppleSiliconMetrics dataclass."""

    def test_initialization(self):
        """Test metrics dataclass initialization."""
        metrics = AppleSiliconMetrics(
            chip_type="M2",
            memory_total_gb=16.0,
            memory_used_gb=8.0,
            memory_available_gb=8.0,
            memory_utilization_percent=50.0,
            mps_available=True,
            mps_utilization_percent=75.0,
            ane_available=True,
            thermal_state=0,
            power_mode="normal",
            core_count=8,
            performance_core_count=4,
            efficiency_core_count=4,
        )

        assert metrics.chip_type == "M2"
        assert metrics.memory_total_gb == 16.0
        assert metrics.mps_available is True
        assert metrics.ane_available is True

    def test_to_dict(self):
        """Test conversion to dictionary."""
        metrics = AppleSiliconMetrics(
            chip_type="M2",
            memory_total_gb=16.0,
            memory_used_gb=8.0,
            memory_available_gb=8.0,
            memory_utilization_percent=50.0,
            mps_available=True,
            ane_available=True,
        )

        metrics_dict = metrics.to_dict()

        assert isinstance(metrics_dict, dict)
        assert metrics_dict["chip_type"] == "M2"
        assert metrics_dict["memory_total_gb"] == 16.0
        assert metrics_dict["mps_available"] is True

    def test_to_mlflow_metrics(self):
        """Test conversion to MLFlow metrics format."""
        metrics = AppleSiliconMetrics(
            chip_type="M2",
            memory_total_gb=16.0,
            memory_used_gb=8.0,
            memory_available_gb=8.0,
            memory_utilization_percent=50.0,
            mps_available=True,
            mps_utilization_percent=75.0,
            ane_available=True,
        )

        mlflow_metrics = metrics.to_mlflow_metrics()

        assert isinstance(mlflow_metrics, dict)
        # Check all values are numeric
        assert all(isinstance(v, (int, float)) for v in mlflow_metrics.values())
        # Check boolean conversion
        assert mlflow_metrics["mps_available"] == 1.0
        assert mlflow_metrics["ane_available"] == 1.0
        # Check MPS utilization included
        assert mlflow_metrics["mps_utilization_percent"] == 75.0

    def test_to_mlflow_metrics_no_mps_utilization(self):
        """Test MLFlow metrics without MPS utilization."""
        metrics = AppleSiliconMetrics(
            chip_type="M2",
            memory_total_gb=16.0,
            memory_used_gb=8.0,
            memory_available_gb=8.0,
            memory_utilization_percent=50.0,
            mps_available=False,
            mps_utilization_percent=None,
            ane_available=True,
        )

        mlflow_metrics = metrics.to_mlflow_metrics()

        assert "mps_utilization_percent" not in mlflow_metrics
        assert mlflow_metrics["mps_available"] == 0.0


class TestDetectAppleSilicon:
    """Test Apple Silicon detection."""

    @patch("mlops.tracking.apple_silicon_metrics.platform.system")
    @patch("mlops.tracking.apple_silicon_metrics.platform.machine")
    @patch("mlops.tracking.apple_silicon_metrics.platform.processor")
    def test_detect_apple_silicon_true(self, mock_processor, mock_machine, mock_system):
        """Test detection on Apple Silicon."""
        mock_system.return_value = "Darwin"
        mock_machine.return_value = "arm64"
        mock_processor.return_value = "arm"

        assert detect_apple_silicon() is True

    @patch("mlops.tracking.apple_silicon_metrics.platform.system")
    @patch("mlops.tracking.apple_silicon_metrics.platform.machine")
    def test_detect_apple_silicon_false_system(self, mock_machine, mock_system):
        """Test detection on non-Darwin system."""
        mock_system.return_value = "Linux"
        mock_machine.return_value = "arm64"

        assert detect_apple_silicon() is False

    @patch("mlops.tracking.apple_silicon_metrics.platform.system")
    @patch("mlops.tracking.apple_silicon_metrics.platform.machine")
    def test_detect_apple_silicon_false_machine(self, mock_machine, mock_system):
        """Test detection on non-ARM machine."""
        mock_system.return_value = "Darwin"
        mock_machine.return_value = "x86_64"

        assert detect_apple_silicon() is False

    @patch("mlops.tracking.apple_silicon_metrics.platform.system")
    def test_detect_apple_silicon_exception(self, mock_system):
        """Test detection with exception."""
        mock_system.side_effect = Exception("Test error")

        assert detect_apple_silicon() is False


class TestChipTypeDetection:
    """Test chip type detection."""

    @patch("mlops.tracking.apple_silicon_metrics.subprocess.run")
    def test_get_chip_type_m1(self, mock_run):
        """Test M1 chip type detection."""
        from mlops.tracking.apple_silicon_metrics import get_chip_type

        mock_run.return_value = Mock(stdout="Apple M1\n")

        chip_type = get_chip_type()

        assert chip_type == "M1"

    @patch("mlops.tracking.apple_silicon_metrics.subprocess.run")
    def test_get_chip_type_m1_pro(self, mock_run):
        """Test M1 Pro chip type detection."""
        from mlops.tracking.apple_silicon_metrics import get_chip_type

        mock_run.return_value = Mock(stdout="Apple M1 Pro\n")

        chip_type = get_chip_type()

        assert chip_type == "M1 Pro"

    @patch("mlops.tracking.apple_silicon_metrics.subprocess.run")
    def test_get_chip_type_m2_max(self, mock_run):
        """Test M2 Max chip type detection."""
        from mlops.tracking.apple_silicon_metrics import get_chip_type

        mock_run.return_value = Mock(stdout="Apple M2 Max\n")

        chip_type = get_chip_type()

        assert chip_type == "M2 Max"

    @patch("mlops.tracking.apple_silicon_metrics.subprocess.run")
    def test_get_chip_type_m3(self, mock_run):
        """Test M3 chip type detection."""
        from mlops.tracking.apple_silicon_metrics import get_chip_type

        mock_run.return_value = Mock(stdout="Apple M3\n")

        chip_type = get_chip_type()

        assert chip_type == "M3"

    @patch("mlops.tracking.apple_silicon_metrics.subprocess.run")
    def test_get_chip_type_unknown(self, mock_run):
        """Test unknown chip type."""
        from mlops.tracking.apple_silicon_metrics import get_chip_type

        mock_run.return_value = Mock(stdout="Some other chip\n")

        chip_type = get_chip_type()

        assert chip_type == "Unknown Apple Silicon"

    @patch("mlops.tracking.apple_silicon_metrics.subprocess.run")
    def test_get_chip_type_error(self, mock_run):
        """Test chip type detection error."""
        from mlops.tracking.apple_silicon_metrics import get_chip_type

        mock_run.side_effect = Exception("Test error")

        chip_type = get_chip_type()

        assert chip_type == "Unknown"


class TestMemoryInfo:
    """Test memory information collection."""

    @patch("mlops.tracking.apple_silicon_metrics.subprocess.run")
    def test_get_memory_info(self, mock_run):
        """Test memory info collection."""
        from mlops.tracking.apple_silicon_metrics import get_memory_info

        # Mock vm_stat output
        mock_run.return_value = Mock(
            stdout="""Mach Virtual Memory Statistics: (page size of 4096 bytes)
Pages free:                              100000.
Pages active:                            200000.
Pages inactive:                          150000.
Pages wired down:                        50000.
"""
        )

        memory_info = get_memory_info()

        assert isinstance(memory_info, dict)
        assert "total_gb" in memory_info
        assert "used_gb" in memory_info
        assert "available_gb" in memory_info
        assert "utilization_percent" in memory_info
        assert all(isinstance(v, float) for v in memory_info.values())

    @patch("mlops.tracking.apple_silicon_metrics.subprocess.run")
    def test_get_memory_info_error(self, mock_run):
        """Test memory info collection error."""
        from mlops.tracking.apple_silicon_metrics import get_memory_info

        mock_run.side_effect = Exception("Test error")

        memory_info = get_memory_info()

        assert memory_info["total_gb"] == 0.0
        assert memory_info["used_gb"] == 0.0


class TestMPSInfo:
    """Test MPS information collection."""

    def test_get_mps_info_available(self):
        """Test MPS info when available."""
        import sys
        from mlops.tracking.apple_silicon_metrics import get_mps_info

        # Mock torch in sys.modules
        mock_torch = MagicMock()
        mock_torch.backends.mps.is_available.return_value = True

        with patch.dict(sys.modules, {"torch": mock_torch}):
            mps_info = get_mps_info()

            assert mps_info["available"] is True
            assert mps_info["utilization_percent"] is None

    def test_get_mps_info_not_available(self):
        """Test MPS info when not available."""
        import sys
        from mlops.tracking.apple_silicon_metrics import get_mps_info

        # Mock torch in sys.modules
        mock_torch = MagicMock()
        mock_torch.backends.mps.is_available.return_value = False

        with patch.dict(sys.modules, {"torch": mock_torch}):
            mps_info = get_mps_info()

            assert mps_info["available"] is False

    def test_get_mps_info_import_error(self):
        """Test MPS info when torch not available."""
        import sys
        from mlops.tracking.apple_silicon_metrics import get_mps_info

        # Remove torch from sys.modules temporarily
        torch_module = sys.modules.pop("torch", None)

        try:
            # Mock the import to raise ImportError
            with patch.dict("sys.modules", {"torch": None}):
                mps_info = get_mps_info()

            assert mps_info["available"] is False

        finally:
            # Restore torch if it was there
            if torch_module is not None:
                sys.modules["torch"] = torch_module


class TestANEInfo:
    """Test ANE information collection."""

    @patch("mlops.tracking.apple_silicon_metrics.detect_apple_silicon")
    def test_get_ane_info_available(self, mock_detect):
        """Test ANE info on Apple Silicon."""
        from mlops.tracking.apple_silicon_metrics import get_ane_info

        mock_detect.return_value = True

        assert get_ane_info() is True

    @patch("mlops.tracking.apple_silicon_metrics.detect_apple_silicon")
    def test_get_ane_info_not_available(self, mock_detect):
        """Test ANE info on non-Apple Silicon."""
        from mlops.tracking.apple_silicon_metrics import get_ane_info

        mock_detect.return_value = False

        assert get_ane_info() is False


class TestThermalState:
    """Test thermal state collection."""

    def test_get_thermal_state(self):
        """Test thermal state collection."""
        from mlops.tracking.apple_silicon_metrics import get_thermal_state

        thermal_state = get_thermal_state()

        assert isinstance(thermal_state, int)
        assert 0 <= thermal_state <= 3


class TestPowerMode:
    """Test power mode detection."""

    @patch("mlops.tracking.apple_silicon_metrics.subprocess.run")
    def test_get_power_mode_low_power(self, mock_run):
        """Test low power mode detection."""
        from mlops.tracking.apple_silicon_metrics import get_power_mode

        mock_run.return_value = Mock(stdout="Now drawing from 'Battery Power'\nLow Power Mode: enabled\n")

        power_mode = get_power_mode()

        assert power_mode == "low_power"

    @patch("mlops.tracking.apple_silicon_metrics.subprocess.run")
    def test_get_power_mode_ac_power(self, mock_run):
        """Test high performance mode detection."""
        from mlops.tracking.apple_silicon_metrics import get_power_mode

        mock_run.return_value = Mock(stdout="Now drawing from 'AC Power'\n")

        power_mode = get_power_mode()

        assert power_mode == "high_performance"

    @patch("mlops.tracking.apple_silicon_metrics.subprocess.run")
    def test_get_power_mode_normal(self, mock_run):
        """Test normal power mode."""
        from mlops.tracking.apple_silicon_metrics import get_power_mode

        mock_run.return_value = Mock(stdout="Now drawing from 'Battery Power'\n")

        power_mode = get_power_mode()

        assert power_mode == "normal"

    @patch("mlops.tracking.apple_silicon_metrics.subprocess.run")
    def test_get_power_mode_error(self, mock_run):
        """Test power mode detection error."""
        from mlops.tracking.apple_silicon_metrics import get_power_mode

        mock_run.side_effect = Exception("Test error")

        power_mode = get_power_mode()

        assert power_mode == "normal"


class TestCoreInfo:
    """Test core information collection."""

    @patch("mlops.tracking.apple_silicon_metrics.subprocess.run")
    def test_get_core_info(self, mock_run):
        """Test core info collection."""
        from mlops.tracking.apple_silicon_metrics import get_core_info

        # Mock sysctl responses
        def mock_run_side_effect(*args, **kwargs):
            cmd = args[0]
            if "hw.ncpu" in cmd:
                return Mock(stdout="8\n")
            elif "hw.perflevel0.logicalcpu" in cmd:
                return Mock(stdout="4\n")
            elif "hw.perflevel1.logicalcpu" in cmd:
                return Mock(stdout="4\n")
            return Mock(stdout="0\n")

        mock_run.side_effect = mock_run_side_effect

        core_info = get_core_info()

        assert core_info["total"] == 8
        assert core_info["performance"] == 4
        assert core_info["efficiency"] == 4

    @patch("mlops.tracking.apple_silicon_metrics.subprocess.run")
    def test_get_core_info_error(self, mock_run):
        """Test core info collection error."""
        from mlops.tracking.apple_silicon_metrics import get_core_info

        mock_run.side_effect = Exception("Test error")

        core_info = get_core_info()

        assert core_info["total"] == 0


class TestCollectMetrics:
    """Test comprehensive metrics collection."""

    @patch("mlops.tracking.apple_silicon_metrics.detect_apple_silicon")
    def test_collect_metrics_not_apple_silicon(self, mock_detect):
        """Test collection on non-Apple Silicon."""
        mock_detect.return_value = False

        with pytest.raises(AppleSiliconMetricsError) as exc_info:
            collect_metrics()

        assert "Not running on Apple Silicon" in str(exc_info.value)

    @patch("mlops.tracking.apple_silicon_metrics.detect_apple_silicon")
    @patch("mlops.tracking.apple_silicon_metrics.get_chip_type")
    @patch("mlops.tracking.apple_silicon_metrics.get_memory_info")
    @patch("mlops.tracking.apple_silicon_metrics.get_mps_info")
    @patch("mlops.tracking.apple_silicon_metrics.get_ane_info")
    @patch("mlops.tracking.apple_silicon_metrics.get_thermal_state")
    @patch("mlops.tracking.apple_silicon_metrics.get_power_mode")
    @patch("mlops.tracking.apple_silicon_metrics.get_core_info")
    def test_collect_metrics_success(
        self,
        mock_core_info,
        mock_power_mode,
        mock_thermal,
        mock_ane,
        mock_mps,
        mock_memory,
        mock_chip,
        mock_detect,
    ):
        """Test successful metrics collection."""
        mock_detect.return_value = True
        mock_chip.return_value = "M2"
        mock_memory.return_value = {
            "total_gb": 16.0,
            "used_gb": 8.0,
            "available_gb": 8.0,
            "utilization_percent": 50.0,
        }
        mock_mps.return_value = {"available": True, "utilization_percent": None}
        mock_ane.return_value = True
        mock_thermal.return_value = 0
        mock_power_mode.return_value = "normal"
        mock_core_info.return_value = {"total": 8, "performance": 4, "efficiency": 4}

        metrics = collect_metrics()

        assert isinstance(metrics, AppleSiliconMetrics)
        assert metrics.chip_type == "M2"
        assert metrics.memory_total_gb == 16.0
        assert metrics.mps_available is True
        assert metrics.core_count == 8


class TestLogMetricsToMLFlow:
    """Test logging metrics to MLFlow."""

    @patch("mlops.tracking.apple_silicon_metrics.collect_metrics")
    def test_log_metrics_to_mlflow_success(self, mock_collect):
        """Test successful logging to MLFlow."""
        mock_collect.return_value = AppleSiliconMetrics(
            chip_type="M2",
            memory_total_gb=16.0,
            memory_used_gb=8.0,
            memory_available_gb=8.0,
            memory_utilization_percent=50.0,
            mps_available=True,
            ane_available=True,
        )

        mock_client = MagicMock()

        log_metrics_to_mlflow(mock_client)

        # Verify metrics were logged
        mock_client.log_apple_silicon_metrics.assert_called_once()
        # Verify tags were set
        assert mock_client.set_tag.call_count == 2

    @patch("mlops.tracking.apple_silicon_metrics.collect_metrics")
    def test_log_metrics_to_mlflow_error(self, mock_collect):
        """Test logging to MLFlow with error."""
        mock_collect.side_effect = Exception("Test error")

        mock_client = MagicMock()

        with pytest.raises(AppleSiliconMetricsError):
            log_metrics_to_mlflow(mock_client)


class TestAppleSiliconMetricsError:
    """Test AppleSiliconMetricsError exception class."""

    def test_error_with_metric(self):
        """Test error with metric information."""
        error = AppleSiliconMetricsError("Test error", metric="memory")

        assert str(error) == "Test error"
        assert error.metric == "memory"
        assert error.details == {}

    def test_error_with_details(self):
        """Test error with details dictionary."""
        details = {"key": "value"}
        error = AppleSiliconMetricsError("Test error", details=details)

        assert error.details == details

    def test_error_minimal(self):
        """Test error with minimal information."""
        error = AppleSiliconMetricsError("Test error")

        assert str(error) == "Test error"
        assert error.metric is None
        assert error.details == {}
