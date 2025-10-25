"""Tests for Apple Silicon Monitor"""

from __future__ import annotations

import platform
from datetime import datetime
from unittest.mock import Mock, patch

import pytest

from mlops.silicon.detector import AppleSiliconDetector
from mlops.silicon.metrics import AppleSiliconMetrics
from mlops.silicon.monitor import AppleSiliconMonitor


class TestAppleSiliconMonitor:
    """Test suite for AppleSiliconMonitor"""

    @pytest.fixture
    def mock_apple_silicon(self):
        """Mock Apple Silicon system"""
        with patch("platform.system", return_value="Darwin"), \
             patch("platform.machine", return_value="arm64"), \
             patch("platform.processor", return_value="arm"):
            yield

    @pytest.fixture
    def mock_detector(self):
        """Mock detector with known hardware info"""
        detector = Mock(spec=AppleSiliconDetector)
        detector.is_apple_silicon.return_value = True
        detector.get_hardware_info.return_value = Mock(
            is_apple_silicon=True,
            chip_type="M1",
            chip_variant="Base",
            memory_total_gb=16.0,
            core_count=8,
            performance_cores=4,
            efficiency_cores=4,
            mlx_available=True,
            mps_available=True,
            ane_available=True,
            thermal_state=0,
            power_mode="normal",
        )
        return detector

    def test_monitor_initialization(self, mock_apple_silicon):
        """Test monitor initializes correctly"""
        monitor = AppleSiliconMonitor(project_name="test_project")
        assert monitor.project_name == "test_project"
        assert monitor.detector is not None
        assert monitor.hardware_info is not None

    def test_monitor_with_custom_detector(self, mock_detector):
        """Test monitor with custom detector"""
        monitor = AppleSiliconMonitor(detector=mock_detector)
        assert monitor.detector == mock_detector

    @patch("psutil.virtual_memory")
    @patch("psutil.cpu_percent")
    @patch("psutil.cpu_freq")
    def test_collect_metrics(
        self,
        mock_cpu_freq,
        mock_cpu_percent,
        mock_memory,
        mock_apple_silicon,
    ):
        """Test metrics collection"""
        # Mock psutil responses
        mock_memory.return_value = Mock(
            used=8 * 1024**3,
            available=8 * 1024**3,
            percent=50.0,
        )
        mock_cpu_percent.return_value = 45.0
        mock_cpu_freq.return_value = Mock(current=3200.0)

        monitor = AppleSiliconMonitor()
        metrics = monitor.collect()

        assert isinstance(metrics, AppleSiliconMetrics)
        assert isinstance(metrics.timestamp, datetime)
        assert metrics.cpu_percent == 45.0
        assert metrics.memory_utilization_percent > 0

    @patch("subprocess.run")
    @patch("psutil.cpu_percent")
    @patch("psutil.cpu_freq")
    def test_collect_memory_vmstat(
        self,
        mock_cpu_freq,
        mock_cpu_percent,
        mock_run,
        mock_apple_silicon,
    ):
        """Test memory collection using vm_stat"""
        # Mock vm_stat output
        vm_stat_output = """Mach Virtual Memory Statistics: (page size of 4096 bytes)
Pages free:                               500000.
Pages active:                             1000000.
Pages inactive:                           500000.
Pages wired down:                         300000."""

        mock_run.return_value = Mock(stdout=vm_stat_output)
        mock_cpu_percent.return_value = 50.0
        mock_cpu_freq.return_value = Mock(current=3200.0)

        monitor = AppleSiliconMonitor()
        metrics = monitor.collect()

        assert metrics.memory_used_gb > 0
        assert metrics.memory_available_gb > 0
        assert metrics.memory_utilization_percent > 0

    @patch("psutil.virtual_memory")
    @patch("psutil.cpu_percent")
    @patch("psutil.cpu_freq")
    def test_collect_memory_psutil_fallback(
        self,
        mock_cpu_freq,
        mock_cpu_percent,
        mock_memory,
        mock_apple_silicon,
    ):
        """Test memory collection fallback to psutil"""
        mock_memory.return_value = Mock(
            used=10 * 1024**3,
            available=6 * 1024**3,
            percent=62.5,
        )
        mock_cpu_percent.return_value = 50.0
        mock_cpu_freq.return_value = Mock(current=3200.0)

        with patch("subprocess.run", side_effect=Exception("vm_stat failed")):
            monitor = AppleSiliconMonitor()
            metrics = monitor.collect()

            assert metrics.memory_used_gb == pytest.approx(10.0, rel=0.1)
            assert metrics.memory_available_gb == pytest.approx(6.0, rel=0.1)

    @patch("psutil.cpu_percent")
    @patch("psutil.cpu_freq")
    @patch("psutil.virtual_memory")
    def test_collect_cpu_metrics(
        self,
        mock_memory,
        mock_cpu_freq,
        mock_cpu_percent,
        mock_apple_silicon,
    ):
        """Test CPU metrics collection"""
        mock_memory.return_value = Mock(
            used=8 * 1024**3,
            available=8 * 1024**3,
            percent=50.0,
        )
        mock_cpu_percent.return_value = 75.5
        mock_cpu_freq.return_value = Mock(current=3456.78)

        monitor = AppleSiliconMonitor()
        metrics = monitor.collect()

        assert metrics.cpu_percent == 75.5
        assert metrics.cpu_freq_mhz == pytest.approx(3456.78, rel=0.01)

    def test_is_apple_silicon_method(self, mock_apple_silicon):
        """Test is_apple_silicon method"""
        monitor = AppleSiliconMonitor()
        assert isinstance(monitor.is_apple_silicon(), bool)

    def test_get_hardware_summary(self, mock_apple_silicon):
        """Test hardware summary retrieval"""
        monitor = AppleSiliconMonitor()
        summary = monitor.get_hardware_summary()

        assert isinstance(summary, dict)
        assert "is_apple_silicon" in summary
        assert "chip_type" in summary
        assert "memory_total_gb" in summary

    @patch("psutil.virtual_memory")
    @patch("psutil.cpu_percent")
    @patch("psutil.cpu_freq")
    def test_check_health_good(
        self,
        mock_cpu_freq,
        mock_cpu_percent,
        mock_memory,
        mock_apple_silicon,
    ):
        """Test health check with good system state"""
        mock_memory.return_value = Mock(
            used=8 * 1024**3,
            available=8 * 1024**3,
            percent=50.0,
        )
        mock_cpu_percent.return_value = 50.0
        mock_cpu_freq.return_value = Mock(current=3200.0)

        monitor = AppleSiliconMonitor()
        health = monitor.check_health()

        assert isinstance(health, dict)
        assert "score" in health
        assert "thermal_throttling" in health
        assert "memory_constrained" in health
        assert "recommendations" in health

        assert health["score"] > 80  # Good health
        assert health["thermal_throttling"] is False
        assert health["memory_constrained"] is False

    @patch("psutil.virtual_memory")
    @patch("psutil.cpu_percent")
    @patch("psutil.cpu_freq")
    def test_check_health_thermal_throttling(
        self,
        mock_cpu_freq,
        mock_cpu_percent,
        mock_memory,
        mock_apple_silicon,
    ):
        """Test health check with thermal throttling"""
        mock_memory.return_value = Mock(
            used=8 * 1024**3,
            available=8 * 1024**3,
            percent=50.0,
        )
        mock_cpu_percent.return_value = 50.0
        mock_cpu_freq.return_value = Mock(current=3200.0)

        monitor = AppleSiliconMonitor()

        # Manually set thermal state to serious
        monitor.hardware_info.thermal_state = 2

        health = monitor.check_health()

        assert health["thermal_throttling"] is True
        assert any("throttling" in r.lower() for r in health["recommendations"])

    @patch("psutil.virtual_memory")
    @patch("psutil.cpu_percent")
    @patch("psutil.cpu_freq")
    def test_check_health_memory_constrained(
        self,
        mock_cpu_freq,
        mock_cpu_percent,
        mock_memory,
        mock_apple_silicon,
    ):
        """Test health check with high memory usage"""
        mock_memory.return_value = Mock(
            used=14 * 1024**3,
            available=2 * 1024**3,
            percent=87.5,
        )
        mock_cpu_percent.return_value = 50.0
        mock_cpu_freq.return_value = Mock(current=3200.0)

        monitor = AppleSiliconMonitor()
        health = monitor.check_health()

        assert health["memory_constrained"] is True
        assert any("memory" in r.lower() for r in health["recommendations"])

    @patch("psutil.virtual_memory")
    @patch("psutil.cpu_percent")
    @patch("psutil.cpu_freq")
    def test_check_health_low_power(
        self,
        mock_cpu_freq,
        mock_cpu_percent,
        mock_memory,
        mock_apple_silicon,
    ):
        """Test health check with low power mode"""
        mock_memory.return_value = Mock(
            used=8 * 1024**3,
            available=8 * 1024**3,
            percent=50.0,
        )
        mock_cpu_percent.return_value = 50.0
        mock_cpu_freq.return_value = Mock(current=3200.0)

        monitor = AppleSiliconMonitor()
        monitor.hardware_info.power_mode = "low_power"

        health = monitor.check_health()

        assert any("power" in r.lower() for r in health["recommendations"])

    def test_metrics_collection_failure(self, mock_apple_silicon):
        """Test graceful handling of metrics collection failure"""
        with patch("psutil.virtual_memory", side_effect=Exception("psutil failed")):
            monitor = AppleSiliconMonitor()

            with pytest.raises(RuntimeError, match="Metrics collection failed"):
                monitor.collect()

    @patch("psutil.virtual_memory")
    @patch("psutil.cpu_percent")
    @patch("psutil.cpu_freq")
    def test_mps_utilization_collection(
        self,
        mock_cpu_freq,
        mock_cpu_percent,
        mock_memory,
        mock_apple_silicon,
    ):
        """Test MPS utilization collection"""
        mock_memory.return_value = Mock(
            used=8 * 1024**3,
            available=8 * 1024**3,
            percent=50.0,
        )
        mock_cpu_percent.return_value = 50.0
        mock_cpu_freq.return_value = Mock(current=3200.0)

        monitor = AppleSiliconMonitor()
        metrics = monitor.collect()

        # MPS utilization is not directly available, should be None
        assert metrics.mps_utilization_percent is None


@pytest.mark.skipif(
    platform.system() != "Darwin" or platform.machine() != "arm64",
    reason="Requires Apple Silicon hardware",
)
class TestAppleSiliconMonitorRealHardware:
    """Tests that require actual Apple Silicon hardware"""

    def test_real_metrics_collection(self):
        """Test metrics collection on real Apple Silicon"""
        monitor = AppleSiliconMonitor(project_name="test")
        metrics = monitor.collect()

        assert isinstance(metrics, AppleSiliconMetrics)
        assert metrics.chip_type in ["M1", "M2", "M3", "M4"]
        assert metrics.memory_total_gb > 0
        assert metrics.memory_used_gb > 0
        assert metrics.cpu_percent >= 0
        assert metrics.cpu_count > 0

    def test_real_health_check(self):
        """Test health check on real Apple Silicon"""
        monitor = AppleSiliconMonitor()
        health = monitor.check_health()

        assert isinstance(health, dict)
        assert 0 <= health["score"] <= 100
        assert isinstance(health["thermal_throttling"], bool)
        assert isinstance(health["memory_constrained"], bool)
        assert len(health["recommendations"]) > 0
