"""Tests for Apple Silicon Integration Helpers"""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from mlops.silicon.integration import (
    AppleSiliconMetricsCollector,
    check_thermal_state,
    collect_apple_silicon_metrics,
    detect_apple_silicon,
    get_chip_type,
    get_memory_info,
    get_optimal_config_for_bentoml,
    get_optimal_config_for_training,
)


class TestIntegrationHelpers:
    """Test suite for integration helper functions"""

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

    def test_detect_apple_silicon_true(self, mock_apple_silicon):
        """Test detect_apple_silicon returns True on Apple Silicon"""
        result = detect_apple_silicon()
        assert isinstance(result, bool)

    def test_detect_apple_silicon_false(self, mock_non_apple_silicon):
        """Test detect_apple_silicon returns False on non-Apple Silicon"""
        result = detect_apple_silicon()
        assert result is False

    @patch("subprocess.run")
    def test_get_chip_type(self, mock_run, mock_apple_silicon):
        """Test get_chip_type returns chip information"""
        mock_run.return_value = Mock(stdout="Apple M2 Pro\n")

        chip_type = get_chip_type()
        assert isinstance(chip_type, str)
        assert len(chip_type) > 0

    @patch("psutil.virtual_memory")
    @patch("psutil.cpu_percent")
    @patch("psutil.cpu_freq")
    def test_collect_apple_silicon_metrics(
        self,
        mock_cpu_freq,
        mock_cpu_percent,
        mock_memory,
        mock_apple_silicon,
    ):
        """Test collect_apple_silicon_metrics returns metrics"""
        mock_memory.return_value = Mock(
            used=8 * 1024**3,
            available=8 * 1024**3,
            percent=50.0,
        )
        mock_cpu_percent.return_value = 50.0
        mock_cpu_freq.return_value = Mock(current=3200.0)

        metrics = collect_apple_silicon_metrics()
        assert metrics is not None
        assert hasattr(metrics, "chip_type")
        assert hasattr(metrics, "memory_total_gb")

    def test_get_optimal_config_for_bentoml_apple_silicon(self, mock_apple_silicon):
        """Test BentoML config for Apple Silicon"""
        config = get_optimal_config_for_bentoml(project_name="test")

        assert isinstance(config, dict)
        assert "workers" in config
        assert "max_batch_size" in config
        assert "apple_silicon" in config

    def test_get_optimal_config_for_bentoml_non_apple_silicon(
        self,
        mock_non_apple_silicon,
    ):
        """Test BentoML config for non-Apple Silicon"""
        config = get_optimal_config_for_bentoml()

        assert config["enable_apple_silicon"] is False
        assert config["workers"] == 1

    def test_get_optimal_config_for_training_apple_silicon(self, mock_apple_silicon):
        """Test training config for Apple Silicon"""
        config = get_optimal_config_for_training()

        assert isinstance(config, dict)
        assert "batch_size" in config
        assert "num_workers" in config
        assert "use_mlx" in config

    def test_get_optimal_config_for_training_memory_intensive(
        self,
        mock_apple_silicon,
    ):
        """Test training config for memory intensive workload"""
        config = get_optimal_config_for_training(memory_intensive=True)

        # Should have reasonable batch size for memory-intensive workload
        assert config["batch_size"] > 0
        assert "memory_limit_gb" in config

    def test_get_optimal_config_for_training_non_apple_silicon(
        self,
        mock_non_apple_silicon,
    ):
        """Test training config for non-Apple Silicon"""
        config = get_optimal_config_for_training()

        assert config["use_mlx"] is False
        assert config["use_mps"] is False

    def test_check_thermal_state(self, mock_apple_silicon):
        """Test check_thermal_state returns state info"""
        state_code, state_name = check_thermal_state()

        assert isinstance(state_code, int)
        assert 0 <= state_code <= 3
        assert isinstance(state_name, str)
        assert state_name in ["nominal", "fair", "serious", "critical", "unknown"]

    @patch("psutil.virtual_memory")
    @patch("psutil.cpu_percent")
    @patch("psutil.cpu_freq")
    def test_get_memory_info(
        self,
        mock_cpu_freq,
        mock_cpu_percent,
        mock_memory,
        mock_apple_silicon,
    ):
        """Test get_memory_info returns memory metrics"""
        mock_memory.return_value = Mock(
            used=10 * 1024**3,
            available=6 * 1024**3,
            percent=62.5,
        )
        mock_cpu_percent.return_value = 50.0
        mock_cpu_freq.return_value = Mock(current=3200.0)

        memory_info = get_memory_info()

        assert isinstance(memory_info, dict)
        assert "total_gb" in memory_info
        assert "used_gb" in memory_info
        assert "available_gb" in memory_info
        assert "utilization_percent" in memory_info


class TestAppleSiliconMetricsCollectorCompat:
    """Test suite for backward-compatible metrics collector"""

    @pytest.fixture
    def mock_apple_silicon(self):
        """Mock Apple Silicon system"""
        with patch("platform.system", return_value="Darwin"), \
             patch("platform.machine", return_value="arm64"), \
             patch("platform.processor", return_value="arm"):
            yield

    def test_collector_initialization(self, mock_apple_silicon):
        """Test collector initializes correctly"""
        collector = AppleSiliconMetricsCollector(project_name="test")
        assert collector.project_name == "test"
        assert collector._monitor is not None

    def test_collector_is_apple_silicon(self, mock_apple_silicon):
        """Test is_apple_silicon method"""
        collector = AppleSiliconMetricsCollector()
        result = collector.is_apple_silicon()
        assert isinstance(result, bool)

    @patch("psutil.virtual_memory")
    @patch("psutil.cpu_percent")
    @patch("psutil.cpu_freq")
    def test_collector_collect(
        self,
        mock_cpu_freq,
        mock_cpu_percent,
        mock_memory,
        mock_apple_silicon,
    ):
        """Test collect method"""
        mock_memory.return_value = Mock(
            used=8 * 1024**3,
            available=8 * 1024**3,
            percent=50.0,
        )
        mock_cpu_percent.return_value = 50.0
        mock_cpu_freq.return_value = Mock(current=3200.0)

        collector = AppleSiliconMetricsCollector()
        metrics = collector.collect()

        assert metrics is not None
        assert hasattr(metrics, "chip_type")
        assert hasattr(metrics, "memory_total_gb")


class TestIntegrationWithMLFlow:
    """Test suite for MLFlow integration"""

    @pytest.fixture
    def mock_apple_silicon(self):
        """Mock Apple Silicon system"""
        with patch("platform.system", return_value="Darwin"), \
             patch("platform.machine", return_value="arm64"), \
             patch("platform.processor", return_value="arm"):
            yield

    @pytest.fixture
    def mock_mlflow_client(self):
        """Mock MLFlow client"""
        client = Mock()
        client.log_apple_silicon_metrics = Mock()
        client.set_tag = Mock()
        return client

    @patch("psutil.virtual_memory")
    @patch("psutil.cpu_percent")
    @patch("psutil.cpu_freq")
    def test_log_metrics_to_mlflow(
        self,
        mock_cpu_freq,
        mock_cpu_percent,
        mock_memory,
        mock_mlflow_client,
        mock_apple_silicon,
    ):
        """Test logging metrics to MLFlow"""
        from mlops.silicon.integration import log_metrics_to_mlflow

        mock_memory.return_value = Mock(
            used=8 * 1024**3,
            available=8 * 1024**3,
            percent=50.0,
        )
        mock_cpu_percent.return_value = 50.0
        mock_cpu_freq.return_value = Mock(current=3200.0)

        log_metrics_to_mlflow(mock_mlflow_client)

        # Verify MLFlow client methods were called
        mock_mlflow_client.log_apple_silicon_metrics.assert_called_once()
        assert mock_mlflow_client.set_tag.call_count >= 2  # chip_type, variant, power_mode

    @patch("psutil.virtual_memory")
    @patch("psutil.cpu_percent")
    @patch("psutil.cpu_freq")
    def test_log_metrics_to_mlflow_failure(
        self,
        mock_cpu_freq,
        mock_cpu_percent,
        mock_memory,
        mock_apple_silicon,
    ):
        """Test graceful handling of MLFlow logging failure"""
        from mlops.silicon.integration import log_metrics_to_mlflow

        mock_memory.return_value = Mock(
            used=8 * 1024**3,
            available=8 * 1024**3,
            percent=50.0,
        )
        mock_cpu_percent.return_value = 50.0
        mock_cpu_freq.return_value = Mock(current=3200.0)

        failing_client = Mock()
        failing_client.log_apple_silicon_metrics.side_effect = Exception("MLFlow error")

        # Should not raise exception
        log_metrics_to_mlflow(failing_client)
