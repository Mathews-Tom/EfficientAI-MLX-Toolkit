"""Tests for Scaling Manager

This module tests the auto-scaling and load balancing functionality.
"""

import pytest
from unittest.mock import patch, MagicMock, Mock
import time

from mlops.serving.scaling_manager import (
    ScalingManager,
    ScalingConfig,
    ScalingMetrics,
    ThermalMonitor,
    create_scaling_manager,
)


class TestScalingConfig:
    """Tests for scaling configuration"""

    def test_default_config(self):
        """Test default scaling configuration"""
        config = ScalingConfig()

        assert config.min_replicas == 1
        assert config.max_replicas == 10
        assert config.target_requests_per_replica == 10
        assert config.thermal_aware is True

    def test_custom_config(self):
        """Test custom scaling configuration"""
        config = ScalingConfig(
            min_replicas=2,
            max_replicas=20,
            target_requests_per_replica=5,
            thermal_aware=False,
        )

        assert config.min_replicas == 2
        assert config.max_replicas == 20
        assert config.target_requests_per_replica == 5
        assert config.thermal_aware is False


class TestScalingManager:
    """Tests for ScalingManager"""

    def test_init(self):
        """Test scaling manager initialization"""
        mock_cluster = Mock()
        manager = ScalingManager(cluster=mock_cluster)

        assert manager.cluster == mock_cluster
        assert manager.config is not None
        assert isinstance(manager.config, ScalingConfig)

    def test_init_with_config(self):
        """Test initialization with custom config"""
        mock_cluster = Mock()
        config = ScalingConfig(max_replicas=5)

        manager = ScalingManager(cluster=mock_cluster, scaling_config=config)

        assert manager.config.max_replicas == 5

    def test_evaluate_scaling_scale_up(self):
        """Test scaling evaluation recommends scale up"""
        mock_cluster = Mock()
        config = ScalingConfig(
            target_requests_per_replica=10,
            scale_up_threshold=0.8,
        )
        manager = ScalingManager(cluster=mock_cluster, scaling_config=config)

        # High load: 20 requests for 2 replicas = 10 req/replica (at threshold)
        current_metrics = {
            "num_replicas": 2,
            "ongoing_requests": 20,  # 10 per replica, above 80% threshold
            "cpu_utilization_pct": 60.0,
            "memory_utilization_pct": 50.0,
        }

        metrics = manager.evaluate_scaling("proj1", "model1", current_metrics)

        assert metrics.current_replicas == 2
        assert metrics.target_replicas == 3  # Should scale up

    def test_evaluate_scaling_scale_down(self):
        """Test scaling evaluation recommends scale down"""
        mock_cluster = Mock()
        config = ScalingConfig(
            target_requests_per_replica=10,
            scale_down_threshold=0.3,
        )
        manager = ScalingManager(cluster=mock_cluster, scaling_config=config)

        # Low load: 2 requests for 2 replicas = 1 req/replica (below 30% threshold)
        current_metrics = {
            "num_replicas": 2,
            "ongoing_requests": 2,
            "cpu_utilization_pct": 20.0,
            "memory_utilization_pct": 15.0,
        }

        metrics = manager.evaluate_scaling("proj1", "model1", current_metrics)

        assert metrics.current_replicas == 2
        assert metrics.target_replicas == 1  # Should scale down

    def test_evaluate_scaling_no_change(self):
        """Test scaling evaluation recommends no change"""
        mock_cluster = Mock()
        manager = ScalingManager(cluster=mock_cluster)

        # Moderate load: within thresholds
        current_metrics = {
            "num_replicas": 2,
            "ongoing_requests": 10,  # 5 per replica, within thresholds
            "cpu_utilization_pct": 50.0,
            "memory_utilization_pct": 40.0,
        }

        metrics = manager.evaluate_scaling("proj1", "model1", current_metrics)

        assert metrics.current_replicas == 2
        assert metrics.target_replicas == 2  # No change

    @patch("mlops.serving.scaling_manager.ThermalMonitor")
    def test_evaluate_scaling_thermal_prevent_scale_up(self, mock_thermal_class):
        """Test thermal state prevents scale up"""
        mock_cluster = Mock()
        config = ScalingConfig(
            thermal_aware=True,
            target_requests_per_replica=10,
        )

        # Mock thermal monitor
        mock_thermal = Mock()
        mock_thermal.get_thermal_state.return_value = "serious"
        mock_thermal_class.return_value = mock_thermal

        manager = ScalingManager(cluster=mock_cluster, scaling_config=config)

        # High load but thermal state is serious
        current_metrics = {
            "num_replicas": 2,
            "ongoing_requests": 30,  # Would normally trigger scale up
            "cpu_utilization_pct": 80.0,
            "memory_utilization_pct": 70.0,
        }

        metrics = manager.evaluate_scaling("proj1", "model1", current_metrics)

        # Should NOT scale up due to thermal state
        assert metrics.target_replicas == 2

    @patch("mlops.serving.scaling_manager.ThermalMonitor")
    def test_evaluate_scaling_thermal_force_scale_down(self, mock_thermal_class):
        """Test critical thermal state forces scale down"""
        mock_cluster = Mock()
        config = ScalingConfig(thermal_aware=True)

        # Mock thermal monitor
        mock_thermal = Mock()
        mock_thermal.get_thermal_state.return_value = "critical"
        mock_thermal_class.return_value = mock_thermal

        manager = ScalingManager(cluster=mock_cluster, scaling_config=config)

        # Normal load but thermal state is critical
        current_metrics = {
            "num_replicas": 4,
            "ongoing_requests": 20,
            "cpu_utilization_pct": 60.0,
            "memory_utilization_pct": 50.0,
        }

        metrics = manager.evaluate_scaling("proj1", "model1", current_metrics)

        # Should force scale down
        assert metrics.target_replicas == 3

    def test_apply_scaling_success(self):
        """Test successfully applying scaling decision"""
        mock_cluster = Mock()
        mock_cluster.scale_deployment = Mock()

        manager = ScalingManager(cluster=mock_cluster)

        metrics = ScalingMetrics(
            current_replicas=2,
            target_replicas=3,
            ongoing_requests=20,
            avg_requests_per_replica=10.0,
            cpu_utilization_pct=60.0,
            memory_utilization_pct=50.0,
        )

        result = manager.apply_scaling("proj1", "model1", metrics)

        assert result is True
        mock_cluster.scale_deployment.assert_called_once_with(
            project_name="proj1",
            model_name="model1",
            num_replicas=3,
        )

        # Should record scaling event
        history = manager.get_scaling_history()
        assert len(history) == 1
        assert history[0]["from_replicas"] == 2
        assert history[0]["to_replicas"] == 3

    def test_apply_scaling_cooldown(self):
        """Test scaling is skipped during cooldown period"""
        mock_cluster = Mock()
        config = ScalingConfig(cooldown_period_s=60)
        manager = ScalingManager(cluster=mock_cluster, scaling_config=config)

        metrics = ScalingMetrics(
            current_replicas=2,
            target_replicas=3,
            ongoing_requests=20,
            avg_requests_per_replica=10.0,
            cpu_utilization_pct=60.0,
            memory_utilization_pct=50.0,
        )

        # First scaling should work
        result1 = manager.apply_scaling("proj1", "model1", metrics)
        assert result1 is True

        # Immediate second scaling should be skipped
        result2 = manager.apply_scaling("proj1", "model1", metrics)
        assert result2 is False

    def test_apply_scaling_no_change_needed(self):
        """Test scaling is skipped when replicas already at target"""
        mock_cluster = Mock()
        manager = ScalingManager(cluster=mock_cluster)

        metrics = ScalingMetrics(
            current_replicas=2,
            target_replicas=2,  # Same as current
            ongoing_requests=10,
            avg_requests_per_replica=5.0,
            cpu_utilization_pct=40.0,
            memory_utilization_pct=30.0,
        )

        result = manager.apply_scaling("proj1", "model1", metrics)

        assert result is False

    def test_get_scaling_history(self):
        """Test getting scaling history"""
        mock_cluster = Mock()
        mock_cluster.scale_deployment = Mock()

        manager = ScalingManager(cluster=mock_cluster)

        # Add multiple scaling events
        for i in range(5):
            metrics = ScalingMetrics(
                current_replicas=i,
                target_replicas=i+1,
                ongoing_requests=10,
                avg_requests_per_replica=5.0,
                cpu_utilization_pct=50.0,
                memory_utilization_pct=40.0,
            )
            # Bypass cooldown for testing
            manager._last_scale_time = {}
            manager.apply_scaling("proj1", "model1", metrics)

        history = manager.get_scaling_history(limit=3)

        assert len(history) == 3  # Limited to 3
        # Most recent first
        assert history[0]["from_replicas"] == 4
        assert history[0]["to_replicas"] == 5

    def test_optimize_cluster_resources_high_cpu(self):
        """Test cluster optimization with high CPU usage"""
        mock_cluster = Mock()
        mock_cluster.get_cluster_resource_usage.return_value = {
            "total_deployments": 5,
            "cpu": {"utilization_pct": 85.0},
        }

        manager = ScalingManager(cluster=mock_cluster)
        optimization = manager.optimize_cluster_resources()

        assert "recommendations" in optimization
        assert len(optimization["recommendations"]) > 0
        assert optimization["recommendations"][0]["type"] == "scale_down"

    def test_optimize_cluster_resources_low_cpu(self):
        """Test cluster optimization with low CPU usage"""
        mock_cluster = Mock()
        mock_cluster.get_cluster_resource_usage.return_value = {
            "total_deployments": 3,
            "cpu": {"utilization_pct": 15.0},
        }

        manager = ScalingManager(cluster=mock_cluster)
        optimization = manager.optimize_cluster_resources()

        assert "recommendations" in optimization
        assert len(optimization["recommendations"]) > 0
        assert optimization["recommendations"][0]["type"] == "consolidate"


class TestThermalMonitor:
    """Tests for thermal monitoring"""

    def test_init(self):
        """Test thermal monitor initialization"""
        monitor = ThermalMonitor()

        assert monitor._cached_state == "nominal"

    def test_get_thermal_state_cached(self):
        """Test thermal state uses cache within interval"""
        monitor = ThermalMonitor()
        monitor._cached_state = "fair"
        monitor._last_check_time = time.time()

        state = monitor.get_thermal_state()

        assert state == "fair"  # Should use cached value

    @patch("subprocess.run")
    def test_check_thermal_state_nominal(self, mock_run):
        """Test thermal state check returns nominal"""
        mock_run.return_value = MagicMock(
            stdout="No thermal pressure",
            returncode=0
        )

        monitor = ThermalMonitor()
        # Force check by clearing cache
        monitor._last_check_time = 0

        state = monitor.get_thermal_state()

        assert state == "nominal"

    @patch("subprocess.run")
    def test_check_thermal_state_serious(self, mock_run):
        """Test thermal state check detects serious state"""
        mock_run.return_value = MagicMock(
            stdout="CPU_Scheduler_Limit = 75",
            returncode=0
        )

        monitor = ThermalMonitor()
        monitor._last_check_time = 0

        state = monitor.get_thermal_state()

        assert state == "serious"

    @patch("subprocess.run")
    def test_check_thermal_state_critical(self, mock_run):
        """Test thermal state check detects critical state"""
        mock_run.return_value = MagicMock(
            stdout="CPU_Scheduler_Limit = 100",
            returncode=0
        )

        monitor = ThermalMonitor()
        monitor._last_check_time = 0

        state = monitor.get_thermal_state()

        assert state == "critical"

    @patch("subprocess.run")
    def test_check_thermal_state_fallback(self, mock_run):
        """Test thermal state defaults to nominal on error"""
        mock_run.side_effect = FileNotFoundError()

        monitor = ThermalMonitor()
        monitor._last_check_time = 0

        state = monitor.get_thermal_state()

        assert state == "nominal"


class TestCreateScalingManager:
    """Tests for scaling manager factory"""

    def test_create_scaling_manager(self):
        """Test creating scaling manager"""
        mock_cluster = Mock()
        manager = create_scaling_manager(cluster=mock_cluster)

        assert isinstance(manager, ScalingManager)
        assert manager.cluster == mock_cluster

    def test_create_scaling_manager_with_config(self):
        """Test creating scaling manager with custom config"""
        mock_cluster = Mock()
        config = ScalingConfig(max_replicas=20)

        manager = create_scaling_manager(cluster=mock_cluster, scaling_config=config)

        assert manager.config.max_replicas == 20
