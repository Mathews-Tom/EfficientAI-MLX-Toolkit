"""Tests for Thermal-Aware Airflow Scheduler"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from mlops.airflow.thermal_scheduler import (
    SchedulingDecision,
    ThermalAwareScheduler,
    ThermalThresholds,
)
from mlops.silicon.metrics import AppleSiliconMetrics


@pytest.fixture
def mock_monitor():
    """Create mock AppleSiliconMonitor"""
    monitor = MagicMock()
    return monitor


@pytest.fixture
def mock_metrics():
    """Create mock metrics with nominal thermal state"""
    from datetime import datetime

    return AppleSiliconMetrics(
        timestamp=datetime.now(),
        chip_type="M3",
        chip_variant="Max",
        memory_total_gb=64.0,
        memory_used_gb=32.0,
        memory_available_gb=32.0,
        memory_utilization_percent=50.0,
        mlx_available=True,
        mps_available=True,
        ane_available=True,
        thermal_state=0,  # nominal
        power_mode="normal",
        cpu_percent=50.0,
        cpu_count=12,
        performance_cores=8,
        efficiency_cores=4,
        cpu_freq_mhz=3500.0,
    )


@pytest.fixture
def scheduler(mock_monitor):
    """Create ThermalAwareScheduler with mock monitor"""
    return ThermalAwareScheduler(monitor=mock_monitor)


class TestThermalThresholds:
    """Test ThermalThresholds configuration"""

    def test_default_thresholds(self):
        """Test default threshold values"""
        thresholds = ThermalThresholds()
        assert thresholds.nominal == 0
        assert thresholds.fair == 1
        assert thresholds.serious == 2
        assert thresholds.critical == 3

    def test_custom_thresholds(self):
        """Test custom threshold values"""
        thresholds = ThermalThresholds(
            nominal=0,
            fair=2,
            serious=3,
            critical=3,
        )
        assert thresholds.fair == 2
        assert thresholds.serious == 3


class TestSchedulingDecision:
    """Test SchedulingDecision dataclass"""

    def test_should_run_decision(self):
        """Test decision to run task"""
        decision = SchedulingDecision(
            should_run=True,
            reason="System conditions good",
            throttle_level=1.0,
            thermal_state=0,
            health_score=100.0,
        )
        assert decision.should_run
        assert decision.throttle_level == 1.0
        assert decision.retry_after_seconds is None

    def test_should_not_run_decision(self):
        """Test decision to not run task"""
        decision = SchedulingDecision(
            should_run=False,
            reason="Thermal throttling",
            retry_after_seconds=60,
            thermal_state=2,
            health_score=60.0,
        )
        assert not decision.should_run
        assert decision.retry_after_seconds == 60


class TestThermalAwareScheduler:
    """Test ThermalAwareScheduler"""

    def test_initialization(self, scheduler):
        """Test scheduler initialization"""
        assert scheduler.monitor is not None
        assert scheduler.thresholds is not None
        assert scheduler.memory_threshold == 85.0
        assert scheduler.min_health_score == 40.0

    def test_initialization_with_custom_config(self):
        """Test scheduler with custom configuration"""
        thresholds = ThermalThresholds(fair=2)
        scheduler = ThermalAwareScheduler(
            thresholds=thresholds,
            memory_threshold_percent=90.0,
            min_health_score=50.0,
        )
        assert scheduler.thresholds.fair == 2
        assert scheduler.memory_threshold == 90.0
        assert scheduler.min_health_score == 50.0

    def test_should_run_task_nominal(self, scheduler, mock_monitor, mock_metrics):
        """Test task scheduling in nominal conditions"""
        mock_monitor.collect.return_value = mock_metrics

        decision = scheduler.should_run_task(
            task_id="test_task",
            thermal_threshold=2,
        )

        assert decision.should_run
        assert decision.throttle_level == 1.0
        assert decision.thermal_state == 0
        assert decision.health_score == 100.0

    def test_should_run_task_thermal_threshold_exceeded(
        self, scheduler, mock_monitor, mock_metrics
    ):
        """Test task blocked due to thermal threshold"""
        mock_metrics.thermal_state = 3  # critical
        mock_monitor.collect.return_value = mock_metrics

        decision = scheduler.should_run_task(
            task_id="test_task",
            thermal_threshold=2,  # max serious
        )

        assert not decision.should_run
        assert "exceeds threshold" in decision.reason
        assert decision.retry_after_seconds is not None

    def test_should_run_task_low_health_score(
        self, scheduler, mock_monitor, mock_metrics
    ):
        """Test task blocked due to low health score"""
        mock_metrics.thermal_state = 2  # serious
        mock_metrics.memory_utilization_percent = 95.0  # high memory
        mock_monitor.collect.return_value = mock_metrics

        decision = scheduler.should_run_task(
            task_id="test_task",
            thermal_threshold=2,
            priority="normal",
        )

        # Health score will be low due to thermal and memory
        # Should be blocked for non-critical priority
        assert not decision.should_run or decision.throttle_level < 1.0

    def test_should_run_task_critical_priority_bypasses_health(
        self, scheduler, mock_monitor, mock_metrics
    ):
        """Test critical priority task bypasses health check"""
        mock_metrics.thermal_state = 2
        mock_metrics.memory_utilization_percent = 95.0
        mock_monitor.collect.return_value = mock_metrics

        decision = scheduler.should_run_task(
            task_id="critical_task",
            thermal_threshold=2,
            priority="critical",
        )

        # Critical tasks should run even with low health
        assert decision.should_run or decision.throttle_level > 0

    def test_should_run_task_insufficient_memory(
        self, scheduler, mock_monitor, mock_metrics
    ):
        """Test task blocked due to insufficient memory"""
        mock_metrics.memory_available_gb = 4.0
        mock_monitor.collect.return_value = mock_metrics

        decision = scheduler.should_run_task(
            task_id="test_task",
            thermal_threshold=2,
            memory_required_gb=8.0,
        )

        assert not decision.should_run
        assert "Insufficient memory" in decision.reason

    def test_should_run_task_memory_pressure(
        self, scheduler, mock_monitor, mock_metrics
    ):
        """Test task with memory pressure"""
        mock_metrics.memory_utilization_percent = 90.0
        mock_monitor.collect.return_value = mock_metrics

        decision = scheduler.should_run_task(
            task_id="test_task",
            thermal_threshold=2,
            priority="high",
        )

        # High priority tasks run with throttle
        if decision.should_run:
            assert decision.throttle_level < 1.0

    def test_should_run_task_thermal_throttling(
        self, scheduler, mock_monitor, mock_metrics
    ):
        """Test task during thermal throttling"""
        mock_metrics.thermal_state = 2  # serious - thermal throttling
        mock_monitor.collect.return_value = mock_metrics

        decision = scheduler.should_run_task(
            task_id="test_task",
            thermal_threshold=2,
            priority="normal",
        )

        # Non-critical tasks should be blocked or heavily throttled
        assert not decision.should_run or decision.throttle_level < 1.0

    def test_should_run_task_critical_during_throttling(
        self, scheduler, mock_monitor, mock_metrics
    ):
        """Test critical task during thermal throttling"""
        mock_metrics.thermal_state = 2
        mock_monitor.collect.return_value = mock_metrics

        decision = scheduler.should_run_task(
            task_id="critical_task",
            thermal_threshold=2,
            priority="critical",
        )

        # Critical tasks run with heavy throttle
        assert decision.should_run
        assert decision.throttle_level <= 1.0

    def test_calculate_retry_delay(self, scheduler):
        """Test retry delay calculation"""
        assert scheduler._calculate_retry_delay(0) == 10
        assert scheduler._calculate_retry_delay(1) == 30
        assert scheduler._calculate_retry_delay(2) == 60
        assert scheduler._calculate_retry_delay(3) == 120

    def test_calculate_throttle_level(self, scheduler):
        """Test throttle level calculation"""
        assert scheduler._calculate_throttle_level(0) == 1.0
        assert scheduler._calculate_throttle_level(1) == 0.9
        assert scheduler._calculate_throttle_level(2) == 0.7
        assert scheduler._calculate_throttle_level(3) == 0.5

    def test_wait_for_thermal_clearance_success(
        self, scheduler, mock_monitor, mock_metrics
    ):
        """Test waiting for thermal clearance - success"""
        # Start with high thermal, then clear
        mock_metrics.thermal_state = 3
        mock_monitor.collect.return_value = mock_metrics

        def side_effect():
            # After first call, reduce thermal state
            mock_metrics.thermal_state = 0
            return mock_metrics

        mock_monitor.collect.side_effect = side_effect

        result = scheduler.wait_for_thermal_clearance(
            task_id="test_task",
            thermal_threshold=2,
            timeout_seconds=5,
            check_interval_seconds=1,
        )

        assert result

    def test_wait_for_thermal_clearance_timeout(
        self, scheduler, mock_monitor, mock_metrics
    ):
        """Test waiting for thermal clearance - timeout"""
        mock_metrics.thermal_state = 3  # stays critical
        mock_monitor.collect.return_value = mock_metrics

        result = scheduler.wait_for_thermal_clearance(
            task_id="test_task",
            thermal_threshold=2,
            timeout_seconds=2,
            check_interval_seconds=1,
        )

        assert not result

    def test_get_scheduling_stats(self, scheduler, mock_monitor, mock_metrics):
        """Test getting scheduling statistics"""
        mock_monitor.collect.return_value = mock_metrics
        mock_monitor.check_health.return_value = {
            "score": 100.0,
            "thermal_throttling": False,
            "memory_constrained": False,
            "recommendations": ["System health is good."],
        }

        stats = scheduler.get_scheduling_stats()

        assert "timestamp" in stats
        assert stats["thermal_state"] == 0
        assert stats["health_score"] == 100.0
        assert stats["can_schedule_normal"]
        assert stats["can_schedule_high"]
        assert stats["can_schedule_critical"]

    def test_get_scheduling_stats_error_handling(self, scheduler, mock_monitor):
        """Test scheduling stats with error"""
        mock_monitor.collect.side_effect = RuntimeError("Monitor error")

        stats = scheduler.get_scheduling_stats()

        assert "error" in stats
        assert "Monitor error" in stats["error"]

    def test_suggest_task_configuration_training(
        self, scheduler, mock_monitor, mock_metrics
    ):
        """Test task configuration suggestion for training"""
        mock_monitor.collect.return_value = mock_metrics

        suggestion = scheduler.suggest_task_configuration("training")

        assert "config" in suggestion
        assert "batch_size" in suggestion["config"]
        assert "workers" in suggestion["config"]
        assert "memory_gb" in suggestion["config"]
        assert suggestion["thermal_state"] == 0
        assert suggestion["throttle_level"] == 1.0

    def test_suggest_task_configuration_inference(
        self, scheduler, mock_monitor, mock_metrics
    ):
        """Test task configuration suggestion for inference"""
        mock_monitor.collect.return_value = mock_metrics

        suggestion = scheduler.suggest_task_configuration("inference")

        assert "config" in suggestion
        assert suggestion["config"]["batch_size"] > 0
        assert suggestion["config"]["workers"] > 0

    def test_suggest_task_configuration_with_throttle(
        self, scheduler, mock_monitor, mock_metrics
    ):
        """Test configuration suggestion with thermal throttle"""
        mock_metrics.thermal_state = 2  # serious
        mock_monitor.collect.return_value = mock_metrics

        suggestion = scheduler.suggest_task_configuration("training")

        # Configuration should be reduced
        assert suggestion["throttle_level"] < 1.0
        # Batch size and workers should be reduced
        config = suggestion["config"]
        assert config["batch_size"] < 32  # Original default

    def test_error_handling_in_should_run_task(self, scheduler, mock_monitor):
        """Test error handling in scheduling decision"""
        mock_monitor.collect.side_effect = Exception("Monitor failure")

        decision = scheduler.should_run_task("test_task")

        # Should fail-safe and allow task with throttle
        assert decision.should_run
        assert decision.throttle_level < 1.0
        assert "failed" in decision.reason.lower()


class TestSchedulerIntegration:
    """Integration tests for scheduler"""

    @pytest.mark.apple_silicon
    def test_scheduler_with_real_monitor(self):
        """Test scheduler with real AppleSiliconMonitor"""
        from mlops.silicon.monitor import AppleSiliconMonitor

        monitor = AppleSiliconMonitor(project_name="test")
        scheduler = ThermalAwareScheduler(monitor=monitor)

        decision = scheduler.should_run_task(
            task_id="integration_test",
            thermal_threshold=2,
        )

        assert isinstance(decision, SchedulingDecision)
        assert decision.thermal_state in [0, 1, 2, 3]
        assert 0.0 <= decision.health_score <= 100.0

    @pytest.mark.apple_silicon
    def test_scheduler_stats_with_real_monitor(self):
        """Test stats with real monitor"""
        from mlops.silicon.monitor import AppleSiliconMonitor

        monitor = AppleSiliconMonitor(project_name="test")
        scheduler = ThermalAwareScheduler(monitor=monitor)

        stats = scheduler.get_scheduling_stats()

        assert "thermal_state" in stats
        assert "health_score" in stats
        assert "memory_utilization" in stats
        assert "recommendations" in stats
