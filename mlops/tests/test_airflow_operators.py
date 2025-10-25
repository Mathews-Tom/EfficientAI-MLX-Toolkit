"""Tests for Thermal-Aware Airflow Operators"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from mlops.airflow.operators import (
    ThermalAwareInferenceOperator,
    ThermalAwareMLXOperator,
    ThermalAwareOperatorMixin,
    ThermalAwareTrainingOperator,
    ThermalThrottleException,
    create_thermal_aware_task,
)
from mlops.airflow.thermal_scheduler import SchedulingDecision


@pytest.fixture
def mock_context():
    """Create mock Airflow context"""
    return {
        "task_instance_key_str": "test_task_123",
        "task_instance": MagicMock(),
    }


@pytest.fixture
def mock_scheduler():
    """Create mock ThermalAwareScheduler"""
    scheduler = MagicMock()
    return scheduler


@pytest.fixture
def mock_monitor():
    """Create mock AppleSiliconMonitor"""
    monitor = MagicMock()
    return monitor


class TestThermalAwareOperatorMixin:
    """Test ThermalAwareOperatorMixin"""

    def test_initialization(self):
        """Test operator mixin initialization"""
        # Create a simple class using the mixin
        class TestOperator(ThermalAwareOperatorMixin):
            pass

        operator = TestOperator(
            thermal_threshold=2,
            retry_on_thermal=True,
            memory_required_gb=8.0,
            priority="high",
        )

        assert operator.thermal_threshold == 2
        assert operator.retry_on_thermal
        assert operator.memory_required_gb == 8.0
        assert operator.priority == "high"
        assert operator.scheduler is not None
        assert operator.monitor is not None

    @patch("mlops.airflow.operators.ThermalAwareScheduler")
    def test_pre_execute_allowed(self, mock_scheduler_class, mock_context):
        """Test pre_execute when task is allowed"""
        # Setup mock
        mock_scheduler = MagicMock()
        mock_scheduler.should_run_task.return_value = SchedulingDecision(
            should_run=True,
            reason="Good conditions",
            throttle_level=1.0,
            thermal_state=0,
            health_score=100.0,
        )
        mock_scheduler_class.return_value = mock_scheduler

        class TestOperator(ThermalAwareOperatorMixin):
            pass

        operator = TestOperator(thermal_threshold=2)
        operator.scheduler = mock_scheduler

        # Should not raise exception
        operator.pre_execute(mock_context)

        # Should store decision in context
        assert "thermal_decision" in mock_context

    @patch("mlops.airflow.operators.ThermalAwareScheduler")
    def test_pre_execute_blocked_with_retry(self, mock_scheduler_class, mock_context):
        """Test pre_execute when task is blocked and retry enabled"""
        mock_scheduler = MagicMock()
        mock_scheduler.should_run_task.return_value = SchedulingDecision(
            should_run=False,
            reason="Thermal throttling",
            retry_after_seconds=60,
            thermal_state=3,
            health_score=40.0,
        )
        mock_scheduler.wait_for_thermal_clearance.return_value = False
        mock_scheduler_class.return_value = mock_scheduler

        class TestOperator(ThermalAwareOperatorMixin):
            pass

        operator = TestOperator(
            thermal_threshold=2,
            retry_on_thermal=True,
            wait_for_clearance=True,
        )
        operator.scheduler = mock_scheduler

        # Should raise ThermalThrottleException
        with pytest.raises(ThermalThrottleException):
            operator.pre_execute(mock_context)

    @patch("mlops.airflow.operators.ThermalAwareScheduler")
    def test_pre_execute_blocked_without_retry(
        self, mock_scheduler_class, mock_context
    ):
        """Test pre_execute when task is blocked and retry disabled"""
        mock_scheduler = MagicMock()
        mock_scheduler.should_run_task.return_value = SchedulingDecision(
            should_run=False,
            reason="Thermal throttling",
            retry_after_seconds=60,
            thermal_state=3,
            health_score=40.0,
        )
        mock_scheduler_class.return_value = mock_scheduler

        class TestOperator(ThermalAwareOperatorMixin):
            pass

        operator = TestOperator(
            thermal_threshold=2,
            retry_on_thermal=False,
            wait_for_clearance=False,
        )
        operator.scheduler = mock_scheduler

        # Should raise RuntimeError
        with pytest.raises(RuntimeError):
            operator.pre_execute(mock_context)

    @patch("mlops.airflow.operators.ThermalAwareScheduler")
    def test_pre_execute_clearance_success(self, mock_scheduler_class, mock_context):
        """Test pre_execute when waiting for clearance succeeds"""
        mock_scheduler = MagicMock()
        mock_scheduler.should_run_task.return_value = SchedulingDecision(
            should_run=False,
            reason="Thermal throttling",
            retry_after_seconds=60,
            thermal_state=3,
            health_score=40.0,
        )
        mock_scheduler.wait_for_thermal_clearance.return_value = True
        mock_scheduler_class.return_value = mock_scheduler

        class TestOperator(ThermalAwareOperatorMixin):
            pass

        operator = TestOperator(
            thermal_threshold=2,
            wait_for_clearance=True,
        )
        operator.scheduler = mock_scheduler

        # Should not raise exception after clearance
        operator.pre_execute(mock_context)

    @patch("mlops.airflow.operators.AppleSiliconMonitor")
    def test_post_execute(self, mock_monitor_class, mock_context):
        """Test post_execute logging"""
        mock_monitor = MagicMock()
        mock_metrics = MagicMock()
        mock_metrics.thermal_state = 0
        mock_metrics.get_health_score.return_value = 100.0
        mock_metrics.to_dict.return_value = {"thermal_state": 0}
        mock_monitor.collect.return_value = mock_metrics
        mock_monitor_class.return_value = mock_monitor

        class TestOperator(ThermalAwareOperatorMixin):
            pass

        operator = TestOperator()
        operator.monitor = mock_monitor

        # Should log metrics without error
        operator.post_execute(mock_context, result="success")

        # Should push metrics to XCom
        mock_context["task_instance"].xcom_push.assert_called_once()


class TestThermalAwareMLXOperator:
    """Test ThermalAwareMLXOperator"""

    def test_initialization(self):
        """Test MLX operator initialization"""

        def dummy_function():
            return "success"

        operator = ThermalAwareMLXOperator(
            task_id="test_task",
            python_callable=dummy_function,
            thermal_threshold=2,
            priority="high",
        )

        assert operator.task_id == "test_task"
        assert operator.python_callable == dummy_function
        assert operator.thermal_threshold == 2
        assert operator.priority == "high"

    @patch("mlops.airflow.operators.ThermalAwareScheduler")
    @patch("mlops.airflow.operators.AppleSiliconMonitor")
    def test_execute_success(
        self, mock_monitor_class, mock_scheduler_class, mock_context
    ):
        """Test successful task execution"""
        # Setup mocks
        mock_scheduler = MagicMock()
        mock_scheduler.should_run_task.return_value = SchedulingDecision(
            should_run=True,
            reason="Good conditions",
            throttle_level=1.0,
            thermal_state=0,
            health_score=100.0,
        )
        mock_scheduler_class.return_value = mock_scheduler

        mock_monitor = MagicMock()
        mock_metrics = MagicMock()
        mock_metrics.thermal_state = 0
        mock_metrics.get_health_score.return_value = 100.0
        mock_metrics.to_dict.return_value = {}
        mock_monitor.collect.return_value = mock_metrics
        mock_monitor_class.return_value = mock_monitor

        def test_function():
            return "success"

        operator = ThermalAwareMLXOperator(
            task_id="test_task",
            python_callable=test_function,
        )
        operator.scheduler = mock_scheduler
        operator.monitor = mock_monitor

        result = operator.execute(mock_context)

        assert result == "success"

    @patch("mlops.airflow.operators.ThermalAwareScheduler")
    @patch("mlops.airflow.operators.AppleSiliconMonitor")
    def test_execute_with_throttle(
        self, mock_monitor_class, mock_scheduler_class, mock_context
    ):
        """Test task execution with throttling"""
        mock_scheduler = MagicMock()
        mock_scheduler.should_run_task.return_value = SchedulingDecision(
            should_run=True,
            reason="Throttled",
            throttle_level=0.7,
            thermal_state=2,
            health_score=70.0,
        )
        mock_scheduler_class.return_value = mock_scheduler

        mock_monitor = MagicMock()
        mock_metrics = MagicMock()
        mock_metrics.thermal_state = 2
        mock_metrics.get_health_score.return_value = 70.0
        mock_metrics.to_dict.return_value = {}
        mock_monitor.collect.return_value = mock_metrics
        mock_monitor_class.return_value = mock_monitor

        def test_function(throttle_level=1.0):
            return f"throttle={throttle_level}"

        operator = ThermalAwareMLXOperator(
            task_id="test_task",
            python_callable=test_function,
        )
        operator.scheduler = mock_scheduler
        operator.monitor = mock_monitor

        # Add decision to context
        mock_context["thermal_decision"] = mock_scheduler.should_run_task.return_value

        result = operator.execute(mock_context)

        assert "throttle=0.7" in result


class TestThermalAwareTrainingOperator:
    """Test ThermalAwareTrainingOperator"""

    def test_initialization(self):
        """Test training operator initialization"""
        operator = ThermalAwareTrainingOperator(
            task_id="train_task",
            model_name="llama-7b",
            dataset_path="/data/train.jsonl",
            output_path="/models/output",
        )

        assert operator.task_id == "train_task"
        assert operator.model_name == "llama-7b"
        assert operator.dataset_path == "/data/train.jsonl"
        assert operator.output_path == "/models/output"
        # Should have training defaults
        assert operator.thermal_threshold == 2
        assert operator.priority == "high"
        assert operator.memory_required_gb == 16.0

    @patch("mlops.airflow.operators.ThermalAwareScheduler")
    @patch("mlops.airflow.operators.AppleSiliconMonitor")
    def test_train_model(
        self, mock_monitor_class, mock_scheduler_class, mock_context
    ):
        """Test training execution"""
        mock_scheduler = MagicMock()
        mock_scheduler.should_run_task.return_value = SchedulingDecision(
            should_run=True,
            reason="Good conditions",
            throttle_level=1.0,
            thermal_state=0,
            health_score=100.0,
        )
        mock_scheduler.suggest_task_configuration.return_value = {
            "config": {
                "batch_size": 32,
                "workers": 4,
                "memory_gb": 16.0,
            },
            "thermal_state": 0,
            "throttle_level": 1.0,
        }
        mock_scheduler_class.return_value = mock_scheduler

        mock_monitor = MagicMock()
        mock_metrics = MagicMock()
        mock_metrics.thermal_state = 0
        mock_metrics.get_health_score.return_value = 100.0
        mock_metrics.to_dict.return_value = {}
        mock_monitor.collect.return_value = mock_metrics
        mock_monitor_class.return_value = mock_monitor

        operator = ThermalAwareTrainingOperator(
            task_id="train_task",
            model_name="llama-7b",
            dataset_path="/data/train.jsonl",
        )
        operator.scheduler = mock_scheduler
        operator.monitor = mock_monitor

        mock_context["thermal_decision"] = mock_scheduler.should_run_task.return_value

        result = operator.execute(mock_context)

        assert result["model_name"] == "llama-7b"
        assert result["status"] == "success"
        assert "config" in result


class TestThermalAwareInferenceOperator:
    """Test ThermalAwareInferenceOperator"""

    def test_initialization(self):
        """Test inference operator initialization"""
        operator = ThermalAwareInferenceOperator(
            task_id="inference_task",
            model_path="/models/llama-7b",
            input_data=["prompt1", "prompt2"],
        )

        assert operator.task_id == "inference_task"
        assert operator.model_path == "/models/llama-7b"
        assert operator.input_data == ["prompt1", "prompt2"]
        # Should have inference defaults
        assert operator.thermal_threshold == 1
        assert operator.priority == "normal"
        assert operator.memory_required_gb == 8.0

    @patch("mlops.airflow.operators.ThermalAwareScheduler")
    @patch("mlops.airflow.operators.AppleSiliconMonitor")
    def test_run_inference(
        self, mock_monitor_class, mock_scheduler_class, mock_context
    ):
        """Test inference execution"""
        mock_scheduler = MagicMock()
        mock_scheduler.should_run_task.return_value = SchedulingDecision(
            should_run=True,
            reason="Good conditions",
            throttle_level=1.0,
            thermal_state=0,
            health_score=100.0,
        )
        mock_scheduler.suggest_task_configuration.return_value = {
            "config": {
                "batch_size": 64,
                "workers": 2,
                "memory_gb": 8.0,
            },
            "thermal_state": 0,
            "throttle_level": 1.0,
        }
        mock_scheduler_class.return_value = mock_scheduler

        mock_monitor = MagicMock()
        mock_metrics = MagicMock()
        mock_metrics.thermal_state = 0
        mock_metrics.get_health_score.return_value = 100.0
        mock_metrics.to_dict.return_value = {}
        mock_monitor.collect.return_value = mock_metrics
        mock_monitor_class.return_value = mock_monitor

        operator = ThermalAwareInferenceOperator(
            task_id="inference_task",
            model_path="/models/llama-7b",
            input_data=["test"],
        )
        operator.scheduler = mock_scheduler
        operator.monitor = mock_monitor

        mock_context["thermal_decision"] = mock_scheduler.should_run_task.return_value

        result = operator.execute(mock_context)

        assert result["model_path"] == "/models/llama-7b"
        assert result["status"] == "success"


class TestFactoryFunction:
    """Test factory function"""

    def test_create_thermal_aware_task(self):
        """Test create_thermal_aware_task factory"""

        def dummy_function():
            return "success"

        task = create_thermal_aware_task(
            task_id="test_task",
            python_callable=dummy_function,
            thermal_threshold=2,
            priority="high",
        )

        assert isinstance(task, ThermalAwareMLXOperator)
        assert task.task_id == "test_task"
        assert task.thermal_threshold == 2
        assert task.priority == "high"
