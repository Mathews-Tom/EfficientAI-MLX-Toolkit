"""Custom Airflow Operators with Thermal-Aware Execution

This module provides custom Airflow operators that integrate with ThermalAwareScheduler
to make intelligent scheduling decisions based on Apple Silicon thermal state.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable

from mlops.airflow.thermal_scheduler import (
    SchedulingDecision,
    ThermalAwareScheduler,
    ThermalThresholds,
)
from mlops.silicon.monitor import AppleSiliconMonitor

logger = logging.getLogger(__name__)


class ThermalAwareOperatorMixin:
    """Mixin class for thermal-aware Airflow operators

    This mixin provides thermal-aware execution capabilities that can be
    added to any Airflow operator. It integrates with ThermalAwareScheduler
    to check thermal conditions before executing tasks.
    """

    def __init__(
        self,
        thermal_threshold: int = 2,
        retry_on_thermal: bool = True,
        memory_required_gb: float | None = None,
        priority: str = "normal",
        wait_for_clearance: bool = True,
        clearance_timeout_seconds: int = 300,
        **kwargs: Any,
    ) -> None:
        """Initialize thermal-aware operator

        Args:
            thermal_threshold: Maximum thermal state allowed (0-3)
            retry_on_thermal: Whether to retry task if thermal state too high
            memory_required_gb: Memory required for task execution
            priority: Task priority (low, normal, high, critical)
            wait_for_clearance: Whether to wait for thermal clearance
            clearance_timeout_seconds: Max time to wait for clearance
            **kwargs: Additional operator arguments
        """
        self.thermal_threshold = thermal_threshold
        self.retry_on_thermal = retry_on_thermal
        self.memory_required_gb = memory_required_gb
        self.priority = priority
        self.wait_for_clearance = wait_for_clearance
        self.clearance_timeout_seconds = clearance_timeout_seconds

        self.scheduler = ThermalAwareScheduler()
        self.monitor = AppleSiliconMonitor(project_name="airflow")

        super().__init__(**kwargs)

        logger.info(
            "ThermalAwareOperator initialized: threshold=%d, priority=%s, wait=%s",
            thermal_threshold,
            priority,
            wait_for_clearance,
        )

    def pre_execute(self, context: dict[str, Any]) -> None:
        """Check thermal conditions before executing task

        Args:
            context: Airflow task context

        Raises:
            ThermalThrottleException: If thermal state too high and retry requested
        """
        task_id = context.get("task_instance_key_str", "unknown")

        logger.info("Pre-execute thermal check for task: %s", task_id)

        # Check if we should run
        decision = self.scheduler.should_run_task(
            task_id=task_id,
            thermal_threshold=self.thermal_threshold,
            memory_required_gb=self.memory_required_gb,
            priority=self.priority,
        )

        if not decision.should_run:
            logger.warning("Task %s blocked: %s", task_id, decision.reason)

            # Wait for clearance if enabled
            if self.wait_for_clearance:
                cleared = self.scheduler.wait_for_thermal_clearance(
                    task_id=task_id,
                    thermal_threshold=self.thermal_threshold,
                    timeout_seconds=self.clearance_timeout_seconds,
                )
                if cleared:
                    logger.info("Task %s cleared after waiting", task_id)
                    return

            # Raise exception to trigger retry
            if self.retry_on_thermal:
                raise ThermalThrottleException(
                    f"Task blocked due to thermal constraints: {decision.reason}"
                )
            else:
                raise RuntimeError(
                    f"Task cannot run due to thermal constraints: {decision.reason}"
                )

        # Log thermal decision
        logger.info(
            "Task %s approved: throttle=%.2f, thermal=%d, health=%.1f",
            task_id,
            decision.throttle_level,
            decision.thermal_state,
            decision.health_score,
        )

        # Store decision in context for task execution
        context["thermal_decision"] = decision

    def post_execute(self, context: dict[str, Any], result: Any = None) -> None:
        """Log metrics after task execution

        Args:
            context: Airflow task context
            result: Task execution result
        """
        task_id = context.get("task_instance_key_str", "unknown")

        try:
            # Collect post-execution metrics
            metrics = self.monitor.collect()

            logger.info(
                "Post-execute metrics for %s: thermal=%d, health=%.1f",
                task_id,
                metrics.thermal_state,
                metrics.get_health_score(),
            )

            # Store metrics in XCom for monitoring
            if "task_instance" in context:
                context["task_instance"].xcom_push(
                    key="thermal_metrics",
                    value=metrics.to_dict(),
                )

        except Exception as e:
            logger.error("Failed to collect post-execute metrics: %s", e)


class ThermalThrottleException(Exception):
    """Exception raised when task is throttled due to thermal constraints"""
    pass


class ThermalAwareMLXOperator(ThermalAwareOperatorMixin):
    """Thermal-aware operator for MLX workloads

    This operator is optimized for Apple Silicon MLX workloads with
    thermal monitoring and intelligent scheduling.

    Example:
        >>> from mlops.airflow.operators import ThermalAwareMLXOperator
        >>>
        >>> task = ThermalAwareMLXOperator(
        ...     task_id="train_model",
        ...     thermal_threshold=2,
        ...     memory_required_gb=16.0,
        ...     priority="high",
        ...     python_callable=train_function,
        ... )
    """

    def __init__(
        self,
        task_id: str,
        python_callable: Callable[..., Any],
        op_args: tuple[Any, ...] | None = None,
        op_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize MLX operator

        Args:
            task_id: Unique task identifier
            python_callable: Function to execute
            op_args: Positional arguments for callable
            op_kwargs: Keyword arguments for callable
            **kwargs: ThermalAwareOperatorMixin arguments
        """
        self.task_id = task_id
        self.python_callable = python_callable
        self.op_args = op_args or ()
        self.op_kwargs = op_kwargs or {}

        super().__init__(**kwargs)

    def execute(self, context: dict[str, Any]) -> Any:
        """Execute the task with thermal monitoring

        Args:
            context: Airflow task context

        Returns:
            Result from python_callable
        """
        # Pre-execute checks
        self.pre_execute(context)

        try:
            # Get thermal decision from context
            decision = context.get("thermal_decision")

            # Apply throttling if needed
            if decision and decision.throttle_level < 1.0:
                logger.info(
                    "Applying throttle %.2f to task %s",
                    decision.throttle_level,
                    self.task_id,
                )
                # Pass throttle info to callable if it accepts it
                if "throttle_level" in self.python_callable.__code__.co_varnames:
                    self.op_kwargs["throttle_level"] = decision.throttle_level

            # Execute the task
            logger.info("Executing task: %s", self.task_id)
            result = self.python_callable(*self.op_args, **self.op_kwargs)

            # Post-execute logging
            self.post_execute(context, result)

            return result

        except Exception as e:
            logger.error("Task %s failed: %s", self.task_id, e)
            raise


class ThermalAwareTrainingOperator(ThermalAwareMLXOperator):
    """Thermal-aware operator optimized for model training

    This operator applies training-specific optimizations and thermal
    monitoring for long-running training tasks.

    Example:
        >>> from mlops.airflow.operators import ThermalAwareTrainingOperator
        >>>
        >>> task = ThermalAwareTrainingOperator(
        ...     task_id="train_llm",
        ...     model_name="llama-7b",
        ...     dataset_path="/data/train.jsonl",
        ...     thermal_threshold=2,
        ...     priority="high",
        ... )
    """

    def __init__(
        self,
        model_name: str,
        dataset_path: str,
        output_path: str | None = None,
        training_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize training operator

        Args:
            model_name: Model identifier or path
            dataset_path: Path to training dataset
            output_path: Path to save trained model
            training_config: Training configuration
            **kwargs: ThermalAwareMLXOperator arguments
        """
        self.model_name = model_name
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.training_config = training_config or {}

        # Training tasks need more resources by default
        kwargs.setdefault("thermal_threshold", 2)
        kwargs.setdefault("priority", "high")
        kwargs.setdefault("memory_required_gb", 16.0)

        # Use training function as callable
        super().__init__(
            python_callable=self._train_model,
            **kwargs,
        )

    def _train_model(self, throttle_level: float = 1.0, **context: Any) -> dict[str, Any]:
        """Execute training with thermal throttling

        Args:
            throttle_level: Throttle level (0.0-1.0) from scheduler
            **context: Additional context

        Returns:
            Training results
        """
        logger.info(
            "Starting training: model=%s, throttle=%.2f",
            self.model_name,
            throttle_level,
        )

        # Get optimal configuration based on thermal state
        suggested_config = self.scheduler.suggest_task_configuration("training")

        # Merge with user config
        config = {**suggested_config["config"], **self.training_config}

        # Apply throttle to batch size and workers
        config["batch_size"] = int(config["batch_size"] * throttle_level)
        config["workers"] = max(1, int(config["workers"] * throttle_level))

        logger.info("Training config: %s", config)

        # This is a placeholder - actual training logic would be implemented here
        # or passed as python_callable
        return {
            "model_name": self.model_name,
            "config": config,
            "throttle_level": throttle_level,
            "status": "success",
        }


class ThermalAwareInferenceOperator(ThermalAwareMLXOperator):
    """Thermal-aware operator optimized for model inference

    This operator is designed for inference workloads with lower thermal
    thresholds and faster retry cycles.

    Example:
        >>> from mlops.airflow.operators import ThermalAwareInferenceOperator
        >>>
        >>> task = ThermalAwareInferenceOperator(
        ...     task_id="run_inference",
        ...     model_path="/models/llama-7b",
        ...     input_data=data,
        ...     thermal_threshold=1,
        ... )
    """

    def __init__(
        self,
        model_path: str,
        input_data: Any,
        inference_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize inference operator

        Args:
            model_path: Path to trained model
            input_data: Input data for inference
            inference_config: Inference configuration
            **kwargs: ThermalAwareMLXOperator arguments
        """
        self.model_path = model_path
        self.input_data = input_data
        self.inference_config = inference_config or {}

        # Inference tasks are less resource intensive
        kwargs.setdefault("thermal_threshold", 1)
        kwargs.setdefault("priority", "normal")
        kwargs.setdefault("memory_required_gb", 8.0)

        super().__init__(
            python_callable=self._run_inference,
            **kwargs,
        )

    def _run_inference(self, throttle_level: float = 1.0, **context: Any) -> dict[str, Any]:
        """Execute inference with thermal throttling

        Args:
            throttle_level: Throttle level (0.0-1.0) from scheduler
            **context: Additional context

        Returns:
            Inference results
        """
        logger.info(
            "Starting inference: model=%s, throttle=%.2f",
            self.model_path,
            throttle_level,
        )

        # Get optimal configuration
        suggested_config = self.scheduler.suggest_task_configuration("inference")
        config = {**suggested_config["config"], **self.inference_config}

        # Apply throttle to batch size
        config["batch_size"] = int(config["batch_size"] * throttle_level)

        logger.info("Inference config: %s", config)

        # Placeholder for actual inference logic
        return {
            "model_path": self.model_path,
            "config": config,
            "throttle_level": throttle_level,
            "status": "success",
        }


def create_thermal_aware_task(
    task_id: str,
    python_callable: Callable[..., Any],
    thermal_threshold: int = 2,
    priority: str = "normal",
    **kwargs: Any,
) -> ThermalAwareMLXOperator:
    """Factory function to create thermal-aware tasks

    Args:
        task_id: Unique task identifier
        python_callable: Function to execute
        thermal_threshold: Maximum thermal state (0-3)
        priority: Task priority
        **kwargs: Additional operator arguments

    Returns:
        ThermalAwareMLXOperator instance
    """
    return ThermalAwareMLXOperator(
        task_id=task_id,
        python_callable=python_callable,
        thermal_threshold=thermal_threshold,
        priority=priority,
        **kwargs,
    )
