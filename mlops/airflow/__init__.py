"""Thermal-Aware Airflow Integration for Apple Silicon

This module provides Airflow operators and schedulers that intelligently
manage task execution based on Apple Silicon thermal state, memory pressure,
and power mode.

Components:
    - ThermalAwareScheduler: Core scheduling logic with thermal monitoring
    - ThermalAwareOperatorMixin: Base mixin for thermal-aware operators
    - ThermalAwareMLXOperator: General-purpose thermal-aware operator
    - ThermalAwareTrainingOperator: Optimized for model training
    - ThermalAwareInferenceOperator: Optimized for model inference

Example:
    >>> from mlops.airflow import ThermalAwareScheduler, ThermalAwareTrainingOperator
    >>>
    >>> # Check if we should run a task
    >>> scheduler = ThermalAwareScheduler()
    >>> decision = scheduler.should_run_task("train_model", thermal_threshold=2)
    >>>
    >>> # Create thermal-aware training task
    >>> task = ThermalAwareTrainingOperator(
    ...     task_id="train_llm",
    ...     model_name="llama-7b",
    ...     dataset_path="/data/train.jsonl",
    ...     thermal_threshold=2,
    ...     priority="high",
    ... )
"""

from mlops.airflow.operators import (
    ThermalAwareInferenceOperator,
    ThermalAwareMLXOperator,
    ThermalAwareOperatorMixin,
    ThermalAwareTrainingOperator,
    ThermalThrottleException,
    create_thermal_aware_task,
)
from mlops.airflow.thermal_scheduler import (
    SchedulingDecision,
    ThermalAwareScheduler,
    ThermalThresholds,
)

__all__ = [
    # Scheduler
    "ThermalAwareScheduler",
    "SchedulingDecision",
    "ThermalThresholds",
    # Operators
    "ThermalAwareOperatorMixin",
    "ThermalAwareMLXOperator",
    "ThermalAwareTrainingOperator",
    "ThermalAwareInferenceOperator",
    "create_thermal_aware_task",
    # Exceptions
    "ThermalThrottleException",
]
