"""Model Performance Monitoring

Monitors model performance metrics including accuracy, precision, recall, F1,
latency, and memory usage with threshold-based alerting.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from evidently.metric_preset import ClassificationPreset, RegressionPreset
from evidently.report import Report

from mlops.monitoring.evidently.config import PerformanceThresholds

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Model performance metrics

    Attributes:
        timestamp: When metrics were collected
        accuracy: Model accuracy (0.0-1.0)
        precision: Model precision (0.0-1.0)
        recall: Model recall (0.0-1.0)
        f1_score: Model F1 score (0.0-1.0)
        latency_ms: Inference latency in milliseconds
        memory_mb: Memory usage in MB
        throughput_qps: Queries per second
        error_rate: Error rate (0.0-1.0)
        total_predictions: Total number of predictions
        degraded: Whether performance is degraded
        degradation_reasons: List of degradation reasons
        additional_metrics: Additional custom metrics
    """

    timestamp: datetime
    accuracy: float | None = None
    precision: float | None = None
    recall: float | None = None
    f1_score: float | None = None
    latency_ms: float | None = None
    memory_mb: float | None = None
    throughput_qps: float | None = None
    error_rate: float | None = None
    total_predictions: int = 0
    degraded: bool = False
    degradation_reasons: list[str] = field(default_factory=list)
    additional_metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary

        Returns:
            Dictionary representation of performance metrics
        """
        return {
            "timestamp": self.timestamp.isoformat(),
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "latency_ms": self.latency_ms,
            "memory_mb": self.memory_mb,
            "throughput_qps": self.throughput_qps,
            "error_rate": self.error_rate,
            "total_predictions": self.total_predictions,
            "degraded": self.degraded,
            "degradation_reasons": self.degradation_reasons,
            "additional_metrics": self.additional_metrics,
        }


class PerformanceMonitor:
    """Model performance monitor using Evidently

    This monitor tracks model performance over time and detects degradation:
    - Classification metrics (accuracy, precision, recall, F1)
    - Regression metrics (MAE, MSE, RMSE, R²)
    - Operational metrics (latency, memory, throughput)
    - Threshold-based degradation detection
    """

    def __init__(
        self,
        project_name: str,
        task_type: str = "classification",
        thresholds: PerformanceThresholds | None = None,
        workspace_path: Path | str | None = None,
    ):
        """Initialize performance monitor

        Args:
            project_name: Name of the project being monitored
            task_type: Type of ML task ('classification' or 'regression')
            thresholds: Performance thresholds for degradation detection
            workspace_path: Path to workspace for saving reports
        """
        self.project_name = project_name
        self.task_type = task_type
        self.thresholds = thresholds or PerformanceThresholds()
        self.workspace_path = (
            Path(workspace_path) if workspace_path else Path("mlops/monitoring/workspace")
        )
        self.workspace_path.mkdir(parents=True, exist_ok=True)

        # Reference performance storage
        self._reference_data: pd.DataFrame | None = None

        logger.info(
            "Initialized PerformanceMonitor for project: %s (task: %s, workspace: %s)",
            project_name,
            task_type,
            self.workspace_path,
        )

    def set_reference_data(
        self, reference_data: pd.DataFrame, target_column: str, prediction_column: str
    ) -> None:
        """Set reference data for performance monitoring

        Args:
            reference_data: Reference dataset with predictions and targets
            target_column: Name of target column
            prediction_column: Name of prediction column

        Raises:
            ValueError: If reference data is invalid
        """
        if reference_data.empty:
            raise ValueError("Reference data cannot be empty")

        if target_column not in reference_data.columns:
            raise ValueError(f"Target column '{target_column}' not found in reference data")

        if prediction_column not in reference_data.columns:
            raise ValueError(
                f"Prediction column '{prediction_column}' not found in reference data"
            )

        self._reference_data = reference_data.copy()
        self._target_column = target_column
        self._prediction_column = prediction_column

        logger.info(
            "Set reference data: %d rows (target: %s, prediction: %s)",
            len(reference_data),
            target_column,
            prediction_column,
        )

    def monitor_performance(
        self,
        current_data: pd.DataFrame,
        target_column: str,
        prediction_column: str,
        latency_ms: float | None = None,
        memory_mb: float | None = None,
    ) -> PerformanceMetrics:
        """Monitor model performance on current data

        Args:
            current_data: Current predictions with targets
            target_column: Name of target column
            prediction_column: Name of prediction column
            latency_ms: Average inference latency in milliseconds
            memory_mb: Memory usage in MB

        Returns:
            PerformanceMetrics with current performance

        Raises:
            ValueError: If data is invalid
        """
        if current_data.empty:
            raise ValueError("Current data cannot be empty")

        if target_column not in current_data.columns:
            raise ValueError(f"Target column '{target_column}' not found in current data")

        if prediction_column not in current_data.columns:
            raise ValueError(f"Prediction column '{prediction_column}' not found in current data")

        try:
            logger.info("Monitoring performance on %d predictions", len(current_data))

            # Create Evidently performance report
            if self.task_type == "classification":
                report = Report(metrics=[ClassificationPreset()])
            else:
                report = Report(metrics=[RegressionPreset()])

            # Run report
            if self._reference_data is not None:
                report.run(
                    reference_data=self._reference_data,
                    current_data=current_data,
                )
            else:
                # Run without reference for initial monitoring
                report.run(
                    reference_data=current_data,
                    current_data=current_data,
                )

            # Extract metrics
            result = report.as_dict()
            metrics_data = result["metrics"][0]["result"]["current"]

            # Create performance metrics
            performance = PerformanceMetrics(
                timestamp=datetime.now(),
                total_predictions=len(current_data),
                latency_ms=latency_ms,
                memory_mb=memory_mb,
            )

            # Extract task-specific metrics
            if self.task_type == "classification":
                performance.accuracy = metrics_data.get("accuracy")
                performance.precision = metrics_data.get("precision")
                performance.recall = metrics_data.get("recall")
                performance.f1_score = metrics_data.get("f1")
            else:
                performance.additional_metrics["mae"] = metrics_data.get("mean_abs_error")
                performance.additional_metrics["mse"] = metrics_data.get("mean_error")
                performance.additional_metrics["rmse"] = metrics_data.get("rmse")
                performance.additional_metrics["r2"] = metrics_data.get("r2_score")

            # Check for performance degradation
            self._check_degradation(performance)

            # Save report
            self._save_report(report, performance)

            logger.info(
                "Performance monitoring complete: degraded=%s, reasons=%s",
                performance.degraded,
                performance.degradation_reasons,
            )

            return performance

        except Exception as e:
            logger.error("Performance monitoring failed: %s", e)
            raise RuntimeError(f"Performance monitoring failed: {e}") from e

    def _check_degradation(self, metrics: PerformanceMetrics) -> None:
        """Check if performance is degraded based on thresholds

        Args:
            metrics: Performance metrics to check
        """
        degradation_reasons = []

        # Check accuracy
        if metrics.accuracy is not None and metrics.accuracy < self.thresholds.accuracy_threshold:
            degradation_reasons.append(
                f"Accuracy {metrics.accuracy:.3f} < threshold {self.thresholds.accuracy_threshold}"
            )

        # Check precision
        if (
            metrics.precision is not None
            and metrics.precision < self.thresholds.precision_threshold
        ):
            degradation_reasons.append(
                f"Precision {metrics.precision:.3f} < threshold {self.thresholds.precision_threshold}"
            )

        # Check recall
        if metrics.recall is not None and metrics.recall < self.thresholds.recall_threshold:
            degradation_reasons.append(
                f"Recall {metrics.recall:.3f} < threshold {self.thresholds.recall_threshold}"
            )

        # Check F1 score
        if metrics.f1_score is not None and metrics.f1_score < self.thresholds.f1_threshold:
            degradation_reasons.append(
                f"F1 score {metrics.f1_score:.3f} < threshold {self.thresholds.f1_threshold}"
            )

        # Check latency
        if (
            metrics.latency_ms is not None
            and metrics.latency_ms > self.thresholds.latency_threshold_ms
        ):
            degradation_reasons.append(
                f"Latency {metrics.latency_ms:.1f}ms > threshold {self.thresholds.latency_threshold_ms}ms"
            )

        # Check memory
        if metrics.memory_mb is not None and metrics.memory_mb > self.thresholds.memory_threshold_mb:
            degradation_reasons.append(
                f"Memory {metrics.memory_mb:.1f}MB > threshold {self.thresholds.memory_threshold_mb}MB"
            )

        metrics.degraded = len(degradation_reasons) > 0
        metrics.degradation_reasons = degradation_reasons

    def _save_report(self, evidently_report: Report, metrics: PerformanceMetrics) -> None:
        """Save performance report to workspace

        Args:
            evidently_report: Evidently report object
            metrics: Performance metrics
        """
        try:
            # Create project-specific directory
            project_dir = self.workspace_path / self.project_name / "performance"
            project_dir.mkdir(parents=True, exist_ok=True)

            # Save Evidently HTML report
            timestamp_str = metrics.timestamp.strftime("%Y%m%d_%H%M%S")
            html_path = project_dir / f"performance_report_{timestamp_str}.html"
            evidently_report.save_html(str(html_path))

            logger.debug("Saved performance report to: %s", html_path)

        except Exception as e:
            logger.warning("Failed to save performance report: %s", e)

    def get_reference_data(self) -> pd.DataFrame | None:
        """Get current reference data

        Returns:
            Reference data DataFrame or None if not set
        """
        return self._reference_data.copy() if self._reference_data is not None else None
