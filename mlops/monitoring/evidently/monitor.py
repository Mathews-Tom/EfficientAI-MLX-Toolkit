"""Evidently Monitoring System

Unified monitoring system integrating drift detection, performance monitoring,
Apple Silicon metrics, and alert management.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from mlops.monitoring.evidently.alert_manager import AlertConfig, AlertManager
from mlops.monitoring.evidently.apple_silicon_metrics import (
    AppleSiliconMetrics,
    AppleSiliconMetricsCollector,
)
from mlops.monitoring.evidently.config import (
    DriftDetectionConfig,
    EvidentlyConfig,
    PerformanceThresholds,
)
from mlops.monitoring.evidently.drift_detector import DriftDetector, DriftReport
from mlops.monitoring.evidently.performance_monitor import (
    PerformanceMetrics,
    PerformanceMonitor,
)

logger = logging.getLogger(__name__)


class EvidentlyMonitor:
    """Unified Evidently monitoring system

    Integrates all monitoring components:
    - Data drift detection
    - Model performance monitoring
    - Apple Silicon metrics collection
    - Alert management
    - Automated retraining suggestions

    This provides a single interface for comprehensive model monitoring
    across all toolkit projects.
    """

    def __init__(
        self,
        project_name: str,
        config: EvidentlyConfig | None = None,
        drift_config: DriftDetectionConfig | None = None,
        performance_thresholds: PerformanceThresholds | None = None,
        alert_config: AlertConfig | None = None,
    ):
        """Initialize Evidently monitoring system

        Args:
            project_name: Name of the project being monitored
            config: Main Evidently configuration
            drift_config: Drift detection configuration
            performance_thresholds: Performance monitoring thresholds
            alert_config: Alert configuration
        """
        self.project_name = project_name
        self.config = config or EvidentlyConfig(project_name=project_name)

        # Initialize components
        self.drift_detector = DriftDetector(
            project_name=project_name,
            config=drift_config,
            workspace_path=self.config.workspace_path,
        ) if self.config.drift_detection_enabled else None

        self.performance_monitor = PerformanceMonitor(
            project_name=project_name,
            thresholds=performance_thresholds,
            workspace_path=self.config.workspace_path,
        ) if self.config.performance_monitoring_enabled else None

        self.apple_silicon_metrics = AppleSiliconMetricsCollector(
            project_name=project_name,
        ) if self.config.apple_silicon_metrics_enabled else None

        self.alert_manager = AlertManager(
            project_name=project_name,
            config=alert_config,
            workspace_path=self.config.workspace_path,
        ) if self.config.alert_enabled else None

        logger.info(
            "Initialized EvidentlyMonitor for project: %s (drift=%s, performance=%s, apple_silicon=%s, alerts=%s)",
            project_name,
            self.config.drift_detection_enabled,
            self.config.performance_monitoring_enabled,
            self.config.apple_silicon_metrics_enabled,
            self.config.alert_enabled,
        )

    def set_reference_data(
        self,
        reference_data: pd.DataFrame,
        target_column: str | None = None,
        prediction_column: str | None = None,
    ) -> None:
        """Set reference data for drift and performance monitoring

        Args:
            reference_data: Reference dataset (training or baseline data)
            target_column: Name of target column (for performance monitoring)
            prediction_column: Name of prediction column (for performance monitoring)

        Raises:
            ValueError: If reference data is invalid
        """
        if self.drift_detector:
            self.drift_detector.set_reference_data(reference_data)

        if self.performance_monitor and target_column and prediction_column:
            self.performance_monitor.set_reference_data(
                reference_data, target_column, prediction_column
            )

        logger.info("Set reference data: %d rows", len(reference_data))

    def monitor(
        self,
        current_data: pd.DataFrame,
        target_column: str | None = None,
        prediction_column: str | None = None,
        latency_ms: float | None = None,
        memory_mb: float | None = None,
    ) -> dict[str, Any]:
        """Run comprehensive monitoring on current data

        Args:
            current_data: Current data to monitor
            target_column: Name of target column (for performance monitoring)
            prediction_column: Name of prediction column (for performance monitoring)
            latency_ms: Average inference latency in milliseconds
            memory_mb: Memory usage in MB

        Returns:
            Dictionary with monitoring results:
            - drift_report: Drift detection results (if enabled)
            - performance_metrics: Performance metrics (if enabled)
            - apple_silicon_metrics: Hardware metrics (if enabled)
            - alerts: List of created alerts

        Raises:
            ValueError: If monitoring fails
        """
        results: dict[str, Any] = {
            "project_name": self.project_name,
            "monitoring_enabled": self.config.monitoring_enabled,
        }

        if not self.config.monitoring_enabled:
            logger.info("Monitoring disabled for project: %s", self.project_name)
            return results

        try:
            logger.info("Running monitoring for project: %s", self.project_name)

            # Drift detection
            drift_report: DriftReport | None = None
            if self.drift_detector:
                try:
                    drift_report = self.drift_detector.detect_drift(current_data)
                    results["drift_report"] = drift_report.to_dict()

                    # Create alert if drift detected
                    if drift_report.dataset_drift and self.alert_manager:
                        alert = self.alert_manager.create_drift_alert(
                            drift_share=drift_report.drift_share,
                            drifted_features=drift_report.drifted_features,
                            total_features=drift_report.total_features,
                        )
                        results.setdefault("alerts", []).append(alert.to_dict())

                except Exception as e:
                    logger.error("Drift detection failed: %s", e)
                    results["drift_error"] = str(e)

            # Performance monitoring
            performance_metrics: PerformanceMetrics | None = None
            if self.performance_monitor and target_column and prediction_column:
                try:
                    performance_metrics = self.performance_monitor.monitor_performance(
                        current_data=current_data,
                        target_column=target_column,
                        prediction_column=prediction_column,
                        latency_ms=latency_ms,
                        memory_mb=memory_mb,
                    )
                    results["performance_metrics"] = performance_metrics.to_dict()

                    # Create alert if performance degraded
                    if performance_metrics.degraded and self.alert_manager:
                        alert = self.alert_manager.create_performance_alert(
                            degradation_reasons=performance_metrics.degradation_reasons,
                            metrics=performance_metrics.to_dict(),
                        )
                        results.setdefault("alerts", []).append(alert.to_dict())

                except Exception as e:
                    logger.error("Performance monitoring failed: %s", e)
                    results["performance_error"] = str(e)

            # Apple Silicon metrics
            apple_silicon_metrics: AppleSiliconMetrics | None = None
            if self.apple_silicon_metrics:
                try:
                    apple_silicon_metrics = self.apple_silicon_metrics.collect()
                    results["apple_silicon_metrics"] = apple_silicon_metrics.to_dict()

                    # Check for Apple Silicon issues
                    if self.alert_manager:
                        self._check_apple_silicon_issues(apple_silicon_metrics, results)

                except Exception as e:
                    logger.error("Apple Silicon metrics collection failed: %s", e)
                    results["apple_silicon_error"] = str(e)

            # Check if retraining needed
            results["retraining_suggested"] = self._should_suggest_retraining(
                drift_report, performance_metrics
            )

            logger.info(
                "Monitoring complete: drift=%s, performance_degraded=%s, retraining_suggested=%s",
                drift_report.dataset_drift if drift_report else None,
                performance_metrics.degraded if performance_metrics else None,
                results["retraining_suggested"],
            )

            return results

        except Exception as e:
            logger.error("Monitoring failed: %s", e)
            raise RuntimeError(f"Monitoring failed: {e}") from e

    def _check_apple_silicon_issues(
        self,
        metrics: AppleSiliconMetrics,
        results: dict[str, Any],
    ) -> None:
        """Check for Apple Silicon-specific issues

        Args:
            metrics: Apple Silicon metrics
            results: Results dictionary to update with alerts
        """
        if not self.alert_manager:
            return

        # Check memory pressure
        if metrics.memory_percent > 90:
            alert = self.alert_manager.create_apple_silicon_alert(
                issue=f"High memory pressure: {metrics.memory_percent:.1f}% used",
                metrics=metrics.to_dict(),
            )
            results.setdefault("alerts", []).append(alert.to_dict())

        # Check thermal state
        if metrics.thermal_state in ["serious", "critical"]:
            alert = self.alert_manager.create_apple_silicon_alert(
                issue=f"Thermal state: {metrics.thermal_state}",
                metrics=metrics.to_dict(),
            )
            results.setdefault("alerts", []).append(alert.to_dict())

    def _should_suggest_retraining(
        self,
        drift_report: DriftReport | None,
        performance_metrics: PerformanceMetrics | None,
    ) -> bool:
        """Determine if retraining should be suggested

        Args:
            drift_report: Drift detection results
            performance_metrics: Performance metrics

        Returns:
            True if retraining is suggested, False otherwise
        """
        # Suggest retraining if significant drift detected
        if drift_report and drift_report.dataset_drift and drift_report.drift_share > 0.5:
            logger.info("Retraining suggested due to significant drift")
            return True

        # Suggest retraining if performance degraded
        if performance_metrics and performance_metrics.degraded:
            logger.info("Retraining suggested due to performance degradation")
            return True

        return False

    def get_monitoring_status(self) -> dict[str, Any]:
        """Get current monitoring system status

        Returns:
            Dictionary with monitoring status information
        """
        return {
            "project_name": self.project_name,
            "monitoring_enabled": self.config.monitoring_enabled,
            "drift_detection_enabled": self.config.drift_detection_enabled,
            "performance_monitoring_enabled": self.config.performance_monitoring_enabled,
            "apple_silicon_metrics_enabled": self.config.apple_silicon_metrics_enabled,
            "alert_enabled": self.config.alert_enabled,
            "workspace_path": str(self.config.workspace_path),
            "drift_detector_ready": self.drift_detector is not None
            and self.drift_detector.get_reference_data() is not None,
            "performance_monitor_ready": self.performance_monitor is not None
            and self.performance_monitor.get_reference_data() is not None,
            "apple_silicon_available": self.apple_silicon_metrics is not None
            and self.apple_silicon_metrics.is_apple_silicon(),
        }


def create_monitor(
    project_name: str,
    workspace_path: Path | str | None = None,
    enable_drift_detection: bool = True,
    enable_performance_monitoring: bool = True,
    enable_apple_silicon_metrics: bool = True,
    enable_alerts: bool = True,
) -> EvidentlyMonitor:
    """Create a pre-configured Evidently monitor

    Args:
        project_name: Name of the project being monitored
        workspace_path: Path to workspace for saving reports
        enable_drift_detection: Enable drift detection
        enable_performance_monitoring: Enable performance monitoring
        enable_apple_silicon_metrics: Enable Apple Silicon metrics
        enable_alerts: Enable alert management

    Returns:
        Configured EvidentlyMonitor instance

    Example:
        >>> monitor = create_monitor("lora-finetuning-mlx")
        >>> monitor.set_reference_data(train_data, "target", "prediction")
        >>> results = monitor.monitor(test_data, "target", "prediction")
    """
    config = EvidentlyConfig(
        project_name=project_name,
        workspace_path=workspace_path or "mlops/monitoring/workspace",
        drift_detection_enabled=enable_drift_detection,
        performance_monitoring_enabled=enable_performance_monitoring,
        apple_silicon_metrics_enabled=enable_apple_silicon_metrics,
        alert_enabled=enable_alerts,
    )

    return EvidentlyMonitor(project_name=project_name, config=config)
