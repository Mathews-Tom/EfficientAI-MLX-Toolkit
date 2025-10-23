"""Evidently Monitoring Dashboard

Evidently-based monitoring infrastructure for data drift detection,
model performance monitoring, and Apple Silicon-specific metrics.
"""

from __future__ import annotations

from mlops.monitoring.evidently.alert_manager import AlertManager, AlertConfig, Alert
from mlops.monitoring.evidently.apple_silicon_metrics import (
    AppleSiliconMetricsCollector,
    AppleSiliconMetrics,
)
from mlops.monitoring.evidently.drift_detector import DriftDetector, DriftReport
from mlops.monitoring.evidently.monitor import EvidentlyMonitor, create_monitor
from mlops.monitoring.evidently.performance_monitor import (
    PerformanceMonitor,
    PerformanceMetrics,
)

__all__ = [
    "Alert",
    "AlertConfig",
    "AlertManager",
    "AppleSiliconMetrics",
    "AppleSiliconMetricsCollector",
    "DriftDetector",
    "DriftReport",
    "EvidentlyMonitor",
    "PerformanceMetrics",
    "PerformanceMonitor",
    "create_monitor",
]
