"""MLOps Monitoring Infrastructure

Provides monitoring and alerting infrastructure for model performance,
data drift detection, and Apple Silicon-specific metrics tracking.
"""

from __future__ import annotations

__all__ = [
    "EvidentlyMonitor",
    "DriftDetector",
    "PerformanceMonitor",
    "AlertManager",
    "AppleSiliconMetricsCollector",
    "create_monitor",
]


def __getattr__(name: str) -> object:
    """Lazy imports for monitoring components"""
    if name == "EvidentlyMonitor":
        from mlops.monitoring.evidently.monitor import EvidentlyMonitor

        return EvidentlyMonitor
    elif name == "DriftDetector":
        from mlops.monitoring.evidently.drift_detector import DriftDetector

        return DriftDetector
    elif name == "PerformanceMonitor":
        from mlops.monitoring.evidently.performance_monitor import PerformanceMonitor

        return PerformanceMonitor
    elif name == "AlertManager":
        from mlops.monitoring.evidently.alert_manager import AlertManager

        return AlertManager
    elif name == "AppleSiliconMetricsCollector":
        from mlops.monitoring.evidently.apple_silicon_metrics import (
            AppleSiliconMetricsCollector,
        )

        return AppleSiliconMetricsCollector
    elif name == "create_monitor":
        from mlops.monitoring.evidently.monitor import create_monitor

        return create_monitor

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
