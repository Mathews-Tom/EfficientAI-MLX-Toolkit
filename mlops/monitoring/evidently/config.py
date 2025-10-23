"""Evidently Configuration

Configuration dataclasses for Evidently monitoring infrastructure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class EvidentlyConfig:
    """Configuration for Evidently monitoring system

    Attributes:
        project_name: Name of the project being monitored
        workspace_path: Path to Evidently workspace directory
        monitoring_enabled: Whether monitoring is enabled
        drift_detection_enabled: Whether drift detection is enabled
        performance_monitoring_enabled: Whether performance monitoring is enabled
        apple_silicon_metrics_enabled: Whether Apple Silicon metrics are collected
        alert_enabled: Whether alerting is enabled
        dashboard_port: Port for Evidently dashboard (default: 8000)
        retention_days: Number of days to retain monitoring data
    """

    project_name: str
    workspace_path: Path | str = field(default="mlops/monitoring/workspace")
    monitoring_enabled: bool = True
    drift_detection_enabled: bool = True
    performance_monitoring_enabled: bool = True
    apple_silicon_metrics_enabled: bool = True
    alert_enabled: bool = True
    dashboard_port: int = 8000
    retention_days: int = 30

    def __post_init__(self) -> None:
        """Convert workspace_path to Path object"""
        if isinstance(self.workspace_path, str):
            self.workspace_path = Path(self.workspace_path)

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> EvidentlyConfig:
        """Create configuration from dictionary

        Args:
            config: Configuration dictionary

        Returns:
            EvidentlyConfig instance
        """
        return cls(**config)

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary

        Returns:
            Configuration dictionary
        """
        return {
            "project_name": self.project_name,
            "workspace_path": str(self.workspace_path),
            "monitoring_enabled": self.monitoring_enabled,
            "drift_detection_enabled": self.drift_detection_enabled,
            "performance_monitoring_enabled": self.performance_monitoring_enabled,
            "apple_silicon_metrics_enabled": self.apple_silicon_metrics_enabled,
            "alert_enabled": self.alert_enabled,
            "dashboard_port": self.dashboard_port,
            "retention_days": self.retention_days,
        }


@dataclass
class DriftDetectionConfig:
    """Configuration for drift detection

    Attributes:
        stattest: Statistical test to use for drift detection
        stattest_threshold: Threshold for statistical test
        drift_share: Share of drifted features to trigger alert
        nbinsx: Number of bins for histograms (x-axis)
        window_size: Size of reference window for drift detection
    """

    stattest: str = "wasserstein"
    stattest_threshold: float = 0.1
    drift_share: float = 0.5
    nbinsx: int = 10
    window_size: int = 1000

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> DriftDetectionConfig:
        """Create configuration from dictionary"""
        return cls(**config)

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "stattest": self.stattest,
            "stattest_threshold": self.stattest_threshold,
            "drift_share": self.drift_share,
            "nbinsx": self.nbinsx,
            "window_size": self.window_size,
        }


@dataclass
class PerformanceThresholds:
    """Performance monitoring thresholds

    Attributes:
        accuracy_threshold: Minimum acceptable accuracy
        precision_threshold: Minimum acceptable precision
        recall_threshold: Minimum acceptable recall
        f1_threshold: Minimum acceptable F1 score
        latency_threshold_ms: Maximum acceptable latency in milliseconds
        memory_threshold_mb: Maximum acceptable memory usage in MB
    """

    accuracy_threshold: float = 0.8
    precision_threshold: float = 0.75
    recall_threshold: float = 0.75
    f1_threshold: float = 0.75
    latency_threshold_ms: float = 100.0
    memory_threshold_mb: float = 2048.0

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> PerformanceThresholds:
        """Create configuration from dictionary"""
        return cls(**config)

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "accuracy_threshold": self.accuracy_threshold,
            "precision_threshold": self.precision_threshold,
            "recall_threshold": self.recall_threshold,
            "f1_threshold": self.f1_threshold,
            "latency_threshold_ms": self.latency_threshold_ms,
            "memory_threshold_mb": self.memory_threshold_mb,
        }
