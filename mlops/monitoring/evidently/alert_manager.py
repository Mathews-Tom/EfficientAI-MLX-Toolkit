"""Alert Management System

Manages alerts for drift detection, performance degradation, and system issues
with configurable notification channels and alert rules.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels"""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertType(Enum):
    """Types of alerts"""

    DRIFT_DETECTED = "drift_detected"
    PERFORMANCE_DEGRADED = "performance_degraded"
    SYSTEM_ERROR = "system_error"
    THRESHOLD_EXCEEDED = "threshold_exceeded"
    APPLE_SILICON_ISSUE = "apple_silicon_issue"


@dataclass
class Alert:
    """Alert information

    Attributes:
        alert_id: Unique alert identifier
        timestamp: When alert was created
        alert_type: Type of alert
        severity: Alert severity level
        project_name: Name of the project
        title: Alert title
        message: Detailed alert message
        metadata: Additional alert metadata
        acknowledged: Whether alert has been acknowledged
        resolved: Whether alert has been resolved
    """

    alert_id: str
    timestamp: datetime
    alert_type: AlertType
    severity: AlertSeverity
    project_name: str
    title: str
    message: str
    metadata: dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False
    resolved: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert alert to dictionary

        Returns:
            Dictionary representation of alert
        """
        return {
            "alert_id": self.alert_id,
            "timestamp": self.timestamp.isoformat(),
            "alert_type": self.alert_type.value,
            "severity": self.severity.value,
            "project_name": self.project_name,
            "title": self.title,
            "message": self.message,
            "metadata": self.metadata,
            "acknowledged": self.acknowledged,
            "resolved": self.resolved,
        }


@dataclass
class AlertConfig:
    """Alert configuration

    Attributes:
        enabled: Whether alerting is enabled
        drift_threshold: Drift share threshold to trigger alert
        performance_degradation_enabled: Whether to alert on performance degradation
        apple_silicon_monitoring_enabled: Whether to monitor Apple Silicon metrics
        notification_channels: List of notification channels
        alert_retention_days: Number of days to retain alerts
    """

    enabled: bool = True
    drift_threshold: float = 0.5
    performance_degradation_enabled: bool = True
    apple_silicon_monitoring_enabled: bool = True
    notification_channels: list[str] = field(default_factory=lambda: ["log"])
    alert_retention_days: int = 30

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> AlertConfig:
        """Create configuration from dictionary"""
        return cls(**config)

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "enabled": self.enabled,
            "drift_threshold": self.drift_threshold,
            "performance_degradation_enabled": self.performance_degradation_enabled,
            "apple_silicon_monitoring_enabled": self.apple_silicon_monitoring_enabled,
            "notification_channels": self.notification_channels,
            "alert_retention_days": self.alert_retention_days,
        }


class AlertManager:
    """Alert management system

    Manages alerts for monitoring events with configurable notification
    channels and alert rules. Supports:
    - Alert creation and tracking
    - Alert acknowledgment and resolution
    - Notification channel integration
    - Alert history and retention
    """

    def __init__(
        self,
        project_name: str,
        config: AlertConfig | None = None,
        workspace_path: Path | str | None = None,
    ):
        """Initialize alert manager

        Args:
            project_name: Name of the project being monitored
            config: Alert configuration
            workspace_path: Path to workspace for storing alerts
        """
        self.project_name = project_name
        self.config = config or AlertConfig()
        self.workspace_path = (
            Path(workspace_path) if workspace_path else Path("mlops/monitoring/workspace")
        )
        self.workspace_path.mkdir(parents=True, exist_ok=True)

        # Alert storage
        self._alerts: dict[str, Alert] = {}
        self._alert_counter = 0

        # Notification handlers
        self._notification_handlers: dict[str, Callable[[Alert], None]] = {
            "log": self._log_notification,
        }

        logger.info(
            "Initialized AlertManager for project: %s (workspace: %s)",
            project_name,
            self.workspace_path,
        )

    def create_alert(
        self,
        alert_type: AlertType,
        severity: AlertSeverity,
        title: str,
        message: str,
        metadata: dict[str, Any] | None = None,
    ) -> Alert:
        """Create and store a new alert

        Args:
            alert_type: Type of alert
            severity: Alert severity level
            title: Alert title
            message: Detailed alert message
            metadata: Additional alert metadata

        Returns:
            Created Alert instance
        """
        if not self.config.enabled:
            logger.debug("Alerting disabled, skipping alert creation")
            return Alert(
                alert_id="disabled",
                timestamp=datetime.now(),
                alert_type=alert_type,
                severity=severity,
                project_name=self.project_name,
                title=title,
                message=message,
                metadata=metadata or {},
            )

        try:
            # Generate alert ID
            self._alert_counter += 1
            alert_id = f"{self.project_name}_{alert_type.value}_{self._alert_counter}"

            # Create alert
            alert = Alert(
                alert_id=alert_id,
                timestamp=datetime.now(),
                alert_type=alert_type,
                severity=severity,
                project_name=self.project_name,
                title=title,
                message=message,
                metadata=metadata or {},
            )

            # Store alert
            self._alerts[alert_id] = alert

            # Send notifications
            self._send_notifications(alert)

            logger.info(
                "Created alert: %s (type: %s, severity: %s)",
                alert_id,
                alert_type.value,
                severity.value,
            )

            return alert

        except Exception as e:
            logger.error("Failed to create alert: %s", e)
            raise RuntimeError(f"Alert creation failed: {e}") from e

    def create_drift_alert(
        self,
        drift_share: float,
        drifted_features: list[str],
        total_features: int,
    ) -> Alert:
        """Create alert for detected data drift

        Args:
            drift_share: Share of features with detected drift
            drifted_features: List of drifted features
            total_features: Total number of features

        Returns:
            Created Alert instance
        """
        severity = (
            AlertSeverity.CRITICAL
            if drift_share > 0.7
            else AlertSeverity.WARNING if drift_share > 0.5 else AlertSeverity.INFO
        )

        return self.create_alert(
            alert_type=AlertType.DRIFT_DETECTED,
            severity=severity,
            title=f"Data Drift Detected: {drift_share:.1%} of features",
            message=f"Detected drift in {len(drifted_features)}/{total_features} features. "
            f"Affected features: {', '.join(drifted_features[:5])}{'...' if len(drifted_features) > 5 else ''}",
            metadata={
                "drift_share": drift_share,
                "drifted_features": drifted_features,
                "total_features": total_features,
            },
        )

    def create_performance_alert(
        self,
        degradation_reasons: list[str],
        metrics: dict[str, Any],
    ) -> Alert:
        """Create alert for performance degradation

        Args:
            degradation_reasons: List of degradation reasons
            metrics: Performance metrics

        Returns:
            Created Alert instance
        """
        return self.create_alert(
            alert_type=AlertType.PERFORMANCE_DEGRADED,
            severity=AlertSeverity.WARNING,
            title="Model Performance Degraded",
            message=f"Performance degradation detected: {'; '.join(degradation_reasons)}",
            metadata={
                "degradation_reasons": degradation_reasons,
                "metrics": metrics,
            },
        )

    def create_apple_silicon_alert(
        self,
        issue: str,
        metrics: dict[str, Any],
    ) -> Alert:
        """Create alert for Apple Silicon issues

        Args:
            issue: Description of the issue
            metrics: Apple Silicon metrics

        Returns:
            Created Alert instance
        """
        return self.create_alert(
            alert_type=AlertType.APPLE_SILICON_ISSUE,
            severity=AlertSeverity.WARNING,
            title="Apple Silicon Issue Detected",
            message=issue,
            metadata={"metrics": metrics},
        )

    def acknowledge_alert(self, alert_id: str) -> None:
        """Acknowledge an alert

        Args:
            alert_id: Alert identifier

        Raises:
            ValueError: If alert not found
        """
        if alert_id not in self._alerts:
            raise ValueError(f"Alert not found: {alert_id}")

        self._alerts[alert_id].acknowledged = True
        logger.info("Acknowledged alert: %s", alert_id)

    def resolve_alert(self, alert_id: str) -> None:
        """Resolve an alert

        Args:
            alert_id: Alert identifier

        Raises:
            ValueError: If alert not found
        """
        if alert_id not in self._alerts:
            raise ValueError(f"Alert not found: {alert_id}")

        self._alerts[alert_id].resolved = True
        logger.info("Resolved alert: %s", alert_id)

    def get_alert(self, alert_id: str) -> Alert:
        """Get an alert by ID

        Args:
            alert_id: Alert identifier

        Returns:
            Alert instance

        Raises:
            ValueError: If alert not found
        """
        if alert_id not in self._alerts:
            raise ValueError(f"Alert not found: {alert_id}")

        return self._alerts[alert_id]

    def get_all_alerts(
        self,
        unresolved_only: bool = False,
        alert_type: AlertType | None = None,
    ) -> list[Alert]:
        """Get all alerts

        Args:
            unresolved_only: Only return unresolved alerts
            alert_type: Filter by alert type

        Returns:
            List of alerts
        """
        alerts = list(self._alerts.values())

        if unresolved_only:
            alerts = [a for a in alerts if not a.resolved]

        if alert_type is not None:
            alerts = [a for a in alerts if a.alert_type == alert_type]

        return alerts

    def _send_notifications(self, alert: Alert) -> None:
        """Send notifications for an alert

        Args:
            alert: Alert to notify about
        """
        for channel in self.config.notification_channels:
            handler = self._notification_handlers.get(channel)
            if handler:
                try:
                    handler(alert)
                except Exception as e:
                    logger.error("Notification failed for channel %s: %s", channel, e)
            else:
                logger.warning("Unknown notification channel: %s", channel)

    def _log_notification(self, alert: Alert) -> None:
        """Log notification handler

        Args:
            alert: Alert to log
        """
        log_level = {
            AlertSeverity.INFO: logging.INFO,
            AlertSeverity.WARNING: logging.WARNING,
            AlertSeverity.ERROR: logging.ERROR,
            AlertSeverity.CRITICAL: logging.CRITICAL,
        }.get(alert.severity, logging.INFO)

        logger.log(
            log_level,
            "ALERT [%s] %s: %s - %s",
            alert.severity.value.upper(),
            alert.alert_type.value,
            alert.title,
            alert.message,
        )

    def register_notification_handler(
        self,
        channel: str,
        handler: Callable[[Alert], None],
    ) -> None:
        """Register a custom notification handler

        Args:
            channel: Channel name
            handler: Notification handler function

        Example:
            >>> def email_handler(alert: Alert):
            ...     send_email(alert.title, alert.message)
            >>> manager.register_notification_handler("email", email_handler)
        """
        self._notification_handlers[channel] = handler
        logger.info("Registered notification handler for channel: %s", channel)
