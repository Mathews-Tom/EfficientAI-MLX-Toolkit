"""
Monitoring and observability for DSPy Integration Framework.
"""

import asyncio
import json
import logging
import os
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

from ..exceptions import DSPyIntegrationError

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for DSPy operations."""

    execution_time: float
    input_tokens: int
    output_tokens: int
    memory_usage: float
    timestamp: float
    success: bool = True
    error_message: str | None = None


@dataclass
class SystemMetrics:
    """System-level metrics."""

    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: float
    timestamp: float


@dataclass
class AlertRule:
    """Alert rule configuration."""

    name: str
    metric_name: str
    threshold: float
    comparison: str  # "gt", "lt", "eq"
    window_minutes: int
    enabled: bool = True


class MetricsCollector:
    """Collects and aggregates metrics."""

    def __init__(self, max_history: int = 10000):
        """Initialize metrics collector."""
        self.max_history = max_history
        self.performance_metrics: deque = deque(maxlen=max_history)
        self.system_metrics: deque = deque(maxlen=max_history)
        self.request_counts = defaultdict(int)
        self.error_counts = defaultdict(int)
        self.response_times = defaultdict(list)

    def record_performance(self, metrics: PerformanceMetrics):
        """Record performance metrics."""
        self.performance_metrics.append(metrics)

        # Update aggregated metrics
        if metrics.success:
            self.request_counts["success"] += 1
        else:
            self.request_counts["error"] += 1
            self.error_counts[metrics.error_message or "unknown"] += 1

        self.response_times["all"].append(metrics.execution_time)

    def record_system(self, metrics: SystemMetrics):
        """Record system metrics."""
        self.system_metrics.append(metrics)

    def get_performance_summary(
        self, window_minutes: int = 60
    ) -> dict[str, str | int | float | bool]:
        """Get performance summary for the specified time window."""
        cutoff_time = time.time() - (window_minutes * 60)

        # Filter metrics within window
        recent_metrics = [m for m in self.performance_metrics if m.timestamp >= cutoff_time]

        if not recent_metrics:
            return {
                "total_requests": 0,
                "success_rate": 0.0,
                "avg_response_time": 0.0,
                "error_rate": 0.0,
            }

        total_requests = len(recent_metrics)
        successful_requests = sum(1 for m in recent_metrics if m.success)

        avg_response_time = sum(m.execution_time for m in recent_metrics) / total_requests
        success_rate = successful_requests / total_requests
        error_rate = 1.0 - success_rate

        return {
            "total_requests": total_requests,
            "success_rate": success_rate,
            "avg_response_time": avg_response_time,
            "error_rate": error_rate,
            "window_minutes": window_minutes,
            "timestamp": time.time(),
        }

    def get_system_summary(self, window_minutes: int = 60) -> dict[str, str | int | float | bool]:
        """Get system metrics summary."""
        cutoff_time = time.time() - (window_minutes * 60)

        recent_metrics = [m for m in self.system_metrics if m.timestamp >= cutoff_time]

        if not recent_metrics:
            return {}

        avg_cpu = sum(m.cpu_usage for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m.memory_usage for m in recent_metrics) / len(recent_metrics)
        avg_disk = sum(m.disk_usage for m in recent_metrics) / len(recent_metrics)

        return {
            "avg_cpu_usage": avg_cpu,
            "avg_memory_usage": avg_memory,
            "avg_disk_usage": avg_disk,
            "window_minutes": window_minutes,
            "timestamp": time.time(),
        }

    def get_error_analysis(self, window_minutes: int = 60) -> dict[str, str | int | float | bool]:
        """Get error analysis."""
        cutoff_time = time.time() - (window_minutes * 60)

        recent_errors = [
            m for m in self.performance_metrics if not m.success and m.timestamp >= cutoff_time
        ]

        error_breakdown = defaultdict(int)
        for error in recent_errors:
            error_breakdown[error.error_message or "unknown"] += 1

        return {
            "total_errors": len(recent_errors),
            "error_breakdown": dict(error_breakdown),
            "window_minutes": window_minutes,
            "timestamp": time.time(),
        }


class AlertManager:
    """Manages alerts and notifications."""

    def __init__(self):
        """Initialize alert manager."""
        self.alert_rules: list[AlertRule] = []
        self.active_alerts: dict[str, dict[str, str | int | float | bool]] = {}
        self.alert_history: list[dict[str, str | int | float | bool]] = []

    def add_alert_rule(self, rule: AlertRule):
        """Add an alert rule."""
        self.alert_rules.append(rule)
        logger.info("Added alert rule: %s", rule.name)

    def remove_alert_rule(self, rule_name: str):
        """Remove an alert rule."""
        self.alert_rules = [r for r in self.alert_rules if r.name != rule_name]
        logger.info("Removed alert rule: %s", rule_name)

    def check_alerts(self, metrics: dict[str, str | int | float | bool]):
        """Check metrics against alert rules."""
        current_time = time.time()

        for rule in self.alert_rules:
            if not rule.enabled:
                continue

            metric_value = metrics.get(rule.metric_name)
            if metric_value is None:
                continue

            # Check threshold
            triggered = False
            if rule.comparison == "gt" and metric_value > rule.threshold:
                triggered = True
            elif rule.comparison == "lt" and metric_value < rule.threshold:
                triggered = True
            elif rule.comparison == "eq" and metric_value == rule.threshold:
                triggered = True

            if triggered:
                self._trigger_alert(rule, metric_value, current_time)
            else:
                self._resolve_alert(rule.name, current_time)

    def _trigger_alert(self, rule: AlertRule, value: float, timestamp: float):
        """Trigger an alert."""
        alert_key = rule.name

        if alert_key not in self.active_alerts:
            alert = {
                "rule_name": rule.name,
                "metric_name": rule.metric_name,
                "threshold": rule.threshold,
                "current_value": value,
                "triggered_at": timestamp,
                "status": "active",
            }

            self.active_alerts[alert_key] = alert
            self.alert_history.append(alert.copy())

            logger.warning(
                "Alert triggered: %s - %s = %s (threshold: %s)",
                rule.name,
                rule.metric_name,
                value,
                rule.threshold,
            )

            # Here you would integrate with notification systems (email, Slack, etc.)
            self._send_notification(alert)

    def _resolve_alert(self, rule_name: str, timestamp: float):
        """Resolve an alert."""
        if rule_name in self.active_alerts:
            alert = self.active_alerts[rule_name]
            alert["status"] = "resolved"
            alert["resolved_at"] = timestamp

            del self.active_alerts[rule_name]
            self.alert_history.append(alert.copy())

            logger.info("Alert resolved: %s", rule_name)

    def _send_notification(self, alert: dict[str, str | int | float | bool]):
        """Send alert notification."""
        # Placeholder for notification integration
        # In a real implementation, this would send emails, Slack messages, etc.
        logger.warning(
            "ALERT: %s - %s = %s", alert["rule_name"], alert["metric_name"], alert["current_value"]
        )

    def get_active_alerts(self) -> list[dict[str, str | int | float | bool]]:
        """Get currently active alerts."""
        return list(self.active_alerts.values())

    def get_alert_history(self, limit: int = 100) -> list[dict[str, str | int | float | bool]]:
        """Get alert history."""
        return self.alert_history[-limit:]


class DSPyMonitor:
    """Main monitoring class for DSPy Integration Framework."""

    def __init__(self, export_path: Path | None = None, enable_system_monitoring: bool = True):
        """Initialize DSPy monitor."""
        self.metrics_collector = MetricsCollector()
        self.alert_manager = AlertManager()
        self.export_path = export_path or Path(".dspy_cache/monitoring")
        self.export_path.mkdir(parents=True, exist_ok=True)

        self.enable_system_monitoring = enable_system_monitoring
        self.monitoring_task = None

        # Setup default alert rules
        self._setup_default_alerts()

        # Start system monitoring if enabled
        if enable_system_monitoring:
            self.start_system_monitoring()

    def _setup_default_alerts(self):
        """Setup default alert rules."""
        default_rules = [
            AlertRule(
                name="high_error_rate",
                metric_name="error_rate",
                threshold=0.1,  # 10% error rate
                comparison="gt",
                window_minutes=5,
            ),
            AlertRule(
                name="slow_response_time",
                metric_name="avg_response_time",
                threshold=5.0,  # 5 seconds
                comparison="gt",
                window_minutes=5,
            ),
            AlertRule(
                name="high_memory_usage",
                metric_name="avg_memory_usage",
                threshold=0.9,  # 90% memory usage
                comparison="gt",
                window_minutes=5,
            ),
        ]

        for rule in default_rules:
            self.alert_manager.add_alert_rule(rule)

    def start_system_monitoring(self):
        """Start system monitoring task."""
        if self.monitoring_task is None:
            self.monitoring_task = asyncio.create_task(self._system_monitoring_loop())
            logger.info("Started system monitoring")

    def stop_system_monitoring(self):
        """Stop system monitoring task."""
        if self.monitoring_task:
            self.monitoring_task.cancel()
            self.monitoring_task = None
            logger.info("Stopped system monitoring")

    async def _system_monitoring_loop(self):
        """System monitoring loop."""
        try:
            while True:
                try:
                    # Collect system metrics
                    system_metrics = await self._collect_system_metrics()
                    self.metrics_collector.record_system(system_metrics)

                    # Check alerts
                    performance_summary = self.metrics_collector.get_performance_summary(5)
                    system_summary = self.metrics_collector.get_system_summary(5)

                    combined_metrics = {**performance_summary, **system_summary}
                    self.alert_manager.check_alerts(combined_metrics)

                    await asyncio.sleep(30)  # Check every 30 seconds

                except Exception as e:
                    logger.error("System monitoring error: %s", e)
                    await asyncio.sleep(60)  # Wait longer on error

        except asyncio.CancelledError:
            logger.info("System monitoring cancelled")

    async def _collect_system_metrics(self) -> SystemMetrics:
        """Collect system metrics."""
        try:
            # This is a simplified implementation
            # In production, you'd use psutil or similar libraries

            # CPU usage (simplified)
            cpu_usage = 0.0
            try:
                with open("/proc/loadavg", "r") as f:
                    load_avg = float(f.read().split()[0])
                    cpu_usage = min(load_avg / os.cpu_count(), 1.0)
            except Exception as _:
                cpu_usage = 0.0

            # Memory usage (simplified)
            memory_usage = 0.0
            try:
                with open("/proc/meminfo", "r") as f:
                    meminfo = f.read()
                    total = int(
                        [line for line in meminfo.split("\n") if "MemTotal" in line][0].split()[1]
                    )
                    available = int(
                        [line for line in meminfo.split("\n") if "MemAvailable" in line][0].split()[
                            1
                        ]
                    )
                    memory_usage = 1.0 - (available / total)
            except Exception as _:
                memory_usage = 0.0

            return SystemMetrics(
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                disk_usage=0.0,  # Would implement disk usage check
                network_io=0.0,  # Would implement network I/O check
                timestamp=time.time(),
            )

        except Exception as e:
            logger.error("Failed to collect system metrics: %s", e)
            return SystemMetrics(0.0, 0.0, 0.0, 0.0, time.time())

    def record_request(self, project_name: str, module_name: str, metrics: PerformanceMetrics):
        """Record a request with performance metrics."""
        self.metrics_collector.record_performance(metrics)
        logger.debug(
            "Recorded request: %s/%s - %.3fs", project_name, module_name, metrics.execution_time
        )

    async def get_metrics(self, window_minutes: int = 60) -> dict[str, str | int | float | bool]:
        """Get comprehensive metrics."""
        performance_summary = self.metrics_collector.get_performance_summary(window_minutes)
        system_summary = self.metrics_collector.get_system_summary(window_minutes)
        error_analysis = self.metrics_collector.get_error_analysis(window_minutes)
        active_alerts = self.alert_manager.get_active_alerts()

        return {
            "performance": performance_summary,
            "system": system_summary,
            "errors": error_analysis,
            "alerts": {
                "active": active_alerts,
                "count": len(active_alerts),
            },
            "timestamp": time.time(),
        }

    async def export_metrics(self, filename: str | None = None):
        """Export metrics to file."""
        try:
            if filename is None:
                filename = f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

            export_file = self.export_path / filename

            metrics_data = {
                "performance_metrics": [
                    asdict(m) for m in self.metrics_collector.performance_metrics
                ],
                "system_metrics": [asdict(m) for m in self.metrics_collector.system_metrics],
                "alert_history": self.alert_manager.get_alert_history(),
                "export_timestamp": time.time(),
            }

            with open(export_file, "w") as f:
                json.dump(metrics_data, f, indent=2)

            logger.info("Metrics exported to %s", export_file)

        except Exception as e:
            logger.error("Failed to export metrics: %s", e)

    def add_alert_rule(self, rule: AlertRule):
        """Add a custom alert rule."""
        self.alert_manager.add_alert_rule(rule)

    def get_dashboard_data(self) -> dict[str, str | int | float | bool]:
        """Get data for monitoring dashboard."""
        try:
            current_metrics = asyncio.run(self.get_metrics(60))

            # Add trend data
            recent_performance = list(self.metrics_collector.performance_metrics)[-100:]
            trend_data = []

            for i in range(0, len(recent_performance), 10):
                chunk = recent_performance[i : i + 10]
                if chunk:
                    avg_time = sum(m.execution_time for m in chunk) / len(chunk)
                    success_rate = sum(1 for m in chunk if m.success) / len(chunk)
                    trend_data.append(
                        {
                            "timestamp": chunk[-1].timestamp,
                            "avg_response_time": avg_time,
                            "success_rate": success_rate,
                        }
                    )

            return {
                **current_metrics,
                "trends": trend_data,
                "health_status": (
                    "healthy" if len(self.alert_manager.get_active_alerts()) == 0 else "degraded"
                ),
            }

        except Exception as e:
            logger.error("Failed to get dashboard data: %s", e)
            return {"error": str(e)}

    def cleanup(self):
        """Cleanup monitoring resources."""
        self.stop_system_monitoring()
        logger.info("DSPy monitor cleanup completed")
