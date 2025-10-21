"""Scaling Manager for Ray Serve

This module provides auto-scaling and load balancing logic for Ray Serve deployments
with Apple Silicon thermal awareness and resource management.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ScalingMetrics:
    """Metrics used for scaling decisions"""
    current_replicas: int
    target_replicas: int
    ongoing_requests: int
    avg_requests_per_replica: float
    cpu_utilization_pct: float
    memory_utilization_pct: float
    thermal_state: str = "nominal"  # nominal, fair, serious, critical


@dataclass
class ScalingConfig:
    """Configuration for auto-scaling behavior"""
    min_replicas: int = 1
    max_replicas: int = 10
    target_requests_per_replica: int = 10
    scale_up_threshold: float = 0.8  # Scale up if above 80% of target
    scale_down_threshold: float = 0.3  # Scale down if below 30% of target
    cooldown_period_s: int = 60  # Wait period between scaling actions
    thermal_aware: bool = True
    thermal_scale_down_threshold: str = "serious"  # Scale down at this thermal state


class ScalingManager:
    """Manages auto-scaling and load balancing for Ray Serve deployments

    This manager monitors deployment metrics and adjusts replica counts based on
    load, resource utilization, and Apple Silicon thermal conditions.
    """

    def __init__(
        self,
        cluster: Any,  # SharedRayCluster
        scaling_config: ScalingConfig | None = None,
    ):
        """Initialize scaling manager

        Args:
            cluster: SharedRayCluster instance to manage
            scaling_config: Scaling configuration (uses defaults if None)
        """
        self.cluster = cluster
        self.config = scaling_config or ScalingConfig()
        self._last_scale_time: dict[str, float] = {}  # deployment_name -> timestamp
        self._scaling_history: list[dict[str, Any]] = []
        self._thermal_monitor = ThermalMonitor() if self.config.thermal_aware else None

    def evaluate_scaling(
        self,
        project_name: str,
        model_name: str,
        current_metrics: dict[str, Any],
    ) -> ScalingMetrics:
        """Evaluate whether scaling is needed for a deployment

        Args:
            project_name: Project identifier
            model_name: Model identifier
            current_metrics: Current deployment metrics

        Returns:
            ScalingMetrics with scaling decision
        """
        current_replicas = current_metrics.get("num_replicas", 1)
        ongoing_requests = current_metrics.get("ongoing_requests", 0)

        # Calculate average requests per replica
        avg_requests = (
            ongoing_requests / current_replicas if current_replicas > 0 else 0
        )

        # Get resource utilization
        cpu_util = current_metrics.get("cpu_utilization_pct", 0.0)
        mem_util = current_metrics.get("memory_utilization_pct", 0.0)

        # Get thermal state if available
        thermal_state = "nominal"
        if self._thermal_monitor:
            thermal_state = self._thermal_monitor.get_thermal_state()

        # Calculate target replicas based on load
        target_replicas = self._calculate_target_replicas(
            current_replicas=current_replicas,
            avg_requests=avg_requests,
            cpu_util=cpu_util,
            thermal_state=thermal_state,
        )

        return ScalingMetrics(
            current_replicas=current_replicas,
            target_replicas=target_replicas,
            ongoing_requests=ongoing_requests,
            avg_requests_per_replica=avg_requests,
            cpu_utilization_pct=cpu_util,
            memory_utilization_pct=mem_util,
            thermal_state=thermal_state,
        )

    def _calculate_target_replicas(
        self,
        current_replicas: int,
        avg_requests: float,
        cpu_util: float,
        thermal_state: str,
    ) -> int:
        """Calculate target number of replicas

        Args:
            current_replicas: Current replica count
            avg_requests: Average requests per replica
            cpu_util: CPU utilization percentage
            thermal_state: Current thermal state

        Returns:
            Target replica count
        """
        target = current_replicas

        # Scale based on request load
        if avg_requests > self.config.target_requests_per_replica * self.config.scale_up_threshold:
            # Scale up
            target = current_replicas + 1
        elif avg_requests < self.config.target_requests_per_replica * self.config.scale_down_threshold:
            # Scale down
            target = max(1, current_replicas - 1)

        # Apply thermal constraints
        if self.config.thermal_aware:
            if thermal_state == "serious" and target > current_replicas:
                # Don't scale up if thermal state is serious
                logger.warning("Preventing scale-up due to thermal state: %s", thermal_state)
                target = current_replicas
            elif thermal_state == "critical":
                # Force scale down if critical
                logger.warning("Forcing scale-down due to critical thermal state")
                target = max(1, current_replicas - 1)

        # Apply min/max constraints
        target = max(self.config.min_replicas, min(self.config.max_replicas, target))

        return target

    def apply_scaling(
        self,
        project_name: str,
        model_name: str,
        metrics: ScalingMetrics,
    ) -> bool:
        """Apply scaling decision to deployment

        Args:
            project_name: Project identifier
            model_name: Model identifier
            metrics: Scaling metrics with target replicas

        Returns:
            True if scaling was applied, False if skipped
        """
        deployment_key = f"{project_name}_{model_name}"

        # Check if in cooldown period
        last_scale_time = self._last_scale_time.get(deployment_key, 0)
        time_since_last_scale = time.time() - last_scale_time

        if time_since_last_scale < self.config.cooldown_period_s:
            logger.debug(
                "Skipping scaling for %s (in cooldown: %.1fs remaining)",
                deployment_key,
                self.config.cooldown_period_s - time_since_last_scale,
            )
            return False

        # Check if scaling is needed
        if metrics.current_replicas == metrics.target_replicas:
            logger.debug("No scaling needed for %s", deployment_key)
            return False

        # Apply scaling
        try:
            logger.info(
                "Scaling %s from %d to %d replicas (load: %.1f req/replica, thermal: %s)",
                deployment_key,
                metrics.current_replicas,
                metrics.target_replicas,
                metrics.avg_requests_per_replica,
                metrics.thermal_state,
            )

            self.cluster.scale_deployment(
                project_name=project_name,
                model_name=model_name,
                num_replicas=metrics.target_replicas,
            )

            # Update last scale time
            self._last_scale_time[deployment_key] = time.time()

            # Record scaling event
            self._scaling_history.append({
                "timestamp": time.time(),
                "deployment": deployment_key,
                "from_replicas": metrics.current_replicas,
                "to_replicas": metrics.target_replicas,
                "reason": "load" if metrics.thermal_state == "nominal" else "thermal",
                "metrics": {
                    "avg_requests": metrics.avg_requests_per_replica,
                    "cpu_util": metrics.cpu_utilization_pct,
                    "thermal_state": metrics.thermal_state,
                },
            })

            return True

        except Exception as e:
            logger.error("Failed to apply scaling for %s: %s", deployment_key, e)
            return False

    def get_scaling_history(
        self,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Get recent scaling history

        Args:
            limit: Maximum number of events to return

        Returns:
            List of scaling events (most recent first)
        """
        return self._scaling_history[-limit:][::-1]

    def optimize_cluster_resources(self) -> dict[str, Any]:
        """Optimize resource allocation across all deployments

        Returns:
            Dictionary with optimization results
        """
        # Get cluster resource usage
        usage = self.cluster.get_cluster_resource_usage()

        total_deployments = usage.get("total_deployments", 0)
        cpu_utilization = usage.get("cpu", {}).get("utilization_pct", 0)

        recommendations = []

        # Check if cluster is over/under-utilized
        if cpu_utilization > 80:
            recommendations.append({
                "type": "scale_down",
                "reason": "High CPU utilization (>80%)",
                "action": "Consider scaling down low-traffic deployments",
            })
        elif cpu_utilization < 20 and total_deployments > 0:
            recommendations.append({
                "type": "consolidate",
                "reason": "Low CPU utilization (<20%)",
                "action": "Consider consolidating deployments or reducing replicas",
            })

        # Check thermal state
        if self._thermal_monitor:
            thermal_state = self._thermal_monitor.get_thermal_state()
            if thermal_state in ["serious", "critical"]:
                recommendations.append({
                    "type": "thermal_management",
                    "reason": f"Thermal state: {thermal_state}",
                    "action": "Reduce replica counts to lower thermal load",
                })

        return {
            "timestamp": time.time(),
            "cpu_utilization_pct": cpu_utilization,
            "total_deployments": total_deployments,
            "recommendations": recommendations,
        }


class ThermalMonitor:
    """Monitors Apple Silicon thermal state

    This is a placeholder implementation. In production, this would interface
    with system APIs to get actual thermal data.
    """

    def __init__(self):
        """Initialize thermal monitor"""
        self._last_check_time = 0.0
        self._cached_state = "nominal"
        self._check_interval_s = 10.0  # Check every 10 seconds

    def get_thermal_state(self) -> str:
        """Get current thermal state

        Returns:
            Thermal state: 'nominal', 'fair', 'serious', or 'critical'
        """
        current_time = time.time()

        # Use cached value if within check interval
        if current_time - self._last_check_time < self._check_interval_s:
            return self._cached_state

        # Update thermal state
        self._cached_state = self._check_thermal_state()
        self._last_check_time = current_time

        return self._cached_state

    def _check_thermal_state(self) -> str:
        """Check thermal state from system

        Returns:
            Thermal state string

        Note:
            This is a placeholder. Production implementation would use:
            - IOKit framework on macOS
            - sysctl machdep.xcpm.cpu_thermal_level
            - Or other system thermal APIs
        """
        try:
            import subprocess

            # Try to get thermal pressure (macOS specific)
            result = subprocess.run(
                ["pmset", "-g", "therm"],
                capture_output=True,
                text=True,
                timeout=1.0,
            )

            output = result.stdout.lower()

            if "cpu_scheduler_limit" in output:
                # System is thermally throttled
                if "100" in output:
                    return "critical"
                elif "75" in output:
                    return "serious"
                elif "50" in output:
                    return "fair"

            return "nominal"

        except (subprocess.SubprocessError, FileNotFoundError, subprocess.TimeoutExpired):
            # Default to nominal if unable to check
            return "nominal"


def create_scaling_manager(
    cluster: Any,
    scaling_config: ScalingConfig | None = None,
) -> ScalingManager:
    """Create a new scaling manager

    Args:
        cluster: SharedRayCluster instance
        scaling_config: Optional scaling configuration

    Returns:
        Initialized ScalingManager instance
    """
    return ScalingManager(cluster=cluster, scaling_config=scaling_config)
