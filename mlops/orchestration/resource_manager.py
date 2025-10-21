"""Apple Silicon Resource Manager for Airflow

This module provides resource management for Airflow tasks optimized for Apple Silicon.
It handles thermal monitoring, memory allocation, and task scheduling based on hardware
capabilities.
"""

from __future__ import annotations

import platform
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from mlops.config import AppleSiliconConfig


class ThermalState(Enum):
    """Thermal states for Apple Silicon"""
    NOMINAL = "nominal"
    MODERATE = "moderate"
    ELEVATED = "elevated"
    CRITICAL = "critical"


class ResourcePriority(Enum):
    """Priority levels for resource allocation"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ResourceAllocation:
    """Resource allocation for a task"""
    cores: int
    memory_gb: float
    priority: ResourcePriority = ResourcePriority.NORMAL
    thermal_aware: bool = True
    max_duration_minutes: int | None = None


@dataclass
class ResourceUsage:
    """Current resource usage metrics"""
    cpu_percent: float
    memory_used_gb: float
    memory_available_gb: float
    thermal_state: ThermalState
    active_tasks: int
    timestamp: float = field(default_factory=lambda: __import__('time').time())


class AppleSiliconResourceManager:
    """Manages resources for Airflow tasks on Apple Silicon

    This manager handles:
    - Thermal monitoring and throttling
    - Memory allocation and tracking
    - CPU core assignment
    - Task scheduling based on hardware state
    """

    def __init__(self, config: AppleSiliconConfig | None = None):
        """Initialize resource manager

        Args:
            config: Apple Silicon configuration (auto-detected if not provided)
        """
        self.config = config or AppleSiliconConfig.detect()
        self._active_allocations: dict[str, ResourceAllocation] = {}
        self._thermal_threshold_moderate = 70.0  # Celsius
        self._thermal_threshold_elevated = 85.0  # Celsius
        self._thermal_threshold_critical = 95.0  # Celsius

    def get_thermal_state(self) -> ThermalState:
        """Get current thermal state of the system

        Returns:
            Current thermal state
        """
        if not self.config.thermal_aware or not self.config.chip_type:
            return ThermalState.NOMINAL

        try:
            # Get CPU temperature using powermetrics
            result = subprocess.run(
                ["sudo", "powermetrics", "--samplers", "thermal", "-n", "1", "-i", "1000"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )

            if result.returncode != 0:
                # Fallback: use sysctl to get thermal pressure
                result = subprocess.run(
                    ["sysctl", "machdep.xcpm.cpu_thermal_level"],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                thermal_level = int(result.stdout.split()[-1])

                if thermal_level >= 75:
                    return ThermalState.CRITICAL
                elif thermal_level >= 50:
                    return ThermalState.ELEVATED
                elif thermal_level >= 25:
                    return ThermalState.MODERATE
                else:
                    return ThermalState.NOMINAL

            # Parse temperature from powermetrics output
            output = result.stdout
            for line in output.split('\n'):
                if 'CPU die temperature' in line:
                    # Extract temperature value
                    temp_str = line.split(':')[-1].strip().split()[0]
                    temperature = float(temp_str)

                    if temperature >= self._thermal_threshold_critical:
                        return ThermalState.CRITICAL
                    elif temperature >= self._thermal_threshold_elevated:
                        return ThermalState.ELEVATED
                    elif temperature >= self._thermal_threshold_moderate:
                        return ThermalState.MODERATE
                    else:
                        return ThermalState.NOMINAL

            return ThermalState.NOMINAL

        except (subprocess.SubprocessError, ValueError, FileNotFoundError):
            # If we can't get thermal data, assume nominal
            return ThermalState.NOMINAL

    def get_memory_usage(self) -> tuple[float, float]:
        """Get current memory usage

        Returns:
            Tuple of (used_gb, available_gb)
        """
        try:
            # Get memory info using vm_stat
            result = subprocess.run(
                ["vm_stat"],
                capture_output=True,
                text=True,
                check=True,
            )

            # Parse vm_stat output
            lines = result.stdout.split('\n')
            page_size = 4096  # Default page size on macOS

            # Extract page statistics
            stats = {}
            for line in lines:
                if ':' in line:
                    key, value = line.split(':')
                    key = key.strip()
                    value = value.strip().rstrip('.')
                    try:
                        stats[key] = int(value)
                    except ValueError:
                        continue

            # Calculate memory usage
            pages_free = stats.get('Pages free', 0)
            pages_active = stats.get('Pages active', 0)
            pages_inactive = stats.get('Pages inactive', 0)
            pages_speculative = stats.get('Pages speculative', 0)
            pages_wired = stats.get('Pages wired down', 0)

            total_pages = pages_free + pages_active + pages_inactive + pages_speculative + pages_wired
            used_pages = pages_active + pages_wired

            total_gb = (total_pages * page_size) / (1024 ** 3)
            used_gb = (used_pages * page_size) / (1024 ** 3)
            available_gb = total_gb - used_gb

            return (used_gb, available_gb)

        except (subprocess.SubprocessError, ValueError, KeyError):
            # Fallback to config values
            return (self.config.memory_gb * 0.5, self.config.memory_gb * 0.5)

    def get_cpu_usage(self) -> float:
        """Get current CPU usage percentage

        Returns:
            CPU usage as percentage (0-100)
        """
        try:
            # Use top command to get CPU usage
            result = subprocess.run(
                ["top", "-l", "1", "-n", "0"],
                capture_output=True,
                text=True,
                check=True,
                timeout=2,
            )

            # Parse CPU usage from top output
            for line in result.stdout.split('\n'):
                if 'CPU usage' in line:
                    # Extract idle percentage
                    idle_str = line.split('idle')[0].split()[-1].rstrip('%')
                    idle = float(idle_str)
                    return 100.0 - idle

            return 0.0

        except (subprocess.SubprocessError, ValueError, FileNotFoundError):
            return 0.0

    def get_resource_usage(self) -> ResourceUsage:
        """Get current resource usage metrics

        Returns:
            ResourceUsage object with current metrics
        """
        cpu_percent = self.get_cpu_usage()
        used_gb, available_gb = self.get_memory_usage()
        thermal_state = self.get_thermal_state()
        active_tasks = len(self._active_allocations)

        return ResourceUsage(
            cpu_percent=cpu_percent,
            memory_used_gb=used_gb,
            memory_available_gb=available_gb,
            thermal_state=thermal_state,
            active_tasks=active_tasks,
        )

    def can_allocate(self, allocation: ResourceAllocation) -> bool:
        """Check if resources can be allocated

        Args:
            allocation: Requested resource allocation

        Returns:
            True if resources can be allocated, False otherwise
        """
        usage = self.get_resource_usage()

        # Check memory availability
        if allocation.memory_gb > usage.memory_available_gb:
            return False

        # Check core availability
        total_allocated_cores = sum(a.cores for a in self._active_allocations.values())
        if total_allocated_cores + allocation.cores > self.config.cores:
            return False

        # Check thermal constraints
        if allocation.thermal_aware and self.config.thermal_aware:
            if usage.thermal_state == ThermalState.CRITICAL:
                return False
            elif usage.thermal_state == ThermalState.ELEVATED:
                # Only allow critical priority tasks
                if allocation.priority != ResourcePriority.CRITICAL:
                    return False
            elif usage.thermal_state == ThermalState.MODERATE:
                # Limit concurrent tasks
                if usage.active_tasks >= self.config.max_concurrent_tasks:
                    return False

        return True

    def allocate(self, task_id: str, allocation: ResourceAllocation) -> bool:
        """Allocate resources for a task

        Args:
            task_id: Unique task identifier
            allocation: Resource allocation request

        Returns:
            True if allocation successful, False otherwise
        """
        if not self.can_allocate(allocation):
            return False

        self._active_allocations[task_id] = allocation
        return True

    def release(self, task_id: str) -> bool:
        """Release resources for a task

        Args:
            task_id: Task identifier to release

        Returns:
            True if resources were released, False if task not found
        """
        if task_id in self._active_allocations:
            del self._active_allocations[task_id]
            return True
        return False

    def get_recommended_allocation(
        self,
        cores: int | None = None,
        memory_gb: float | None = None,
        priority: ResourcePriority = ResourcePriority.NORMAL,
    ) -> ResourceAllocation:
        """Get recommended resource allocation based on current state

        Args:
            cores: Requested cores (auto-determined if not provided)
            memory_gb: Requested memory (auto-determined if not provided)
            priority: Task priority

        Returns:
            Recommended ResourceAllocation
        """
        usage = self.get_resource_usage()

        # Auto-determine cores if not provided
        if cores is None:
            if usage.thermal_state == ThermalState.CRITICAL:
                cores = 1
            elif usage.thermal_state == ThermalState.ELEVATED:
                cores = max(1, self.config.cores // 4)
            elif usage.thermal_state == ThermalState.MODERATE:
                cores = max(1, self.config.cores // 2)
            else:
                cores = self.config.cores

        # Auto-determine memory if not provided
        if memory_gb is None:
            # Leave 20% headroom
            memory_gb = min(
                usage.memory_available_gb * 0.8,
                self.config.memory_gb * 0.8,
            )

        return ResourceAllocation(
            cores=cores,
            memory_gb=memory_gb,
            priority=priority,
            thermal_aware=self.config.thermal_aware,
        )

    def get_allocation_stats(self) -> dict[str, Any]:
        """Get statistics about current allocations

        Returns:
            Dictionary with allocation statistics
        """
        if not self._active_allocations:
            return {
                'active_tasks': 0,
                'total_cores_allocated': 0,
                'total_memory_allocated_gb': 0.0,
                'utilization_cores': 0.0,
                'utilization_memory': 0.0,
            }

        total_cores = sum(a.cores for a in self._active_allocations.values())
        total_memory = sum(a.memory_gb for a in self._active_allocations.values())

        return {
            'active_tasks': len(self._active_allocations),
            'total_cores_allocated': total_cores,
            'total_memory_allocated_gb': total_memory,
            'utilization_cores': (total_cores / self.config.cores) * 100,
            'utilization_memory': (total_memory / self.config.memory_gb) * 100,
            'tasks_by_priority': {
                priority.value: sum(
                    1 for a in self._active_allocations.values()
                    if a.priority == priority
                )
                for priority in ResourcePriority
            },
        }

    def optimize_for_thermal_state(self) -> dict[str, Any]:
        """Optimize resource allocations based on thermal state

        Returns:
            Dictionary with optimization actions taken
        """
        thermal_state = self.get_thermal_state()
        actions = {
            'thermal_state': thermal_state.value,
            'tasks_throttled': [],
            'tasks_paused': [],
        }

        if thermal_state == ThermalState.CRITICAL:
            # Pause all non-critical tasks
            for task_id, allocation in list(self._active_allocations.items()):
                if allocation.priority != ResourcePriority.CRITICAL:
                    actions['tasks_paused'].append(task_id)

        elif thermal_state == ThermalState.ELEVATED:
            # Throttle high-resource tasks
            for task_id, allocation in list(self._active_allocations.items()):
                if allocation.cores > self.config.cores // 2:
                    actions['tasks_throttled'].append(task_id)

        return actions
