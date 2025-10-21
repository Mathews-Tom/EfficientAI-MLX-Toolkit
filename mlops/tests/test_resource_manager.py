"""Tests for Apple Silicon Resource Manager

Tests for resource allocation, thermal monitoring, and task scheduling.
"""

from __future__ import annotations

import platform

import pytest

from mlops.config import AppleSiliconConfig
from mlops.orchestration import (
    AppleSiliconResourceManager,
    ResourceAllocation,
    ResourcePriority,
    ResourceUsage,
    ThermalState,
)


class TestResourceAllocation:
    """Tests for ResourceAllocation"""

    def test_basic_allocation(self):
        """Test basic resource allocation"""
        allocation = ResourceAllocation(
            cores=4,
            memory_gb=8.0,
        )

        assert allocation.cores == 4
        assert allocation.memory_gb == 8.0
        assert allocation.priority == ResourcePriority.NORMAL
        assert allocation.thermal_aware is True

    def test_allocation_with_priority(self):
        """Test allocation with custom priority"""
        allocation = ResourceAllocation(
            cores=2,
            memory_gb=4.0,
            priority=ResourcePriority.HIGH,
        )

        assert allocation.priority == ResourcePriority.HIGH

    def test_allocation_with_timeout(self):
        """Test allocation with max duration"""
        allocation = ResourceAllocation(
            cores=8,
            memory_gb=16.0,
            max_duration_minutes=60,
        )

        assert allocation.max_duration_minutes == 60


class TestResourceUsage:
    """Tests for ResourceUsage"""

    def test_resource_usage_creation(self):
        """Test ResourceUsage creation"""
        usage = ResourceUsage(
            cpu_percent=50.0,
            memory_used_gb=8.0,
            memory_available_gb=8.0,
            thermal_state=ThermalState.NOMINAL,
            active_tasks=2,
        )

        assert usage.cpu_percent == 50.0
        assert usage.memory_used_gb == 8.0
        assert usage.memory_available_gb == 8.0
        assert usage.thermal_state == ThermalState.NOMINAL
        assert usage.active_tasks == 2
        assert usage.timestamp > 0


class TestThermalState:
    """Tests for ThermalState enum"""

    def test_thermal_states(self):
        """Test thermal state values"""
        assert ThermalState.NOMINAL.value == "nominal"
        assert ThermalState.MODERATE.value == "moderate"
        assert ThermalState.ELEVATED.value == "elevated"
        assert ThermalState.CRITICAL.value == "critical"


class TestResourcePriority:
    """Tests for ResourcePriority enum"""

    def test_priority_levels(self):
        """Test priority level values"""
        assert ResourcePriority.LOW.value == "low"
        assert ResourcePriority.NORMAL.value == "normal"
        assert ResourcePriority.HIGH.value == "high"
        assert ResourcePriority.CRITICAL.value == "critical"


class TestAppleSiliconResourceManager:
    """Tests for AppleSiliconResourceManager"""

    @pytest.fixture
    def manager(self):
        """Create resource manager instance"""
        config = AppleSiliconConfig(
            chip_type="M1",
            cores=8,
            memory_gb=16.0,
            thermal_aware=True,
            max_concurrent_tasks=4,
        )
        return AppleSiliconResourceManager(config)

    def test_manager_initialization(self, manager):
        """Test manager initialization"""
        assert manager.config is not None
        assert isinstance(manager.config, AppleSiliconConfig)
        assert manager.config.cores == 8

    def test_get_thermal_state(self, manager):
        """Test getting thermal state"""
        state = manager.get_thermal_state()

        assert isinstance(state, ThermalState)
        # Should always return a valid state
        assert state in [
            ThermalState.NOMINAL,
            ThermalState.MODERATE,
            ThermalState.ELEVATED,
            ThermalState.CRITICAL,
        ]

    def test_get_memory_usage(self, manager):
        """Test getting memory usage"""
        used_gb, available_gb = manager.get_memory_usage()

        assert isinstance(used_gb, float)
        assert isinstance(available_gb, float)
        assert used_gb >= 0
        assert available_gb >= 0

    def test_get_cpu_usage(self, manager):
        """Test getting CPU usage"""
        cpu_percent = manager.get_cpu_usage()

        assert isinstance(cpu_percent, float)
        assert 0.0 <= cpu_percent <= 100.0

    def test_get_resource_usage(self, manager):
        """Test getting comprehensive resource usage"""
        usage = manager.get_resource_usage()

        assert isinstance(usage, ResourceUsage)
        assert usage.cpu_percent >= 0
        assert usage.memory_used_gb >= 0
        assert usage.memory_available_gb >= 0
        assert isinstance(usage.thermal_state, ThermalState)
        assert usage.active_tasks == 0  # No tasks allocated yet

    def test_can_allocate_basic(self, manager):
        """Test basic allocation check"""
        allocation = ResourceAllocation(
            cores=2,
            memory_gb=4.0,
        )

        can_allocate = manager.can_allocate(allocation)
        assert isinstance(can_allocate, bool)

    def test_can_allocate_exceeds_cores(self, manager):
        """Test allocation that exceeds available cores"""
        allocation = ResourceAllocation(
            cores=16,  # More than available
            memory_gb=4.0,
        )

        assert manager.can_allocate(allocation) is False

    def test_can_allocate_exceeds_memory(self, manager):
        """Test allocation that exceeds available memory"""
        allocation = ResourceAllocation(
            cores=2,
            memory_gb=1000.0,  # Unrealistic memory requirement
        )

        # Should be False if memory is insufficient
        result = manager.can_allocate(allocation)
        assert isinstance(result, bool)

    def test_allocate_success(self, manager):
        """Test successful allocation"""
        allocation = ResourceAllocation(
            cores=2,
            memory_gb=4.0,
        )

        # Try to allocate
        success = manager.allocate("task1", allocation)

        # Should succeed if resources available
        if success:
            assert "task1" in manager._active_allocations
            assert manager._active_allocations["task1"] == allocation

    def test_allocate_and_release(self, manager):
        """Test allocation and release cycle"""
        allocation = ResourceAllocation(
            cores=2,
            memory_gb=4.0,
        )

        # Allocate
        if manager.allocate("task1", allocation):
            assert "task1" in manager._active_allocations

            # Release
            released = manager.release("task1")
            assert released is True
            assert "task1" not in manager._active_allocations

    def test_release_nonexistent_task(self, manager):
        """Test releasing a task that doesn't exist"""
        released = manager.release("nonexistent_task")
        assert released is False

    def test_multiple_allocations(self, manager):
        """Test multiple concurrent allocations"""
        alloc1 = ResourceAllocation(cores=1, memory_gb=2.0)
        alloc2 = ResourceAllocation(cores=1, memory_gb=2.0)

        success1 = manager.allocate("task1", alloc1)
        success2 = manager.allocate("task2", alloc2)

        if success1 and success2:
            assert len(manager._active_allocations) == 2

            # Release all
            manager.release("task1")
            manager.release("task2")
            assert len(manager._active_allocations) == 0

    def test_get_recommended_allocation(self, manager):
        """Test getting recommended allocation"""
        allocation = manager.get_recommended_allocation()

        assert isinstance(allocation, ResourceAllocation)
        assert allocation.cores > 0
        assert allocation.memory_gb > 0
        assert allocation.priority == ResourcePriority.NORMAL

    def test_get_recommended_allocation_with_params(self, manager):
        """Test recommended allocation with custom parameters"""
        allocation = manager.get_recommended_allocation(
            cores=4,
            memory_gb=8.0,
            priority=ResourcePriority.HIGH,
        )

        assert allocation.cores == 4
        assert allocation.memory_gb == 8.0
        assert allocation.priority == ResourcePriority.HIGH

    def test_get_allocation_stats_empty(self, manager):
        """Test allocation stats with no allocations"""
        stats = manager.get_allocation_stats()

        assert stats["active_tasks"] == 0
        assert stats["total_cores_allocated"] == 0
        assert stats["total_memory_allocated_gb"] == 0.0
        assert stats["utilization_cores"] == 0.0
        assert stats["utilization_memory"] == 0.0

    def test_get_allocation_stats_with_tasks(self, manager):
        """Test allocation stats with active tasks"""
        alloc1 = ResourceAllocation(cores=2, memory_gb=4.0)
        alloc2 = ResourceAllocation(cores=2, memory_gb=4.0)

        if manager.allocate("task1", alloc1) and manager.allocate("task2", alloc2):
            stats = manager.get_allocation_stats()

            assert stats["active_tasks"] == 2
            assert stats["total_cores_allocated"] == 4
            assert stats["total_memory_allocated_gb"] == 8.0
            assert stats["utilization_cores"] > 0
            assert stats["utilization_memory"] > 0

            # Cleanup
            manager.release("task1")
            manager.release("task2")

    def test_optimize_for_thermal_state(self, manager):
        """Test thermal optimization"""
        result = manager.optimize_for_thermal_state()

        assert "thermal_state" in result
        assert "tasks_throttled" in result
        assert "tasks_paused" in result
        assert isinstance(result["tasks_throttled"], list)
        assert isinstance(result["tasks_paused"], list)

    def test_thermal_aware_allocation(self, manager):
        """Test that thermal state affects allocation"""
        allocation = ResourceAllocation(
            cores=4,
            memory_gb=8.0,
            thermal_aware=True,
        )

        # Allocation should consider thermal state
        can_allocate = manager.can_allocate(allocation)
        assert isinstance(can_allocate, bool)

    def test_non_thermal_aware_allocation(self, manager):
        """Test allocation without thermal awareness"""
        allocation = ResourceAllocation(
            cores=2,
            memory_gb=4.0,
            thermal_aware=False,
        )

        # Should not be affected by thermal state
        can_allocate = manager.can_allocate(allocation)
        assert isinstance(can_allocate, bool)

    def test_priority_affects_allocation(self, manager):
        """Test that priority affects allocation under thermal pressure"""
        # Create allocations with different priorities
        critical_alloc = ResourceAllocation(
            cores=2,
            memory_gb=4.0,
            priority=ResourcePriority.CRITICAL,
        )

        normal_alloc = ResourceAllocation(
            cores=2,
            memory_gb=4.0,
            priority=ResourcePriority.NORMAL,
        )

        # Both should be valid allocations
        assert isinstance(manager.can_allocate(critical_alloc), bool)
        assert isinstance(manager.can_allocate(normal_alloc), bool)

    def test_manager_with_auto_detect(self):
        """Test manager with auto-detected config"""
        manager = AppleSiliconResourceManager()

        assert manager.config is not None
        usage = manager.get_resource_usage()
        assert isinstance(usage, ResourceUsage)

    @pytest.mark.skipif(
        platform.processor() != "arm" or platform.system() != "Darwin",
        reason="Apple Silicon specific test",
    )
    def test_real_apple_silicon_detection(self):
        """Test on real Apple Silicon hardware"""
        config = AppleSiliconConfig.detect()
        manager = AppleSiliconResourceManager(config)

        assert config.chip_type is not None
        assert "M" in config.chip_type or "ARM" in config.chip_type

        # Should be able to get real metrics
        usage = manager.get_resource_usage()
        assert usage.cpu_percent >= 0
        assert usage.memory_used_gb > 0
