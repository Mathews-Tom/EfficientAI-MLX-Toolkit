"""Tests for Thermal-Aware Configuration"""

from __future__ import annotations

import pytest

from mlops.airflow.config import (
    TASK_PROFILES,
    TaskConfig,
    ThermalSchedulingConfig,
    create_task_config,
    get_task_profile,
)


class TestThermalSchedulingConfig:
    """Test ThermalSchedulingConfig"""

    def test_default_config(self):
        """Test default configuration values"""
        config = ThermalSchedulingConfig()

        assert config.thermal_threshold_nominal == 0
        assert config.thermal_threshold_fair == 1
        assert config.thermal_threshold_serious == 2
        assert config.thermal_threshold_critical == 3
        assert config.memory_threshold_percent == 85.0
        assert config.min_health_score == 40.0
        assert config.clearance_timeout_seconds == 300
        assert config.check_interval_seconds == 10

    def test_custom_config(self):
        """Test custom configuration"""
        config = ThermalSchedulingConfig(
            memory_threshold_percent=90.0,
            min_health_score=50.0,
            clearance_timeout_seconds=600,
        )

        assert config.memory_threshold_percent == 90.0
        assert config.min_health_score == 50.0
        assert config.clearance_timeout_seconds == 600

    def test_retry_delays(self):
        """Test retry delay configuration"""
        config = ThermalSchedulingConfig()

        assert config.retry_delays[0] == 10
        assert config.retry_delays[1] == 30
        assert config.retry_delays[2] == 60
        assert config.retry_delays[3] == 120

    def test_throttle_levels(self):
        """Test throttle level configuration"""
        config = ThermalSchedulingConfig()

        assert config.throttle_levels[0] == 1.0
        assert config.throttle_levels[1] == 0.9
        assert config.throttle_levels[2] == 0.7
        assert config.throttle_levels[3] == 0.5

    def test_to_dict(self):
        """Test conversion to dictionary"""
        config = ThermalSchedulingConfig()
        config_dict = config.to_dict()

        assert "thermal_threshold_nominal" in config_dict
        assert "memory_threshold_percent" in config_dict
        assert "retry_delays" in config_dict
        assert "throttle_levels" in config_dict

    def test_from_dict(self):
        """Test creation from dictionary"""
        config_dict = {
            "memory_threshold_percent": 90.0,
            "min_health_score": 50.0,
        }

        config = ThermalSchedulingConfig.from_dict(config_dict)

        assert config.memory_threshold_percent == 90.0
        assert config.min_health_score == 50.0


class TestTaskConfig:
    """Test TaskConfig"""

    def test_default_config(self):
        """Test default task configuration"""
        config = TaskConfig(task_id="test_task")

        assert config.task_id == "test_task"
        assert config.thermal_threshold == 2
        assert config.memory_required_gb is None
        assert config.priority == "normal"
        assert config.retry_on_thermal
        assert config.wait_for_clearance
        assert config.max_retries == 3
        assert config.retry_delay_seconds == 60

    def test_custom_config(self):
        """Test custom task configuration"""
        config = TaskConfig(
            task_id="custom_task",
            thermal_threshold=1,
            memory_required_gb=8.0,
            priority="high",
            retry_on_thermal=False,
            max_retries=5,
        )

        assert config.task_id == "custom_task"
        assert config.thermal_threshold == 1
        assert config.memory_required_gb == 8.0
        assert config.priority == "high"
        assert not config.retry_on_thermal
        assert config.max_retries == 5

    def test_to_dict(self):
        """Test task config to dictionary"""
        config = TaskConfig(
            task_id="test_task",
            thermal_threshold=2,
            memory_required_gb=16.0,
        )

        config_dict = config.to_dict()

        assert config_dict["task_id"] == "test_task"
        assert config_dict["thermal_threshold"] == 2
        assert config_dict["memory_required_gb"] == 16.0

    def test_from_dict(self):
        """Test task config from dictionary"""
        config_dict = {
            "task_id": "test_task",
            "thermal_threshold": 1,
            "priority": "high",
        }

        config = TaskConfig.from_dict(config_dict)

        assert config.task_id == "test_task"
        assert config.thermal_threshold == 1
        assert config.priority == "high"


class TestTaskProfiles:
    """Test predefined task profiles"""

    def test_training_profile(self):
        """Test training task profile"""
        profile = TASK_PROFILES["training"]

        assert profile["thermal_threshold"] == 2
        assert profile["memory_required_gb"] == 16.0
        assert profile["priority"] == "high"
        assert profile["retry_on_thermal"]
        assert profile["wait_for_clearance"]
        assert profile["max_retries"] == 5

    def test_inference_profile(self):
        """Test inference task profile"""
        profile = TASK_PROFILES["inference"]

        assert profile["thermal_threshold"] == 1
        assert profile["memory_required_gb"] == 8.0
        assert profile["priority"] == "normal"
        assert profile["retry_on_thermal"]
        assert not profile["wait_for_clearance"]
        assert profile["max_retries"] == 3

    def test_preprocessing_profile(self):
        """Test preprocessing task profile"""
        profile = TASK_PROFILES["preprocessing"]

        assert profile["thermal_threshold"] == 1
        assert profile["memory_required_gb"] == 4.0
        assert profile["priority"] == "normal"
        assert not profile["retry_on_thermal"]

    def test_evaluation_profile(self):
        """Test evaluation task profile"""
        profile = TASK_PROFILES["evaluation"]

        assert profile["thermal_threshold"] == 1
        assert profile["memory_required_gb"] == 8.0
        assert profile["priority"] == "normal"

    def test_deployment_profile(self):
        """Test deployment task profile"""
        profile = TASK_PROFILES["deployment"]

        assert profile["thermal_threshold"] == 3
        assert profile["priority"] == "critical"
        assert profile["wait_for_clearance"]


class TestHelperFunctions:
    """Test helper functions"""

    def test_get_task_profile(self):
        """Test get_task_profile function"""
        profile = get_task_profile("training")

        assert profile["thermal_threshold"] == 2
        assert profile["memory_required_gb"] == 16.0
        assert isinstance(profile, dict)

    def test_get_task_profile_unknown(self):
        """Test get_task_profile with unknown profile"""
        with pytest.raises(ValueError) as exc_info:
            get_task_profile("unknown_profile")

        assert "Unknown task profile" in str(exc_info.value)

    def test_create_task_config_no_profile(self):
        """Test create_task_config without profile"""
        config = create_task_config("test_task")

        assert config.task_id == "test_task"
        assert config.thermal_threshold == 2  # default

    def test_create_task_config_with_profile(self):
        """Test create_task_config with profile"""
        config = create_task_config("train_task", profile="training")

        assert config.task_id == "train_task"
        assert config.thermal_threshold == 2
        assert config.memory_required_gb == 16.0
        assert config.priority == "high"

    def test_create_task_config_with_overrides(self):
        """Test create_task_config with overrides"""
        config = create_task_config(
            "train_task",
            profile="training",
            thermal_threshold=1,  # Override
            memory_required_gb=32.0,  # Override
        )

        assert config.task_id == "train_task"
        assert config.thermal_threshold == 1  # Overridden
        assert config.memory_required_gb == 32.0  # Overridden
        assert config.priority == "high"  # From profile

    def test_create_task_config_inference_profile(self):
        """Test create_task_config with inference profile"""
        config = create_task_config("inference_task", profile="inference")

        assert config.task_id == "inference_task"
        assert config.thermal_threshold == 1
        assert config.memory_required_gb == 8.0
        assert config.priority == "normal"

    def test_create_task_config_deployment_profile(self):
        """Test create_task_config with deployment profile"""
        config = create_task_config("deploy_task", profile="deployment")

        assert config.task_id == "deploy_task"
        assert config.thermal_threshold == 3  # Can run even in critical
        assert config.priority == "critical"
