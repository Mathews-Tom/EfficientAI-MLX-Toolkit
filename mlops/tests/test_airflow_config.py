"""Tests for Airflow Configuration Module

Tests for Apple Silicon detection, configuration management, and environment-specific
settings.
"""

from __future__ import annotations

import platform
from pathlib import Path

import pytest

from mlops.config import (
    AirflowConfig,
    AppleSiliconConfig,
    Environment,
    ExecutorType,
    get_airflow_config,
)


class TestAppleSiliconConfig:
    """Tests for AppleSiliconConfig"""

    def test_detect_creates_config(self):
        """Test that detect() creates a valid configuration"""
        config = AppleSiliconConfig.detect()

        assert config is not None
        assert isinstance(config.cores, int)
        assert config.cores > 0
        assert isinstance(config.memory_gb, float)
        assert config.memory_gb > 0
        assert isinstance(config.thermal_aware, bool)
        assert isinstance(config.max_concurrent_tasks, int)
        assert config.max_concurrent_tasks > 0

    def test_detect_on_apple_silicon(self):
        """Test detection on Apple Silicon"""
        is_apple_silicon = (
            platform.processor() == "arm" and platform.system() == "Darwin"
        )

        config = AppleSiliconConfig.detect()

        if is_apple_silicon:
            assert config.chip_type is not None
            assert config.thermal_aware is True
        else:
            # On non-Apple Silicon, chip_type should be None
            assert config.chip_type is None or config.chip_type == "Unknown ARM"

    def test_manual_config_creation(self):
        """Test manual creation of AppleSiliconConfig"""
        config = AppleSiliconConfig(
            chip_type="M1",
            cores=8,
            memory_gb=16.0,
            thermal_aware=True,
            max_concurrent_tasks=4,
        )

        assert config.chip_type == "M1"
        assert config.cores == 8
        assert config.memory_gb == 16.0
        assert config.thermal_aware is True
        assert config.max_concurrent_tasks == 4

    def test_default_values(self):
        """Test default values for AppleSiliconConfig"""
        config = AppleSiliconConfig()

        assert config.chip_type is None
        assert config.cores == 8
        assert config.memory_gb == 16.0
        assert config.thermal_aware is True
        assert config.max_concurrent_tasks == 4


class TestAirflowConfig:
    """Tests for AirflowConfig"""

    def test_default_config_creation(self):
        """Test creation with default values"""
        config = AirflowConfig()

        assert config.executor == ExecutorType.LOCAL
        assert config.environment == Environment.DEVELOPMENT
        assert isinstance(config.dags_folder, Path)
        assert isinstance(config.base_log_folder, Path)
        assert isinstance(config.plugins_folder, Path)

    def test_detect_creates_valid_config(self):
        """Test that detect() creates a valid configuration"""
        config = AirflowConfig.detect()

        assert config is not None
        assert isinstance(config.apple_silicon, AppleSiliconConfig)
        assert config.enable_apple_silicon_optimization is True

    def test_apple_silicon_optimization_applied(self):
        """Test that Apple Silicon optimizations are applied"""
        apple_config = AppleSiliconConfig(
            chip_type="M1",
            cores=8,
            memory_gb=16.0,
            thermal_aware=True,
            max_concurrent_tasks=4,
        )

        config = AirflowConfig(
            apple_silicon=apple_config,
            enable_apple_silicon_optimization=True,
            parallelism=64,  # Will be adjusted
        )

        # Parallelism should be adjusted to cores * 4 (8 * 4 = 32)
        assert config.parallelism <= apple_config.cores * 4

    def test_to_airflow_config_dict(self):
        """Test conversion to Airflow configuration dictionary"""
        config = AirflowConfig()
        airflow_dict = config.to_airflow_config()

        assert "core" in airflow_dict
        assert "scheduler" in airflow_dict
        assert "webserver" in airflow_dict

        # Check core settings
        assert airflow_dict["core"]["executor"] == ExecutorType.LOCAL.value
        assert "dags_folder" in airflow_dict["core"]
        assert "sql_alchemy_conn" in airflow_dict["core"]

    def test_celery_config_included(self):
        """Test that Celery config is included when using CeleryExecutor"""
        config = AirflowConfig(executor=ExecutorType.CELERY)
        airflow_dict = config.to_airflow_config()

        assert "celery" in airflow_dict
        assert "broker_url" in airflow_dict["celery"]
        assert "result_backend" in airflow_dict["celery"]

    def test_environment_specific_configs(self):
        """Test environment-specific configurations"""
        dev_config = AirflowConfig.for_environment(Environment.DEVELOPMENT)
        assert dev_config.executor == ExecutorType.LOCAL
        assert dev_config.environment == Environment.DEVELOPMENT
        assert dev_config.max_active_runs_per_dag == 1

        staging_config = AirflowConfig.for_environment(Environment.STAGING)
        assert staging_config.executor == ExecutorType.LOCAL
        assert staging_config.environment == Environment.STAGING
        assert staging_config.max_active_runs_per_dag == 4

        prod_config = AirflowConfig.for_environment(Environment.PRODUCTION)
        assert prod_config.executor == ExecutorType.CELERY
        assert prod_config.environment == Environment.PRODUCTION
        assert prod_config.max_active_runs_per_dag == 16

    def test_path_conversion(self):
        """Test that paths are converted to Path objects"""
        config = AirflowConfig(
            dags_folder="mlops/airflow/dags",
            base_log_folder="mlops/airflow/logs",
        )

        assert isinstance(config.dags_folder, Path)
        assert isinstance(config.base_log_folder, Path)

    def test_thermal_aware_adjustments(self):
        """Test thermal-aware adjustments"""
        apple_config = AppleSiliconConfig(
            chip_type="M2",
            cores=8,
            thermal_aware=True,
            max_concurrent_tasks=3,
        )

        config = AirflowConfig(
            apple_silicon=apple_config,
            enable_apple_silicon_optimization=True,
            max_active_runs_per_dag=10,
        )

        # Should be limited to max_concurrent_tasks
        assert config.max_active_runs_per_dag <= apple_config.max_concurrent_tasks


class TestGetAirflowConfig:
    """Tests for get_airflow_config function"""

    def test_get_airflow_config_default(self):
        """Test getting config with defaults"""
        config = get_airflow_config()

        assert config is not None
        assert isinstance(config, AirflowConfig)

    def test_get_airflow_config_with_environment(self):
        """Test getting config for specific environment"""
        config = get_airflow_config(Environment.PRODUCTION)

        assert config.environment == Environment.PRODUCTION
        assert config.executor == ExecutorType.CELERY

    def test_get_airflow_config_development(self):
        """Test getting development config"""
        config = get_airflow_config(Environment.DEVELOPMENT)

        assert config.environment == Environment.DEVELOPMENT
        assert config.executor == ExecutorType.LOCAL
        assert config.parallelism == 4


class TestExecutorType:
    """Tests for ExecutorType enum"""

    def test_executor_values(self):
        """Test executor type values"""
        assert ExecutorType.LOCAL.value == "LocalExecutor"
        assert ExecutorType.CELERY.value == "CeleryExecutor"
        assert ExecutorType.SEQUENTIAL.value == "SequentialExecutor"
        assert ExecutorType.KUBERNETES.value == "KubernetesExecutor"


class TestEnvironment:
    """Tests for Environment enum"""

    def test_environment_values(self):
        """Test environment values"""
        assert Environment.DEVELOPMENT.value == "development"
        assert Environment.STAGING.value == "staging"
        assert Environment.PRODUCTION.value == "production"
