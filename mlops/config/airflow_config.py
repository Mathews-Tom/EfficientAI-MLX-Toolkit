"""Airflow Configuration Module

This module provides configuration management for Apache Airflow with Apple Silicon
optimization support. It handles executor settings, resource allocation, and
environment-specific configurations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import platform


class ExecutorType(Enum):
    """Airflow executor types"""
    LOCAL = "LocalExecutor"
    CELERY = "CeleryExecutor"
    SEQUENTIAL = "SequentialExecutor"
    KUBERNETES = "KubernetesExecutor"


class Environment(Enum):
    """Deployment environment types"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class AppleSiliconConfig:
    """Apple Silicon specific configuration"""
    chip_type: str | None = None
    cores: int = 8
    memory_gb: float = 16.0
    thermal_aware: bool = True
    max_concurrent_tasks: int = 4

    @classmethod
    def detect(cls) -> AppleSiliconConfig:
        """Detect Apple Silicon configuration from system"""
        chip = platform.processor()
        is_apple_silicon = chip == "arm" and platform.system() == "Darwin"

        if not is_apple_silicon:
            return cls(chip_type=None, cores=4, memory_gb=8.0, thermal_aware=False)

        # Detect chip type from sysctl
        import subprocess
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                check=True
            )
            brand = result.stdout.strip()

            if "M1" in brand:
                chip_type = "M1"
                cores = 8
                memory_gb = 16.0
            elif "M2" in brand:
                chip_type = "M2"
                cores = 8
                memory_gb = 24.0
            elif "M3" in brand:
                chip_type = "M3"
                cores = 12
                memory_gb = 36.0
            else:
                chip_type = "Unknown ARM"
                cores = 8
                memory_gb = 16.0

            # Adjust max concurrent tasks based on cores
            max_concurrent = max(2, cores // 2)

            return cls(
                chip_type=chip_type,
                cores=cores,
                memory_gb=memory_gb,
                thermal_aware=True,
                max_concurrent_tasks=max_concurrent
            )
        except (subprocess.SubprocessError, FileNotFoundError):
            return cls(chip_type="ARM", cores=8, memory_gb=16.0, thermal_aware=True)


@dataclass
class AirflowConfig:
    """Airflow configuration with Apple Silicon support"""

    # Core Airflow settings
    executor: ExecutorType = ExecutorType.LOCAL
    environment: Environment = Environment.DEVELOPMENT
    dags_folder: Path = field(default_factory=lambda: Path("mlops/airflow/dags"))
    base_log_folder: Path = field(default_factory=lambda: Path("mlops/airflow/logs"))
    plugins_folder: Path = field(default_factory=lambda: Path("mlops/airflow/plugins"))

    # Database settings
    sql_alchemy_conn: str = "sqlite:///mlops/airflow/airflow.db"
    load_examples: bool = False

    # Scheduler settings
    parallelism: int = 32
    dag_concurrency: int = 16
    max_active_runs_per_dag: int = 16
    worker_concurrency: int = 16

    # Celery settings (when using CeleryExecutor)
    broker_url: str = "redis://localhost:6379/0"
    result_backend: str = "db+postgresql://airflow:airflow@localhost/airflow"
    celery_app_name: str = "airflow.executors.celery_executor"

    # Apple Silicon specific
    apple_silicon: AppleSiliconConfig = field(default_factory=AppleSiliconConfig.detect)
    enable_apple_silicon_optimization: bool = True

    def __post_init__(self):
        """Post-initialization configuration adjustments"""
        # Ensure paths are Path objects
        if not isinstance(self.dags_folder, Path):
            self.dags_folder = Path(self.dags_folder)
        if not isinstance(self.base_log_folder, Path):
            self.base_log_folder = Path(self.base_log_folder)
        if not isinstance(self.plugins_folder, Path):
            self.plugins_folder = Path(self.plugins_folder)

        # Apply Apple Silicon optimizations if enabled
        if self.enable_apple_silicon_optimization and self.apple_silicon.chip_type:
            self._apply_apple_silicon_optimizations()

    def _apply_apple_silicon_optimizations(self):
        """Apply Apple Silicon specific optimizations"""
        if not self.apple_silicon.chip_type:
            return

        # Adjust parallelism based on Apple Silicon cores
        self.parallelism = min(self.parallelism, self.apple_silicon.cores * 4)
        self.dag_concurrency = min(self.dag_concurrency, self.apple_silicon.cores * 2)
        self.worker_concurrency = min(self.worker_concurrency, self.apple_silicon.cores * 2)

        # Thermal-aware adjustments
        if self.apple_silicon.thermal_aware:
            self.max_active_runs_per_dag = min(
                self.max_active_runs_per_dag,
                self.apple_silicon.max_concurrent_tasks
            )

    def to_airflow_config(self) -> dict[str, Any]:
        """Convert to Airflow configuration dictionary"""
        config = {
            "core": {
                "executor": self.executor.value,
                "dags_folder": str(self.dags_folder.absolute()),
                "base_log_folder": str(self.base_log_folder.absolute()),
                "plugins_folder": str(self.plugins_folder.absolute()),
                "sql_alchemy_conn": self.sql_alchemy_conn,
                "load_examples": self.load_examples,
                "parallelism": self.parallelism,
                "dag_concurrency": self.dag_concurrency,
                "max_active_runs_per_dag": self.max_active_runs_per_dag,
            },
            "scheduler": {
                "catchup_by_default": False,
                "max_threads": min(2, self.apple_silicon.cores // 4) if self.apple_silicon.chip_type else 2,
            },
            "webserver": {
                "web_server_port": 8080,
                "web_server_host": "0.0.0.0",
                "secret_key": "temporary_secret_key",  # Should be replaced in production
            },
        }

        # Add Celery configuration if using CeleryExecutor
        if self.executor == ExecutorType.CELERY:
            config["celery"] = {
                "broker_url": self.broker_url,
                "result_backend": self.result_backend,
                "celery_app_name": self.celery_app_name,
                "worker_concurrency": self.worker_concurrency,
            }

        # Add Apple Silicon metadata
        if self.apple_silicon.chip_type:
            config["apple_silicon"] = {
                "chip_type": self.apple_silicon.chip_type,
                "cores": self.apple_silicon.cores,
                "memory_gb": self.apple_silicon.memory_gb,
                "thermal_aware": self.apple_silicon.thermal_aware,
                "max_concurrent_tasks": self.apple_silicon.max_concurrent_tasks,
            }

        return config

    @classmethod
    def for_environment(cls, env: Environment) -> AirflowConfig:
        """Create configuration for specific environment"""
        base_config = cls.detect()

        if env == Environment.DEVELOPMENT:
            base_config.executor = ExecutorType.LOCAL
            base_config.parallelism = 4
            base_config.dag_concurrency = 2
            base_config.max_active_runs_per_dag = 1
            base_config.load_examples = False
        elif env == Environment.STAGING:
            base_config.executor = ExecutorType.LOCAL
            base_config.parallelism = 16
            base_config.dag_concurrency = 8
            base_config.max_active_runs_per_dag = 4
        elif env == Environment.PRODUCTION:
            base_config.executor = ExecutorType.CELERY
            base_config.parallelism = 32
            base_config.dag_concurrency = 16
            base_config.max_active_runs_per_dag = 16

        base_config.environment = env
        return base_config

    @classmethod
    def detect(cls) -> AirflowConfig:
        """Detect optimal configuration from system"""
        apple_silicon = AppleSiliconConfig.detect()
        return cls(apple_silicon=apple_silicon)


def get_airflow_config(environment: Environment | None = None) -> AirflowConfig:
    """Get Airflow configuration for environment

    Args:
        environment: Target environment (default: auto-detect from ENV var or development)

    Returns:
        Configured AirflowConfig instance
    """
    if environment is None:
        import os
        env_str = os.getenv("AIRFLOW_ENV", "development")
        environment = Environment(env_str)

    return AirflowConfig.for_environment(environment)
