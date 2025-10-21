"""Ray Serve Configuration Module

This module provides configuration management for Ray Serve with Apple Silicon
optimization support. It handles cluster settings, deployment configurations,
and resource allocation for model serving.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import platform


class DeploymentMode(Enum):
    """Ray Serve deployment modes"""
    LOCAL = "local"
    CLUSTER = "cluster"
    KUBERNETES = "kubernetes"


class ScalingMode(Enum):
    """Auto-scaling modes for Ray Serve"""
    FIXED = "fixed"
    AUTO = "auto"
    MANUAL = "manual"


@dataclass
class AppleSiliconConfig:
    """Apple Silicon specific configuration for Ray Serve"""
    chip_type: str | None = None
    cores: int = 8
    memory_gb: float = 16.0
    thermal_aware: bool = True
    max_replicas: int = 4
    unified_memory: bool = True
    mps_available: bool = False

    @classmethod
    def detect(cls) -> AppleSiliconConfig:
        """Detect Apple Silicon configuration from system"""
        chip = platform.processor()
        is_apple_silicon = chip == "arm" and platform.system() == "Darwin"

        if not is_apple_silicon:
            return cls(
                chip_type=None,
                cores=4,
                memory_gb=8.0,
                thermal_aware=False,
                unified_memory=False,
                mps_available=False
            )

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
                max_replicas = 4
            elif "M2" in brand:
                chip_type = "M2"
                cores = 8
                memory_gb = 24.0
                max_replicas = 6
            elif "M3" in brand:
                chip_type = "M3"
                cores = 12
                memory_gb = 36.0
                max_replicas = 8
            else:
                chip_type = "Unknown ARM"
                cores = 8
                memory_gb = 16.0
                max_replicas = 4

            return cls(
                chip_type=chip_type,
                cores=cores,
                memory_gb=memory_gb,
                thermal_aware=True,
                max_replicas=max_replicas,
                unified_memory=True,
                mps_available=True
            )
        except (subprocess.SubprocessError, FileNotFoundError):
            return cls(
                chip_type="ARM",
                cores=8,
                memory_gb=16.0,
                thermal_aware=True,
                unified_memory=True,
                mps_available=False
            )


@dataclass
class RayServeConfig:
    """Ray Serve configuration with Apple Silicon support"""

    # Core Ray Serve settings
    deployment_mode: DeploymentMode = DeploymentMode.LOCAL
    host: str = "0.0.0.0"
    port: int = 8000
    http_options: dict[str, Any] = field(default_factory=dict)

    # Ray cluster settings
    ray_address: str | None = None  # None = start local cluster
    num_cpus: int | None = None  # Auto-detect from system
    num_gpus: int = 0  # Apple Silicon uses unified memory, not discrete GPUs
    object_store_memory: int | None = None  # Auto-configure based on system

    # Deployment settings
    scaling_mode: ScalingMode = ScalingMode.AUTO
    num_replicas: int = 1
    max_replicas: int = 4
    min_replicas: int = 1
    target_num_ongoing_requests_per_replica: int = 10

    # Resource allocation per replica
    num_cpus_per_replica: float = 1.0
    memory_per_replica_mb: int = 1024

    # Health check settings
    health_check_period_s: int = 10
    health_check_timeout_s: int = 30

    # Logging and monitoring
    log_dir: Path = field(default_factory=lambda: Path("mlops/serving/logs"))
    enable_metrics: bool = True
    metrics_port: int = 9090

    # Apple Silicon specific
    apple_silicon: AppleSiliconConfig = field(default_factory=AppleSiliconConfig.detect)
    enable_apple_silicon_optimization: bool = True
    enable_mlx_optimization: bool = True

    # Project multi-tenancy
    project_name: str = "default"
    deployment_tag: str = "latest"

    def __post_init__(self):
        """Post-initialization configuration adjustments"""
        # Ensure paths are Path objects
        if not isinstance(self.log_dir, Path):
            self.log_dir = Path(self.log_dir)

        # Auto-configure Ray cluster based on Apple Silicon
        if self.enable_apple_silicon_optimization and self.apple_silicon.chip_type:
            self._apply_apple_silicon_optimizations()

        # Set HTTP options defaults
        if not self.http_options:
            self.http_options = {
                "host": self.host,
                "port": self.port,
                "location": "EveryNode",
            }

    def _apply_apple_silicon_optimizations(self):
        """Apply Apple Silicon specific optimizations"""
        if not self.apple_silicon.chip_type:
            return

        # Auto-configure CPUs based on Apple Silicon cores
        if self.num_cpus is None:
            self.num_cpus = self.apple_silicon.cores

        # Adjust replicas based on Apple Silicon capabilities
        if self.scaling_mode == ScalingMode.AUTO:
            self.max_replicas = min(
                self.max_replicas,
                self.apple_silicon.max_replicas
            )

        # Configure object store memory (30% of total RAM for unified memory)
        if self.object_store_memory is None:
            self.object_store_memory = int(
                self.apple_silicon.memory_gb * 1024 * 1024 * 1024 * 0.3
            )

        # Thermal-aware replica limits
        if self.apple_silicon.thermal_aware:
            # Reduce max replicas to prevent thermal throttling
            self.max_replicas = min(
                self.max_replicas,
                max(1, self.apple_silicon.cores // 2)
            )

        # Adjust memory per replica based on available memory
        available_memory_mb = int(self.apple_silicon.memory_gb * 1024)
        # Reserve 4GB for system, divide rest by max replicas
        usable_memory_mb = max(1024, available_memory_mb - 4096)
        self.memory_per_replica_mb = min(
            self.memory_per_replica_mb,
            usable_memory_mb // self.max_replicas
        )

    def to_ray_init_config(self) -> dict[str, Any]:
        """Convert to Ray initialization configuration"""
        config: dict[str, Any] = {}

        if self.ray_address:
            config["address"] = self.ray_address
        else:
            # Local cluster configuration
            if self.num_cpus is not None:
                config["num_cpus"] = self.num_cpus
            if self.num_gpus > 0:
                config["num_gpus"] = self.num_gpus
            if self.object_store_memory:
                config["object_store_memory"] = self.object_store_memory

            # Add logging
            config["logging_level"] = "info"
            config["log_to_driver"] = True

        return config

    def to_serve_config(self) -> dict[str, Any]:
        """Convert to Ray Serve configuration"""
        config = {
            "http_options": self.http_options.copy(),
        }

        # Add Apple Silicon metadata for monitoring
        if self.apple_silicon.chip_type:
            config["apple_silicon_metadata"] = {
                "chip_type": self.apple_silicon.chip_type,
                "cores": self.apple_silicon.cores,
                "memory_gb": self.apple_silicon.memory_gb,
                "unified_memory": self.apple_silicon.unified_memory,
                "mps_available": self.apple_silicon.mps_available,
            }

        return config

    def to_deployment_config(self) -> dict[str, Any]:
        """Convert to Ray Serve deployment configuration"""
        config = {
            "num_replicas": self.num_replicas,
            "ray_actor_options": {
                "num_cpus": self.num_cpus_per_replica,
            },
            "health_check_period_s": self.health_check_period_s,
            "health_check_timeout_s": self.health_check_timeout_s,
        }

        # Add auto-scaling config if enabled
        if self.scaling_mode == ScalingMode.AUTO:
            config["autoscaling_config"] = {
                "min_replicas": self.min_replicas,
                "max_replicas": self.max_replicas,
                "target_num_ongoing_requests_per_replica": (
                    self.target_num_ongoing_requests_per_replica
                ),
            }

        # Add Apple Silicon specific metadata
        if self.apple_silicon.chip_type:
            config["user_config"] = {
                "apple_silicon": True,
                "chip_type": self.apple_silicon.chip_type,
                "mlx_optimization": self.enable_mlx_optimization,
                "project_name": self.project_name,
                "deployment_tag": self.deployment_tag,
            }

        return config

    @classmethod
    def for_deployment_mode(cls, mode: DeploymentMode) -> RayServeConfig:
        """Create configuration for specific deployment mode"""
        base_config = cls.detect()

        if mode == DeploymentMode.LOCAL:
            base_config.deployment_mode = DeploymentMode.LOCAL
            base_config.ray_address = None
            base_config.num_replicas = 1
            base_config.max_replicas = 2
        elif mode == DeploymentMode.CLUSTER:
            base_config.deployment_mode = DeploymentMode.CLUSTER
            base_config.ray_address = "auto"  # Connect to existing cluster
            base_config.num_replicas = 2
            base_config.max_replicas = 10
        elif mode == DeploymentMode.KUBERNETES:
            base_config.deployment_mode = DeploymentMode.KUBERNETES
            base_config.ray_address = "auto"
            base_config.num_replicas = 3
            base_config.max_replicas = 20

        return base_config

    @classmethod
    def detect(cls) -> RayServeConfig:
        """Detect optimal configuration from system"""
        apple_silicon = AppleSiliconConfig.detect()
        return cls(apple_silicon=apple_silicon)


def get_ray_serve_config(
    deployment_mode: DeploymentMode | None = None,
    project_name: str = "default"
) -> RayServeConfig:
    """Get Ray Serve configuration for deployment mode

    Args:
        deployment_mode: Target deployment mode (default: auto-detect or LOCAL)
        project_name: Project identifier for multi-tenant serving

    Returns:
        Configured RayServeConfig instance
    """
    if deployment_mode is None:
        import os
        mode_str = os.getenv("RAY_SERVE_MODE", "local")
        deployment_mode = DeploymentMode(mode_str)

    config = RayServeConfig.for_deployment_mode(deployment_mode)
    config.project_name = project_name
    return config
