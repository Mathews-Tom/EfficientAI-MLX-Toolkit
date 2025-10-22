"""BentoML Configuration Module

Configuration management for BentoML model packaging and serving with Apple Silicon
optimization support.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import platform


class ModelFramework(Enum):
    """Supported ML frameworks for BentoML"""
    MLX = "mlx"
    PYTORCH = "pytorch"
    TRANSFORMERS = "transformers"
    ONNX = "onnx"


class ServingBackend(Enum):
    """Serving backend options"""
    BENTOML = "bentoml"
    RAY_SERVE = "ray_serve"
    HYBRID = "hybrid"  # BentoML packaging + Ray Serve runtime


@dataclass
class AppleSiliconOptimization:
    """Apple Silicon specific optimizations"""
    enable_mps: bool = True
    enable_mlx: bool = True
    enable_unified_memory: bool = True
    enable_ane: bool = False  # Apple Neural Engine
    thermal_aware: bool = True
    max_batch_size: int = 32
    prefetch_batches: int = 2


@dataclass
class BentoMLConfig:
    """BentoML configuration with Apple Silicon support"""

    # Service metadata
    service_name: str = "mlx_model_service"
    service_version: str = "v1.0"
    description: str = "MLX-optimized model service"
    project_name: str = "default"

    # Model configuration
    model_framework: ModelFramework = ModelFramework.MLX
    model_path: Path = field(default_factory=lambda: Path("models"))
    model_name: str = "model"
    model_version: str | None = None

    # Serving configuration
    serving_backend: ServingBackend = ServingBackend.HYBRID
    workers: int = 1
    timeout: int = 300
    max_batch_size: int = 32
    max_latency_ms: int = 10000

    # API configuration
    host: str = "0.0.0.0"
    port: int = 3000
    api_version: str = "v1"
    enable_swagger: bool = True
    enable_metrics: bool = True

    # Resource configuration
    cpu_per_worker: float = 1.0
    memory_per_worker_mb: int = 2048
    gpu_per_worker: int = 0  # Not used on Apple Silicon

    # Apple Silicon optimization
    apple_silicon: AppleSiliconOptimization = field(
        default_factory=AppleSiliconOptimization
    )
    enable_apple_silicon_optimization: bool = True

    # Storage and registry
    bento_store_path: Path = field(default_factory=lambda: Path("bentos"))
    model_store_path: Path = field(default_factory=lambda: Path("models"))

    # Ray Serve integration (when using HYBRID mode)
    ray_serve_enabled: bool = True
    ray_address: str | None = None
    ray_deployment_name: str | None = None

    def __post_init__(self) -> None:
        """Post-initialization configuration"""
        # Convert string paths to Path objects
        if not isinstance(self.model_path, Path):
            self.model_path = Path(self.model_path)
        if not isinstance(self.bento_store_path, Path):
            self.bento_store_path = Path(self.bento_store_path)
        if not isinstance(self.model_store_path, Path):
            self.model_store_path = Path(self.model_store_path)

        # Auto-configure based on Apple Silicon
        if self.enable_apple_silicon_optimization:
            self._apply_apple_silicon_optimizations()

        # Set Ray deployment name from service name if not provided
        if self.ray_deployment_name is None:
            self.ray_deployment_name = f"{self.project_name}_{self.service_name}"

    def _apply_apple_silicon_optimizations(self) -> None:
        """Apply Apple Silicon specific optimizations"""
        system = platform.system()
        machine = platform.machine()

        if system != "Darwin" or machine != "arm64":
            # Not on Apple Silicon, disable optimizations
            self.apple_silicon.enable_mps = False
            self.apple_silicon.enable_mlx = False
            self.apple_silicon.enable_unified_memory = False
            self.apple_silicon.enable_ane = False
            return

        # Adjust batch size for unified memory
        if self.apple_silicon.enable_unified_memory:
            self.max_batch_size = min(
                self.max_batch_size,
                self.apple_silicon.max_batch_size
            )

        # Adjust workers for thermal management
        if self.apple_silicon.thermal_aware:
            # Conservative worker count to prevent thermal throttling
            self.workers = min(self.workers, 2)

    def to_bentoml_config(self) -> dict[str, Any]:
        """Convert to BentoML service configuration"""
        config = {
            "service": {
                "name": self.service_name,
                "version": self.service_version,
                "description": self.description,
            },
            "api_server": {
                "workers": self.workers,
                "timeout": self.timeout,
                "host": self.host,
                "port": self.port,
            },
            "runner": {
                "batching": {
                    "enabled": True,
                    "max_batch_size": self.max_batch_size,
                    "max_latency_ms": self.max_latency_ms,
                },
                "resources": {
                    "cpu": self.cpu_per_worker,
                    "memory": f"{self.memory_per_worker_mb}Mi",
                },
            },
        }

        # Add Apple Silicon metadata
        if self.enable_apple_silicon_optimization:
            config["apple_silicon"] = {
                "enabled": True,
                "mps": self.apple_silicon.enable_mps,
                "mlx": self.apple_silicon.enable_mlx,
                "unified_memory": self.apple_silicon.enable_unified_memory,
                "ane": self.apple_silicon.enable_ane,
            }

        return config

    def to_ray_serve_config(self) -> dict[str, Any]:
        """Convert to Ray Serve deployment configuration"""
        config = {
            "deployment_name": self.ray_deployment_name,
            "num_replicas": self.workers,
            "ray_actor_options": {
                "num_cpus": self.cpu_per_worker,
            },
            "max_concurrent_queries": self.max_batch_size,
        }

        # Add Apple Silicon metadata
        if self.enable_apple_silicon_optimization:
            config["user_config"] = {
                "apple_silicon": True,
                "mlx_optimization": self.apple_silicon.enable_mlx,
                "project_name": self.project_name,
            }

        return config

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> BentoMLConfig:
        """Create configuration from dictionary

        Args:
            config_dict: Configuration dictionary

        Returns:
            BentoMLConfig instance
        """
        # Extract nested configs
        apple_silicon_dict = config_dict.pop("apple_silicon", {})
        apple_silicon = AppleSiliconOptimization(**apple_silicon_dict)

        # Convert enum strings
        if "model_framework" in config_dict:
            config_dict["model_framework"] = ModelFramework(
                config_dict["model_framework"]
            )
        if "serving_backend" in config_dict:
            config_dict["serving_backend"] = ServingBackend(
                config_dict["serving_backend"]
            )

        return cls(apple_silicon=apple_silicon, **config_dict)

    @classmethod
    def detect(cls, project_name: str = "default") -> BentoMLConfig:
        """Auto-detect optimal configuration from system

        Args:
            project_name: Project identifier

        Returns:
            Configured BentoMLConfig instance
        """
        system = platform.system()
        machine = platform.machine()
        is_apple_silicon = system == "Darwin" and machine == "arm64"

        config = cls(project_name=project_name)

        if is_apple_silicon:
            # Auto-configure for Apple Silicon
            config.enable_apple_silicon_optimization = True
            config.model_framework = ModelFramework.MLX
            config.serving_backend = ServingBackend.HYBRID
        else:
            # Standard configuration
            config.enable_apple_silicon_optimization = False
            config.model_framework = ModelFramework.PYTORCH
            config.serving_backend = ServingBackend.BENTOML

        return config


def get_bentoml_config(
    project_name: str = "default",
    model_framework: ModelFramework | None = None,
) -> BentoMLConfig:
    """Get BentoML configuration for project

    Args:
        project_name: Project identifier
        model_framework: Override model framework (default: auto-detect)

    Returns:
        Configured BentoMLConfig instance
    """
    config = BentoMLConfig.detect(project_name=project_name)

    if model_framework is not None:
        config.model_framework = model_framework

    return config
