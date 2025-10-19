"""MLOps Configuration Module

Provides configuration management for MLOps infrastructure components.
"""

from mlops.config.airflow_config import (
    AirflowConfig,
    AppleSiliconConfig,
    Environment,
    ExecutorType,
    get_airflow_config,
)

try:
    from mlops.config.ray_config import (
        RayServeConfig,
        DeploymentMode,
        ScalingMode,
        get_ray_serve_config,
    )
    RAY_CONFIG_AVAILABLE = True
except ImportError:
    RAY_CONFIG_AVAILABLE = False

__all__ = [
    "AirflowConfig",
    "AppleSiliconConfig",
    "Environment",
    "ExecutorType",
    "get_airflow_config",
]

if RAY_CONFIG_AVAILABLE:
    __all__.extend([
        "RayServeConfig",
        "DeploymentMode",
        "ScalingMode",
        "get_ray_serve_config",
    ])
