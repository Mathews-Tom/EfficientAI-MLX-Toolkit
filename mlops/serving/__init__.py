"""Ray Serve Model Serving Module

This module provides model serving infrastructure using Ray Serve with Apple Silicon
optimization and MLX integration. It supports multi-project model deployment with
auto-scaling and load balancing.
"""

from __future__ import annotations

__version__ = "0.1.0"

# Import key classes when ray is available
try:
    from mlops.serving.ray_serve import RayServeError, SharedRayCluster
    from mlops.serving.model_wrapper import MLXModelWrapper, ModelWrapper
    from mlops.serving.scaling_manager import ScalingManager

    __all__ = [
        "SharedRayCluster",
        "RayServeError",
        "ModelWrapper",
        "MLXModelWrapper",
        "ScalingManager",
    ]

    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    __all__ = []


def is_ray_available() -> bool:
    """Check if Ray Serve is available

    Returns:
        True if ray[serve] is installed and importable
    """
    return RAY_AVAILABLE
