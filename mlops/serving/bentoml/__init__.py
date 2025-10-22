"""BentoML Model Packaging Module

This module provides BentoML integration for packaging and serving models with
Apple Silicon optimization. It supports MLX models, Ray Serve integration, and
multi-project model registry.
"""

from __future__ import annotations

__version__ = "0.1.0"

# Import key classes when BentoML is available
try:
    from mlops.serving.bentoml.service import MLXBentoService, BentoMLError
    from mlops.serving.bentoml.packager import ModelPackager, PackageConfig
    from mlops.serving.bentoml.runner import MLXModelRunner
    from mlops.serving.bentoml.config import BentoMLConfig

    __all__ = [
        "MLXBentoService",
        "BentoMLError",
        "ModelPackager",
        "PackageConfig",
        "MLXModelRunner",
        "BentoMLConfig",
    ]

    BENTOML_AVAILABLE = True
except ImportError:
    BENTOML_AVAILABLE = False
    __all__ = []


def is_bentoml_available() -> bool:
    """Check if BentoML is available

    Returns:
        True if bentoml is installed and importable
    """
    return BENTOML_AVAILABLE
