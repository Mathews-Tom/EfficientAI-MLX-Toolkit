"""
Data versioning infrastructure for EfficientAI-MLX-Toolkit.

This package provides DVC-based data versioning with support for multiple
storage backends (S3, GCS, Azure Blob, Local) and project namespace isolation.
"""

from mlops.config.dvc_config import (
    DVCConfig,
    DVCConfigError,
    StorageBackend,
    get_default_config,
    load_config_from_file,
)

__all__ = [
    "DVCConfig",
    "DVCConfigError",
    "StorageBackend",
    "get_default_config",
    "load_config_from_file",
]
