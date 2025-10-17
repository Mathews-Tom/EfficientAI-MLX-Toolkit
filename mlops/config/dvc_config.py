"""
DVC configuration module with storage backend abstraction.

This module provides centralized DVC configuration management for the
EfficientAI-MLX-Toolkit, supporting multiple storage backends (S3, GCS, Azure Blob,
Local) with validation and project namespace isolation.
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

StorageBackend = Literal["s3", "gcs", "azure", "local"]
EnvironmentType = Literal["development", "production", "testing"]


class DVCConfigError(Exception):
    """Raised when DVC configuration operations fail."""

    def __init__(
        self,
        message: str,
        environment: str | None = None,
        backend: str | None = None,
        details: dict[str, str | int | float | bool] | None = None,
    ) -> None:
        super().__init__(message)
        self.environment = environment
        self.backend = backend
        self.details = dict(details or {})


@dataclass
class DVCConfig:
    """
    DVC configuration with storage backend abstraction.

    Attributes:
        storage_backend: Storage backend type (s3, gcs, azure, local)
        remote_name: Name of the DVC remote
        remote_url: URL/path to remote storage
        cache_dir: Local cache directory for DVC
        project_namespace: Project namespace for isolation
        environment: Environment type (development, production, testing)
        enable_autostage: Automatically stage DVC files in git
        enable_symlinks: Use symlinks for cache
        enable_hardlinks: Use hardlinks for cache
        verify_checksums: Verify file checksums on pull
        jobs: Number of parallel jobs for push/pull
        s3_region: AWS S3 region (for S3 backend)
        s3_profile: AWS profile name (for S3 backend)
        s3_endpoint_url: Custom S3 endpoint URL (for S3-compatible services)
        gcs_project: GCP project ID (for GCS backend)
        azure_account_name: Azure storage account name (for Azure backend)
        azure_container_name: Azure container name (for Azure backend)
        tags: Additional metadata tags
    """

    storage_backend: StorageBackend = "local"
    remote_name: str = "storage"
    remote_url: str = "./dvc-storage"
    cache_dir: str = ".dvc/cache"
    project_namespace: str = "default"
    environment: EnvironmentType = "development"
    enable_autostage: bool = True
    enable_symlinks: bool = True
    enable_hardlinks: bool = True
    verify_checksums: bool = True
    jobs: int = 4
    s3_region: str | None = None
    s3_profile: str | None = None
    s3_endpoint_url: str | None = None
    gcs_project: str | None = None
    azure_account_name: str | None = None
    azure_container_name: str | None = None
    tags: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        self._validate_config()
        self._apply_environment_overrides()
        self._set_defaults_by_environment()

    def _validate_config(self) -> None:
        """Validate configuration settings."""
        if not self.remote_name:
            raise DVCConfigError(
                "remote_name cannot be empty",
                environment=self.environment,
                backend=self.storage_backend,
            )

        if not self.remote_url:
            raise DVCConfigError(
                "remote_url cannot be empty",
                environment=self.environment,
                backend=self.storage_backend,
            )

        if not self.project_namespace:
            raise DVCConfigError(
                "project_namespace cannot be empty",
                environment=self.environment,
                backend=self.storage_backend,
            )

        # Validate storage backend
        if self.storage_backend not in ["s3", "gcs", "azure", "local"]:
            raise DVCConfigError(
                f"Invalid storage backend: {self.storage_backend}",
                environment=self.environment,
                backend=self.storage_backend,
                details={"valid_backends": ["s3", "gcs", "azure", "local"]},
            )

        # Validate environment type
        if self.environment not in ["development", "production", "testing"]:
            raise DVCConfigError(
                f"Invalid environment: {self.environment}",
                environment=self.environment,
                backend=self.storage_backend,
                details={"valid_environments": ["development", "production", "testing"]},
            )

        # Validate backend-specific requirements
        self._validate_backend_config()

    def _validate_backend_config(self) -> None:
        """Validate backend-specific configuration."""
        if self.storage_backend == "s3":
            if not self.remote_url.startswith("s3://"):
                raise DVCConfigError(
                    "S3 backend requires remote_url to start with 's3://'",
                    environment=self.environment,
                    backend=self.storage_backend,
                )

        elif self.storage_backend == "gcs":
            if not self.remote_url.startswith("gs://"):
                raise DVCConfigError(
                    "GCS backend requires remote_url to start with 'gs://'",
                    environment=self.environment,
                    backend=self.storage_backend,
                )

        elif self.storage_backend == "azure":
            if not self.remote_url.startswith("azure://"):
                raise DVCConfigError(
                    "Azure backend requires remote_url to start with 'azure://'",
                    environment=self.environment,
                    backend=self.storage_backend,
                )
            if not self.azure_account_name:
                raise DVCConfigError(
                    "Azure backend requires azure_account_name",
                    environment=self.environment,
                    backend=self.storage_backend,
                )

        elif self.storage_backend == "local":
            # Local backend can use relative or absolute paths
            pass

    def _apply_environment_overrides(self) -> None:
        """Apply environment variable overrides to configuration."""
        env_mapping = {
            "DVC_STORAGE_BACKEND": "storage_backend",
            "DVC_REMOTE_NAME": "remote_name",
            "DVC_REMOTE_URL": "remote_url",
            "DVC_CACHE_DIR": "cache_dir",
            "DVC_PROJECT_NAMESPACE": "project_namespace",
            "DVC_ENVIRONMENT": "environment",
            "DVC_S3_REGION": "s3_region",
            "DVC_S3_PROFILE": "s3_profile",
            "DVC_S3_ENDPOINT_URL": "s3_endpoint_url",
            "DVC_GCS_PROJECT": "gcs_project",
            "DVC_AZURE_ACCOUNT_NAME": "azure_account_name",
            "DVC_AZURE_CONTAINER_NAME": "azure_container_name",
        }

        for env_var, attr_name in env_mapping.items():
            if env_value := os.environ.get(env_var):
                setattr(self, attr_name, env_value)
                logger.debug("Applied environment override: %s = %s", attr_name, env_value)

        # Boolean environment variables
        bool_mapping = {
            "DVC_ENABLE_AUTOSTAGE": "enable_autostage",
            "DVC_ENABLE_SYMLINKS": "enable_symlinks",
            "DVC_ENABLE_HARDLINKS": "enable_hardlinks",
            "DVC_VERIFY_CHECKSUMS": "verify_checksums",
        }

        for env_var, attr_name in bool_mapping.items():
            if env_value := os.environ.get(env_var):
                setattr(self, attr_name, env_value.lower() in ("true", "1", "yes", "on"))
                logger.debug("Applied boolean override: %s = %s", attr_name, getattr(self, attr_name))

        # Integer environment variables
        if jobs_value := os.environ.get("DVC_JOBS"):
            try:
                self.jobs = int(jobs_value)
                logger.debug("Applied jobs override: %s", self.jobs)
            except ValueError:
                logger.warning("Invalid DVC_JOBS value: %s, using default", jobs_value)

    def _set_defaults_by_environment(self) -> None:
        """Set default values based on environment type."""
        if self.environment == "development":
            # Development defaults - local storage
            if self.storage_backend == "local" and self.remote_url == "./dvc-storage":
                self.remote_url = f"./dvc-storage/{self.project_namespace}"

        elif self.environment == "production":
            # Production defaults - warn about local storage
            if self.storage_backend == "local":
                logger.warning(
                    "Production environment using local storage backend. "
                    "Consider using S3, GCS, or Azure for better reliability."
                )

        elif self.environment == "testing":
            # Testing defaults - temporary local storage
            if self.storage_backend == "local":
                self.remote_url = f"/tmp/dvc-test-storage/{self.project_namespace}"
                self.cache_dir = "/tmp/dvc-test-cache"

    def to_dict(self) -> dict[str, str | int | bool | dict[str, str] | None]:
        """
        Convert configuration to dictionary.

        Returns:
            Dictionary representation of configuration
        """
        return {
            "storage_backend": self.storage_backend,
            "remote_name": self.remote_name,
            "remote_url": self.remote_url,
            "cache_dir": self.cache_dir,
            "project_namespace": self.project_namespace,
            "environment": self.environment,
            "enable_autostage": self.enable_autostage,
            "enable_symlinks": self.enable_symlinks,
            "enable_hardlinks": self.enable_hardlinks,
            "verify_checksums": self.verify_checksums,
            "jobs": self.jobs,
            "s3_region": self.s3_region,
            "s3_profile": self.s3_profile,
            "s3_endpoint_url": self.s3_endpoint_url,
            "gcs_project": self.gcs_project,
            "azure_account_name": self.azure_account_name,
            "azure_container_name": self.azure_container_name,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, config_dict: dict[str, str | int | bool | dict[str, str] | None]) -> "DVCConfig":
        """
        Create configuration from dictionary.

        Args:
            config_dict: Configuration dictionary

        Returns:
            DVCConfig instance
        """
        # Filter out None values and unknown keys
        valid_keys = {
            "storage_backend",
            "remote_name",
            "remote_url",
            "cache_dir",
            "project_namespace",
            "environment",
            "enable_autostage",
            "enable_symlinks",
            "enable_hardlinks",
            "verify_checksums",
            "jobs",
            "s3_region",
            "s3_profile",
            "s3_endpoint_url",
            "gcs_project",
            "azure_account_name",
            "azure_container_name",
            "tags",
        }

        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_keys and v is not None}

        return cls(**filtered_dict)

    @classmethod
    def from_environment(
        cls,
        environment: EnvironmentType = "development",
        project_namespace: str = "default",
    ) -> "DVCConfig":
        """
        Create configuration for specific environment and project.

        Args:
            environment: Environment type
            project_namespace: Project namespace for isolation

        Returns:
            DVCConfig instance configured for environment
        """
        config = cls(environment=environment, project_namespace=project_namespace)
        logger.info(
            "Created DVC configuration for environment: %s, project: %s",
            environment,
            project_namespace,
        )
        return config

    def validate(self) -> bool:
        """
        Validate current configuration.

        Returns:
            True if configuration is valid

        Raises:
            DVCConfigError: If validation fails
        """
        self._validate_config()
        logger.info("DVC configuration validation passed")
        return True

    def get_connection_info(self) -> dict[str, str | None]:
        """
        Get connection information for DVC remote.

        Returns:
            Dictionary with connection details
        """
        info: dict[str, str | None] = {
            "storage_backend": self.storage_backend,
            "remote_name": self.remote_name,
            "remote_url": self.remote_url,
            "project_namespace": self.project_namespace,
            "environment": self.environment,
        }

        # Add backend-specific info
        if self.storage_backend == "s3":
            info.update(
                {
                    "s3_region": self.s3_region,
                    "s3_profile": self.s3_profile,
                    "s3_endpoint_url": self.s3_endpoint_url,
                }
            )
        elif self.storage_backend == "gcs":
            info["gcs_project"] = self.gcs_project
        elif self.storage_backend == "azure":
            info.update(
                {
                    "azure_account_name": self.azure_account_name,
                    "azure_container_name": self.azure_container_name,
                }
            )

        return info

    def update_tags(self, new_tags: dict[str, str]) -> None:
        """
        Update configuration tags.

        Args:
            new_tags: Tags to add or update
        """
        self.tags.update(new_tags)
        logger.debug("Updated tags: %s", new_tags)

    def get_cache_path(self, relative_path: str = "") -> Path:
        """
        Get absolute cache path.

        Args:
            relative_path: Relative path within cache directory

        Returns:
            Absolute path to cache
        """
        base_path = Path(self.cache_dir)
        if relative_path:
            return base_path / relative_path
        return base_path

    def ensure_cache_dir(self) -> Path:
        """
        Ensure cache directory exists.

        Returns:
            Path to cache directory

        Raises:
            DVCConfigError: If directory creation fails
        """
        try:
            cache_path = Path(self.cache_dir)
            cache_path.mkdir(parents=True, exist_ok=True)
            logger.info("Cache directory ensured: %s", cache_path)
            return cache_path
        except Exception as e:
            raise DVCConfigError(
                f"Failed to create cache directory: {e}",
                environment=self.environment,
                backend=self.storage_backend,
            ) from e

    def get_remote_path(self, relative_path: str = "") -> str:
        """
        Get remote path with project namespace.

        Args:
            relative_path: Relative path within project namespace

        Returns:
            Full remote path
        """
        if self.storage_backend == "local":
            base = Path(self.remote_url)
            if relative_path:
                return str(base / relative_path)
            return str(base)

        # For cloud backends, append to URL
        base_url = self.remote_url.rstrip("/")
        if relative_path:
            return f"{base_url}/{self.project_namespace}/{relative_path}"
        return f"{base_url}/{self.project_namespace}"


def get_default_config(
    environment: EnvironmentType = "development",
    project_namespace: str = "default",
) -> DVCConfig:
    """
    Get default DVC configuration for environment and project.

    Args:
        environment: Environment type
        project_namespace: Project namespace

    Returns:
        Default DVCConfig instance

    Example:
        >>> config = get_default_config("production", "lora-finetuning-mlx")
        >>> print(config.remote_url)
    """
    return DVCConfig.from_environment(environment, project_namespace)


def load_config_from_file(
    config_path: Path,
    environment: EnvironmentType | None = None,
    project_namespace: str | None = None,
) -> DVCConfig:
    """
    Load DVC configuration from file.

    Args:
        config_path: Path to configuration file (YAML, JSON, or TOML)
        environment: Optional environment override
        project_namespace: Optional project namespace override

    Returns:
        DVCConfig instance

    Raises:
        DVCConfigError: If loading fails
    """
    try:
        from utils.config_manager import ConfigManager

        # Check if file exists
        if not config_path.exists():
            raise DVCConfigError(
                f"Configuration file not found: {config_path}",
                details={"config_path": str(config_path)},
            )

        config_manager = ConfigManager(config_path)
        dvc_config = config_manager.get("dvc", {})

        if not isinstance(dvc_config, dict):
            raise DVCConfigError(
                "Invalid DVC configuration format in file",
                details={"config_path": str(config_path)},
            )

        # Override environment and project_namespace if specified
        if environment:
            dvc_config["environment"] = environment
        if project_namespace:
            dvc_config["project_namespace"] = project_namespace

        return DVCConfig.from_dict(dvc_config)

    except DVCConfigError:
        # Re-raise DVCConfigError without wrapping
        raise
    except Exception as e:
        raise DVCConfigError(
            f"Failed to load configuration from {config_path}",
            details={"error": str(e)},
        ) from e
