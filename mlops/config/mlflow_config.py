"""
MLFlow configuration module with environment-based settings.

This module provides centralized MLFlow configuration management for the
EfficientAI-MLX-Toolkit, supporting multiple environments (development, production)
with validation and Apple Silicon optimization settings.
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

EnvironmentType = Literal["development", "production", "testing"]


class MLFlowConfigError(Exception):
    """Raised when MLFlow configuration operations fail."""

    def __init__(
        self,
        message: str,
        environment: str | None = None,
        details: dict[str, str | int | float | bool] | None = None,
    ) -> None:
        super().__init__(message)
        self.environment = environment
        self.details = dict(details or {})


@dataclass
class MLFlowConfig:
    """
    MLFlow configuration with environment-based settings.

    Attributes:
        tracking_uri: MLFlow tracking server URI
        experiment_name: Default experiment name
        artifact_location: Path for storing artifacts
        backend_store_uri: Backend database URI for MLFlow
        default_artifact_root: Default root for artifacts
        registry_uri: Model registry URI
        environment: Environment type (development, production, testing)
        enable_system_metrics: Enable automatic system metrics logging
        enable_apple_silicon_metrics: Enable Apple Silicon specific metrics
        log_models: Automatically log models
        log_artifacts: Automatically log artifacts
        log_params: Automatically log parameters
        log_metrics: Automatically log metrics
    """

    tracking_uri: str = "http://localhost:5000"
    experiment_name: str = "default"
    artifact_location: str = "./mlruns"
    backend_store_uri: str = "sqlite:///mlflow.db"
    default_artifact_root: str | None = None
    registry_uri: str | None = None
    environment: EnvironmentType = "development"
    enable_system_metrics: bool = True
    enable_apple_silicon_metrics: bool = True
    log_models: bool = True
    log_artifacts: bool = True
    log_params: bool = True
    log_metrics: bool = True
    tags: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        self._validate_config()
        self._apply_environment_overrides()
        self._set_defaults_by_environment()

    def _validate_config(self) -> None:
        """Validate configuration settings."""
        if not self.tracking_uri:
            raise MLFlowConfigError(
                "tracking_uri cannot be empty",
                environment=self.environment,
            )

        if not self.experiment_name:
            raise MLFlowConfigError(
                "experiment_name cannot be empty",
                environment=self.environment,
            )

        # Validate environment type
        if self.environment not in ["development", "production", "testing"]:
            raise MLFlowConfigError(
                f"Invalid environment: {self.environment}",
                environment=self.environment,
                details={"valid_environments": ["development", "production", "testing"]},
            )

    def _apply_environment_overrides(self) -> None:
        """Apply environment variable overrides to configuration."""
        env_mapping = {
            "MLFLOW_TRACKING_URI": "tracking_uri",
            "MLFLOW_EXPERIMENT_NAME": "experiment_name",
            "MLFLOW_ARTIFACT_LOCATION": "artifact_location",
            "MLFLOW_BACKEND_STORE_URI": "backend_store_uri",
            "MLFLOW_DEFAULT_ARTIFACT_ROOT": "default_artifact_root",
            "MLFLOW_REGISTRY_URI": "registry_uri",
            "MLFLOW_ENVIRONMENT": "environment",
        }

        for env_var, attr_name in env_mapping.items():
            if env_value := os.environ.get(env_var):
                setattr(self, attr_name, env_value)
                logger.debug("Applied environment override: %s = %s", attr_name, env_value)

        # Boolean environment variables
        bool_mapping = {
            "MLFLOW_ENABLE_SYSTEM_METRICS": "enable_system_metrics",
            "MLFLOW_ENABLE_APPLE_SILICON_METRICS": "enable_apple_silicon_metrics",
            "MLFLOW_LOG_MODELS": "log_models",
            "MLFLOW_LOG_ARTIFACTS": "log_artifacts",
            "MLFLOW_LOG_PARAMS": "log_params",
            "MLFLOW_LOG_METRICS": "log_metrics",
        }

        for env_var, attr_name in bool_mapping.items():
            if env_value := os.environ.get(env_var):
                setattr(self, attr_name, env_value.lower() in ("true", "1", "yes", "on"))
                logger.debug("Applied boolean override: %s = %s", attr_name, getattr(self, attr_name))

    def _set_defaults_by_environment(self) -> None:
        """Set default values based on environment type."""
        if self.environment == "development":
            # Development defaults - local storage
            if self.tracking_uri == "http://localhost:5000":
                self.tracking_uri = "http://localhost:5000"
            if self.backend_store_uri == "sqlite:///mlflow.db":
                self.backend_store_uri = "sqlite:///mlflow.db"

        elif self.environment == "production":
            # Production defaults - require proper URIs
            if "localhost" in self.tracking_uri:
                logger.warning(
                    "Production environment using localhost tracking URI. "
                    "Consider using a dedicated MLFlow server."
                )
            if "sqlite" in self.backend_store_uri:
                logger.warning(
                    "Production environment using SQLite backend. "
                    "Consider using PostgreSQL or MySQL for better concurrency."
                )

        elif self.environment == "testing":
            # Testing defaults - in-memory/local
            if self.tracking_uri == "http://localhost:5000":
                self.tracking_uri = "file:///tmp/mlflow-test"
            if self.backend_store_uri == "sqlite:///mlflow.db":
                self.backend_store_uri = "sqlite:///tmp/mlflow-test.db"

    def to_dict(self) -> dict[str, str | bool | dict[str, str] | None]:
        """
        Convert configuration to dictionary.

        Returns:
            Dictionary representation of configuration
        """
        return {
            "tracking_uri": self.tracking_uri,
            "experiment_name": self.experiment_name,
            "artifact_location": self.artifact_location,
            "backend_store_uri": self.backend_store_uri,
            "default_artifact_root": self.default_artifact_root,
            "registry_uri": self.registry_uri,
            "environment": self.environment,
            "enable_system_metrics": self.enable_system_metrics,
            "enable_apple_silicon_metrics": self.enable_apple_silicon_metrics,
            "log_models": self.log_models,
            "log_artifacts": self.log_artifacts,
            "log_params": self.log_params,
            "log_metrics": self.log_metrics,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, config_dict: dict[str, str | bool | dict[str, str] | None]) -> "MLFlowConfig":
        """
        Create configuration from dictionary.

        Args:
            config_dict: Configuration dictionary

        Returns:
            MLFlowConfig instance
        """
        # Filter out None values and unknown keys
        valid_keys = {
            "tracking_uri",
            "experiment_name",
            "artifact_location",
            "backend_store_uri",
            "default_artifact_root",
            "registry_uri",
            "environment",
            "enable_system_metrics",
            "enable_apple_silicon_metrics",
            "log_models",
            "log_artifacts",
            "log_params",
            "log_metrics",
            "tags",
        }

        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_keys and v is not None}

        return cls(**filtered_dict)

    @classmethod
    def from_environment(cls, environment: EnvironmentType = "development") -> "MLFlowConfig":
        """
        Create configuration for specific environment.

        Args:
            environment: Environment type

        Returns:
            MLFlowConfig instance configured for environment
        """
        config = cls(environment=environment)
        logger.info("Created MLFlow configuration for environment: %s", environment)
        return config

    def validate(self) -> bool:
        """
        Validate current configuration.

        Returns:
            True if configuration is valid

        Raises:
            MLFlowConfigError: If validation fails
        """
        self._validate_config()
        logger.info("MLFlow configuration validation passed")
        return True

    def get_connection_info(self) -> dict[str, str]:
        """
        Get connection information for MLFlow server.

        Returns:
            Dictionary with connection details
        """
        return {
            "tracking_uri": self.tracking_uri,
            "registry_uri": self.registry_uri or self.tracking_uri,
            "backend_store_uri": self.backend_store_uri,
            "environment": self.environment,
        }

    def update_tags(self, new_tags: dict[str, str]) -> None:
        """
        Update configuration tags.

        Args:
            new_tags: Tags to add or update
        """
        self.tags.update(new_tags)
        logger.debug("Updated tags: %s", new_tags)

    def get_artifact_path(self, relative_path: str = "") -> Path:
        """
        Get absolute artifact path.

        Args:
            relative_path: Relative path within artifact location

        Returns:
            Absolute path to artifact
        """
        base_path = Path(self.artifact_location)
        if relative_path:
            return base_path / relative_path
        return base_path

    def ensure_artifact_location(self) -> Path:
        """
        Ensure artifact location directory exists.

        Returns:
            Path to artifact location

        Raises:
            MLFlowConfigError: If directory creation fails
        """
        try:
            artifact_path = Path(self.artifact_location)
            artifact_path.mkdir(parents=True, exist_ok=True)
            logger.info("Artifact location ensured: %s", artifact_path)
            return artifact_path
        except Exception as e:
            raise MLFlowConfigError(
                f"Failed to create artifact location: {e}",
                environment=self.environment,
            ) from e


def get_default_config(environment: EnvironmentType = "development") -> MLFlowConfig:
    """
    Get default MLFlow configuration for environment.

    Args:
        environment: Environment type

    Returns:
        Default MLFlowConfig instance

    Example:
        >>> config = get_default_config("production")
        >>> print(config.tracking_uri)
    """
    return MLFlowConfig.from_environment(environment)


def load_config_from_file(config_path: Path, environment: EnvironmentType | None = None) -> MLFlowConfig:
    """
    Load MLFlow configuration from file.

    Args:
        config_path: Path to configuration file (YAML, JSON, or TOML)
        environment: Optional environment override

    Returns:
        MLFlowConfig instance

    Raises:
        MLFlowConfigError: If loading fails
    """
    try:
        from utils.config_manager import ConfigManager

        # Check if file exists
        if not config_path.exists():
            raise MLFlowConfigError(
                f"Configuration file not found: {config_path}",
                details={"config_path": str(config_path)},
            )

        config_manager = ConfigManager(config_path)
        mlflow_config = config_manager.get("mlflow", {})

        if not isinstance(mlflow_config, dict):
            raise MLFlowConfigError(
                "Invalid MLFlow configuration format in file",
                details={"config_path": str(config_path)},
            )

        # Override environment if specified
        if environment:
            mlflow_config["environment"] = environment

        return MLFlowConfig.from_dict(mlflow_config)

    except MLFlowConfigError:
        # Re-raise MLFlowConfigError without wrapping
        raise
    except Exception as e:
        raise MLFlowConfigError(
            f"Failed to load configuration from {config_path}",
            details={"error": str(e)},
        ) from e
