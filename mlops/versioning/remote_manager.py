"""
Remote storage path management with project namespace isolation.

This module provides utilities for managing project-specific remote paths
across different storage backends (S3, GCS, Azure, Local).
"""

import logging
from pathlib import Path
from typing import Any

from mlops.config.dvc_config import DVCConfig, StorageBackend

logger = logging.getLogger(__name__)


class RemoteManagerError(Exception):
    """Raised when remote management operations fail."""

    def __init__(
        self,
        message: str,
        backend: str | None = None,
        details: dict[str, str | int | float | bool] | None = None,
    ) -> None:
        super().__init__(message)
        self.backend = backend
        self.details = dict(details or {})


class RemotePathManager:
    """
    Manage remote storage paths with project namespace isolation.

    This class provides utilities for constructing, validating, and managing
    project-specific remote storage paths across different backends.

    Attributes:
        config: DVC configuration
        backend: Storage backend type
        project_namespace: Project namespace for isolation
    """

    def __init__(self, config: DVCConfig) -> None:
        """
        Initialize remote path manager.

        Args:
            config: DVC configuration
        """
        self.config = config
        self.backend = config.storage_backend
        self.project_namespace = config.project_namespace

    def get_project_remote_path(self, subdirectory: str = "") -> str:
        """
        Get full remote path for project with optional subdirectory.

        Args:
            subdirectory: Optional subdirectory within project namespace

        Returns:
            Full remote path

        Example:
            >>> manager = RemotePathManager(config)
            >>> manager.get_project_remote_path("datasets")
            's3://bucket/my-project/datasets'
        """
        return self.config.get_remote_path(subdirectory)

    def get_dataset_path(self, dataset_name: str) -> str:
        """
        Get remote path for a specific dataset.

        Args:
            dataset_name: Dataset name

        Returns:
            Remote path for dataset

        Example:
            >>> manager.get_dataset_path("train.csv")
            's3://bucket/my-project/datasets/train.csv'
        """
        return self.get_project_remote_path(f"datasets/{dataset_name}")

    def get_model_path(self, model_name: str, version: str = "latest") -> str:
        """
        Get remote path for a model artifact.

        Args:
            model_name: Model name
            version: Model version (default: "latest")

        Returns:
            Remote path for model

        Example:
            >>> manager.get_model_path("lora-adapter", "v1.0")
            's3://bucket/my-project/models/lora-adapter/v1.0'
        """
        return self.get_project_remote_path(f"models/{model_name}/{version}")

    def get_checkpoint_path(self, checkpoint_id: str) -> str:
        """
        Get remote path for a training checkpoint.

        Args:
            checkpoint_id: Checkpoint identifier

        Returns:
            Remote path for checkpoint

        Example:
            >>> manager.get_checkpoint_path("epoch_10")
            's3://bucket/my-project/checkpoints/epoch_10'
        """
        return self.get_project_remote_path(f"checkpoints/{checkpoint_id}")

    def get_artifact_path(self, artifact_type: str, artifact_name: str) -> str:
        """
        Get remote path for a generic artifact.

        Args:
            artifact_type: Artifact type (e.g., 'logs', 'metrics', 'plots')
            artifact_name: Artifact name

        Returns:
            Remote path for artifact

        Example:
            >>> manager.get_artifact_path("metrics", "training_metrics.json")
            's3://bucket/my-project/artifacts/metrics/training_metrics.json'
        """
        return self.get_project_remote_path(f"artifacts/{artifact_type}/{artifact_name}")

    def validate_remote_path(self, path: str) -> bool:
        """
        Validate a remote path for the configured backend.

        Args:
            path: Remote path to validate

        Returns:
            True if path is valid for backend

        Raises:
            RemoteManagerError: If validation fails
        """
        if self.backend == "s3":
            if not path.startswith("s3://"):
                raise RemoteManagerError(
                    f"Invalid S3 path: {path}",
                    backend="s3",
                    details={"expected_prefix": "s3://"},
                )
        elif self.backend == "gcs":
            if not path.startswith("gs://"):
                raise RemoteManagerError(
                    f"Invalid GCS path: {path}",
                    backend="gcs",
                    details={"expected_prefix": "gs://"},
                )
        elif self.backend == "azure":
            if not path.startswith("azure://"):
                raise RemoteManagerError(
                    f"Invalid Azure path: {path}",
                    backend="azure",
                    details={"expected_prefix": "azure://"},
                )
        elif self.backend == "local":
            # Local paths can be relative or absolute
            pass

        return True

    def parse_remote_path(self, path: str) -> dict[str, str]:
        """
        Parse a remote path into components.

        Args:
            path: Remote path to parse

        Returns:
            Dictionary with path components

        Example:
            >>> manager.parse_remote_path("s3://bucket/project/datasets/train.csv")
            {
                'backend': 's3',
                'bucket': 'bucket',
                'project': 'project',
                'subdirectory': 'datasets',
                'filename': 'train.csv'
            }
        """
        components: dict[str, str] = {"backend": self.backend}

        if self.backend in ["s3", "gcs", "azure"]:
            # Cloud storage path parsing
            if self.backend == "s3":
                prefix = "s3://"
            elif self.backend == "gcs":
                prefix = "gs://"
            else:  # azure
                prefix = "azure://"

            if path.startswith(prefix):
                path_without_prefix = path[len(prefix) :]
                parts = path_without_prefix.split("/")

                if parts:
                    components["bucket"] = parts[0]
                if len(parts) > 1:
                    components["project"] = parts[1]
                if len(parts) > 2:
                    components["subdirectory"] = "/".join(parts[2:-1]) if len(parts) > 3 else parts[2]
                if len(parts) > 3:
                    components["filename"] = parts[-1]

        elif self.backend == "local":
            # Local path parsing
            path_obj = Path(path)
            components["directory"] = str(path_obj.parent)
            components["filename"] = path_obj.name

        return components

    def list_project_directories(self) -> list[str]:
        """
        Get list of standard project directories.

        Returns:
            List of standard directory paths

        Example:
            >>> manager.list_project_directories()
            ['datasets', 'models', 'checkpoints', 'artifacts']
        """
        directories = [
            "datasets",
            "models",
            "checkpoints",
            "artifacts",
            "logs",
            "metrics",
        ]

        return [self.get_project_remote_path(d) for d in directories]

    def get_backend_info(self) -> dict[str, str | None]:
        """
        Get information about the storage backend.

        Returns:
            Dictionary with backend information
        """
        info: dict[str, str | None] = {
            "backend": self.backend,
            "project_namespace": self.project_namespace,
            "base_remote_url": self.config.remote_url,
            "remote_name": self.config.remote_name,
        }

        if self.backend == "s3":
            info.update(
                {
                    "region": self.config.s3_region,
                    "profile": self.config.s3_profile,
                    "endpoint_url": self.config.s3_endpoint_url,
                }
            )
        elif self.backend == "gcs":
            info["gcs_project"] = self.config.gcs_project
        elif self.backend == "azure":
            info.update(
                {
                    "account_name": self.config.azure_account_name,
                    "container_name": self.config.azure_container_name,
                }
            )

        return info


class MultiProjectRemoteManager:
    """
    Manage remote storage for multiple projects.

    This class coordinates remote storage across multiple project namespaces,
    ensuring proper isolation and organization.

    Attributes:
        backend: Storage backend type
        base_remote_url: Base remote URL for all projects
        projects: Dictionary of project managers
    """

    def __init__(self, backend: StorageBackend, base_remote_url: str) -> None:
        """
        Initialize multi-project remote manager.

        Args:
            backend: Storage backend type
            base_remote_url: Base remote URL
        """
        self.backend = backend
        self.base_remote_url = base_remote_url
        self.projects: dict[str, RemotePathManager] = {}

    def register_project(self, project_namespace: str) -> RemotePathManager:
        """
        Register a new project and get its remote manager.

        Args:
            project_namespace: Project namespace

        Returns:
            Remote path manager for the project

        Example:
            >>> multi_manager = MultiProjectRemoteManager("s3", "s3://bucket/data")
            >>> project_manager = multi_manager.register_project("lora-finetuning-mlx")
        """
        config = DVCConfig(
            storage_backend=self.backend,
            remote_url=self.base_remote_url,
            project_namespace=project_namespace,
        )

        manager = RemotePathManager(config)
        self.projects[project_namespace] = manager

        logger.info("Registered project: %s", project_namespace)

        return manager

    def get_project_manager(self, project_namespace: str) -> RemotePathManager | None:
        """
        Get remote manager for a project.

        Args:
            project_namespace: Project namespace

        Returns:
            Remote path manager or None if not registered
        """
        return self.projects.get(project_namespace)

    def list_projects(self) -> list[str]:
        """
        List all registered projects.

        Returns:
            List of project namespaces
        """
        return list(self.projects.keys())

    def get_all_project_paths(self) -> dict[str, str]:
        """
        Get remote paths for all registered projects.

        Returns:
            Dictionary mapping project names to remote paths
        """
        return {
            project: manager.get_project_remote_path()
            for project, manager in self.projects.items()
        }


def create_remote_manager(config: DVCConfig) -> RemotePathManager:
    """
    Create a remote path manager from configuration.

    Args:
        config: DVC configuration

    Returns:
        Initialized remote path manager

    Example:
        >>> config = DVCConfig(storage_backend="s3", ...)
        >>> manager = create_remote_manager(config)
    """
    return RemotePathManager(config)


def create_multi_project_manager(
    backend: StorageBackend,
    base_remote_url: str,
) -> MultiProjectRemoteManager:
    """
    Create a multi-project remote manager.

    Args:
        backend: Storage backend type
        base_remote_url: Base remote URL

    Returns:
        Initialized multi-project manager

    Example:
        >>> manager = create_multi_project_manager("s3", "s3://bucket/data")
        >>> manager.register_project("project-1")
        >>> manager.register_project("project-2")
    """
    return MultiProjectRemoteManager(backend, base_remote_url)
