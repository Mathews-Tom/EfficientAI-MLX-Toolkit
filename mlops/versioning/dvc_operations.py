"""
High-level DVC operations for data versioning workflows.

This module provides convenient functions for common DVC workflows,
built on top of the DVCClient API.
"""

import logging
from pathlib import Path
from typing import Any

from mlops.client.dvc_client import DVCClient, DVCClientError
from mlops.config.dvc_config import DVCConfig

logger = logging.getLogger(__name__)


class DVCOperationsError(Exception):
    """Raised when DVC operations fail."""

    def __init__(
        self,
        message: str,
        operation: str | None = None,
        details: dict[str, str | int | float | bool] | None = None,
    ) -> None:
        super().__init__(message)
        self.operation = operation
        self.details = dict(details or {})


def initialize_project(
    project_namespace: str,
    storage_backend: str = "local",
    remote_url: str | None = None,
    repo_root: str | Path | None = None,
) -> DVCClient:
    """
    Initialize DVC for a project with remote storage.

    Args:
        project_namespace: Project namespace for isolation
        storage_backend: Storage backend type (s3, gcs, azure, local)
        remote_url: Optional remote URL (auto-generated if None)
        repo_root: Optional repository root

    Returns:
        Configured DVC client

    Raises:
        DVCOperationsError: If initialization fails

    Example:
        >>> client = initialize_project("lora-finetuning-mlx", "s3", "s3://my-bucket/data")
    """
    try:
        # Create configuration
        config = DVCConfig(
            storage_backend=storage_backend,  # type: ignore
            project_namespace=project_namespace,
            remote_url=remote_url or f"./.dvc-storage/{project_namespace}",
        )

        # Create client
        client = DVCClient(config=config, repo_root=repo_root)

        # Initialize DVC
        client.init()

        # Add remote
        client.remote_add()

        logger.info(
            "Initialized DVC for project: %s with backend: %s",
            project_namespace,
            storage_backend,
        )

        return client

    except DVCClientError as e:
        raise DVCOperationsError(
            f"Failed to initialize project: {e}",
            operation="initialize_project",
            details={"project": project_namespace, "backend": storage_backend},
        ) from e


def track_dataset(
    client: DVCClient,
    dataset_path: str | Path,
    push_to_remote: bool = True,
) -> dict[str, Any]:
    """
    Track a dataset file or directory with DVC.

    Args:
        client: DVC client instance
        dataset_path: Path to dataset file or directory
        push_to_remote: Whether to push to remote after tracking

    Returns:
        Dictionary with tracking information

    Raises:
        DVCOperationsError: If tracking fails

    Example:
        >>> info = track_dataset(client, "datasets/train.csv")
    """
    try:
        # Add to DVC tracking
        add_info = client.add(dataset_path)

        # Push to remote if requested
        if push_to_remote:
            push_info = client.push(targets=[Path(dataset_path)])
            add_info["push_status"] = push_info

        logger.info("Tracked dataset: %s", dataset_path)

        return add_info

    except DVCClientError as e:
        raise DVCOperationsError(
            f"Failed to track dataset: {e}",
            operation="track_dataset",
            details={"path": str(dataset_path)},
        ) from e


def sync_data(
    client: DVCClient,
    targets: list[str | Path] | None = None,
    direction: str = "pull",
    force: bool = False,
) -> dict[str, Any]:
    """
    Synchronize data with remote storage.

    Args:
        client: DVC client instance
        targets: Optional list of specific targets
        direction: Sync direction ('pull' or 'push')
        force: Force operation even if cache exists

    Returns:
        Dictionary with sync status

    Raises:
        DVCOperationsError: If sync fails

    Example:
        >>> sync_data(client, direction="pull")  # Download from remote
        >>> sync_data(client, direction="push")  # Upload to remote
    """
    try:
        if direction == "pull":
            result = client.pull(targets=targets, force=force)
        elif direction == "push":
            result = client.push(targets=targets)
        else:
            raise DVCOperationsError(
                f"Invalid sync direction: {direction}",
                operation="sync_data",
                details={"valid_directions": ["pull", "push"]},
            )

        logger.info("Synced data with remote (%s)", direction)

        return result

    except DVCClientError as e:
        raise DVCOperationsError(
            f"Failed to sync data: {e}",
            operation="sync_data",
            details={"direction": direction},
        ) from e


def get_data_status(
    client: DVCClient,
    targets: list[str | Path] | None = None,
    check_remote: bool = False,
) -> dict[str, Any]:
    """
    Get status of tracked data.

    Args:
        client: DVC client instance
        targets: Optional list of specific targets
        check_remote: Whether to check remote status

    Returns:
        Dictionary with status information

    Raises:
        DVCOperationsError: If status check fails

    Example:
        >>> status = get_data_status(client, check_remote=True)
        >>> print(status["is_clean"])
    """
    try:
        remote = client.config.remote_name if check_remote else None
        status = client.status(targets=targets, remote=remote)

        logger.debug("Retrieved data status")

        return status

    except DVCClientError as e:
        raise DVCOperationsError(
            f"Failed to get status: {e}",
            operation="get_data_status",
        ) from e


def restore_data_version(
    client: DVCClient,
    targets: list[str | Path] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """
    Restore data files to their tracked versions.

    Args:
        client: DVC client instance
        targets: Optional list of specific targets
        force: Force checkout even if files exist

    Returns:
        Dictionary with restore status

    Raises:
        DVCOperationsError: If restore fails

    Example:
        >>> restore_data_version(client, targets=["datasets/train.csv"])
    """
    try:
        result = client.checkout(targets=targets, force=force)

        logger.info("Restored data to tracked versions")

        return result

    except DVCClientError as e:
        raise DVCOperationsError(
            f"Failed to restore data: {e}",
            operation="restore_data_version",
        ) from e


def configure_remote_storage(
    client: DVCClient,
    remote_name: str,
    remote_url: str,
    set_as_default: bool = True,
) -> dict[str, Any]:
    """
    Configure a remote storage location.

    Args:
        client: DVC client instance
        remote_name: Name for the remote
        remote_url: URL for remote storage
        set_as_default: Set as default remote

    Returns:
        Dictionary with remote configuration

    Raises:
        DVCOperationsError: If configuration fails

    Example:
        >>> configure_remote_storage(
        ...     client,
        ...     "production",
        ...     "s3://prod-bucket/data"
        ... )
    """
    try:
        result = client.remote_add(
            name=remote_name,
            url=remote_url,
            default=set_as_default,
            force=True,
        )

        logger.info("Configured remote storage: %s", remote_name)

        return result

    except DVCClientError as e:
        raise DVCOperationsError(
            f"Failed to configure remote: {e}",
            operation="configure_remote_storage",
            details={"remote": remote_name},
        ) from e


def list_remotes(client: DVCClient) -> list[dict[str, str]]:
    """
    List all configured remote storage locations.

    Args:
        client: DVC client instance

    Returns:
        List of remote configurations

    Raises:
        DVCOperationsError: If listing fails

    Example:
        >>> remotes = list_remotes(client)
        >>> for remote in remotes:
        ...     print(f"{remote['name']}: {remote['url']}")
    """
    try:
        remotes = client.remote_list()

        logger.debug("Listed %d remotes", len(remotes))

        return remotes

    except DVCClientError as e:
        raise DVCOperationsError(
            f"Failed to list remotes: {e}",
            operation="list_remotes",
        ) from e


def create_workflow(
    project_namespace: str,
    dataset_paths: list[str | Path],
    storage_backend: str = "local",
    remote_url: str | None = None,
) -> dict[str, Any]:
    """
    Create a complete DVC workflow for a project.

    This function:
    1. Initializes DVC for the project
    2. Tracks all specified datasets
    3. Pushes data to remote storage

    Args:
        project_namespace: Project namespace
        dataset_paths: List of dataset paths to track
        storage_backend: Storage backend type
        remote_url: Optional remote URL

    Returns:
        Dictionary with workflow results

    Raises:
        DVCOperationsError: If workflow creation fails

    Example:
        >>> workflow = create_workflow(
        ...     "lora-finetuning-mlx",
        ...     ["datasets/train.csv", "datasets/test.csv"],
        ...     storage_backend="s3",
        ...     remote_url="s3://my-bucket/lora-data"
        ... )
    """
    try:
        # Initialize project
        client = initialize_project(
            project_namespace=project_namespace,
            storage_backend=storage_backend,
            remote_url=remote_url,
        )

        # Track all datasets
        tracked = []
        for dataset_path in dataset_paths:
            try:
                info = track_dataset(client, dataset_path, push_to_remote=True)
                tracked.append(info)
            except DVCOperationsError as e:
                logger.warning("Failed to track %s: %s", dataset_path, e)

        logger.info(
            "Created DVC workflow for %s with %d/%d datasets tracked",
            project_namespace,
            len(tracked),
            len(dataset_paths),
        )

        return {
            "project": project_namespace,
            "backend": storage_backend,
            "tracked_datasets": tracked,
            "total_datasets": len(dataset_paths),
            "success": len(tracked) > 0,
        }

    except DVCOperationsError:
        raise
    except Exception as e:
        raise DVCOperationsError(
            f"Failed to create workflow: {e}",
            operation="create_workflow",
            details={"project": project_namespace},
        ) from e
