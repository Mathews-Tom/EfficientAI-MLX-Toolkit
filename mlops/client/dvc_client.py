"""
DVC client API with data versioning methods.

This module provides a high-level DVC client interface for data versioning,
remote storage operations, and artifact tracking with project namespace isolation.
"""

import logging
import subprocess
from pathlib import Path
from typing import Any

from mlops.config.dvc_config import DVCConfig

logger = logging.getLogger(__name__)


class DVCClientError(Exception):
    """Raised when DVC client operations fail."""

    def __init__(
        self,
        message: str,
        operation: str | None = None,
        details: dict[str, str | int | float | bool] | None = None,
    ) -> None:
        super().__init__(message)
        self.operation = operation
        self.details = dict(details or {})


class DVCClient:
    """
    High-level DVC client for data versioning.

    This class provides a simplified interface to DVC operations,
    with built-in support for multiple storage backends and project
    namespace isolation.

    Attributes:
        config: DVC configuration
        repo_root: Repository root directory
        dvc_dir: DVC directory path (.dvc)
    """

    def __init__(
        self,
        config: DVCConfig | None = None,
        repo_root: str | Path | None = None,
    ) -> None:
        """
        Initialize DVC client.

        Args:
            config: DVC configuration (uses default if None)
            repo_root: Repository root directory (uses current dir if None)
        """
        from mlops.config.dvc_config import get_default_config

        self.config = config or get_default_config()
        self.repo_root = Path(repo_root) if repo_root else Path.cwd()
        self.dvc_dir = self.repo_root / ".dvc"

        # Mock mode for testing (no real DVC commands)
        self._mock_mode = False
        self._mock_responses: dict[str, Any] = {}

    def _run_dvc_command(
        self,
        command: list[str],
        check: bool = True,
        capture_output: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        """
        Run a DVC command.

        Args:
            command: DVC command arguments
            check: Whether to check return code
            capture_output: Whether to capture output

        Returns:
            Completed process

        Raises:
            DVCClientError: If command fails
        """
        if self._mock_mode:
            # Return mock response in testing
            cmd_key = " ".join(command)
            return self._mock_responses.get(cmd_key, subprocess.CompletedProcess(command, 0, "", ""))

        try:
            result = subprocess.run(
                ["dvc"] + command,
                cwd=self.repo_root,
                check=check,
                capture_output=capture_output,
                text=True,
            )
            return result

        except subprocess.CalledProcessError as e:
            raise DVCClientError(
                f"DVC command failed: {e.stderr or e.stdout}",
                operation=" ".join(command),
                details={"returncode": e.returncode},
            ) from e
        except FileNotFoundError as e:
            raise DVCClientError(
                "DVC is not installed or not in PATH",
                operation="run_command",
            ) from e

    def init(self, force: bool = False) -> None:
        """
        Initialize DVC in repository.

        Args:
            force: Force re-initialization

        Raises:
            DVCClientError: If initialization fails
        """
        try:
            command = ["init"]
            if force:
                command.append("--force")

            self._run_dvc_command(command)

            logger.info("DVC initialized in %s", self.repo_root)

        except DVCClientError as e:
            if not force and "already exists" in str(e):
                logger.debug("DVC already initialized")
                return
            raise

    def add(self, path: str | Path, recursive: bool = False) -> dict[str, Any]:
        """
        Add file or directory to DVC tracking.

        Args:
            path: Path to file or directory
            recursive: Whether to add directory recursively

        Returns:
            Dictionary with tracking information

        Raises:
            DVCClientError: If adding fails
        """
        try:
            path_obj = Path(path)
            if not path_obj.exists():
                raise DVCClientError(
                    f"Path not found: {path}",
                    operation="add",
                    details={"path": str(path)},
                )

            command = ["add", str(path)]
            if recursive and path_obj.is_dir():
                command.append("--recursive")

            result = self._run_dvc_command(command)

            # Generate .dvc file path
            dvc_file = path_obj.with_suffix(path_obj.suffix + ".dvc")

            logger.info("Added %s to DVC tracking", path)

            return {
                "path": str(path),
                "dvc_file": str(dvc_file),
                "success": True,
            }

        except DVCClientError:
            raise
        except Exception as e:
            raise DVCClientError(
                f"Failed to add {path}: {e}",
                operation="add",
                details={"path": str(path)},
            ) from e

    def remove(self, path: str | Path) -> dict[str, Any]:
        """
        Remove file or directory from DVC tracking.

        Args:
            path: Path to .dvc file or tracked path

        Returns:
            Dictionary with removal information

        Raises:
            DVCClientError: If removal fails
        """
        try:
            command = ["remove", str(path)]
            self._run_dvc_command(command)

            logger.info("Removed %s from DVC tracking", path)

            return {
                "path": str(path),
                "success": True,
            }

        except Exception as e:
            raise DVCClientError(
                f"Failed to remove {path}: {e}",
                operation="remove",
                details={"path": str(path)},
            ) from e

    def push(
        self,
        targets: list[str | Path] | None = None,
        remote: str | None = None,
        jobs: int | None = None,
    ) -> dict[str, Any]:
        """
        Push tracked data to remote storage.

        Args:
            targets: Optional list of specific targets to push
            remote: Optional remote name (uses default if None)
            jobs: Optional number of parallel jobs

        Returns:
            Dictionary with push status

        Raises:
            DVCClientError: If push fails
        """
        try:
            command = ["push"]

            if remote:
                command.extend(["--remote", remote])
            elif self.config.remote_name:
                command.extend(["--remote", self.config.remote_name])

            if jobs is not None:
                command.extend(["--jobs", str(jobs)])
            elif self.config.jobs:
                command.extend(["--jobs", str(self.config.jobs)])

            if targets:
                command.extend([str(t) for t in targets])

            result = self._run_dvc_command(command)

            logger.info("Pushed data to remote")

            return {
                "success": True,
                "remote": remote or self.config.remote_name,
                "targets": [str(t) for t in targets] if targets else ["all"],
            }

        except Exception as e:
            raise DVCClientError(
                f"Failed to push data: {e}",
                operation="push",
                details={"remote": remote or self.config.remote_name},
            ) from e

    def pull(
        self,
        targets: list[str | Path] | None = None,
        remote: str | None = None,
        jobs: int | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        """
        Pull tracked data from remote storage.

        Args:
            targets: Optional list of specific targets to pull
            remote: Optional remote name (uses default if None)
            jobs: Optional number of parallel jobs
            force: Force download even if cache exists

        Returns:
            Dictionary with pull status

        Raises:
            DVCClientError: If pull fails
        """
        try:
            command = ["pull"]

            if remote:
                command.extend(["--remote", remote])
            elif self.config.remote_name:
                command.extend(["--remote", self.config.remote_name])

            if jobs is not None:
                command.extend(["--jobs", str(jobs)])
            elif self.config.jobs:
                command.extend(["--jobs", str(self.config.jobs)])

            if force:
                command.append("--force")

            if targets:
                command.extend([str(t) for t in targets])

            result = self._run_dvc_command(command)

            logger.info("Pulled data from remote")

            return {
                "success": True,
                "remote": remote or self.config.remote_name,
                "targets": [str(t) for t in targets] if targets else ["all"],
            }

        except Exception as e:
            raise DVCClientError(
                f"Failed to pull data: {e}",
                operation="pull",
                details={"remote": remote or self.config.remote_name},
            ) from e

    def status(
        self,
        targets: list[str | Path] | None = None,
        remote: str | None = None,
    ) -> dict[str, Any]:
        """
        Get status of tracked data.

        Args:
            targets: Optional list of specific targets
            remote: Optional remote name for cloud status

        Returns:
            Dictionary with status information

        Raises:
            DVCClientError: If status check fails
        """
        try:
            command = ["status"]

            if remote:
                command.extend(["--remote", remote])
            elif self.config.remote_name:
                command.extend(["--remote", self.config.remote_name])

            if targets:
                command.extend([str(t) for t in targets])

            result = self._run_dvc_command(command, check=False)

            # Parse status output
            status_info = {
                "success": True,
                "output": result.stdout.strip(),
                "is_clean": result.returncode == 0,
            }

            logger.debug("DVC status: %s", status_info["output"])

            return status_info

        except Exception as e:
            raise DVCClientError(
                f"Failed to get status: {e}",
                operation="status",
            ) from e

    def checkout(
        self,
        targets: list[str | Path] | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        """
        Checkout data files from DVC cache.

        Args:
            targets: Optional list of specific targets
            force: Force checkout even if files exist

        Returns:
            Dictionary with checkout status

        Raises:
            DVCClientError: If checkout fails
        """
        try:
            command = ["checkout"]

            if force:
                command.append("--force")

            if targets:
                command.extend([str(t) for t in targets])

            result = self._run_dvc_command(command)

            logger.info("Checked out data files")

            return {
                "success": True,
                "targets": [str(t) for t in targets] if targets else ["all"],
            }

        except Exception as e:
            raise DVCClientError(
                f"Failed to checkout: {e}",
                operation="checkout",
            ) from e

    def remote_add(
        self,
        name: str | None = None,
        url: str | None = None,
        default: bool = True,
        force: bool = False,
    ) -> dict[str, Any]:
        """
        Add a remote storage location.

        Args:
            name: Remote name (uses config if None)
            url: Remote URL (uses config if None)
            default: Set as default remote
            force: Overwrite existing remote

        Returns:
            Dictionary with remote information

        Raises:
            DVCClientError: If adding remote fails
        """
        try:
            remote_name = name or self.config.remote_name
            remote_url = url or self.config.get_remote_path()

            command = ["remote", "add"]

            if force:
                command.append("--force")

            if default:
                command.append("--default")

            command.extend([remote_name, remote_url])

            # Add backend-specific configuration
            self._run_dvc_command(command)

            # Configure backend-specific settings
            self._configure_remote_backend(remote_name)

            logger.info("Added remote: %s -> %s", remote_name, remote_url)

            return {
                "name": remote_name,
                "url": remote_url,
                "backend": self.config.storage_backend,
                "success": True,
            }

        except Exception as e:
            raise DVCClientError(
                f"Failed to add remote: {e}",
                operation="remote_add",
                details={"name": remote_name, "url": remote_url},
            ) from e

    def _configure_remote_backend(self, remote_name: str) -> None:
        """
        Configure backend-specific remote settings.

        Args:
            remote_name: Remote name to configure
        """
        try:
            if self.config.storage_backend == "s3":
                if self.config.s3_region:
                    self._run_dvc_command(
                        ["remote", "modify", remote_name, "region", self.config.s3_region]
                    )
                if self.config.s3_profile:
                    self._run_dvc_command(
                        ["remote", "modify", remote_name, "profile", self.config.s3_profile]
                    )
                if self.config.s3_endpoint_url:
                    self._run_dvc_command(
                        ["remote", "modify", remote_name, "endpointurl", self.config.s3_endpoint_url]
                    )

            elif self.config.storage_backend == "gcs":
                if self.config.gcs_project:
                    self._run_dvc_command(
                        ["remote", "modify", remote_name, "projectname", self.config.gcs_project]
                    )

            elif self.config.storage_backend == "azure":
                if self.config.azure_account_name:
                    self._run_dvc_command(
                        [
                            "remote",
                            "modify",
                            remote_name,
                            "account_name",
                            self.config.azure_account_name,
                        ]
                    )

        except DVCClientError:
            # Configuration failures are non-fatal
            logger.warning("Failed to configure remote backend settings")

    def remote_list(self) -> list[dict[str, str]]:
        """
        List all configured remotes.

        Returns:
            List of remote configurations

        Raises:
            DVCClientError: If listing fails
        """
        try:
            result = self._run_dvc_command(["remote", "list"])

            remotes = []
            for line in result.stdout.strip().split("\n"):
                if line.strip():
                    parts = line.split("\t")
                    if len(parts) >= 2:
                        remotes.append({"name": parts[0].strip(), "url": parts[1].strip()})

            return remotes

        except Exception as e:
            raise DVCClientError(
                f"Failed to list remotes: {e}",
                operation="remote_list",
            ) from e

    def get_connection_info(self) -> dict[str, str | None]:
        """
        Get connection information for DVC remote.

        Returns:
            Dictionary with connection details
        """
        return self.config.get_connection_info()


def create_client(
    config: DVCConfig | None = None,
    repo_root: str | Path | None = None,
) -> DVCClient:
    """
    Create a new DVC client.

    Args:
        config: Optional DVC configuration
        repo_root: Optional repository root directory

    Returns:
        Initialized DVC client

    Example:
        >>> client = create_client()
        >>> client.add("datasets/train.csv")
        >>> client.push()
    """
    return DVCClient(config=config, repo_root=repo_root)
