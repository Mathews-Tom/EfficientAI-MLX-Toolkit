"""Project Workspace Manager

Manages project-specific workspaces for MLOps operations, providing
isolated directories for experiments, models, data versioning, and monitoring
across all toolkit projects.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


class WorkspaceError(Exception):
    """Raised when workspace operations fail"""

    def __init__(
        self,
        message: str,
        operation: str | None = None,
        workspace: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.operation = operation
        self.workspace = workspace
        self.details = dict(details or {})


@dataclass
class ProjectWorkspace:
    """Project workspace configuration and paths

    Provides isolated workspace for a single project with dedicated
    directories for MLFlow experiments, DVC data, BentoML models,
    and Evidently monitoring.

    Attributes:
        project_name: Project namespace identifier (e.g., "lora-finetuning-mlx")
        root_path: Root directory for all workspace data
        mlflow_experiment_id: MLFlow experiment ID (auto-created)
        mlflow_tracking_uri: MLFlow tracking server URI
        dvc_remote_path: DVC remote storage path
        bentoml_tag_prefix: BentoML model tag prefix
        metadata: Additional project metadata
        created_at: Workspace creation timestamp
        updated_at: Workspace last update timestamp
    """

    project_name: str
    root_path: Path
    mlflow_experiment_id: str | None = None
    mlflow_tracking_uri: str | None = None
    dvc_remote_path: str | None = None
    bentoml_tag_prefix: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def mlflow_path(self) -> Path:
        """MLFlow artifacts directory"""
        return self.root_path / "mlflow"

    @property
    def dvc_path(self) -> Path:
        """DVC data directory"""
        return self.root_path / "dvc"

    @property
    def monitoring_path(self) -> Path:
        """Monitoring data directory"""
        return self.root_path / "monitoring"

    @property
    def models_path(self) -> Path:
        """Models registry directory"""
        return self.root_path / "models"

    @property
    def outputs_path(self) -> Path:
        """Project outputs directory"""
        return self.root_path / "outputs"

    @property
    def metadata_file(self) -> Path:
        """Workspace metadata file path"""
        return self.root_path / "workspace.yaml"

    def ensure_directories(self) -> None:
        """Create all workspace directories if they don't exist"""
        directories = [
            self.root_path,
            self.mlflow_path,
            self.dvc_path,
            self.monitoring_path,
            self.models_path,
            self.outputs_path,
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug("Ensured directory exists: %s", directory)

    def save_metadata(self) -> None:
        """Save workspace metadata to YAML file"""
        self.updated_at = datetime.now(timezone.utc)

        metadata = {
            "project_name": self.project_name,
            "root_path": str(self.root_path),
            "mlflow_experiment_id": self.mlflow_experiment_id,
            "mlflow_tracking_uri": self.mlflow_tracking_uri,
            "dvc_remote_path": self.dvc_remote_path,
            "bentoml_tag_prefix": self.bentoml_tag_prefix,
            "metadata": dict(self.metadata),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

        with open(self.metadata_file, "w") as f:
            yaml.safe_dump(metadata, f, default_flow_style=False, sort_keys=False)

        logger.debug("Saved workspace metadata: %s", self.metadata_file)

    @classmethod
    def load_metadata(cls, root_path: Path) -> ProjectWorkspace:
        """Load workspace from metadata file

        Args:
            root_path: Root directory containing workspace.yaml

        Returns:
            ProjectWorkspace instance loaded from metadata

        Raises:
            WorkspaceError: If metadata file is missing or invalid
        """
        metadata_file = root_path / "workspace.yaml"

        if not metadata_file.exists():
            raise WorkspaceError(
                f"Workspace metadata not found: {metadata_file}",
                operation="load_metadata",
                workspace=str(root_path),
            )

        try:
            with open(metadata_file) as f:
                data = yaml.safe_load(f)

            # Parse timestamps
            created_at = datetime.fromisoformat(data["created_at"])
            updated_at = datetime.fromisoformat(data["updated_at"])

            return cls(
                project_name=data["project_name"],
                root_path=Path(data["root_path"]),
                mlflow_experiment_id=data.get("mlflow_experiment_id"),
                mlflow_tracking_uri=data.get("mlflow_tracking_uri"),
                dvc_remote_path=data.get("dvc_remote_path"),
                bentoml_tag_prefix=data.get("bentoml_tag_prefix"),
                metadata=dict(data.get("metadata", {})),
                created_at=created_at,
                updated_at=updated_at,
            )

        except Exception as e:
            raise WorkspaceError(
                f"Failed to load workspace metadata: {e}",
                operation="load_metadata",
                workspace=str(root_path),
                details={"error": str(e)},
            ) from e

    def to_dict(self) -> dict[str, Any]:
        """Convert workspace to dictionary

        Returns:
            Dictionary representation of workspace
        """
        return {
            "project_name": self.project_name,
            "root_path": str(self.root_path),
            "mlflow_path": str(self.mlflow_path),
            "dvc_path": str(self.dvc_path),
            "monitoring_path": str(self.monitoring_path),
            "models_path": str(self.models_path),
            "outputs_path": str(self.outputs_path),
            "mlflow_experiment_id": self.mlflow_experiment_id,
            "mlflow_tracking_uri": self.mlflow_tracking_uri,
            "dvc_remote_path": self.dvc_remote_path,
            "bentoml_tag_prefix": self.bentoml_tag_prefix,
            "metadata": dict(self.metadata),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class WorkspaceManager:
    """Manager for project workspaces

    Provides centralized management of project workspaces across the toolkit,
    enabling isolated MLOps operations per project while maintaining a unified
    interface for cross-project analytics and monitoring.

    Attributes:
        base_path: Base directory for all workspaces
        workspaces: Cache of loaded workspace instances
    """

    def __init__(self, base_path: str | Path | None = None, repo_root: str | Path | None = None) -> None:
        """Initialize workspace manager

        Args:
            base_path: Base directory for workspaces (default: repo_root/mlops/workspace)
            repo_root: Repository root directory (default: current directory)
        """
        if repo_root is None:
            repo_root = Path.cwd()
        else:
            repo_root = Path(repo_root)

        if base_path is None:
            self.base_path = repo_root / "mlops" / "workspace"
        else:
            self.base_path = Path(base_path)

        self.base_path.mkdir(parents=True, exist_ok=True)
        self.workspaces: dict[str, ProjectWorkspace] = {}

        logger.info("WorkspaceManager initialized: %s", self.base_path)

    def create_workspace(
        self,
        project_name: str,
        mlflow_experiment_id: str | None = None,
        mlflow_tracking_uri: str | None = None,
        dvc_remote_path: str | None = None,
        bentoml_tag_prefix: str | None = None,
        metadata: dict[str, Any] | None = None,
        force: bool = False,
    ) -> ProjectWorkspace:
        """Create a new project workspace

        Creates a workspace with isolated directories for MLFlow, DVC,
        monitoring, and model storage. Automatically configures integration
        points for all MLOps components.

        Args:
            project_name: Project namespace identifier
            mlflow_experiment_id: Optional MLFlow experiment ID
            mlflow_tracking_uri: Optional MLFlow tracking URI
            dvc_remote_path: Optional DVC remote path
            bentoml_tag_prefix: Optional BentoML tag prefix
            metadata: Optional project metadata
            force: Overwrite existing workspace if True

        Returns:
            Created ProjectWorkspace instance

        Raises:
            WorkspaceError: If workspace already exists (unless force=True)

        Example:
            >>> mgr = WorkspaceManager()
            >>> workspace = mgr.create_workspace("lora-finetuning-mlx")
            >>> print(workspace.mlflow_path)
        """
        workspace_path = self.base_path / project_name

        # Check if workspace exists
        if workspace_path.exists() and not force:
            raise WorkspaceError(
                f"Workspace already exists: {project_name}",
                operation="create_workspace",
                workspace=project_name,
                details={"path": str(workspace_path)},
            )

        # Create workspace instance
        workspace = ProjectWorkspace(
            project_name=project_name,
            root_path=workspace_path,
            mlflow_experiment_id=mlflow_experiment_id,
            mlflow_tracking_uri=mlflow_tracking_uri or "file:///tmp/mlflow",
            dvc_remote_path=dvc_remote_path or str(workspace_path / "dvc" / "remote"),
            bentoml_tag_prefix=bentoml_tag_prefix or f"{project_name}:",
            metadata=dict(metadata or {}),
        )

        # Create directories and save metadata
        try:
            workspace.ensure_directories()
            workspace.save_metadata()

            # Cache workspace
            self.workspaces[project_name] = workspace

            logger.info("Created workspace: %s at %s", project_name, workspace_path)
            return workspace

        except Exception as e:
            raise WorkspaceError(
                f"Failed to create workspace: {e}",
                operation="create_workspace",
                workspace=project_name,
                details={"error": str(e)},
            ) from e

    def get_workspace(self, project_name: str) -> ProjectWorkspace:
        """Get existing workspace for project

        Args:
            project_name: Project namespace identifier

        Returns:
            ProjectWorkspace instance

        Raises:
            WorkspaceError: If workspace does not exist

        Example:
            >>> mgr = WorkspaceManager()
            >>> workspace = mgr.get_workspace("lora-finetuning-mlx")
        """
        # Check cache first
        if project_name in self.workspaces:
            return self.workspaces[project_name]

        # Load from disk
        workspace_path = self.base_path / project_name

        if not workspace_path.exists():
            raise WorkspaceError(
                f"Workspace does not exist: {project_name}",
                operation="get_workspace",
                workspace=project_name,
                details={"path": str(workspace_path)},
            )

        try:
            workspace = ProjectWorkspace.load_metadata(workspace_path)
            self.workspaces[project_name] = workspace
            logger.debug("Loaded workspace: %s", project_name)
            return workspace

        except Exception as e:
            raise WorkspaceError(
                f"Failed to load workspace: {e}",
                operation="get_workspace",
                workspace=project_name,
                details={"error": str(e)},
            ) from e

    def get_or_create_workspace(
        self,
        project_name: str,
        **kwargs: Any,
    ) -> ProjectWorkspace:
        """Get existing workspace or create if it doesn't exist

        Args:
            project_name: Project namespace identifier
            **kwargs: Additional arguments for create_workspace

        Returns:
            ProjectWorkspace instance

        Example:
            >>> mgr = WorkspaceManager()
            >>> workspace = mgr.get_or_create_workspace("lora-finetuning-mlx")
        """
        try:
            return self.get_workspace(project_name)
        except WorkspaceError:
            return self.create_workspace(project_name, **kwargs)

    def list_workspaces(self) -> list[ProjectWorkspace]:
        """List all workspaces

        Returns:
            List of all ProjectWorkspace instances

        Example:
            >>> mgr = WorkspaceManager()
            >>> workspaces = mgr.list_workspaces()
            >>> for ws in workspaces:
            ...     print(f"{ws.project_name}: {ws.mlflow_experiment_id}")
        """
        workspaces = []

        if not self.base_path.exists():
            return workspaces

        for workspace_dir in self.base_path.iterdir():
            if not workspace_dir.is_dir():
                continue

            metadata_file = workspace_dir / "workspace.yaml"
            if not metadata_file.exists():
                continue

            try:
                workspace = self.get_workspace(workspace_dir.name)
                workspaces.append(workspace)
            except WorkspaceError as e:
                logger.warning("Failed to load workspace %s: %s", workspace_dir.name, e)
                continue

        return workspaces

    def delete_workspace(self, project_name: str, force: bool = False) -> None:
        """Delete a workspace

        Args:
            project_name: Project namespace identifier
            force: Delete without confirmation (USE WITH CAUTION)

        Raises:
            WorkspaceError: If workspace does not exist or deletion fails

        Warning:
            This permanently deletes all workspace data including experiments,
            models, and monitoring data. Use with extreme caution.
        """
        if not force:
            raise WorkspaceError(
                "Workspace deletion requires force=True for safety",
                operation="delete_workspace",
                workspace=project_name,
            )

        workspace_path = self.base_path / project_name

        if not workspace_path.exists():
            raise WorkspaceError(
                f"Workspace does not exist: {project_name}",
                operation="delete_workspace",
                workspace=project_name,
            )

        try:
            import shutil
            shutil.rmtree(workspace_path)

            # Remove from cache
            self.workspaces.pop(project_name, None)

            logger.warning("Deleted workspace: %s", project_name)

        except Exception as e:
            raise WorkspaceError(
                f"Failed to delete workspace: {e}",
                operation="delete_workspace",
                workspace=project_name,
                details={"error": str(e)},
            ) from e

    def update_workspace_metadata(
        self,
        project_name: str,
        mlflow_experiment_id: str | None = None,
        mlflow_tracking_uri: str | None = None,
        dvc_remote_path: str | None = None,
        bentoml_tag_prefix: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ProjectWorkspace:
        """Update workspace metadata

        Args:
            project_name: Project namespace identifier
            mlflow_experiment_id: Optional MLFlow experiment ID
            mlflow_tracking_uri: Optional MLFlow tracking URI
            dvc_remote_path: Optional DVC remote path
            bentoml_tag_prefix: Optional BentoML tag prefix
            metadata: Optional project metadata to merge

        Returns:
            Updated ProjectWorkspace instance

        Raises:
            WorkspaceError: If workspace does not exist or update fails
        """
        workspace = self.get_workspace(project_name)

        # Update fields
        if mlflow_experiment_id is not None:
            workspace.mlflow_experiment_id = mlflow_experiment_id
        if mlflow_tracking_uri is not None:
            workspace.mlflow_tracking_uri = mlflow_tracking_uri
        if dvc_remote_path is not None:
            workspace.dvc_remote_path = dvc_remote_path
        if bentoml_tag_prefix is not None:
            workspace.bentoml_tag_prefix = bentoml_tag_prefix
        if metadata is not None:
            workspace.metadata.update(metadata)

        # Save updated metadata
        try:
            workspace.save_metadata()
            logger.info("Updated workspace metadata: %s", project_name)
            return workspace

        except Exception as e:
            raise WorkspaceError(
                f"Failed to update workspace metadata: {e}",
                operation="update_workspace_metadata",
                workspace=project_name,
                details={"error": str(e)},
            ) from e

    def get_workspace_status(self, project_name: str) -> dict[str, Any]:
        """Get workspace status and statistics

        Args:
            project_name: Project namespace identifier

        Returns:
            Dictionary with workspace status information

        Raises:
            WorkspaceError: If workspace does not exist
        """
        workspace = self.get_workspace(project_name)

        status = workspace.to_dict()

        # Add directory statistics
        status["directory_stats"] = {
            "mlflow_exists": workspace.mlflow_path.exists(),
            "dvc_exists": workspace.dvc_path.exists(),
            "monitoring_exists": workspace.monitoring_path.exists(),
            "models_exists": workspace.models_path.exists(),
            "outputs_exists": workspace.outputs_path.exists(),
        }

        # Count files in each directory
        for path_name, path in [
            ("mlflow", workspace.mlflow_path),
            ("dvc", workspace.dvc_path),
            ("monitoring", workspace.monitoring_path),
            ("models", workspace.models_path),
            ("outputs", workspace.outputs_path),
        ]:
            if path.exists():
                file_count = sum(1 for _ in path.rglob("*") if _.is_file())
                status["directory_stats"][f"{path_name}_files"] = file_count

        return status

    def get_all_workspaces_status(self) -> list[dict[str, Any]]:
        """Get status for all workspaces

        Returns:
            List of status dictionaries for all workspaces
        """
        workspaces = self.list_workspaces()
        return [self.get_workspace_status(ws.project_name) for ws in workspaces]
