"""Tests for Project Workspace Manager"""

from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import yaml

from mlops.workspace.manager import (
    ProjectWorkspace,
    WorkspaceError,
    WorkspaceManager,
)


@pytest.fixture
def temp_base_path():
    """Create temporary base path for workspaces"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def workspace_manager(temp_base_path):
    """Create workspace manager with temporary base path"""
    return WorkspaceManager(base_path=temp_base_path)


@pytest.fixture
def sample_workspace(temp_base_path):
    """Create sample workspace for testing"""
    workspace_path = temp_base_path / "test-project"
    workspace_path.mkdir(parents=True)

    workspace = ProjectWorkspace(
        project_name="test-project",
        root_path=workspace_path,
        mlflow_experiment_id="exp-123",
        mlflow_tracking_uri="http://localhost:5000",
        dvc_remote_path="/tmp/dvc-remote",
        bentoml_tag_prefix="test-project:",
        metadata={"version": "1.0", "description": "Test project"},
    )

    workspace.ensure_directories()
    workspace.save_metadata()

    return workspace


class TestProjectWorkspace:
    """Test ProjectWorkspace dataclass"""

    def test_workspace_initialization(self, temp_base_path):
        """Test workspace initialization with all fields"""
        workspace_path = temp_base_path / "test-project"

        workspace = ProjectWorkspace(
            project_name="test-project",
            root_path=workspace_path,
            mlflow_experiment_id="exp-123",
            mlflow_tracking_uri="http://localhost:5000",
            dvc_remote_path="/tmp/dvc-remote",
            bentoml_tag_prefix="test:",
            metadata={"key": "value"},
        )

        assert workspace.project_name == "test-project"
        assert workspace.root_path == workspace_path
        assert workspace.mlflow_experiment_id == "exp-123"
        assert workspace.mlflow_tracking_uri == "http://localhost:5000"
        assert workspace.dvc_remote_path == "/tmp/dvc-remote"
        assert workspace.bentoml_tag_prefix == "test:"
        assert workspace.metadata == {"key": "value"}
        assert isinstance(workspace.created_at, datetime)
        assert isinstance(workspace.updated_at, datetime)

    def test_workspace_paths(self, temp_base_path):
        """Test workspace path properties"""
        workspace_path = temp_base_path / "test-project"
        workspace = ProjectWorkspace(
            project_name="test-project",
            root_path=workspace_path,
        )

        assert workspace.mlflow_path == workspace_path / "mlflow"
        assert workspace.dvc_path == workspace_path / "dvc"
        assert workspace.monitoring_path == workspace_path / "monitoring"
        assert workspace.models_path == workspace_path / "models"
        assert workspace.outputs_path == workspace_path / "outputs"
        assert workspace.metadata_file == workspace_path / "workspace.yaml"

    def test_ensure_directories(self, temp_base_path):
        """Test directory creation"""
        workspace_path = temp_base_path / "test-project"
        workspace = ProjectWorkspace(
            project_name="test-project",
            root_path=workspace_path,
        )

        workspace.ensure_directories()

        assert workspace.root_path.exists()
        assert workspace.mlflow_path.exists()
        assert workspace.dvc_path.exists()
        assert workspace.monitoring_path.exists()
        assert workspace.models_path.exists()
        assert workspace.outputs_path.exists()

    def test_save_metadata(self, temp_base_path):
        """Test metadata saving to YAML"""
        workspace_path = temp_base_path / "test-project"
        workspace = ProjectWorkspace(
            project_name="test-project",
            root_path=workspace_path,
            mlflow_experiment_id="exp-123",
            metadata={"key": "value"},
        )

        workspace.ensure_directories()
        workspace.save_metadata()

        assert workspace.metadata_file.exists()

        with open(workspace.metadata_file) as f:
            data = yaml.safe_load(f)

        assert data["project_name"] == "test-project"
        assert data["mlflow_experiment_id"] == "exp-123"
        assert data["metadata"] == {"key": "value"}
        assert "created_at" in data
        assert "updated_at" in data

    def test_load_metadata(self, sample_workspace):
        """Test loading workspace from metadata file"""
        loaded_workspace = ProjectWorkspace.load_metadata(sample_workspace.root_path)

        assert loaded_workspace.project_name == sample_workspace.project_name
        assert loaded_workspace.mlflow_experiment_id == sample_workspace.mlflow_experiment_id
        assert loaded_workspace.mlflow_tracking_uri == sample_workspace.mlflow_tracking_uri
        assert loaded_workspace.dvc_remote_path == sample_workspace.dvc_remote_path
        assert loaded_workspace.bentoml_tag_prefix == sample_workspace.bentoml_tag_prefix
        assert loaded_workspace.metadata == sample_workspace.metadata

    def test_load_metadata_missing_file(self, temp_base_path):
        """Test loading metadata when file is missing"""
        workspace_path = temp_base_path / "nonexistent"

        with pytest.raises(WorkspaceError) as exc_info:
            ProjectWorkspace.load_metadata(workspace_path)

        assert "Workspace metadata not found" in str(exc_info.value)
        assert exc_info.value.operation == "load_metadata"

    def test_load_metadata_invalid_yaml(self, temp_base_path):
        """Test loading metadata with invalid YAML"""
        workspace_path = temp_base_path / "invalid"
        workspace_path.mkdir()

        metadata_file = workspace_path / "workspace.yaml"
        with open(metadata_file, "w") as f:
            f.write("invalid: yaml: content: [")

        with pytest.raises(WorkspaceError) as exc_info:
            ProjectWorkspace.load_metadata(workspace_path)

        assert "Failed to load workspace metadata" in str(exc_info.value)

    def test_to_dict(self, sample_workspace):
        """Test converting workspace to dictionary"""
        workspace_dict = sample_workspace.to_dict()

        assert workspace_dict["project_name"] == "test-project"
        assert workspace_dict["mlflow_experiment_id"] == "exp-123"
        assert "mlflow_path" in workspace_dict
        assert "dvc_path" in workspace_dict
        assert "monitoring_path" in workspace_dict
        assert "models_path" in workspace_dict
        assert "outputs_path" in workspace_dict
        assert "created_at" in workspace_dict
        assert "updated_at" in workspace_dict


class TestWorkspaceManager:
    """Test WorkspaceManager operations"""

    def test_manager_initialization(self, temp_base_path):
        """Test workspace manager initialization"""
        manager = WorkspaceManager(base_path=temp_base_path)

        assert manager.base_path == temp_base_path
        assert manager.base_path.exists()
        assert isinstance(manager.workspaces, dict)

    def test_manager_initialization_default_path(self):
        """Test workspace manager with default path"""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = WorkspaceManager(repo_root=tmpdir)
            expected_path = Path(tmpdir) / "mlops" / "workspace"
            assert manager.base_path == expected_path
            assert manager.base_path.exists()

    def test_create_workspace(self, workspace_manager):
        """Test creating a new workspace"""
        workspace = workspace_manager.create_workspace(
            project_name="lora-finetuning-mlx",
            mlflow_experiment_id="exp-456",
            mlflow_tracking_uri="http://localhost:5000",
            metadata={"description": "LoRA fine-tuning"},
        )

        assert workspace.project_name == "lora-finetuning-mlx"
        assert workspace.mlflow_experiment_id == "exp-456"
        assert workspace.root_path.exists()
        assert workspace.mlflow_path.exists()
        assert workspace.dvc_path.exists()
        assert workspace.metadata_file.exists()

        # Check that workspace is cached
        assert "lora-finetuning-mlx" in workspace_manager.workspaces

    def test_create_workspace_already_exists(self, workspace_manager, sample_workspace):
        """Test creating workspace that already exists"""
        with pytest.raises(WorkspaceError) as exc_info:
            workspace_manager.create_workspace("test-project")

        assert "Workspace already exists" in str(exc_info.value)
        assert exc_info.value.operation == "create_workspace"

    def test_create_workspace_force_overwrite(self, workspace_manager, sample_workspace):
        """Test creating workspace with force overwrite"""
        workspace = workspace_manager.create_workspace(
            project_name="test-project",
            mlflow_experiment_id="exp-new",
            force=True,
        )

        assert workspace.project_name == "test-project"
        assert workspace.mlflow_experiment_id == "exp-new"

    def test_get_workspace(self, workspace_manager, sample_workspace):
        """Test getting existing workspace"""
        workspace = workspace_manager.get_workspace("test-project")

        assert workspace.project_name == "test-project"
        assert workspace.mlflow_experiment_id == "exp-123"
        assert workspace.metadata == {"version": "1.0", "description": "Test project"}

    def test_get_workspace_from_cache(self, workspace_manager):
        """Test getting workspace from cache"""
        # Create and cache workspace
        workspace_manager.create_workspace("cached-project")

        # Get workspace (should come from cache)
        workspace = workspace_manager.get_workspace("cached-project")

        assert workspace.project_name == "cached-project"
        assert "cached-project" in workspace_manager.workspaces

    def test_get_workspace_not_found(self, workspace_manager):
        """Test getting non-existent workspace"""
        with pytest.raises(WorkspaceError) as exc_info:
            workspace_manager.get_workspace("nonexistent")

        assert "Workspace does not exist" in str(exc_info.value)
        assert exc_info.value.operation == "get_workspace"

    def test_get_or_create_workspace_existing(self, workspace_manager, sample_workspace):
        """Test get_or_create for existing workspace"""
        workspace = workspace_manager.get_or_create_workspace("test-project")

        assert workspace.project_name == "test-project"
        assert workspace.mlflow_experiment_id == "exp-123"

    def test_get_or_create_workspace_new(self, workspace_manager):
        """Test get_or_create for new workspace"""
        workspace = workspace_manager.get_or_create_workspace(
            "new-project",
            mlflow_experiment_id="exp-789",
        )

        assert workspace.project_name == "new-project"
        assert workspace.mlflow_experiment_id == "exp-789"
        assert workspace.root_path.exists()

    def test_list_workspaces_empty(self, workspace_manager):
        """Test listing workspaces when none exist"""
        workspaces = workspace_manager.list_workspaces()
        assert workspaces == []

    def test_list_workspaces(self, workspace_manager):
        """Test listing multiple workspaces"""
        # Create multiple workspaces
        workspace_manager.create_workspace("project-1")
        workspace_manager.create_workspace("project-2")
        workspace_manager.create_workspace("project-3")

        workspaces = workspace_manager.list_workspaces()

        assert len(workspaces) == 3
        project_names = {ws.project_name for ws in workspaces}
        assert project_names == {"project-1", "project-2", "project-3"}

    def test_list_workspaces_with_invalid(self, workspace_manager, temp_base_path):
        """Test listing workspaces with invalid entries"""
        # Create valid workspace
        workspace_manager.create_workspace("valid-project")

        # Create invalid directory without metadata
        invalid_dir = temp_base_path / "invalid-project"
        invalid_dir.mkdir()

        workspaces = workspace_manager.list_workspaces()

        # Should only return valid workspace
        assert len(workspaces) == 1
        assert workspaces[0].project_name == "valid-project"

    def test_delete_workspace_requires_force(self, workspace_manager, sample_workspace):
        """Test that delete requires force flag"""
        with pytest.raises(WorkspaceError) as exc_info:
            workspace_manager.delete_workspace("test-project")

        assert "requires force=True" in str(exc_info.value)

    def test_delete_workspace(self, workspace_manager, sample_workspace):
        """Test deleting workspace with force"""
        workspace_path = sample_workspace.root_path

        workspace_manager.delete_workspace("test-project", force=True)

        assert not workspace_path.exists()
        assert "test-project" not in workspace_manager.workspaces

    def test_delete_workspace_not_found(self, workspace_manager):
        """Test deleting non-existent workspace"""
        with pytest.raises(WorkspaceError) as exc_info:
            workspace_manager.delete_workspace("nonexistent", force=True)

        assert "Workspace does not exist" in str(exc_info.value)

    def test_update_workspace_metadata(self, workspace_manager, sample_workspace):
        """Test updating workspace metadata"""
        updated_workspace = workspace_manager.update_workspace_metadata(
            project_name="test-project",
            mlflow_experiment_id="exp-updated",
            metadata={"new_key": "new_value"},
        )

        assert updated_workspace.mlflow_experiment_id == "exp-updated"
        assert "new_key" in updated_workspace.metadata
        assert updated_workspace.metadata["new_key"] == "new_value"
        # Original metadata should still exist
        assert updated_workspace.metadata["version"] == "1.0"

    def test_update_workspace_metadata_not_found(self, workspace_manager):
        """Test updating non-existent workspace"""
        with pytest.raises(WorkspaceError):
            workspace_manager.update_workspace_metadata(
                "nonexistent",
                mlflow_experiment_id="exp-123",
            )

    def test_get_workspace_status(self, workspace_manager, sample_workspace):
        """Test getting workspace status"""
        # Create some test files
        (sample_workspace.mlflow_path / "test.txt").write_text("test")
        (sample_workspace.models_path / "model.pkl").write_text("model")

        status = workspace_manager.get_workspace_status("test-project")

        assert status["project_name"] == "test-project"
        assert status["mlflow_experiment_id"] == "exp-123"
        assert "directory_stats" in status
        assert status["directory_stats"]["mlflow_exists"] is True
        assert status["directory_stats"]["dvc_exists"] is True
        assert status["directory_stats"]["mlflow_files"] >= 1
        assert status["directory_stats"]["models_files"] >= 1

    def test_get_all_workspaces_status(self, workspace_manager):
        """Test getting status for all workspaces"""
        # Create multiple workspaces
        workspace_manager.create_workspace("project-1")
        workspace_manager.create_workspace("project-2")

        all_status = workspace_manager.get_all_workspaces_status()

        assert len(all_status) == 2
        project_names = {status["project_name"] for status in all_status}
        assert project_names == {"project-1", "project-2"}

    def test_workspace_isolation(self, workspace_manager):
        """Test that workspaces are properly isolated"""
        workspace1 = workspace_manager.create_workspace("project-1")
        workspace2 = workspace_manager.create_workspace("project-2")

        # Write files to workspace 1
        (workspace1.mlflow_path / "exp1.txt").write_text("exp1")

        # Verify workspace 2 doesn't have the file
        assert not (workspace2.mlflow_path / "exp1.txt").exists()

        # Verify paths are different
        assert workspace1.root_path != workspace2.root_path


class TestWorkspaceIntegration:
    """Integration tests for workspace management"""

    def test_complete_workspace_lifecycle(self, workspace_manager):
        """Test complete workspace lifecycle"""
        # Create workspace
        workspace = workspace_manager.create_workspace(
            project_name="lifecycle-test",
            mlflow_experiment_id="exp-001",
            metadata={"stage": "created"},
        )

        assert workspace.root_path.exists()

        # Update workspace
        updated_workspace = workspace_manager.update_workspace_metadata(
            "lifecycle-test",
            mlflow_experiment_id="exp-002",
            metadata={"stage": "updated"},
        )

        assert updated_workspace.mlflow_experiment_id == "exp-002"
        assert updated_workspace.metadata["stage"] == "updated"

        # List workspaces
        workspaces = workspace_manager.list_workspaces()
        assert len(workspaces) == 1

        # Get status
        status = workspace_manager.get_workspace_status("lifecycle-test")
        assert status["mlflow_experiment_id"] == "exp-002"

        # Delete workspace
        workspace_manager.delete_workspace("lifecycle-test", force=True)

        # Verify deletion
        workspaces = workspace_manager.list_workspaces()
        assert len(workspaces) == 0

    def test_multiple_projects_workflow(self, workspace_manager):
        """Test managing multiple projects simultaneously"""
        projects = ["lora-finetuning-mlx", "model-compression-mlx", "coreml-style-transfer"]

        # Create workspaces for all projects
        for project in projects:
            workspace_manager.create_workspace(
                project_name=project,
                mlflow_experiment_id=f"exp-{project}",
                metadata={"project_type": "ml_toolkit"},
            )

        # List all workspaces
        workspaces = workspace_manager.list_workspaces()
        assert len(workspaces) == 3

        # Get status for all
        all_status = workspace_manager.get_all_workspaces_status()
        assert len(all_status) == 3

        # Verify each workspace is isolated
        for project in projects:
            workspace = workspace_manager.get_workspace(project)
            assert workspace.project_name == project
            assert workspace.mlflow_experiment_id == f"exp-{project}"

    def test_workspace_persistence_across_managers(self, temp_base_path):
        """Test that workspaces persist across manager instances"""
        # Create workspace with first manager
        manager1 = WorkspaceManager(base_path=temp_base_path)
        workspace1 = manager1.create_workspace(
            "persistent-project",
            mlflow_experiment_id="exp-persist",
        )

        # Create second manager instance
        manager2 = WorkspaceManager(base_path=temp_base_path)
        workspace2 = manager2.get_workspace("persistent-project")

        # Verify workspace was loaded correctly
        assert workspace2.project_name == workspace1.project_name
        assert workspace2.mlflow_experiment_id == workspace1.mlflow_experiment_id


class TestWorkspaceError:
    """Test WorkspaceError exception"""

    def test_workspace_error_attributes(self):
        """Test WorkspaceError stores error context"""
        error = WorkspaceError(
            "Test error",
            operation="test_op",
            workspace="test-workspace",
            details={"key": "value"},
        )

        assert str(error) == "Test error"
        assert error.operation == "test_op"
        assert error.workspace == "test-workspace"
        assert error.details == {"key": "value"}

    def test_workspace_error_minimal(self):
        """Test WorkspaceError with minimal arguments"""
        error = WorkspaceError("Simple error")

        assert str(error) == "Simple error"
        assert error.operation is None
        assert error.workspace is None
        assert error.details == {}
