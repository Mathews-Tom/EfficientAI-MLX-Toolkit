"""
Tests for DVC remote storage path management.

This test suite validates remote path management with project namespace
isolation across different storage backends.
"""

import pytest

from mlops.config.dvc_config import DVCConfig
from mlops.versioning.remote_manager import (
    MultiProjectRemoteManager,
    RemoteManagerError,
    RemotePathManager,
    create_multi_project_manager,
    create_remote_manager,
)


class TestRemotePathManager:
    """Test remote path manager."""

    def test_initialization(self):
        """Test manager initialization."""
        config = DVCConfig(
            storage_backend="s3",
            remote_url="s3://bucket/data",
            project_namespace="test-project",
        )

        manager = RemotePathManager(config)

        assert manager.backend == "s3"
        assert manager.project_namespace == "test-project"

    def test_get_project_remote_path_s3(self):
        """Test getting project remote path for S3."""
        config = DVCConfig(
            storage_backend="s3",
            remote_url="s3://bucket/data",
            project_namespace="test-project",
        )

        manager = RemotePathManager(config)
        path = manager.get_project_remote_path()

        assert path == "s3://bucket/data/test-project"

    def test_get_project_remote_path_with_subdirectory(self):
        """Test getting project path with subdirectory."""
        config = DVCConfig(
            storage_backend="s3",
            remote_url="s3://bucket/data",
            project_namespace="test-project",
        )

        manager = RemotePathManager(config)
        path = manager.get_project_remote_path("datasets")

        assert path == "s3://bucket/data/test-project/datasets"

    def test_get_dataset_path(self):
        """Test getting dataset path."""
        config = DVCConfig(
            storage_backend="s3",
            remote_url="s3://bucket/data",
            project_namespace="lora-finetuning",
        )

        manager = RemotePathManager(config)
        path = manager.get_dataset_path("train.csv")

        assert path == "s3://bucket/data/lora-finetuning/datasets/train.csv"

    def test_get_model_path(self):
        """Test getting model path."""
        config = DVCConfig(
            storage_backend="gcs",
            remote_url="gs://bucket/models",
            project_namespace="compression",
        )

        manager = RemotePathManager(config)
        path = manager.get_model_path("quantized-model", "v1.0")

        assert path == "gs://bucket/models/compression/models/quantized-model/v1.0"

    def test_get_model_path_latest(self):
        """Test getting latest model path."""
        config = DVCConfig(
            storage_backend="s3",
            remote_url="s3://bucket/data",
            project_namespace="test",
        )

        manager = RemotePathManager(config)
        path = manager.get_model_path("my-model")

        assert "latest" in path

    def test_get_checkpoint_path(self):
        """Test getting checkpoint path."""
        config = DVCConfig(
            storage_backend="azure",
            remote_url="azure://container/data",
            azure_account_name="myaccount",
            project_namespace="training",
        )

        manager = RemotePathManager(config)
        path = manager.get_checkpoint_path("epoch_10")

        assert path == "azure://container/data/training/checkpoints/epoch_10"

    def test_get_artifact_path(self):
        """Test getting artifact path."""
        config = DVCConfig(
            storage_backend="local",
            remote_url="./storage",
            project_namespace="test",
            environment="testing",
        )

        manager = RemotePathManager(config)
        path = manager.get_artifact_path("metrics", "training_metrics.json")

        assert "artifacts/metrics/training_metrics.json" in path

    def test_validate_remote_path_s3_valid(self):
        """Test validating valid S3 path."""
        config = DVCConfig(
            storage_backend="s3",
            remote_url="s3://bucket/data",
            project_namespace="test",
        )

        manager = RemotePathManager(config)
        assert manager.validate_remote_path("s3://bucket/data/test") is True

    def test_validate_remote_path_s3_invalid(self):
        """Test validating invalid S3 path."""
        config = DVCConfig(
            storage_backend="s3",
            remote_url="s3://bucket/data",
            project_namespace="test",
        )

        manager = RemotePathManager(config)

        with pytest.raises(RemoteManagerError, match="Invalid S3 path"):
            manager.validate_remote_path("gs://bucket/data")

    def test_validate_remote_path_gcs_valid(self):
        """Test validating valid GCS path."""
        config = DVCConfig(
            storage_backend="gcs",
            remote_url="gs://bucket/data",
            project_namespace="test",
        )

        manager = RemotePathManager(config)
        assert manager.validate_remote_path("gs://bucket/data/test") is True

    def test_validate_remote_path_gcs_invalid(self):
        """Test validating invalid GCS path."""
        config = DVCConfig(
            storage_backend="gcs",
            remote_url="gs://bucket/data",
            project_namespace="test",
        )

        manager = RemotePathManager(config)

        with pytest.raises(RemoteManagerError, match="Invalid GCS path"):
            manager.validate_remote_path("s3://bucket/data")

    def test_validate_remote_path_azure_valid(self):
        """Test validating valid Azure path."""
        config = DVCConfig(
            storage_backend="azure",
            remote_url="azure://container/data",
            azure_account_name="myaccount",
            project_namespace="test",
        )

        manager = RemotePathManager(config)
        assert manager.validate_remote_path("azure://container/data/test") is True

    def test_validate_remote_path_azure_invalid(self):
        """Test validating invalid Azure path."""
        config = DVCConfig(
            storage_backend="azure",
            remote_url="azure://container/data",
            azure_account_name="myaccount",
            project_namespace="test",
        )

        manager = RemotePathManager(config)

        with pytest.raises(RemoteManagerError, match="Invalid Azure path"):
            manager.validate_remote_path("s3://bucket/data")

    def test_validate_remote_path_local(self):
        """Test validating local path."""
        config = DVCConfig(
            storage_backend="local",
            remote_url="./storage",
            project_namespace="test",
            environment="testing",
        )

        manager = RemotePathManager(config)
        # Local paths are always valid (no strict validation)
        assert manager.validate_remote_path("./any/path") is True
        assert manager.validate_remote_path("/absolute/path") is True

    def test_parse_remote_path_s3(self):
        """Test parsing S3 remote path."""
        config = DVCConfig(
            storage_backend="s3",
            remote_url="s3://bucket/data",
            project_namespace="test",
        )

        manager = RemotePathManager(config)
        components = manager.parse_remote_path("s3://my-bucket/project/datasets/train.csv")

        assert components["backend"] == "s3"
        assert components["bucket"] == "my-bucket"
        assert components["project"] == "project"
        assert components["subdirectory"] == "datasets"
        assert components["filename"] == "train.csv"

    def test_parse_remote_path_gcs(self):
        """Test parsing GCS remote path."""
        config = DVCConfig(
            storage_backend="gcs",
            remote_url="gs://bucket/data",
            project_namespace="test",
        )

        manager = RemotePathManager(config)
        components = manager.parse_remote_path("gs://my-bucket/project/models/model.pkl")

        assert components["backend"] == "gcs"
        assert components["bucket"] == "my-bucket"
        assert components["project"] == "project"
        assert components["filename"] == "model.pkl"

    def test_parse_remote_path_local(self):
        """Test parsing local remote path."""
        config = DVCConfig(
            storage_backend="local",
            remote_url="./storage",
            project_namespace="test",
            environment="testing",
        )

        manager = RemotePathManager(config)
        components = manager.parse_remote_path("./storage/project/data.csv")

        assert components["backend"] == "local"
        assert components["filename"] == "data.csv"

    def test_list_project_directories(self):
        """Test listing project directories."""
        config = DVCConfig(
            storage_backend="s3",
            remote_url="s3://bucket/data",
            project_namespace="test-project",
        )

        manager = RemotePathManager(config)
        directories = manager.list_project_directories()

        assert len(directories) == 6
        assert any("datasets" in d for d in directories)
        assert any("models" in d for d in directories)
        assert any("checkpoints" in d for d in directories)
        assert any("artifacts" in d for d in directories)

    def test_get_backend_info_s3(self):
        """Test getting S3 backend info."""
        config = DVCConfig(
            storage_backend="s3",
            remote_url="s3://bucket/data",
            project_namespace="test",
            s3_region="us-west-2",
            s3_profile="myprofile",
        )

        manager = RemotePathManager(config)
        info = manager.get_backend_info()

        assert info["backend"] == "s3"
        assert info["project_namespace"] == "test"
        assert info["region"] == "us-west-2"
        assert info["profile"] == "myprofile"

    def test_get_backend_info_gcs(self):
        """Test getting GCS backend info."""
        config = DVCConfig(
            storage_backend="gcs",
            remote_url="gs://bucket/data",
            project_namespace="test",
            gcs_project="my-gcp-project",
        )

        manager = RemotePathManager(config)
        info = manager.get_backend_info()

        assert info["backend"] == "gcs"
        assert info["gcs_project"] == "my-gcp-project"

    def test_get_backend_info_azure(self):
        """Test getting Azure backend info."""
        config = DVCConfig(
            storage_backend="azure",
            remote_url="azure://container/data",
            project_namespace="test",
            azure_account_name="myaccount",
            azure_container_name="mycontainer",
        )

        manager = RemotePathManager(config)
        info = manager.get_backend_info()

        assert info["backend"] == "azure"
        assert info["account_name"] == "myaccount"
        assert info["container_name"] == "mycontainer"


class TestMultiProjectRemoteManager:
    """Test multi-project remote manager."""

    def test_initialization(self):
        """Test multi-project manager initialization."""
        manager = MultiProjectRemoteManager("s3", "s3://bucket/data")

        assert manager.backend == "s3"
        assert manager.base_remote_url == "s3://bucket/data"
        assert len(manager.projects) == 0

    def test_register_project(self):
        """Test registering a project."""
        manager = MultiProjectRemoteManager("s3", "s3://bucket/data")
        project_manager = manager.register_project("lora-finetuning")

        assert isinstance(project_manager, RemotePathManager)
        assert project_manager.project_namespace == "lora-finetuning"
        assert "lora-finetuning" in manager.projects

    def test_register_multiple_projects(self):
        """Test registering multiple projects."""
        manager = MultiProjectRemoteManager("gcs", "gs://bucket/data")

        manager.register_project("project-1")
        manager.register_project("project-2")
        manager.register_project("project-3")

        assert len(manager.projects) == 3

    def test_get_project_manager(self):
        """Test getting project manager."""
        manager = MultiProjectRemoteManager("s3", "s3://bucket/data")
        manager.register_project("test-project")

        project_manager = manager.get_project_manager("test-project")

        assert project_manager is not None
        assert project_manager.project_namespace == "test-project"

    def test_get_project_manager_not_found(self):
        """Test getting non-existent project manager."""
        manager = MultiProjectRemoteManager("s3", "s3://bucket/data")

        project_manager = manager.get_project_manager("nonexistent")

        assert project_manager is None

    def test_list_projects(self):
        """Test listing projects."""
        manager = MultiProjectRemoteManager("s3", "s3://bucket/data")

        manager.register_project("project-1")
        manager.register_project("project-2")

        projects = manager.list_projects()

        assert len(projects) == 2
        assert "project-1" in projects
        assert "project-2" in projects

    def test_get_all_project_paths(self):
        """Test getting all project paths."""
        manager = MultiProjectRemoteManager("s3", "s3://bucket/data")

        manager.register_project("project-1")
        manager.register_project("project-2")

        paths = manager.get_all_project_paths()

        assert len(paths) == 2
        assert "project-1" in paths
        assert "project-2" in paths
        assert "s3://bucket/data/project-1" in paths["project-1"]
        assert "s3://bucket/data/project-2" in paths["project-2"]


class TestCreateRemoteManager:
    """Test remote manager factory function."""

    def test_create_remote_manager(self):
        """Test creating remote manager from config."""
        config = DVCConfig(
            storage_backend="s3",
            remote_url="s3://bucket/data",
            project_namespace="test",
        )

        manager = create_remote_manager(config)

        assert isinstance(manager, RemotePathManager)
        assert manager.backend == "s3"


class TestCreateMultiProjectManager:
    """Test multi-project manager factory function."""

    def test_create_multi_project_manager(self):
        """Test creating multi-project manager."""
        manager = create_multi_project_manager("s3", "s3://bucket/data")

        assert isinstance(manager, MultiProjectRemoteManager)
        assert manager.backend == "s3"
        assert manager.base_remote_url == "s3://bucket/data"


class TestRemoteManagerError:
    """Test RemoteManagerError exception."""

    def test_error_with_backend(self):
        """Test error with backend information."""
        error = RemoteManagerError(
            "Test error",
            backend="s3",
        )

        assert str(error) == "Test error"
        assert error.backend == "s3"
        assert error.details == {}

    def test_error_with_details(self):
        """Test error with details."""
        details = {"path": "s3://bucket/data", "expected": "gs://"}
        error = RemoteManagerError(
            "Test error",
            backend="gcs",
            details=details,
        )

        assert error.details == details
