"""
Integration tests for DVC data versioning system.

This test suite validates end-to-end workflows with mocked DVC operations
and storage backends (no real DVC or cloud services required).
"""

import subprocess
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mlops.client.dvc_client import DVCClient
from mlops.config.dvc_config import DVCConfig
from mlops.versioning.dvc_operations import (
    DVCOperationsError,
    configure_remote_storage,
    create_workflow,
    get_data_status,
    initialize_project,
    list_remotes,
    restore_data_version,
    sync_data,
    track_dataset,
)
from mlops.versioning.remote_manager import (
    MultiProjectRemoteManager,
    RemotePathManager,
)


class TestDVCIntegrationWorkflow:
    """Test complete DVC workflow integration."""

    @patch("subprocess.run")
    def test_initialize_and_track_workflow(self, mock_run):
        """Test initializing project and tracking datasets."""
        mock_run.return_value = subprocess.CompletedProcess([], 0, "", "")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test dataset
            dataset_path = Path(tmpdir) / "train.csv"
            dataset_path.write_text("col1,col2\n1,2\n3,4")

            # Initialize project
            client = initialize_project(
                project_namespace="test-project",
                storage_backend="local",
                remote_url=str(Path(tmpdir) / "storage"),
                repo_root=tmpdir,
            )

            assert client.config.project_namespace == "test-project"
            assert mock_run.call_count >= 2  # init + remote add

            # Track dataset
            result = track_dataset(client, dataset_path, push_to_remote=True)

            assert result["success"] is True
            assert "push_status" in result

    @patch("subprocess.run")
    def test_sync_workflow_push_pull(self, mock_run):
        """Test synchronizing data with remote storage."""
        mock_run.return_value = subprocess.CompletedProcess([], 0, "", "")

        config = DVCConfig(
            storage_backend="s3",
            remote_url="s3://bucket/data",
            project_namespace="test",
        )
        client = DVCClient(config=config)

        # Push data
        push_result = sync_data(client, direction="push")
        assert push_result["success"] is True

        # Pull data
        pull_result = sync_data(client, direction="pull")
        assert pull_result["success"] is True

    def test_sync_invalid_direction(self):
        """Test sync with invalid direction."""
        config = DVCConfig(project_namespace="test")
        client = DVCClient(config=config)

        with pytest.raises(DVCOperationsError, match="Invalid sync direction"):
            sync_data(client, direction="invalid")

    @patch("subprocess.run")
    def test_status_workflow(self, mock_run):
        """Test checking data status."""
        mock_run.return_value = subprocess.CompletedProcess([], 0, "All files up to date", "")

        config = DVCConfig(project_namespace="test")
        client = DVCClient(config=config)

        status = get_data_status(client, check_remote=True)

        assert status["success"] is True
        assert status["is_clean"] is True

    @patch("subprocess.run")
    def test_restore_workflow(self, mock_run):
        """Test restoring data versions."""
        mock_run.return_value = subprocess.CompletedProcess([], 0, "", "")

        config = DVCConfig(project_namespace="test")
        client = DVCClient(config=config)

        result = restore_data_version(client, targets=["data.csv"])

        assert result["success"] is True
        assert "data.csv" in result["targets"]

    @patch("subprocess.run")
    def test_remote_configuration_workflow(self, mock_run):
        """Test configuring remote storage."""
        mock_run.return_value = subprocess.CompletedProcess([], 0, "", "")

        config = DVCConfig(project_namespace="test")
        client = DVCClient(config=config)

        result = configure_remote_storage(
            client,
            remote_name="production",
            remote_url="s3://prod-bucket/data",
        )

        assert result["success"] is True
        assert result["name"] == "production"

    @patch("subprocess.run")
    def test_list_remotes_workflow(self, mock_run):
        """Test listing configured remotes."""
        mock_run.return_value = subprocess.CompletedProcess(
            [], 0, "storage\t./dvc-storage\nbackup\ts3://bucket", ""
        )

        config = DVCConfig(project_namespace="test")
        client = DVCClient(config=config)

        remotes = list_remotes(client)

        assert len(remotes) == 2
        assert remotes[0]["name"] == "storage"


class TestMultiProjectIntegration:
    """Test multi-project integration scenarios."""

    def test_multiple_projects_isolation(self):
        """Test project namespace isolation."""
        manager = MultiProjectRemoteManager("s3", "s3://bucket/mlops-data")

        # Register multiple projects
        project1 = manager.register_project("lora-finetuning")
        project2 = manager.register_project("model-compression")
        project3 = manager.register_project("coreml-diffusion")

        # Verify isolation
        paths = manager.get_all_project_paths()

        assert len(paths) == 3
        assert "lora-finetuning" in paths["lora-finetuning"]
        assert "model-compression" in paths["model-compression"]
        assert "coreml-diffusion" in paths["coreml-diffusion"]

        # Verify no cross-contamination
        assert "lora-finetuning" not in paths["model-compression"]
        assert "model-compression" not in paths["coreml-diffusion"]

    def test_shared_backend_different_namespaces(self):
        """Test shared backend with different project namespaces."""
        backend = "s3"
        base_url = "s3://shared-bucket/data"

        manager = MultiProjectRemoteManager(backend, base_url)

        # Register projects
        projects = ["project-a", "project-b", "project-c"]
        for project in projects:
            manager.register_project(project)

        # Verify all use same backend but different paths
        for project in projects:
            proj_manager = manager.get_project_manager(project)
            assert proj_manager is not None
            assert proj_manager.backend == backend
            assert project in proj_manager.get_project_remote_path()


class TestStorageBackendIntegration:
    """Test integration with different storage backends."""

    @patch("subprocess.run")
    def test_s3_backend_workflow(self, mock_run):
        """Test workflow with S3 backend."""
        mock_run.return_value = subprocess.CompletedProcess([], 0, "", "")

        config = DVCConfig(
            storage_backend="s3",
            remote_url="s3://my-bucket/data",
            project_namespace="test-project",
            s3_region="us-west-2",
            s3_profile="default",
        )

        client = DVCClient(config=config)
        client.init()
        client.remote_add()

        # S3 configuration should be applied
        assert mock_run.call_count >= 3  # init + remote add + region config

    @patch("subprocess.run")
    def test_gcs_backend_workflow(self, mock_run):
        """Test workflow with GCS backend."""
        mock_run.return_value = subprocess.CompletedProcess([], 0, "", "")

        config = DVCConfig(
            storage_backend="gcs",
            remote_url="gs://my-bucket/data",
            project_namespace="test-project",
            gcs_project="my-gcp-project",
        )

        client = DVCClient(config=config)
        client.init()
        client.remote_add()

        assert mock_run.call_count >= 2  # init + remote add

    @patch("subprocess.run")
    def test_azure_backend_workflow(self, mock_run):
        """Test workflow with Azure backend."""
        mock_run.return_value = subprocess.CompletedProcess([], 0, "", "")

        config = DVCConfig(
            storage_backend="azure",
            remote_url="azure://container/data",
            project_namespace="test-project",
            azure_account_name="myaccount",
        )

        client = DVCClient(config=config)
        client.init()
        client.remote_add()

        assert mock_run.call_count >= 2

    def test_local_backend_workflow(self):
        """Test workflow with local backend."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = DVCConfig(
                storage_backend="local",
                remote_url=str(Path(tmpdir) / "storage"),
                project_namespace="test-project",
                environment="development",  # Use development to get namespace in path
            )

            manager = RemotePathManager(config)

            # Local paths should work
            dataset_path = manager.get_dataset_path("train.csv")
            # In development mode, local paths include project namespace
            assert "test-project" in dataset_path or "datasets" in dataset_path
            assert "datasets" in dataset_path


class TestRemotePathIntegration:
    """Test remote path management integration."""

    def test_complete_project_structure(self):
        """Test creating complete project structure."""
        config = DVCConfig(
            storage_backend="s3",
            remote_url="s3://bucket/mlops",
            project_namespace="lora-finetuning-mlx",
        )

        manager = RemotePathManager(config)

        # Test all path types
        dataset_path = manager.get_dataset_path("training_data.jsonl")
        model_path = manager.get_model_path("lora-adapter", "v1.0")
        checkpoint_path = manager.get_checkpoint_path("epoch_10")
        artifact_path = manager.get_artifact_path("metrics", "loss.json")

        # Verify all paths contain project namespace
        for path in [dataset_path, model_path, checkpoint_path, artifact_path]:
            assert "lora-finetuning-mlx" in path
            assert path.startswith("s3://bucket/mlops")

    def test_path_validation_across_backends(self):
        """Test path validation for all backends."""
        backends = [
            ("s3", "s3://bucket/data", "s3://bucket/data/test"),
            ("gcs", "gs://bucket/data", "gs://bucket/data/test"),
            ("azure", "azure://container/data", "azure://container/data/test"),
            ("local", "./storage", "./storage/test"),
        ]

        for backend, remote_url, test_path in backends:
            if backend == "azure":
                config = DVCConfig(
                    storage_backend=backend,  # type: ignore
                    remote_url=remote_url,
                    project_namespace="test",
                    azure_account_name="account",
                )
            else:
                config = DVCConfig(
                    storage_backend=backend,  # type: ignore
                    remote_url=remote_url,
                    project_namespace="test",
                )

            manager = RemotePathManager(config)
            assert manager.validate_remote_path(test_path) is True


class TestEndToEndWorkflow:
    """Test complete end-to-end workflows."""

    @patch("subprocess.run")
    def test_full_project_lifecycle(self, mock_run):
        """Test complete project lifecycle from init to sync."""
        mock_run.return_value = subprocess.CompletedProcess([], 0, "", "")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test datasets
            dataset1 = Path(tmpdir) / "train.csv"
            dataset2 = Path(tmpdir) / "test.csv"
            dataset1.write_text("data1")
            dataset2.write_text("data2")

            # Initialize project
            client = initialize_project(
                project_namespace="full-lifecycle-test",
                storage_backend="local",
                remote_url=str(Path(tmpdir) / "storage"),
                repo_root=tmpdir,
            )

            # Track datasets
            track_dataset(client, dataset1, push_to_remote=True)
            track_dataset(client, dataset2, push_to_remote=True)

            # Check status
            status = get_data_status(client)
            assert status["success"] is True

            # Simulate modification and restore
            dataset1.write_text("modified")
            restore_data_version(client, targets=[dataset1])

    @patch("subprocess.run")
    def test_multi_project_shared_storage(self, mock_run):
        """Test multiple projects sharing storage backend."""
        mock_run.return_value = subprocess.CompletedProcess([], 0, "", "")

        with tempfile.TemporaryDirectory() as tmpdir:
            storage_url = str(Path(tmpdir) / "shared-storage")

            # Create multiple projects
            projects = ["project-a", "project-b", "project-c"]
            clients = []

            for project in projects:
                client = initialize_project(
                    project_namespace=project,
                    storage_backend="local",
                    remote_url=storage_url,
                    repo_root=tmpdir,
                )
                clients.append(client)

            # Verify all clients configured
            assert len(clients) == 3

            # Verify isolation via path manager
            manager = MultiProjectRemoteManager("local", storage_url)
            for project in projects:
                manager.register_project(project)

            paths = manager.get_all_project_paths()
            assert len(paths) == 3


class TestErrorHandling:
    """Test error handling in integration scenarios."""

    def test_invalid_backend_initialization(self):
        """Test initialization with invalid backend."""
        with pytest.raises(Exception):  # DVCConfigError from validation
            DVCConfig(
                storage_backend="invalid",  # type: ignore
                project_namespace="test",
            )

    @patch("subprocess.run")
    def test_track_nonexistent_dataset(self, mock_run):
        """Test tracking nonexistent dataset."""
        config = DVCConfig(project_namespace="test")
        client = DVCClient(config=config)

        with pytest.raises(DVCOperationsError):
            track_dataset(client, "/nonexistent/file.csv")

    @patch("subprocess.run")
    def test_dvc_command_failure_handling(self, mock_run):
        """Test handling of DVC command failures."""
        mock_run.side_effect = subprocess.CalledProcessError(1, ["dvc"], stderr="Error")

        config = DVCConfig(project_namespace="test")
        client = DVCClient(config=config)

        with pytest.raises(DVCOperationsError):
            sync_data(client, direction="push")


class TestPerformanceIntegration:
    """Test performance-related integration scenarios."""

    @patch("subprocess.run")
    def test_parallel_jobs_configuration(self, mock_run):
        """Test parallel jobs configuration."""
        mock_run.return_value = subprocess.CompletedProcess([], 0, "", "")

        config = DVCConfig(
            project_namespace="test",
            jobs=16,  # High parallelism
        )

        client = DVCClient(config=config)
        client.push()

        # Verify jobs parameter was used
        call_args = mock_run.call_args
        assert "--jobs" in call_args[0][0]
        assert "16" in call_args[0][0]

    @patch("subprocess.run")
    def test_large_dataset_batch_operations(self, mock_run):
        """Test batch operations on multiple datasets."""
        mock_run.return_value = subprocess.CompletedProcess([], 0, "", "")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create multiple datasets
            datasets = []
            for i in range(10):
                dataset = Path(tmpdir) / f"dataset_{i}.csv"
                dataset.write_text(f"data{i}")
                datasets.append(dataset)

            # Track all datasets
            config = DVCConfig(
                storage_backend="local",
                remote_url=str(Path(tmpdir) / "storage"),
                project_namespace="batch-test",
            )
            client = DVCClient(config=config, repo_root=tmpdir)

            # Batch push
            client.push(targets=datasets)

            # Should have made single push call
            assert mock_run.call_count >= 1


class TestConfigurationIntegration:
    """Test configuration integration scenarios."""

    def test_environment_based_configuration(self):
        """Test configuration across environments."""
        environments = ["development", "production", "testing"]

        for env in environments:
            config = DVCConfig(
                project_namespace="test",
                environment=env,  # type: ignore
            )

            assert config.environment == env

            # Verify environment-specific defaults
            if env == "testing":
                assert "/tmp/dvc-test-storage" in config.remote_url

    def test_backend_specific_configuration(self):
        """Test backend-specific configuration settings."""
        # S3 configuration
        s3_config = DVCConfig(
            storage_backend="s3",
            remote_url="s3://bucket/data",
            project_namespace="test",
            s3_region="us-west-2",
            s3_profile="default",
            s3_endpoint_url="http://localhost:9000",
        )

        info = s3_config.get_connection_info()
        assert info["s3_region"] == "us-west-2"
        assert info["s3_profile"] == "default"
        assert info["s3_endpoint_url"] == "http://localhost:9000"

        # GCS configuration
        gcs_config = DVCConfig(
            storage_backend="gcs",
            remote_url="gs://bucket/data",
            project_namespace="test",
            gcs_project="my-project",
        )

        info = gcs_config.get_connection_info()
        assert info["gcs_project"] == "my-project"
