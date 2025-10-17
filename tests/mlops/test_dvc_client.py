"""
Tests for DVC client API.

This test suite validates DVC client operations with mocked DVC commands
to avoid requiring actual DVC installation or external storage.
"""

import subprocess
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mlops.client.dvc_client import DVCClient, DVCClientError, create_client
from mlops.config.dvc_config import DVCConfig


class TestDVCClient:
    """Test DVC client core functionality."""

    def test_client_initialization(self):
        """Test client initialization with default config."""
        client = DVCClient()

        assert client.config is not None
        assert client.repo_root == Path.cwd()
        assert client.dvc_dir == Path.cwd() / ".dvc"

    def test_client_with_custom_config(self):
        """Test client with custom configuration."""
        config = DVCConfig(
            storage_backend="s3",
            remote_url="s3://bucket/data",
            project_namespace="test",
        )

        client = DVCClient(config=config)

        assert client.config.storage_backend == "s3"
        assert client.config.project_namespace == "test"

    def test_client_with_custom_repo_root(self):
        """Test client with custom repository root."""
        with tempfile.TemporaryDirectory() as tmpdir:
            client = DVCClient(repo_root=tmpdir)

            assert client.repo_root == Path(tmpdir)
            assert client.dvc_dir == Path(tmpdir) / ".dvc"

    @patch("subprocess.run")
    def test_init(self, mock_run):
        """Test DVC initialization."""
        mock_run.return_value = subprocess.CompletedProcess([], 0, "", "")

        client = DVCClient()
        client.init()

        mock_run.assert_called_once()
        call_args = mock_run.call_args
        assert call_args[0][0][0] == "dvc"
        assert call_args[0][0][1] == "init"

    @patch("subprocess.run")
    def test_init_force(self, mock_run):
        """Test forced DVC initialization."""
        mock_run.return_value = subprocess.CompletedProcess([], 0, "", "")

        client = DVCClient()
        client.init(force=True)

        call_args = mock_run.call_args
        assert "--force" in call_args[0][0]

    @patch("subprocess.run")
    def test_add_file(self, mock_run):
        """Test adding file to DVC tracking."""
        mock_run.return_value = subprocess.CompletedProcess([], 0, "", "")

        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.txt"
            test_file.write_text("test data")

            client = DVCClient(repo_root=tmpdir)
            result = client.add(test_file)

            assert result["success"] is True
            assert result["path"] == str(test_file)
            assert ".dvc" in result["dvc_file"]

    def test_add_nonexistent_file(self):
        """Test adding nonexistent file fails."""
        client = DVCClient()

        with pytest.raises(DVCClientError, match="Path not found"):
            client.add("/nonexistent/file.txt")

    @patch("subprocess.run")
    def test_remove(self, mock_run):
        """Test removing file from DVC tracking."""
        mock_run.return_value = subprocess.CompletedProcess([], 0, "", "")

        client = DVCClient()
        result = client.remove("test.txt.dvc")

        assert result["success"] is True
        assert result["path"] == "test.txt.dvc"

    @patch("subprocess.run")
    def test_push_default(self, mock_run):
        """Test pushing to default remote."""
        mock_run.return_value = subprocess.CompletedProcess([], 0, "", "")

        config = DVCConfig(
            remote_name="storage",
            project_namespace="test",
        )
        client = DVCClient(config=config)
        result = client.push()

        assert result["success"] is True
        assert result["remote"] == "storage"

    @patch("subprocess.run")
    def test_push_with_targets(self, mock_run):
        """Test pushing specific targets."""
        mock_run.return_value = subprocess.CompletedProcess([], 0, "", "")

        client = DVCClient()
        result = client.push(targets=["data.csv", "model.pkl"])

        assert result["success"] is True
        assert len(result["targets"]) == 2

    @patch("subprocess.run")
    def test_push_with_jobs(self, mock_run):
        """Test pushing with parallel jobs."""
        mock_run.return_value = subprocess.CompletedProcess([], 0, "", "")

        client = DVCClient()
        client.push(jobs=8)

        call_args = mock_run.call_args
        assert "--jobs" in call_args[0][0]
        assert "8" in call_args[0][0]

    @patch("subprocess.run")
    def test_pull_default(self, mock_run):
        """Test pulling from default remote."""
        mock_run.return_value = subprocess.CompletedProcess([], 0, "", "")

        config = DVCConfig(
            remote_name="storage",
            project_namespace="test",
        )
        client = DVCClient(config=config)
        result = client.pull()

        assert result["success"] is True
        assert result["remote"] == "storage"

    @patch("subprocess.run")
    def test_pull_with_force(self, mock_run):
        """Test forced pull."""
        mock_run.return_value = subprocess.CompletedProcess([], 0, "", "")

        client = DVCClient()
        client.pull(force=True)

        call_args = mock_run.call_args
        assert "--force" in call_args[0][0]

    @patch("subprocess.run")
    def test_status_clean(self, mock_run):
        """Test status check when clean."""
        mock_run.return_value = subprocess.CompletedProcess([], 0, "Nothing to commit", "")

        client = DVCClient()
        result = client.status()

        assert result["success"] is True
        assert result["is_clean"] is True

    @patch("subprocess.run")
    def test_status_with_changes(self, mock_run):
        """Test status check with changes."""
        mock_run.return_value = subprocess.CompletedProcess([], 1, "Changes detected", "")

        client = DVCClient()
        result = client.status()

        assert result["success"] is True
        assert result["is_clean"] is False

    @patch("subprocess.run")
    def test_checkout(self, mock_run):
        """Test checking out data files."""
        mock_run.return_value = subprocess.CompletedProcess([], 0, "", "")

        client = DVCClient()
        result = client.checkout(targets=["data.csv"])

        assert result["success"] is True
        assert "data.csv" in result["targets"]

    @patch("subprocess.run")
    def test_remote_add_default(self, mock_run):
        """Test adding default remote."""
        mock_run.return_value = subprocess.CompletedProcess([], 0, "", "")

        config = DVCConfig(
            storage_backend="local",
            remote_name="storage",
            remote_url="./dvc-storage",
            project_namespace="test",
        )
        client = DVCClient(config=config)
        result = client.remote_add()

        assert result["success"] is True
        assert result["name"] == "storage"
        assert result["backend"] == "local"

    @patch("subprocess.run")
    def test_remote_add_s3(self, mock_run):
        """Test adding S3 remote with configuration."""
        mock_run.return_value = subprocess.CompletedProcess([], 0, "", "")

        config = DVCConfig(
            storage_backend="s3",
            remote_name="s3-storage",
            remote_url="s3://bucket/data",
            project_namespace="test",
            s3_region="us-west-2",
            s3_profile="myprofile",
        )
        client = DVCClient(config=config)
        result = client.remote_add()

        assert result["success"] is True
        assert result["backend"] == "s3"

        # Should have multiple subprocess calls for S3 configuration
        assert mock_run.call_count >= 3  # add + region + profile

    @patch("subprocess.run")
    def test_remote_list(self, mock_run):
        """Test listing remotes."""
        mock_run.return_value = subprocess.CompletedProcess(
            [], 0, "storage\t./dvc-storage\nbackup\ts3://bucket/backup", ""
        )

        client = DVCClient()
        remotes = client.remote_list()

        assert len(remotes) == 2
        assert remotes[0]["name"] == "storage"
        assert remotes[1]["name"] == "backup"

    @patch("subprocess.run")
    def test_remote_list_empty(self, mock_run):
        """Test listing remotes when none configured."""
        mock_run.return_value = subprocess.CompletedProcess([], 0, "", "")

        client = DVCClient()
        remotes = client.remote_list()

        assert len(remotes) == 0

    def test_get_connection_info(self):
        """Test getting connection information."""
        config = DVCConfig(
            storage_backend="s3",
            remote_url="s3://bucket/data",
            project_namespace="test",
            s3_region="us-east-1",
        )
        client = DVCClient(config=config)

        info = client.get_connection_info()

        assert info["storage_backend"] == "s3"
        assert info["project_namespace"] == "test"
        assert info["s3_region"] == "us-east-1"

    @patch("subprocess.run")
    def test_dvc_command_failure(self, mock_run):
        """Test handling of DVC command failures."""
        mock_run.side_effect = subprocess.CalledProcessError(
            1, ["dvc", "push"], stderr="Push failed"
        )

        client = DVCClient()

        with pytest.raises(DVCClientError, match="DVC command failed"):
            client.push()

    @patch("subprocess.run")
    def test_dvc_not_installed(self, mock_run):
        """Test handling when DVC is not installed."""
        mock_run.side_effect = FileNotFoundError()

        client = DVCClient()

        with pytest.raises(DVCClientError, match="DVC is not installed"):
            client.init()


class TestCreateClient:
    """Test client factory function."""

    def test_create_default_client(self):
        """Test creating client with defaults."""
        client = create_client()

        assert isinstance(client, DVCClient)
        assert client.config is not None

    def test_create_client_with_config(self):
        """Test creating client with custom config."""
        config = DVCConfig(
            storage_backend="s3",
            remote_url="s3://bucket/data",
            project_namespace="test",
        )

        client = create_client(config=config)

        assert client.config.storage_backend == "s3"

    def test_create_client_with_repo_root(self):
        """Test creating client with custom repo root."""
        with tempfile.TemporaryDirectory() as tmpdir:
            client = create_client(repo_root=tmpdir)

            assert client.repo_root == Path(tmpdir)


class TestDVCClientMockMode:
    """Test DVC client mock mode for testing."""

    def test_mock_mode_initialization(self):
        """Test enabling mock mode."""
        client = DVCClient()
        client._mock_mode = True
        client._mock_responses = {
            "init": subprocess.CompletedProcess([], 0, "", "")
        }

        # Should not raise even without subprocess mock
        result = client._run_dvc_command(["init"])
        assert result.returncode == 0

    def test_mock_mode_add(self):
        """Test add operation in mock mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.txt"
            test_file.write_text("test")

            client = DVCClient(repo_root=tmpdir)
            client._mock_mode = True
            client._mock_responses = {
                "add " + str(test_file): subprocess.CompletedProcess([], 0, "", "")
            }

            result = client.add(test_file)
            assert result["success"] is True


class TestDVCClientError:
    """Test DVCClientError exception."""

    def test_error_with_operation(self):
        """Test error with operation information."""
        error = DVCClientError(
            "Test error",
            operation="push",
        )

        assert str(error) == "Test error"
        assert error.operation == "push"
        assert error.details == {}

    def test_error_with_details(self):
        """Test error with details."""
        details = {"remote": "storage", "files": 5}
        error = DVCClientError(
            "Test error",
            operation="push",
            details=details,
        )

        assert error.details == details
