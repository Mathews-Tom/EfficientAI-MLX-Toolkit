"""
Tests for DVC configuration module.

This test suite validates DVC configuration with storage backend abstraction,
environment-based settings, and validation logic.
"""

import os
import tempfile
from pathlib import Path

import pytest

from mlops.config.dvc_config import (
    DVCConfig,
    DVCConfigError,
    get_default_config,
    load_config_from_file,
)


class TestDVCConfig:
    """Test DVC configuration dataclass."""

    def test_default_config(self):
        """Test default DVC configuration."""
        config = DVCConfig()

        assert config.storage_backend == "local"
        assert config.remote_name == "storage"
        # Development environment auto-appends project namespace to local paths
        assert config.remote_url == "./dvc-storage/default"
        assert config.cache_dir == ".dvc/cache"
        assert config.project_namespace == "default"
        assert config.environment == "development"
        assert config.enable_autostage is True
        assert config.enable_symlinks is True
        assert config.enable_hardlinks is True
        assert config.verify_checksums is True
        assert config.jobs == 4

    def test_custom_config(self):
        """Test custom DVC configuration."""
        config = DVCConfig(
            storage_backend="s3",
            remote_url="s3://my-bucket/data",
            project_namespace="test-project",
            environment="production",
            jobs=8,
        )

        assert config.storage_backend == "s3"
        assert config.remote_url == "s3://my-bucket/data"
        assert config.project_namespace == "test-project"
        assert config.environment == "production"
        assert config.jobs == 8

    def test_s3_backend_validation(self):
        """Test S3 backend validation."""
        # Valid S3 config
        config = DVCConfig(
            storage_backend="s3",
            remote_url="s3://bucket/path",
            project_namespace="test",
        )
        assert config.storage_backend == "s3"

        # Invalid S3 URL
        with pytest.raises(DVCConfigError, match="S3 backend requires remote_url to start with 's3://'"):
            DVCConfig(
                storage_backend="s3",
                remote_url="http://bucket/path",
                project_namespace="test",
            )

    def test_gcs_backend_validation(self):
        """Test GCS backend validation."""
        # Valid GCS config
        config = DVCConfig(
            storage_backend="gcs",
            remote_url="gs://bucket/path",
            project_namespace="test",
        )
        assert config.storage_backend == "gcs"

        # Invalid GCS URL
        with pytest.raises(DVCConfigError, match="GCS backend requires remote_url to start with 'gs://'"):
            DVCConfig(
                storage_backend="gcs",
                remote_url="s3://bucket/path",
                project_namespace="test",
            )

    def test_azure_backend_validation(self):
        """Test Azure backend validation."""
        # Valid Azure config
        config = DVCConfig(
            storage_backend="azure",
            remote_url="azure://container/path",
            azure_account_name="myaccount",
            project_namespace="test",
        )
        assert config.storage_backend == "azure"
        assert config.azure_account_name == "myaccount"

        # Missing account name
        with pytest.raises(DVCConfigError, match="Azure backend requires azure_account_name"):
            DVCConfig(
                storage_backend="azure",
                remote_url="azure://container/path",
                project_namespace="test",
            )

        # Invalid Azure URL
        with pytest.raises(DVCConfigError, match="Azure backend requires remote_url to start with 'azure://'"):
            DVCConfig(
                storage_backend="azure",
                remote_url="s3://bucket/path",
                azure_account_name="myaccount",
                project_namespace="test",
            )

    def test_local_backend_validation(self):
        """Test local backend validation."""
        # Valid local paths
        config1 = DVCConfig(
            storage_backend="local",
            remote_url="./dvc-storage",
            project_namespace="test",
        )
        assert config1.storage_backend == "local"

        config2 = DVCConfig(
            storage_backend="local",
            remote_url="/absolute/path/storage",
            project_namespace="test",
        )
        assert config2.storage_backend == "local"

    def test_invalid_storage_backend(self):
        """Test invalid storage backend."""
        with pytest.raises(DVCConfigError, match="Invalid storage backend"):
            DVCConfig(
                storage_backend="invalid",  # type: ignore
                project_namespace="test",
            )

    def test_invalid_environment(self):
        """Test invalid environment type."""
        with pytest.raises(DVCConfigError, match="Invalid environment"):
            DVCConfig(
                environment="invalid",  # type: ignore
                project_namespace="test",
            )

    def test_empty_remote_name(self):
        """Test validation of empty remote name."""
        with pytest.raises(DVCConfigError, match="remote_name cannot be empty"):
            DVCConfig(
                remote_name="",
                project_namespace="test",
            )

    def test_empty_remote_url(self):
        """Test validation of empty remote URL."""
        with pytest.raises(DVCConfigError, match="remote_url cannot be empty"):
            DVCConfig(
                remote_url="",
                project_namespace="test",
            )

    def test_empty_project_namespace(self):
        """Test validation of empty project namespace."""
        with pytest.raises(DVCConfigError, match="project_namespace cannot be empty"):
            DVCConfig(
                project_namespace="",
            )

    def test_environment_overrides(self, monkeypatch):
        """Test environment variable overrides."""
        monkeypatch.setenv("DVC_STORAGE_BACKEND", "s3")
        monkeypatch.setenv("DVC_REMOTE_URL", "s3://env-bucket/data")
        monkeypatch.setenv("DVC_PROJECT_NAMESPACE", "env-project")
        monkeypatch.setenv("DVC_ENABLE_AUTOSTAGE", "false")
        monkeypatch.setenv("DVC_JOBS", "16")

        config = DVCConfig()

        assert config.storage_backend == "s3"
        assert config.remote_url == "s3://env-bucket/data"
        assert config.project_namespace == "env-project"
        assert config.enable_autostage is False
        assert config.jobs == 16

    def test_s3_environment_overrides(self, monkeypatch):
        """Test S3-specific environment overrides."""
        monkeypatch.setenv("DVC_S3_REGION", "us-west-2")
        monkeypatch.setenv("DVC_S3_PROFILE", "my-profile")
        monkeypatch.setenv("DVC_S3_ENDPOINT_URL", "http://localhost:9000")

        config = DVCConfig(
            storage_backend="s3",
            remote_url="s3://bucket/path",
            project_namespace="test",
        )

        assert config.s3_region == "us-west-2"
        assert config.s3_profile == "my-profile"
        assert config.s3_endpoint_url == "http://localhost:9000"

    def test_development_environment_defaults(self):
        """Test development environment defaults."""
        config = DVCConfig(
            environment="development",
            project_namespace="test-project",
        )

        assert config.environment == "development"
        assert config.remote_url == "./dvc-storage/test-project"

    def test_production_environment_warnings(self, caplog):
        """Test production environment warnings."""
        config = DVCConfig(
            environment="production",
            storage_backend="local",
            project_namespace="test",
        )

        assert "Production environment using local storage backend" in caplog.text

    def test_testing_environment_defaults(self):
        """Test testing environment defaults."""
        config = DVCConfig(
            environment="testing",
            project_namespace="test-project",
        )

        assert config.environment == "testing"
        assert config.remote_url == "/tmp/dvc-test-storage/test-project"
        assert config.cache_dir == "/tmp/dvc-test-cache"

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = DVCConfig(
            storage_backend="s3",
            remote_url="s3://bucket/path",
            project_namespace="test",
            s3_region="us-east-1",
        )

        config_dict = config.to_dict()

        assert config_dict["storage_backend"] == "s3"
        assert config_dict["remote_url"] == "s3://bucket/path"
        assert config_dict["project_namespace"] == "test"
        assert config_dict["s3_region"] == "us-east-1"
        assert "tags" in config_dict

    def test_from_dict(self):
        """Test creation from dictionary."""
        config_dict = {
            "storage_backend": "gcs",
            "remote_url": "gs://bucket/path",
            "project_namespace": "test",
            "gcs_project": "my-project",
            "jobs": 8,
        }

        config = DVCConfig.from_dict(config_dict)

        assert config.storage_backend == "gcs"
        assert config.remote_url == "gs://bucket/path"
        assert config.project_namespace == "test"
        assert config.gcs_project == "my-project"
        assert config.jobs == 8

    def test_from_dict_filters_invalid_keys(self):
        """Test from_dict filters invalid keys."""
        config_dict = {
            "storage_backend": "local",
            "remote_url": "./storage",
            "project_namespace": "test",
            "invalid_key": "should_be_ignored",
        }

        config = DVCConfig.from_dict(config_dict)

        assert config.storage_backend == "local"
        assert not hasattr(config, "invalid_key")

    def test_from_environment(self):
        """Test creation from environment."""
        config = DVCConfig.from_environment(
            environment="production",
            project_namespace="prod-project",
        )

        assert config.environment == "production"
        assert config.project_namespace == "prod-project"

    def test_validate(self):
        """Test validate method."""
        config = DVCConfig(project_namespace="test")
        assert config.validate() is True

    def test_get_connection_info(self):
        """Test get_connection_info method."""
        config = DVCConfig(
            storage_backend="s3",
            remote_url="s3://bucket/path",
            project_namespace="test",
            s3_region="us-west-2",
        )

        info = config.get_connection_info()

        assert info["storage_backend"] == "s3"
        assert info["remote_url"] == "s3://bucket/path"
        assert info["project_namespace"] == "test"
        assert info["s3_region"] == "us-west-2"

    def test_update_tags(self):
        """Test update_tags method."""
        config = DVCConfig(project_namespace="test")
        config.update_tags({"version": "1.0", "team": "ml"})

        assert config.tags["version"] == "1.0"
        assert config.tags["team"] == "ml"

        config.update_tags({"version": "2.0"})
        assert config.tags["version"] == "2.0"
        assert config.tags["team"] == "ml"

    def test_get_cache_path(self):
        """Test get_cache_path method."""
        config = DVCConfig(
            cache_dir=".dvc/cache",
            project_namespace="test",
        )

        base_path = config.get_cache_path()
        assert base_path == Path(".dvc/cache")

        relative_path = config.get_cache_path("models")
        assert relative_path == Path(".dvc/cache/models")

    def test_ensure_cache_dir(self):
        """Test ensure_cache_dir method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "cache"
            config = DVCConfig(
                cache_dir=str(cache_dir),
                project_namespace="test",
            )

            result = config.ensure_cache_dir()

            assert result == cache_dir
            assert cache_dir.exists()
            assert cache_dir.is_dir()

    def test_get_remote_path_local(self):
        """Test get_remote_path for local backend."""
        config = DVCConfig(
            storage_backend="local",
            remote_url="./dvc-storage",
            project_namespace="test",
            environment="testing",  # Avoid auto-append in development mode
        )

        base = config.get_remote_path()
        # Remote URL gets updated to /tmp/dvc-test-storage/test in testing mode
        assert base == "/tmp/dvc-test-storage/test"

        relative = config.get_remote_path("datasets")
        assert relative == str(Path("/tmp/dvc-test-storage/test") / "datasets")

    def test_get_remote_path_s3(self):
        """Test get_remote_path for S3 backend."""
        config = DVCConfig(
            storage_backend="s3",
            remote_url="s3://bucket/data",
            project_namespace="test-project",
        )

        base = config.get_remote_path()
        assert base == "s3://bucket/data/test-project"

        relative = config.get_remote_path("datasets")
        assert relative == "s3://bucket/data/test-project/datasets"


class TestGetDefaultConfig:
    """Test get_default_config function."""

    def test_default_config(self):
        """Test getting default configuration."""
        config = get_default_config()

        assert config.environment == "development"
        assert config.project_namespace == "default"

    def test_custom_environment(self):
        """Test custom environment."""
        config = get_default_config(environment="production", project_namespace="prod-project")

        assert config.environment == "production"
        assert config.project_namespace == "prod-project"


class TestLoadConfigFromFile:
    """Test load_config_from_file function."""

    def test_load_yaml_config(self):
        """Test loading configuration from YAML file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text(
                """
dvc:
  storage_backend: s3
  remote_url: s3://bucket/data
  project_namespace: yaml-project
  s3_region: us-east-1
"""
            )

            config = load_config_from_file(config_path)

            assert config.storage_backend == "s3"
            assert config.remote_url == "s3://bucket/data"
            assert config.project_namespace == "yaml-project"
            assert config.s3_region == "us-east-1"

    def test_load_with_environment_override(self):
        """Test loading with environment override."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text(
                """
dvc:
  storage_backend: local
  remote_url: ./storage
  project_namespace: test
  environment: development
"""
            )

            config = load_config_from_file(config_path, environment="production")

            assert config.environment == "production"

    def test_load_with_namespace_override(self):
        """Test loading with project namespace override."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text(
                """
dvc:
  storage_backend: local
  remote_url: ./storage
  project_namespace: default
"""
            )

            config = load_config_from_file(config_path, project_namespace="override-project")

            assert config.project_namespace == "override-project"

    def test_load_nonexistent_file(self):
        """Test loading from nonexistent file."""
        with pytest.raises(DVCConfigError, match="Configuration file not found"):
            load_config_from_file(Path("/nonexistent/config.yaml"))

    def test_load_invalid_format(self):
        """Test loading from file with invalid format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text("dvc: invalid_format")

            with pytest.raises(DVCConfigError, match="Invalid DVC configuration format"):
                load_config_from_file(config_path)


class TestDVCConfigError:
    """Test DVCConfigError exception."""

    def test_error_with_environment(self):
        """Test error with environment information."""
        error = DVCConfigError(
            "Test error",
            environment="production",
        )

        assert str(error) == "Test error"
        assert error.environment == "production"
        assert error.backend is None
        assert error.details == {}

    def test_error_with_backend(self):
        """Test error with backend information."""
        error = DVCConfigError(
            "Test error",
            backend="s3",
        )

        assert error.backend == "s3"

    def test_error_with_details(self):
        """Test error with details."""
        details = {"key": "value", "count": 42}
        error = DVCConfigError(
            "Test error",
            environment="development",
            backend="local",
            details=details,
        )

        assert error.details == details
