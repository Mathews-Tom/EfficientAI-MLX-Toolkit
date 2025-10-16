"""
Tests for MLFlow configuration module.

This module tests the MLFlowConfig class and related functions for
environment-based configuration management.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from mlops.config import (
    MLFlowConfig,
    MLFlowConfigError,
    get_default_config,
    load_config_from_file,
)


class TestMLFlowConfig:
    """Test MLFlowConfig dataclass and methods."""

    def test_default_initialization(self):
        """Test default configuration initialization."""
        config = MLFlowConfig()

        assert config.tracking_uri == "http://localhost:5000"
        assert config.experiment_name == "default"
        assert config.environment == "development"
        assert config.enable_system_metrics is True
        assert config.enable_apple_silicon_metrics is True

    def test_custom_initialization(self):
        """Test configuration with custom values."""
        config = MLFlowConfig(
            tracking_uri="http://mlflow.example.com:5000",
            experiment_name="test-experiment",
            environment="production",
            enable_system_metrics=False,
        )

        assert config.tracking_uri == "http://mlflow.example.com:5000"
        assert config.experiment_name == "test-experiment"
        assert config.environment == "production"
        assert config.enable_system_metrics is False

    def test_environment_overrides(self):
        """Test environment variable overrides."""
        with patch.dict(
            os.environ,
            {
                "MLFLOW_TRACKING_URI": "http://test-server:5000",
                "MLFLOW_EXPERIMENT_NAME": "env-experiment",
                "MLFLOW_ENABLE_SYSTEM_METRICS": "false",
            },
        ):
            config = MLFlowConfig()

            assert config.tracking_uri == "http://test-server:5000"
            assert config.experiment_name == "env-experiment"
            assert config.enable_system_metrics is False

    def test_invalid_environment_type(self):
        """Test validation fails for invalid environment type."""
        with pytest.raises(MLFlowConfigError) as exc_info:
            MLFlowConfig(environment="invalid")

        assert "Invalid environment" in str(exc_info.value)
        assert exc_info.value.environment == "invalid"

    def test_empty_tracking_uri(self):
        """Test validation fails for empty tracking URI."""
        with pytest.raises(MLFlowConfigError) as exc_info:
            MLFlowConfig(tracking_uri="")

        assert "tracking_uri cannot be empty" in str(exc_info.value)

    def test_empty_experiment_name(self):
        """Test validation fails for empty experiment name."""
        with pytest.raises(MLFlowConfigError) as exc_info:
            MLFlowConfig(experiment_name="")

        assert "experiment_name cannot be empty" in str(exc_info.value)

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = MLFlowConfig(
            tracking_uri="http://test:5000",
            experiment_name="test",
            environment="production",
        )

        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert config_dict["tracking_uri"] == "http://test:5000"
        assert config_dict["experiment_name"] == "test"
        assert config_dict["environment"] == "production"
        assert "tags" in config_dict

    def test_from_dict(self):
        """Test creation from dictionary."""
        config_dict = {
            "tracking_uri": "http://test:5000",
            "experiment_name": "test",
            "environment": "production",
            "enable_system_metrics": False,
        }

        config = MLFlowConfig.from_dict(config_dict)

        assert config.tracking_uri == "http://test:5000"
        assert config.experiment_name == "test"
        assert config.environment == "production"
        assert config.enable_system_metrics is False

    def test_from_dict_filters_invalid_keys(self):
        """Test from_dict filters out invalid keys."""
        config_dict = {
            "tracking_uri": "http://test:5000",
            "invalid_key": "should_be_ignored",
            "another_invalid": 123,
        }

        config = MLFlowConfig.from_dict(config_dict)

        assert config.tracking_uri == "http://test:5000"
        assert not hasattr(config, "invalid_key")

    def test_from_environment_development(self):
        """Test creation for development environment."""
        config = MLFlowConfig.from_environment("development")

        assert config.environment == "development"
        assert config.tracking_uri == "http://localhost:5000"
        assert config.backend_store_uri == "sqlite:///mlflow.db"

    def test_from_environment_production(self):
        """Test creation for production environment."""
        config = MLFlowConfig.from_environment("production")

        assert config.environment == "production"

    def test_from_environment_testing(self):
        """Test creation for testing environment."""
        config = MLFlowConfig.from_environment("testing")

        assert config.environment == "testing"
        assert "file://" in config.tracking_uri or "tmp" in config.tracking_uri

    def test_validate_success(self):
        """Test successful validation."""
        config = MLFlowConfig()

        assert config.validate() is True

    def test_validate_failure(self):
        """Test validation failure."""
        config = MLFlowConfig()
        config.tracking_uri = ""  # Invalid

        with pytest.raises(MLFlowConfigError):
            config.validate()

    def test_get_connection_info(self):
        """Test getting connection information."""
        config = MLFlowConfig(
            tracking_uri="http://test:5000",
            registry_uri="http://registry:5000",
        )

        conn_info = config.get_connection_info()

        assert conn_info["tracking_uri"] == "http://test:5000"
        assert conn_info["registry_uri"] == "http://registry:5000"
        assert conn_info["environment"] == "development"

    def test_update_tags(self):
        """Test updating configuration tags."""
        config = MLFlowConfig()
        config.update_tags({"project": "test", "version": "1.0"})

        assert config.tags["project"] == "test"
        assert config.tags["version"] == "1.0"

        # Update existing tag
        config.update_tags({"version": "2.0"})
        assert config.tags["version"] == "2.0"

    def test_get_artifact_path(self):
        """Test getting artifact path."""
        config = MLFlowConfig(artifact_location="./mlruns")

        artifact_path = config.get_artifact_path()
        assert isinstance(artifact_path, Path)
        assert artifact_path == Path("./mlruns")

        # Test with relative path
        relative_artifact_path = config.get_artifact_path("models/model1")
        assert relative_artifact_path == Path("./mlruns/models/model1")

    def test_ensure_artifact_location(self):
        """Test ensuring artifact location directory exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_dir = Path(tmpdir) / "mlruns" / "artifacts"
            config = MLFlowConfig(artifact_location=str(artifact_dir))

            result_path = config.ensure_artifact_location()

            assert result_path.exists()
            assert result_path.is_dir()
            assert result_path == artifact_dir

    def test_production_warnings(self, caplog):
        """Test warnings for production environment with development settings."""
        import logging

        caplog.set_level(logging.WARNING)

        MLFlowConfig(
            environment="production",
            tracking_uri="http://localhost:5000",
            backend_store_uri="sqlite:///mlflow.db",
        )

        assert any("localhost" in record.message for record in caplog.records)
        assert any("SQLite" in record.message for record in caplog.records)

    def test_boolean_environment_variables(self):
        """Test boolean environment variable parsing."""
        test_cases = [
            ("true", True),
            ("True", True),
            ("1", True),
            ("yes", True),
            ("on", True),
            ("false", False),
            ("False", False),
            ("0", False),
            ("no", False),
            ("off", False),
        ]

        for env_value, expected in test_cases:
            with patch.dict(os.environ, {"MLFLOW_ENABLE_SYSTEM_METRICS": env_value}):
                config = MLFlowConfig()
                assert config.enable_system_metrics is expected


class TestHelperFunctions:
    """Test module-level helper functions."""

    def test_get_default_config_development(self):
        """Test getting default development configuration."""
        config = get_default_config("development")

        assert isinstance(config, MLFlowConfig)
        assert config.environment == "development"

    def test_get_default_config_production(self):
        """Test getting default production configuration."""
        config = get_default_config("production")

        assert isinstance(config, MLFlowConfig)
        assert config.environment == "production"

    def test_load_config_from_file_yaml(self):
        """Test loading configuration from YAML file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("""
mlflow:
  tracking_uri: http://custom:5000
  experiment_name: yaml-experiment
  environment: production
  enable_system_metrics: false
            """)
            config_path = Path(f.name)

        try:
            config = load_config_from_file(config_path)

            assert config.tracking_uri == "http://custom:5000"
            assert config.experiment_name == "yaml-experiment"
            assert config.environment == "production"
            assert config.enable_system_metrics is False

        finally:
            config_path.unlink()

    def test_load_config_from_file_json(self):
        """Test loading configuration from JSON file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("""
{
  "mlflow": {
    "tracking_uri": "http://custom:5000",
    "experiment_name": "json-experiment",
    "environment": "production"
  }
}
            """)
            config_path = Path(f.name)

        try:
            config = load_config_from_file(config_path)

            assert config.tracking_uri == "http://custom:5000"
            assert config.experiment_name == "json-experiment"
            assert config.environment == "production"

        finally:
            config_path.unlink()

    def test_load_config_from_file_environment_override(self):
        """Test loading configuration with environment override."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("""
mlflow:
  tracking_uri: http://custom:5000
  experiment_name: test
  environment: development
            """)
            config_path = Path(f.name)

        try:
            config = load_config_from_file(config_path, environment="production")

            assert config.environment == "production"
            assert config.tracking_uri == "http://custom:5000"

        finally:
            config_path.unlink()

    def test_load_config_from_file_invalid_format(self):
        """Test loading configuration with invalid format."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("""
mlflow: "not a dict"
            """)
            config_path = Path(f.name)

        try:
            with pytest.raises(MLFlowConfigError) as exc_info:
                load_config_from_file(config_path)

            assert "Invalid MLFlow configuration format" in str(exc_info.value)

        finally:
            config_path.unlink()

    def test_load_config_from_file_missing_file(self):
        """Test loading configuration from non-existent file."""
        config_path = Path("/non/existent/config.yaml")

        with pytest.raises(MLFlowConfigError):
            load_config_from_file(config_path)


class TestMLFlowConfigError:
    """Test MLFlowConfigError exception class."""

    def test_error_with_environment(self):
        """Test error with environment information."""
        error = MLFlowConfigError("Test error", environment="production")

        assert str(error) == "Test error"
        assert error.environment == "production"
        assert error.details == {}

    def test_error_with_details(self):
        """Test error with details dictionary."""
        details = {"key": "value", "count": 42}
        error = MLFlowConfigError("Test error", details=details)

        assert error.details == details

    def test_error_minimal(self):
        """Test error with minimal information."""
        error = MLFlowConfigError("Test error")

        assert str(error) == "Test error"
        assert error.environment is None
        assert error.details == {}
