"""
Integration tests for MLFlow tracking infrastructure.

This module tests end-to-end integration between MLFlow configuration,
client, and Apple Silicon metrics collection.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mlops.client import MLFlowClient, create_client
from mlops.config import MLFlowConfig
from mlops.tracking import AppleSiliconMetrics, log_metrics_to_mlflow


@pytest.fixture
def mock_mlflow():
    """Mock MLFlow module for integration tests."""
    with patch("mlops.client.mlflow_client.mlflow") as mock:
        # Setup default mock returns
        mock.get_experiment_by_name.return_value = None
        mock.create_experiment.return_value = "integration-test-experiment"

        # Mock active run
        mock_run = MagicMock()
        mock_run.info.run_id = "integration-test-run"
        mock.start_run.return_value = mock_run

        # Mock experiment
        mock_experiment = MagicMock()
        mock_experiment.experiment_id = "integration-test-experiment"
        mock_experiment.name = "integration-test"
        mock_experiment.artifact_location = "./mlruns"
        mock_experiment.lifecycle_stage = "active"
        mock.get_experiment.return_value = mock_experiment

        yield mock


@pytest.fixture
def mock_mlflow_client():
    """Mock MLFlow tracking client for integration tests."""
    with patch("mlops.client.mlflow_client.BaseMlflowClient") as mock:
        mock_instance = MagicMock()
        mock.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def integration_config():
    """Integration test MLFlow configuration."""
    return MLFlowConfig(
        tracking_uri="http://integration-test:5000",
        experiment_name="integration-test",
        environment="testing",
        enable_system_metrics=True,
        enable_apple_silicon_metrics=True,
    )


class TestMLFlowIntegration:
    """Test end-to-end MLFlow integration."""

    def test_end_to_end_experiment_tracking(
        self, mock_mlflow, mock_mlflow_client, integration_config
    ):
        """Test complete experiment tracking flow."""
        client = MLFlowClient(config=integration_config)

        # Start run
        run = client.start_run(run_name="integration-test-run")
        assert run is not None
        assert client.is_active_run

        # Log parameters
        params = {
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 10,
            "model": "transformer",
        }
        client.log_params(params)
        mock_mlflow.log_params.assert_called_once()

        # Log metrics
        metrics = {"loss": 0.42, "accuracy": 0.95, "f1_score": 0.93}
        client.log_metrics(metrics, step=1)
        mock_mlflow.log_metrics.assert_called()

        # Log artifact
        with tempfile.NamedTemporaryFile(delete=False) as f:
            artifact_path = Path(f.name)
            f.write(b"test model data")

        try:
            client.log_artifact(artifact_path)
            mock_mlflow.log_artifact.assert_called()
        finally:
            artifact_path.unlink()

        # Set tags
        client.set_tags({"version": "1.0", "stage": "testing"})
        mock_mlflow.set_tags.assert_called()

        # End run
        client.end_run()
        assert not client.is_active_run
        mock_mlflow.end_run.assert_called_with(status="FINISHED")

    def test_configuration_integration(self, mock_mlflow, mock_mlflow_client):
        """Test configuration integration with client."""
        # Test development environment
        dev_config = MLFlowConfig.from_environment("development")
        dev_client = MLFlowClient(config=dev_config)

        assert dev_client.config.environment == "development"
        assert dev_client.config.enable_system_metrics is True

        # Test production environment
        prod_config = MLFlowConfig.from_environment("production")
        prod_client = MLFlowClient(config=prod_config)

        assert prod_client.config.environment == "production"

    def test_apple_silicon_metrics_integration(
        self, mock_mlflow, mock_mlflow_client, integration_config
    ):
        """Test Apple Silicon metrics integration with MLFlow client."""
        client = MLFlowClient(config=integration_config)

        with client.run(run_name="metrics-test"):
            # Log Apple Silicon metrics
            as_metrics = {
                "mps_utilization": 85.5,
                "memory_used_gb": 12.5,
                "thermal_state": 0,
            }
            client.log_apple_silicon_metrics(as_metrics)

            # Verify metrics were logged with prefix
            mock_mlflow.log_metrics.assert_called()
            call_args = mock_mlflow.log_metrics.call_args[0][0]

            # Check prefixing
            assert any(k.startswith("apple_silicon_") for k in call_args.keys())

    def test_context_manager_integration(
        self, mock_mlflow, mock_mlflow_client, integration_config
    ):
        """Test context manager integration."""
        client = MLFlowClient(config=integration_config)

        # Successful run
        with client.run(run_name="context-test"):
            client.log_params({"param": "value"})
            client.log_metrics({"metric": 1.0})

        mock_mlflow.end_run.assert_called_with(status="FINISHED")

        # Failed run
        with pytest.raises(Exception):
            with client.run(run_name="failed-context-test"):
                raise ValueError("Test error")

        # Verify run was marked as failed
        assert mock_mlflow.end_run.call_args.kwargs["status"] == "FAILED"

    def test_multiple_experiments(self, mock_mlflow, mock_mlflow_client):
        """Test handling multiple experiments."""
        config1 = MLFlowConfig(
            experiment_name="experiment-1", environment="testing"
        )
        config2 = MLFlowConfig(
            experiment_name="experiment-2", environment="testing"
        )

        client1 = MLFlowClient(config=config1)
        client2 = MLFlowClient(config=config2)

        assert client1.config.experiment_name == "experiment-1"
        assert client2.config.experiment_name == "experiment-2"

    def test_artifact_location_integration(
        self, mock_mlflow, mock_mlflow_client, integration_config
    ):
        """Test artifact location handling."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = integration_config
            config.artifact_location = tmpdir

            client = MLFlowClient(config=config)

            # Ensure artifact location
            artifact_path = client.config.ensure_artifact_location()

            assert artifact_path.exists()
            assert artifact_path == Path(tmpdir)

    def test_experiment_info_retrieval(
        self, mock_mlflow, mock_mlflow_client, integration_config
    ):
        """Test retrieving experiment information."""
        client = MLFlowClient(config=integration_config)

        info = client.get_experiment_info()

        assert info["experiment_id"] == "integration-test-experiment"
        assert info["name"] == "integration-test"

    def test_configuration_validation_integration(self, mock_mlflow, mock_mlflow_client):
        """Test configuration validation during client initialization."""
        # Valid configuration
        valid_config = MLFlowConfig(
            tracking_uri="http://test:5000",
            experiment_name="test",
            environment="testing",
        )

        client = MLFlowClient(config=valid_config)
        assert client.config.validate() is True

        # Invalid configuration should fail at config level
        with pytest.raises(Exception):
            invalid_config = MLFlowConfig(
                tracking_uri="",  # Empty URI
                experiment_name="test",
            )

    def test_run_search_integration(
        self, mock_mlflow, mock_mlflow_client, integration_config
    ):
        """Test run search integration."""
        client = MLFlowClient(config=integration_config)

        # Mock search results
        mock_df = MagicMock()
        mock_df.empty = False
        mock_df.to_dict.return_value = [
            {"run_id": "run1", "metrics.loss": 0.5},
            {"run_id": "run2", "metrics.loss": 0.3},
        ]
        mock_mlflow.search_runs.return_value = mock_df

        runs = client.search_runs(filter_string="metrics.loss < 0.6")

        assert len(runs) == 2
        assert runs[0]["run_id"] == "run1"

    def test_model_logging_integration(
        self, mock_mlflow, mock_mlflow_client, integration_config
    ):
        """Test model logging integration."""
        client = MLFlowClient(config=integration_config)

        with client.run(run_name="model-test"):
            # Mock model
            mock_model = MagicMock()

            # Log model
            client.log_model(
                mock_model, artifact_path="model", registered_model_name="test-model"
            )

            mock_mlflow.sklearn.log_model.assert_called_once()

    def test_disabled_features_integration(
        self, mock_mlflow, mock_mlflow_client, integration_config
    ):
        """Test integration with disabled features."""
        config = integration_config
        config.log_params = False
        config.log_metrics = False
        config.log_artifacts = False

        client = MLFlowClient(config=config)

        with client.run(run_name="disabled-features-test"):
            # These should not log anything
            client.log_params({"param": "value"})
            client.log_metrics({"metric": 1.0})

            with tempfile.NamedTemporaryFile() as f:
                client.log_artifact(f.name)

        # Verify nothing was logged
        mock_mlflow.log_params.assert_not_called()
        mock_mlflow.log_metrics.assert_not_called()
        mock_mlflow.log_artifact.assert_not_called()


class TestAppleSiliconMetricsIntegration:
    """Test Apple Silicon metrics integration."""

    @patch("mlops.tracking.apple_silicon_metrics.detect_apple_silicon")
    @patch("mlops.tracking.apple_silicon_metrics.get_chip_type")
    @patch("mlops.tracking.apple_silicon_metrics.get_memory_info")
    @patch("mlops.tracking.apple_silicon_metrics.get_mps_info")
    @patch("mlops.tracking.apple_silicon_metrics.get_ane_info")
    @patch("mlops.tracking.apple_silicon_metrics.get_thermal_state")
    @patch("mlops.tracking.apple_silicon_metrics.get_power_mode")
    @patch("mlops.tracking.apple_silicon_metrics.get_core_info")
    def test_full_metrics_collection_and_logging(
        self,
        mock_core_info,
        mock_power_mode,
        mock_thermal,
        mock_ane,
        mock_mps,
        mock_memory,
        mock_chip,
        mock_detect,
        mock_mlflow,
        mock_mlflow_client,
        integration_config,
    ):
        """Test full Apple Silicon metrics collection and logging."""
        # Setup mocks
        mock_detect.return_value = True
        mock_chip.return_value = "M2 Pro"
        mock_memory.return_value = {
            "total_gb": 32.0,
            "used_gb": 16.0,
            "available_gb": 16.0,
            "utilization_percent": 50.0,
        }
        mock_mps.return_value = {"available": True, "utilization_percent": None}
        mock_ane.return_value = True
        mock_thermal.return_value = 0
        mock_power_mode.return_value = "high_performance"
        mock_core_info.return_value = {"total": 12, "performance": 8, "efficiency": 4}

        # Create client and log metrics
        client = MLFlowClient(config=integration_config)

        with client.run(run_name="full-metrics-test"):
            log_metrics_to_mlflow(client)

            # Verify metrics were logged
            assert mock_mlflow.log_metrics.called

            # Verify tags were set
            assert mock_mlflow.set_tags.call_count >= 2


class TestHelperFunctionIntegration:
    """Test helper function integration."""

    def test_create_client_integration(self, mock_mlflow, mock_mlflow_client):
        """Test create_client helper integration."""
        client = create_client()

        assert isinstance(client, MLFlowClient)
        assert client.config is not None

    def test_create_client_with_config_integration(
        self, mock_mlflow, mock_mlflow_client, integration_config
    ):
        """Test create_client with custom config."""
        client = create_client(config=integration_config)

        assert isinstance(client, MLFlowClient)
        assert client.config == integration_config


class TestErrorHandlingIntegration:
    """Test error handling in integration scenarios."""

    def test_client_initialization_failure(self, mock_mlflow, mock_mlflow_client):
        """Test client initialization with MLFlow connection failure."""
        mock_mlflow.set_tracking_uri.side_effect = Exception("Connection failed")

        with pytest.raises(Exception):
            MLFlowClient()

    def test_run_operations_without_active_run(
        self, mock_mlflow, mock_mlflow_client, integration_config
    ):
        """Test operations without active run."""
        from mlops.client import MLFlowClientError

        client = MLFlowClient(config=integration_config)

        with pytest.raises(MLFlowClientError):
            client.log_params({"param": "value"})

        with pytest.raises(MLFlowClientError):
            client.log_metrics({"metric": 1.0})

    def test_artifact_logging_with_missing_file(
        self, mock_mlflow, mock_mlflow_client, integration_config
    ):
        """Test artifact logging with non-existent file."""
        from mlops.client import MLFlowClientError

        client = MLFlowClient(config=integration_config)

        with client.run(run_name="missing-artifact-test"):
            with pytest.raises(MLFlowClientError):
                client.log_artifact("/non/existent/file.txt")


class TestConfigurationPersistence:
    """Test configuration persistence and loading."""

    def test_config_to_from_dict_integration(self):
        """Test config serialization and deserialization."""
        original_config = MLFlowConfig(
            tracking_uri="http://test:5000",
            experiment_name="test-exp",
            environment="production",
            enable_apple_silicon_metrics=True,
        )

        # Convert to dict
        config_dict = original_config.to_dict()

        # Reconstruct from dict
        loaded_config = MLFlowConfig.from_dict(config_dict)

        assert loaded_config.tracking_uri == original_config.tracking_uri
        assert loaded_config.experiment_name == original_config.experiment_name
        assert loaded_config.environment == original_config.environment
        assert (
            loaded_config.enable_apple_silicon_metrics
            == original_config.enable_apple_silicon_metrics
        )

    def test_config_file_loading_integration(self):
        """Test loading configuration from file."""
        from mlops.config import load_config_from_file

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("""
mlflow:
  tracking_uri: http://file-config:5000
  experiment_name: file-experiment
  environment: production
  enable_apple_silicon_metrics: true
            """)
            config_path = Path(f.name)

        try:
            config = load_config_from_file(config_path)

            assert config.tracking_uri == "http://file-config:5000"
            assert config.experiment_name == "file-experiment"
            assert config.environment == "production"
            assert config.enable_apple_silicon_metrics is True

        finally:
            config_path.unlink()
