"""
Tests for MLFlow client API.

This module tests the MLFlowClient class with mocked MLFlow server.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from mlops.client import MLFlowClient, MLFlowClientError, create_client
from mlops.config import MLFlowConfig


@pytest.fixture
def mock_mlflow():
    """Mock MLFlow module."""
    with patch("mlops.client.mlflow_client.mlflow") as mock:
        # Setup default mock returns
        mock.get_experiment_by_name.return_value = None
        mock.create_experiment.return_value = "test-experiment-id"

        # Mock active run
        mock_run = MagicMock()
        mock_run.info.run_id = "test-run-id"
        mock.start_run.return_value = mock_run

        # Mock experiment
        mock_experiment = MagicMock()
        mock_experiment.experiment_id = "test-experiment-id"
        mock_experiment.name = "test-experiment"
        mock_experiment.artifact_location = "./mlruns"
        mock_experiment.lifecycle_stage = "active"
        mock.get_experiment.return_value = mock_experiment

        yield mock


@pytest.fixture
def mock_mlflow_client():
    """Mock MLFlow tracking client."""
    with patch("mlops.client.mlflow_client.BaseMlflowClient") as mock:
        mock_instance = MagicMock()
        mock.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def test_config():
    """Test MLFlow configuration."""
    return MLFlowConfig(
        tracking_uri="http://test:5000",
        experiment_name="test-experiment",
        environment="testing",
    )


class TestMLFlowClient:
    """Test MLFlowClient class."""

    def test_initialization_default_config(self, mock_mlflow, mock_mlflow_client):
        """Test client initialization with default config."""
        client = MLFlowClient()

        assert client.config is not None
        assert client.experiment_id == "test-experiment-id"
        mock_mlflow.set_tracking_uri.assert_called_once()

    def test_initialization_custom_config(self, mock_mlflow, mock_mlflow_client, test_config):
        """Test client initialization with custom config."""
        client = MLFlowClient(config=test_config)

        assert client.config == test_config
        assert client.config.experiment_name == "test-experiment"
        mock_mlflow.set_tracking_uri.assert_called_with("http://test:5000")

    def test_initialization_creates_new_experiment(self, mock_mlflow, mock_mlflow_client, test_config):
        """Test initialization creates experiment if not exists."""
        mock_mlflow.get_experiment_by_name.return_value = None

        client = MLFlowClient(config=test_config)

        mock_mlflow.create_experiment.assert_called_once()
        assert client.experiment_id == "test-experiment-id"

    def test_initialization_uses_existing_experiment(self, mock_mlflow, mock_mlflow_client, test_config):
        """Test initialization uses existing experiment."""
        mock_experiment = MagicMock()
        mock_experiment.experiment_id = "existing-id"
        mock_mlflow.get_experiment_by_name.return_value = mock_experiment

        client = MLFlowClient(config=test_config)

        mock_mlflow.create_experiment.assert_not_called()
        assert client.experiment_id == "existing-id"

    def test_client_property(self, mock_mlflow, mock_mlflow_client, test_config):
        """Test client property returns underlying client."""
        client = MLFlowClient(config=test_config)

        assert client.client is not None

    def test_experiment_id_property(self, mock_mlflow, mock_mlflow_client, test_config):
        """Test experiment_id property."""
        client = MLFlowClient(config=test_config)

        assert client.experiment_id == "test-experiment-id"

    def test_run_id_property_no_active_run(self, mock_mlflow, mock_mlflow_client, test_config):
        """Test run_id property when no active run."""
        client = MLFlowClient(config=test_config)

        assert client.run_id is None
        assert not client.is_active_run

    def test_start_run(self, mock_mlflow, mock_mlflow_client, test_config):
        """Test starting a new run."""
        client = MLFlowClient(config=test_config)

        run = client.start_run(run_name="test-run", tags={"key": "value"})

        assert run is not None
        assert client.is_active_run
        assert client.run_id == "test-run-id"
        mock_mlflow.start_run.assert_called_once()

    def test_start_run_with_description(self, mock_mlflow, mock_mlflow_client, test_config):
        """Test starting run with description."""
        client = MLFlowClient(config=test_config)

        client.start_run(run_name="test-run", description="Test description")

        # Check that description was added as note tag
        call_kwargs = mock_mlflow.start_run.call_args.kwargs
        assert "mlflow.note.content" in call_kwargs["tags"]

    def test_end_run(self, mock_mlflow, mock_mlflow_client, test_config):
        """Test ending a run."""
        client = MLFlowClient(config=test_config)
        client.start_run(run_name="test-run")

        client.end_run(status="FINISHED")

        assert not client.is_active_run
        assert client.run_id is None
        mock_mlflow.end_run.assert_called_with(status="FINISHED")

    def test_end_run_no_active_run(self, mock_mlflow, mock_mlflow_client, test_config):
        """Test ending run when no active run."""
        client = MLFlowClient(config=test_config)

        # Should not raise, just log warning
        client.end_run()

    def test_run_context_manager_success(self, mock_mlflow, mock_mlflow_client, test_config):
        """Test run context manager with successful completion."""
        client = MLFlowClient(config=test_config)

        with client.run(run_name="test-run"):
            assert client.is_active_run

        assert not client.is_active_run
        mock_mlflow.end_run.assert_called_with(status="FINISHED")

    def test_run_context_manager_failure(self, mock_mlflow, mock_mlflow_client, test_config):
        """Test run context manager with exception."""
        client = MLFlowClient(config=test_config)

        with pytest.raises(MLFlowClientError):
            with client.run(run_name="test-run"):
                raise ValueError("Test error")

        mock_mlflow.end_run.assert_called_with(status="FAILED")

    def test_log_params(self, mock_mlflow, mock_mlflow_client, test_config):
        """Test logging parameters."""
        client = MLFlowClient(config=test_config)
        client.start_run(run_name="test-run")

        params = {"lr": 0.001, "epochs": 10, "model": "test"}
        client.log_params(params)

        mock_mlflow.log_params.assert_called_once()
        # Check that values are converted to strings
        call_args = mock_mlflow.log_params.call_args[0][0]
        assert all(isinstance(v, str) for v in call_args.values())

    def test_log_params_disabled(self, mock_mlflow, mock_mlflow_client, test_config):
        """Test logging parameters when disabled."""
        test_config.log_params = False
        client = MLFlowClient(config=test_config)
        client.start_run(run_name="test-run")

        client.log_params({"lr": 0.001})

        mock_mlflow.log_params.assert_not_called()

    def test_log_params_no_active_run(self, mock_mlflow, mock_mlflow_client, test_config):
        """Test logging parameters without active run."""
        client = MLFlowClient(config=test_config)

        with pytest.raises(MLFlowClientError) as exc_info:
            client.log_params({"lr": 0.001})

        assert "No active run" in str(exc_info.value)

    def test_log_param(self, mock_mlflow, mock_mlflow_client, test_config):
        """Test logging single parameter."""
        client = MLFlowClient(config=test_config)
        client.start_run(run_name="test-run")

        client.log_param("lr", 0.001)

        mock_mlflow.log_params.assert_called_once()

    def test_log_metrics(self, mock_mlflow, mock_mlflow_client, test_config):
        """Test logging metrics."""
        client = MLFlowClient(config=test_config)
        client.start_run(run_name="test-run")

        metrics = {"loss": 0.5, "accuracy": 0.95}
        client.log_metrics(metrics, step=1)

        mock_mlflow.log_metrics.assert_called_once()
        call_args = mock_mlflow.log_metrics.call_args
        assert call_args.kwargs["step"] == 1

    def test_log_metrics_no_step(self, mock_mlflow, mock_mlflow_client, test_config):
        """Test logging metrics without step."""
        client = MLFlowClient(config=test_config)
        client.start_run(run_name="test-run")

        client.log_metrics({"loss": 0.5})

        mock_mlflow.log_metrics.assert_called_once()
        call_args = mock_mlflow.log_metrics.call_args
        assert "step" not in call_args.kwargs or call_args.kwargs.get("step") is None

    def test_log_metrics_disabled(self, mock_mlflow, mock_mlflow_client, test_config):
        """Test logging metrics when disabled."""
        test_config.log_metrics = False
        client = MLFlowClient(config=test_config)
        client.start_run(run_name="test-run")

        client.log_metrics({"loss": 0.5})

        mock_mlflow.log_metrics.assert_not_called()

    def test_log_metric(self, mock_mlflow, mock_mlflow_client, test_config):
        """Test logging single metric."""
        client = MLFlowClient(config=test_config)
        client.start_run(run_name="test-run")

        client.log_metric("loss", 0.5, step=1)

        mock_mlflow.log_metrics.assert_called_once()

    def test_log_artifact(self, mock_mlflow, mock_mlflow_client, test_config):
        """Test logging artifact file."""
        client = MLFlowClient(config=test_config)
        client.start_run(run_name="test-run")

        with tempfile.NamedTemporaryFile(delete=False) as f:
            artifact_path = Path(f.name)
            f.write(b"test content")

        try:
            client.log_artifact(artifact_path, artifact_path="models")

            mock_mlflow.log_artifact.assert_called_once()
            call_args = mock_mlflow.log_artifact.call_args
            assert call_args.kwargs["artifact_path"] == "models"

        finally:
            artifact_path.unlink()

    def test_log_artifact_file_not_found(self, mock_mlflow, mock_mlflow_client, test_config):
        """Test logging non-existent artifact."""
        client = MLFlowClient(config=test_config)
        client.start_run(run_name="test-run")

        with pytest.raises(MLFlowClientError) as exc_info:
            client.log_artifact("/non/existent/file.txt")

        assert "not found" in str(exc_info.value).lower()

    def test_log_artifact_disabled(self, mock_mlflow, mock_mlflow_client, test_config):
        """Test logging artifact when disabled."""
        test_config.log_artifacts = False
        client = MLFlowClient(config=test_config)
        client.start_run(run_name="test-run")

        with tempfile.NamedTemporaryFile() as f:
            client.log_artifact(f.name)

        mock_mlflow.log_artifact.assert_not_called()

    def test_log_artifacts(self, mock_mlflow, mock_mlflow_client, test_config):
        """Test logging artifacts directory."""
        client = MLFlowClient(config=test_config)
        client.start_run(run_name="test-run")

        with tempfile.TemporaryDirectory() as tmpdir:
            client.log_artifacts(tmpdir, artifact_path="outputs")

            mock_mlflow.log_artifacts.assert_called_once()

    def test_log_artifacts_directory_not_found(self, mock_mlflow, mock_mlflow_client, test_config):
        """Test logging non-existent artifacts directory."""
        client = MLFlowClient(config=test_config)
        client.start_run(run_name="test-run")

        with pytest.raises(MLFlowClientError) as exc_info:
            client.log_artifacts("/non/existent/directory")

        assert "not found" in str(exc_info.value).lower()

    def test_log_model(self, mock_mlflow, mock_mlflow_client, test_config):
        """Test logging model."""
        client = MLFlowClient(config=test_config)
        client.start_run(run_name="test-run")

        mock_model = MagicMock()

        client.log_model(mock_model, artifact_path="model", registered_model_name="test-model")

        mock_mlflow.sklearn.log_model.assert_called_once()

    def test_log_model_disabled(self, mock_mlflow, mock_mlflow_client, test_config):
        """Test logging model when disabled."""
        test_config.log_models = False
        client = MLFlowClient(config=test_config)
        client.start_run(run_name="test-run")

        mock_model = MagicMock()
        client.log_model(mock_model, artifact_path="model")

        mock_mlflow.sklearn.log_model.assert_not_called()

    def test_set_tags(self, mock_mlflow, mock_mlflow_client, test_config):
        """Test setting tags."""
        client = MLFlowClient(config=test_config)
        client.start_run(run_name="test-run")

        tags = {"key1": "value1", "key2": "value2"}
        client.set_tags(tags)

        mock_mlflow.set_tags.assert_called_once_with(tags)

    def test_set_tag(self, mock_mlflow, mock_mlflow_client, test_config):
        """Test setting single tag."""
        client = MLFlowClient(config=test_config)
        client.start_run(run_name="test-run")

        client.set_tag("key", "value")

        mock_mlflow.set_tags.assert_called_once()

    def test_get_run_current(self, mock_mlflow, mock_mlflow_client, test_config):
        """Test getting current run."""
        client = MLFlowClient(config=test_config)
        client.start_run(run_name="test-run")

        mock_mlflow_client.get_run.return_value = MagicMock()

        run = client.get_run()

        assert run is not None
        mock_mlflow_client.get_run.assert_called_once_with("test-run-id")

    def test_get_run_specific_id(self, mock_mlflow, mock_mlflow_client, test_config):
        """Test getting specific run by ID."""
        client = MLFlowClient(config=test_config)

        mock_mlflow_client.get_run.return_value = MagicMock()

        run = client.get_run(run_id="other-run-id")

        assert run is not None
        mock_mlflow_client.get_run.assert_called_once_with("other-run-id")

    def test_get_run_no_run_id(self, mock_mlflow, mock_mlflow_client, test_config):
        """Test getting run without run ID."""
        client = MLFlowClient(config=test_config)

        with pytest.raises(MLFlowClientError) as exc_info:
            client.get_run()

        assert "No run ID" in str(exc_info.value)

    def test_search_runs(self, mock_mlflow, mock_mlflow_client, test_config):
        """Test searching runs."""
        client = MLFlowClient(config=test_config)

        mock_df = MagicMock()
        mock_df.empty = False
        mock_df.to_dict.return_value = [{"run_id": "run1"}, {"run_id": "run2"}]
        mock_mlflow.search_runs.return_value = mock_df

        runs = client.search_runs(filter_string="metrics.loss < 0.5", max_results=10)

        assert len(runs) == 2
        mock_mlflow.search_runs.assert_called_once()

    def test_search_runs_empty(self, mock_mlflow, mock_mlflow_client, test_config):
        """Test searching runs with no results."""
        client = MLFlowClient(config=test_config)

        mock_df = MagicMock()
        mock_df.empty = True
        mock_mlflow.search_runs.return_value = mock_df

        runs = client.search_runs()

        assert runs == []

    def test_log_apple_silicon_metrics(self, mock_mlflow, mock_mlflow_client, test_config):
        """Test logging Apple Silicon metrics."""
        client = MLFlowClient(config=test_config)
        client.start_run(run_name="test-run")

        metrics = {"mps_util": 85.5, "thermal_state": 0}
        client.log_apple_silicon_metrics(metrics)

        # Check that metrics were logged with prefix
        mock_mlflow.log_metrics.assert_called_once()
        call_args = mock_mlflow.log_metrics.call_args[0][0]
        assert "apple_silicon_mps_util" in call_args
        assert "apple_silicon_thermal_state" in call_args

    def test_log_apple_silicon_metrics_disabled(self, mock_mlflow, mock_mlflow_client, test_config):
        """Test logging Apple Silicon metrics when disabled."""
        test_config.enable_apple_silicon_metrics = False
        client = MLFlowClient(config=test_config)
        client.start_run(run_name="test-run")

        client.log_apple_silicon_metrics({"mps_util": 85.5})

        mock_mlflow.log_metrics.assert_not_called()

    def test_get_experiment_info(self, mock_mlflow, mock_mlflow_client, test_config):
        """Test getting experiment information."""
        client = MLFlowClient(config=test_config)

        info = client.get_experiment_info()

        assert info["experiment_id"] == "test-experiment-id"
        assert info["name"] == "test-experiment"
        assert info["lifecycle_stage"] == "active"


class TestMLFlowClientError:
    """Test MLFlowClientError exception class."""

    def test_error_with_operation(self):
        """Test error with operation information."""
        error = MLFlowClientError("Test error", operation="test_operation")

        assert str(error) == "Test error"
        assert error.operation == "test_operation"
        assert error.details == {}

    def test_error_with_details(self):
        """Test error with details dictionary."""
        details = {"key": "value", "count": 42}
        error = MLFlowClientError("Test error", details=details)

        assert error.details == details

    def test_error_minimal(self):
        """Test error with minimal information."""
        error = MLFlowClientError("Test error")

        assert str(error) == "Test error"
        assert error.operation is None
        assert error.details == {}


class TestHelperFunctions:
    """Test module-level helper functions."""

    def test_create_client_default(self, mock_mlflow, mock_mlflow_client):
        """Test creating client with default config."""
        client = create_client()

        assert isinstance(client, MLFlowClient)
        assert client.config is not None

    def test_create_client_custom_config(self, mock_mlflow, mock_mlflow_client, test_config):
        """Test creating client with custom config."""
        client = create_client(config=test_config)

        assert isinstance(client, MLFlowClient)
        assert client.config == test_config
