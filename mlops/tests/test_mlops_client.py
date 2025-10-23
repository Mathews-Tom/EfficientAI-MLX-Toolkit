"""Tests for unified MLOps client"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pytest

from mlops.client.mlops_client import MLOpsClient, MLOpsClientError, create_client
from mlops.config.dvc_config import DVCConfig
from mlops.config.mlflow_config import MLFlowConfig


@pytest.fixture
def temp_repo():
    """Create temporary repository"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mlflow_config():
    """Create test MLFlow configuration"""
    return MLFlowConfig(
        tracking_uri="http://localhost:5000",
        experiment_name="test-project",
        environment="testing",
    )


@pytest.fixture
def dvc_config():
    """Create test DVC configuration"""
    return DVCConfig(
        storage_backend="local",
        project_namespace="test-project",
        environment="testing",
    )


@pytest.fixture
def mock_mlflow_client():
    """Create mock MLFlow client"""
    with patch("mlops.client.mlops_client.MLFlowClient") as mock:
        client = Mock()
        client.is_active_run = True
        client.run = MagicMock()
        client.log_params = Mock()
        client.log_metrics = Mock()
        client.log_apple_silicon_metrics = Mock()
        client.log_artifact = Mock()
        client.log_model = Mock()
        client.get_experiment_info = Mock(return_value={"experiment_id": "123"})
        mock.return_value = client
        yield mock


@pytest.fixture
def mock_dvc_client():
    """Create mock DVC client"""
    with patch("mlops.client.mlops_client.DVCClient") as mock:
        client = Mock()
        client.add = Mock(return_value={"success": True, "dvc_file": "data.dvc"})
        client.push = Mock(return_value={"success": True})
        client.pull = Mock(return_value={"success": True})
        client.get_connection_info = Mock(return_value={"remote": "test"})
        mock.return_value = client
        yield mock


class TestMLOpsClientInitialization:
    """Test MLOps client initialization"""

    def test_init_with_defaults(self, temp_repo, mock_mlflow_client, mock_dvc_client):
        """Test client initialization with defaults"""
        client = MLOpsClient(
            project_name="test-project",
            repo_root=temp_repo,
        )

        assert client.project_name == "test-project"
        assert client.repo_root == temp_repo
        assert client.workspace_path.exists()
        assert client._mlflow_available
        assert client._dvc_available

    def test_init_with_custom_configs(
        self, temp_repo, mlflow_config, dvc_config, mock_mlflow_client, mock_dvc_client
    ):
        """Test client initialization with custom configurations"""
        client = MLOpsClient(
            project_name="test-project",
            mlflow_config=mlflow_config,
            dvc_config=dvc_config,
            repo_root=temp_repo,
        )

        assert client.project_name == "test-project"
        mock_mlflow_client.assert_called_once()
        mock_dvc_client.assert_called_once()

    def test_from_project(self, temp_repo, mock_mlflow_client, mock_dvc_client):
        """Test creation from project name"""
        client = MLOpsClient.from_project("test-project", repo_root=temp_repo)

        assert client.project_name == "test-project"
        assert isinstance(client, MLOpsClient)

    def test_create_client_function(self, temp_repo, mock_mlflow_client, mock_dvc_client):
        """Test create_client function"""
        client = create_client("test-project", repo_root=temp_repo)

        assert client.project_name == "test-project"
        assert isinstance(client, MLOpsClient)

    def test_init_with_mlflow_failure(self, temp_repo, mock_dvc_client):
        """Test graceful handling of MLFlow initialization failure"""
        with patch("mlops.client.mlops_client.MLFlowClient", side_effect=Exception("MLFlow error")):
            client = MLOpsClient(project_name="test-project", repo_root=temp_repo)

            assert not client._mlflow_available
            assert client._dvc_available

    def test_init_with_dvc_failure(self, temp_repo, mock_mlflow_client):
        """Test graceful handling of DVC initialization failure"""
        with patch("mlops.client.mlops_client.DVCClient", side_effect=Exception("DVC error")):
            client = MLOpsClient(project_name="test-project", repo_root=temp_repo)

            assert client._mlflow_available
            assert not client._dvc_available

    def test_workspace_creation(self, temp_repo, mock_mlflow_client, mock_dvc_client):
        """Test workspace directory creation"""
        workspace = temp_repo / "custom_workspace"
        client = MLOpsClient(
            project_name="test-project",
            repo_root=temp_repo,
            workspace_path=workspace,
        )

        assert client.workspace_path == workspace
        assert workspace.exists()


class TestMLOpsClientExperimentTracking:
    """Test experiment tracking operations"""

    def test_start_run_context_manager(self, temp_repo, mock_mlflow_client, mock_dvc_client):
        """Test start_run as context manager"""
        client = MLOpsClient(project_name="test-project", repo_root=temp_repo)

        with patch.object(client.mlflow_client, "run") as mock_run:
            mock_run.return_value.__enter__ = Mock(return_value="run")
            mock_run.return_value.__exit__ = Mock(return_value=False)

            with client.start_run(run_name="test-run"):
                pass

            mock_run.assert_called_once_with(
                run_name="test-run",
                nested=False,
                tags=None,
                description=None,
            )

    def test_start_run_mlflow_unavailable(self, temp_repo, mock_dvc_client):
        """Test start_run when MLFlow unavailable"""
        with patch("mlops.client.mlops_client.MLFlowClient", side_effect=Exception("MLFlow error")):
            client = MLOpsClient(project_name="test-project", repo_root=temp_repo)

            with pytest.raises(MLOpsClientError, match="MLFlow is not available"):
                with client.start_run():
                    pass

    def test_log_params(self, temp_repo, mock_mlflow_client, mock_dvc_client):
        """Test logging parameters"""
        client = MLOpsClient(project_name="test-project", repo_root=temp_repo)

        params = {"lr": 0.001, "epochs": 10}
        client.log_params(params)

        client.mlflow_client.log_params.assert_called_once_with(params)

    def test_log_params_mlflow_unavailable(self, temp_repo, mock_dvc_client):
        """Test log_params when MLFlow unavailable"""
        with patch("mlops.client.mlops_client.MLFlowClient", side_effect=Exception("MLFlow error")):
            client = MLOpsClient(project_name="test-project", repo_root=temp_repo)

            # Should not raise, just log warning
            client.log_params({"lr": 0.001})

    def test_log_metrics(self, temp_repo, mock_mlflow_client, mock_dvc_client):
        """Test logging metrics"""
        client = MLOpsClient(project_name="test-project", repo_root=temp_repo)

        metrics = {"loss": 0.5, "accuracy": 0.95}
        client.log_metrics(metrics, step=1)

        client.mlflow_client.log_metrics.assert_called_once_with(metrics, step=1)

    def test_log_apple_silicon_metrics(self, temp_repo, mock_mlflow_client, mock_dvc_client):
        """Test logging Apple Silicon metrics"""
        client = MLOpsClient(project_name="test-project", repo_root=temp_repo)

        metrics = {"mps_utilization": 87.5, "memory_gb": 14.2}
        client.log_apple_silicon_metrics(metrics)

        client.mlflow_client.log_apple_silicon_metrics.assert_called_once_with(metrics)

    def test_log_artifact(self, temp_repo, mock_mlflow_client, mock_dvc_client):
        """Test logging artifact"""
        client = MLOpsClient(project_name="test-project", repo_root=temp_repo)

        artifact_path = temp_repo / "model.safetensors"
        artifact_path.touch()

        client.log_artifact(artifact_path, artifact_path="models")

        client.mlflow_client.log_artifact.assert_called_once_with(
            artifact_path, artifact_path="models"
        )

    def test_log_model(self, temp_repo, mock_mlflow_client, mock_dvc_client):
        """Test logging model"""
        client = MLOpsClient(project_name="test-project", repo_root=temp_repo)

        model = Mock()
        client.log_model(model, artifact_path="model", registered_model_name="test-model")

        client.mlflow_client.log_model.assert_called_once()


class TestMLOpsClientDataVersioning:
    """Test data versioning operations"""

    def test_dvc_add(self, temp_repo, mock_mlflow_client, mock_dvc_client):
        """Test adding files to DVC"""
        client = MLOpsClient(project_name="test-project", repo_root=temp_repo)

        data_file = temp_repo / "data.csv"
        data_file.touch()

        result = client.dvc_add(data_file)

        assert result["success"]
        client.dvc_client.add.assert_called_once_with(data_file, recursive=False)

    def test_dvc_add_unavailable(self, temp_repo, mock_mlflow_client):
        """Test dvc_add when DVC unavailable"""
        with patch("mlops.client.mlops_client.DVCClient", side_effect=Exception("DVC error")):
            client = MLOpsClient(project_name="test-project", repo_root=temp_repo)

            with pytest.raises(MLOpsClientError, match="DVC is not available"):
                client.dvc_add("data.csv")

    def test_dvc_push(self, temp_repo, mock_mlflow_client, mock_dvc_client):
        """Test pushing to DVC remote"""
        client = MLOpsClient(project_name="test-project", repo_root=temp_repo)

        result = client.dvc_push()

        assert result["success"]
        client.dvc_client.push.assert_called_once()

    def test_dvc_push_with_targets(self, temp_repo, mock_mlflow_client, mock_dvc_client):
        """Test pushing specific targets"""
        client = MLOpsClient(project_name="test-project", repo_root=temp_repo)

        targets = ["data.csv", "model.pkl"]
        result = client.dvc_push(targets=targets)

        assert result["success"]
        client.dvc_client.push.assert_called_once_with(targets=targets, remote=None)

    def test_dvc_pull(self, temp_repo, mock_mlflow_client, mock_dvc_client):
        """Test pulling from DVC remote"""
        client = MLOpsClient(project_name="test-project", repo_root=temp_repo)

        result = client.dvc_pull()

        assert result["success"]
        client.dvc_client.pull.assert_called_once()

    def test_dvc_pull_with_force(self, temp_repo, mock_mlflow_client, mock_dvc_client):
        """Test force pulling"""
        client = MLOpsClient(project_name="test-project", repo_root=temp_repo)

        result = client.dvc_pull(force=True)

        assert result["success"]
        client.dvc_client.pull.assert_called_once_with(targets=None, remote=None, force=True)


class TestMLOpsClientDeployment:
    """Test model deployment operations"""

    @patch("mlops.client.mlops_client.MLOpsClient._check_bentoml_available", return_value=True)
    def test_deploy_model(
        self, mock_bentoml_check, temp_repo, mock_mlflow_client, mock_dvc_client
    ):
        """Test model deployment"""
        with patch("mlops.serving.bentoml.packager.package_model") as mock_package:
            mock_package.return_value = {
                "success": True,
                "model_tag": "test-model:v1",
            }

            client = MLOpsClient(project_name="test-project", repo_root=temp_repo)

            model_path = temp_repo / "model"
            model_path.mkdir()

            result = client.deploy_model(
                model_path=model_path,
                model_name="test-model",
                model_version="v1",
            )

            assert result["success"]
            assert result["model_tag"] == "test-model:v1"
            mock_package.assert_called_once()

    def test_deploy_model_bentoml_unavailable(
        self, temp_repo, mock_mlflow_client, mock_dvc_client
    ):
        """Test deploy_model when BentoML unavailable"""
        with patch.object(MLOpsClient, "_check_bentoml_available", return_value=False):
            client = MLOpsClient(project_name="test-project", repo_root=temp_repo)

            with pytest.raises(MLOpsClientError, match="BentoML is not available"):
                client.deploy_model(
                    model_path=temp_repo / "model",
                    model_name="test-model",
                )


class TestMLOpsClientMonitoring:
    """Test monitoring operations"""

    @patch("mlops.client.mlops_client.MLOpsClient._check_evidently_available", return_value=True)
    def test_set_reference_data(
        self, mock_evidently_check, temp_repo, mock_mlflow_client, mock_dvc_client
    ):
        """Test setting reference data"""
        with patch("mlops.monitoring.evidently.monitor.create_monitor") as mock_monitor:
            monitor = Mock()
            monitor.set_reference_data = Mock()
            mock_monitor.return_value = monitor

            client = MLOpsClient(project_name="test-project", repo_root=temp_repo)

            reference_data = pd.DataFrame({"feature": [1, 2, 3], "target": [0, 1, 0]})
            client.set_reference_data(reference_data, target_column="target")

            monitor.set_reference_data.assert_called_once()

    @patch("mlops.client.mlops_client.MLOpsClient._check_evidently_available", return_value=True)
    def test_monitor_predictions(
        self, mock_evidently_check, temp_repo, mock_mlflow_client, mock_dvc_client
    ):
        """Test monitoring predictions"""
        with patch("mlops.monitoring.evidently.monitor.create_monitor") as mock_monitor:
            monitor = Mock()
            monitor.monitor = Mock(
                return_value={
                    "drift_report": {"dataset_drift": True, "drift_share": 0.6},
                    "performance_metrics": {"degraded": False},
                }
            )
            mock_monitor.return_value = monitor

            client = MLOpsClient(project_name="test-project", repo_root=temp_repo)

            current_data = pd.DataFrame({"feature": [4, 5, 6], "target": [1, 1, 0]})
            results = client.monitor_predictions(current_data, target_column="target")

            assert "drift_report" in results
            assert results["drift_report"]["dataset_drift"]
            monitor.monitor.assert_called_once()

    def test_monitor_predictions_evidently_unavailable(
        self, temp_repo, mock_mlflow_client, mock_dvc_client
    ):
        """Test monitor_predictions when Evidently unavailable"""
        with patch.object(MLOpsClient, "_check_evidently_available", return_value=False):
            client = MLOpsClient(project_name="test-project", repo_root=temp_repo)

            current_data = pd.DataFrame({"feature": [1, 2, 3]})
            results = client.monitor_predictions(current_data)

            assert not results.get("monitoring_available", True)


class TestMLOpsClientWorkspace:
    """Test workspace management"""

    def test_get_workspace_path(self, temp_repo, mock_mlflow_client, mock_dvc_client):
        """Test getting workspace path"""
        client = MLOpsClient(project_name="test-project", repo_root=temp_repo)

        workspace = client.get_workspace_path()
        assert workspace.exists()
        assert "test-project" in str(workspace)

    def test_get_workspace_path_with_subdir(
        self, temp_repo, mock_mlflow_client, mock_dvc_client
    ):
        """Test getting workspace subdirectory"""
        client = MLOpsClient(project_name="test-project", repo_root=temp_repo)

        subdir = client.get_workspace_path("models")
        assert subdir.exists()
        assert "models" in str(subdir)

    def test_get_status(self, temp_repo, mock_mlflow_client, mock_dvc_client):
        """Test getting client status"""
        client = MLOpsClient(project_name="test-project", repo_root=temp_repo)

        status = client.get_status()

        assert status["project_name"] == "test-project"
        assert status["mlflow_available"]
        assert status["dvc_available"
]
        assert "workspace_path" in status

    def test_get_status_with_component_info(
        self, temp_repo, mock_mlflow_client, mock_dvc_client
    ):
        """Test status includes component information"""
        client = MLOpsClient(project_name="test-project", repo_root=temp_repo)

        status = client.get_status()

        assert "mlflow_experiment" in status
        assert "dvc_connection" in status


class TestMLOpsClientIntegration:
    """Integration tests for MLOps client"""

    def test_full_workflow(self, temp_repo, mock_mlflow_client, mock_dvc_client):
        """Test complete MLOps workflow"""
        client = MLOpsClient.from_project("test-project", repo_root=temp_repo)

        # Mock context manager
        with patch.object(client.mlflow_client, "run") as mock_run:
            mock_run.return_value.__enter__ = Mock(return_value="run")
            mock_run.return_value.__exit__ = Mock(return_value=False)

            # Start experiment
            with client.start_run(run_name="integration-test"):
                # Log parameters
                client.log_params({"lr": 0.001, "epochs": 10})

                # Log metrics
                client.log_metrics({"loss": 0.5, "accuracy": 0.95})

                # Log Apple Silicon metrics
                client.log_apple_silicon_metrics({"mps_utilization": 87.5})

            # Version data
            data_file = temp_repo / "data.csv"
            data_file.touch()
            client.dvc_add(data_file)
            client.dvc_push()

            # Get status
            status = client.get_status()
            assert status["project_name"] == "test-project"

    def test_error_handling(self, temp_repo, mock_mlflow_client, mock_dvc_client):
        """Test error handling across components"""
        client = MLOpsClient(project_name="test-project", repo_root=temp_repo)

        # Mock MLFlow error - wrap in MLFlowClientError
        from mlops.client.mlflow_client import MLFlowClientError as MLFlowError

        with patch.object(
            client.mlflow_client, "log_params", side_effect=MLFlowError("MLFlow error", operation="log_params")
        ):
            with pytest.raises(MLOpsClientError, match="Failed to log parameters"):
                client.log_params({"lr": 0.001})

        # Mock DVC error - wrap in DVCClientError
        from mlops.client.dvc_client import DVCClientError as DVCError

        with patch.object(client.dvc_client, "add", side_effect=DVCError("DVC error", operation="add")):
            with pytest.raises(MLOpsClientError, match="Failed to add to DVC"):
                client.dvc_add("nonexistent.csv")


class TestMLOpsClientAvailabilityChecks:
    """Test component availability checks"""

    def test_check_bentoml_available_true(self):
        """Test BentoML availability check when available"""
        with patch.dict("sys.modules", {"bentoml": Mock()}):
            assert MLOpsClient._check_bentoml_available()

    def test_check_bentoml_available_false(self):
        """Test BentoML availability check when unavailable"""
        with patch.dict("sys.modules", {"bentoml": None}):
            with patch("builtins.__import__", side_effect=ImportError):
                assert not MLOpsClient._check_bentoml_available()

    def test_check_evidently_available_true(self):
        """Test Evidently availability check when available"""
        with patch.dict("sys.modules", {"evidently": Mock()}):
            assert MLOpsClient._check_evidently_available()

    def test_check_evidently_available_false(self):
        """Test Evidently availability check when unavailable"""
        with patch.dict("sys.modules", {"evidently": None}):
            with patch("builtins.__import__", side_effect=ImportError):
                assert not MLOpsClient._check_evidently_available()
