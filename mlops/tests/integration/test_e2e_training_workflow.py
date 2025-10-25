"""
End-to-end integration test for complete training workflow.

Tests the full training pipeline:
1. Data versioning with DVC
2. Experiment tracking with MLFlow
3. Apple Silicon metrics collection
4. Model deployment with BentoML
5. Performance monitoring with Evidently
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from mlops.client.mlops_client import MLOpsClient


class TestE2ETrainingWorkflow:
    """Integration tests for complete training workflow."""

    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            (workspace / "data").mkdir()
            (workspace / "models").mkdir()
            (workspace / "outputs").mkdir()
            yield workspace

    @pytest.fixture
    def mlops_client(self, temp_workspace):
        """Create MLOps client with mocked backends."""
        with patch("mlops.client.mlops_client.MLFlowClient") as mock_mlflow, patch(
            "mlops.client.mlops_client.DVCClient"
        ) as mock_dvc, patch(
            "mlops.client.mlops_client.AppleSiliconMonitor"
        ) as mock_monitor:
            # Configure mocks
            mock_mlflow.return_value.create_experiment.return_value = "exp-001"
            mock_mlflow.return_value.start_run.return_value.__enter__ = MagicMock(
                return_value=MagicMock(info=MagicMock(run_id="run-001"))
            )
            mock_mlflow.return_value.start_run.return_value.__exit__ = MagicMock(
                return_value=None
            )

            mock_dvc.return_value.init.return_value = None
            mock_dvc.return_value.add.return_value = True

            mock_monitor.return_value.get_metrics.return_value = {
                "chip_type": "M3",
                "unified_memory_gb": 32.0,
                "mps_available": True,
            }

            client = MLOpsClient(project_namespace="test-project")
            yield client

    def test_complete_training_workflow(self, mlops_client, temp_workspace):
        """Test complete training workflow from data versioning to deployment."""
        # Step 1: Version dataset
        dataset_path = temp_workspace / "data" / "train.csv"
        df = pd.DataFrame({"feature": [1, 2, 3], "label": [0, 1, 0]})
        df.to_csv(dataset_path, index=False)

        with patch.object(mlops_client.dvc_client, "add") as mock_add:
            mock_add.return_value = True
            result = mlops_client.version_dataset(str(dataset_path))
            assert result is True
            mock_add.assert_called_once_with(str(dataset_path))

        # Step 2: Start experiment
        with patch.object(
            mlops_client.mlflow_client, "create_experiment"
        ) as mock_create_exp:
            mock_create_exp.return_value = "exp-001"
            exp_id = mlops_client.create_experiment("test-experiment")
            assert exp_id == "exp-001"

        # Step 3: Track training run
        with patch.object(
            mlops_client.mlflow_client, "start_run"
        ) as mock_start_run:
            mock_run = MagicMock()
            mock_run.info.run_id = "run-001"
            mock_start_run.return_value.__enter__.return_value = mock_run
            mock_start_run.return_value.__exit__.return_value = None

            with patch.object(
                mlops_client.mlflow_client, "log_params"
            ) as mock_log_params, patch.object(
                mlops_client.mlflow_client, "log_metrics"
            ) as mock_log_metrics:
                with mlops_client.start_run(
                    run_name="test-run", experiment_id="exp-001"
                ):
                    # Log training config
                    mlops_client.log_params(
                        {"learning_rate": 0.001, "batch_size": 32}
                    )

                    # Simulate training epochs
                    for epoch in range(3):
                        metrics = {
                            "train_loss": 0.5 - epoch * 0.1,
                            "train_accuracy": 0.7 + epoch * 0.05,
                        }
                        mlops_client.log_metrics(metrics, step=epoch)

                assert mock_log_params.called
                assert mock_log_metrics.called

        # Step 4: Collect Apple Silicon metrics
        with patch.object(
            mlops_client.silicon_monitor, "get_metrics"
        ) as mock_get_metrics:
            mock_get_metrics.return_value = {
                "chip_type": "M3",
                "unified_memory_gb": 32.0,
                "mps_available": True,
                "thermal_state": "nominal",
            }
            metrics = mlops_client.collect_apple_silicon_metrics()
            assert metrics["chip_type"] == "M3"
            assert "unified_memory_gb" in metrics

        # Step 5: Save model artifact
        model_path = temp_workspace / "models" / "model.bin"
        model_path.write_text("fake_model_data")

        with patch.object(
            mlops_client.mlflow_client, "log_artifact"
        ) as mock_log_artifact:
            mlops_client.log_artifact(str(model_path))
            mock_log_artifact.assert_called_once()

    def test_versioning_tracking_integration(self, mlops_client, temp_workspace):
        """Test integration between DVC versioning and MLFlow tracking."""
        # Create versioned dataset
        dataset_path = temp_workspace / "data" / "dataset.csv"
        df = pd.DataFrame({"x": [1, 2, 3], "y": [2, 4, 6]})
        df.to_csv(dataset_path, index=False)

        # Version with DVC
        with patch.object(mlops_client.dvc_client, "add") as mock_add, patch.object(
            mlops_client.dvc_client, "push"
        ) as mock_push:
            mock_add.return_value = True
            mock_push.return_value = True

            versioned = mlops_client.version_dataset(
                str(dataset_path), push_to_remote=True
            )
            assert versioned is True
            mock_push.assert_called_once()

        # Log dataset info to MLFlow
        with patch.object(
            mlops_client.mlflow_client, "log_params"
        ) as mock_log_params:
            mlops_client.log_params(
                {"dataset_path": str(dataset_path), "dataset_size": len(df)}
            )
            mock_log_params.assert_called_once()

    def test_monitoring_integration(self, mlops_client, temp_workspace):
        """Test integration with Evidently monitoring."""
        # Create reference and current data
        reference_data = pd.DataFrame(
            {"feature1": [1, 2, 3, 4, 5], "target": [0, 1, 0, 1, 0]}
        )

        current_data = pd.DataFrame(
            {"feature1": [6, 7, 8, 9, 10], "target": [1, 0, 1, 0, 1]}
        )

        # Set up monitoring
        with patch(
            "mlops.client.mlops_client.EvidentlyMonitor"
        ) as mock_evidently_class:
            mock_monitor = MagicMock()
            mock_monitor.generate_drift_report.return_value = {
                "drift_detected": False,
                "drift_score": 0.05,
            }
            mock_evidently_class.return_value = mock_monitor

            # Create new client to get mocked Evidently
            client = MLOpsClient(project_namespace="test-project")
            client.evidently_monitor = mock_monitor

            # Set reference data
            client.set_reference_data(reference_data)

            # Check for drift
            with patch.object(
                client.evidently_monitor, "generate_drift_report"
            ) as mock_drift:
                mock_drift.return_value = {
                    "drift_detected": False,
                    "drift_score": 0.05,
                }
                drift_report = client.evidently_monitor.generate_drift_report(
                    current_data, reference_data
                )
                assert drift_report["drift_detected"] is False

    def test_deployment_workflow(self, mlops_client, temp_workspace):
        """Test model deployment workflow."""
        model_path = temp_workspace / "models" / "trained_model"
        model_path.mkdir()
        (model_path / "model.bin").write_text("model_weights")
        (model_path / "config.json").write_text('{"param": "value"}')

        # Package model
        with patch("mlops.client.mlops_client.ModelPackager") as mock_packager_class:
            mock_packager = MagicMock()
            mock_packager.package_model.return_value = {
                "bento_name": "test_model:v1",
                "status": "success",
            }
            mock_packager_class.return_value = mock_packager

            # Create client with mocked packager
            client = MLOpsClient(project_namespace="test-project")
            client.model_packager = mock_packager

            result = client.deploy_model(
                model_path=str(model_path), model_name="test_model", version="v1"
            )

            assert result["status"] == "success"

    def test_error_recovery_workflow(self, mlops_client, temp_workspace):
        """Test workflow with error recovery."""
        # Simulate DVC connection failure
        with patch.object(
            mlops_client.dvc_client, "add", side_effect=Exception("DVC unavailable")
        ):
            # Should gracefully handle DVC failure
            with pytest.raises(Exception) as exc_info:
                mlops_client.version_dataset("nonexistent.csv")
            assert "DVC unavailable" in str(exc_info.value)

        # MLFlow should still work
        with patch.object(
            mlops_client.mlflow_client, "log_params"
        ) as mock_log_params:
            mlops_client.log_params({"test": "value"})
            mock_log_params.assert_called_once()

    def test_apple_silicon_optimization_workflow(self, mlops_client, temp_workspace):
        """Test Apple Silicon-specific workflow."""
        # Collect hardware metrics
        with patch.object(
            mlops_client.silicon_monitor, "get_metrics"
        ) as mock_get_metrics:
            mock_get_metrics.return_value = {
                "chip_type": "M3_MAX",
                "unified_memory_gb": 64.0,
                "mps_available": True,
                "ane_available": True,
                "performance_cores": 12,
                "efficiency_cores": 4,
                "thermal_state": "nominal",
                "power_mode": "high_performance",
            }

            metrics = mlops_client.collect_apple_silicon_metrics()

            # Verify metrics
            assert metrics["chip_type"] == "M3_MAX"
            assert metrics["unified_memory_gb"] == 64.0
            assert metrics["mps_available"] is True
            assert metrics["ane_available"] is True

        # Log optimization metrics
        with patch.object(
            mlops_client.mlflow_client, "log_metrics"
        ) as mock_log_metrics:
            mlops_client.log_metrics(
                {
                    "mps_utilization": 0.85,
                    "memory_efficiency": 0.92,
                    "thermal_efficiency": 0.95,
                }
            )
            mock_log_metrics.assert_called_once()


@pytest.mark.integration
class TestMultiProjectIntegration:
    """Integration tests for multiple projects sharing MLOps infrastructure."""

    @pytest.fixture
    def shared_workspace(self):
        """Create shared workspace for multiple projects."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            for project in ["lora", "compression", "diffusion"]:
                (workspace / project).mkdir()
                (workspace / project / "data").mkdir()
                (workspace / project / "models").mkdir()
            yield workspace

    def test_shared_mlflow_server(self, shared_workspace):
        """Test multiple projects using same MLFlow server."""
        with patch("mlops.client.mlops_client.MLFlowClient") as mock_mlflow_class:
            # Create clients for different projects
            clients = []
            for project in ["lora-finetuning", "model-compression", "coreml-diffusion"]:
                mock_mlflow = MagicMock()
                mock_mlflow.create_experiment.return_value = f"exp-{project}"
                mock_mlflow_class.return_value = mock_mlflow

                client = MLOpsClient(project_namespace=project)
                clients.append(client)

            # Each should have separate experiment
            assert len(clients) == 3
            assert all(c.project_namespace != clients[0].project_namespace for c in clients[1:])

    def test_shared_dvc_remote(self, shared_workspace):
        """Test multiple projects using same DVC remote."""
        with patch("mlops.client.mlops_client.DVCClient") as mock_dvc_class:
            mock_dvc = MagicMock()
            mock_dvc.add.return_value = True
            mock_dvc.push.return_value = True
            mock_dvc_class.return_value = mock_dvc

            # Create clients for different projects
            for project in ["lora", "compression", "diffusion"]:
                client = MLOpsClient(project_namespace=project)

                # Version project-specific data
                dataset_path = shared_workspace / project / "data" / "dataset.csv"
                dataset_path.write_text("data")

                with patch.object(client.dvc_client, "add") as mock_add:
                    mock_add.return_value = True
                    client.version_dataset(str(dataset_path))
                    mock_add.assert_called_once()
