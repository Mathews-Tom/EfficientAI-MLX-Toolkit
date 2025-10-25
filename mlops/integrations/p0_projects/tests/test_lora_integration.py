"""Tests for LoRA Fine-tuning MLOps Integration"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pytest

from mlops.integrations.p0_projects.lora_finetuning.integration import (
    LoRAMLOpsTracker,
    create_lora_mlops_client,
)


@pytest.fixture
def mock_mlops_client():
    """Create mock MLOps client"""
    client = MagicMock()
    client.start_run = Mock()
    client.log_params = Mock()
    client.log_metrics = Mock()
    client.log_apple_silicon_metrics = Mock()
    client.log_artifact = Mock()
    client.log_model = Mock()
    client.dvc_add = Mock(return_value={"dvc_file": "data.dvc", "status": "success"})
    client.dvc_push = Mock(return_value={"status": "success"})
    client.deploy_model = Mock(return_value={"model_tag": "lora_adapter:v1", "status": "deployed"})
    client.monitor_predictions = Mock(return_value={"monitoring_available": True, "drift_detected": False})
    client.get_status = Mock(return_value={
        "mlflow_available": True,
        "dvc_available": True,
        "bentoml_available": True,
        "evidently_available": True,
    })
    return client


@pytest.fixture
def tracker(mock_mlops_client):
    """Create tracker with mock client"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = LoRAMLOpsTracker(client=mock_mlops_client, repo_root=tmpdir)
        yield tracker


class TestLoRAMLOpsTracker:
    """Test LoRA MLOps tracker"""

    def test_initialization(self, tracker):
        """Test tracker initialization"""
        assert tracker.client is not None
        assert tracker.repo_root is not None

    def test_start_training_run(self, tracker):
        """Test starting training run"""
        # Create mock context manager
        mock_run = MagicMock()
        tracker.client.start_run.return_value.__enter__ = Mock(return_value=mock_run)
        tracker.client.start_run.return_value.__exit__ = Mock(return_value=False)

        with tracker.start_training_run(run_name="test-run") as run:
            assert run is not None

        tracker.client.start_run.assert_called_once()
        call_args = tracker.client.start_run.call_args
        assert call_args.kwargs["run_name"] == "test-run"
        assert "task" in call_args.kwargs["tags"]
        assert call_args.kwargs["tags"]["task"] == "training"

    def test_log_training_config(self, tracker):
        """Test logging training configuration"""
        # Create mock configs
        lora_config = MagicMock()
        lora_config.rank = 16
        lora_config.alpha = 32
        lora_config.dropout = 0.1
        lora_config.target_modules = ["q_proj", "v_proj"]
        lora_config.mlx_precision = "float32"

        training_config = MagicMock()
        training_config.model_name = "test-model"
        training_config.num_epochs = 3
        training_config.batch_size = 4
        training_config.learning_rate = 1e-4
        training_config.optimizer = "adam"
        training_config.use_mlx = True

        # Log configuration
        tracker.log_training_config(lora_config, training_config)

        # Verify params were logged
        tracker.client.log_params.assert_called_once()
        params = tracker.client.log_params.call_args[0][0]

        assert params["lora_rank"] == 16
        assert params["lora_alpha"] == 32
        assert params["model_name"] == "test-model"
        assert params["num_epochs"] == 3
        assert params["use_mlx"] is True

    def test_log_training_metrics(self, tracker):
        """Test logging training metrics"""
        metrics = {
            "train_loss": 0.45,
            "learning_rate": 0.0001,
            "tokens_per_second": 125.5,
        }

        tracker.log_training_metrics(metrics, epoch=2)

        tracker.client.log_metrics.assert_called_once_with(metrics, step=2)

    def test_log_apple_silicon_metrics(self, tracker):
        """Test logging Apple Silicon metrics"""
        metrics = {
            "mps_utilization": 87.5,
            "memory_gb": 14.2,
        }

        tracker.log_apple_silicon_metrics(metrics, step=1)

        tracker.client.log_apple_silicon_metrics.assert_called_once_with(metrics, step=1)

    def test_save_model_artifact(self, tracker):
        """Test saving model artifact"""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint"
            checkpoint_path.mkdir()

            tracker.save_model_artifact(checkpoint_path)

            tracker.client.log_artifact.assert_called_once()
            call_args = tracker.client.log_artifact.call_args
            assert call_args[0][0] == checkpoint_path

    def test_version_dataset(self, tracker):
        """Test versioning dataset"""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "train.jsonl"
            dataset_path.touch()

            result = tracker.version_dataset(dataset_path, push_to_remote=False)

            tracker.client.dvc_add.assert_called_once()
            assert result["dvc_file"] == "data.dvc"

    def test_version_dataset_with_push(self, tracker):
        """Test versioning dataset with remote push"""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "train.jsonl"
            dataset_path.touch()

            result = tracker.version_dataset(dataset_path, push_to_remote=True)

            tracker.client.dvc_add.assert_called_once()
            tracker.client.dvc_push.assert_called_once()

    def test_deploy_adapter(self, tracker):
        """Test deploying LoRA adapter"""
        result = tracker.deploy_adapter(
            adapter_path="outputs/checkpoint",
            model_name="lora_adapter",
            model_version="v1.0",
        )

        tracker.client.deploy_model.assert_called_once()
        assert result["model_tag"] == "lora_adapter:v1"
        assert result["status"] == "deployed"

    def test_monitor_inference(self, tracker):
        """Test monitoring inference"""
        prompts = ["Hello", "AI is amazing"]
        generated_texts = ["Hello world", "AI is amazing technology"]
        latencies_ms = [150.2, 142.8]
        memory_mb = 512.5

        results = tracker.monitor_inference(
            prompts=prompts,
            generated_texts=generated_texts,
            latencies_ms=latencies_ms,
            memory_mb=memory_mb,
        )

        tracker.client.monitor_predictions.assert_called_once()
        assert results["monitoring_available"] is True
        assert results["drift_detected"] is False

        # Verify dataframe was created correctly
        call_args = tracker.client.monitor_predictions.call_args
        df = call_args.kwargs["current_data"]
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "prompt" in df.columns
        assert "generated_text" in df.columns
        assert "latency_ms" in df.columns

    def test_get_status(self, tracker):
        """Test getting status"""
        status = tracker.get_status()

        assert status["mlflow_available"] is True
        assert status["dvc_available"] is True
        assert status["bentoml_available"] is True


class TestLoRAMLOpsTrackerIntegration:
    """Integration tests for LoRA MLOps tracker"""

    @pytest.mark.integration
    def test_full_training_workflow(self, tracker):
        """Test full training workflow with MLOps tracking"""
        # Mock configs
        lora_config = MagicMock()
        lora_config.rank = 16
        lora_config.alpha = 32
        training_config = MagicMock()
        training_config.model_name = "test-model"
        training_config.num_epochs = 2

        # Setup mock context manager
        mock_run = MagicMock()
        tracker.client.start_run.return_value.__enter__ = Mock(return_value=mock_run)
        tracker.client.start_run.return_value.__exit__ = Mock(return_value=False)

        # Run training workflow
        with tracker.start_training_run(run_name="integration-test") as run:
            # Log config
            tracker.log_training_config(lora_config, training_config)

            # Simulate training epochs
            for epoch in range(2):
                metrics = {
                    "train_loss": 0.5 - (epoch * 0.1),
                    "learning_rate": 1e-4,
                }
                tracker.log_training_metrics(metrics, epoch=epoch)

            # Save model
            with tempfile.TemporaryDirectory() as tmpdir:
                checkpoint_path = Path(tmpdir) / "checkpoint"
                checkpoint_path.mkdir()
                tracker.save_model_artifact(checkpoint_path)

        # Verify all operations were called
        assert tracker.client.log_params.called
        assert tracker.client.log_metrics.called
        assert tracker.client.log_artifact.called

    @pytest.mark.integration
    def test_full_deployment_workflow(self, tracker):
        """Test full deployment workflow"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Version dataset
            dataset_path = Path(tmpdir) / "train.jsonl"
            dataset_path.touch()
            tracker.version_dataset(dataset_path, push_to_remote=True)

            # Deploy model
            adapter_path = Path(tmpdir) / "adapter"
            adapter_path.mkdir()
            result = tracker.deploy_adapter(
                adapter_path=adapter_path,
                model_name="test_adapter",
            )

            # Verify
            assert tracker.client.dvc_add.called
            assert tracker.client.dvc_push.called
            assert tracker.client.deploy_model.called
            assert "model_tag" in result


def test_create_lora_mlops_client():
    """Test creating LoRA MLOps client"""
    with tempfile.TemporaryDirectory() as tmpdir:
        client = create_lora_mlops_client(repo_root=tmpdir)
        assert client is not None
        assert client.project_name == "lora-finetuning-mlx"
