"""Tests for CoreML Stable Diffusion Style Transfer MLOps Integration"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pytest

from mlops.integrations.p0_projects.coreml_diffusion.integration import (
    DiffusionMLOpsTracker,
    create_diffusion_mlops_client,
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
    client.dvc_add = Mock(return_value={"dvc_file": "data.dvc", "status": "success"})
    client.dvc_push = Mock(return_value={"status": "success"})
    client.deploy_model = Mock(return_value={"model_tag": "style_transfer:v1", "status": "deployed"})
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
        tracker = DiffusionMLOpsTracker(client=mock_mlops_client, repo_root=tmpdir)
        yield tracker


class TestDiffusionMLOpsTracker:
    """Test Diffusion MLOps tracker"""

    def test_initialization(self, tracker):
        """Test tracker initialization"""
        assert tracker.client is not None
        assert tracker.repo_root is not None

    def test_start_transfer_run(self, tracker):
        """Test starting style transfer run"""
        mock_run = MagicMock()
        tracker.client.start_run.return_value.__enter__ = Mock(return_value=mock_run)
        tracker.client.start_run.return_value.__exit__ = Mock(return_value=False)

        with tracker.start_transfer_run(run_name="artistic-001") as run:
            assert run is not None

        tracker.client.start_run.assert_called_once()
        call_args = tracker.client.start_run.call_args
        assert call_args.kwargs["run_name"] == "artistic-001"
        assert call_args.kwargs["tags"]["task"] == "style_transfer"

    def test_start_conversion_run(self, tracker):
        """Test starting CoreML conversion run"""
        mock_run = MagicMock()
        tracker.client.start_run.return_value.__enter__ = Mock(return_value=mock_run)
        tracker.client.start_run.return_value.__exit__ = Mock(return_value=False)

        with tracker.start_conversion_run(run_name="convert-001") as run:
            assert run is not None

        call_args = tracker.client.start_run.call_args
        assert call_args.kwargs["tags"]["task"] == "coreml_conversion"

    def test_log_diffusion_config(self, tracker):
        """Test logging diffusion configuration"""
        diffusion_config = MagicMock()
        diffusion_config.model_name = "stabilityai/stable-diffusion-2"
        diffusion_config.num_inference_steps = 50
        diffusion_config.guidance_scale = 7.5
        diffusion_config.height = 512
        diffusion_config.width = 512
        diffusion_config.use_mlx = True

        tracker.log_diffusion_config(diffusion_config)

        tracker.client.log_params.assert_called_once()
        params = tracker.client.log_params.call_args[0][0]

        assert params["model_name"] == "stabilityai/stable-diffusion-2"
        assert params["num_inference_steps"] == 50
        assert params["use_mlx"] is True

    def test_log_transfer_config(self, tracker):
        """Test logging style transfer configuration"""
        style_config = MagicMock()
        style_config.style_strength = 0.8
        style_config.content_strength = 0.6
        style_config.output_resolution = (512, 512)
        style_config.num_inference_steps = 50
        style_config.guidance_scale = 7.5
        style_config.preserve_color = False

        tracker.log_transfer_config(style_config)

        tracker.client.log_params.assert_called_once()
        params = tracker.client.log_params.call_args[0][0]

        assert params["style_strength"] == 0.8
        assert params["content_strength"] == 0.6
        assert params["preserve_color"] is False

    def test_log_coreml_config(self, tracker):
        """Test logging CoreML configuration"""
        coreml_config = MagicMock()
        coreml_config.optimize_for_apple_silicon = True
        coreml_config.compute_units = "all"
        coreml_config.precision = "float16"
        coreml_config.quantization = "linear"
        coreml_config.use_ane = True

        tracker.log_coreml_config(coreml_config)

        tracker.client.log_params.assert_called_once()
        params = tracker.client.log_params.call_args[0][0]

        assert params["optimize_for_apple_silicon"] is True
        assert params["compute_units"] == "all"
        assert params["use_ane"] is True

    def test_log_transfer_metrics(self, tracker):
        """Test logging transfer metrics"""
        metrics = {
            "style_similarity": 0.87,
            "content_preservation": 0.92,
            "transfer_time_s": 3.5,
            "quality_score": 0.89,
        }

        tracker.log_transfer_metrics(metrics)

        tracker.client.log_metrics.assert_called_once_with(metrics, step=None)

    def test_log_conversion_metrics(self, tracker):
        """Test logging conversion metrics"""
        metrics = {
            "conversion_time_s": 45.2,
            "model_size_mb": 256.8,
            "ane_compatible": True,
        }

        tracker.log_conversion_metrics(metrics)

        tracker.client.log_metrics.assert_called_once_with(metrics)

    def test_log_benchmark_metrics(self, tracker):
        """Test logging benchmark comparison"""
        pytorch_metrics = {
            "inference_time_s": 5.2,
            "memory_mb": 2048.0,
        }

        coreml_metrics = {
            "inference_time_s": 2.1,
            "memory_mb": 1024.0,
        }

        tracker.log_benchmark_metrics(pytorch_metrics, coreml_metrics)

        # Verify multiple log_metrics calls
        assert tracker.client.log_metrics.call_count >= 2

        # Verify speedup was calculated
        all_calls = tracker.client.log_metrics.call_args_list
        metrics_logged = {}
        for call in all_calls:
            metrics_logged.update(call[0][0])

        assert "pytorch_inference_time_s" in metrics_logged
        assert "coreml_inference_time_s" in metrics_logged

        if "coreml_speedup" in metrics_logged:
            speedup = metrics_logged["coreml_speedup"]
            expected_speedup = 5.2 / 2.1
            assert abs(speedup - expected_speedup) < 0.01

    def test_save_output_artifact(self, tracker):
        """Test saving output artifact"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.png"
            output_path.touch()

            tracker.save_output_artifact(output_path)

            tracker.client.log_artifact.assert_called_once()

    def test_save_coreml_model(self, tracker):
        """Test saving CoreML model"""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.mlpackage"
            model_path.mkdir()

            tracker.save_coreml_model(model_path)

            tracker.client.log_artifact.assert_called_once()

    def test_version_model(self, tracker):
        """Test versioning CoreML model"""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.mlpackage"
            model_path.mkdir()

            result = tracker.version_model(model_path, push_to_remote=False)

            tracker.client.dvc_add.assert_called_once()
            call_args = tracker.client.dvc_add.call_args
            assert call_args[1]["recursive"] is True

    def test_version_images(self, tracker):
        """Test versioning image directory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            image_dir = Path(tmpdir) / "images"
            image_dir.mkdir()

            result = tracker.version_images(image_dir, push_to_remote=True)

            tracker.client.dvc_add.assert_called_once()
            tracker.client.dvc_push.assert_called_once()

    def test_deploy_coreml_model(self, tracker):
        """Test deploying CoreML model"""
        result = tracker.deploy_coreml_model(
            model_path="outputs/model.mlpackage",
            model_name="style_transfer",
            model_version="v1.0",
        )

        tracker.client.deploy_model.assert_called_once()
        assert result["model_tag"] == "style_transfer:v1"

    def test_monitor_inference(self, tracker):
        """Test monitoring inference"""
        image_sizes = [(512, 512), (768, 768), (1024, 1024)]
        latencies_ms = [3500, 5200, 8900]
        memory_mb = 1024.5
        quality_scores = [0.87, 0.89, 0.91]

        results = tracker.monitor_inference(
            image_sizes=image_sizes,
            latencies_ms=latencies_ms,
            memory_mb=memory_mb,
            quality_scores=quality_scores,
        )

        tracker.client.monitor_predictions.assert_called_once()
        assert results["monitoring_available"] is True

        # Verify dataframe
        call_args = tracker.client.monitor_predictions.call_args
        df = call_args.kwargs["current_data"]
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert "image_width" in df.columns
        assert "image_height" in df.columns
        assert "latency_ms" in df.columns
        assert "quality_score" in df.columns


class TestDiffusionMLOpsTrackerIntegration:
    """Integration tests for Diffusion MLOps tracker"""

    @pytest.mark.integration
    def test_full_transfer_workflow(self, tracker):
        """Test full style transfer workflow"""
        style_config = MagicMock()
        style_config.style_strength = 0.8
        style_config.content_strength = 0.6

        mock_run = MagicMock()
        tracker.client.start_run.return_value.__enter__ = Mock(return_value=mock_run)
        tracker.client.start_run.return_value.__exit__ = Mock(return_value=False)

        with tracker.start_transfer_run("artistic-001") as run:
            # Log config
            tracker.log_transfer_config(style_config)

            # Log metrics
            metrics = {
                "style_similarity": 0.87,
                "transfer_time_s": 3.5,
            }
            tracker.log_transfer_metrics(metrics)

            # Save output
            with tempfile.TemporaryDirectory() as tmpdir:
                output_path = Path(tmpdir) / "output.png"
                output_path.touch()
                tracker.save_output_artifact(output_path)

        assert tracker.client.log_params.called
        assert tracker.client.log_metrics.called
        assert tracker.client.log_artifact.called

    @pytest.mark.integration
    def test_full_conversion_workflow(self, tracker):
        """Test full CoreML conversion workflow"""
        coreml_config = MagicMock()
        coreml_config.optimize_for_apple_silicon = True
        coreml_config.compute_units = "all"

        mock_run = MagicMock()
        tracker.client.start_run.return_value.__enter__ = Mock(return_value=mock_run)
        tracker.client.start_run.return_value.__exit__ = Mock(return_value=False)

        with tracker.start_conversion_run("convert-001") as run:
            # Log config
            tracker.log_coreml_config(coreml_config)

            # Log metrics
            metrics = {
                "conversion_time_s": 45.2,
                "model_size_mb": 256.8,
            }
            tracker.log_conversion_metrics(metrics)

            # Save model
            with tempfile.TemporaryDirectory() as tmpdir:
                model_path = Path(tmpdir) / "model.mlpackage"
                model_path.mkdir()
                tracker.save_coreml_model(model_path)

        assert tracker.client.log_params.called
        assert tracker.client.log_metrics.called
        assert tracker.client.log_artifact.called


def test_create_diffusion_mlops_client():
    """Test creating Diffusion MLOps client"""
    with tempfile.TemporaryDirectory() as tmpdir:
        client = create_diffusion_mlops_client(repo_root=tmpdir)
        assert client is not None
        assert client.project_name == "coreml-stable-diffusion-style-transfer"
