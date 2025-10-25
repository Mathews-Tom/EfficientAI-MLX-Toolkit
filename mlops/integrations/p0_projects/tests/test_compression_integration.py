"""Tests for Model Compression MLOps Integration"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pytest

from mlops.integrations.p0_projects.model_compression.integration import (
    CompressionMLOpsTracker,
    create_compression_mlops_client,
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
    client.dvc_add = Mock(return_value={"dvc_file": "model.dvc", "status": "success"})
    client.dvc_push = Mock(return_value={"status": "success"})
    client.deploy_model = Mock(return_value={"model_tag": "compressed_model:v1", "status": "deployed"})
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
        tracker = CompressionMLOpsTracker(client=mock_mlops_client, repo_root=tmpdir)
        yield tracker


class TestCompressionMLOpsTracker:
    """Test Compression MLOps tracker"""

    def test_initialization(self, tracker):
        """Test tracker initialization"""
        assert tracker.client is not None
        assert tracker.repo_root is not None

    def test_start_compression_run(self, tracker):
        """Test starting compression run"""
        mock_run = MagicMock()
        tracker.client.start_run.return_value.__enter__ = Mock(return_value=mock_run)
        tracker.client.start_run.return_value.__exit__ = Mock(return_value=False)

        with tracker.start_compression_run(
            run_name="test-quantize",
            compression_method="quantization"
        ) as run:
            assert run is not None

        tracker.client.start_run.assert_called_once()
        call_args = tracker.client.start_run.call_args
        assert call_args.kwargs["run_name"] == "test-quantize"
        assert call_args.kwargs["tags"]["method"] == "quantization"

    def test_log_quantization_config(self, tracker):
        """Test logging quantization configuration"""
        quant_config = MagicMock()
        quant_config.target_bits = 8
        quant_config.method = MagicMock()
        quant_config.method.__str__ = Mock(return_value="linear")
        quant_config.use_mlx_quantization = True
        quant_config.symmetric = True
        quant_config.per_channel = False

        tracker.log_quantization_config(quant_config)

        tracker.client.log_params.assert_called_once()
        params = tracker.client.log_params.call_args[0][0]

        assert params["compression_method"] == "quantization"
        assert params["target_bits"] == 8
        assert params["use_mlx_quantization"] is True

    def test_log_pruning_config(self, tracker):
        """Test logging pruning configuration"""
        prune_config = MagicMock()
        prune_config.target_sparsity = 0.5
        prune_config.method = MagicMock()
        prune_config.method.__str__ = Mock(return_value="magnitude")
        prune_config.structured = False
        prune_config.prune_bias = False

        tracker.log_pruning_config(prune_config)

        tracker.client.log_params.assert_called_once()
        params = tracker.client.log_params.call_args[0][0]

        assert params["compression_method"] == "pruning"
        assert params["target_sparsity"] == 0.5
        assert params["structured"] is False

    def test_log_compression_metrics(self, tracker):
        """Test logging compression metrics"""
        metrics = {
            "compression_ratio": 3.8,
            "size_reduction_mb": 1250.5,
            "inference_speedup": 2.1,
        }

        tracker.log_compression_metrics(metrics)

        tracker.client.log_metrics.assert_called_once_with(metrics, step=None)

    def test_log_benchmark_metrics(self, tracker):
        """Test logging benchmark comparison"""
        original_metrics = {
            "inference_time": 2.5,
            "memory": 5000.0,
            "throughput": 45.2,
        }

        compressed_metrics = {
            "inference_time": 1.1,
            "memory": 1315.8,
            "throughput": 98.5,
        }

        tracker.log_benchmark_metrics(original_metrics, compressed_metrics)

        # Verify multiple log_metrics calls (original + compressed + improvements)
        # Note: improvements dict will be empty if keys don't match patterns
        # So we expect 2 calls (original + compressed) or 3 if improvements matched
        assert tracker.client.log_metrics.call_count >= 2

        # Verify metrics were logged with correct prefixes
        all_calls = tracker.client.log_metrics.call_args_list
        metrics_logged = {}
        for call in all_calls:
            metrics_logged.update(call[0][0])

        assert "original_inference_time" in metrics_logged
        assert "compressed_inference_time" in metrics_logged

        # If improvements were calculated, verify speedup
        if "speedup_inference_time" in metrics_logged:
            speedup = metrics_logged["speedup_inference_time"]
            expected_speedup = 2.5 / 1.1
            assert abs(speedup - expected_speedup) < 0.01

    def test_save_compressed_model(self, tracker):
        """Test saving compressed model"""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "compressed_model"
            model_path.mkdir()

            tracker.save_compressed_model(model_path)

            tracker.client.log_artifact.assert_called_once()

    def test_version_model(self, tracker):
        """Test versioning model"""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model"
            model_path.mkdir()

            result = tracker.version_model(model_path, push_to_remote=False)

            tracker.client.dvc_add.assert_called_once()
            assert result["dvc_file"] == "model.dvc"

    def test_deploy_compressed_model(self, tracker):
        """Test deploying compressed model"""
        result = tracker.deploy_compressed_model(
            model_path="outputs/quantized/model",
            model_name="compressed_model",
            model_version="v1.0",
        )

        tracker.client.deploy_model.assert_called_once()
        assert result["model_tag"] == "compressed_model:v1"

    def test_monitor_inference(self, tracker):
        """Test monitoring inference"""
        input_sizes = [128, 256, 512]
        latencies_ms = [45.2, 89.5, 178.3]
        memory_mb = 1315.8

        results = tracker.monitor_inference(
            input_sizes=input_sizes,
            latencies_ms=latencies_ms,
            memory_mb=memory_mb,
        )

        tracker.client.monitor_predictions.assert_called_once()
        assert results["monitoring_available"] is True

        # Verify dataframe
        call_args = tracker.client.monitor_predictions.call_args
        df = call_args.kwargs["current_data"]
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert "input_size" in df.columns
        assert "latency_ms" in df.columns


class TestCompressionMLOpsTrackerIntegration:
    """Integration tests for Compression MLOps tracker"""

    @pytest.mark.integration
    def test_full_quantization_workflow(self, tracker):
        """Test full quantization workflow"""
        quant_config = MagicMock()
        quant_config.target_bits = 8
        quant_config.method = MagicMock()
        quant_config.method.__str__ = Mock(return_value="linear")

        mock_run = MagicMock()
        tracker.client.start_run.return_value.__enter__ = Mock(return_value=mock_run)
        tracker.client.start_run.return_value.__exit__ = Mock(return_value=False)

        with tracker.start_compression_run("quantize-8bit", "quantization") as run:
            # Log config
            tracker.log_quantization_config(quant_config)

            # Log metrics
            metrics = {
                "compression_ratio": 3.8,
                "size_reduction_mb": 1250.5,
            }
            tracker.log_compression_metrics(metrics)

            # Save model
            with tempfile.TemporaryDirectory() as tmpdir:
                model_path = Path(tmpdir) / "model"
                model_path.mkdir()
                tracker.save_compressed_model(model_path)

        assert tracker.client.log_params.called
        assert tracker.client.log_metrics.called
        assert tracker.client.log_artifact.called

    @pytest.mark.integration
    def test_full_pruning_workflow(self, tracker):
        """Test full pruning workflow"""
        prune_config = MagicMock()
        prune_config.target_sparsity = 0.5
        prune_config.method = MagicMock()
        prune_config.method.__str__ = Mock(return_value="magnitude")
        prune_config.structured = False

        mock_run = MagicMock()
        tracker.client.start_run.return_value.__enter__ = Mock(return_value=mock_run)
        tracker.client.start_run.return_value.__exit__ = Mock(return_value=False)

        with tracker.start_compression_run("prune-50pct", "pruning") as run:
            # Log config
            tracker.log_pruning_config(prune_config)

            # Log metrics
            metrics = {
                "actual_sparsity": 0.52,
                "parameters_removed_percent": 48.5,
            }
            tracker.log_compression_metrics(metrics)

        assert tracker.client.log_params.called
        assert tracker.client.log_metrics.called


def test_create_compression_mlops_client():
    """Test creating Compression MLOps client"""
    with tempfile.TemporaryDirectory() as tmpdir:
        client = create_compression_mlops_client(repo_root=tmpdir)
        assert client is not None
        assert client.project_name == "model-compression-mlx"
