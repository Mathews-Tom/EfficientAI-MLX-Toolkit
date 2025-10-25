"""Model Compression MLOps Integration

Provides MLOps integration for model compression workflows including:
- Experiment tracking for compression experiments
- Model versioning for compressed models
- Deployment of compressed models
- Performance monitoring and benchmarking
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from mlops.client.mlops_client import MLOpsClient, create_client

logger = logging.getLogger(__name__)

PROJECT_NAME = "model-compression-mlx"


class CompressionMLOpsTracker:
    """MLOps tracking wrapper for model compression workflows

    This class provides convenient methods to integrate MLOps operations
    into quantization, pruning, and compression workflows.

    Example:
        >>> tracker = CompressionMLOpsTracker()
        >>>
        >>> # Track compression run
        >>> with tracker.start_compression_run(run_name="quantize-8bit") as run:
        ...     tracker.log_compression_config(quant_config)
        ...
        ...     # Compression
        ...     compressed_model = quantize_model(...)
        ...     metrics = benchmark_model(compressed_model)
        ...
        ...     tracker.log_compression_metrics(metrics)
        ...     tracker.save_compressed_model(model_path)
        >>>
        >>> # Deploy compressed model
        >>> tracker.deploy_compressed_model(
        ...     model_path="outputs/quantized/model_8bit",
        ...     model_name="llama_8bit",
        ... )
    """

    def __init__(
        self,
        client: MLOpsClient | None = None,
        repo_root: str | Path | None = None,
    ) -> None:
        """Initialize Compression MLOps tracker

        Args:
            client: Optional MLOps client (creates default if not provided)
            repo_root: Optional repository root directory
        """
        self.client = client or create_client(PROJECT_NAME, repo_root=repo_root)
        self.repo_root = Path(repo_root) if repo_root else Path.cwd()
        logger.info("Compression MLOps tracker initialized for project: %s", PROJECT_NAME)

    def start_compression_run(
        self,
        run_name: str | None = None,
        compression_method: str | None = None,
        description: str | None = None,
        tags: dict[str, str] | None = None,
    ):
        """Start an MLFlow experiment run for compression

        Args:
            run_name: Optional run name (auto-generated if not provided)
            compression_method: Compression method (quantization, pruning, distillation)
            description: Optional run description
            tags: Optional tags for the run

        Returns:
            Context manager for active MLFlow run

        Example:
            >>> with tracker.start_compression_run(
            ...     run_name="quantize-8bit",
            ...     compression_method="quantization",
            ... ) as run:
            ...     tracker.log_compression_config(quant_config)
            ...     # ... compression code ...
        """
        default_tags = {"task": "compression", "framework": "mlx"}
        if compression_method:
            default_tags["method"] = compression_method
        if tags:
            default_tags.update(tags)

        return self.client.start_run(
            run_name=run_name,
            tags=default_tags,
            description=description,
        )

    def log_quantization_config(
        self,
        quant_config: Any,
    ) -> None:
        """Log quantization configuration

        Args:
            quant_config: Quantization configuration object
        """
        params = {
            "compression_method": "quantization",
            "target_bits": getattr(quant_config, "target_bits", None),
            "quantization_method": str(getattr(quant_config, "method", None)),
            "use_mlx_quantization": getattr(quant_config, "use_mlx_quantization", None),
            "symmetric": getattr(quant_config, "symmetric", None),
            "per_channel": getattr(quant_config, "per_channel", None),
            "calibration_samples": getattr(quant_config, "calibration_samples", None),
        }

        # Filter out None values
        params = {k: v for k, v in params.items() if v is not None}

        self.client.log_params(params)
        logger.info("Logged quantization configuration with %d parameters", len(params))

    def log_pruning_config(
        self,
        prune_config: Any,
    ) -> None:
        """Log pruning configuration

        Args:
            prune_config: Pruning configuration object
        """
        params = {
            "compression_method": "pruning",
            "target_sparsity": getattr(prune_config, "target_sparsity", None),
            "pruning_method": str(getattr(prune_config, "method", None)),
            "structured": getattr(prune_config, "structured", None),
            "prune_bias": getattr(prune_config, "prune_bias", None),
            "iterative": getattr(prune_config, "iterative", None),
            "num_iterations": getattr(prune_config, "num_iterations", None),
        }

        # Filter out None values
        params = {k: v for k, v in params.items() if v is not None}

        self.client.log_params(params)
        logger.info("Logged pruning configuration with %d parameters", len(params))

    def log_compression_config(
        self,
        comp_config: Any,
    ) -> None:
        """Log comprehensive compression configuration

        Args:
            comp_config: Compression configuration object
        """
        params = {
            "compression_method": "comprehensive",
            "enabled_methods": str(getattr(comp_config, "enabled_methods", [])),
            "model_name": getattr(comp_config, "model_name", None),
            "use_mlx": getattr(comp_config, "use_mlx", None),
            "sequential_compression": getattr(comp_config, "sequential_compression", None),
        }

        # Filter out None values
        params = {k: v for k, v in params.items() if v is not None}

        self.client.log_params(params)
        logger.info("Logged compression configuration with %d parameters", len(params))

    def log_compression_metrics(
        self,
        metrics: dict[str, float | int],
        step: int | None = None,
    ) -> None:
        """Log compression metrics

        Args:
            metrics: Dictionary of metrics (compression_ratio, size_reduction, etc.)
            step: Optional step number

        Example:
            >>> tracker.log_compression_metrics({
            ...     "compression_ratio": 3.8,
            ...     "size_reduction_mb": 1250.5,
            ...     "actual_sparsity": 0.75,
            ...     "inference_speedup": 2.1,
            ... })
        """
        self.client.log_metrics(metrics, step=step)
        logger.debug("Logged compression metrics")

    def log_benchmark_metrics(
        self,
        original_metrics: dict[str, float | int],
        compressed_metrics: dict[str, float | int],
    ) -> None:
        """Log benchmark comparison between original and compressed models

        Args:
            original_metrics: Metrics for original model
            compressed_metrics: Metrics for compressed model
        """
        # Log original metrics with prefix
        original_prefixed = {f"original_{k}": v for k, v in original_metrics.items()}
        self.client.log_metrics(original_prefixed)

        # Log compressed metrics with prefix
        compressed_prefixed = {f"compressed_{k}": v for k, v in compressed_metrics.items()}
        self.client.log_metrics(compressed_prefixed)

        # Calculate and log improvement ratios
        improvements = {}
        for key in original_metrics:
            if key in compressed_metrics and key.endswith(("_time", "_latency", "_memory")):
                # For time/latency/memory, lower is better
                if compressed_metrics[key] > 0:
                    improvements[f"speedup_{key}"] = original_metrics[key] / compressed_metrics[key]
            elif key in compressed_metrics and key.endswith(("_throughput", "_fps")):
                # For throughput/fps, higher is better
                if original_metrics[key] > 0:
                    improvements[f"improvement_{key}"] = compressed_metrics[key] / original_metrics[key]

        if improvements:
            self.client.log_metrics(improvements)

        logger.info("Logged benchmark comparison metrics")

    def log_apple_silicon_metrics(
        self,
        metrics: dict[str, float | int],
        step: int | None = None,
    ) -> None:
        """Log Apple Silicon specific metrics

        Args:
            metrics: Dictionary of Apple Silicon metrics
            step: Optional step number
        """
        self.client.log_apple_silicon_metrics(metrics, step=step)

    def save_compressed_model(
        self,
        model_path: str | Path,
        artifact_path: str = "compressed_models",
    ) -> None:
        """Save compressed model as MLFlow artifact

        Args:
            model_path: Path to compressed model directory
            artifact_path: Optional subdirectory in artifact store
        """
        self.client.log_artifact(model_path, artifact_path=artifact_path)
        logger.info("Saved compressed model artifact: %s", model_path)

    def version_model(
        self,
        model_path: str | Path,
        push_to_remote: bool = False,
    ) -> dict[str, Any]:
        """Version compressed model with DVC

        Args:
            model_path: Path to model file or directory
            push_to_remote: Whether to push to remote storage

        Returns:
            Dictionary with versioning information
        """
        model_path = Path(model_path)
        if not model_path.is_absolute():
            model_path = self.repo_root / model_path

        result = self.client.dvc_add(model_path)
        logger.info("Versioned model: %s", model_path)

        if push_to_remote:
            self.client.dvc_push()
            logger.info("Pushed model to remote storage")

        return result

    def deploy_compressed_model(
        self,
        model_path: str | Path,
        model_name: str,
        model_version: str | None = None,
        build_bento: bool = True,
    ) -> dict[str, Any]:
        """Deploy compressed model using BentoML

        Args:
            model_path: Path to compressed model
            model_name: Model identifier
            model_version: Optional model version
            build_bento: Whether to build Bento package

        Returns:
            Dictionary with deployment information

        Example:
            >>> result = tracker.deploy_compressed_model(
            ...     model_path="outputs/quantized/model_8bit",
            ...     model_name="llama_8bit",
            ...     model_version="v1.0",
            ... )
        """
        result = self.client.deploy_model(
            model_path=model_path,
            model_name=model_name,
            model_version=model_version,
            build_bento=build_bento,
        )
        logger.info("Deployed compressed model: %s", result.get("model_tag"))
        return result

    def monitor_inference(
        self,
        input_sizes: list[int],
        latencies_ms: list[float],
        memory_mb: float,
        additional_data: dict[str, list[Any]] | None = None,
    ) -> dict[str, Any]:
        """Monitor inference performance for compressed model

        Args:
            input_sizes: List of input sizes (tokens, pixels, etc.)
            latencies_ms: List of inference latencies in milliseconds
            memory_mb: Memory usage in MB
            additional_data: Optional additional monitoring data

        Returns:
            Dictionary with monitoring results
        """
        # Create monitoring dataframe
        df_data = {
            "input_size": input_sizes,
            "latency_ms": latencies_ms,
        }

        if additional_data:
            df_data.update(additional_data)

        df = pd.DataFrame(df_data)

        # Calculate average latency
        avg_latency_ms = sum(latencies_ms) / len(latencies_ms) if latencies_ms else 0.0

        results = self.client.monitor_predictions(
            current_data=df,
            latency_ms=avg_latency_ms,
            memory_mb=memory_mb,
        )

        logger.info("Monitored %d inference samples", len(input_sizes))
        return results

    def get_status(self) -> dict[str, Any]:
        """Get MLOps integration status

        Returns:
            Dictionary with status information
        """
        return self.client.get_status()


def create_compression_mlops_client(repo_root: str | Path | None = None) -> MLOpsClient:
    """Create MLOps client for model compression project

    Args:
        repo_root: Optional repository root directory

    Returns:
        Configured MLOpsClient instance

    Example:
        >>> client = create_compression_mlops_client()
        >>> with client.start_run(run_name="quantize-8bit"):
        ...     client.log_params({"bits": 8})
        ...     client.log_metrics({"compression_ratio": 3.8})
    """
    return create_client(PROJECT_NAME, repo_root=repo_root)
