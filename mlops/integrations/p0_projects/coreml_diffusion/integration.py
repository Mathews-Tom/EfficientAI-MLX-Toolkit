"""CoreML Stable Diffusion Style Transfer MLOps Integration

Provides MLOps integration for style transfer workflows including:
- Experiment tracking for style transfer runs
- Model versioning for CoreML models
- Deployment of optimized models
- Performance monitoring and benchmarking
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from mlops.client.mlops_client import MLOpsClient, create_client

logger = logging.getLogger(__name__)

PROJECT_NAME = "coreml-stable-diffusion-style-transfer"


class DiffusionMLOpsTracker:
    """MLOps tracking wrapper for CoreML style transfer workflows

    This class provides convenient methods to integrate MLOps operations
    into style transfer, CoreML conversion, and inference workflows.

    Example:
        >>> tracker = DiffusionMLOpsTracker()
        >>>
        >>> # Track style transfer run
        >>> with tracker.start_transfer_run(run_name="artistic-style") as run:
        ...     tracker.log_transfer_config(style_config)
        ...
        ...     # Style transfer
        ...     result_image = transfer_style(content, style)
        ...     metrics = compute_quality_metrics(result_image)
        ...
        ...     tracker.log_transfer_metrics(metrics)
        ...     tracker.save_output_artifact(result_image)
        >>>
        >>> # Deploy CoreML model
        >>> tracker.deploy_coreml_model(
        ...     model_path="outputs/coreml/style_transfer.mlpackage",
        ...     model_name="style_transfer_v1",
        ... )
    """

    def __init__(
        self,
        client: MLOpsClient | None = None,
        repo_root: str | Path | None = None,
    ) -> None:
        """Initialize Diffusion MLOps tracker

        Args:
            client: Optional MLOps client (creates default if not provided)
            repo_root: Optional repository root directory
        """
        self.client = client or create_client(PROJECT_NAME, repo_root=repo_root)
        self.repo_root = Path(repo_root) if repo_root else Path.cwd()
        logger.info("Diffusion MLOps tracker initialized for project: %s", PROJECT_NAME)

    def start_transfer_run(
        self,
        run_name: str | None = None,
        description: str | None = None,
        tags: dict[str, str] | None = None,
    ):
        """Start an MLFlow experiment run for style transfer

        Args:
            run_name: Optional run name (auto-generated if not provided)
            description: Optional run description
            tags: Optional tags for the run

        Returns:
            Context manager for active MLFlow run

        Example:
            >>> with tracker.start_transfer_run(run_name="artistic-style") as run:
            ...     tracker.log_transfer_config(style_config)
            ...     # ... transfer code ...
        """
        default_tags = {"task": "style_transfer", "framework": "coreml", "model": "stable_diffusion"}
        if tags:
            default_tags.update(tags)

        return self.client.start_run(
            run_name=run_name,
            tags=default_tags,
            description=description,
        )

    def start_conversion_run(
        self,
        run_name: str | None = None,
        description: str | None = None,
        tags: dict[str, str] | None = None,
    ):
        """Start an MLFlow experiment run for CoreML conversion

        Args:
            run_name: Optional run name
            description: Optional run description
            tags: Optional tags for the run

        Returns:
            Context manager for active MLFlow run
        """
        default_tags = {"task": "coreml_conversion", "framework": "coreml"}
        if tags:
            default_tags.update(tags)

        return self.client.start_run(
            run_name=run_name,
            tags=default_tags,
            description=description,
        )

    def log_diffusion_config(
        self,
        diffusion_config: Any,
    ) -> None:
        """Log diffusion model configuration

        Args:
            diffusion_config: Diffusion configuration object
        """
        params = {
            "model_name": getattr(diffusion_config, "model_name", None),
            "num_inference_steps": getattr(diffusion_config, "num_inference_steps", None),
            "guidance_scale": getattr(diffusion_config, "guidance_scale", None),
            "height": getattr(diffusion_config, "height", None),
            "width": getattr(diffusion_config, "width", None),
            "use_mlx": getattr(diffusion_config, "use_mlx", None),
        }

        # Filter out None values
        params = {k: v for k, v in params.items() if v is not None}

        self.client.log_params(params)
        logger.info("Logged diffusion configuration with %d parameters", len(params))

    def log_transfer_config(
        self,
        style_config: Any,
    ) -> None:
        """Log style transfer configuration

        Args:
            style_config: Style transfer configuration object
        """
        params = {
            "style_strength": getattr(style_config, "style_strength", None),
            "content_strength": getattr(style_config, "content_strength", None),
            "output_resolution": str(getattr(style_config, "output_resolution", None)),
            "num_inference_steps": getattr(style_config, "num_inference_steps", None),
            "guidance_scale": getattr(style_config, "guidance_scale", None),
            "preserve_color": getattr(style_config, "preserve_color", None),
        }

        # Filter out None values
        params = {k: v for k, v in params.items() if v is not None}

        self.client.log_params(params)
        logger.info("Logged style transfer configuration with %d parameters", len(params))

    def log_coreml_config(
        self,
        coreml_config: Any,
    ) -> None:
        """Log CoreML conversion configuration

        Args:
            coreml_config: CoreML configuration object
        """
        params = {
            "optimize_for_apple_silicon": getattr(coreml_config, "optimize_for_apple_silicon", None),
            "compute_units": getattr(coreml_config, "compute_units", None),
            "precision": getattr(coreml_config, "precision", None),
            "quantization": getattr(coreml_config, "quantization", None),
            "use_ane": getattr(coreml_config, "use_ane", None),
        }

        # Filter out None values
        params = {k: v for k, v in params.items() if v is not None}

        self.client.log_params(params)
        logger.info("Logged CoreML configuration with %d parameters", len(params))

    def log_transfer_metrics(
        self,
        metrics: dict[str, float | int],
        step: int | None = None,
    ) -> None:
        """Log style transfer metrics

        Args:
            metrics: Dictionary of metrics (quality scores, transfer time, etc.)
            step: Optional step number

        Example:
            >>> tracker.log_transfer_metrics({
            ...     "style_similarity": 0.87,
            ...     "content_preservation": 0.92,
            ...     "transfer_time_s": 3.5,
            ...     "memory_mb": 1024.5,
            ... })
        """
        self.client.log_metrics(metrics, step=step)
        logger.debug("Logged style transfer metrics")

    def log_conversion_metrics(
        self,
        metrics: dict[str, float | int],
    ) -> None:
        """Log CoreML conversion metrics

        Args:
            metrics: Dictionary of conversion metrics

        Example:
            >>> tracker.log_conversion_metrics({
            ...     "conversion_time_s": 45.2,
            ...     "model_size_mb": 256.8,
            ...     "ane_compatible": True,
            ... })
        """
        self.client.log_metrics(metrics)
        logger.info("Logged CoreML conversion metrics")

    def log_benchmark_metrics(
        self,
        pytorch_metrics: dict[str, float | int],
        coreml_metrics: dict[str, float | int],
    ) -> None:
        """Log benchmark comparison between PyTorch and CoreML models

        Args:
            pytorch_metrics: Metrics for PyTorch model
            coreml_metrics: Metrics for CoreML model
        """
        # Log PyTorch metrics with prefix
        pytorch_prefixed = {f"pytorch_{k}": v for k, v in pytorch_metrics.items()}
        self.client.log_metrics(pytorch_prefixed)

        # Log CoreML metrics with prefix
        coreml_prefixed = {f"coreml_{k}": v for k, v in coreml_metrics.items()}
        self.client.log_metrics(coreml_prefixed)

        # Calculate and log speedup
        if "inference_time_s" in pytorch_metrics and "inference_time_s" in coreml_metrics:
            if coreml_metrics["inference_time_s"] > 0:
                speedup = pytorch_metrics["inference_time_s"] / coreml_metrics["inference_time_s"]
                self.client.log_metrics({"coreml_speedup": speedup})

        logger.info("Logged benchmark comparison metrics")

    def log_apple_silicon_metrics(
        self,
        metrics: dict[str, float | int],
        step: int | None = None,
    ) -> None:
        """Log Apple Silicon specific metrics

        Args:
            metrics: Dictionary of Apple Silicon metrics (ANE usage, memory, etc.)
            step: Optional step number

        Example:
            >>> tracker.log_apple_silicon_metrics({
            ...     "ane_utilization": 92.5,
            ...     "memory_gb": 8.5,
            ...     "thermal_state": "nominal",
            ... })
        """
        self.client.log_apple_silicon_metrics(metrics, step=step)

    def save_output_artifact(
        self,
        output_path: str | Path,
        artifact_path: str = "outputs",
    ) -> None:
        """Save style transfer output as MLFlow artifact

        Args:
            output_path: Path to output file (image, video, etc.)
            artifact_path: Optional subdirectory in artifact store
        """
        self.client.log_artifact(output_path, artifact_path=artifact_path)
        logger.info("Saved output artifact: %s", output_path)

    def save_coreml_model(
        self,
        model_path: str | Path,
        artifact_path: str = "coreml_models",
    ) -> None:
        """Save CoreML model as MLFlow artifact

        Args:
            model_path: Path to CoreML model (.mlpackage or .mlmodel)
            artifact_path: Optional subdirectory in artifact store
        """
        self.client.log_artifact(model_path, artifact_path=artifact_path)
        logger.info("Saved CoreML model artifact: %s", model_path)

    def version_model(
        self,
        model_path: str | Path,
        push_to_remote: bool = False,
    ) -> dict[str, Any]:
        """Version CoreML model with DVC

        Args:
            model_path: Path to CoreML model file or directory
            push_to_remote: Whether to push to remote storage

        Returns:
            Dictionary with versioning information
        """
        model_path = Path(model_path)
        if not model_path.is_absolute():
            model_path = self.repo_root / model_path

        result = self.client.dvc_add(model_path, recursive=model_path.is_dir())
        logger.info("Versioned CoreML model: %s", model_path)

        if push_to_remote:
            self.client.dvc_push()
            logger.info("Pushed model to remote storage")

        return result

    def version_images(
        self,
        image_dir: str | Path,
        push_to_remote: bool = False,
    ) -> dict[str, Any]:
        """Version image datasets with DVC

        Args:
            image_dir: Path to image directory
            push_to_remote: Whether to push to remote storage

        Returns:
            Dictionary with versioning information
        """
        image_dir = Path(image_dir)
        if not image_dir.is_absolute():
            image_dir = self.repo_root / image_dir

        result = self.client.dvc_add(image_dir, recursive=True)
        logger.info("Versioned image directory: %s", image_dir)

        if push_to_remote:
            self.client.dvc_push()
            logger.info("Pushed images to remote storage")

        return result

    def deploy_coreml_model(
        self,
        model_path: str | Path,
        model_name: str,
        model_version: str | None = None,
        build_bento: bool = True,
    ) -> dict[str, Any]:
        """Deploy CoreML model using BentoML

        Args:
            model_path: Path to CoreML model
            model_name: Model identifier
            model_version: Optional model version
            build_bento: Whether to build Bento package

        Returns:
            Dictionary with deployment information

        Example:
            >>> result = tracker.deploy_coreml_model(
            ...     model_path="outputs/coreml/style_transfer.mlpackage",
            ...     model_name="style_transfer",
            ...     model_version="v1.0",
            ... )
        """
        result = self.client.deploy_model(
            model_path=model_path,
            model_name=model_name,
            model_version=model_version,
            build_bento=build_bento,
        )
        logger.info("Deployed CoreML model: %s", result.get("model_tag"))
        return result

    def monitor_inference(
        self,
        image_sizes: list[tuple[int, int]],
        latencies_ms: list[float],
        memory_mb: float,
        quality_scores: list[float] | None = None,
    ) -> dict[str, Any]:
        """Monitor inference performance for CoreML model

        Args:
            image_sizes: List of image sizes (width, height)
            latencies_ms: List of inference latencies in milliseconds
            memory_mb: Memory usage in MB
            quality_scores: Optional list of quality scores

        Returns:
            Dictionary with monitoring results

        Example:
            >>> results = tracker.monitor_inference(
            ...     image_sizes=[(512, 512), (768, 768)],
            ...     latencies_ms=[3500, 5200],
            ...     memory_mb=1024.5,
            ...     quality_scores=[0.87, 0.91],
            ... )
        """
        # Create monitoring dataframe
        df_data = {
            "image_width": [w for w, h in image_sizes],
            "image_height": [h for w, h in image_sizes],
            "latency_ms": latencies_ms,
            "image_pixels": [w * h for w, h in image_sizes],
        }

        if quality_scores:
            df_data["quality_score"] = quality_scores

        df = pd.DataFrame(df_data)

        # Calculate average latency
        avg_latency_ms = sum(latencies_ms) / len(latencies_ms) if latencies_ms else 0.0

        results = self.client.monitor_predictions(
            current_data=df,
            latency_ms=avg_latency_ms,
            memory_mb=memory_mb,
        )

        logger.info("Monitored %d inference samples", len(image_sizes))
        return results

    def get_status(self) -> dict[str, Any]:
        """Get MLOps integration status

        Returns:
            Dictionary with status information
        """
        return self.client.get_status()


def create_diffusion_mlops_client(repo_root: str | Path | None = None) -> MLOpsClient:
    """Create MLOps client for CoreML diffusion project

    Args:
        repo_root: Optional repository root directory

    Returns:
        Configured MLOpsClient instance

    Example:
        >>> client = create_diffusion_mlops_client()
        >>> with client.start_run(run_name="style-transfer-001"):
        ...     client.log_params({"style_strength": 0.8})
        ...     client.log_metrics({"quality_score": 0.87})
    """
    return create_client(PROJECT_NAME, repo_root=repo_root)
