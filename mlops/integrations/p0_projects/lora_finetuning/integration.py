"""LoRA Fine-tuning MLOps Integration

Provides MLOps integration for LoRA fine-tuning workflows including:
- Experiment tracking for training runs
- Data versioning for training datasets
- Model deployment for trained adapters
- Performance monitoring for inference
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd

from mlops.client.mlops_client import MLOpsClient, create_client

logger = logging.getLogger(__name__)

PROJECT_NAME = "lora-finetuning-mlx"


class LoRAMLOpsTracker:
    """MLOps tracking wrapper for LoRA fine-tuning workflows

    This class provides convenient methods to integrate MLOps operations
    into LoRA training, inference, and deployment workflows.

    Example:
        >>> tracker = LoRAMLOpsTracker()
        >>>
        >>> # Track training run
        >>> with tracker.start_training_run(run_name="experiment-001") as run:
        ...     tracker.log_training_config(lora_config, training_config)
        ...
        ...     # Training loop
        ...     for epoch in range(num_epochs):
        ...         metrics = train_epoch(...)
        ...         tracker.log_training_metrics(metrics, epoch=epoch)
        ...
        ...     tracker.save_model_artifact(checkpoint_path)
        >>>
        >>> # Version training data
        >>> tracker.version_dataset("data/samples/train.jsonl")
        >>>
        >>> # Deploy trained model
        >>> tracker.deploy_adapter(
        ...     adapter_path="outputs/checkpoints/checkpoint_epoch_2",
        ...     model_name="lora_adapter_v1",
        ... )
    """

    def __init__(
        self,
        client: MLOpsClient | None = None,
        repo_root: str | Path | None = None,
    ) -> None:
        """Initialize LoRA MLOps tracker

        Args:
            client: Optional MLOps client (creates default if not provided)
            repo_root: Optional repository root directory
        """
        self.client = client or create_client(PROJECT_NAME, repo_root=repo_root)
        self.repo_root = Path(repo_root) if repo_root else Path.cwd()
        logger.info("LoRA MLOps tracker initialized for project: %s", PROJECT_NAME)

    def start_training_run(
        self,
        run_name: str | None = None,
        description: str | None = None,
        tags: dict[str, str] | None = None,
    ):
        """Start an MLFlow experiment run for training

        Args:
            run_name: Optional run name (auto-generated if not provided)
            description: Optional run description
            tags: Optional tags for the run

        Returns:
            Context manager for active MLFlow run

        Example:
            >>> with tracker.start_training_run(run_name="lora-exp-001") as run:
            ...     tracker.log_training_config(lora_config, training_config)
            ...     # ... training code ...
        """
        default_tags = {"task": "training", "framework": "mlx", "method": "lora"}
        if tags:
            default_tags.update(tags)

        return self.client.start_run(
            run_name=run_name,
            tags=default_tags,
            description=description,
        )

    def log_training_config(
        self,
        lora_config: Any,
        training_config: Any,
    ) -> None:
        """Log LoRA and training configurations

        Args:
            lora_config: LoRA configuration object
            training_config: Training configuration object
        """
        params = {
            # LoRA parameters
            "lora_rank": getattr(lora_config, "rank", None),
            "lora_alpha": getattr(lora_config, "alpha", None),
            "lora_dropout": getattr(lora_config, "dropout", None),
            "lora_target_modules": str(getattr(lora_config, "target_modules", [])),

            # Training parameters
            "model_name": getattr(training_config, "model_name", None),
            "num_epochs": getattr(training_config, "num_epochs", None),
            "batch_size": getattr(training_config, "batch_size", None),
            "learning_rate": getattr(training_config, "learning_rate", None),
            "optimizer": getattr(training_config, "optimizer", None),
            "warmup_steps": getattr(training_config, "warmup_steps", None),
            "gradient_accumulation_steps": getattr(training_config, "gradient_accumulation_steps", None),
            "max_grad_norm": getattr(training_config, "max_grad_norm", None),

            # MLX parameters
            "use_mlx": getattr(training_config, "use_mlx", True),
            "mlx_precision": getattr(lora_config, "mlx_precision", "float32"),
        }

        # Filter out None values
        params = {k: v for k, v in params.items() if v is not None}

        self.client.log_params(params)
        logger.info("Logged training configuration with %d parameters", len(params))

    def log_training_metrics(
        self,
        metrics: dict[str, float | int],
        epoch: int | None = None,
        step: int | None = None,
    ) -> None:
        """Log training metrics

        Args:
            metrics: Dictionary of metrics (loss, accuracy, etc.)
            epoch: Optional epoch number
            step: Optional step number (used if epoch not provided)
        """
        # Use epoch as step if provided, otherwise use step
        metric_step = epoch if epoch is not None else step

        self.client.log_metrics(metrics, step=metric_step)

        if epoch is not None:
            logger.debug("Logged training metrics for epoch %d", epoch)
        else:
            logger.debug("Logged training metrics")

    def log_apple_silicon_metrics(
        self,
        metrics: dict[str, float | int],
        step: int | None = None,
    ) -> None:
        """Log Apple Silicon specific metrics

        Args:
            metrics: Dictionary of Apple Silicon metrics (MPS utilization, memory, etc.)
            step: Optional step number

        Example:
            >>> tracker.log_apple_silicon_metrics({
            ...     "mps_utilization": 87.5,
            ...     "memory_gb": 14.2,
            ...     "thermal_state": "nominal",
            ... })
        """
        self.client.log_apple_silicon_metrics(metrics, step=step)

    def save_model_artifact(
        self,
        checkpoint_path: str | Path,
        artifact_path: str = "checkpoints",
    ) -> None:
        """Save checkpoint as MLFlow artifact

        Args:
            checkpoint_path: Path to checkpoint directory
            artifact_path: Optional subdirectory in artifact store
        """
        self.client.log_artifact(checkpoint_path, artifact_path=artifact_path)
        logger.info("Saved checkpoint artifact: %s", checkpoint_path)

    def log_model(
        self,
        model: Any,
        artifact_path: str = "model",
        registered_model_name: str | None = None,
    ) -> None:
        """Log model to MLFlow

        Args:
            model: Model object to log
            artifact_path: Path within artifacts to store model
            registered_model_name: Optional name for model registry
        """
        self.client.log_model(
            model,
            artifact_path=artifact_path,
            registered_model_name=registered_model_name,
        )
        logger.info("Logged model to MLFlow")

    def version_dataset(
        self,
        dataset_path: str | Path,
        push_to_remote: bool = False,
    ) -> dict[str, Any]:
        """Version training dataset with DVC

        Args:
            dataset_path: Path to dataset file or directory
            push_to_remote: Whether to push to remote storage

        Returns:
            Dictionary with versioning information

        Example:
            >>> result = tracker.version_dataset("data/samples/train.jsonl")
            >>> print(f"DVC file: {result['dvc_file']}")
        """
        dataset_path = Path(dataset_path)
        if not dataset_path.is_absolute():
            dataset_path = self.repo_root / dataset_path

        result = self.client.dvc_add(dataset_path)
        logger.info("Versioned dataset: %s", dataset_path)

        if push_to_remote:
            self.client.dvc_push()
            logger.info("Pushed dataset to remote storage")

        return result

    def deploy_adapter(
        self,
        adapter_path: str | Path,
        model_name: str,
        model_version: str | None = None,
        build_bento: bool = True,
    ) -> dict[str, Any]:
        """Deploy LoRA adapter using BentoML

        Args:
            adapter_path: Path to LoRA adapter checkpoint
            model_name: Model identifier
            model_version: Optional model version
            build_bento: Whether to build Bento package

        Returns:
            Dictionary with deployment information

        Example:
            >>> result = tracker.deploy_adapter(
            ...     adapter_path="outputs/checkpoints/checkpoint_epoch_2",
            ...     model_name="lora_adapter",
            ...     model_version="v1.0",
            ... )
            >>> print(f"Model tag: {result['model_tag']}")
        """
        result = self.client.deploy_model(
            model_path=adapter_path,
            model_name=model_name,
            model_version=model_version,
            build_bento=build_bento,
        )
        logger.info("Deployed LoRA adapter: %s", result.get("model_tag"))
        return result

    def monitor_inference(
        self,
        prompts: list[str],
        generated_texts: list[str],
        latencies_ms: list[float],
        memory_mb: float,
    ) -> dict[str, Any]:
        """Monitor inference performance and quality

        Args:
            prompts: List of input prompts
            generated_texts: List of generated outputs
            latencies_ms: List of inference latencies in milliseconds
            memory_mb: Memory usage in MB

        Returns:
            Dictionary with monitoring results

        Example:
            >>> results = tracker.monitor_inference(
            ...     prompts=["Hello", "AI is"],
            ...     generated_texts=["Hello world", "AI is amazing"],
            ...     latencies_ms=[150.2, 142.8],
            ...     memory_mb=512.5,
            ... )
        """
        # Create monitoring dataframe
        df = pd.DataFrame({
            "prompt": prompts,
            "generated_text": generated_texts,
            "latency_ms": latencies_ms,
            "prompt_length": [len(p) for p in prompts],
            "generated_length": [len(g) for g in generated_texts],
        })

        # Calculate average latency
        avg_latency_ms = sum(latencies_ms) / len(latencies_ms) if latencies_ms else 0.0

        results = self.client.monitor_predictions(
            current_data=df,
            latency_ms=avg_latency_ms,
            memory_mb=memory_mb,
        )

        logger.info("Monitored %d inference samples", len(prompts))
        return results

    def get_status(self) -> dict[str, Any]:
        """Get MLOps integration status

        Returns:
            Dictionary with status information
        """
        return self.client.get_status()


def create_lora_mlops_client(repo_root: str | Path | None = None) -> MLOpsClient:
    """Create MLOps client for LoRA fine-tuning project

    Args:
        repo_root: Optional repository root directory

    Returns:
        Configured MLOpsClient instance

    Example:
        >>> client = create_lora_mlops_client()
        >>> with client.start_run(run_name="experiment-001"):
        ...     client.log_params({"lr": 0.001})
        ...     client.log_metrics({"loss": 0.5})
    """
    return create_client(PROJECT_NAME, repo_root=repo_root)
