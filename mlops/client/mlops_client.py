"""Unified MLOps Client

Provides a single interface for all MLOps operations across the toolkit,
integrating MLFlow experiment tracking, DVC data versioning, BentoML deployment,
and Evidently monitoring with Apple Silicon optimization.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator

import pandas as pd

from mlops.client.dvc_client import DVCClient, DVCClientError
from mlops.client.mlflow_client import MLFlowClient, MLFlowClientError
from mlops.config.dvc_config import DVCConfig
from mlops.config.mlflow_config import MLFlowConfig
from mlops.workspace.manager import WorkspaceManager, WorkspaceError

logger = logging.getLogger(__name__)


class MLOpsClientError(Exception):
    """Raised when MLOps client operations fail."""

    def __init__(
        self,
        message: str,
        operation: str | None = None,
        component: str | None = None,
        details: dict[str, str | int | float | bool] | None = None,
    ) -> None:
        super().__init__(message)
        self.operation = operation
        self.component = component
        self.details = dict(details or {})


class MLOpsClient:
    """Unified MLOps client for all toolkit operations

    This client provides a single, unified interface for:
    - Experiment tracking (MLFlow)
    - Data versioning (DVC)
    - Model deployment (BentoML)
    - Performance monitoring (Evidently)
    - Apple Silicon metrics collection

    All operations are automatically configured based on the project
    namespace, providing seamless integration for all toolkit projects.

    Attributes:
        project_name: Project namespace identifier
        mlflow_client: MLFlow client instance
        dvc_client: DVC client instance
        bentoml_available: Whether BentoML is available
        evidently_available: Whether Evidently is available
        workspace: Project workspace instance
        workspace_path: Project-specific workspace path (legacy, use workspace.root_path)
    """

    def __init__(
        self,
        project_name: str,
        mlflow_config: MLFlowConfig | None = None,
        dvc_config: DVCConfig | None = None,
        repo_root: str | Path | None = None,
        workspace_path: str | Path | None = None,
        workspace_manager: WorkspaceManager | None = None,
    ) -> None:
        """Initialize MLOps client for a project

        Args:
            project_name: Project namespace identifier
            mlflow_config: Optional MLFlow configuration
            dvc_config: Optional DVC configuration
            repo_root: Repository root directory
            workspace_path: Optional workspace path for outputs (legacy)
            workspace_manager: Optional workspace manager instance
        """
        self.project_name = project_name
        self.repo_root = Path(repo_root) if repo_root else Path.cwd()

        # Setup workspace using WorkspaceManager
        # If workspace_path is provided explicitly, use legacy behavior
        if workspace_path:
            self.workspace_path = Path(workspace_path)
            self.workspace_path.mkdir(parents=True, exist_ok=True)
            self.workspace = None  # type: ignore
            logger.info("Using custom workspace path (legacy): %s", self.workspace_path)
        else:
            # Use WorkspaceManager for automatic workspace management
            if workspace_manager is None:
                workspace_manager = WorkspaceManager(repo_root=self.repo_root)

            # Get or create workspace
            try:
                self.workspace = workspace_manager.get_or_create_workspace(
                    project_name=project_name,
                    mlflow_tracking_uri=(
                        mlflow_config.tracking_uri if mlflow_config else None
                    ),
                )
                self.workspace_path = self.workspace.root_path
                logger.info("Workspace loaded for project: %s", project_name)
            except WorkspaceError as e:
                logger.warning("Failed to load workspace: %s", e)
                # Fallback to default path
                self.workspace_path = self.repo_root / "mlops" / "workspace" / project_name
                self.workspace_path.mkdir(parents=True, exist_ok=True)
                self.workspace = None  # type: ignore

        # Initialize MLFlow client
        try:
            if mlflow_config is None:
                mlflow_config = MLFlowConfig(
                    experiment_name=project_name,
                    tags={"project": project_name},
                )
            self.mlflow_client = MLFlowClient(config=mlflow_config)
            self._mlflow_available = True
            logger.info("MLFlow client initialized for project: %s", project_name)

            # Update workspace with experiment ID
            if self.workspace and hasattr(self.mlflow_client, "_experiment_id"):
                try:
                    workspace_manager.update_workspace_metadata(
                        project_name=project_name,
                        mlflow_experiment_id=str(self.mlflow_client._experiment_id),
                    )
                except Exception as meta_error:
                    logger.debug("Failed to update workspace with experiment ID: %s", meta_error)

        except Exception as e:
            self._mlflow_available = False
            self.mlflow_client = None  # type: ignore
            logger.warning("MLFlow client initialization failed: %s", e)

        # Initialize DVC client
        try:
            if dvc_config is None:
                dvc_config = DVCConfig(
                    project_namespace=project_name,
                    tags={"project": project_name},
                )
            self.dvc_client = DVCClient(config=dvc_config, repo_root=repo_root)
            self._dvc_available = True
            logger.info("DVC client initialized for project: %s", project_name)
        except Exception as e:
            self._dvc_available = False
            self.dvc_client = None  # type: ignore
            logger.warning("DVC client initialization failed: %s", e)

        # Check for optional components
        self.bentoml_available = self._check_bentoml_available()
        self.evidently_available = self._check_evidently_available()

        # Initialize monitoring if available
        self._monitor = None
        if self.evidently_available:
            try:
                from mlops.monitoring.evidently.monitor import create_monitor

                self._monitor = create_monitor(
                    project_name=project_name,
                    workspace_path=str(self.workspace_path / "monitoring"),
                )
                logger.info("Evidently monitor initialized for project: %s", project_name)
            except Exception as e:
                logger.warning("Evidently monitor initialization failed: %s", e)

        logger.info(
            "MLOpsClient initialized: mlflow=%s, dvc=%s, bentoml=%s, evidently=%s",
            self._mlflow_available,
            self._dvc_available,
            self.bentoml_available,
            self.evidently_available,
        )

    @staticmethod
    def _check_bentoml_available() -> bool:
        """Check if BentoML is available"""
        try:
            import bentoml  # noqa: F401

            return True
        except ImportError:
            return False

    @staticmethod
    def _check_evidently_available() -> bool:
        """Check if Evidently is available"""
        try:
            import evidently  # noqa: F401

            return True
        except ImportError:
            return False

    @classmethod
    def from_project(
        cls,
        project_name: str,
        repo_root: str | Path | None = None,
    ) -> MLOpsClient:
        """Create MLOps client with auto-configuration for project

        This is the recommended way to create a client as it automatically
        configures all components based on the project namespace.

        Args:
            project_name: Project namespace identifier
            repo_root: Optional repository root directory

        Returns:
            Configured MLOpsClient instance

        Example:
            >>> client = MLOpsClient.from_project("lora-finetuning-mlx")
            >>> with client.start_run(run_name="experiment-001"):
            ...     client.log_params({"lr": 0.0001})
            ...     client.log_metrics({"loss": 0.42})
        """
        return cls(
            project_name=project_name,
            repo_root=repo_root,
        )

    # ==================== Experiment Tracking (MLFlow) ====================

    @contextmanager
    def start_run(
        self,
        run_name: str | None = None,
        nested: bool = False,
        tags: dict[str, str] | None = None,
        description: str | None = None,
    ) -> Generator[Any, None, None]:
        """Start an MLFlow experiment run (context manager)

        Args:
            run_name: Optional run name
            nested: Whether this is a nested run
            tags: Optional tags for the run
            description: Optional run description

        Yields:
            Active MLFlow run

        Raises:
            MLOpsClientError: If MLFlow is not available or run fails

        Example:
            >>> with client.start_run(run_name="training"):
            ...     client.log_params({"epochs": 10})
            ...     client.log_metrics({"loss": 0.5})
        """
        if not self._mlflow_available:
            raise MLOpsClientError(
                "MLFlow is not available",
                operation="start_run",
                component="mlflow",
            )

        try:
            with self.mlflow_client.run(
                run_name=run_name,
                nested=nested,
                tags=tags,
                description=description,
            ) as run:
                yield run
        except MLFlowClientError as e:
            raise MLOpsClientError(
                f"Failed to start run: {e}",
                operation="start_run",
                component="mlflow",
                details={"run_name": run_name or "unnamed"},
            ) from e

    def log_params(self, params: dict[str, str | int | float | bool]) -> None:
        """Log parameters to current run

        Args:
            params: Dictionary of parameters to log

        Raises:
            MLOpsClientError: If logging fails
        """
        if not self._mlflow_available:
            logger.warning("MLFlow not available, skipping parameter logging")
            return

        try:
            self.mlflow_client.log_params(params)
        except MLFlowClientError as e:
            raise MLOpsClientError(
                f"Failed to log parameters: {e}",
                operation="log_params",
                component="mlflow",
            ) from e

    def log_metrics(
        self,
        metrics: dict[str, float | int],
        step: int | None = None,
    ) -> None:
        """Log metrics to current run

        Args:
            metrics: Dictionary of metrics to log
            step: Optional step number

        Raises:
            MLOpsClientError: If logging fails
        """
        if not self._mlflow_available:
            logger.warning("MLFlow not available, skipping metric logging")
            return

        try:
            self.mlflow_client.log_metrics(metrics, step=step)
        except MLFlowClientError as e:
            raise MLOpsClientError(
                f"Failed to log metrics: {e}",
                operation="log_metrics",
                component="mlflow",
            ) from e

    def log_apple_silicon_metrics(
        self,
        metrics: dict[str, float | int],
        step: int | None = None,
    ) -> None:
        """Log Apple Silicon specific metrics

        Args:
            metrics: Dictionary of Apple Silicon metrics
            step: Optional step number

        Raises:
            MLOpsClientError: If logging fails

        Example:
            >>> client.log_apple_silicon_metrics({
            ...     "mps_utilization": 87.5,
            ...     "memory_gb": 14.2,
            ...     "thermal_state": "nominal",
            ... })
        """
        if not self._mlflow_available:
            logger.warning("MLFlow not available, skipping Apple Silicon metrics")
            return

        try:
            self.mlflow_client.log_apple_silicon_metrics(metrics)
            if step is not None:
                self.mlflow_client.log_metrics(
                    {f"apple_silicon_{k}": v for k, v in metrics.items()},
                    step=step,
                )
        except MLFlowClientError as e:
            raise MLOpsClientError(
                f"Failed to log Apple Silicon metrics: {e}",
                operation="log_apple_silicon_metrics",
                component="mlflow",
            ) from e

    def log_artifact(
        self,
        local_path: str | Path,
        artifact_path: str | None = None,
    ) -> None:
        """Log an artifact (file) to current run

        Args:
            local_path: Path to local file
            artifact_path: Optional subdirectory in artifact store

        Raises:
            MLOpsClientError: If logging fails
        """
        if not self._mlflow_available:
            logger.warning("MLFlow not available, skipping artifact logging")
            return

        try:
            self.mlflow_client.log_artifact(local_path, artifact_path=artifact_path)
        except MLFlowClientError as e:
            raise MLOpsClientError(
                f"Failed to log artifact: {e}",
                operation="log_artifact",
                component="mlflow",
                details={"local_path": str(local_path)},
            ) from e

    def log_model(
        self,
        model: Any,
        artifact_path: str,
        registered_model_name: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Log a model to current run

        Args:
            model: Model object to log
            artifact_path: Path within artifacts to store model
            registered_model_name: Optional name for model registry
            **kwargs: Additional arguments for model logging

        Raises:
            MLOpsClientError: If logging fails
        """
        if not self._mlflow_available:
            logger.warning("MLFlow not available, skipping model logging")
            return

        try:
            self.mlflow_client.log_model(
                model,
                artifact_path=artifact_path,
                registered_model_name=registered_model_name,
                **kwargs,
            )
        except MLFlowClientError as e:
            raise MLOpsClientError(
                f"Failed to log model: {e}",
                operation="log_model",
                component="mlflow",
                details={"artifact_path": artifact_path},
            ) from e

    # ==================== Data Versioning (DVC) ====================

    def dvc_add(
        self,
        path: str | Path,
        recursive: bool = False,
    ) -> dict[str, Any]:
        """Add file or directory to DVC tracking

        Args:
            path: Path to file or directory
            recursive: Whether to add directory recursively

        Returns:
            Dictionary with tracking information

        Raises:
            MLOpsClientError: If adding fails

        Example:
            >>> result = client.dvc_add("datasets/train.jsonl")
            >>> print(result["dvc_file"])
        """
        if not self._dvc_available:
            raise MLOpsClientError(
                "DVC is not available",
                operation="dvc_add",
                component="dvc",
            )

        try:
            return self.dvc_client.add(path, recursive=recursive)
        except DVCClientError as e:
            raise MLOpsClientError(
                f"Failed to add to DVC: {e}",
                operation="dvc_add",
                component="dvc",
                details={"path": str(path)},
            ) from e

    def dvc_push(
        self,
        targets: list[str | Path] | None = None,
        remote: str | None = None,
    ) -> dict[str, Any]:
        """Push tracked data to remote storage

        Args:
            targets: Optional list of specific targets to push
            remote: Optional remote name

        Returns:
            Dictionary with push status

        Raises:
            MLOpsClientError: If push fails
        """
        if not self._dvc_available:
            raise MLOpsClientError(
                "DVC is not available",
                operation="dvc_push",
                component="dvc",
            )

        try:
            return self.dvc_client.push(targets=targets, remote=remote)
        except DVCClientError as e:
            raise MLOpsClientError(
                f"Failed to push to DVC: {e}",
                operation="dvc_push",
                component="dvc",
            ) from e

    def dvc_pull(
        self,
        targets: list[str | Path] | None = None,
        remote: str | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        """Pull tracked data from remote storage

        Args:
            targets: Optional list of specific targets to pull
            remote: Optional remote name
            force: Force download even if cache exists

        Returns:
            Dictionary with pull status

        Raises:
            MLOpsClientError: If pull fails
        """
        if not self._dvc_available:
            raise MLOpsClientError(
                "DVC is not available",
                operation="dvc_pull",
                component="dvc",
            )

        try:
            return self.dvc_client.pull(targets=targets, remote=remote, force=force)
        except DVCClientError as e:
            raise MLOpsClientError(
                f"Failed to pull from DVC: {e}",
                operation="dvc_pull",
                component="dvc",
            ) from e

    # ==================== Model Deployment (BentoML) ====================

    def deploy_model(
        self,
        model_path: str | Path,
        model_name: str,
        model_version: str | None = None,
        build_bento: bool = True,
    ) -> dict[str, Any]:
        """Deploy model using BentoML

        Args:
            model_path: Path to model files
            model_name: Model identifier
            model_version: Optional model version
            build_bento: Whether to build Bento package

        Returns:
            Dictionary with deployment information

        Raises:
            MLOpsClientError: If deployment fails

        Example:
            >>> result = client.deploy_model(
            ...     model_path="outputs/checkpoints/checkpoint_epoch_2",
            ...     model_name="lora_adapter",
            ...     model_version="v1.0",
            ... )
        """
        if not self.bentoml_available:
            raise MLOpsClientError(
                "BentoML is not available",
                operation="deploy_model",
                component="bentoml",
            )

        try:
            from mlops.serving.bentoml.config import ModelFramework
            from mlops.serving.bentoml.packager import package_model

            result = package_model(
                model_path=model_path,
                model_name=model_name,
                project_name=self.project_name,
                model_framework=ModelFramework.MLX,
                build_bento=build_bento,
            )

            logger.info("Model deployed: %s", result.get("model_tag"))
            return result

        except Exception as e:
            raise MLOpsClientError(
                f"Failed to deploy model: {e}",
                operation="deploy_model",
                component="bentoml",
                details={"model_name": model_name},
            ) from e

    # ==================== Monitoring (Evidently) ====================

    def set_reference_data(
        self,
        reference_data: pd.DataFrame,
        target_column: str | None = None,
        prediction_column: str | None = None,
    ) -> None:
        """Set reference data for drift and performance monitoring

        Args:
            reference_data: Reference dataset (training or baseline data)
            target_column: Name of target column
            prediction_column: Name of prediction column

        Raises:
            MLOpsClientError: If setting reference data fails
        """
        if not self.evidently_available or self._monitor is None:
            logger.warning("Evidently not available, skipping reference data")
            return

        try:
            self._monitor.set_reference_data(
                reference_data,
                target_column=target_column,
                prediction_column=prediction_column,
            )
            logger.info("Reference data set for monitoring")
        except Exception as e:
            raise MLOpsClientError(
                f"Failed to set reference data: {e}",
                operation="set_reference_data",
                component="evidently",
            ) from e

    def monitor_predictions(
        self,
        current_data: pd.DataFrame,
        target_column: str | None = None,
        prediction_column: str | None = None,
        latency_ms: float | None = None,
        memory_mb: float | None = None,
    ) -> dict[str, Any]:
        """Monitor predictions for drift and performance

        Args:
            current_data: Current data to monitor
            target_column: Name of target column
            prediction_column: Name of prediction column
            latency_ms: Average inference latency in milliseconds
            memory_mb: Memory usage in MB

        Returns:
            Dictionary with monitoring results

        Raises:
            MLOpsClientError: If monitoring fails
        """
        if not self.evidently_available or self._monitor is None:
            logger.warning("Evidently not available, skipping monitoring")
            return {"monitoring_available": False}

        try:
            results = self._monitor.monitor(
                current_data,
                target_column=target_column,
                prediction_column=prediction_column,
                latency_ms=latency_ms,
                memory_mb=memory_mb,
            )

            # Log monitoring results to MLFlow if available
            if self._mlflow_available and self.mlflow_client.is_active_run:
                if "drift_report" in results:
                    drift = results["drift_report"]
                    self.log_metrics({
                        "monitoring_dataset_drift": 1.0 if drift.get("dataset_drift") else 0.0,
                        "monitoring_drift_share": drift.get("drift_share", 0.0),
                    })

                if "performance_metrics" in results:
                    perf = results["performance_metrics"]
                    self.log_metrics({
                        "monitoring_performance_degraded": 1.0 if perf.get("degraded") else 0.0,
                    })

            return results

        except Exception as e:
            raise MLOpsClientError(
                f"Failed to monitor predictions: {e}",
                operation="monitor_predictions",
                component="evidently",
            ) from e

    # ==================== Workspace Management ====================

    def get_workspace_path(self, subdir: str | None = None) -> Path:
        """Get workspace path for outputs

        Args:
            subdir: Optional subdirectory within workspace

        Returns:
            Path to workspace or subdirectory
        """
        if subdir:
            path = self.workspace_path / subdir
            path.mkdir(parents=True, exist_ok=True)
            return path
        return self.workspace_path

    def get_status(self) -> dict[str, Any]:
        """Get MLOps client status

        Returns:
            Dictionary with status information
        """
        status = {
            "project_name": self.project_name,
            "workspace_path": str(self.workspace_path),
            "mlflow_available": self._mlflow_available,
            "dvc_available": self._dvc_available,
            "bentoml_available": self.bentoml_available,
            "evidently_available": self.evidently_available,
        }

        # Add workspace information if available
        if self.workspace:
            status["workspace"] = self.workspace.to_dict()

        if self._mlflow_available:
            try:
                status["mlflow_experiment"] = self.mlflow_client.get_experiment_info()
            except Exception as e:
                status["mlflow_error"] = str(e)

        if self._dvc_available:
            try:
                status["dvc_connection"] = self.dvc_client.get_connection_info()
            except Exception as e:
                status["dvc_error"] = str(e)

        if self.evidently_available and self._monitor:
            try:
                status["evidently_status"] = self._monitor.get_monitoring_status()
            except Exception as e:
                status["evidently_error"] = str(e)

        return status


def create_client(project_name: str, repo_root: str | Path | None = None) -> MLOpsClient:
    """Create a new MLOps client with auto-configuration

    Args:
        project_name: Project namespace identifier
        repo_root: Optional repository root directory

    Returns:
        Configured MLOpsClient instance

    Example:
        >>> client = create_client("lora-finetuning-mlx")
        >>> with client.start_run(run_name="experiment-001"):
        ...     client.log_params({"lr": 0.001})
        ...     client.log_metrics({"loss": 0.5})
        ...     client.dvc_add("datasets/train.csv")
        ...     client.dvc_push()
    """
    return MLOpsClient.from_project(project_name, repo_root=repo_root)
