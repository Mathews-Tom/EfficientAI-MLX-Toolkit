"""
MLFlow client API with experiment tracking methods.

This module provides a high-level MLFlow client interface for experiment tracking,
model logging, and artifact management, integrated with Apple Silicon metrics.
"""

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator

import mlflow
from mlflow.entities import Run
from mlflow.tracking import MlflowClient as BaseMlflowClient

from mlops.config import MLFlowConfig

logger = logging.getLogger(__name__)


class MLFlowClientError(Exception):
    """Raised when MLFlow client operations fail."""

    def __init__(
        self,
        message: str,
        operation: str | None = None,
        details: dict[str, str | int | float | bool] | None = None,
    ) -> None:
        super().__init__(message)
        self.operation = operation
        self.details = dict(details or {})


class MLFlowClient:
    """
    High-level MLFlow client for experiment tracking.

    This class provides a simplified interface to MLFlow tracking operations,
    with built-in support for Apple Silicon metrics and automatic configuration.

    Attributes:
        config: MLFlow configuration
        client: Underlying MLFlow tracking client
        experiment_id: Current experiment ID
        run_id: Current run ID (if active)
    """

    def __init__(self, config: MLFlowConfig | None = None) -> None:
        """
        Initialize MLFlow client.

        Args:
            config: MLFlow configuration (uses default if None)
        """
        from mlops.config import get_default_config

        self.config = config or get_default_config()
        self._client: BaseMlflowClient | None = None
        self._experiment_id: str | None = None
        self._run_id: str | None = None
        self._active_run: Run | None = None

        # Initialize MLFlow tracking
        self._initialize_tracking()

    def _initialize_tracking(self) -> None:
        """Initialize MLFlow tracking with configuration."""
        try:
            # Set tracking URI
            mlflow.set_tracking_uri(self.config.tracking_uri)

            # Initialize client
            self._client = BaseMlflowClient(tracking_uri=self.config.tracking_uri)

            # Set or create experiment
            experiment = mlflow.get_experiment_by_name(self.config.experiment_name)
            if experiment is None:
                self._experiment_id = mlflow.create_experiment(
                    name=self.config.experiment_name,
                    artifact_location=self.config.artifact_location,
                    tags=self.config.tags,
                )
                logger.info("Created experiment: %s", self.config.experiment_name)
            else:
                self._experiment_id = experiment.experiment_id
                logger.info("Using existing experiment: %s", self.config.experiment_name)

            # Set experiment
            mlflow.set_experiment(self.config.experiment_name)

        except Exception as e:
            raise MLFlowClientError(
                f"Failed to initialize MLFlow tracking: {e}",
                operation="initialize_tracking",
                details={"tracking_uri": self.config.tracking_uri},
            ) from e

    @property
    def client(self) -> BaseMlflowClient:
        """Get underlying MLFlow client."""
        if self._client is None:
            raise MLFlowClientError(
                "MLFlow client not initialized",
                operation="get_client",
            )
        return self._client

    @property
    def experiment_id(self) -> str:
        """Get current experiment ID."""
        if self._experiment_id is None:
            raise MLFlowClientError(
                "No experiment set",
                operation="get_experiment_id",
            )
        return self._experiment_id

    @property
    def run_id(self) -> str | None:
        """Get current run ID (None if no active run)."""
        return self._run_id

    @property
    def is_active_run(self) -> bool:
        """Check if there's an active run."""
        return self._run_id is not None

    def start_run(
        self,
        run_name: str | None = None,
        nested: bool = False,
        tags: dict[str, str] | None = None,
        description: str | None = None,
    ) -> Run:
        """
        Start a new MLFlow run.

        Args:
            run_name: Optional run name
            nested: Whether this is a nested run
            tags: Optional tags for the run
            description: Optional run description

        Returns:
            Started MLFlow run

        Raises:
            MLFlowClientError: If starting run fails
        """
        try:
            run_tags = dict(self.config.tags)
            if tags:
                run_tags.update(tags)

            if description:
                run_tags["mlflow.note.content"] = description

            self._active_run = mlflow.start_run(
                run_name=run_name,
                experiment_id=self._experiment_id,
                nested=nested,
                tags=run_tags,
            )

            self._run_id = self._active_run.info.run_id

            logger.info("Started run: %s (ID: %s)", run_name or "unnamed", self._run_id)

            # Log system metrics if enabled
            if self.config.enable_system_metrics:
                mlflow.set_system_metrics_sampling_interval(1)
                mlflow.enable_system_metrics_logging()

            return self._active_run

        except Exception as e:
            raise MLFlowClientError(
                f"Failed to start run: {e}",
                operation="start_run",
                details={"run_name": run_name or "unnamed"},
            ) from e

    def end_run(self, status: str = "FINISHED") -> None:
        """
        End the current MLFlow run.

        Args:
            status: Run status (FINISHED, FAILED, KILLED)

        Raises:
            MLFlowClientError: If ending run fails
        """
        try:
            if not self.is_active_run:
                logger.warning("No active run to end")
                return

            mlflow.end_run(status=status)

            logger.info("Ended run: %s with status %s", self._run_id, status)

            self._run_id = None
            self._active_run = None

        except Exception as e:
            raise MLFlowClientError(
                f"Failed to end run: {e}",
                operation="end_run",
                details={"run_id": self._run_id or "none"},
            ) from e

    @contextmanager
    def run(
        self,
        run_name: str | None = None,
        nested: bool = False,
        tags: dict[str, str] | None = None,
        description: str | None = None,
    ) -> Generator[Run, None, None]:
        """
        Context manager for MLFlow runs.

        Args:
            run_name: Optional run name
            nested: Whether this is a nested run
            tags: Optional tags for the run
            description: Optional run description

        Yields:
            Active MLFlow run

        Example:
            >>> with client.run(run_name="experiment-001"):
            ...     client.log_params({"lr": 0.001})
            ...     client.log_metrics({"loss": 0.5})
        """
        run = self.start_run(run_name=run_name, nested=nested, tags=tags, description=description)
        try:
            yield run
        except Exception as e:
            self.end_run(status="FAILED")
            raise MLFlowClientError(
                f"Run failed: {e}",
                operation="run_context",
                details={"run_id": self._run_id or "none"},
            ) from e
        else:
            self.end_run(status="FINISHED")

    def log_params(self, params: dict[str, str | int | float | bool]) -> None:
        """
        Log parameters to current run.

        Args:
            params: Dictionary of parameters to log

        Raises:
            MLFlowClientError: If logging fails
        """
        try:
            if not self.config.log_params:
                logger.debug("Parameter logging disabled, skipping")
                return

            if not self.is_active_run:
                raise MLFlowClientError(
                    "No active run for logging parameters",
                    operation="log_params",
                )

            # Convert all values to strings for MLFlow
            str_params = {k: str(v) for k, v in params.items()}

            mlflow.log_params(str_params)

            logger.debug("Logged %d parameters", len(params))

        except Exception as e:
            raise MLFlowClientError(
                f"Failed to log parameters: {e}",
                operation="log_params",
                details={"run_id": self._run_id or "none"},
            ) from e

    def log_param(self, key: str, value: str | int | float | bool) -> None:
        """
        Log a single parameter to current run.

        Args:
            key: Parameter name
            value: Parameter value

        Raises:
            MLFlowClientError: If logging fails
        """
        self.log_params({key: value})

    def log_metrics(
        self, metrics: dict[str, float | int], step: int | None = None
    ) -> None:
        """
        Log metrics to current run.

        Args:
            metrics: Dictionary of metrics to log
            step: Optional step number

        Raises:
            MLFlowClientError: If logging fails
        """
        try:
            if not self.config.log_metrics:
                logger.debug("Metric logging disabled, skipping")
                return

            if not self.is_active_run:
                raise MLFlowClientError(
                    "No active run for logging metrics",
                    operation="log_metrics",
                )

            # Convert all values to float
            float_metrics = {k: float(v) for k, v in metrics.items()}

            if step is not None:
                mlflow.log_metrics(float_metrics, step=step)
            else:
                mlflow.log_metrics(float_metrics)

            logger.debug("Logged %d metrics", len(metrics))

        except Exception as e:
            raise MLFlowClientError(
                f"Failed to log metrics: {e}",
                operation="log_metrics",
                details={"run_id": self._run_id or "none"},
            ) from e

    def log_metric(self, key: str, value: float | int, step: int | None = None) -> None:
        """
        Log a single metric to current run.

        Args:
            key: Metric name
            value: Metric value
            step: Optional step number

        Raises:
            MLFlowClientError: If logging fails
        """
        self.log_metrics({key: value}, step=step)

    def log_artifact(self, local_path: str | Path, artifact_path: str | None = None) -> None:
        """
        Log an artifact (file) to current run.

        Args:
            local_path: Path to local file
            artifact_path: Optional subdirectory in artifact store

        Raises:
            MLFlowClientError: If logging fails
        """
        try:
            if not self.config.log_artifacts:
                logger.debug("Artifact logging disabled, skipping")
                return

            if not self.is_active_run:
                raise MLFlowClientError(
                    "No active run for logging artifacts",
                    operation="log_artifact",
                )

            local_path_obj = Path(local_path)
            if not local_path_obj.exists():
                raise MLFlowClientError(
                    f"Artifact file not found: {local_path}",
                    operation="log_artifact",
                    details={"local_path": str(local_path)},
                )

            mlflow.log_artifact(str(local_path), artifact_path=artifact_path)

            logger.debug("Logged artifact: %s", local_path)

        except Exception as e:
            raise MLFlowClientError(
                f"Failed to log artifact: {e}",
                operation="log_artifact",
                details={"local_path": str(local_path)},
            ) from e

    def log_artifacts(self, local_dir: str | Path, artifact_path: str | None = None) -> None:
        """
        Log all files in a directory as artifacts.

        Args:
            local_dir: Path to local directory
            artifact_path: Optional subdirectory in artifact store

        Raises:
            MLFlowClientError: If logging fails
        """
        try:
            if not self.config.log_artifacts:
                logger.debug("Artifact logging disabled, skipping")
                return

            if not self.is_active_run:
                raise MLFlowClientError(
                    "No active run for logging artifacts",
                    operation="log_artifacts",
                )

            local_dir_obj = Path(local_dir)
            if not local_dir_obj.exists():
                raise MLFlowClientError(
                    f"Artifact directory not found: {local_dir}",
                    operation="log_artifacts",
                    details={"local_dir": str(local_dir)},
                )

            mlflow.log_artifacts(str(local_dir), artifact_path=artifact_path)

            logger.debug("Logged artifacts from: %s", local_dir)

        except Exception as e:
            raise MLFlowClientError(
                f"Failed to log artifacts: {e}",
                operation="log_artifacts",
                details={"local_dir": str(local_dir)},
            ) from e

    def log_model(
        self,
        model: Any,
        artifact_path: str,
        registered_model_name: str | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Log a model to current run.

        Args:
            model: Model object to log
            artifact_path: Path within artifacts to store model
            registered_model_name: Optional name for model registry
            **kwargs: Additional arguments for model logging

        Raises:
            MLFlowClientError: If logging fails
        """
        try:
            if not self.config.log_models:
                logger.debug("Model logging disabled, skipping")
                return

            if not self.is_active_run:
                raise MLFlowClientError(
                    "No active run for logging model",
                    operation="log_model",
                )

            # Try to infer model flavor and log appropriately
            # This is a simplified version - in production you'd detect model type
            mlflow.sklearn.log_model(
                model,
                artifact_path=artifact_path,
                registered_model_name=registered_model_name,
                **kwargs,
            )

            logger.info("Logged model to: %s", artifact_path)

        except Exception as e:
            raise MLFlowClientError(
                f"Failed to log model: {e}",
                operation="log_model",
                details={"artifact_path": artifact_path},
            ) from e

    def set_tags(self, tags: dict[str, str]) -> None:
        """
        Set tags on current run.

        Args:
            tags: Dictionary of tags to set

        Raises:
            MLFlowClientError: If setting tags fails
        """
        try:
            if not self.is_active_run:
                raise MLFlowClientError(
                    "No active run for setting tags",
                    operation="set_tags",
                )

            mlflow.set_tags(tags)

            logger.debug("Set %d tags", len(tags))

        except Exception as e:
            raise MLFlowClientError(
                f"Failed to set tags: {e}",
                operation="set_tags",
                details={"run_id": self._run_id or "none"},
            ) from e

    def set_tag(self, key: str, value: str) -> None:
        """
        Set a single tag on current run.

        Args:
            key: Tag name
            value: Tag value

        Raises:
            MLFlowClientError: If setting tag fails
        """
        self.set_tags({key: value})

    def get_run(self, run_id: str | None = None) -> Run:
        """
        Get run information.

        Args:
            run_id: Optional run ID (uses current run if None)

        Returns:
            Run object

        Raises:
            MLFlowClientError: If getting run fails
        """
        try:
            target_run_id = run_id or self._run_id

            if target_run_id is None:
                raise MLFlowClientError(
                    "No run ID specified and no active run",
                    operation="get_run",
                )

            return self.client.get_run(target_run_id)

        except Exception as e:
            raise MLFlowClientError(
                f"Failed to get run: {e}",
                operation="get_run",
                details={"run_id": target_run_id or "none"},
            ) from e

    def search_runs(
        self,
        filter_string: str = "",
        max_results: int = 1000,
        order_by: list[str] | None = None,
    ) -> list[Run]:
        """
        Search for runs in current experiment.

        Args:
            filter_string: MLFlow filter string
            max_results: Maximum number of results
            order_by: Optional list of order_by clauses

        Returns:
            List of matching runs

        Raises:
            MLFlowClientError: If search fails
        """
        try:
            runs = mlflow.search_runs(
                experiment_ids=[self.experiment_id],
                filter_string=filter_string,
                max_results=max_results,
                order_by=order_by,
            )

            logger.debug("Found %d runs", len(runs))

            return runs.to_dict("records") if not runs.empty else []

        except Exception as e:
            raise MLFlowClientError(
                f"Failed to search runs: {e}",
                operation="search_runs",
                details={"filter": filter_string},
            ) from e

    def log_apple_silicon_metrics(self, metrics: dict[str, float | int]) -> None:
        """
        Log Apple Silicon specific metrics.

        Args:
            metrics: Dictionary of Apple Silicon metrics

        Raises:
            MLFlowClientError: If logging fails
        """
        if not self.config.enable_apple_silicon_metrics:
            logger.debug("Apple Silicon metrics logging disabled, skipping")
            return

        # Prefix metrics with apple_silicon_ for clarity
        prefixed_metrics = {f"apple_silicon_{k}": v for k, v in metrics.items()}

        self.log_metrics(prefixed_metrics)

    def get_experiment_info(self) -> dict[str, str | None]:
        """
        Get information about current experiment.

        Returns:
            Dictionary with experiment information
        """
        experiment = mlflow.get_experiment(self.experiment_id)

        return {
            "experiment_id": experiment.experiment_id,
            "name": experiment.name,
            "artifact_location": experiment.artifact_location,
            "lifecycle_stage": experiment.lifecycle_stage,
        }


def create_client(config: MLFlowConfig | None = None) -> MLFlowClient:
    """
    Create a new MLFlow client.

    Args:
        config: Optional MLFlow configuration

    Returns:
        Initialized MLFlow client

    Example:
        >>> client = create_client()
        >>> with client.run(run_name="experiment-001"):
        ...     client.log_params({"lr": 0.001})
    """
    return MLFlowClient(config=config)
