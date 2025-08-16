"""
MLflow integration for DSPy experiment tracking and model management.
"""

# Standard library imports
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# Optional third-party imports
try:
    import mlflow
    import mlflow.pyfunc
    from mlflow.tracking import MlflowClient

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    mlflow = None
    MlflowClient = None

# Third-party imports
import dspy

# Local imports
from ..exceptions import DSPyIntegrationError
from ..framework import DSPyFramework
from ..types import DSPyConfig, OptimizationResult

logger = logging.getLogger(__name__)


class DSPyMLflowModel(mlflow.pyfunc.PythonModel if MLFLOW_AVAILABLE else object):
    """MLflow wrapper for DSPy modules."""

    def __init__(self, module: dspy.Module, metadata: dict[str, str | int | float | bool]):
        """Initialize DSPy MLflow model."""
        self.module = module
        self.metadata = metadata

    def predict(self, context, model_input):
        """Predict using DSPy module."""
        try:
            if isinstance(model_input, dict):
                result = self.module(**model_input)
            else:
                # Handle pandas DataFrame or other formats
                inputs = (
                    model_input.to_dict("records")[0]
                    if hasattr(model_input, "to_dict")
                    else model_input
                )
                result = self.module(**inputs)

            # Format result for MLflow
            if hasattr(result, "__dict__"):
                return result.__dict__
            elif isinstance(result, dict):
                return result
            else:
                return {"result": str(result)}

        except Exception as e:
            logger.error("DSPy MLflow model prediction failed: %s", e)
            return {"error": str(e)}


class DSPyMLflowTracker:
    """MLflow tracker for DSPy experiments and models."""

    def __init__(
        self,
        tracking_uri: str | None = None,
        experiment_name: str = "dspy-integration",
        registry_uri: str | None = None,
    ):
        """Initialize MLflow tracker."""

        if not MLFLOW_AVAILABLE:
            raise DSPyIntegrationError("MLflow is not available. Please install mlflow.")

        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.registry_uri = registry_uri

        # Setup MLflow
        self._setup_mlflow()

        # Get or create experiment
        self.experiment = self._get_or_create_experiment()

        # Initialize client
        self.client = MlflowClient(tracking_uri=tracking_uri, registry_uri=registry_uri)

        logger.info("MLflow tracker initialized for experiment: %s", experiment_name)

    def _setup_mlflow(self):
        """Setup MLflow configuration."""
        if self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)

        if self.registry_uri:
            mlflow.set_registry_uri(self.registry_uri)

    def _get_or_create_experiment(self):
        """Get or create MLflow experiment."""
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(self.experiment_name)
                experiment = mlflow.get_experiment(experiment_id)

            mlflow.set_experiment(self.experiment_name)
            return experiment

        except Exception as e:
            logger.error("Failed to setup MLflow experiment: %s", e)
            raise DSPyIntegrationError("MLflow experiment setup failed") from e

    def start_run(self, run_name: str | None = None, tags: dict[str, str | None] = None) -> str:
        """Start a new MLflow run."""
        try:
            run = mlflow.start_run(run_name=run_name, tags=tags)
            logger.info("Started MLflow run: %s", run.info.run_id)
            return run.info.run_id

        except Exception as e:
            logger.error("Failed to start MLflow run: %s", e)
            raise DSPyIntegrationError("MLflow run start failed") from e

    def log_dspy_config(self, config: DSPyConfig):
        """Log DSPy configuration to MLflow."""
        try:
            mlflow.log_params(
                {
                    "model_provider": config.model_provider,
                    "model_name": config.model_name,
                    "optimization_level": config.optimization_level,
                    "enable_tracing": config.enable_tracing,
                    "max_retries": config.max_retries,
                }
            )

            # Log config as artifact
            config_dict = {
                "model_provider": config.model_provider,
                "model_name": config.model_name,
                "optimization_level": config.optimization_level,
                "cache_dir": str(config.cache_dir),
                "enable_tracing": config.enable_tracing,
                "max_retries": config.max_retries,
            }

            config_path = Path("dspy_config.json")
            with open(config_path, "w") as f:
                json.dump(config_dict, f, indent=2)

            mlflow.log_artifact(str(config_path))
            config_path.unlink()  # Clean up

        except Exception as e:
            logger.error("Failed to log DSPy config: %s", e)

    def log_optimization_result(self, result: OptimizationResult, prefix: str = ""):
        """Log optimization result to MLflow."""
        try:
            # Log metrics
            prefix = f"{prefix}_" if prefix else ""

            mlflow.log_params(
                {
                    f"{prefix}optimizer_used": result.optimizer_used,
                    f"{prefix}num_examples_used": result.num_examples_used,
                }
            )

            mlflow.log_metrics(
                {
                    f"{prefix}optimization_time": result.optimization_time,
                }
            )

            # Log original performance
            for metric, value in result.original_performance.items():
                mlflow.log_metric(f"{prefix}original_{metric}", value)

            # Log optimized performance
            for metric, value in result.optimized_performance.items():
                mlflow.log_metric(f"{prefix}optimized_{metric}", value)

            # Calculate and log improvement
            for metric in result.original_performance.keys():
                if metric in result.optimized_performance:
                    orig = result.original_performance[metric]
                    opt = result.optimized_performance[metric]
                    if orig > 0:
                        improvement = (opt - orig) / orig
                        mlflow.log_metric(f"{prefix}improvement_{metric}", improvement)

            # Log metadata as tags
            for key, value in result.metadata.items():
                mlflow.set_tag(f"{prefix}metadata_{key}", str(value))

        except Exception as e:
            logger.error("Failed to log optimization result: %s", e)

    def log_dspy_module(
        self,
        module: dspy.Module,
        module_name: str,
        metadata: dict[str, Any | None] = None,
        signature: str | None = None,
    ) -> str:
        """Log DSPy module as MLflow model."""
        try:
            # Prepare metadata
            model_metadata = {
                "module_name": module_name,
                "module_type": type(module).__name__,
                "timestamp": datetime.now().isoformat(),
                "signature": signature,
                **(metadata or {}),
            }

            # Create MLflow model
            mlflow_model = DSPyMLflowModel(module, model_metadata)

            # Log model
            model_info = mlflow.pyfunc.log_model(
                artifact_path=f"dspy_module_{module_name}",
                python_model=mlflow_model,
                registered_model_name=f"dspy_{module_name}",
                metadata=model_metadata,
            )

            logger.info("Logged DSPy module %s to MLflow: %s", module_name, model_info.model_uri)
            return model_info.model_uri

        except Exception as e:
            logger.error("Failed to log DSPy module: %s", e)
            raise DSPyIntegrationError("MLflow model logging failed") from e

    def log_framework_stats(self, framework: DSPyFramework):
        """Log framework statistics to MLflow."""
        try:
            stats = framework.get_framework_stats()

            # Log framework config
            framework_config = stats.get("framework", {}).get("config", {})
            for key, value in framework_config.items():
                mlflow.log_param(f"framework_{key}", value)

            # Log component stats
            for component, component_stats in stats.items():
                if isinstance(component_stats, dict) and component != "framework":
                    for key, value in component_stats.items():
                        if isinstance(value, (int, float)):
                            mlflow.log_metric(f"{component}_{key}", value)
                        else:
                            mlflow.set_tag(f"{component}_{key}", str(value))

        except Exception as e:
            logger.error("Failed to log framework stats: %s", e)

    def log_performance_metrics(
        self, metrics: dict[str, str | int | float | bool], step: int | None = None
    ):
        """Log performance metrics to MLflow."""
        try:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(key, value, step=step)
                else:
                    mlflow.set_tag(key, str(value))

        except Exception as e:
            logger.error("Failed to log performance metrics: %s", e)

    def register_model(
        self,
        model_uri: str,
        model_name: str,
        description: str | None = None,
        tags: dict[str, str | None] = None,
    ) -> str:
        """Register model in MLflow model registry."""
        try:
            model_version = mlflow.register_model(model_uri=model_uri, name=model_name, tags=tags)

            if description:
                self.client.update_model_version(
                    name=model_name,
                    version=model_version.version,
                    description=description,
                )

            logger.info("Registered model %s version %s", model_name, model_version.version)
            return f"{model_name}/{model_version.version}"

        except Exception as e:
            logger.error("Failed to register model: %s", e)
            raise DSPyIntegrationError("Model registration failed") from e

    def load_dspy_module(self, model_uri: str) -> dspy.Module:
        """Load DSPy module from MLflow."""
        try:
            # Load MLflow model
            mlflow_model = mlflow.pyfunc.load_model(model_uri)

            # Extract DSPy module
            if hasattr(mlflow_model, "_model_impl") and hasattr(mlflow_model._model_impl, "module"):
                return mlflow_model._model_impl.module
            else:
                raise DSPyIntegrationError("Invalid DSPy model format in MLflow")

        except Exception as e:
            logger.error("Failed to load DSPy module from MLflow: %s", e)
            raise DSPyIntegrationError("MLflow model loading failed") from e

    def search_runs(
        self, filter_string: str | None = None, max_results: int = 100
    ) -> list[dict[str, str | int | float | bool]]:
        """Search MLflow runs."""
        try:
            runs = mlflow.search_runs(
                experiment_ids=[self.experiment.experiment_id],
                filter_string=filter_string,
                max_results=max_results,
            )

            return runs.to_dict("records") if hasattr(runs, "to_dict") else []

        except Exception as e:
            logger.error("Failed to search MLflow runs: %s", e)
            return []

    def get_best_run(self, metric_name: str, ascending: bool = False) -> dict[str, Any | None]:
        """Get the best run based on a metric."""
        try:
            runs = self.search_runs()
            if not runs:
                return None

            # Filter runs that have the metric
            valid_runs = [
                run
                for run in runs
                if f"metrics.{metric_name}" in run and run[f"metrics.{metric_name}"] is not None
            ]

            if not valid_runs:
                return None

            # Sort by metric
            best_run = (
                min(valid_runs, key=lambda x: x[f"metrics.{metric_name}"])
                if ascending
                else max(valid_runs, key=lambda x: x[f"metrics.{metric_name}"])
            )

            return best_run

        except Exception as e:
            logger.error("Failed to get best run: %s", e)
            return None

    def end_run(self, status: str = "FINISHED"):
        """End the current MLflow run."""
        try:
            mlflow.end_run(status=status)
            logger.info("Ended MLflow run")

        except Exception as e:
            logger.error("Failed to end MLflow run: %s", e)

    def create_experiment_comparison(
        self, experiment_names: list[str]
    ) -> dict[str, str | int | float | bool]:
        """Create comparison between experiments."""
        try:
            comparison = {
                "experiments": {},
                "summary": {},
                "timestamp": datetime.now().isoformat(),
            }

            for exp_name in experiment_names:
                exp = mlflow.get_experiment_by_name(exp_name)
                if exp:
                    runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
                    comparison["experiments"][exp_name] = {
                        "experiment_id": exp.experiment_id,
                        "num_runs": len(runs),
                        "runs": (runs.to_dict("records") if hasattr(runs, "to_dict") else []),
                    }

            return comparison

        except Exception as e:
            logger.error("Failed to create experiment comparison: %s", e)
            return {"error": str(e)}


def setup_mlflow_tracking(
    config: DSPyConfig,
    tracking_uri: str | None = None,
    experiment_name: str | None = None,
) -> DSPyMLflowTracker:
    """Setup MLflow tracking for DSPy integration."""
    try:
        experiment_name = experiment_name or f"dspy-{config.model_provider}-{int(time.time())}"

        tracker = DSPyMLflowTracker(tracking_uri=tracking_uri, experiment_name=experiment_name)

        logger.info("MLflow tracking setup completed for experiment: %s", experiment_name)
        return tracker

    except Exception as e:
        logger.error("Failed to setup MLflow tracking: %s", e)
        raise DSPyIntegrationError("MLflow tracking setup failed") from e


# Context manager for MLflow runs
class MLflowRunContext:
    """Context manager for MLflow runs with DSPy integration."""

    def __init__(
        self,
        tracker: DSPyMLflowTracker,
        run_name: str | None = None,
        tags: dict[str, str | None] = None,
    ):
        self.tracker = tracker
        self.run_name = run_name
        self.tags = tags
        self.run_id = None

    def __enter__(self):
        self.run_id = self.tracker.start_run(self.run_name, self.tags)
        return self.tracker

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.tracker.end_run("FAILED")
        else:
            self.tracker.end_run("FINISHED")
