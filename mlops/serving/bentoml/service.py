"""BentoML Service Implementation

Service implementation for MLX models with Apple Silicon optimization.
"""

from __future__ import annotations

import logging
from typing import Any

import bentoml
from bentoml.io import JSON

from mlops.serving.bentoml.runner import MLXModelRunner, create_runner

logger = logging.getLogger(__name__)


class BentoMLError(Exception):
    """Exception raised for BentoML service errors"""

    def __init__(
        self,
        message: str,
        operation: str | None = None,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.operation = operation
        self.details = dict(details or {})


class MLXBentoService:
    """Base BentoML service for MLX models with Apple Silicon optimization

    This service provides a standardized interface for serving MLX models
    with support for:
    - Apple Silicon hardware acceleration (MPS, unified memory)
    - MLX-optimized inference
    - Batching and async operations
    - Health checks and metrics
    - Multi-project deployment
    """

    def __init__(
        self,
        runner: MLXModelRunner,
        service_name: str = "mlx_model_service",
        project_name: str = "default",
    ):
        """Initialize BentoML service

        Args:
            runner: MLX model runner
            service_name: Service identifier
            project_name: Project identifier
        """
        self.runner = runner
        self.service_name = service_name
        self.project_name = project_name
        self._is_ready = False

        logger.info(
            "Initialized MLXBentoService: %s (project: %s)",
            service_name,
            project_name,
        )

    def load(self) -> None:
        """Load model and prepare service"""
        try:
            logger.info("Loading model for service: %s", self.service_name)
            self.runner.load_model()
            self._is_ready = True
            logger.info("Service ready: %s", self.service_name)

        except Exception as e:
            logger.error("Failed to load service: %s", e)
            raise BentoMLError(
                f"Service initialization failed: {e}",
                operation="load",
                details={"service_name": self.service_name},
            ) from e

    def predict(self, input_data: Any) -> dict[str, Any]:
        """Run model prediction

        Args:
            input_data: Input data for prediction

        Returns:
            Prediction results

        Raises:
            BentoMLError: If service is not ready or prediction fails
        """
        if not self._is_ready:
            raise BentoMLError(
                "Service not ready. Call load() first.",
                operation="predict",
                details={"service_name": self.service_name},
            )

        try:
            result = self.runner.predict(input_data)

            # Add service metadata
            if isinstance(result, dict):
                result["service_name"] = self.service_name
                result["project_name"] = self.project_name
            else:
                result = {
                    "prediction": result,
                    "service_name": self.service_name,
                    "project_name": self.project_name,
                }

            return result

        except Exception as e:
            logger.error("Prediction failed: %s", e)
            raise BentoMLError(
                f"Prediction failed: {e}",
                operation="predict",
                details={"service_name": self.service_name},
            ) from e

    def health_check(self) -> dict[str, Any]:
        """Check service health

        Returns:
            Health status information
        """
        try:
            memory_usage = self.runner.get_memory_usage()

            return {
                "status": "healthy" if self._is_ready else "not_ready",
                "service_name": self.service_name,
                "project_name": self.project_name,
                "model_loaded": self.runner.is_loaded,
                "memory_usage_mb": memory_usage.get("total_mb", 0.0),
                "mlx_available": memory_usage.get("mlx_available", False),
            }

        except Exception as e:
            logger.error("Health check failed: %s", e)
            return {
                "status": "unhealthy",
                "service_name": self.service_name,
                "error": str(e),
            }

    def unload(self) -> None:
        """Unload model and cleanup resources"""
        try:
            logger.info("Unloading service: %s", self.service_name)
            self.runner.unload_model()
            self._is_ready = False
            logger.info("Service unloaded: %s", self.service_name)

        except Exception as e:
            logger.error("Failed to unload service: %s", e)
            raise BentoMLError(
                f"Service unload failed: {e}",
                operation="unload",
                details={"service_name": self.service_name},
            ) from e

    @property
    def is_ready(self) -> bool:
        """Check if service is ready"""
        return self._is_ready


def create_bentoml_service(
    model_path: str,
    service_name: str = "mlx_model_service",
    project_name: str = "default",
    model_type: str = "mlx",
    **runner_kwargs: Any,
) -> bentoml.Service:
    """Create a BentoML service for MLX models

    This function creates a complete BentoML service with API endpoints
    for prediction, health checks, and metrics.

    Args:
        model_path: Path to model files
        service_name: Service identifier
        project_name: Project identifier
        model_type: Model type ('mlx', 'lora', etc.)
        **runner_kwargs: Additional runner configuration

    Returns:
        BentoML Service instance

    Example:
        >>> service = create_bentoml_service(
        ...     model_path="models/lora-adapter",
        ...     service_name="lora_service",
        ...     project_name="lora-finetuning-mlx",
        ...     model_type="lora",
        ... )
    """
    # Create runner
    runner = create_runner(model_path, model_type=model_type, **runner_kwargs)

    # Create BentoML runner
    bento_runner = bentoml.Runner(
        runner.predict,
        name=f"{service_name}_runner",
        runnable_method="predict",
    )

    # Create service
    svc = bentoml.Service(service_name, runners=[bento_runner])

    # Create service wrapper for lifecycle management
    service_wrapper = MLXBentoService(runner, service_name, project_name)

    # Define API endpoints
    @svc.api(
        input=JSON(),
        output=JSON(),
        route="/predict",
    )
    async def predict(input_data: dict[str, Any]) -> dict[str, Any]:
        """Prediction endpoint"""
        return service_wrapper.predict(input_data)

    @svc.api(
        input=JSON(),
        output=JSON(),
        route="/health",
    )
    async def health() -> dict[str, Any]:
        """Health check endpoint"""
        return service_wrapper.health_check()

    @svc.api(
        input=JSON(),
        output=JSON(),
        route="/metrics",
    )
    async def metrics() -> dict[str, Any]:
        """Metrics endpoint"""
        memory_usage = runner.get_memory_usage()
        return {
            "service_name": service_name,
            "project_name": project_name,
            "memory_usage": memory_usage,
            "model_loaded": runner.is_loaded,
        }

    # Initialize service on startup
    @svc.on_startup
    async def startup():
        """Service startup handler"""
        logger.info("Starting service: %s", service_name)
        service_wrapper.load()
        logger.info("Service started successfully: %s", service_name)

    @svc.on_shutdown
    async def shutdown():
        """Service shutdown handler"""
        logger.info("Shutting down service: %s", service_name)
        service_wrapper.unload()
        logger.info("Service shutdown complete: %s", service_name)

    return svc


def create_lora_service(
    model_path: str,
    service_name: str = "lora_service",
    project_name: str = "lora-finetuning-mlx",
) -> bentoml.Service:
    """Create a BentoML service specifically for LoRA adapter models

    Args:
        model_path: Path to LoRA adapter
        service_name: Service identifier
        project_name: Project identifier

    Returns:
        BentoML Service instance

    Example:
        >>> service = create_lora_service(
        ...     model_path="outputs/checkpoints/checkpoint_epoch_2",
        ...     service_name="my_lora_adapter",
        ... )
    """
    return create_bentoml_service(
        model_path=model_path,
        service_name=service_name,
        project_name=project_name,
        model_type="lora",
    )
