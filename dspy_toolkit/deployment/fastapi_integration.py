"""
FastAPI integration for DSPy modules with async support.
"""

# Standard library imports
import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Any

# Third-party imports
import dspy
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ..exceptions import DSPyIntegrationError, handle_async_dspy_errors
from ..framework import DSPyFramework
from ..types import DSPyConfig
from .monitoring import DSPyMonitor, PerformanceMetrics

logger = logging.getLogger(__name__)


class DSPyRequest(BaseModel):
    """Base request model for DSPy endpoints."""

    inputs: dict[str, str | int | float | bool] = Field(
        ..., description="Input parameters for DSPy module"
    )
    module_name: str | None = Field(None, description="Specific module name to use")
    project_name: str | None = Field(None, description="Project name for module lookup")
    options: dict[str, str | int | float | bool] = Field(
        default_factory=dict, description="Additional options"
    )


class DSPyResponse(BaseModel):
    """Base response model for DSPy endpoints."""

    outputs: dict[str, str | int | float | bool] = Field(..., description="Output from DSPy module")
    metadata: dict[str, str | int | float | bool] = Field(
        default_factory=dict, description="Response metadata"
    )
    performance: PerformanceMetrics | None = Field(None, description="Performance metrics")
    timestamp: float = Field(default_factory=time.time, description="Response timestamp")


class DSPyStreamRequest(BaseModel):
    """Request model for streaming DSPy endpoints."""

    inputs: dict[str, str | int | float | bool] = Field(
        ..., description="Input parameters for streaming"
    )
    stream_options: dict[str, str | int | float | bool] = Field(
        default_factory=dict, description="Streaming options"
    )


class DSPyHealthResponse(BaseModel):
    """Health check response model."""

    status: str = Field(..., description="Health status")
    framework_health: dict[str, str | int | float | bool] = Field(
        ..., description="Framework health details"
    )
    timestamp: float = Field(default_factory=time.time, description="Health check timestamp")


class DSPyFastAPIApp:
    """FastAPI application wrapper for DSPy modules."""

    def __init__(
        self,
        framework: DSPyFramework,
        title: str = "DSPy Integration API",
        description: str = "API for DSPy-powered intelligent automation",
        version: str = "1.0.0",
    ):
        """Initialize DSPy FastAPI application."""

        # FastAPI availability check removed; integration assumes FastAPI and its dependencies are installed

        self.framework = framework
        self.monitor = DSPyMonitor()

        # Create FastAPI app with lifespan
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # Startup
            logger.info("Starting DSPy FastAPI application")
            await self._startup()
            yield
            # Shutdown
            logger.info("Shutting down DSPy FastAPI application")
            await self._shutdown()

        self.app = FastAPI(title=title, description=description, version=version, lifespan=lifespan)

        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Setup routes
        self._setup_routes()

    async def _startup(self):
        """Application startup tasks."""
        try:
            # Perform framework health check
            health = self.framework.health_check()
            if health["overall_status"] == "unhealthy":
                logger.error("Framework is unhealthy at startup")
                raise DSPyIntegrationError("Framework health check failed")

            logger.info("DSPy FastAPI application started successfully")

        except Exception as e:
            logger.error("Startup failed: %s", e)
            raise

    async def _shutdown(self):
        """Application shutdown tasks."""
        try:
            # Export monitoring data
            await self.monitor.export_metrics()
            logger.info("DSPy FastAPI application shutdown completed")
        except Exception as e:
            logger.error("Shutdown error: %s", e)

    def _setup_routes(self):
        """Setup API routes."""

        @self.app.get("/health", response_model=DSPyHealthResponse)
        async def health_check():
            """Health check endpoint."""
            try:
                framework_health = self.framework.health_check()
                status = (
                    "healthy" if framework_health["overall_status"] == "healthy" else "unhealthy"
                )

                return DSPyHealthResponse(status=status, framework_health=framework_health)
            except Exception as e:
                logger.error("Health check failed: %s", e)
                raise HTTPException(status_code=500, detail="Health check failed") from e

        @self.app.post("/predict", response_model=DSPyResponse)
        async def predict(request: DSPyRequest, background_tasks: BackgroundTasks):
            """General prediction endpoint."""
            return await self._handle_prediction(request, background_tasks)

        @self.app.post("/predict/{project_name}", response_model=DSPyResponse)
        async def predict_project(
            project_name: str, request: DSPyRequest, background_tasks: BackgroundTasks
        ):
            """Project-specific prediction endpoint."""
            request.project_name = project_name
            return await self._handle_prediction(request, background_tasks)

        @self.app.post("/predict/{project_name}/{module_name}", response_model=DSPyResponse)
        async def predict_module(
            project_name: str,
            module_name: str,
            request: DSPyRequest,
            background_tasks: BackgroundTasks,
        ):
            """Module-specific prediction endpoint."""
            request.project_name = project_name
            request.module_name = module_name
            return await self._handle_prediction(request, background_tasks)

        @self.app.post("/stream/{project_name}/{module_name}")
        async def stream_predict(project_name: str, module_name: str, request: DSPyStreamRequest):
            """Streaming prediction endpoint."""
            return StreamingResponse(
                self._handle_streaming_prediction(project_name, module_name, request),
                media_type="text/plain",
            )

        @self.app.get("/modules")
        async def list_modules():
            """List available modules."""
            try:
                modules = self.framework.module_manager.list_modules()
                return {"modules": modules}
            except Exception as e:
                logger.error("Failed to list modules: %s", e)
                raise HTTPException(status_code=500, detail="Failed to list modules") from e

        @self.app.get("/signatures/{project_name}")
        async def list_project_signatures(project_name: str):
            """List signatures for a project."""
            try:
                signatures = self.framework.signature_registry.list_project_signatures(project_name)
                return {"project": project_name, "signatures": signatures}
            except Exception as e:
                logger.error("Failed to list signatures for %s: %s", project_name, e)
                raise HTTPException(status_code=500, detail="Failed to list signatures") from e

        @self.app.get("/stats")
        async def get_framework_stats():
            """Get framework statistics."""
            try:
                stats = self.framework.get_framework_stats()
                return stats
            except Exception as e:
                logger.error("Failed to get framework stats: %s", e)
                raise HTTPException(status_code=500, detail="Failed to get stats") from e

        @self.app.get("/metrics")
        async def get_performance_metrics():
            """Get performance metrics."""
            try:
                metrics = await self.monitor.get_metrics()
                return metrics
            except Exception as e:
                logger.error("Failed to get metrics: %s", e)
                raise HTTPException(status_code=500, detail="Failed to get metrics") from e

    @handle_async_dspy_errors()
    async def _handle_prediction(
        self, request: DSPyRequest, background_tasks: BackgroundTasks
    ) -> DSPyResponse:
        """Handle prediction requests."""
        start_time = time.time()

        try:
            # Get the appropriate module
            if request.project_name and request.module_name:
                module = self.framework.get_project_module(
                    request.project_name, request.module_name
                )
                if not module:
                    raise HTTPException(
                        status_code=404,
                        detail="Module not found for project",
                    )
            else:
                # Use a default module or create one dynamically
                raise HTTPException(
                    status_code=400, detail="Project name and module name are required"
                )

            # Execute the module
            result = await self._execute_module_async(module, request.inputs)

            # Calculate performance metrics
            execution_time = time.time() - start_time
            performance = PerformanceMetrics(
                execution_time=execution_time,
                input_tokens=len(str(request.inputs)),
                output_tokens=len(str(result)),
                memory_usage=0,  # Would need actual memory monitoring
                timestamp=time.time(),
            )

            # Record metrics in background
            background_tasks.add_task(
                self.monitor.record_request,
                request.project_name or "unknown",
                request.module_name or "unknown",
                performance,
            )

            # Format response
            if hasattr(result, "__dict__"):
                outputs = result.__dict__
            elif isinstance(result, dict):
                outputs = result
            else:
                outputs = {"result": str(result)}

            return DSPyResponse(
                outputs=outputs,
                metadata={
                    "project_name": request.project_name,
                    "module_name": request.module_name,
                    "options": request.options,
                },
                performance=performance,
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error("Prediction failed: %s", e)
            raise HTTPException(status_code=500, detail="Prediction failed") from e

    async def _execute_module_async(
        self, module: dspy.Module, inputs: dict[str, str | int | float | bool]
    ) -> Any:
        """Execute DSPy module asynchronously."""
        try:
            # Check if module has async support
            if hasattr(module, "__call__") and asyncio.iscoroutinefunction(module.__call__):
                return await module(**inputs)
            else:
                # Run in thread pool for sync modules
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, lambda: module(**inputs))

        except Exception as e:
            logger.error("Module execution failed: %s", e)
            raise DSPyIntegrationError("Module execution failed") from e

    async def _handle_streaming_prediction(
        self, project_name: str, module_name: str, request: DSPyStreamRequest
    ):
        """Handle streaming prediction requests."""
        try:
            module = self.framework.get_project_module(project_name, module_name)
            if not module:
                yield "data: {{'error': 'Module not found'}}\n\n"
                return

            result = await self._execute_module_async(module, request.inputs)

            # Stream the result in chunks
            result_str = str(result)
            chunk_size = request.stream_options.get("chunk_size", 50)

            for i in range(0, len(result_str), chunk_size):
                chunk = result_str[i : i + chunk_size]
                yield f"data: {{'chunk': '{chunk}', 'index': {i // chunk_size}}}\n\n"
                await asyncio.sleep(0.1)  # Small delay for streaming effect

            yield "data: {{'done': true}}\n\n"

        except Exception as e:
            logger.error("Streaming prediction failed: %s", e)
            yield "data: {'error': 'Streaming prediction failed'}\n\n"

    def get_app(self) -> FastAPI:
        """Get the FastAPI application instance."""
        return self.app


def create_dspy_app(config: DSPyConfig, **kwargs) -> DSPyFastAPIApp:
    """Create a DSPy FastAPI application."""
    try:
        # Initialize framework
        framework = DSPyFramework(config)

        # Create FastAPI app
        app = DSPyFastAPIApp(framework, **kwargs)

        logger.info("DSPy FastAPI application created successfully")
        return app

    except Exception as e:
        logger.error("Failed to create DSPy FastAPI application: %s", e)
        raise DSPyIntegrationError("FastAPI app creation failed") from e


# Utility functions for common deployment patterns
async def create_ensemble_endpoint(
    frameworks: list[DSPyFramework], weights: list[float | None] = None
):
    """Create an ensemble endpoint that combines multiple DSPy frameworks."""
    if not frameworks:
        raise DSPyIntegrationError("At least one framework is required for ensemble")

    if weights is None:
        weights = [1.0 / len(frameworks)] * len(frameworks)

    if len(weights) != len(frameworks):
        raise DSPyIntegrationError("Number of weights must match number of frameworks")

    async def ensemble_predict(
        inputs: dict[str, str | int | float | bool]
    ) -> dict[str, str | int | float | bool]:
        """Ensemble prediction function."""
        results = []

        # Get predictions from all frameworks
        for framework in frameworks:
            try:
                # This would need to be implemented based on specific ensemble logic
                result = await framework.predict_async(inputs)  # This method would need to be added
                results.append(result)
            except Exception as e:
                logger.warning("Framework prediction failed in ensemble: %s", e)
                results.append(None)

        # Combine results based on weights
        # This is a simplified combination - real ensemble would be more sophisticated
        combined_result = {}
        for i, (result, weight) in enumerate(zip(results, weights)):
            if result:
                for key, value in result.items():
                    if key not in combined_result:
                        combined_result[key] = []
                    combined_result[key].append((value, weight))

        # Weighted average or voting logic would go here
        final_result = {}
        for key, values in combined_result.items():
            if values:
                # Simple weighted average for numeric values
                try:
                    weighted_sum = sum(float(v) * w for v, w in values)
                    total_weight = sum(w for v, w in values)
                    final_result[key] = weighted_sum / total_weight if total_weight > 0 else 0
                except (ValueError, TypeError):
                    # For non-numeric values, use majority voting
                    final_result[key] = max(values, key=lambda x: x[1])[0]

        return final_result

    return ensemble_predict
