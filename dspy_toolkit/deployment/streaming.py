"""
Streaming endpoints and ensemble methods for DSPy Integration Framework.
"""

# Standard library imports
import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, AsyncGenerator

# Third-party imports
import dspy
from fastapi.responses import StreamingResponse
from sse_starlette import EventSourceResponse

from ..exceptions import DSPyIntegrationError, handle_async_dspy_errors
from ..framework import DSPyFramework

logger = logging.getLogger(__name__)


@dataclass
class StreamingConfig:
    """Configuration for streaming endpoints."""

    chunk_size: int = 50
    delay_ms: int = 100
    max_tokens: int = 1000
    timeout_seconds: int = 30
    enable_sse: bool = True


class DSPyStreamingEndpoint:
    """Streaming endpoint for DSPy modules."""

    def __init__(self, framework: DSPyFramework, config: StreamingConfig = None):
        """Initialize streaming endpoint."""

        self.framework = framework
        self.config = config or StreamingConfig()

    @handle_async_dspy_errors()
    async def stream_prediction(
        self,
        project_name: str,
        module_name: str,
        inputs: dict[str, str | int | float | bool],
        stream_options: dict[str, Any | None] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream prediction results."""

        start_time = time.time()

        try:
            # Get module
            module = self.framework.get_project_module(project_name, module_name)
            if not module:
                yield self._format_error(
                    f"Module {module_name} not found for project {project_name}"
                )
                return

            # Merge stream options
            options = {**self.config.__dict__, **(stream_options or {})}

            # Check if module supports streaming
            if hasattr(module, "stream") and callable(module.stream):
                # Use native streaming
                async for chunk in self._stream_native(module, inputs, options):
                    yield chunk
            else:
                # Simulate streaming for non-streaming modules
                async for chunk in self._stream_simulated(module, inputs, options):
                    yield chunk

            # Send completion event
            execution_time = time.time() - start_time
            yield self._format_completion(execution_time)

        except Exception as e:
            logger.error("Streaming prediction failed: %s", e)
            yield self._format_error(str(e))

    async def _stream_native(
        self,
        module: dspy.Module,
        inputs: dict[str, str | int | float | bool],
        options: dict[str, str | int | float | bool],
    ) -> AsyncGenerator[str, None]:
        """Stream using native module streaming."""
        try:
            async for result in module.stream(**inputs):
                yield self._format_chunk(result, "native")
                await asyncio.sleep(options.get("delay_ms", 100) / 1000)

        except Exception as e:
            logger.error("Native streaming failed: %s", e)
            yield self._format_error(str(e))

    async def _stream_simulated(
        self,
        module: dspy.Module,
        inputs: dict[str, str | int | float | bool],
        options: dict[str, str | int | float | bool],
    ) -> AsyncGenerator[str, None]:
        """Simulate streaming for non-streaming modules."""
        try:
            # Execute module
            result = await self._execute_module_async(module, inputs)

            # Convert result to string for streaming
            result_str = self._format_result_for_streaming(result)

            # Stream in chunks
            chunk_size = options.get("chunk_size", 50)
            delay_ms = options.get("delay_ms", 100)

            for i in range(0, len(result_str), chunk_size):
                chunk = result_str[i : i + chunk_size]
                yield self._format_chunk(chunk, "simulated", i // chunk_size)
                await asyncio.sleep(delay_ms / 1000)

        except Exception as e:
            logger.error("Simulated streaming failed: %s", e)
            yield self._format_error(str(e))

    async def _execute_module_async(
        self, module: dspy.Module, inputs: dict[str, str | int | float | bool]
    ) -> Any:
        """Execute DSPy module asynchronously."""
        try:
            if hasattr(module, "__call__") and asyncio.iscoroutinefunction(module.__call__):
                return await module(**inputs)
            else:
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, lambda: module(**inputs))
        except Exception as e:
            raise DSPyIntegrationError(f"Module execution failed: {e}") from e

    def _format_result_for_streaming(self, result: Any) -> str:
        """Format result for streaming."""
        if hasattr(result, "__dict__"):
            return json.dumps(result.__dict__, indent=2)
        elif isinstance(result, dict):
            return json.dumps(result, indent=2)
        else:
            return str(result)

    def _format_chunk(self, content: Any, stream_type: str, index: int = 0) -> str:
        """Format chunk for streaming."""
        chunk_data = {
            "type": "chunk",
            "content": content,
            "stream_type": stream_type,
            "index": index,
            "timestamp": time.time(),
        }

        if self.config.enable_sse:
            return f"data: {json.dumps(chunk_data)}\n\n"
        else:
            return json.dumps(chunk_data) + "\n"

    def _format_error(self, error_message: str) -> str:
        """Format error for streaming."""
        error_data = {
            "type": "error",
            "message": error_message,
            "timestamp": time.time(),
        }

        if self.config.enable_sse:
            return f"data: {json.dumps(error_data)}\n\n"
        else:
            return json.dumps(error_data) + "\n"

    def _format_completion(self, execution_time: float) -> str:
        """Format completion event."""
        completion_data = {
            "type": "completion",
            "execution_time": execution_time,
            "timestamp": time.time(),
        }

        if self.config.enable_sse:
            return f"data: {json.dumps(completion_data)}\n\n"
        else:
            return json.dumps(completion_data) + "\n"


class DSPyEnsembleStreaming:
    """Ensemble streaming for multiple DSPy modules."""

    def __init__(
        self,
        frameworks: list[DSPyFramework],
        weights: list[float | None] = None,
        config: StreamingConfig = None,
    ):
        """Initialize ensemble streaming."""
        if not frameworks:
            raise DSPyIntegrationError("At least one framework is required for ensemble")

        self.frameworks = frameworks
        self.weights = weights or [1.0 / len(frameworks)] * len(frameworks)
        self.config = config or StreamingConfig()

        if len(self.weights) != len(self.frameworks):
            raise DSPyIntegrationError("Number of weights must match number of frameworks")

    async def stream_ensemble_prediction(
        self,
        project_name: str,
        module_name: str,
        inputs: dict[str, str | int | float | bool],
    ) -> AsyncGenerator[str, None]:
        """Stream ensemble predictions."""
        try:
            # Start all predictions concurrently
            tasks = []
            for i, framework in enumerate(self.frameworks):
                task = asyncio.create_task(
                    self._get_framework_prediction(framework, project_name, module_name, inputs, i)
                )
                tasks.append(task)

            # Stream results as they become available
            completed_results = []

            for task in asyncio.as_completed(tasks):
                try:
                    framework_id, result = await task
                    completed_results.append((framework_id, result))

                    # Stream intermediate result
                    yield self._format_ensemble_chunk(framework_id, result, len(completed_results))

                except Exception as e:
                    logger.error("Framework prediction failed: %s", e)
                    yield self._format_ensemble_error(f"Framework prediction failed: {e}")

            # Combine final results
            if completed_results:
                final_result = self._combine_ensemble_results(completed_results)
                yield self._format_ensemble_completion(final_result)
            else:
                yield self._format_ensemble_error("All framework predictions failed")

        except Exception as e:
            logger.error("Ensemble streaming failed: %s", e)
            yield self._format_ensemble_error(str(e))

    async def _get_framework_prediction(
        self,
        framework: DSPyFramework,
        project_name: str,
        module_name: str,
        inputs: dict[str, str | int | float | bool],
        framework_id: int,
    ) -> tuple:
        """Get prediction from a single framework."""
        try:
            module = framework.get_project_module(project_name, module_name)
            if not module:
                raise DSPyIntegrationError(f"Module not found in framework {framework_id}")

            # Execute module
            if hasattr(module, "__call__") and asyncio.iscoroutinefunction(module.__call__):
                result = await module(**inputs)
            else:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, lambda: module(**inputs))

            return framework_id, result

        except Exception as e:
            logger.error("Framework %d prediction failed: %s", framework_id, e)
            raise

    def _combine_ensemble_results(
        self, results: list[tuple]
    ) -> dict[str, str | int | float | bool]:
        """Combine ensemble results using weights."""
        try:
            combined = {}

            for framework_id, result in results:
                weight = self.weights[framework_id]

                # Convert result to dict format
                if hasattr(result, "__dict__"):
                    result_dict = result.__dict__
                elif isinstance(result, dict):
                    result_dict = result
                else:
                    result_dict = {"result": str(result)}

                # Weighted combination
                for key, value in result_dict.items():
                    if key not in combined:
                        combined[key] = []
                    combined[key].append((value, weight))

            # Calculate weighted averages or consensus
            final_result = {}
            for key, values in combined.items():
                if values:
                    try:
                        # Try numeric weighted average
                        weighted_sum = sum(float(v) * w for v, w in values)
                        total_weight = sum(w for v, w in values)
                        final_result[key] = weighted_sum / total_weight if total_weight > 0 else 0
                    except (ValueError, TypeError):
                        # Use majority voting for non-numeric values
                        final_result[key] = max(values, key=lambda x: x[1])[0]

            return final_result

        except Exception as e:
            logger.error("Failed to combine ensemble results: %s", e)
            return {"error": str(e)}

    def _format_ensemble_chunk(self, framework_id: int, result: Any, completed_count: int) -> str:
        """Format ensemble chunk."""
        chunk_data = {
            "type": "ensemble_chunk",
            "framework_id": framework_id,
            "result": result.__dict__ if hasattr(result, "__dict__") else result,
            "completed_count": completed_count,
            "total_frameworks": len(self.frameworks),
            "timestamp": time.time(),
        }

        if self.config.enable_sse:
            return f"data: {json.dumps(chunk_data)}\n\n"
        else:
            return json.dumps(chunk_data) + "\n"

    def _format_ensemble_completion(self, final_result: dict[str, str | int | float | bool]) -> str:
        """Format ensemble completion."""
        completion_data = {
            "type": "ensemble_completion",
            "final_result": final_result,
            "frameworks_used": len(self.frameworks),
            "timestamp": time.time(),
        }

        if self.config.enable_sse:
            return f"data: {json.dumps(completion_data)}\n\n"
        else:
            return json.dumps(completion_data) + "\n"

    def _format_ensemble_error(self, error_message: str) -> str:
        """Format ensemble error."""
        error_data = {
            "type": "ensemble_error",
            "message": error_message,
            "timestamp": time.time(),
        }

        if self.config.enable_sse:
            return f"data: {json.dumps(error_data)}\n\n"
        else:
            return json.dumps(error_data) + "\n"


def create_streaming_endpoint(
    framework: DSPyFramework, config: StreamingConfig | None = None
) -> DSPyStreamingEndpoint:
    """Create a streaming endpoint for DSPy framework."""
    try:
        endpoint = DSPyStreamingEndpoint(framework, config)
        logger.info("Created DSPy streaming endpoint")
        return endpoint

    except Exception as e:
        logger.error("Failed to create streaming endpoint: %s", e)
        raise DSPyIntegrationError("Streaming endpoint creation failed") from e


def create_ensemble_streaming(
    frameworks: list[DSPyFramework],
    weights: list[float | None] = None,
    config: StreamingConfig | None = None,
) -> DSPyEnsembleStreaming:
    """Create ensemble streaming for multiple frameworks."""
    try:
        ensemble = DSPyEnsembleStreaming(frameworks, weights, config)
        logger.info("Created DSPy ensemble streaming with %d frameworks", len(frameworks))
        return ensemble

    except Exception as e:
        logger.error("Failed to create ensemble streaming: %s", e)
        raise DSPyIntegrationError("Ensemble streaming creation failed") from e


# Utility functions for streaming integration
async def stream_to_fastapi_response(
    stream_generator: AsyncGenerator[str, None], media_type: str = "text/plain"
) -> StreamingResponse:
    """Convert async generator to FastAPI StreamingResponse."""

    return StreamingResponse(stream_generator, media_type=media_type)


async def stream_to_sse_response(
    stream_generator: AsyncGenerator[str, None]
) -> EventSourceResponse:
    """Convert async generator to Server-Sent Events response."""

    return EventSourceResponse(stream_generator)
