"""
MLX LLM Provider for Apple Silicon optimization in DSPy Integration Framework.
"""

# Standard library imports
import json
import logging
import time

# Optional third-party imports
try:
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import generate, load, stream_generate

    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    mx = None
    nn = None

try:
    import litellm
    from litellm import Choices, CustomLLM, Message, ModelResponse, Usage

    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False
    CustomLLM = object
    ModelResponse = None
    Choices = None
    Message = None
    Usage = None

from ..exceptions import HardwareCompatibilityError, MLXProviderError
from ..types import DSPyConfig, HardwareInfo
from .base_provider import BaseLLMProvider

logger = logging.getLogger(__name__)


class MLXLLMProvider(BaseLLMProvider, CustomLLM if LITELLM_AVAILABLE else object):
    """Custom LiteLLM provider for MLX models optimized for Apple Silicon."""

    def __init__(self, config: DSPyConfig):
        """Initialize MLX LLM provider."""
        if not MLX_AVAILABLE:
            raise MLXProviderError("MLX is not available. Please install mlx and mlx-lm packages.")

        if not LITELLM_AVAILABLE:
            raise MLXProviderError("LiteLLM is not available. Please install litellm package.")

        super().__init__(config)
        CustomLLM.__init__(self)

        self.model = None
        self.tokenizer = None
        self._load_mlx_model()
        self.initialize()

    def _load_mlx_model(self) -> None:
        """Load MLX model with Apple Silicon optimizations."""
        try:
            # Extract model path from config (remove mlx/ prefix if present)
            model_path = self.config.model_name.replace("mlx/", "")

            logger.info("Loading MLX model: %s", model_path)
            self.model, self.tokenizer = load(model_path)

            # Configure MLX for optimal Apple Silicon performance
            if mx.metal.is_available():
                # Set memory limit based on optimization level
                memory_limits = {
                    0: 4 * 1024**3,  # 4GB - Conservative
                    1: 8 * 1024**3,  # 8GB - Balanced
                    2: 12 * 1024**3,  # 12GB - Aggressive
                    3: 16 * 1024**3,  # 16GB - Maximum
                }
                memory_limit = memory_limits.get(self.config.optimization_level, 8 * 1024**3)
                mx.metal.set_memory_limit(memory_limit)

                logger.info("Set MLX memory limit to %dGB", memory_limit // 1024**3)

        except Exception as e:
            raise MLXProviderError(f"Failed to load MLX model {self.config.model_name}: {e}")

    def detect_hardware(self) -> HardwareInfo:
        """Detect Apple Silicon capabilities."""
        if not MLX_AVAILABLE:
            return HardwareInfo(
                device_type="cpu",
                total_memory=0,
                available_memory=0,
                metal_available=False,
                mps_available=False,
                optimization_level=0,
            )

        try:
            metal_available = mx.metal.is_available()

            # Try to detect MPS availability
            mps_available = False
            try:
                import torch

                mps_available = torch.backends.mps.is_available()
            except ImportError:
                pass

            # Estimate memory (simplified detection)
            total_memory = 16  # Default assumption for Apple Silicon
            available_memory = 12  # Conservative estimate

            if metal_available:
                try:
                    memory_limit = mx.metal.get_memory_limit()
                    total_memory = memory_limit // 1024**3
                    available_memory = int(total_memory * 0.8)  # 80% available
                except:
                    pass

            # Detect chip type (simplified)
            device_type = "apple_silicon"
            if metal_available:
                device_type = "m1"  # Default to M1, could be enhanced with more detection

            return HardwareInfo(
                device_type=device_type,
                total_memory=total_memory,
                available_memory=available_memory,
                metal_available=metal_available,
                mps_available=mps_available,
                optimization_level=self.config.optimization_level,
            )

        except Exception as e:
            logger.warning("Hardware detection failed: %s", e)
            return HardwareInfo(
                device_type="cpu",
                total_memory=8,
                available_memory=6,
                metal_available=False,
                mps_available=False,
                optimization_level=0,
            )

    def is_available(self) -> bool:
        """Check if MLX provider is available and functional."""
        return (
            MLX_AVAILABLE
            and LITELLM_AVAILABLE
            and self.model is not None
            and self.tokenizer is not None
        )

    def completion(self, *args, **kwargs) -> ModelResponse:
        """Generate completion using MLX model."""
        if not self.is_available():
            raise MLXProviderError("MLX provider is not available")

        try:
            # Extract parameters from kwargs
            messages = kwargs.get("messages", [])
            max_tokens = kwargs.get("max_tokens", 512)
            temperature = kwargs.get("temperature", 0.7)
            stream = kwargs.get("stream", False)

            # Get the prompt from messages
            if messages:
                prompt = messages[-1].get("content", "")
            else:
                prompt = kwargs.get("prompt", "")

            if not prompt:
                raise MLXProviderError("No prompt provided")

            # Generate response using MLX
            start_time = time.time()

            if stream:
                # Use streaming generation
                response_text = ""
                for token in stream_generate(
                    self.model,
                    self.tokenizer,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temp=temperature,
                ):
                    response_text += token
            else:
                # Use regular generation
                response_text = generate(
                    self.model,
                    self.tokenizer,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temp=temperature,
                )

            generation_time = time.time() - start_time

            # Calculate token usage (approximate)
            prompt_tokens = len(self.tokenizer.encode(prompt))
            completion_tokens = len(self.tokenizer.encode(response_text))
            total_tokens = prompt_tokens + completion_tokens

            # Create LiteLLM compatible response
            return self._format_response(
                response_text,
                prompt_tokens,
                completion_tokens,
                total_tokens,
                generation_time,
            )

        except Exception as e:
            logger.error("MLX completion failed: %s", e)
            raise MLXProviderError("Completion generation failed") from e

    def _format_response(
        self,
        response: str,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        generation_time: float,
    ) -> ModelResponse:
        """Format response for LiteLLM compatibility."""
        try:
            return ModelResponse(
                id=f"mlx-{int(time.time())}",
                choices=[
                    Choices(
                        message=Message(content=response, role="assistant"),
                        finish_reason="stop",
                        index=0,
                    )
                ],
                created=int(time.time()),
                model=self.config.model_name,
                object="chat.completion",
                usage=Usage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                ),
                # Add custom metadata
                _hidden_params={
                    "generation_time": generation_time,
                    "hardware_type": self.hardware_info.device_type,
                    "optimization_level": self.config.optimization_level,
                },
            )
        except Exception as e:
            logger.error("Response formatting failed: %s", e)
            raise MLXProviderError("Failed to format response") from e

    async def acompletion(self, *args, **kwargs) -> ModelResponse:
        """Async completion wrapper."""
        # For now, we'll use sync completion
        # In the future, this could be enhanced with proper async MLX support
        return self.completion(*args, **kwargs)

    def get_model_info(self) -> dict[str, str | int | float | bool]:
        """Get information about the loaded model."""
        if not self.model:
            return {}

        try:
            # Get basic model information
            info = {
                "model_name": self.config.model_name,
                "model_type": type(self.model).__name__,
                "hardware_optimized": self.hardware_info.metal_available,
                "memory_limit": (mx.metal.get_memory_limit() if mx.metal.is_available() else 0),
            }

            # Add model-specific parameters if available
            if hasattr(self.model, "args"):
                info.update(
                    {
                        "model_args": str(self.model.args),
                    }
                )

            return info

        except Exception as e:
            logger.warning("Failed to get model info: %s", e)
            return {"error": str(e)}

    def benchmark_performance(self, test_prompt: str = "Hello, world!") -> dict[str, float]:
        """Benchmark the MLX provider performance."""
        if not self.is_available():
            raise MLXProviderError("MLX provider is not available for benchmarking")

        try:
            # Warm up
            self.completion(messages=[{"content": test_prompt}], max_tokens=10)

            # Benchmark multiple runs
            times = []
            for _ in range(5):
                start_time = time.time()
                response = self.completion(messages=[{"content": test_prompt}], max_tokens=50)
                end_time = time.time()
                times.append(end_time - start_time)

            # Calculate statistics
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)

            # Estimate tokens per second
            tokens_per_second = 50 / avg_time  # Approximate based on max_tokens

            return {
                "average_time": avg_time,
                "min_time": min_time,
                "max_time": max_time,
                "tokens_per_second": tokens_per_second,
                "hardware_type": self.hardware_info.device_type,
                "optimization_level": self.config.optimization_level,
            }

        except Exception as e:
            logger.error("Benchmarking failed: %s", e)
            raise MLXProviderError("Performance benchmarking failed") from e


def setup_mlx_provider_for_dspy(config: DSPyConfig) -> None:
    """Setup MLX provider for DSPy integration."""
    try:
        # Create MLX provider instance
        mlx_provider = MLXLLMProvider(config)

        # Register with LiteLLM
        if LITELLM_AVAILABLE:
            litellm.custom_provider_map = [{"provider": "mlx", "custom_handler": mlx_provider}]

            # Configure DSPy to use MLX provider
            import dspy

            dspy.configure(lm=dspy.LM(model=config.model_name))

            logger.info("Successfully configured DSPy with MLX provider: %s", config.model_name)
        else:
            raise MLXProviderError("LiteLLM not available for DSPy integration")

    except Exception as e:
        logger.error("Failed to setup MLX provider for DSPy: %s", e)
        raise MLXProviderError("DSPy MLX setup failed") from e
