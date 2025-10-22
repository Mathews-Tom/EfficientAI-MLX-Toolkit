"""BentoML Model Runner

Runner implementation for MLX models with Apple Silicon optimization.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import bentoml

logger = logging.getLogger(__name__)


class MLXModelRunner:
    """Runner for MLX models with Apple Silicon optimization

    This runner handles model loading, inference, and resource management
    for MLX models with unified memory and MPS acceleration support.
    """

    def __init__(
        self,
        model_path: str | Path,
        model_config: dict[str, Any] | None = None,
        enable_mps: bool = True,
        enable_unified_memory: bool = True,
    ):
        """Initialize MLX model runner

        Args:
            model_path: Path to MLX model files
            model_config: Optional model configuration
            enable_mps: Enable Metal Performance Shaders acceleration
            enable_unified_memory: Enable unified memory optimization
        """
        self.model_path = Path(model_path)
        self.model_config = model_config or {}
        self.enable_mps = enable_mps
        self.enable_unified_memory = enable_unified_memory
        self.model: Any | None = None
        self._is_loaded = False
        self._mlx_available = self._check_mlx_available()

    def _check_mlx_available(self) -> bool:
        """Check if MLX is available"""
        try:
            import mlx.core as mx
            return True
        except ImportError:
            logger.warning("MLX not available, falling back to standard inference")
            return False

    def load_model(self) -> None:
        """Load MLX model with Apple Silicon optimizations"""
        if self._is_loaded:
            logger.debug("Model already loaded")
            return

        try:
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model path not found: {self.model_path}")

            if self._mlx_available:
                import mlx.core as mx

                logger.info("Loading MLX model from %s", self.model_path)

                # Configure MLX for optimal performance
                if self.enable_mps:
                    logger.debug("MPS acceleration enabled")

                # Load model weights
                # This is extensible - subclasses can implement specific loading
                self.model = self._load_mlx_model()

                logger.info("MLX model loaded successfully")
            else:
                # Fallback to standard loading
                logger.info("Loading model with standard method")
                self.model = self._load_standard_model()

            self._is_loaded = True

        except Exception as e:
            logger.error("Failed to load model: %s", e)
            raise

    def _load_mlx_model(self) -> Any:
        """Load model using MLX

        Returns:
            Loaded MLX model

        Note:
            Override this method in subclasses for specific model types
        """
        import mlx.core as mx

        # Placeholder for MLX model loading
        # Subclasses should implement specific loading logic
        logger.debug("Base MLX model loading - override for specific model types")
        return {}

    def _load_standard_model(self) -> Any:
        """Load model using standard PyTorch/transformers

        Returns:
            Loaded standard model
        """
        logger.debug("Standard model loading - override for specific model types")
        return {}

    def predict(self, input_data: Any) -> Any:
        """Run MLX-optimized inference

        Args:
            input_data: Model input

        Returns:
            Model predictions
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        try:
            if self._mlx_available:
                return self._mlx_predict(input_data)
            else:
                return self._standard_predict(input_data)

        except Exception as e:
            logger.error("Prediction failed: %s", e)
            raise

    def _mlx_predict(self, input_data: Any) -> Any:
        """MLX-specific prediction

        Args:
            input_data: Model input

        Returns:
            Predictions using MLX
        """
        import mlx.core as mx

        # Base implementation - override in subclasses
        logger.debug("Processing input with MLX")
        return {"prediction": None, "mlx_optimized": True}

    def _standard_predict(self, input_data: Any) -> Any:
        """Standard prediction without MLX

        Args:
            input_data: Model input

        Returns:
            Predictions using standard framework
        """
        logger.debug("Running standard prediction")
        return {"prediction": None, "mlx_optimized": False}

    def unload_model(self) -> None:
        """Unload MLX model and free memory"""
        if not self._is_loaded:
            logger.debug("Model not loaded, nothing to unload")
            return

        try:
            if self._mlx_available:
                import mlx.core as mx

                # Clear MLX cache
                mx.metal.clear_cache()
                logger.debug("Cleared MLX cache")

            # Release model reference
            self.model = None
            self._is_loaded = False

            logger.info("Model unloaded successfully")

        except Exception as e:
            logger.error("Failed to unload model: %s", e)
            raise

    def get_memory_usage(self) -> dict[str, float]:
        """Get current memory usage

        Returns:
            Dictionary with memory usage metrics (in MB)
        """
        if not self._mlx_available:
            return {"total_mb": 0.0, "mlx_available": False}

        try:
            import mlx.core as mx

            # Get MLX memory stats
            cache_size = mx.metal.get_cache_memory()
            active_size = mx.metal.get_active_memory()

            return {
                "cache_mb": cache_size / (1024 * 1024),
                "active_mb": active_size / (1024 * 1024),
                "total_mb": (cache_size + active_size) / (1024 * 1024),
                "mlx_available": True,
            }

        except Exception as e:
            logger.warning("Failed to get memory usage: %s", e)
            return {"total_mb": 0.0, "error": str(e)}

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self._is_loaded


class LoRAModelRunner(MLXModelRunner):
    """Runner specifically for LoRA adapter models"""

    def _load_mlx_model(self) -> Any:
        """Load LoRA adapter using MLX

        Returns:
            Loaded LoRA adapter model
        """
        try:
            # Import mlx_lm for LoRA models
            from mlx_lm import load

            logger.info("Loading LoRA adapter from %s", self.model_path)

            # Load base model and adapter
            model, tokenizer = load(str(self.model_path))

            return {"model": model, "tokenizer": tokenizer}

        except ImportError:
            logger.error("mlx_lm not available for LoRA models")
            raise
        except Exception as e:
            logger.error("Failed to load LoRA model: %s", e)
            raise

    def _mlx_predict(self, input_data: Any) -> Any:
        """LoRA-specific prediction

        Args:
            input_data: Text input or dict with 'text' key

        Returns:
            Model predictions
        """
        if not self.model:
            raise RuntimeError("Model not loaded")

        try:
            # Extract text from input
            if isinstance(input_data, dict):
                text = input_data.get("text", "")
            else:
                text = str(input_data)

            # Generate response
            from mlx_lm import generate

            model = self.model["model"]
            tokenizer = self.model["tokenizer"]

            # Generate
            max_tokens = input_data.get("max_tokens", 100) if isinstance(input_data, dict) else 100
            temp = input_data.get("temperature", 0.7) if isinstance(input_data, dict) else 0.7

            response = generate(
                model,
                tokenizer,
                prompt=text,
                max_tokens=max_tokens,
                temp=temp,
            )

            return {
                "prediction": response,
                "mlx_optimized": True,
                "model_type": "lora",
            }

        except Exception as e:
            logger.error("LoRA prediction failed: %s", e)
            raise


def create_runner(
    model_path: str | Path,
    model_type: str = "mlx",
    **kwargs: Any,
) -> MLXModelRunner:
    """Factory function to create appropriate model runner

    Args:
        model_path: Path to model files
        model_type: Model type ('mlx', 'lora', etc.)
        **kwargs: Additional runner-specific arguments

    Returns:
        Appropriate MLXModelRunner instance
    """
    if model_type.lower() == "lora":
        return LoRAModelRunner(model_path, **kwargs)
    elif model_type.lower() == "mlx":
        return MLXModelRunner(model_path, **kwargs)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
