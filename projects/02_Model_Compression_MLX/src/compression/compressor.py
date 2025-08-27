"""
Main model compressor that orchestrates different compression techniques.
"""

import logging
from pathlib import Path
from typing import Any

from pruning import MLXPruner
from quantization import MLXQuantizer

from .config import CompressionConfig

# Import shared utilities with fallback
try:
    from utils.logging_utils import get_logger

    SHARED_UTILS_AVAILABLE = True
except ImportError:
    SHARED_UTILS_AVAILABLE = False

    def get_logger(name: str) -> logging.Logger:
        return logging.getLogger(name)


logger = get_logger(__name__)


class ModelCompressor:
    """
    Main model compressor that orchestrates different compression techniques.
    """

    def __init__(self, config: CompressionConfig):
        """Initialize model compressor."""
        self.config = config
        self.results = {}

    def compress(self, model_path: str) -> Any:
        """
        Apply comprehensive compression to a model.

        Args:
            model_path: Path to model to compress

        Returns:
            Compressed model
        """
        logger.info(f"Starting compression of model: {model_path}")

        model = None  # Will be loaded by individual compressors

        for method in self.config.enabled_methods:
            if method == "quantization":
                model = self._apply_quantization(model_path)
            elif method == "pruning":
                model = self._apply_pruning(model_path)
            else:
                logger.warning(f"Unknown compression method: {method}")

        return model

    def _apply_quantization(self, model_path: str) -> Any:
        """Apply quantization compression."""
        logger.info("Applying quantization compression")

        quantizer = MLXQuantizer(self.config.quantization)
        quantized_model = quantizer.quantize(model_path)

        # Store results
        self.results["quantization"] = quantizer.get_quantization_info()

        return quantized_model

    def _apply_pruning(self, model_path: str) -> Any:
        """Apply pruning compression."""
        logger.info("Applying pruning compression")

        from mlx_lm.utils import load as load_model_and_tokenizer

        model, tokenizer = load_model_and_tokenizer(model_path)

        pruner = MLXPruner(self.config.pruning)
        pruned_model = pruner.prune(model)

        # Store results
        self.results["pruning"] = pruner.get_pruning_info()

        return pruned_model

    def get_compression_results(self) -> dict[str, Any]:
        """Get comprehensive compression results."""
        return self.results
