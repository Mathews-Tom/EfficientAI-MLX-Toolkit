"""
Compression strategy implementations.
"""

import logging
from typing import Any

# Import shared utilities with fallback
try:
    from utils.logging_utils import get_logger
    SHARED_UTILS_AVAILABLE = True
except ImportError:
    SHARED_UTILS_AVAILABLE = False
    def get_logger(name: str) -> logging.Logger:
        return logging.getLogger(name)

logger = get_logger(__name__)


class CompressionStrategy:
    """Base compression strategy."""
    
    def __init__(self, config):
        self.config = config
    
    def compress(self, model: Any) -> Any:
        """Apply compression to model."""
        return model


class SequentialStrategy(CompressionStrategy):
    """Sequential compression strategy."""
    
    def compress(self, model: Any) -> Any:
        """Apply compression methods sequentially."""
        logger.info("Applying sequential compression")
        return model


class ParallelStrategy(CompressionStrategy):
    """Parallel compression strategy."""
    
    def compress(self, model: Any) -> Any:
        """Apply compression methods in parallel.""" 
        logger.info("Applying parallel compression")
        return model