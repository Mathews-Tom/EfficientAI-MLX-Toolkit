"""
Pruning scheduler implementations.
"""

import logging
from typing import Dict, Any, List

# Import shared utilities with fallback
try:
    from utils.logging_utils import get_logger
    SHARED_UTILS_AVAILABLE = True
except ImportError:
    SHARED_UTILS_AVAILABLE = False
    def get_logger(name: str) -> logging.Logger:
        return logging.getLogger(name)

logger = get_logger(__name__)


class PruningScheduler:
    """Base pruning scheduler."""
    
    def __init__(self, config):
        self.config = config
    
    def get_schedule_info(self) -> Dict[str, Any]:
        """Get scheduler information."""
        return {"scheduler_type": "base"}


class GradualPruningScheduler(PruningScheduler):
    """Gradual pruning scheduler."""
    
    def __init__(self, config):
        super().__init__(config)
    
    def get_schedule_info(self) -> Dict[str, Any]:
        """Get scheduler information."""
        return {
            "scheduler_type": "gradual",
            "start_epoch": self.config.start_epoch,
            "end_epoch": self.config.end_epoch,
            "frequency": self.config.frequency,
        }