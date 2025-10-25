"""Apple Silicon Optimization Module

Centralized Apple Silicon detection, optimization, and monitoring for the MLOps
infrastructure. This module consolidates all Apple Silicon-related functionality
to avoid duplication across components.

Public API:
    - AppleSiliconDetector: Hardware detection and capability checking
    - AppleSiliconOptimizer: Configuration optimization recommendations
    - AppleSiliconMonitor: Real-time metrics collection and monitoring
    - AppleSiliconMetrics: Metrics data structure

Example:
    >>> from mlops.silicon import AppleSiliconDetector, AppleSiliconOptimizer
    >>>
    >>> detector = AppleSiliconDetector()
    >>> if detector.is_apple_silicon:
    ...     info = detector.get_hardware_info()
    ...     optimizer = AppleSiliconOptimizer(info)
    ...     config = optimizer.get_optimal_config()
    ...     print(f"Recommended workers: {config['workers']}")
"""

from mlops.silicon.detector import AppleSiliconDetector, HardwareInfo
from mlops.silicon.metrics import AppleSiliconMetrics
from mlops.silicon.monitor import AppleSiliconMonitor
from mlops.silicon.optimizer import AppleSiliconOptimizer, OptimalConfig

__all__ = [
    "AppleSiliconDetector",
    "AppleSiliconOptimizer",
    "AppleSiliconMonitor",
    "AppleSiliconMetrics",
    "HardwareInfo",
    "OptimalConfig",
]

__version__ = "1.0.0"
