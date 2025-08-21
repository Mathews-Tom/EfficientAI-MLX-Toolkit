"""
Pruning module for MLX-native model compression.

Provides pruning capabilities including:
- Structured pruning (channels, blocks)
- Unstructured pruning (magnitude, gradient-based)
- Gradual pruning with recovery training
- MLX-optimized pruning methods
"""

from .config import PruningConfig, PruningMethod, PruningSchedule
from .pruner import MLXPruner
from .strategies import (
    MagnitudePruner,
    GradientPruner,
    StructuredPruner,
    UnstructuredPruner,
)
from .scheduler import PruningScheduler, GradualPruningScheduler
from .utils import (
    calculate_sparsity,
    analyze_pruning_impact,
    create_pruning_mask,
    apply_pruning_mask,
)

__all__ = [
    "PruningConfig",
    "PruningMethod",
    "PruningSchedule", 
    "MLXPruner",
    "MagnitudePruner",
    "GradientPruner",
    "StructuredPruner",
    "UnstructuredPruner",
    "PruningScheduler",
    "GradualPruningScheduler",
    "calculate_sparsity",
    "analyze_pruning_impact",
    "create_pruning_mask",
    "apply_pruning_mask",
]