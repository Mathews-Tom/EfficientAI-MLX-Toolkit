"""
Automated hyperparameter optimization for LoRA fine-tuning.

Provides intelligent hyperparameter search with Bayesian optimization,
early stopping, and Apple Silicon performance awareness.
"""

from .objectives import (
    BLEUObjective,
    MultiObjective,
    OptimizationObjective,
    PerplexityObjective,
    ROUGEObjective,
)
from .search import (
    BayesianOptimization,
    GridSearch,
    HalvingRandomSearch,
    RandomSearch,
    SearchStrategy,
)
from .tuner import AutoTuner, HyperparameterSpace, OptimizationResult, run_optimization

__all__ = [
    "AutoTuner",
    "OptimizationResult",
    "HyperparameterSpace",
    "run_optimization",
    "SearchStrategy",
    "RandomSearch",
    "BayesianOptimization",
    "GridSearch",
    "HalvingRandomSearch",
    "OptimizationObjective",
    "PerplexityObjective",
    "BLEUObjective",
    "ROUGEObjective",
    "MultiObjective",
]
