"""
Automated hyperparameter optimization for LoRA fine-tuning.

Provides intelligent hyperparameter search with Bayesian optimization,
early stopping, and Apple Silicon performance awareness.
"""

from .tuner import AutoTuner, OptimizationResult, HyperparameterSpace
from .search import (
    SearchStrategy,
    RandomSearch,
    BayesianOptimization,
    GridSearch,
    HalvingRandomSearch,
)
from .objectives import (
    OptimizationObjective,
    PerplexityObjective,
    BLEUObjective,
    ROUGEObjective,
    MultiObjective,
)

__all__ = [
    "AutoTuner",
    "OptimizationResult", 
    "HyperparameterSpace",
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