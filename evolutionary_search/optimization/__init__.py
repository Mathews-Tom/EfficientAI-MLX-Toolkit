"""
Multi-objective optimization for evolutionary architecture search.

This module provides NSGA-II/III implementations, Pareto front tracking,
and surrogate models for efficient architecture evaluation.
"""

from evolutionary_search.optimization.nsga import (
    NSGAII,
    NSGAIII,
    NSGAConfig,
)
from evolutionary_search.optimization.pareto import (
    ParetoFront,
    ParetoArchive,
    dominates,
)
from evolutionary_search.optimization.surrogate import (
    SurrogateModel,
    GaussianProcessSurrogate,
    RandomForestSurrogate,
)

__all__ = [
    "NSGAII",
    "NSGAIII",
    "NSGAConfig",
    "ParetoFront",
    "ParetoArchive",
    "dominates",
    "SurrogateModel",
    "GaussianProcessSurrogate",
    "RandomForestSurrogate",
]
