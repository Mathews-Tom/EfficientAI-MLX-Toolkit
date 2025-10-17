"""
Evolutionary Diffusion Architecture Search

This module implements evolutionary algorithms for neural architecture search
optimized for Apple Silicon and MLX framework. It provides multi-objective
optimization for discovering diffusion model architectures with optimal
trade-offs between quality, speed, and memory usage.
"""

from __future__ import annotations

__version__ = "0.1.0"
__all__ = [
    "ArchitectureGenome",
    "EvolutionaryDiffusionSearch",
    "GeneticOperators",
    "PerformanceEvaluator",
]
