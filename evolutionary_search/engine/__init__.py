"""
Evolution engine for evolutionary architecture search.

This module provides the main evolution engine that coordinates
genetic operators and manages the evolutionary process.
"""

from evolutionary_search.engine.evolution import (
    EvolutionEngine,
    EvolutionConfig,
    EvolutionResult,
)

__all__ = [
    "EvolutionEngine",
    "EvolutionConfig",
    "EvolutionResult",
]
