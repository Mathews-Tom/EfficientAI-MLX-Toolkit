"""
Selection operators for evolutionary architecture search.

This module implements various selection strategies for choosing
parents and survivors in the evolutionary process.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod

import numpy as np

from evolutionary_search.search_space import ArchitectureGenome

__all__ = [
    "SelectionOperator",
    "TournamentSelection",
    "RouletteSelection",
    "ElitistSelection",
    "RankSelection",
]


class SelectionOperator(ABC):
    """Base class for selection operators."""

    def __init__(self, seed: int | None = None):
        """
        Initialize selection operator.

        Args:
            seed: Random seed for reproducibility
        """
        self.rng = np.random.RandomState(seed)
        random.seed(seed)

    @abstractmethod
    def select(
        self, population: list[ArchitectureGenome], num_selected: int
    ) -> list[ArchitectureGenome]:
        """
        Select individuals from population.

        Args:
            population: Population to select from
            num_selected: Number of individuals to select

        Returns:
            Selected individuals
        """
        pass


class TournamentSelection(SelectionOperator):
    """
    Tournament selection operator.

    Randomly samples tournament_size individuals and selects the best.
    """

    def __init__(self, tournament_size: int = 3, seed: int | None = None):
        """
        Initialize tournament selection.

        Args:
            tournament_size: Number of individuals in each tournament
            seed: Random seed
        """
        super().__init__(seed)
        self.tournament_size = tournament_size

    def select(
        self, population: list[ArchitectureGenome], num_selected: int
    ) -> list[ArchitectureGenome]:
        """Select individuals via tournament selection."""
        selected: list[ArchitectureGenome] = []

        for _ in range(num_selected):
            # Run tournament
            tournament = random.sample(
                population, min(self.tournament_size, len(population))
            )

            # Select winner (highest fitness)
            winner = max(tournament, key=lambda g: g.fitness_score)
            selected.append(winner)

        return selected


class RouletteSelection(SelectionOperator):
    """
    Roulette wheel selection (fitness proportionate).

    Selection probability proportional to fitness score.
    """

    def select(
        self, population: list[ArchitectureGenome], num_selected: int
    ) -> list[ArchitectureGenome]:
        """Select individuals via roulette wheel selection."""
        # Compute fitness sum
        fitness_scores = np.array([g.fitness_score for g in population])

        # Handle negative fitness
        if np.min(fitness_scores) < 0:
            fitness_scores = fitness_scores - np.min(fitness_scores)

        # Normalize
        total_fitness = np.sum(fitness_scores)
        if total_fitness == 0:
            # Uniform selection if all fitness are zero
            probabilities = np.ones(len(population)) / len(population)
        else:
            probabilities = fitness_scores / total_fitness

        # Select
        selected_indices = self.rng.choice(
            len(population), size=num_selected, replace=True, p=probabilities
        )

        return [population[i] for i in selected_indices]


class ElitistSelection(SelectionOperator):
    """
    Elitist selection operator.

    Selects top individuals by fitness score.
    """

    def __init__(self, elite_fraction: float = 0.1, seed: int | None = None):
        """
        Initialize elitist selection.

        Args:
            elite_fraction: Fraction of population to preserve as elite
            seed: Random seed
        """
        super().__init__(seed)
        self.elite_fraction = elite_fraction

    def select(
        self, population: list[ArchitectureGenome], num_selected: int
    ) -> list[ArchitectureGenome]:
        """Select top individuals."""
        # Sort by fitness
        sorted_population = sorted(
            population, key=lambda g: g.fitness_score, reverse=True
        )

        # Select top individuals
        return sorted_population[:num_selected]


class RankSelection(SelectionOperator):
    """
    Rank-based selection operator.

    Selection probability based on fitness rank rather than absolute fitness.
    """

    def __init__(self, selection_pressure: float = 1.5, seed: int | None = None):
        """
        Initialize rank selection.

        Args:
            selection_pressure: Selection pressure (1.0-2.0, higher = more pressure)
            seed: Random seed
        """
        super().__init__(seed)
        self.selection_pressure = max(1.0, min(2.0, selection_pressure))

    def select(
        self, population: list[ArchitectureGenome], num_selected: int
    ) -> list[ArchitectureGenome]:
        """Select individuals via rank-based selection."""
        # Sort by fitness and assign ranks
        sorted_population = sorted(population, key=lambda g: g.fitness_score)
        n = len(sorted_population)

        # Compute rank-based probabilities
        probabilities = []
        for rank in range(n):
            # Linear ranking: P(i) = (2-SP)/N + 2*i*(SP-1)/(N*(N-1))
            prob = (2 - self.selection_pressure) / n + (
                2 * rank * (self.selection_pressure - 1) / (n * (n - 1))
            )
            probabilities.append(prob)

        # Normalize
        probabilities = np.array(probabilities)
        probabilities = probabilities / np.sum(probabilities)

        # Select
        selected_indices = self.rng.choice(
            n, size=num_selected, replace=True, p=probabilities
        )

        return [sorted_population[i] for i in selected_indices]
