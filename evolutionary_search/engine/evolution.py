"""
Main evolution engine for evolutionary architecture search.

This module implements the core evolution loop with convergence
criteria, elitism, and diversity preservation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from evolutionary_search.fitness import FitnessEvaluator, FitnessMetrics
from evolutionary_search.operators import (
    CompositeMutation,
    CrossoverOperator,
    ElitistSelection,
    MutationOperator,
    SelectionOperator,
    UniformCrossover,
)
from evolutionary_search.population import DiversityMetrics, PopulationGenerator
from evolutionary_search.search_space import ArchitectureGenome, SearchSpaceConfig

__all__ = [
    "EvolutionConfig",
    "EvolutionResult",
    "EvolutionEngine",
]


@dataclass
class EvolutionConfig:
    """
    Configuration for evolutionary search.

    Attributes:
        population_size: Number of individuals in population
        num_generations: Maximum number of generations
        elite_size: Number of elite individuals to preserve
        crossover_rate: Probability of crossover
        mutation_rate: Probability of mutation
        convergence_threshold: Fitness improvement threshold for convergence
        convergence_generations: Generations without improvement before stopping
        diversity_threshold: Minimum diversity to maintain
        seed: Random seed for reproducibility
    """

    population_size: int = 50
    num_generations: int = 100
    elite_size: int = 5
    crossover_rate: float = 0.8
    mutation_rate: float = 0.3
    convergence_threshold: float = 0.001
    convergence_generations: int = 10
    diversity_threshold: float = 0.1
    seed: int | None = None


@dataclass
class EvolutionResult:
    """
    Results from evolutionary search.

    Attributes:
        best_genome: Best architecture found
        best_fitness: Fitness of best architecture
        generation_history: Fitness statistics per generation
        final_population: Final population
        convergence_generation: Generation at which search converged
        total_evaluations: Total number of fitness evaluations
    """

    best_genome: ArchitectureGenome
    best_fitness: FitnessMetrics
    generation_history: list[dict[str, Any]]
    final_population: list[ArchitectureGenome]
    convergence_generation: int
    total_evaluations: int


class EvolutionEngine:
    """
    Main engine for evolutionary architecture search.

    Coordinates genetic operators, fitness evaluation, and convergence.
    """

    def __init__(
        self,
        search_space: SearchSpaceConfig,
        config: EvolutionConfig,
        fitness_evaluator: FitnessEvaluator | None = None,
        crossover_operator: CrossoverOperator | None = None,
        mutation_operator: MutationOperator | None = None,
        selection_operator: SelectionOperator | None = None,
    ):
        """
        Initialize evolution engine.

        Args:
            search_space: Search space configuration
            config: Evolution configuration
            fitness_evaluator: Custom fitness evaluator (optional)
            crossover_operator: Custom crossover operator (optional)
            mutation_operator: Custom mutation operator (optional)
            selection_operator: Custom selection operator (optional)
        """
        self.search_space = search_space
        self.config = config

        # Initialize components
        self.fitness_evaluator = fitness_evaluator or FitnessEvaluator()
        self.crossover_operator = crossover_operator or UniformCrossover(
            search_space, seed=config.seed
        )
        self.mutation_operator = mutation_operator or CompositeMutation(
            search_space, mutation_rate=config.mutation_rate, seed=config.seed
        )
        self.selection_operator = selection_operator or ElitistSelection(
            elite_fraction=config.elite_size / config.population_size, seed=config.seed
        )

        self.population_generator = PopulationGenerator(search_space, seed=config.seed)

        # Evolution state
        self.population: list[ArchitectureGenome] = []
        self.generation = 0
        self.best_genome: ArchitectureGenome | None = None
        self.best_fitness: float = -float("inf")
        self.generation_history: list[dict[str, Any]] = []
        self.total_evaluations = 0

        # Convergence tracking
        self.generations_without_improvement = 0
        self.previous_best_fitness = -float("inf")

    def evolve(self) -> EvolutionResult:
        """
        Run evolutionary search.

        Returns:
            EvolutionResult with best architecture and statistics
        """
        # Initialize population
        self._initialize_population()

        # Evolution loop
        for gen in range(self.config.num_generations):
            self.generation = gen

            # Evaluate population
            self._evaluate_population()

            # Record statistics
            self._record_generation_stats()

            # Check convergence
            if self._check_convergence():
                break

            # Create next generation
            self._create_next_generation()

            # Maintain diversity
            self._maintain_diversity()

        # Prepare result
        return EvolutionResult(
            best_genome=self.best_genome,
            best_fitness=self.fitness_evaluator.evaluate(self.best_genome),
            generation_history=self.generation_history,
            final_population=self.population,
            convergence_generation=self.generation,
            total_evaluations=self.total_evaluations,
        )

    def _initialize_population(self) -> None:
        """Initialize population with diverse architectures."""
        self.population = self.population_generator.generate_population(
            self.config.population_size
        )

    def _evaluate_population(self) -> None:
        """Evaluate fitness for all individuals in population."""
        for genome in self.population:
            if genome.fitness_score == 0.0:  # Not yet evaluated
                metrics = self.fitness_evaluator.evaluate(genome)
                genome.fitness_score = metrics.combined_score
                genome.generation = self.generation
                self.total_evaluations += 1

                # Update best
                if genome.fitness_score > self.best_fitness:
                    self.best_fitness = genome.fitness_score
                    self.best_genome = genome

    def _record_generation_stats(self) -> None:
        """Record statistics for current generation."""
        fitness_scores = [g.fitness_score for g in self.population]

        stats = {
            "generation": self.generation,
            "best_fitness": float(np.max(fitness_scores)),
            "mean_fitness": float(np.mean(fitness_scores)),
            "std_fitness": float(np.std(fitness_scores)),
            "worst_fitness": float(np.min(fitness_scores)),
            "diversity": DiversityMetrics.compute_diversity(self.population),
        }

        self.generation_history.append(stats)

    def _check_convergence(self) -> bool:
        """
        Check if evolution has converged.

        Returns:
            True if converged, False otherwise
        """
        # Check fitness improvement
        improvement = self.best_fitness - self.previous_best_fitness

        if improvement < self.config.convergence_threshold:
            self.generations_without_improvement += 1
        else:
            self.generations_without_improvement = 0

        self.previous_best_fitness = self.best_fitness

        # Converged if no improvement for N generations
        return self.generations_without_improvement >= self.config.convergence_generations

    def _create_next_generation(self) -> None:
        """Create next generation through selection, crossover, and mutation."""
        next_generation: list[ArchitectureGenome] = []

        # Elitism: preserve best individuals
        elite = self.selection_operator.select(self.population, self.config.elite_size)
        next_generation.extend(elite)

        # Generate offspring
        while len(next_generation) < self.config.population_size:
            # Select parents
            parents = self.selection_operator.select(self.population, 2)

            # Crossover
            if np.random.random() < self.config.crossover_rate:
                offspring1, offspring2 = self.crossover_operator.crossover(
                    parents[0], parents[1]
                )
            else:
                offspring1, offspring2 = parents[0], parents[1]

            # Mutation
            offspring1 = self.mutation_operator.mutate(offspring1)
            offspring2 = self.mutation_operator.mutate(offspring2)

            # Reset fitness for re-evaluation
            offspring1.fitness_score = 0.0
            offspring2.fitness_score = 0.0

            # Add to next generation
            next_generation.append(offspring1)
            if len(next_generation) < self.config.population_size:
                next_generation.append(offspring2)

        self.population = next_generation[: self.config.population_size]

    def _maintain_diversity(self) -> None:
        """Maintain diversity in population."""
        diversity = DiversityMetrics.compute_diversity(self.population)

        # If diversity too low, inject new random individuals
        if diversity < self.config.diversity_threshold:
            num_inject = max(1, int(self.config.population_size * 0.1))

            # Remove worst individuals
            sorted_population = sorted(
                self.population, key=lambda g: g.fitness_score, reverse=True
            )
            self.population = sorted_population[:-num_inject]

            # Add new random individuals
            new_individuals = self.population_generator.generate_population(num_inject)
            self.population.extend(new_individuals)
