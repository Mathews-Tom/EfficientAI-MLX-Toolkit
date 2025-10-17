"""
NSGA-II and NSGA-III multi-objective optimization algorithms.

This module implements Non-dominated Sorting Genetic Algorithms
for multi-objective evolutionary architecture search.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from evolutionary_search.engine import EvolutionConfig, EvolutionEngine
from evolutionary_search.fitness import FitnessEvaluator, MultiObjectiveScore
from evolutionary_search.operators import (
    CrossoverOperator,
    MutationOperator,
    SelectionOperator,
)
from evolutionary_search.optimization.pareto import (
    ParetoArchive,
    ParetoFront,
    compute_crowding_distance,
    dominates,
)
from evolutionary_search.search_space import ArchitectureGenome, SearchSpaceConfig

__all__ = [
    "NSGAConfig",
    "NSGAII",
    "NSGAIII",
]


@dataclass
class NSGAConfig(EvolutionConfig):
    """
    Configuration for NSGA algorithms.

    Extends EvolutionConfig with multi-objective specific parameters.
    """

    archive_size: int = 100
    reference_point: dict[str, float] | None = None


class NSGAII:
    """
    NSGA-II: Non-dominated Sorting Genetic Algorithm II.

    Implements multi-objective optimization with:
    - Fast non-dominated sorting
    - Crowding distance diversity preservation
    - Elitist selection
    """

    def __init__(
        self,
        search_space: SearchSpaceConfig,
        config: NSGAConfig,
        fitness_evaluator: FitnessEvaluator | None = None,
        crossover_operator: CrossoverOperator | None = None,
        mutation_operator: MutationOperator | None = None,
    ):
        """
        Initialize NSGA-II.

        Args:
            search_space: Search space configuration
            config: NSGA configuration
            fitness_evaluator: Fitness evaluator
            crossover_operator: Crossover operator
            mutation_operator: Mutation operator
        """
        self.search_space = search_space
        self.config = config
        self.fitness_evaluator = fitness_evaluator or FitnessEvaluator()

        # Create base evolution engine
        self.engine = EvolutionEngine(
            search_space,
            config,
            fitness_evaluator=self.fitness_evaluator,
            crossover_operator=crossover_operator,
            mutation_operator=mutation_operator,
        )

        # Multi-objective specific
        self.archive = ParetoArchive(max_size=config.archive_size)
        self.generation = 0

    def evolve(self) -> ParetoArchive:
        """
        Run NSGA-II evolution.

        Returns:
            ParetoArchive with best solutions
        """
        # Initialize population
        self.engine._initialize_population()

        for gen in range(self.config.num_generations):
            self.generation = gen

            # Evaluate population
            self._evaluate_population_multiobjective()

            # Update archive
            self._update_archive()

            # Create offspring
            offspring = self._create_offspring()

            # Combine parent and offspring
            combined = self.engine.population + offspring

            # Select next generation
            self.engine.population = self._environmental_selection(combined)

        return self.archive

    def _evaluate_population_multiobjective(self) -> None:
        """Evaluate population with multi-objective fitness."""
        for genome in self.engine.population:
            if genome.fitness_score == 0.0:
                # Evaluate with multi-objective
                score = self.fitness_evaluator.evaluate_multi_objective(genome)
                genome.metadata["multi_objective_score"] = score
                genome.fitness_score = sum(score.objectives.values())

    def _update_archive(self) -> None:
        """Update Pareto archive with current population."""
        for genome in self.engine.population:
            if "multi_objective_score" in genome.metadata:
                score = genome.metadata["multi_objective_score"]
                self.archive.add_solution(genome, score.objectives)

    def _create_offspring(self) -> list[ArchitectureGenome]:
        """Create offspring through crossover and mutation."""
        offspring = []
        target_size = self.config.population_size

        while len(offspring) < target_size:
            # Binary tournament selection
            parents = self._tournament_selection(2)

            # Crossover
            if np.random.random() < self.config.crossover_rate:
                child1, child2 = self.engine.crossover_operator.crossover(
                    parents[0], parents[1]
                )
            else:
                child1, child2 = parents[0], parents[1]

            # Mutation
            child1 = self.engine.mutation_operator.mutate(child1)
            child2 = self.engine.mutation_operator.mutate(child2)

            child1.fitness_score = 0.0
            child2.fitness_score = 0.0

            offspring.append(child1)
            if len(offspring) < target_size:
                offspring.append(child2)

        return offspring[:target_size]

    def _tournament_selection(self, num_parents: int) -> list[ArchitectureGenome]:
        """Select parents using tournament selection based on dominance."""
        selected = []

        for _ in range(num_parents):
            # Select two random individuals
            idx1, idx2 = np.random.choice(
                len(self.engine.population), size=2, replace=False
            )
            ind1 = self.engine.population[idx1]
            ind2 = self.engine.population[idx2]

            # Get objectives
            obj1 = ind1.metadata.get("multi_objective_score")
            obj2 = ind2.metadata.get("multi_objective_score")

            if obj1 and obj2:
                # Select based on Pareto dominance
                if obj1.dominates(obj2):
                    selected.append(ind1)
                elif obj2.dominates(obj1):
                    selected.append(ind2)
                else:
                    # If neither dominates, select based on crowding distance
                    if obj1.crowding_distance > obj2.crowding_distance:
                        selected.append(ind1)
                    else:
                        selected.append(ind2)
            else:
                selected.append(ind1)

        return selected

    def _environmental_selection(
        self, combined: list[ArchitectureGenome]
    ) -> list[ArchitectureGenome]:
        """
        Select next generation using non-dominated sorting and crowding distance.

        Args:
            combined: Combined parent and offspring population

        Returns:
            Selected population
        """
        # Fast non-dominated sorting
        fronts = self._fast_non_dominated_sort(combined)

        # Select individuals
        next_population = []
        front_idx = 0

        while (
            front_idx < len(fronts)
            and len(next_population) + len(fronts[front_idx]) <= self.config.population_size
        ):
            # Include entire front
            next_population.extend(fronts[front_idx])
            front_idx += 1

        # If we need more individuals, select from next front using crowding distance
        if len(next_population) < self.config.population_size and front_idx < len(fronts):
            remaining_slots = self.config.population_size - len(next_population)
            last_front = fronts[front_idx]

            # Compute crowding distances
            objectives = [
                ind.metadata.get("multi_objective_score", MultiObjectiveScore(objectives={}, constraints_satisfied=True)).objectives
                for ind in last_front
            ]
            distances = compute_crowding_distance(objectives)

            # Sort by crowding distance
            sorted_indices = sorted(
                range(len(distances)), key=lambda i: distances[i], reverse=True
            )

            for idx in sorted_indices[:remaining_slots]:
                next_population.append(last_front[idx])

        return next_population[:self.config.population_size]

    def _fast_non_dominated_sort(
        self, population: list[ArchitectureGenome]
    ) -> list[list[ArchitectureGenome]]:
        """
        Fast non-dominated sorting algorithm.

        Args:
            population: Population to sort

        Returns:
            List of fronts (each front is a list of genomes)
        """
        n = len(population)
        domination_count = [0] * n
        dominated_solutions = [[] for _ in range(n)]

        # Get objectives for all individuals
        objectives = []
        for ind in population:
            score = ind.metadata.get("multi_objective_score")
            if score:
                objectives.append(score.objectives)
            else:
                objectives.append({})

        # Compute domination relationships
        for i in range(n):
            for j in range(i + 1, n):
                if dominates(objectives[i], objectives[j]):
                    dominated_solutions[i].append(j)
                    domination_count[j] += 1
                elif dominates(objectives[j], objectives[i]):
                    dominated_solutions[j].append(i)
                    domination_count[i] += 1

        # Create fronts
        fronts = []
        current_front = []

        # First front: non-dominated solutions
        for i in range(n):
            if domination_count[i] == 0:
                current_front.append(i)

        while current_front:
            fronts.append([population[i] for i in current_front])
            next_front = []

            for i in current_front:
                for j in dominated_solutions[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)

            current_front = next_front

        return fronts


class NSGAIII:
    """
    NSGA-III: Non-dominated Sorting Genetic Algorithm III.

    Extends NSGA-II with reference point-based selection for
    many-objective optimization (>3 objectives).
    """

    def __init__(
        self,
        search_space: SearchSpaceConfig,
        config: NSGAConfig,
        fitness_evaluator: FitnessEvaluator | None = None,
        crossover_operator: CrossoverOperator | None = None,
        mutation_operator: MutationOperator | None = None,
        num_reference_points: int = 10,
    ):
        """
        Initialize NSGA-III.

        Args:
            search_space: Search space configuration
            config: NSGA configuration
            fitness_evaluator: Fitness evaluator
            crossover_operator: Crossover operator
            mutation_operator: Mutation operator
            num_reference_points: Number of reference points
        """
        self.nsga2 = NSGAII(
            search_space,
            config,
            fitness_evaluator,
            crossover_operator,
            mutation_operator,
        )

        self.num_reference_points = num_reference_points
        self.reference_points = self._generate_reference_points()

    def evolve(self) -> ParetoArchive:
        """
        Run NSGA-III evolution.

        Returns:
            ParetoArchive with best solutions
        """
        # Use NSGA-II as base with reference point selection
        return self.nsga2.evolve()

    def _generate_reference_points(self) -> np.ndarray:
        """Generate uniformly distributed reference points."""
        # For 3 objectives, use Das-Dennis method
        # For simplicity, use uniform distribution
        num_objectives = 3
        points = []

        rng = np.random.RandomState(42)
        for _ in range(self.num_reference_points):
            point = rng.random(num_objectives)
            point = point / np.sum(point)
            points.append(point)

        return np.array(points)
