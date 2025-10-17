"""
Pareto front management for multi-objective optimization.

This module provides utilities for tracking and managing Pareto fronts
in multi-objective evolutionary search.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from evolutionary_search.fitness import MultiObjectiveScore
from evolutionary_search.search_space import ArchitectureGenome

__all__ = [
    "ParetoFront",
    "ParetoArchive",
    "dominates",
    "compute_crowding_distance",
]


def dominates(objectives1: dict[str, float], objectives2: dict[str, float]) -> bool:
    """
    Check if objectives1 Pareto-dominates objectives2.

    A solution dominates another if it is better in at least one objective
    and not worse in any objective (for maximization).

    Args:
        objectives1: First objective dictionary
        objectives2: Second objective dictionary

    Returns:
        True if objectives1 dominates objectives2
    """
    better_in_any = False
    for key in objectives1:
        if key not in objectives2:
            continue

        if objectives1[key] < objectives2[key]:
            return False  # Worse in this objective
        if objectives1[key] > objectives2[key]:
            better_in_any = True

    return better_in_any


def compute_crowding_distance(
    objectives_list: list[dict[str, float]]
) -> list[float]:
    """
    Compute crowding distance for a set of solutions.

    Crowding distance measures the density of solutions around each point.
    Higher values indicate more isolated solutions (better diversity).

    Args:
        objectives_list: List of objective dictionaries

    Returns:
        List of crowding distances for each solution
    """
    n = len(objectives_list)
    if n <= 2:
        return [float("inf")] * n

    # Initialize distances
    distances = [0.0] * n

    # Get objective keys
    if not objectives_list:
        return distances

    obj_keys = list(objectives_list[0].keys())

    # Compute crowding distance for each objective
    for obj_key in obj_keys:
        # Sort by this objective
        sorted_indices = sorted(
            range(n), key=lambda i: objectives_list[i].get(obj_key, 0.0)
        )

        # Boundary points get infinite distance
        distances[sorted_indices[0]] = float("inf")
        distances[sorted_indices[-1]] = float("inf")

        # Get objective range
        obj_min = objectives_list[sorted_indices[0]].get(obj_key, 0.0)
        obj_max = objectives_list[sorted_indices[-1]].get(obj_key, 0.0)
        obj_range = obj_max - obj_min

        if obj_range == 0:
            continue

        # Compute distances for interior points
        for i in range(1, n - 1):
            idx = sorted_indices[i]
            idx_prev = sorted_indices[i - 1]
            idx_next = sorted_indices[i + 1]

            if distances[idx] != float("inf"):
                distances[idx] += (
                    objectives_list[idx_next].get(obj_key, 0.0)
                    - objectives_list[idx_prev].get(obj_key, 0.0)
                ) / obj_range

    return distances


@dataclass
class ParetoFront:
    """
    Represents a Pareto front of non-dominated solutions.

    Attributes:
        solutions: List of genomes in the front
        objectives: List of objective values for each solution
        rank: Pareto rank (0 = non-dominated)
    """

    solutions: list[ArchitectureGenome]
    objectives: list[dict[str, float]]
    rank: int = 0

    def __post_init__(self) -> None:
        """Validate front after initialization."""
        assert len(self.solutions) == len(
            self.objectives
        ), "Solutions and objectives length mismatch"

    def compute_crowding_distances(self) -> list[float]:
        """
        Compute crowding distances for all solutions in front.

        Returns:
            List of crowding distances
        """
        return compute_crowding_distance(self.objectives)

    def get_hypervolume(self, reference_point: dict[str, float]) -> float:
        """
        Compute hypervolume indicator.

        Hypervolume measures the volume of objective space dominated
        by the front relative to a reference point.

        Args:
            reference_point: Reference point for hypervolume calculation

        Returns:
            Hypervolume value
        """
        if not self.objectives:
            return 0.0

        # Simple 2D/3D hypervolume calculation
        if len(self.objectives[0]) == 2:
            return self._hypervolume_2d(reference_point)
        else:
            # For >3 objectives, use approximation
            return self._hypervolume_approximation(reference_point)

    def _hypervolume_2d(self, reference_point: dict[str, float]) -> float:
        """Compute exact 2D hypervolume."""
        if not self.objectives:
            return 0.0

        obj_keys = list(self.objectives[0].keys())
        if len(obj_keys) != 2:
            return 0.0

        # Sort by first objective
        sorted_objs = sorted(
            self.objectives, key=lambda x: x[obj_keys[0]], reverse=True
        )

        volume = 0.0
        prev_y = reference_point[obj_keys[1]]

        for obj in sorted_objs:
            width = max(0, obj[obj_keys[0]] - reference_point[obj_keys[0]])
            height = max(0, obj[obj_keys[1]] - prev_y)
            volume += width * height
            prev_y = max(prev_y, obj[obj_keys[1]])

        return volume

    def _hypervolume_approximation(self, reference_point: dict[str, float]) -> float:
        """Approximate hypervolume for high-dimensional objectives."""
        if not self.objectives:
            return 0.0

        # Monte Carlo approximation
        num_samples = 10000
        dominated_count = 0

        # Get objective ranges
        obj_keys = list(self.objectives[0].keys())
        obj_min = {key: reference_point[key] for key in obj_keys}
        obj_max = {
            key: max(obj[key] for obj in self.objectives) for key in obj_keys
        }

        rng = np.random.RandomState(42)

        for _ in range(num_samples):
            # Generate random point
            point = {
                key: rng.uniform(obj_min[key], obj_max[key]) for key in obj_keys
            }

            # Check if point is dominated by any solution
            for obj in self.objectives:
                if dominates(obj, point):
                    dominated_count += 1
                    break

        # Estimate volume
        total_volume = np.prod(
            [obj_max[key] - obj_min[key] for key in obj_keys]
        )
        return (dominated_count / num_samples) * total_volume


class ParetoArchive:
    """
    Archive for maintaining Pareto-optimal solutions across generations.

    Tracks best solutions found during evolution and maintains
    multiple Pareto fronts.
    """

    def __init__(self, max_size: int = 100):
        """
        Initialize Pareto archive.

        Args:
            max_size: Maximum number of solutions to maintain
        """
        self.max_size = max_size
        self.fronts: list[ParetoFront] = []
        self.all_solutions: list[ArchitectureGenome] = []
        self.all_objectives: list[dict[str, float]] = []

    def add_solution(
        self, genome: ArchitectureGenome, objectives: dict[str, float]
    ) -> bool:
        """
        Add a solution to the archive.

        Args:
            genome: Architecture genome
            objectives: Objective values

        Returns:
            True if solution was added (non-dominated)
        """
        # Check if solution is dominated by existing solutions
        for existing_obj in self.all_objectives:
            if dominates(existing_obj, objectives):
                return False

        # Remove solutions dominated by new solution
        non_dominated_solutions = []
        non_dominated_objectives = []

        for sol, obj in zip(self.all_solutions, self.all_objectives):
            if not dominates(objectives, obj):
                non_dominated_solutions.append(sol)
                non_dominated_objectives.append(obj)

        # Add new solution
        self.all_solutions = non_dominated_solutions + [genome]
        self.all_objectives = non_dominated_objectives + [objectives]

        # Trim if exceeds max size
        if len(self.all_solutions) > self.max_size:
            self._trim_archive()

        # Rebuild fronts
        self._rebuild_fronts()

        return True

    def get_best_front(self) -> ParetoFront | None:
        """Get the first (best) Pareto front."""
        return self.fronts[0] if self.fronts else None

    def _rebuild_fronts(self) -> None:
        """Rebuild Pareto fronts from current solutions."""
        if not self.all_solutions:
            self.fronts = []
            return

        remaining_indices = list(range(len(self.all_solutions)))
        fronts = []
        rank = 0

        while remaining_indices:
            current_front_indices = []

            # Find non-dominated solutions in remaining
            for i in remaining_indices:
                dominated = False
                for j in remaining_indices:
                    if i != j and dominates(
                        self.all_objectives[j], self.all_objectives[i]
                    ):
                        dominated = True
                        break

                if not dominated:
                    current_front_indices.append(i)

            # Create front
            front_solutions = [self.all_solutions[i] for i in current_front_indices]
            front_objectives = [self.all_objectives[i] for i in current_front_indices]

            fronts.append(
                ParetoFront(
                    solutions=front_solutions, objectives=front_objectives, rank=rank
                )
            )

            # Remove front from remaining
            remaining_indices = [
                i for i in remaining_indices if i not in current_front_indices
            ]
            rank += 1

        self.fronts = fronts

    def _trim_archive(self) -> None:
        """Trim archive to max size using crowding distance."""
        # Rebuild fronts first
        self._rebuild_fronts()

        # Keep solutions from fronts until max size
        kept_solutions = []
        kept_objectives = []

        for front in self.fronts:
            if len(kept_solutions) + len(front.solutions) <= self.max_size:
                kept_solutions.extend(front.solutions)
                kept_objectives.extend(front.objectives)
            else:
                # Partially include this front based on crowding distance
                remaining_slots = self.max_size - len(kept_solutions)
                if remaining_slots > 0:
                    distances = front.compute_crowding_distances()
                    sorted_indices = sorted(
                        range(len(distances)), key=lambda i: distances[i], reverse=True
                    )

                    for idx in sorted_indices[:remaining_slots]:
                        kept_solutions.append(front.solutions[idx])
                        kept_objectives.append(front.objectives[idx])
                break

        self.all_solutions = kept_solutions
        self.all_objectives = kept_objectives
