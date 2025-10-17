"""
Fitness metrics for evaluating diffusion model architectures.

This module implements multi-objective fitness evaluation including
quality metrics (FID, IS), speed metrics (inference time), and
memory efficiency metrics for Apple Silicon.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from evolutionary_search.search_space import ArchitectureGenome

__all__ = [
    "FitnessMetrics",
    "FitnessEvaluator",
    "MultiObjectiveScore",
]


@dataclass
class FitnessMetrics:
    """
    Container for fitness evaluation results.

    Attributes:
        quality_score: Image quality score (0-1, higher is better)
        speed_score: Inference speed score (0-1, higher is better)
        memory_score: Memory efficiency score (0-1, higher is better)
        combined_score: Weighted combination of metrics
        raw_metrics: Dictionary of raw metric values
    """

    quality_score: float
    speed_score: float
    memory_score: float
    combined_score: float
    raw_metrics: dict[str, Any]

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary format."""
        return {
            "quality": self.quality_score,
            "speed": self.speed_score,
            "memory": self.memory_score,
            "combined": self.combined_score,
        }


@dataclass
class MultiObjectiveScore:
    """
    Multi-objective fitness score for Pareto front analysis.

    Objectives (all to be maximized):
        - Quality: Generation quality (FID/IS based)
        - Speed: Inference throughput
        - Efficiency: Memory/compute efficiency
    """

    objectives: dict[str, float]
    constraints_satisfied: bool
    pareto_rank: int = 0
    crowding_distance: float = 0.0

    def dominates(self, other: MultiObjectiveScore) -> bool:
        """Check if this solution dominates another (Pareto dominance)."""
        if not self.constraints_satisfied:
            return False
        if not other.constraints_satisfied:
            return True

        better_in_any = False
        for key in self.objectives:
            if self.objectives[key] < other.objectives.get(key, 0):
                return False
            if self.objectives[key] > other.objectives.get(key, 0):
                better_in_any = True

        return better_in_any


class FitnessEvaluator:
    """
    Evaluates fitness of architecture genomes.

    Implements multi-objective evaluation with normalized scores
    and hardware-aware metrics for Apple Silicon.
    """

    def __init__(
        self,
        quality_weight: float = 0.4,
        speed_weight: float = 0.3,
        memory_weight: float = 0.3,
        target_inference_ms: float = 100.0,
        target_memory_mb: float = 2048.0,
    ):
        """
        Initialize fitness evaluator.

        Args:
            quality_weight: Weight for quality score
            speed_weight: Weight for speed score
            memory_weight: Weight for memory score
            target_inference_ms: Target inference time in milliseconds
            target_memory_mb: Target memory usage in MB
        """
        self.quality_weight = quality_weight
        self.speed_weight = speed_weight
        self.memory_weight = memory_weight
        self.target_inference_ms = target_inference_ms
        self.target_memory_mb = target_memory_mb

        # Normalization statistics (updated during evolution)
        self.quality_stats = {"mean": 0.5, "std": 0.1}
        self.speed_stats = {"mean": 100.0, "std": 50.0}
        self.memory_stats = {"mean": 2048.0, "std": 1024.0}

    def evaluate(
        self,
        genome: ArchitectureGenome,
        benchmark_results: dict[str, Any] | None = None,
    ) -> FitnessMetrics:
        """
        Evaluate fitness of an architecture genome.

        Args:
            genome: Architecture genome to evaluate
            benchmark_results: Optional benchmark results (for actual evaluation)

        Returns:
            FitnessMetrics with normalized scores
        """
        if benchmark_results:
            # Use actual benchmark results
            quality_score = self._evaluate_quality(benchmark_results)
            speed_score = self._evaluate_speed(benchmark_results)
            memory_score = self._evaluate_memory(benchmark_results)
            raw_metrics = benchmark_results
        else:
            # Estimate from genome structure
            quality_score = self._estimate_quality(genome)
            speed_score = self._estimate_speed(genome)
            memory_score = self._estimate_memory(genome)
            raw_metrics = {
                "estimated": True,
                "parameter_count": genome.count_parameters(),
            }

        # Compute combined score
        combined_score = (
            self.quality_weight * quality_score
            + self.speed_weight * speed_score
            + self.memory_weight * memory_score
        )

        return FitnessMetrics(
            quality_score=quality_score,
            speed_score=speed_score,
            memory_score=memory_score,
            combined_score=combined_score,
            raw_metrics=raw_metrics,
        )

    def evaluate_multi_objective(
        self, genome: ArchitectureGenome, benchmark_results: dict[str, Any] | None = None
    ) -> MultiObjectiveScore:
        """
        Evaluate genome for multi-objective optimization.

        Returns:
            MultiObjectiveScore for Pareto analysis
        """
        metrics = self.evaluate(genome, benchmark_results)

        objectives = {
            "quality": metrics.quality_score,
            "speed": metrics.speed_score,
            "memory": metrics.memory_score,
        }

        # Check constraints
        constraints_satisfied = self._check_constraints(genome, metrics)

        return MultiObjectiveScore(
            objectives=objectives, constraints_satisfied=constraints_satisfied
        )

    def _evaluate_quality(self, results: dict[str, Any]) -> float:
        """Evaluate quality from benchmark results."""
        # Assuming FID (lower is better) or IS (higher is better)
        if "fid" in results:
            fid = results["fid"]
            # Normalize FID: typical range 10-200, target <50
            quality_score = max(0.0, min(1.0, 1.0 - (fid - 10.0) / 190.0))
        elif "inception_score" in results:
            is_score = results["inception_score"]
            # Normalize IS: typical range 1-10, target >5
            quality_score = max(0.0, min(1.0, (is_score - 1.0) / 9.0))
        else:
            quality_score = 0.5  # Default

        return quality_score

    def _evaluate_speed(self, results: dict[str, Any]) -> float:
        """Evaluate speed from benchmark results."""
        inference_time = results.get("inference_time_ms", self.target_inference_ms)

        # Normalize: faster is better
        speed_score = max(
            0.0, min(1.0, self.target_inference_ms / max(inference_time, 1.0))
        )

        return speed_score

    def _evaluate_memory(self, results: dict[str, Any]) -> float:
        """Evaluate memory efficiency from benchmark results."""
        memory_usage = results.get("memory_mb", self.target_memory_mb)

        # Normalize: lower memory is better
        memory_score = max(
            0.0, min(1.0, self.target_memory_mb / max(memory_usage, 1.0))
        )

        return memory_score

    def _estimate_quality(self, genome: ArchitectureGenome) -> float:
        """Estimate quality from genome structure."""
        # Heuristic: more attention blocks and deeper networks tend to be better
        attention_count = sum(
            1
            for layer in genome.layers
            if "attention" in layer.component_type.value
        )
        depth = len(genome.layers)

        # Normalize
        quality_estimate = min(
            1.0, (attention_count * 0.1 + depth * 0.02)
        )

        return quality_estimate

    def _estimate_speed(self, genome: ArchitectureGenome) -> float:
        """Estimate speed from genome structure."""
        # Heuristic: fewer layers and parameters = faster
        param_count = genome.count_parameters()
        layer_count = len(genome.layers)

        # Normalize (inverse relationship)
        speed_estimate = max(
            0.0, 1.0 - (param_count / 1e9 + layer_count / 100)
        )

        return speed_estimate

    def _estimate_memory(self, genome: ArchitectureGenome) -> float:
        """Estimate memory efficiency from genome structure."""
        # Heuristic: parameter count correlates with memory
        param_count = genome.count_parameters()

        # Normalize (inverse relationship)
        memory_estimate = max(0.0, 1.0 - param_count / 1e9)

        return memory_estimate

    def _check_constraints(
        self, genome: ArchitectureGenome, metrics: FitnessMetrics
    ) -> bool:
        """Check if genome satisfies hardware constraints."""
        # Check parameter count
        if genome.count_parameters() > 1e9:  # 1B parameter limit
            return False

        # Check estimated memory
        if metrics.raw_metrics.get("memory_mb", 0) > 16384:  # 16GB limit
            return False

        return True

    def update_normalization_stats(self, population_metrics: list[FitnessMetrics]) -> None:
        """Update normalization statistics from population."""
        if not population_metrics:
            return

        qualities = [m.quality_score for m in population_metrics]
        speeds = [m.speed_score for m in population_metrics]
        memories = [m.memory_score for m in population_metrics]

        self.quality_stats = {
            "mean": float(np.mean(qualities)),
            "std": float(np.std(qualities)),
        }
        self.speed_stats = {
            "mean": float(np.mean(speeds)),
            "std": float(np.std(speeds)),
        }
        self.memory_stats = {
            "mean": float(np.mean(memories)),
            "std": float(np.std(memories)),
        }
