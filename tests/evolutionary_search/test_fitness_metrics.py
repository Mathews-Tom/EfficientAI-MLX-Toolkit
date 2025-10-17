"""
Tests for fitness metrics and evaluation.
"""

from __future__ import annotations

import pytest

from evolutionary_search.fitness import (
    FitnessEvaluator,
    FitnessMetrics,
    MultiObjectiveScore,
)
from evolutionary_search.search_space import (
    ArchitectureComponent,
    ArchitectureGenome,
    LayerConfig,
)


class TestFitnessMetrics:
    """Tests for FitnessMetrics dataclass."""

    def test_fitness_metrics_creation(self) -> None:
        """Test creating fitness metrics."""
        metrics = FitnessMetrics(
            quality_score=0.8,
            speed_score=0.7,
            memory_score=0.9,
            combined_score=0.8,
            raw_metrics={"fid": 25.0},
        )

        assert metrics.quality_score == 0.8
        assert metrics.speed_score == 0.7
        assert metrics.memory_score == 0.9
        assert metrics.combined_score == 0.8

    def test_to_dict(self) -> None:
        """Test converting metrics to dictionary."""
        metrics = FitnessMetrics(
            quality_score=0.8,
            speed_score=0.7,
            memory_score=0.9,
            combined_score=0.8,
            raw_metrics={},
        )

        metrics_dict = metrics.to_dict()

        assert metrics_dict["quality"] == 0.8
        assert metrics_dict["speed"] == 0.7
        assert metrics_dict["memory"] == 0.9
        assert metrics_dict["combined"] == 0.8


class TestMultiObjectiveScore:
    """Tests for MultiObjectiveScore."""

    def test_multi_objective_score_creation(self) -> None:
        """Test creating multi-objective score."""
        score = MultiObjectiveScore(
            objectives={"quality": 0.8, "speed": 0.7, "memory": 0.9},
            constraints_satisfied=True,
        )

        assert score.objectives["quality"] == 0.8
        assert score.constraints_satisfied

    def test_dominance_check(self) -> None:
        """Test Pareto dominance checking."""
        # Score 1 dominates Score 2 (better in all objectives)
        score1 = MultiObjectiveScore(
            objectives={"quality": 0.9, "speed": 0.8, "memory": 0.9},
            constraints_satisfied=True,
        )
        score2 = MultiObjectiveScore(
            objectives={"quality": 0.7, "speed": 0.6, "memory": 0.7},
            constraints_satisfied=True,
        )

        assert score1.dominates(score2)
        assert not score2.dominates(score1)

    def test_non_dominance(self) -> None:
        """Test non-dominated solutions."""
        # Neither dominates the other (trade-offs)
        score1 = MultiObjectiveScore(
            objectives={"quality": 0.9, "speed": 0.6, "memory": 0.7},
            constraints_satisfied=True,
        )
        score2 = MultiObjectiveScore(
            objectives={"quality": 0.7, "speed": 0.9, "memory": 0.8},
            constraints_satisfied=True,
        )

        assert not score1.dominates(score2)
        assert not score2.dominates(score1)

    def test_constraint_violation_dominance(self) -> None:
        """Test dominance with constraint violations."""
        # Valid solution dominates invalid one
        valid_score = MultiObjectiveScore(
            objectives={"quality": 0.5, "speed": 0.5, "memory": 0.5},
            constraints_satisfied=True,
        )
        invalid_score = MultiObjectiveScore(
            objectives={"quality": 0.9, "speed": 0.9, "memory": 0.9},
            constraints_satisfied=False,
        )

        assert valid_score.dominates(invalid_score)
        assert not invalid_score.dominates(valid_score)


class TestFitnessEvaluator:
    """Tests for FitnessEvaluator."""

    @pytest.fixture
    def simple_genome(self) -> ArchitectureGenome:
        """Create a simple genome for testing."""
        layers = [
            LayerConfig(
                component_type=ArchitectureComponent.CONV_BLOCK,
                parameters={"kernel_size": 3, "stride": 1, "padding": 1},
                layer_index=i,
                in_channels=64,
                out_channels=64,
            )
            for i in range(4)
        ]
        return ArchitectureGenome(layers=layers)

    def test_evaluator_initialization(self) -> None:
        """Test fitness evaluator initialization."""
        evaluator = FitnessEvaluator(
            quality_weight=0.5, speed_weight=0.3, memory_weight=0.2
        )

        assert evaluator.quality_weight == 0.5
        assert evaluator.speed_weight == 0.3
        assert evaluator.memory_weight == 0.2

    def test_evaluate_with_estimation(self, simple_genome: ArchitectureGenome) -> None:
        """Test fitness evaluation with estimation (no benchmark results)."""
        evaluator = FitnessEvaluator()
        metrics = evaluator.evaluate(simple_genome)

        assert 0.0 <= metrics.quality_score <= 1.0
        assert 0.0 <= metrics.speed_score <= 1.0
        assert 0.0 <= metrics.memory_score <= 1.0
        assert 0.0 <= metrics.combined_score <= 1.0
        assert metrics.raw_metrics["estimated"] is True

    def test_evaluate_with_benchmark_results(
        self, simple_genome: ArchitectureGenome
    ) -> None:
        """Test fitness evaluation with actual benchmark results."""
        evaluator = FitnessEvaluator(target_inference_ms=100.0, target_memory_mb=2048.0)

        benchmark_results = {
            "fid": 30.0,
            "inference_time_ms": 80.0,
            "memory_mb": 1024.0,
        }

        metrics = evaluator.evaluate(simple_genome, benchmark_results)

        assert metrics.quality_score > 0.0
        assert metrics.speed_score > 0.0
        assert metrics.memory_score > 0.0
        assert "estimated" not in metrics.raw_metrics

    def test_quality_evaluation_fid(self, simple_genome: ArchitectureGenome) -> None:
        """Test quality evaluation using FID score."""
        evaluator = FitnessEvaluator()

        # Good FID (low value)
        results_good = {"fid": 20.0}
        metrics_good = evaluator.evaluate(simple_genome, results_good)

        # Bad FID (high value)
        results_bad = {"fid": 150.0}
        metrics_bad = evaluator.evaluate(simple_genome, results_bad)

        assert metrics_good.quality_score > metrics_bad.quality_score

    def test_quality_evaluation_inception_score(
        self, simple_genome: ArchitectureGenome
    ) -> None:
        """Test quality evaluation using Inception Score."""
        evaluator = FitnessEvaluator()

        # Good IS (high value)
        results_good = {"inception_score": 7.5}
        metrics_good = evaluator.evaluate(simple_genome, results_good)

        # Bad IS (low value)
        results_bad = {"inception_score": 2.0}
        metrics_bad = evaluator.evaluate(simple_genome, results_bad)

        assert metrics_good.quality_score > metrics_bad.quality_score

    def test_speed_evaluation(self, simple_genome: ArchitectureGenome) -> None:
        """Test speed evaluation."""
        evaluator = FitnessEvaluator(target_inference_ms=100.0)

        # Fast inference
        results_fast = {"inference_time_ms": 50.0}
        metrics_fast = evaluator.evaluate(simple_genome, results_fast)

        # Slow inference
        results_slow = {"inference_time_ms": 200.0}
        metrics_slow = evaluator.evaluate(simple_genome, results_slow)

        assert metrics_fast.speed_score > metrics_slow.speed_score

    def test_memory_evaluation(self, simple_genome: ArchitectureGenome) -> None:
        """Test memory efficiency evaluation."""
        evaluator = FitnessEvaluator(target_memory_mb=2048.0)

        # Low memory usage
        results_efficient = {"memory_mb": 1024.0}
        metrics_efficient = evaluator.evaluate(simple_genome, results_efficient)

        # High memory usage
        results_inefficient = {"memory_mb": 4096.0}
        metrics_inefficient = evaluator.evaluate(simple_genome, results_inefficient)

        assert metrics_efficient.memory_score > metrics_inefficient.memory_score

    def test_combined_score_weights(self, simple_genome: ArchitectureGenome) -> None:
        """Test combined score with custom weights."""
        # Equal weights
        evaluator_equal = FitnessEvaluator(
            quality_weight=0.33, speed_weight=0.33, memory_weight=0.34
        )
        metrics_equal = evaluator_equal.evaluate(simple_genome)

        # Quality-focused weights
        evaluator_quality = FitnessEvaluator(
            quality_weight=0.8, speed_weight=0.1, memory_weight=0.1
        )
        metrics_quality = evaluator_quality.evaluate(simple_genome)

        # Scores should be different due to weights
        assert metrics_equal.combined_score != metrics_quality.combined_score

    def test_multi_objective_evaluation(
        self, simple_genome: ArchitectureGenome
    ) -> None:
        """Test multi-objective evaluation."""
        evaluator = FitnessEvaluator()
        mo_score = evaluator.evaluate_multi_objective(simple_genome)

        assert "quality" in mo_score.objectives
        assert "speed" in mo_score.objectives
        assert "memory" in mo_score.objectives
        assert isinstance(mo_score.constraints_satisfied, bool)

    def test_constraint_checking(self, simple_genome: ArchitectureGenome) -> None:
        """Test constraint satisfaction checking."""
        evaluator = FitnessEvaluator()

        # Small genome (should satisfy constraints)
        mo_score = evaluator.evaluate_multi_objective(simple_genome)
        assert mo_score.constraints_satisfied

    def test_normalization_stats_update(
        self, simple_genome: ArchitectureGenome
    ) -> None:
        """Test updating normalization statistics."""
        evaluator = FitnessEvaluator()

        # Evaluate multiple genomes
        metrics_list = [evaluator.evaluate(simple_genome) for _ in range(10)]

        # Update stats
        evaluator.update_normalization_stats(metrics_list)

        assert evaluator.quality_stats["mean"] > 0
        assert evaluator.speed_stats["mean"] > 0
        assert evaluator.memory_stats["mean"] > 0

    def test_estimate_quality_from_structure(
        self, simple_genome: ArchitectureGenome
    ) -> None:
        """Test quality estimation from genome structure."""
        evaluator = FitnessEvaluator()

        # Genome with attention (should score higher)
        layers_with_attention = [
            LayerConfig(
                component_type=ArchitectureComponent.ATTENTION_BLOCK,
                parameters={"num_heads": 8, "embed_dim": 512},
                layer_index=i,
                in_channels=512,
                out_channels=512,
            )
            for i in range(8)
        ]
        genome_attention = ArchitectureGenome(layers=layers_with_attention)

        metrics_simple = evaluator.evaluate(simple_genome)
        metrics_attention = evaluator.evaluate(genome_attention)

        # More attention and depth should lead to higher quality estimate
        assert metrics_attention.quality_score >= metrics_simple.quality_score
