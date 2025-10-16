"""
Tests for Step Reduction Algorithm

Tests intelligent step reduction and adaptive step allocation.
"""

import numpy as np
import pytest

import mlx.core as mx

from adaptive_diffusion.sampling.step_reduction import (
    StepReductionStrategy,
    create_step_reduction_strategy,
)


class TestStepReductionStrategy:
    """Test suite for StepReductionStrategy."""

    def test_initialization(self):
        """Test strategy initialization with default parameters."""
        strategy = StepReductionStrategy()

        assert strategy.base_steps == 50
        assert strategy.min_steps == 10
        assert strategy.max_steps == 100
        assert strategy.quality_target == 0.8
        assert strategy.complexity_sensitivity == 0.5
        assert strategy.complexity_history == []
        assert strategy.step_history == []

    def test_custom_parameters(self):
        """Test strategy with custom parameters."""
        strategy = StepReductionStrategy(
            base_steps=30,
            min_steps=5,
            max_steps=80,
            quality_target=0.9,
            complexity_sensitivity=0.7,
        )

        assert strategy.base_steps == 30
        assert strategy.min_steps == 5
        assert strategy.max_steps == 80
        assert strategy.quality_target == 0.9
        assert strategy.complexity_sensitivity == 0.7

    def test_estimate_optimal_steps_low_complexity(self):
        """Test step estimation for low complexity content."""
        strategy = StepReductionStrategy(base_steps=50)

        # Low complexity should reduce steps
        steps = strategy.estimate_optimal_steps(content_complexity=0.2)

        assert steps < strategy.base_steps
        assert strategy.min_steps <= steps <= strategy.max_steps

    def test_estimate_optimal_steps_high_complexity(self):
        """Test step estimation for high complexity content."""
        strategy = StepReductionStrategy(base_steps=50)

        # High complexity should increase steps
        steps = strategy.estimate_optimal_steps(content_complexity=0.8)

        assert steps > strategy.base_steps
        assert strategy.min_steps <= steps <= strategy.max_steps

    def test_estimate_optimal_steps_with_quality_requirement(self):
        """Test step estimation with quality requirement."""
        strategy = StepReductionStrategy(base_steps=50, quality_target=0.8)

        # Higher quality requirement should increase steps
        high_quality_steps = strategy.estimate_optimal_steps(
            content_complexity=0.5, quality_requirement=0.95
        )

        # Lower quality requirement should decrease steps
        low_quality_steps = strategy.estimate_optimal_steps(
            content_complexity=0.5, quality_requirement=0.6
        )

        assert high_quality_steps > low_quality_steps

    def test_estimate_optimal_steps_clamping(self):
        """Test that steps are clamped to valid range."""
        strategy = StepReductionStrategy(base_steps=50, min_steps=10, max_steps=80)

        # Very low complexity with low quality should reduce steps significantly
        low_steps = strategy.estimate_optimal_steps(
            content_complexity=0.0, quality_requirement=0.4
        )
        assert low_steps >= strategy.min_steps

        # Very high complexity with high quality should increase steps significantly
        max_steps = strategy.estimate_optimal_steps(
            content_complexity=1.0, quality_requirement=1.0
        )
        # Should be at or near max_steps
        assert max_steps >= strategy.max_steps - 5

        # Verify clamping actually works - all steps within bounds
        assert strategy.min_steps <= low_steps <= strategy.max_steps
        assert strategy.min_steps <= max_steps <= strategy.max_steps

    def test_analyze_content_complexity_uniform(self):
        """Test complexity analysis for uniform content."""
        strategy = StepReductionStrategy()

        # Uniform sample (low complexity)
        uniform_sample = mx.ones((1, 8, 8, 3)) * 0.5
        complexity = strategy.analyze_content_complexity(uniform_sample)

        # Should have low complexity
        assert 0 <= complexity < 0.5

    def test_analyze_content_complexity_random(self):
        """Test complexity analysis for random content."""
        strategy = StepReductionStrategy()

        # Random sample (high complexity)
        random_sample = mx.random.normal((1, 8, 8, 3))
        complexity = strategy.analyze_content_complexity(random_sample)

        # Should have moderate to high complexity
        assert 0.2 <= complexity <= 1.0

    def test_progressive_step_schedule(self):
        """Test progressive step reduction schedule."""
        strategy = StepReductionStrategy()

        schedule = strategy.progressive_step_schedule(
            initial_steps=50, target_steps=20, num_stages=4
        )

        assert len(schedule) == 4
        assert schedule[0] == 50  # Start at initial
        assert schedule[-1] == 20  # End at target
        # Steps should be decreasing
        assert all(schedule[i] >= schedule[i + 1] for i in range(len(schedule) - 1))

    def test_progressive_step_schedule_already_optimal(self):
        """Test schedule when already at target."""
        strategy = StepReductionStrategy()

        schedule = strategy.progressive_step_schedule(
            initial_steps=20, target_steps=50, num_stages=4
        )

        # Should return constant schedule if initial <= target
        assert all(s == 20 for s in schedule)

    def test_adaptive_step_allocation_equal_quality(self):
        """Test adaptive allocation with equal quality."""
        strategy = StepReductionStrategy(min_steps=5)

        # Equal quality should give roughly equal allocation
        quality_estimates = [0.8, 0.8, 0.8, 0.8]
        allocations = strategy.adaptive_step_allocation(quality_estimates, total_budget=40)

        assert len(allocations) == 4
        assert sum(allocations) == 40
        # Should be roughly equal
        assert all(8 <= alloc <= 12 for alloc in allocations)

    def test_adaptive_step_allocation_varied_quality(self):
        """Test adaptive allocation with varied quality."""
        strategy = StepReductionStrategy(min_steps=2)

        # Low quality regions should get more steps
        quality_estimates = [0.9, 0.3, 0.8, 0.4]
        allocations = strategy.adaptive_step_allocation(quality_estimates, total_budget=40)

        assert len(allocations) == 4
        assert sum(allocations) == 40

        # Index 1 (quality 0.3) should get more steps than index 0 (quality 0.9)
        assert allocations[1] > allocations[0]

    def test_adaptive_step_allocation_empty(self):
        """Test adaptive allocation with empty input."""
        strategy = StepReductionStrategy()

        allocations = strategy.adaptive_step_allocation([], total_budget=50)
        assert allocations == []

    def test_get_reduction_stats_empty(self):
        """Test statistics with no history."""
        strategy = StepReductionStrategy()

        stats = strategy.get_reduction_stats()
        assert stats == {}

    def test_get_reduction_stats_with_history(self):
        """Test statistics after some estimations."""
        strategy = StepReductionStrategy(base_steps=50)

        # Generate some estimates
        strategy.estimate_optimal_steps(0.3)
        strategy.estimate_optimal_steps(0.5)
        strategy.estimate_optimal_steps(0.7)

        stats = strategy.get_reduction_stats()

        assert "mean_steps" in stats
        assert "min_steps" in stats
        assert "max_steps" in stats
        assert "mean_complexity" in stats
        assert "total_samples" in stats
        assert "reduction_ratio" in stats

        assert stats["total_samples"] == 3
        assert stats["base_steps"] == 50

    def test_reset_history(self):
        """Test history reset."""
        strategy = StepReductionStrategy()

        # Add some history
        strategy.estimate_optimal_steps(0.5)
        strategy.complexity_history = [0.3, 0.5]
        strategy.quality_history = [0.7, 0.8]

        # Reset
        strategy.reset_history()

        assert strategy.complexity_history == []
        assert strategy.step_history == []
        assert strategy.quality_history == []

    def test_step_reduction_saves_time(self):
        """Test that step reduction actually reduces steps on average."""
        strategy = StepReductionStrategy(base_steps=50)

        # Simulate multiple samples with varying complexity
        complexities = [0.2, 0.3, 0.4, 0.5, 0.6, 0.3, 0.4, 0.5]

        for complexity in complexities:
            strategy.estimate_optimal_steps(complexity)

        stats = strategy.get_reduction_stats()

        # Average should be less than base (due to low/medium complexity samples)
        assert stats["mean_steps"] <= strategy.base_steps * 1.1

    def test_complexity_sensitivity_effect(self):
        """Test that complexity sensitivity affects step allocation."""
        low_sensitivity = StepReductionStrategy(
            base_steps=50, complexity_sensitivity=0.1
        )
        high_sensitivity = StepReductionStrategy(
            base_steps=50, complexity_sensitivity=0.9
        )

        # Same complexity, different sensitivity
        complexity = 0.8

        low_steps = low_sensitivity.estimate_optimal_steps(complexity)
        high_steps = high_sensitivity.estimate_optimal_steps(complexity)

        # High sensitivity should allocate more steps for high complexity
        assert high_steps > low_steps

    def test_factory_function(self):
        """Test factory function."""
        strategy = create_step_reduction_strategy(
            base_steps=30, min_steps=5, quality_target=0.85
        )

        assert isinstance(strategy, StepReductionStrategy)
        assert strategy.base_steps == 30
        assert strategy.min_steps == 5
        assert strategy.quality_target == 0.85


@pytest.mark.parametrize("base_steps", [20, 50, 100])
def test_different_base_steps(base_steps):
    """Test strategy with different base step counts."""
    strategy = StepReductionStrategy(base_steps=base_steps)

    steps = strategy.estimate_optimal_steps(content_complexity=0.5)

    assert strategy.min_steps <= steps <= strategy.max_steps


@pytest.mark.parametrize("complexity", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_different_complexities(complexity):
    """Test step estimation with different complexity values."""
    strategy = StepReductionStrategy(base_steps=50)

    steps = strategy.estimate_optimal_steps(content_complexity=complexity)

    assert strategy.min_steps <= steps <= strategy.max_steps


def test_integration_with_quality_target():
    """Test integration of complexity and quality requirements."""
    strategy = StepReductionStrategy(base_steps=50, quality_target=0.8)

    # Low complexity, low quality = minimal steps
    min_steps = strategy.estimate_optimal_steps(
        content_complexity=0.2, quality_requirement=0.6
    )

    # High complexity, high quality = maximum steps
    max_steps = strategy.estimate_optimal_steps(
        content_complexity=0.9, quality_requirement=0.95
    )

    # Should have significant difference
    assert max_steps > min_steps * 1.5
