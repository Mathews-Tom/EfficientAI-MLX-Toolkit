"""
Tests for Optimization Pipeline
"""

import pytest

from adaptive_diffusion.optimization.domain_adapter import DomainType
from adaptive_diffusion.optimization.pipeline import (
    OptimizationPipeline,
    create_optimization_pipeline,
)
from adaptive_diffusion.schedulers.adaptive import AdaptiveScheduler
from adaptive_diffusion.sampling.quality_guided import QualityGuidedSampler


class TestOptimizationPipeline:
    """Test suite for optimization pipeline."""

    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        pipeline = OptimizationPipeline(verbose=0)

        assert pipeline.domain_adapter is not None
        assert pipeline.use_domain_adaptation is True
        assert pipeline.use_rl_optimization is True

    def test_pipeline_without_domain_adaptation(self):
        """Test pipeline without domain adaptation."""
        pipeline = OptimizationPipeline(use_domain_adaptation=False, verbose=0)

        assert pipeline.use_domain_adaptation is False

    def test_pipeline_without_rl(self):
        """Test pipeline without RL optimization."""
        pipeline = OptimizationPipeline(use_rl_optimization=False, verbose=0)

        assert pipeline.use_rl_optimization is False
        assert pipeline.rl_agent is None

    def test_pipeline_optimize_with_domain_only(self):
        """Test optimization with only domain adaptation."""
        pipeline = OptimizationPipeline(
            use_domain_adaptation=True, use_rl_optimization=False, verbose=0
        )

        results = pipeline.optimize(
            prompt="A realistic photograph", num_training_steps=0
        )

        assert results["domain_type"] == DomainType.PHOTOREALISTIC
        assert results["domain_config"] is not None
        assert results["final_config"] is not None

    def test_pipeline_optimize_with_rl(self):
        """Test full optimization with RL."""
        pipeline = OptimizationPipeline(
            use_domain_adaptation=True, use_rl_optimization=True, verbose=0
        )

        results = pipeline.optimize(
            prompt="Abstract geometric patterns",
            num_training_steps=200,
            num_optimization_episodes=2,
        )

        assert results["domain_type"] == DomainType.ABSTRACT
        assert results["training_stats"] is not None
        assert results["optimization_results"] is not None
        assert results["final_config"] is not None

    def test_pipeline_create_optimized_scheduler(self):
        """Test creating optimized scheduler."""
        pipeline = OptimizationPipeline(use_rl_optimization=False, verbose=0)

        scheduler = pipeline.create_optimized_scheduler(
            config={"num_steps": 60, "adaptive_threshold": 0.65, "progress_power": 2.5}
        )

        assert isinstance(scheduler, AdaptiveScheduler)
        assert scheduler.num_inference_steps == 60

    def test_pipeline_create_optimized_sampler(self):
        """Test creating optimized sampler."""
        pipeline = OptimizationPipeline(use_rl_optimization=False, verbose=0)

        sampler = pipeline.create_optimized_sampler(
            config={"adaptive_threshold": 0.7, "num_steps": 50}
        )

        assert isinstance(sampler, QualityGuidedSampler)

    def test_pipeline_history_tracking(self):
        """Test optimization history tracking."""
        pipeline = OptimizationPipeline(use_rl_optimization=False, verbose=0)

        # Run multiple optimizations
        pipeline.optimize(prompt="Portrait", num_training_steps=0)
        pipeline.optimize(prompt="Landscape", num_training_steps=0)

        history = pipeline.get_history()

        assert len(history) == 2
        assert history[0]["domain_type"] == DomainType.PORTRAIT
        assert history[1]["domain_type"] == DomainType.LANDSCAPE

    def test_create_optimization_pipeline_factory(self):
        """Test pipeline factory function."""
        pipeline = create_optimization_pipeline(
            use_domain_adaptation=True, use_rl_optimization=False
        )

        assert isinstance(pipeline, OptimizationPipeline)
        assert pipeline.use_domain_adaptation is True
        assert pipeline.use_rl_optimization is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
