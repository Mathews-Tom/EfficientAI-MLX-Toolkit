"""
Validation Metrics Tests

Tests for validation metrics infrastructure.
"""

from __future__ import annotations

import pytest
import sys
from pathlib import Path

# Import benchmarking infrastructure (using pythonpath from pyproject.toml)
from benchmarks.adaptive_diffusion.validation_metrics import (
    QualityMetrics,
    PerformanceMetrics,
    ValidationMetrics,
    MetricsCalculator,
)


class TestQualityMetrics:
    """Tests for QualityMetrics."""

    def test_creation(self):
        """Test creating quality metrics."""
        metrics = QualityMetrics(
            fid_score=25.5,
            clip_score=0.85,
            ssim=0.92,
            psnr=28.5,
            num_samples=10,
        )

        assert metrics.fid_score == 25.5
        assert metrics.clip_score == 0.85
        assert metrics.ssim == 0.92
        assert metrics.psnr == 28.5
        assert metrics.num_samples == 10

    def test_to_dict(self):
        """Test converting to dictionary."""
        metrics = QualityMetrics(fid_score=25.0, num_samples=5)
        metrics_dict = metrics.to_dict()

        assert isinstance(metrics_dict, dict)
        assert "fid_score" in metrics_dict
        assert "num_samples" in metrics_dict
        assert metrics_dict["fid_score"] == 25.0
        assert metrics_dict["num_samples"] == 5

    def test_default_values(self):
        """Test default values."""
        metrics = QualityMetrics()

        assert metrics.fid_score is None
        assert metrics.clip_score is None
        assert metrics.num_samples == 0


class TestPerformanceMetrics:
    """Tests for PerformanceMetrics."""

    def test_creation(self):
        """Test creating performance metrics."""
        metrics = PerformanceMetrics(
            total_time=10.5,
            avg_time_per_image=1.05,
            avg_time_per_step=0.035,
            images_per_sec=0.95,
            steps_per_sec=28.5,
            total_images=10,
            total_steps=300,
        )

        assert metrics.total_time == 10.5
        assert metrics.avg_time_per_image == 1.05
        assert metrics.images_per_sec == 0.95
        assert metrics.total_images == 10
        assert metrics.total_steps == 300

    def test_to_dict(self):
        """Test converting to dictionary."""
        metrics = PerformanceMetrics(
            total_time=10.0,
            avg_time_per_image=1.0,
            images_per_sec=1.0,
        )
        metrics_dict = metrics.to_dict()

        assert isinstance(metrics_dict, dict)
        assert "total_time" in metrics_dict
        assert "images_per_sec" in metrics_dict
        assert metrics_dict["total_time"] == 10.0

    def test_memory_metrics(self):
        """Test memory-related metrics."""
        metrics = PerformanceMetrics(
            total_time=10.0,
            peak_memory_mb=512.5,
            avg_memory_mb=400.2,
            memory_efficiency=200.1,
        )

        assert metrics.peak_memory_mb == 512.5
        assert metrics.avg_memory_mb == 400.2
        assert metrics.memory_efficiency == 200.1


class TestValidationMetrics:
    """Tests for ValidationMetrics."""

    def test_creation(self):
        """Test creating validation metrics."""
        quality = QualityMetrics(fid_score=25.0)
        performance = PerformanceMetrics(total_time=10.0, images_per_sec=1.0)

        metrics = ValidationMetrics(
            quality=quality,
            performance=performance,
            scheduler_name="TestScheduler",
            model_name="test-model",
        )

        assert metrics.scheduler_name == "TestScheduler"
        assert metrics.model_name == "test-model"
        assert metrics.quality.fid_score == 25.0
        assert metrics.performance.images_per_sec == 1.0

    def test_to_dict(self):
        """Test converting to dictionary."""
        quality = QualityMetrics(fid_score=25.0)
        performance = PerformanceMetrics(total_time=10.0, images_per_sec=1.0)
        metrics = ValidationMetrics(quality=quality, performance=performance)

        metrics_dict = metrics.to_dict()

        assert isinstance(metrics_dict, dict)
        assert "quality" in metrics_dict
        assert "performance" in metrics_dict
        assert "scheduler_name" in metrics_dict

    def test_overall_score_computation(self):
        """Test computing overall score."""
        quality = QualityMetrics(fid_score=50.0)
        performance = PerformanceMetrics(
            total_time=10.0,
            images_per_sec=5.0,
            avg_time_per_image=0.2,
            steps_per_sec=150.0,
        )

        metrics = ValidationMetrics(quality=quality, performance=performance)
        overall = metrics.compute_overall_score(quality_weight=0.6)

        assert isinstance(overall, float)
        assert 0 <= overall <= 1
        assert metrics.overall_score == overall
        assert metrics.quality_performance_ratio is not None
        assert metrics.quality_performance_ratio > 0

    def test_overall_score_with_clip(self):
        """Test overall score with CLIP score instead of FID."""
        quality = QualityMetrics(clip_score=0.85)  # High CLIP score
        performance = PerformanceMetrics(total_time=10.0, images_per_sec=5.0)

        metrics = ValidationMetrics(quality=quality, performance=performance)
        overall = metrics.compute_overall_score(quality_weight=0.7)

        assert isinstance(overall, float)
        assert 0 <= overall <= 1
        # With high CLIP score and good performance, should be > 0.5
        assert overall > 0.5


class TestMetricsCalculator:
    """Tests for MetricsCalculator."""

    def test_compute_performance_metrics(self):
        """Test computing performance metrics from timing data."""
        iteration_times = [1.0, 1.1, 0.9, 1.0, 1.2]

        metrics = MetricsCalculator.compute_performance_metrics(
            iteration_times=iteration_times,
            num_images=10,
            num_steps=30,
            batch_size=2,
        )

        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.total_time == sum(iteration_times)
        assert metrics.total_images == 10
        assert metrics.total_steps == 10 * 30
        assert metrics.batch_size == 2
        assert metrics.images_per_sec > 0
        assert metrics.steps_per_sec > 0
        assert metrics.std_time is not None
        assert metrics.min_time == min(iteration_times)
        assert metrics.max_time == max(iteration_times)

    def test_compute_performance_with_memory(self):
        """Test computing performance metrics with memory data."""
        iteration_times = [1.0, 1.0, 1.0]
        memory_usage = [512.0, 520.0, 518.0]

        metrics = MetricsCalculator.compute_performance_metrics(
            iteration_times=iteration_times,
            num_images=6,
            num_steps=30,
            batch_size=2,
            memory_usage=memory_usage,
        )

        assert metrics.peak_memory_mb is not None
        assert metrics.avg_memory_mb is not None
        assert metrics.memory_efficiency is not None
        assert metrics.peak_memory_mb >= metrics.avg_memory_mb
        assert metrics.memory_efficiency == metrics.avg_memory_mb / 2  # batch_size=2


# MLX-dependent tests (only run if MLX is available)
try:
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False


@pytest.mark.skipif(not MLX_AVAILABLE, reason="MLX not available")
class TestMetricsCalculatorWithMLX:
    """Tests for MetricsCalculator that require MLX."""

    def test_compute_quality_metrics(self):
        """Test computing quality metrics from images."""
        # Generate random test images
        images = mx.random.normal((4, 64, 64, 3))

        metrics = MetricsCalculator.compute_quality_metrics(images)

        assert isinstance(metrics, QualityMetrics)
        assert metrics.num_samples == 4
        assert metrics.pixel_variance is not None
        assert metrics.pixel_variance > 0
        assert metrics.edge_sharpness is not None
        assert metrics.edge_sharpness > 0
        assert metrics.fid_score is not None
        assert metrics.evaluation_time is not None
        assert metrics.evaluation_time > 0

    def test_compute_quality_with_color_diversity(self):
        """Test quality metrics include color diversity for RGB images."""
        # RGB images
        images = mx.random.normal((4, 64, 64, 3))

        metrics = MetricsCalculator.compute_quality_metrics(images)

        assert metrics.color_diversity is not None
        assert metrics.color_diversity > 0

    def test_compute_quality_with_reference(self):
        """Test quality metrics with reference images."""
        generated = mx.random.normal((2, 64, 64, 3))
        reference = mx.random.normal((2, 64, 64, 3))

        metrics = MetricsCalculator.compute_quality_metrics(
            generated, reference_images=reference
        )

        assert metrics.ssim is not None
        assert metrics.psnr is not None
        assert 0 <= metrics.ssim <= 1

    def test_edge_sharpness_computation(self):
        """Test edge sharpness computation."""
        # Create image with some edges
        images = mx.random.normal((2, 32, 32, 3))

        sharpness = MetricsCalculator._compute_edge_sharpness(images)

        assert isinstance(sharpness, float)
        assert sharpness > 0

    def test_ssim_computation(self):
        """Test SSIM computation."""
        # Same images should have high SSIM
        mx.random.seed(42)
        images1 = mx.random.normal((2, 32, 32, 3))
        images2 = mx.array(images1)  # Create copy using mx.array

        ssim = MetricsCalculator._compute_ssim(images1, images2)

        assert isinstance(ssim, float)
        assert ssim > 0.9  # Should be very high for identical images

    def test_psnr_computation(self):
        """Test PSNR computation."""
        # Same images should have very high PSNR
        mx.random.seed(42)
        images1 = mx.random.normal((2, 32, 32, 3))
        images2 = mx.array(images1)  # Create copy using mx.array

        psnr = MetricsCalculator._compute_psnr(images1, images2)

        assert isinstance(psnr, float)
        assert psnr > 50  # Should be very high for identical images

    def test_fid_proxy_estimation(self):
        """Test FID proxy estimation."""
        images = mx.random.normal((4, 64, 64, 3))

        fid_proxy = MetricsCalculator._estimate_fid_proxy(images)

        assert isinstance(fid_proxy, float)
        assert fid_proxy >= 0

    def test_skewness_computation(self):
        """Test skewness computation."""
        data = mx.random.normal((100,))

        skewness = MetricsCalculator._compute_skewness(data)

        assert isinstance(skewness, float)
        # Normal distribution should have skewness near 0
        assert abs(skewness) < 1.0

    def test_kurtosis_computation(self):
        """Test kurtosis computation."""
        data = mx.random.normal((100,))

        kurtosis = MetricsCalculator._compute_kurtosis(data)

        assert isinstance(kurtosis, float)
        # Normal distribution should have kurtosis near 3
        assert 2.0 < kurtosis < 4.0


# scipy-dependent tests (only run if scipy is available)
try:
    from scipy import stats
    from adaptive_diffusion.validation_metrics import compute_statistical_significance
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


@pytest.mark.skipif(not SCIPY_AVAILABLE, reason="scipy not available")
class TestStatisticalSignificance:
    """Tests for statistical significance computation."""

    def test_compute_significance(self):
        """Test computing statistical significance."""
        baseline_times = [1.0, 1.1, 0.9, 1.0, 1.2]
        optimized_times = [0.8, 0.9, 0.7, 0.85, 0.95]

        significance = compute_statistical_significance(
            baseline_times, optimized_times, confidence=0.95
        )

        assert "t_statistic" in significance
        assert "p_value" in significance
        assert "cohens_d" in significance
        assert "speedup" in significance
        assert "is_significant" in significance

        # Optimized should be faster
        assert significance["speedup"] > 1.0

    def test_non_significant_difference(self):
        """Test when difference is not significant."""
        baseline_times = [1.0, 1.0, 1.0]
        optimized_times = [1.0, 1.0, 1.0]

        significance = compute_statistical_significance(
            baseline_times, optimized_times
        )

        # Same times should not be significant
        assert significance["is_significant"] is False
        assert significance["speedup"] == pytest.approx(1.0, rel=0.01)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
