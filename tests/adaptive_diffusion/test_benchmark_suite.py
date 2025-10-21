"""
Benchmark Suite Tests

Tests for the comprehensive benchmarking infrastructure.
"""

from __future__ import annotations

import platform
from pathlib import Path
import tempfile

import pytest

# Import benchmarking infrastructure (using pythonpath from pyproject.toml)
from benchmarks.adaptive_diffusion.benchmark_suite import (
    BenchmarkRunner,
    BenchmarkConfig,
    BenchmarkResult,
    run_comprehensive_benchmarks,
)
from benchmarks.adaptive_diffusion.validation_metrics import (
    ValidationMetrics,
    QualityMetrics,
    PerformanceMetrics,
    MetricsCalculator,
)


# Skip tests if not on Apple Silicon
pytestmark = pytest.mark.skipif(
    not (platform.machine() == "arm64" and platform.system() == "Darwin"),
    reason="Apple Silicon required for MLX tests",
)


class TestBenchmarkConfig:
    """Tests for BenchmarkConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = BenchmarkConfig()

        assert config.model_name == "diffusion-base"
        assert config.image_size == (64, 64)
        assert config.num_iterations == 3
        assert config.batch_size == 2
        assert config.num_inference_steps == 30
        assert config.track_memory is True
        assert config.save_results is True
        assert config.verbose is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = BenchmarkConfig(
            model_name="test-model",
            image_size=(128, 128),
            num_iterations=5,
            batch_size=4,
            num_inference_steps=50,
            verbose=False,
        )

        assert config.model_name == "test-model"
        assert config.image_size == (128, 128)
        assert config.num_iterations == 5
        assert config.batch_size == 4
        assert config.num_inference_steps == 50
        assert config.verbose is False


class TestBenchmarkResult:
    """Tests for BenchmarkResult."""

    def test_result_creation(self):
        """Test creating a benchmark result."""
        result = BenchmarkResult(
            config={"test": "config"},
            scheduler_name="TestScheduler",
            model_name="test-model",
            total_time=10.5,
            avg_time_per_image=1.2,
            images_per_sec=0.83,
            steps_per_sec=25.0,
            device="apple_silicon",
        )

        assert result.scheduler_name == "TestScheduler"
        assert result.total_time == 10.5
        assert result.device == "apple_silicon"

    def test_result_to_dict(self):
        """Test converting result to dictionary."""
        result = BenchmarkResult(
            config={},
            scheduler_name="TestScheduler",
            model_name="test-model",
            total_time=10.0,
            avg_time_per_image=1.0,
            images_per_sec=1.0,
            steps_per_sec=30.0,
        )

        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert "scheduler_name" in result_dict
        assert "total_time" in result_dict
        assert result_dict["scheduler_name"] == "TestScheduler"

    def test_result_to_json(self):
        """Test converting result to JSON."""
        result = BenchmarkResult(
            config={},
            scheduler_name="TestScheduler",
            model_name="test-model",
            total_time=10.0,
            avg_time_per_image=1.0,
            images_per_sec=1.0,
            steps_per_sec=30.0,
        )

        json_str = result.to_json()

        assert isinstance(json_str, str)
        assert "TestScheduler" in json_str
        assert "total_time" in json_str


class TestBenchmarkRunner:
    """Tests for BenchmarkRunner."""

    def test_runner_initialization(self):
        """Test runner initialization."""
        config = BenchmarkConfig(verbose=False)
        runner = BenchmarkRunner(config)

        assert runner.config == config
        assert runner.device in ["apple_silicon", "cpu"]
        assert len(runner.results) == 0

    def test_device_detection(self):
        """Test device detection."""
        runner = BenchmarkRunner()
        device = runner._detect_device()

        assert device in ["apple_silicon", "cpu"]
        # On macOS ARM64, should detect Apple Silicon
        if platform.machine() == "arm64" and platform.system() == "Darwin":
            assert device == "apple_silicon"

    def test_benchmark_scheduler(self):
        """Test benchmarking a single scheduler."""
        from adaptive_diffusion.baseline import DDIMScheduler

        config = BenchmarkConfig(
            num_iterations=2,
            batch_size=1,
            num_inference_steps=10,
            warmup_iterations=1,
            verbose=False,
        )
        runner = BenchmarkRunner(config)

        scheduler = DDIMScheduler()
        result = runner.benchmark_scheduler("DDIM", scheduler)

        # Validate result
        assert isinstance(result, BenchmarkResult)
        assert result.scheduler_name == "DDIM"
        assert result.total_time > 0
        assert result.images_per_sec > 0
        assert result.steps_per_sec > 0
        assert result.device == runner.device

    def test_run_all_benchmarks(self):
        """Test running all benchmarks."""
        config = BenchmarkConfig(
            num_iterations=2,
            batch_size=1,
            num_inference_steps=10,
            warmup_iterations=1,
            verbose=False,
            save_results=False,
        )
        runner = BenchmarkRunner(config)

        results = runner.run_all_benchmarks()

        # Should benchmark all 4 schedulers
        assert len(results) == 4
        scheduler_names = [r.scheduler_name for r in results]
        assert "DDPM" in scheduler_names
        assert "DDIM" in scheduler_names
        assert "DPM-Solver" in scheduler_names
        assert "Adaptive" in scheduler_names

        # All results should have valid metrics
        for result in results:
            assert result.total_time > 0
            assert result.images_per_sec > 0

    def test_save_results(self):
        """Test saving benchmark results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            config = BenchmarkConfig(
                num_iterations=2,
                batch_size=1,
                num_inference_steps=10,
                warmup_iterations=1,
                output_dir=output_dir,
                save_results=True,
                verbose=False,
            )
            runner = BenchmarkRunner(config)

            results = runner.run_all_benchmarks()

            # Check that files were created
            summary_file = output_dir / "benchmark_summary.json"
            assert summary_file.exists()

            # Check individual result files
            for result in results:
                result_file = output_dir / f"benchmark_{result.scheduler_name.lower().replace(' ', '_')}.json"
                assert result_file.exists()


class TestValidationMetrics:
    """Tests for validation metrics classes."""

    def test_quality_metrics_creation(self):
        """Test creating quality metrics."""
        metrics = QualityMetrics(
            fid_score=25.5,
            clip_score=0.85,
            ssim=0.92,
            num_samples=10,
        )

        assert metrics.fid_score == 25.5
        assert metrics.clip_score == 0.85
        assert metrics.ssim == 0.92
        assert metrics.num_samples == 10

    def test_quality_metrics_to_dict(self):
        """Test converting quality metrics to dict."""
        metrics = QualityMetrics(fid_score=25.5, num_samples=10)
        metrics_dict = metrics.to_dict()

        assert isinstance(metrics_dict, dict)
        assert "fid_score" in metrics_dict
        assert "num_samples" in metrics_dict
        assert metrics_dict["fid_score"] == 25.5

    def test_performance_metrics_creation(self):
        """Test creating performance metrics."""
        metrics = PerformanceMetrics(
            total_time=10.5,
            avg_time_per_image=1.05,
            images_per_sec=0.95,
            steps_per_sec=28.5,
            total_images=10,
            total_steps=300,
        )

        assert metrics.total_time == 10.5
        assert metrics.images_per_sec == 0.95
        assert metrics.total_images == 10

    def test_validation_metrics_creation(self):
        """Test creating validation metrics."""
        quality = QualityMetrics(fid_score=25.0)
        performance = PerformanceMetrics(total_time=10.0, images_per_sec=1.0)

        metrics = ValidationMetrics(
            quality=quality,
            performance=performance,
            scheduler_name="TestScheduler",
        )

        assert metrics.scheduler_name == "TestScheduler"
        assert metrics.quality.fid_score == 25.0
        assert metrics.performance.images_per_sec == 1.0

    def test_overall_score_computation(self):
        """Test computing overall score."""
        quality = QualityMetrics(fid_score=50.0)  # FID score
        performance = PerformanceMetrics(
            total_time=10.0,
            images_per_sec=5.0,  # 5 images/sec
            avg_time_per_image=0.2,
            steps_per_sec=150.0,
        )

        metrics = ValidationMetrics(quality=quality, performance=performance)
        overall = metrics.compute_overall_score(quality_weight=0.6)

        assert isinstance(overall, float)
        assert 0 <= overall <= 1
        assert metrics.overall_score == overall
        assert metrics.quality_performance_ratio is not None


class TestMetricsCalculator:
    """Tests for MetricsCalculator."""

    def test_compute_quality_metrics(self):
        """Test computing quality metrics from images."""
        import mlx.core as mx

        # Generate random test images
        images = mx.random.normal((4, 64, 64, 3))

        metrics = MetricsCalculator.compute_quality_metrics(images)

        assert isinstance(metrics, QualityMetrics)
        assert metrics.num_samples == 4
        assert metrics.pixel_variance is not None
        assert metrics.pixel_variance > 0
        assert metrics.edge_sharpness is not None
        assert metrics.fid_score is not None

    def test_compute_performance_metrics(self):
        """Test computing performance metrics."""
        iteration_times = [1.0, 1.1, 0.9, 1.0]

        metrics = MetricsCalculator.compute_performance_metrics(
            iteration_times=iteration_times,
            num_images=8,
            num_steps=30,
            batch_size=2,
        )

        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.total_time == sum(iteration_times)
        assert metrics.total_images == 8
        assert metrics.total_steps == 8 * 30
        assert metrics.images_per_sec > 0
        assert metrics.std_time is not None


@pytest.mark.integration
class TestComprehensiveBenchmarks:
    """Integration tests for comprehensive benchmarks."""

    def test_run_comprehensive_benchmarks(self):
        """Test running comprehensive benchmarks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            results = run_comprehensive_benchmarks(
                output_dir=output_dir,
                verbose=False,
            )

            # Should have results for all schedulers
            assert len(results) >= 4

            # Should have created summary file
            summary_file = output_dir / "benchmark_summary.json"
            assert summary_file.exists()
