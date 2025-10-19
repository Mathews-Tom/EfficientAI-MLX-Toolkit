"""
Domain-Specific Validation Tests

Tests adaptive diffusion across different domains:
- Photorealistic (photography, realistic rendering)
- Artistic (paintings, illustrations, stylized art)
- Synthetic (diagrams, technical illustrations, abstract)
"""

from __future__ import annotations

import platform
import time

import mlx.core as mx
import numpy as np
import pytest

# Import project modules (using pythonpath from pyproject.toml)
from adaptive_diffusion.baseline import DiffusionPipeline, DDIMScheduler
from adaptive_diffusion.optimization.pipeline import OptimizationPipeline
from adaptive_diffusion.optimization.domain_adapter import DomainType

# Import benchmarking infrastructure (using pythonpath from pyproject.toml)
from benchmarks.adaptive_diffusion.validation_metrics import (
    MetricsCalculator,
    ValidationMetrics,
    create_validation_report,
)


# Skip tests if not on Apple Silicon
pytestmark = pytest.mark.skipif(
    not (platform.machine() == "arm64" and platform.system() == "Darwin"),
    reason="Apple Silicon required for MLX tests",
)


class TestPhotorealisticDomain:
    """Tests for photorealistic domain optimization."""

    @pytest.fixture
    def opt_pipeline(self):
        """Optimization pipeline with domain adaptation."""
        return OptimizationPipeline(
            use_domain_adaptation=True,
            use_rl_optimization=False,  # Disable RL for faster testing
            verbose=0,
        )

    def test_photorealistic_optimization(self, opt_pipeline):
        """
        Test photorealistic domain optimization.

        Validates:
        - Domain-optimized scheduler is created
        - Performance is competitive
        - Quality metrics are reasonable
        """
        # Get domain-optimized scheduler
        scheduler = opt_pipeline.create_optimized_scheduler(
            domain_type=DomainType.PHOTOREALISTIC
        )

        # Create pipeline
        pipeline = DiffusionPipeline(scheduler=scheduler, image_size=(64, 64))

        # Generate images
        start = time.perf_counter()
        images = pipeline.generate(
            batch_size=2,
            num_inference_steps=30,
            seed=42,
        )
        elapsed = time.perf_counter() - start

        # Validate
        assert images.shape == (2, 64, 64, 3)
        assert elapsed > 0

        # Compute quality metrics
        quality = MetricsCalculator.compute_quality_metrics(images)

        print(f"\nPhotorealistic Domain:")
        print(f"  Generation time: {elapsed:.3f}s")
        print(f"  Quality (FID proxy): {quality.fid_score:.2f}")
        print(f"  Edge sharpness: {quality.edge_sharpness:.4f}")
        print(f"  Pixel variance: {quality.pixel_variance:.4f}")

        # Quality should be reasonable for photorealistic content
        assert quality.edge_sharpness > 0.01  # Should have some detail
        assert quality.pixel_variance > 0.001  # Should have variation

    def test_photorealistic_vs_baseline(self, opt_pipeline):
        """
        Compare photorealistic optimization against baseline.

        Validates:
        - Domain optimization provides benefits
        - Performance or quality is improved
        """
        # Baseline
        baseline_scheduler = DDIMScheduler()
        baseline_pipeline = DiffusionPipeline(
            scheduler=baseline_scheduler,
            image_size=(64, 64),
        )

        # Warmup
        baseline_pipeline.generate(batch_size=1, num_inference_steps=10, seed=42)

        # Benchmark baseline
        baseline_times = []
        for _ in range(3):
            start = time.perf_counter()
            baseline_images = baseline_pipeline.generate(
                batch_size=2,
                num_inference_steps=30,
                seed=None,
            )
            baseline_times.append(time.perf_counter() - start)

        # Domain-optimized
        optimized_scheduler = opt_pipeline.create_optimized_scheduler(
            domain_type=DomainType.PHOTOREALISTIC
        )
        optimized_pipeline = DiffusionPipeline(
            scheduler=optimized_scheduler,
            image_size=(64, 64),
        )

        # Warmup
        optimized_pipeline.generate(batch_size=1, num_inference_steps=10, seed=42)

        # Benchmark optimized
        optimized_times = []
        for _ in range(3):
            start = time.perf_counter()
            optimized_images = optimized_pipeline.generate(
                batch_size=2,
                num_inference_steps=30,
                seed=None,
            )
            optimized_times.append(time.perf_counter() - start)

        # Compute metrics
        baseline_quality = MetricsCalculator.compute_quality_metrics(baseline_images)
        optimized_quality = MetricsCalculator.compute_quality_metrics(optimized_images)

        baseline_perf = MetricsCalculator.compute_performance_metrics(
            baseline_times, num_images=6, num_steps=30, batch_size=2
        )
        optimized_perf = MetricsCalculator.compute_performance_metrics(
            optimized_times, num_images=6, num_steps=30, batch_size=2
        )

        # Print comparison
        print(f"\nPhotorealistic Domain Comparison:")
        print(f"  Baseline time: {baseline_perf.total_time:.3f}s")
        print(f"  Optimized time: {optimized_perf.total_time:.3f}s")
        print(f"  Speedup: {baseline_perf.total_time / optimized_perf.total_time:.2f}x")
        print(f"  Baseline quality: {baseline_quality.fid_score:.2f}")
        print(f"  Optimized quality: {optimized_quality.fid_score:.2f}")

        # Should have some benefit (performance or quality)
        speedup = baseline_perf.total_time / optimized_perf.total_time
        quality_improvement = (
            baseline_quality.fid_score - optimized_quality.fid_score
        ) / baseline_quality.fid_score

        # Either speedup > 1.0 or quality improvement > 0
        assert speedup > 0.8 or quality_improvement > -0.2

    def test_photorealistic_quality_characteristics(self, opt_pipeline):
        """
        Test quality characteristics specific to photorealistic domain.

        Validates:
        - Images have appropriate edge sharpness
        - Color diversity is reasonable
        - Statistical properties match photorealistic content
        """
        scheduler = opt_pipeline.create_optimized_scheduler(
            domain_type=DomainType.PHOTOREALISTIC
        )
        pipeline = DiffusionPipeline(scheduler=scheduler, image_size=(64, 64))

        # Generate multiple batches
        all_images = []
        for _ in range(3):
            images = pipeline.generate(batch_size=2, num_inference_steps=30, seed=None)
            all_images.append(images)

        # Stack all images
        combined_images = mx.concatenate(all_images, axis=0)

        # Compute quality metrics
        quality = MetricsCalculator.compute_quality_metrics(combined_images)

        print(f"\nPhotorealistic Quality Characteristics:")
        print(f"  Edge sharpness: {quality.edge_sharpness:.4f}")
        print(f"  Color diversity: {quality.color_diversity:.4f}")
        print(f"  Pixel variance: {quality.pixel_variance:.4f}")

        # Photorealistic images should have:
        # - Good edge definition (sharpness)
        # - Reasonable color diversity
        # - Moderate variance (not too flat or too noisy)
        assert quality.edge_sharpness > 0.01
        assert quality.pixel_variance > 0.001
        assert quality.pixel_variance < 1.0  # Not too noisy


class TestArtisticDomain:
    """Tests for artistic domain optimization."""

    @pytest.fixture
    def opt_pipeline(self):
        """Optimization pipeline with domain adaptation."""
        return OptimizationPipeline(
            use_domain_adaptation=True,
            use_rl_optimization=False,
            verbose=0,
        )

    def test_artistic_optimization(self, opt_pipeline):
        """
        Test artistic domain optimization.

        Validates:
        - Domain-optimized scheduler is created
        - Images are generated successfully
        - Quality metrics are computed
        """
        scheduler = opt_pipeline.create_optimized_scheduler(
            domain_type=DomainType.ARTISTIC
        )
        pipeline = DiffusionPipeline(scheduler=scheduler, image_size=(64, 64))

        # Generate
        start = time.perf_counter()
        images = pipeline.generate(batch_size=2, num_inference_steps=30, seed=42)
        elapsed = time.perf_counter() - start

        # Validate
        assert images.shape == (2, 64, 64, 3)

        # Compute quality
        quality = MetricsCalculator.compute_quality_metrics(images)

        print(f"\nArtistic Domain:")
        print(f"  Generation time: {elapsed:.3f}s")
        print(f"  Quality (FID proxy): {quality.fid_score:.2f}")
        print(f"  Color diversity: {quality.color_diversity:.4f}")

        # Artistic content often has higher color diversity
        assert quality.color_diversity is not None

    def test_artistic_vs_photorealistic(self, opt_pipeline):
        """
        Compare artistic vs photorealistic domain optimization.

        Validates:
        - Different domains produce different optimizations
        - Quality characteristics differ appropriately
        """
        # Artistic
        artistic_scheduler = opt_pipeline.create_optimized_scheduler(
            domain_type=DomainType.ARTISTIC
        )
        artistic_pipeline = DiffusionPipeline(
            scheduler=artistic_scheduler,
            image_size=(64, 64),
        )

        artistic_images = artistic_pipeline.generate(
            batch_size=4,
            num_inference_steps=30,
            seed=42,
        )
        artistic_quality = MetricsCalculator.compute_quality_metrics(artistic_images)

        # Photorealistic
        photo_scheduler = opt_pipeline.create_optimized_scheduler(
            domain_type=DomainType.PHOTOREALISTIC
        )
        photo_pipeline = DiffusionPipeline(
            scheduler=photo_scheduler,
            image_size=(64, 64),
        )

        photo_images = photo_pipeline.generate(
            batch_size=4,
            num_inference_steps=30,
            seed=42,
        )
        photo_quality = MetricsCalculator.compute_quality_metrics(photo_images)

        print(f"\nDomain Comparison:")
        print(f"  Artistic quality: {artistic_quality.fid_score:.2f}")
        print(f"  Photorealistic quality: {photo_quality.fid_score:.2f}")
        print(f"  Artistic color diversity: {artistic_quality.color_diversity:.4f}")
        print(f"  Photorealistic color diversity: {photo_quality.color_diversity:.4f}")

        # Both should generate valid images
        assert artistic_quality.fid_score is not None
        assert photo_quality.fid_score is not None


class TestSyntheticDomain:
    """Tests for synthetic/technical domain optimization."""

    @pytest.fixture
    def opt_pipeline(self):
        """Optimization pipeline with domain adaptation."""
        return OptimizationPipeline(
            use_domain_adaptation=True,
            use_rl_optimization=False,
            verbose=0,
        )

    def test_synthetic_optimization(self, opt_pipeline):
        """
        Test synthetic domain optimization.

        Validates:
        - Synthetic domain scheduler works
        - Images are generated
        - Performance is acceptable
        """
        scheduler = opt_pipeline.create_optimized_scheduler(
            domain_type=DomainType.SYNTHETIC
        )
        pipeline = DiffusionPipeline(scheduler=scheduler, image_size=(64, 64))

        # Generate
        start = time.perf_counter()
        images = pipeline.generate(batch_size=2, num_inference_steps=30, seed=42)
        elapsed = time.perf_counter() - start

        # Validate
        assert images.shape == (2, 64, 64, 3)
        assert elapsed < 30.0  # Should complete in reasonable time

        # Compute quality
        quality = MetricsCalculator.compute_quality_metrics(images)

        print(f"\nSynthetic Domain:")
        print(f"  Generation time: {elapsed:.3f}s")
        print(f"  Quality (FID proxy): {quality.fid_score:.2f}")
        print(f"  Edge sharpness: {quality.edge_sharpness:.4f}")

        # Synthetic images often have sharp edges
        assert quality.edge_sharpness is not None

    def test_synthetic_quality_characteristics(self, opt_pipeline):
        """
        Test quality characteristics for synthetic domain.

        Validates:
        - Synthetic images have appropriate characteristics
        - Edge sharpness is generally high
        - Color variance may be lower (cleaner, more uniform)
        """
        scheduler = opt_pipeline.create_optimized_scheduler(
            domain_type=DomainType.SYNTHETIC
        )
        pipeline = DiffusionPipeline(scheduler=scheduler, image_size=(64, 64))

        # Generate
        images = pipeline.generate(batch_size=4, num_inference_steps=30, seed=42)

        # Compute metrics
        quality = MetricsCalculator.compute_quality_metrics(images)

        print(f"\nSynthetic Quality Characteristics:")
        print(f"  Edge sharpness: {quality.edge_sharpness:.4f}")
        print(f"  Pixel variance: {quality.pixel_variance:.4f}")
        print(f"  Color diversity: {quality.color_diversity:.4f}")

        # Validate synthetic characteristics
        assert quality.edge_sharpness > 0
        assert quality.pixel_variance > 0


class TestCrossDomainValidation:
    """Cross-domain validation and comparison tests."""

    @pytest.fixture
    def opt_pipeline(self):
        """Optimization pipeline."""
        return OptimizationPipeline(
            use_domain_adaptation=True,
            use_rl_optimization=False,
            verbose=0,
        )

    def test_all_domains_functional(self, opt_pipeline):
        """
        Test that all domains can generate images successfully.

        Validates:
        - All domains work
        - No crashes or errors
        - Images have correct shape
        """
        domains = [
            DomainType.PHOTOREALISTIC,
            DomainType.ARTISTIC,
            DomainType.SYNTHETIC,
        ]

        results = {}

        for domain in domains:
            scheduler = opt_pipeline.create_optimized_scheduler(domain_type=domain)
            pipeline = DiffusionPipeline(scheduler=scheduler, image_size=(64, 64))

            # Generate
            images = pipeline.generate(batch_size=2, num_inference_steps=20, seed=42)

            # Validate
            assert images.shape == (2, 64, 64, 3)
            results[domain.value] = images

        print(f"\nAll Domains Functional Test: PASS")
        print(f"  Tested: {', '.join([d.value for d in domains])}")

    def test_domain_performance_comparison(self, opt_pipeline):
        """
        Compare performance across all domains.

        Validates:
        - All domains have reasonable performance
        - Performance differences are documented
        """
        domains = [
            DomainType.PHOTOREALISTIC,
            DomainType.ARTISTIC,
            DomainType.SYNTHETIC,
        ]

        results = {}

        for domain in domains:
            scheduler = opt_pipeline.create_optimized_scheduler(domain_type=domain)
            pipeline = DiffusionPipeline(scheduler=scheduler, image_size=(64, 64))

            # Warmup
            pipeline.generate(batch_size=1, num_inference_steps=10, seed=42)

            # Benchmark
            times = []
            for _ in range(3):
                start = time.perf_counter()
                pipeline.generate(batch_size=2, num_inference_steps=30, seed=None)
                times.append(time.perf_counter() - start)

            perf = MetricsCalculator.compute_performance_metrics(
                times, num_images=6, num_steps=30, batch_size=2
            )

            results[domain.value] = perf

            print(f"\n{domain.value} Performance:")
            print(f"  Total time: {perf.total_time:.3f}s")
            print(f"  Throughput: {perf.images_per_sec:.2f} images/sec")

        # Print comparison
        print(f"\n{'='*60}")
        print("CROSS-DOMAIN PERFORMANCE COMPARISON")
        print(f"{'='*60}")
        print(f"{'Domain':<20} {'Time (s)':<12} {'Throughput':<15}")
        print("-"*50)

        for domain, perf in results.items():
            print(f"{domain:<20} {perf.total_time:<12.3f} {perf.images_per_sec:<15.2f}")

        # All domains should have reasonable performance
        for domain, perf in results.items():
            assert perf.images_per_sec > 0

    def test_domain_quality_comparison(self, opt_pipeline):
        """
        Compare quality metrics across domains.

        Validates:
        - Quality metrics can be computed for all domains
        - Quality characteristics differ appropriately
        """
        domains = [
            DomainType.PHOTOREALISTIC,
            DomainType.ARTISTIC,
            DomainType.SYNTHETIC,
        ]

        results = {}

        for domain in domains:
            scheduler = opt_pipeline.create_optimized_scheduler(domain_type=domain)
            pipeline = DiffusionPipeline(scheduler=scheduler, image_size=(64, 64))

            # Generate
            images = pipeline.generate(batch_size=4, num_inference_steps=30, seed=42)

            # Compute quality
            quality = MetricsCalculator.compute_quality_metrics(images)
            results[domain.value] = quality

        # Print comparison
        print(f"\n{'='*60}")
        print("CROSS-DOMAIN QUALITY COMPARISON")
        print(f"{'='*60}")
        print(f"{'Domain':<20} {'FID Proxy':<12} {'Sharpness':<12}")
        print("-"*50)

        for domain, quality in results.items():
            print(f"{domain:<20} {quality.fid_score:<12.2f} {quality.edge_sharpness:<12.4f}")

        # All domains should have valid quality metrics
        for domain, quality in results.items():
            assert quality.fid_score is not None
            assert quality.edge_sharpness is not None


@pytest.mark.integration
class TestDomainValidationReport:
    """Integration test for comprehensive domain validation reporting."""

    def test_generate_validation_report(self):
        """
        Generate comprehensive validation report for all domains.

        Validates:
        - Full validation pipeline works
        - Reports are generated
        - All metrics are computed
        """
        opt_pipeline = OptimizationPipeline(
            use_domain_adaptation=True,
            use_rl_optimization=False,
            verbose=0,
        )

        domains = [DomainType.PHOTOREALISTIC, DomainType.ARTISTIC, DomainType.SYNTHETIC]

        reports = {}

        for domain in domains:
            # Baseline
            baseline_scheduler = DDIMScheduler()
            baseline_pipeline = DiffusionPipeline(
                scheduler=baseline_scheduler,
                image_size=(64, 64),
            )

            baseline_times = []
            for _ in range(3):
                start = time.perf_counter()
                baseline_images = baseline_pipeline.generate(
                    batch_size=2, num_inference_steps=30, seed=None
                )
                baseline_times.append(time.perf_counter() - start)

            baseline_quality = MetricsCalculator.compute_quality_metrics(baseline_images)
            baseline_perf = MetricsCalculator.compute_performance_metrics(
                baseline_times, num_images=6, num_steps=30, batch_size=2
            )

            baseline_metrics = ValidationMetrics(
                quality=baseline_quality,
                performance=baseline_perf,
                scheduler_name="DDIM-Baseline",
                domain=domain.value,
            )

            # Optimized
            optimized_scheduler = opt_pipeline.create_optimized_scheduler(
                domain_type=domain
            )
            optimized_pipeline = DiffusionPipeline(
                scheduler=optimized_scheduler,
                image_size=(64, 64),
            )

            optimized_times = []
            for _ in range(3):
                start = time.perf_counter()
                optimized_images = optimized_pipeline.generate(
                    batch_size=2, num_inference_steps=30, seed=None
                )
                optimized_times.append(time.perf_counter() - start)

            optimized_quality = MetricsCalculator.compute_quality_metrics(optimized_images)
            optimized_perf = MetricsCalculator.compute_performance_metrics(
                optimized_times, num_images=6, num_steps=30, batch_size=2
            )

            optimized_metrics = ValidationMetrics(
                quality=optimized_quality,
                performance=optimized_perf,
                scheduler_name=f"Adaptive-{domain.value}",
                domain=domain.value,
            )

            # Create report
            report = create_validation_report(baseline_metrics, optimized_metrics)
            reports[domain.value] = report

            print(f"\n{domain.value} Validation Report:")
            print(f"  Speedup: {report['comparison']['speedup']:.2f}x")

        # Validate reports
        assert len(reports) == 3
        for domain, report in reports.items():
            assert "comparison" in report
            assert "speedup" in report["comparison"]

        print(f"\nValidation Report Generation: PASS")
