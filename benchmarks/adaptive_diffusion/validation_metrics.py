"""
Validation Metrics for Adaptive Diffusion Optimization

Implements comprehensive validation metrics:
- Quality metrics (FID, CLIP score, perceptual distance)
- Performance metrics (speed, memory, throughput)
- Statistical validation (significance testing)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx
import numpy as np


@dataclass
class QualityMetrics:
    """Quality assessment metrics for generated images."""

    # Perceptual quality metrics
    fid_score: float | None = None  # Fréchet Inception Distance
    clip_score: float | None = None  # CLIP similarity score
    inception_score: float | None = None  # Inception Score

    # Image quality metrics
    ssim: float | None = None  # Structural Similarity Index
    psnr: float | None = None  # Peak Signal-to-Noise Ratio
    lpips: float | None = None  # Learned Perceptual Image Patch Similarity

    # Statistical metrics
    pixel_variance: float | None = None
    color_diversity: float | None = None
    edge_sharpness: float | None = None

    # Metadata
    num_samples: int = 0
    evaluation_time: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "fid_score": self.fid_score,
            "clip_score": self.clip_score,
            "inception_score": self.inception_score,
            "ssim": self.ssim,
            "psnr": self.psnr,
            "lpips": self.lpips,
            "pixel_variance": self.pixel_variance,
            "color_diversity": self.color_diversity,
            "edge_sharpness": self.edge_sharpness,
            "num_samples": self.num_samples,
            "evaluation_time": self.evaluation_time,
        }


@dataclass
class PerformanceMetrics:
    """Performance metrics for diffusion sampling."""

    # Speed metrics
    total_time: float = 0.0
    avg_time_per_image: float = 0.0
    avg_time_per_step: float = 0.0
    images_per_sec: float = 0.0
    steps_per_sec: float = 0.0

    # Memory metrics
    peak_memory_mb: float | None = None
    avg_memory_mb: float | None = None
    memory_efficiency: float | None = None  # MB per image

    # Throughput metrics
    total_images: int = 0
    total_steps: int = 0
    batch_size: int = 1

    # Statistical metrics
    std_time: float | None = None
    min_time: float | None = None
    max_time: float | None = None
    time_variance: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_time": self.total_time,
            "avg_time_per_image": self.avg_time_per_image,
            "avg_time_per_step": self.avg_time_per_step,
            "images_per_sec": self.images_per_sec,
            "steps_per_sec": self.steps_per_sec,
            "peak_memory_mb": self.peak_memory_mb,
            "avg_memory_mb": self.avg_memory_mb,
            "memory_efficiency": self.memory_efficiency,
            "total_images": self.total_images,
            "total_steps": self.total_steps,
            "batch_size": self.batch_size,
            "std_time": self.std_time,
            "min_time": self.min_time,
            "max_time": self.max_time,
            "time_variance": self.time_variance,
        }


@dataclass
class ValidationMetrics:
    """
    Comprehensive validation metrics combining quality and performance.

    Tracks:
    - Quality metrics (image quality, perceptual metrics)
    - Performance metrics (speed, memory, throughput)
    - Statistical significance
    """

    quality: QualityMetrics = field(default_factory=QualityMetrics)
    performance: PerformanceMetrics = field(default_factory=PerformanceMetrics)

    # Overall metrics
    overall_score: float | None = None
    quality_performance_ratio: float | None = None

    # Metadata
    scheduler_name: str = "unknown"
    model_name: str = "unknown"
    domain: str | None = None
    timestamp: str | None = None

    def compute_overall_score(self, quality_weight: float = 0.6) -> float:
        """
        Compute overall score combining quality and performance.

        Args:
            quality_weight: Weight for quality metrics (0-1)

        Returns:
            Overall score (0-1 scale)
        """
        # Normalize quality score (lower FID is better, invert to 0-1 scale)
        quality_score = 0.5  # Default if no metrics
        if self.quality.fid_score is not None:
            # FID typically ranges 0-300+, invert and normalize
            quality_score = 1.0 / (1.0 + self.quality.fid_score / 100.0)
        elif self.quality.clip_score is not None:
            quality_score = self.quality.clip_score  # Already 0-1

        # Normalize performance score (higher throughput is better)
        performance_score = 0.5  # Default
        if self.performance.images_per_sec > 0:
            # Normalize to reasonable range (assume 10 images/sec is excellent)
            performance_score = min(1.0, self.performance.images_per_sec / 10.0)

        # Combine scores
        overall = quality_weight * quality_score + (1 - quality_weight) * performance_score
        self.overall_score = overall
        self.quality_performance_ratio = (
            quality_score / performance_score if performance_score > 0 else 0
        )

        return overall

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "quality": self.quality.to_dict(),
            "performance": self.performance.to_dict(),
            "overall_score": self.overall_score,
            "quality_performance_ratio": self.quality_performance_ratio,
            "scheduler_name": self.scheduler_name,
            "model_name": self.model_name,
            "domain": self.domain,
            "timestamp": self.timestamp,
        }


class MetricsCalculator:
    """
    Calculator for validation metrics.

    Provides methods to compute quality and performance metrics from
    generated images and timing information.
    """

    @staticmethod
    def compute_quality_metrics(
        generated_images: mx.array, reference_images: mx.array | None = None
    ) -> QualityMetrics:
        """
        Compute quality metrics for generated images.

        Args:
            generated_images: Generated image samples (B, H, W, C)
            reference_images: Optional reference images for comparison

        Returns:
            Quality metrics
        """
        start_time = time.perf_counter()

        metrics = QualityMetrics(num_samples=generated_images.shape[0])

        # Pixel-level statistics
        metrics.pixel_variance = float(mx.var(generated_images))

        # Color diversity (std across channels)
        if generated_images.shape[-1] == 3:  # RGB
            channel_means = mx.mean(generated_images, axis=(0, 1, 2))
            metrics.color_diversity = float(mx.std(channel_means))

        # Edge sharpness (gradient magnitude)
        edge_metrics = MetricsCalculator._compute_edge_sharpness(generated_images)
        metrics.edge_sharpness = edge_metrics

        # If reference images provided, compute comparison metrics
        if reference_images is not None:
            metrics.ssim = MetricsCalculator._compute_ssim(
                generated_images, reference_images
            )
            metrics.psnr = MetricsCalculator._compute_psnr(
                generated_images, reference_images
            )

        # FID and CLIP scores require pretrained models (compute only if available)
        # For now, use simplified proxy metrics
        metrics.fid_score = MetricsCalculator._estimate_fid_proxy(generated_images)

        metrics.evaluation_time = time.perf_counter() - start_time

        return metrics

    @staticmethod
    def compute_performance_metrics(
        iteration_times: list[float],
        num_images: int,
        num_steps: int,
        batch_size: int = 1,
        memory_usage: list[float] | None = None,
    ) -> PerformanceMetrics:
        """
        Compute performance metrics from timing data.

        Args:
            iteration_times: List of iteration times
            num_images: Total number of images generated
            num_steps: Number of diffusion steps
            batch_size: Batch size used
            memory_usage: Optional memory usage data (MB)

        Returns:
            Performance metrics
        """
        total_time = sum(iteration_times)
        avg_time = np.mean(iteration_times)
        std_time = np.std(iteration_times)

        metrics = PerformanceMetrics(
            total_time=total_time,
            avg_time_per_image=avg_time / batch_size,
            avg_time_per_step=avg_time / num_steps,
            images_per_sec=num_images / total_time,
            steps_per_sec=(num_images * num_steps) / total_time,
            total_images=num_images,
            total_steps=num_images * num_steps,
            batch_size=batch_size,
            std_time=std_time,
            min_time=np.min(iteration_times),
            max_time=np.max(iteration_times),
            time_variance=float(np.var(iteration_times)),
        )

        # Memory metrics
        if memory_usage:
            metrics.peak_memory_mb = np.max(memory_usage)
            metrics.avg_memory_mb = np.mean(memory_usage)
            metrics.memory_efficiency = metrics.avg_memory_mb / batch_size

        return metrics

    @staticmethod
    def _compute_edge_sharpness(images: mx.array) -> float:
        """
        Compute edge sharpness using gradient magnitude.

        Args:
            images: Image batch (B, H, W, C)

        Returns:
            Average edge sharpness
        """
        # Compute gradients
        dx = images[:, 1:, :, :] - images[:, :-1, :, :]
        dy = images[:, :, 1:, :] - images[:, :, :-1, :]

        # Align shapes
        min_h = min(dx.shape[1], dy.shape[1])
        min_w = min(dx.shape[2], dy.shape[2])
        dx_aligned = dx[:, :min_h, :min_w, :]
        dy_aligned = dy[:, :min_h, :min_w, :]

        # Gradient magnitude
        gradient_mag = mx.sqrt(dx_aligned**2 + dy_aligned**2)
        sharpness = float(mx.mean(gradient_mag))

        return sharpness

    @staticmethod
    def _compute_ssim(
        generated: mx.array, reference: mx.array, window_size: int = 11
    ) -> float:
        """
        Compute simplified SSIM (Structural Similarity Index).

        Args:
            generated: Generated images
            reference: Reference images
            window_size: Window size for local statistics

        Returns:
            SSIM score (0-1)
        """
        # Simplified SSIM using correlation
        # Full SSIM would require window-based statistics

        # Normalize images
        gen_norm = (generated - mx.mean(generated)) / (mx.std(generated) + 1e-8)
        ref_norm = (reference - mx.mean(reference)) / (mx.std(reference) + 1e-8)

        # Compute correlation
        correlation = float(mx.mean(gen_norm * ref_norm))

        # Map correlation to SSIM-like score (0-1)
        ssim_score = (correlation + 1) / 2  # Map from [-1, 1] to [0, 1]

        return ssim_score

    @staticmethod
    def _compute_psnr(generated: mx.array, reference: mx.array) -> float:
        """
        Compute Peak Signal-to-Noise Ratio.

        Args:
            generated: Generated images
            reference: Reference images

        Returns:
            PSNR in dB
        """
        # Compute MSE
        mse = float(mx.mean((generated - reference) ** 2))

        # Avoid division by zero
        if mse < 1e-10:
            return 100.0  # Perfect match

        # Assuming images in range [0, 1]
        max_pixel = 1.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))

        return float(psnr)

    @staticmethod
    def _estimate_fid_proxy(images: mx.array) -> float:
        """
        Estimate FID-like score using statistical properties.

        Note: This is a simplified proxy. Real FID requires Inception network.

        Args:
            images: Generated images

        Returns:
            FID proxy score
        """
        # Compute statistical moments
        mean = float(mx.mean(images))
        std = float(mx.std(images))
        skewness = MetricsCalculator._compute_skewness(images)
        kurtosis = MetricsCalculator._compute_kurtosis(images)

        # Combine into FID-like score (lower is better)
        # This is a heuristic proxy based on distribution properties
        fid_proxy = abs(mean - 0.5) * 100 + abs(std - 0.25) * 50
        fid_proxy += abs(skewness) * 10 + abs(kurtosis - 3) * 5

        return float(fid_proxy)

    @staticmethod
    def _compute_skewness(data: mx.array) -> float:
        """Compute skewness of data."""
        mean = mx.mean(data)
        std = mx.std(data)
        centered = data - mean
        skewness = mx.mean((centered / (std + 1e-8)) ** 3)
        return float(skewness)

    @staticmethod
    def _compute_kurtosis(data: mx.array) -> float:
        """Compute kurtosis of data."""
        mean = mx.mean(data)
        std = mx.std(data)
        centered = data - mean
        kurtosis = mx.mean((centered / (std + 1e-8)) ** 4)
        return float(kurtosis)


def compute_statistical_significance(
    baseline_times: list[float], optimized_times: list[float], confidence: float = 0.95
) -> dict[str, Any]:
    """
    Compute statistical significance of performance difference.

    Uses t-test to determine if performance improvement is significant.

    Args:
        baseline_times: Timing data for baseline
        optimized_times: Timing data for optimized version
        confidence: Confidence level (default 0.95 for 95%)

    Returns:
        Dictionary with significance test results
    """
    from scipy import stats

    # Perform two-sample t-test
    t_statistic, p_value = stats.ttest_ind(baseline_times, optimized_times)

    # Compute effect size (Cohen's d)
    mean_baseline = np.mean(baseline_times)
    mean_optimized = np.mean(optimized_times)
    std_pooled = np.sqrt(
        (np.var(baseline_times) + np.var(optimized_times)) / 2
    )
    cohens_d = (mean_baseline - mean_optimized) / (std_pooled + 1e-8)

    # Speedup
    speedup = mean_baseline / mean_optimized

    # Determine significance
    alpha = 1 - confidence
    is_significant = p_value < alpha

    return {
        "t_statistic": float(t_statistic),
        "p_value": float(p_value),
        "cohens_d": float(cohens_d),
        "speedup": float(speedup),
        "is_significant": bool(is_significant),
        "confidence": confidence,
        "baseline_mean": float(mean_baseline),
        "optimized_mean": float(mean_optimized),
        "baseline_std": float(np.std(baseline_times)),
        "optimized_std": float(np.std(optimized_times)),
    }


def create_validation_report(
    baseline_metrics: ValidationMetrics,
    optimized_metrics: ValidationMetrics,
    statistical_test: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Create comprehensive validation report comparing baseline and optimized.

    Args:
        baseline_metrics: Metrics for baseline
        optimized_metrics: Metrics for optimized version
        statistical_test: Optional statistical significance test results

    Returns:
        Validation report dictionary
    """
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "baseline": baseline_metrics.to_dict(),
        "optimized": optimized_metrics.to_dict(),
        "comparison": {
            "speedup": (
                baseline_metrics.performance.total_time
                / optimized_metrics.performance.total_time
            ),
            "memory_reduction": (
                (
                    baseline_metrics.performance.peak_memory_mb
                    - optimized_metrics.performance.peak_memory_mb
                )
                / baseline_metrics.performance.peak_memory_mb
                * 100
                if baseline_metrics.performance.peak_memory_mb
                and optimized_metrics.performance.peak_memory_mb
                else None
            ),
            "quality_improvement": (
                (
                    optimized_metrics.quality.fid_score
                    - baseline_metrics.quality.fid_score
                )
                / baseline_metrics.quality.fid_score
                * 100
                if baseline_metrics.quality.fid_score
                and optimized_metrics.quality.fid_score
                else None
            ),
        },
        "statistical_significance": statistical_test,
    }

    return report
