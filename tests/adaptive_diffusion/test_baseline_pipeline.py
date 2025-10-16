"""
Tests for baseline diffusion pipeline and schedulers.

Tests DDPM, DDIM, and DPM-Solver schedulers with MLX-optimized pipeline.
"""

import sys
from pathlib import Path

import mlx.core as mx
import pytest

# Add project to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "projects" / "04_Adaptive_Diffusion_Optimizer" / "src"))

from adaptive_diffusion.baseline import (
    DDIMScheduler,
    DDPMScheduler,
    DiffusionPipeline,
    DPMSolverScheduler,
)
from adaptive_diffusion.baseline.schedulers import get_scheduler


class TestSchedulers:
    """Test suite for diffusion schedulers."""

    def test_ddpm_scheduler_initialization(self):
        """Test DDPM scheduler initialization."""
        scheduler = DDPMScheduler(num_train_timesteps=1000)

        assert scheduler.num_train_timesteps == 1000
        assert len(scheduler.betas) == 1000
        assert len(scheduler.alphas) == 1000
        assert len(scheduler.alphas_cumprod) == 1000

    def test_ddim_scheduler_initialization(self):
        """Test DDIM scheduler initialization."""
        scheduler = DDIMScheduler(num_train_timesteps=1000, num_inference_steps=50)

        assert scheduler.num_train_timesteps == 1000
        assert scheduler.num_inference_steps == 50
        assert len(scheduler.timesteps) == 50

    def test_dpm_solver_scheduler_initialization(self):
        """Test DPM-Solver scheduler initialization."""
        scheduler = DPMSolverScheduler(
            num_train_timesteps=1000, num_inference_steps=20, solver_order=2
        )

        assert scheduler.num_train_timesteps == 1000
        assert scheduler.num_inference_steps == 20
        assert scheduler.solver_order == 2

    def test_scheduler_factory(self):
        """Test scheduler factory function."""
        ddpm = get_scheduler("ddpm", num_train_timesteps=1000)
        assert isinstance(ddpm, DDPMScheduler)

        ddim = get_scheduler("ddim", num_train_timesteps=1000)
        assert isinstance(ddim, DDIMScheduler)

        dpm = get_scheduler("dpm-solver", num_train_timesteps=1000)
        assert isinstance(dpm, DPMSolverScheduler)

        with pytest.raises(ValueError):
            get_scheduler("invalid_scheduler")

    def test_add_noise_ddpm(self):
        """Test DDPM noise addition."""
        scheduler = DDPMScheduler(num_train_timesteps=1000)

        # Create sample image [B, H, W, C] (NHWC format)
        batch_size = 2
        image = mx.random.normal((batch_size, 32, 32, 3))
        noise = mx.random.normal((batch_size, 32, 32, 3))
        timesteps = mx.array([100, 200])

        # Add noise
        noisy_image = scheduler.add_noise(image, noise, timesteps)

        # Check shape
        assert noisy_image.shape == image.shape

        # Check that noise increases with timestep
        # (higher timestep = more noise)
        # For t=0, should be close to original
        timesteps_zero = mx.array([0, 0])
        noisy_zero = scheduler.add_noise(image, noise, timesteps_zero)
        diff_zero = mx.mean(mx.abs(noisy_zero - image))

        # For t=999, should be mostly noise
        timesteps_max = mx.array([999, 999])
        noisy_max = scheduler.add_noise(image, noise, timesteps_max)
        diff_max = mx.mean(mx.abs(noisy_max - image))

        assert diff_max > diff_zero

    def test_ddpm_step(self):
        """Test DDPM denoising step."""
        scheduler = DDPMScheduler(num_train_timesteps=100)

        # Create sample [B, H, W, C]
        sample = mx.random.normal((2, 32, 32, 3))
        model_output = mx.random.normal((2, 32, 32, 3))
        timestep = 50

        # Perform step
        prev_sample = scheduler.step(model_output, timestep, sample)

        # Check shape
        assert prev_sample.shape == sample.shape

    def test_ddim_step(self):
        """Test DDIM denoising step."""
        scheduler = DDIMScheduler(num_train_timesteps=1000, num_inference_steps=50)

        sample = mx.random.normal((2, 32, 32, 3))
        model_output = mx.random.normal((2, 32, 32, 3))
        timestep_idx = 0

        prev_sample = scheduler.step(model_output, timestep_idx, sample)

        assert prev_sample.shape == sample.shape

    def test_dpm_solver_step(self):
        """Test DPM-Solver denoising step."""
        scheduler = DPMSolverScheduler(
            num_train_timesteps=1000, num_inference_steps=20
        )

        sample = mx.random.normal((2, 32, 32, 3))
        model_output = mx.random.normal((2, 32, 32, 3))
        timestep_idx = 0

        prev_sample = scheduler.step(model_output, timestep_idx, sample)

        assert prev_sample.shape == sample.shape

    def test_beta_schedules(self):
        """Test different beta schedules."""
        schedules = ["linear", "scaled_linear", "squaredcos_cap_v2"]

        for schedule in schedules:
            scheduler = DDPMScheduler(beta_schedule=schedule)
            assert len(scheduler.betas) == 1000
            assert mx.all(scheduler.betas > 0)
            assert mx.all(scheduler.betas < 1)


class TestDiffusionPipeline:
    """Test suite for diffusion pipeline."""

    def test_pipeline_initialization(self):
        """Test pipeline initialization with default parameters."""
        pipeline = DiffusionPipeline()

        assert pipeline.model is not None
        assert pipeline.scheduler is not None
        assert pipeline.image_size == (256, 256)
        assert pipeline.in_channels == 3

    def test_pipeline_with_custom_scheduler(self):
        """Test pipeline with custom scheduler."""
        scheduler = DDIMScheduler(num_inference_steps=20)
        pipeline = DiffusionPipeline(scheduler=scheduler)

        assert isinstance(pipeline.scheduler, DDIMScheduler)
        assert pipeline.scheduler.num_inference_steps == 20

    def test_pipeline_with_scheduler_name(self):
        """Test pipeline initialization with scheduler name."""
        pipeline = DiffusionPipeline(scheduler="dpm-solver")

        assert isinstance(pipeline.scheduler, DPMSolverScheduler)

    @pytest.mark.slow
    def test_generate_images(self):
        """Test image generation from noise."""
        pipeline = DiffusionPipeline(
            scheduler="ddim", image_size=(64, 64), in_channels=3
        )
        pipeline.scheduler.set_timesteps(10)  # Fast test

        # Generate images
        images = pipeline.generate(batch_size=2, seed=42)

        # Check output shape [B, H, W, C]
        assert images.shape == (2, 64, 64, 3)

    @pytest.mark.slow
    def test_generate_with_intermediates(self):
        """Test generation with intermediate steps."""
        pipeline = DiffusionPipeline(scheduler="ddim", image_size=(32, 32))
        pipeline.scheduler.set_timesteps(5)

        images, intermediates = pipeline.generate(
            batch_size=1, return_intermediates=True, seed=42
        )

        assert images.shape == (1, 32, 32, 3)
        assert len(intermediates) == 5
        assert all(x.shape == images.shape for x in intermediates)

    def test_add_noise(self):
        """Test noise addition to clean images."""
        pipeline = DiffusionPipeline()

        images = mx.random.normal((2, 256, 256, 3))
        timesteps = mx.array([100, 200])

        noisy_images = pipeline.add_noise(images, timesteps)

        assert noisy_images.shape == images.shape

    def test_denoise_image(self):
        """Test denoising from intermediate timestep."""
        pipeline = DiffusionPipeline(scheduler="ddim", image_size=(32, 32))
        pipeline.scheduler.set_timesteps(10)

        # Create noisy image [B, H, W, C]
        noisy_image = mx.random.normal((1, 32, 32, 3))

        # Denoise from timestep 500
        denoised = pipeline.denoise_image(noisy_image, timestep=500)

        assert denoised.shape == noisy_image.shape

    def test_scheduler_info(self):
        """Test scheduler information retrieval."""
        pipeline = DiffusionPipeline(scheduler="ddpm")

        info = pipeline.get_scheduler_info()

        assert "scheduler_type" in info
        assert "num_train_timesteps" in info
        assert "beta_schedule" in info
        assert info["scheduler_type"] == "DDPMScheduler"

    @pytest.mark.skip(reason="MLX GroupNorm serialization issue - known limitation")
    def test_save_and_load_model(self, tmp_path):
        """Test model saving and loading."""
        pipeline = DiffusionPipeline(image_size=(32, 32))

        # Save model
        model_path = tmp_path / "model.npz"
        pipeline.save_model(model_path)

        assert model_path.exists()

        # Create new pipeline and load weights
        new_pipeline = DiffusionPipeline(image_size=(32, 32))
        new_pipeline.load_model(model_path)

        # Models should produce similar outputs
        # (Note: exact comparison may fail due to random initialization)
        assert new_pipeline.model is not None


class TestSchedulerComparison:
    """Compare different schedulers."""

    @pytest.mark.slow
    def test_scheduler_speed_comparison(self):
        """Compare inference speed across schedulers."""
        image_size = (32, 32)
        batch_size = 1

        schedulers = [
            ("ddpm", 50),
            ("ddim", 50),
            ("dpm-solver", 20),
        ]

        results = {}

        for scheduler_name, steps in schedulers:
            pipeline = DiffusionPipeline(
                scheduler=scheduler_name, image_size=image_size
            )
            pipeline.scheduler.set_timesteps(steps)

            # Generate (measure would require actual timing)
            images = pipeline.generate(batch_size=batch_size, seed=42)

            results[scheduler_name] = {
                "steps": steps,
                "shape": images.shape,
            }

        # Verify all generated images [B, H, W, C]
        for scheduler_name, result in results.items():
            assert result["shape"] == (batch_size, *image_size, 3)

    def test_scheduler_consistency(self):
        """Test that schedulers produce consistent output with same seed."""
        scheduler = DDIMScheduler(num_inference_steps=10)
        pipeline = DiffusionPipeline(scheduler=scheduler, image_size=(32, 32))

        # Generate twice with same seed
        images1 = pipeline.generate(batch_size=1, seed=42)
        images2 = pipeline.generate(batch_size=1, seed=42)

        # Should be identical
        assert mx.allclose(images1, images2, atol=1e-5)


@pytest.mark.benchmark
class TestPerformance:
    """Performance benchmarks for baseline pipeline."""

    @pytest.mark.slow
    def test_ddim_50_steps(self):
        """Benchmark DDIM with 50 steps."""
        pipeline = DiffusionPipeline(scheduler="ddim", image_size=(256, 256))
        pipeline.scheduler.set_timesteps(50)

        images = pipeline.generate(batch_size=1, seed=42)
        assert images.shape == (1, 256, 256, 3)

    @pytest.mark.slow
    def test_dpm_solver_20_steps(self):
        """Benchmark DPM-Solver with 20 steps."""
        pipeline = DiffusionPipeline(scheduler="dpm-solver", image_size=(256, 256))
        pipeline.scheduler.set_timesteps(20)

        images = pipeline.generate(batch_size=1, seed=42)
        assert images.shape == (1, 256, 256, 3)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
