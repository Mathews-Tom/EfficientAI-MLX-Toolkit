"""
Baseline Diffusion Pipeline

MLX-optimized diffusion pipeline supporting multiple schedulers and efficient
inference on Apple Silicon hardware.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from adaptive_diffusion.baseline.schedulers import BaseScheduler, get_scheduler


def tree_flatten(tree: dict) -> list[tuple[str, Any]]:
    """Flatten nested dict structure to list of (path, value) tuples."""
    items = []
    for key, value in tree.items():
        if isinstance(value, dict):
            sub_items = tree_flatten(value)
            items.extend([(f"{key}.{sub_key}", val) for sub_key, val in sub_items])
        else:
            items.append((key, value))
    return items


def tree_unflatten(items: list[tuple[str, Any]]) -> dict:
    """Unflatten list of (path, value) tuples to nested dict."""
    tree = {}
    for path, value in items:
        parts = path.split(".")
        current = tree
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value
    return tree


class SimpleUNet(nn.Module):
    """
    Simplified U-Net for diffusion models (MLX-compatible).

    This is a lightweight implementation for prototyping and testing.
    Uses NHWC format (MLX default) for convolutions.
    Production use should integrate with Stable Diffusion U-Net or similar.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        hidden_dims: list[int] | None = None,
        time_embed_dim: int = 128,
    ):
        """
        Initialize simple U-Net.

        Args:
            in_channels: Input image channels
            out_channels: Output image channels
            hidden_dims: Hidden dimensions for encoder/decoder
            time_embed_dim: Timestep embedding dimension
        """
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [32, 64]  # Simplified for testing

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_dims = hidden_dims
        self.time_embed_dim = time_embed_dim

        # Time embedding layers
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # Simplified model: just input -> hidden -> output
        # Skip complex U-Net architecture for baseline testing
        self.conv_in = nn.Conv2d(in_channels, hidden_dims[0], kernel_size=3, padding=1)
        self.conv_mid = nn.Conv2d(hidden_dims[0], hidden_dims[0], kernel_size=3, padding=1)
        self.conv_out = nn.Conv2d(hidden_dims[0], out_channels, kernel_size=3, padding=1)

        self.norm1 = nn.GroupNorm(4, hidden_dims[0])
        self.norm2 = nn.GroupNorm(4, hidden_dims[0])

        self.act = nn.SiLU()

    def __call__(self, x: mx.array, t: mx.array | int) -> mx.array:
        """
        Forward pass through U-Net.

        Args:
            x: Noisy input image [B, H, W, C] (NHWC format for MLX)
            t: Timestep [B] or scalar int

        Returns:
            Predicted noise [B, H, W, C]
        """
        # Convert timestep to array if scalar
        if isinstance(t, (int, float)):
            t = mx.array([float(t)] * x.shape[0])
        elif not isinstance(t, mx.array):
            t = mx.array(t)

        # Timestep embedding (not used in this simple version)
        t_emb = self.time_mlp(mx.expand_dims(t, axis=-1))

        # Simple forward pass
        h = self.conv_in(x)
        h = self.norm1(h)
        h = self.act(h)

        h = self.conv_mid(h)
        h = self.norm2(h)
        h = self.act(h)

        h = self.conv_out(h)

        return h


class DiffusionPipeline:
    """
    MLX-optimized diffusion pipeline for image generation.

    Supports multiple schedulers (DDPM, DDIM, DPM-Solver) and efficient
    inference on Apple Silicon.
    """

    def __init__(
        self,
        model: nn.Module | None = None,
        scheduler: BaseScheduler | str | None = None,
        image_size: tuple[int, int] = (256, 256),
        in_channels: int = 3,
    ):
        """
        Initialize diffusion pipeline.

        Args:
            model: U-Net model (if None, uses SimpleUNet)
            scheduler: Scheduler instance or name ('ddpm', 'ddim', 'dpm-solver')
            image_size: Generated image size (H, W)
            in_channels: Number of image channels
        """
        # Initialize model
        if model is None:
            self.model = SimpleUNet(in_channels=in_channels, out_channels=in_channels)
        else:
            self.model = model

        # Initialize scheduler
        if scheduler is None:
            self.scheduler = get_scheduler("ddim", num_inference_steps=50)
        elif isinstance(scheduler, str):
            self.scheduler = get_scheduler(scheduler)
        else:
            self.scheduler = scheduler

        self.image_size = image_size
        self.in_channels = in_channels

        # Device detection
        self.device = self._detect_device()

    def _detect_device(self) -> str:
        """Detect Apple Silicon for optimization."""
        import platform

        if platform.machine() == "arm64" and platform.system() == "Darwin":
            return "apple_silicon"
        return "cpu"

    def generate(
        self,
        batch_size: int = 1,
        num_inference_steps: int | None = None,
        seed: int | None = None,
        return_intermediates: bool = False,
    ) -> mx.array | tuple[mx.array, list[mx.array]]:
        """
        Generate images from random noise.

        Args:
            batch_size: Number of images to generate
            num_inference_steps: Number of denoising steps (overrides scheduler default)
            seed: Random seed for reproducibility
            return_intermediates: If True, return intermediate denoising steps

        Returns:
            Generated images [B, H, W, C] (NHWC format) or (images, intermediates)
        """
        if seed is not None:
            mx.random.seed(seed)

        # Update scheduler timesteps if specified
        if num_inference_steps is not None:
            self.scheduler.set_timesteps(num_inference_steps)

        # Initialize from random noise [B, H, W, C] (NHWC format)
        shape = (batch_size, *self.image_size, self.in_channels)
        x_t = mx.random.normal(shape)

        # Store intermediates if requested
        intermediates = [] if return_intermediates else None

        # Denoising loop
        for i, t in enumerate(self.scheduler.timesteps):
            # Predict noise
            t_batch = mx.array([t] * batch_size)
            model_output = self.model(x_t, t_batch)

            # Denoise step
            x_t = self.scheduler.step(model_output, i, x_t)

            if return_intermediates:
                intermediates.append(x_t)

        if return_intermediates:
            return x_t, intermediates
        return x_t

    def denoise_image(
        self,
        noisy_image: mx.array,
        timestep: int,
        num_inference_steps: int | None = None,
        return_intermediates: bool = False,
    ) -> mx.array | tuple[mx.array, list[mx.array]]:
        """
        Denoise an image starting from given timestep.

        Args:
            noisy_image: Noisy input image [B, H, W, C] (NHWC format)
            timestep: Starting timestep
            num_inference_steps: Number of denoising steps
            return_intermediates: If True, return intermediate steps

        Returns:
            Denoised image [B, H, W, C] or (image, intermediates)
        """
        if num_inference_steps is not None:
            self.scheduler.set_timesteps(num_inference_steps)

        # Find closest timestep in scheduler
        timestep_idx = 0
        for i, t in enumerate(self.scheduler.timesteps):
            if t <= timestep:
                timestep_idx = i
                break

        x_t = noisy_image
        intermediates = [] if return_intermediates else None

        # Denoise from timestep to 0
        for i in range(timestep_idx, len(self.scheduler.timesteps)):
            t = self.scheduler.timesteps[i]
            t_batch = mx.array([t] * x_t.shape[0])

            model_output = self.model(x_t, t_batch)
            x_t = self.scheduler.step(model_output, i, x_t)

            if return_intermediates:
                intermediates.append(x_t)

        if return_intermediates:
            return x_t, intermediates
        return x_t

    def add_noise(
        self, images: mx.array, timesteps: mx.array, noise: mx.array | None = None
    ) -> mx.array:
        """
        Add noise to images at given timesteps (forward process).

        Args:
            images: Clean images [B, H, W, C] (NHWC format)
            timesteps: Timesteps for each image [B]
            noise: Noise to add (if None, samples random noise)

        Returns:
            Noisy images [B, H, W, C]
        """
        if noise is None:
            noise = mx.random.normal(images.shape)

        return self.scheduler.add_noise(images, noise, timesteps)

    def save_model(self, path: str | Path):
        """Save model weights."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Convert parameters to dict, handling nested structures
        params = dict(tree_flatten(self.model.parameters()))
        mx.savez(str(path), **params)

    def load_model(self, path: str | Path):
        """Load model weights."""
        weights = mx.load(str(path))
        # Reconstruct tree structure and update model
        self.model.update(tree_unflatten(list(weights.items())))

    def get_scheduler_info(self) -> dict[str, Any]:
        """Get information about current scheduler configuration."""
        return {
            "scheduler_type": self.scheduler.__class__.__name__,
            "num_train_timesteps": self.scheduler.num_train_timesteps,
            "num_inference_steps": getattr(
                self.scheduler, "num_inference_steps", None
            ),
            "beta_schedule": self.scheduler.beta_schedule,
            "device": self.device,
        }
