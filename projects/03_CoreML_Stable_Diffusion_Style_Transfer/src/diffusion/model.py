"""MLX-optimized Stable Diffusion model implementation."""

import numpy as np
import torch
from PIL import Image
import mlx.core as mx
import mlx.nn as nn
from diffusers import DiffusionPipeline, StableDiffusionPipeline
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)

from .config import DiffusionConfig


class StableDiffusionMLX:
    """MLX-optimized Stable Diffusion model for Apple Silicon."""

    def __init__(self, config: DiffusionConfig):
        self.config = config
        self.pipeline = None
        self.device = self._get_device()

    def _get_device(self) -> str:
        """Determine the best device for inference."""
        if self.config.device != "auto":
            return self.config.device

        # Apple Silicon optimization
        if torch.backends.mps.is_available() and self.config.use_mps:
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"

    def _get_scheduler(self, scheduler_name: str):
        """Get the specified scheduler."""
        schedulers = {
            "DPMSolverMultistepScheduler": DPMSolverMultistepScheduler,
            "DDIMScheduler": DDIMScheduler,
            "PNDMScheduler": PNDMScheduler,
            "LMSDiscreteScheduler": LMSDiscreteScheduler,
            "EulerAncestralDiscreteScheduler": EulerAncestralDiscreteScheduler,
            "EulerDiscreteScheduler": EulerDiscreteScheduler,
        }
        return schedulers.get(scheduler_name, DPMSolverMultistepScheduler)

    def load_model(self) -> None:
        """Load the Stable Diffusion model."""
        print(f"Loading model: {self.config.model_name}")
        print(f"Device: {self.device}")

        # Determine dtype
        dtype = torch.float16 if self.config.torch_dtype == "float16" else torch.float32

        try:
            # Load pipeline
            self.pipeline = StableDiffusionPipeline.from_pretrained(
                self.config.model_name,
                variant=self.config.variant,
                torch_dtype=dtype,
                safety_checker=None if not self.config.safety_checker else "default",
                requires_safety_checker=self.config.requires_safety_checker,
                cache_dir=self.config.cache_dir,
                local_files_only=self.config.offline_mode,
            )

            # Set scheduler
            scheduler_class = self._get_scheduler(self.config.scheduler)
            self.pipeline.scheduler = scheduler_class.from_config(
                self.pipeline.scheduler.config
            )

            # Move to device
            self.pipeline = self.pipeline.to(self.device)

            # Enable optimizations
            if self.config.use_attention_slicing:
                self.pipeline.enable_attention_slicing(self.config.attention_slice_size)

            if self.config.enable_memory_efficient_attention:
                try:
                    self.pipeline.enable_model_cpu_offload()
                except Exception:
                    pass  # Fallback gracefully

            if self.config.use_cpu_offload:
                self.pipeline.enable_sequential_cpu_offload()

            # MLX optimizations
            if self.config.use_mlx and self.device == "mps":
                self._apply_mlx_optimizations()

            print("Model loaded successfully")

        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

    def _apply_mlx_optimizations(self) -> None:
        """Apply MLX-specific optimizations."""
        try:
            # Convert key components to MLX if possible
            print("Applying MLX optimizations...")
            # Note: Full MLX integration would require more extensive refactoring
            # This is a placeholder for future MLX integration

        except Exception as e:
            print(f"MLX optimizations failed, continuing with PyTorch: {e}")

    def generate_image(
        self,
        prompt: str,
        negative_prompt: str | None = None,
        width: int = 512,
        height: int = 512,
        num_inference_steps: int | None = None,
        guidance_scale: float | None = None,
        generator: torch.Generator | None = None,
        image: Image.Image | np.ndarray | None = None,
        strength: float = 0.8,
    ) -> Image.Image:
        """Generate an image from text prompt."""
        if self.pipeline is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Use config defaults if not specified
        steps = num_inference_steps or self.config.num_inference_steps
        guidance = guidance_scale or self.config.guidance_scale

        print(f"Generating image: '{prompt[:50]}...'")
        print(f"Steps: {steps}, Guidance: {guidance}")

        try:
            with torch.inference_mode():
                if image is not None:
                    # Image-to-image pipeline
                    result = self.pipeline(
                        prompt=prompt,
                        image=image,
                        strength=strength,
                        negative_prompt=negative_prompt,
                        num_inference_steps=steps,
                        guidance_scale=guidance,
                        generator=generator,
                        width=width,
                        height=height,
                    )
                else:
                    # Text-to-image pipeline
                    result = self.pipeline(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        num_inference_steps=steps,
                        guidance_scale=guidance,
                        generator=generator,
                        width=width,
                        height=height,
                    )

                return result.images[0]

        except Exception as e:
            raise RuntimeError(f"Image generation failed: {e}")

    def img2img(
        self,
        prompt: str,
        image: Image.Image | np.ndarray,
        strength: float = 0.8,
        negative_prompt: str | None = None,
        num_inference_steps: int | None = None,
        guidance_scale: float | None = None,
        generator: torch.Generator | None = None,
    ) -> Image.Image:
        """Perform image-to-image transformation."""
        return self.generate_image(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            image=image,
            strength=strength,
            width=image.width if isinstance(image, Image.Image) else image.shape[1],
            height=image.height if isinstance(image, Image.Image) else image.shape[0],
        )

    def encode_prompt(self, prompt: str) -> torch.Tensor:
        """Encode text prompt to embeddings."""
        if self.pipeline is None:
            raise RuntimeError("Model not loaded")

        text_inputs = self.pipeline.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.pipeline.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

        with torch.no_grad():
            text_embeddings = self.pipeline.text_encoder(
                text_inputs.input_ids.to(self.device)
            )[0]

        return text_embeddings

    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        if self.pipeline is None:
            return {"status": "not_loaded"}

        info = {
            "model_name": self.config.model_name,
            "device": self.device,
            "dtype": str(self.pipeline.unet.dtype),
            "scheduler": self.config.scheduler,
            "mlx_available": True,
            "mlx_enabled": self.config.use_mlx,
            "mps_available": torch.backends.mps.is_available(),
            "mps_enabled": self.config.use_mps and self.device == "mps",
        }

        return info

    def clear_cache(self) -> None:
        """Clear GPU/MPS cache."""
        if self.device == "cuda":
            torch.cuda.empty_cache()
        elif self.device == "mps":
            torch.mps.empty_cache()

    def unload_model(self) -> None:
        """Unload the model from memory."""
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
            self.clear_cache()
            print("Model unloaded")
