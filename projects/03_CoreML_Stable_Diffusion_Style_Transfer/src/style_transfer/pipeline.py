"""Main style transfer pipeline implementation."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from diffusion import DiffusionConfig, DiffusionPipeline
from .config import StyleTransferConfig


class StyleTransferPipeline:
    """Main pipeline for performing style transfer operations."""

    def __init__(self, config: StyleTransferConfig):
        self.config = config
        self.diffusion_pipeline = None
        self._device = self._get_device()

    def _get_device(self) -> str:
        """Determine the best device for processing."""
        if self.config.device != "auto":
            return self.config.device

        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"

    def _initialize_diffusion_pipeline(self) -> None:
        """Initialize the diffusion pipeline if needed."""
        if self.diffusion_pipeline is None and self.config.method == "diffusion":
            # Create diffusion config from style transfer config
            diffusion_config = DiffusionConfig(
                num_inference_steps=self.config.num_inference_steps,
                guidance_scale=self.config.guidance_scale,
                scheduler=self.config.scheduler,
                device=self._device,
                use_attention_slicing=self.config.enable_attention_slicing,
            )

            self.diffusion_pipeline = DiffusionPipeline(diffusion_config)
            self.diffusion_pipeline.load_model()

    def _prepare_image(
        self,
        image: Image.Image | np.ndarray | str | Path,
        target_size: tuple[int, int] | None = None,
    ) -> Image.Image:
        """Prepare image for processing."""
        # Load image if path provided
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            if image.dtype == np.uint8:
                image = Image.fromarray(image)
            else:
                image = Image.fromarray((image * 255).astype(np.uint8))

        # Resize if needed
        if target_size is None:
            target_size = self.config.output_resolution

        if self.config.preserve_aspect_ratio:
            # Resize maintaining aspect ratio
            image.thumbnail(target_size, Image.Resampling.LANCZOS)
            # Create new image with target size and paste the resized image
            new_image = Image.new("RGB", target_size, (255, 255, 255))
            paste_x = (target_size[0] - image.width) // 2
            paste_y = (target_size[1] - image.height) // 2
            new_image.paste(image, (paste_x, paste_y))
            image = new_image
        else:
            image = image.resize(target_size, Image.Resampling.LANCZOS)

        return image

    def _postprocess_image(self, image: Image.Image) -> Image.Image:
        """Apply post-processing to the generated image."""
        # Apply upscaling if specified
        if self.config.upscale_factor != 1.0:
            new_width = int(image.width * self.config.upscale_factor)
            new_height = int(image.height * self.config.upscale_factor)
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        return image

    def transfer_style(
        self,
        content_image: Image.Image | np.ndarray | str | Path,
        style_image: Image.Image | np.ndarray | str | Path | None = None,
        style_description: str | None = None,
        seed: int | None = None,
    ) -> Image.Image:
        """Perform style transfer on content image."""
        if style_image is None and style_description is None:
            raise ValueError("Either style_image or style_description must be provided")

        # Prepare content image
        content_img = self._prepare_image(content_image)

        if self.config.method == "diffusion":
            return self._diffusion_style_transfer(
                content_img, style_image, style_description, seed
            )
        elif self.config.method == "neural_style":
            return self._neural_style_transfer(content_img, style_image)
        else:
            raise ValueError(f"Unknown method: {self.config.method}")

    def _diffusion_style_transfer(
        self,
        content_image: Image.Image,
        style_image: Image.Image | np.ndarray | str | Path | None,
        style_description: str | None,
        seed: int | None,
    ) -> Image.Image:
        """Perform style transfer using diffusion models."""
        self._initialize_diffusion_pipeline()

        if style_description is None and style_image is not None:
            # Analyze style image to create description
            style_img = self._prepare_image(style_image)
            style_description = self._analyze_style_image(style_img)
        elif style_description is None:
            style_description = "artistic style"

        # Use dual image style transfer if both content and style images provided
        if style_image is not None:
            style_img = self._prepare_image(style_image)
            result = self.diffusion_pipeline.dual_image_style_transfer(
                content_image=content_image,
                style_image=style_img,
                style_strength=self.config.style_strength,
                content_strength=self.config.content_strength,
                seed=seed,
            )
        else:
            # Use image-to-image with style description
            result = self.diffusion_pipeline.image_to_image_stylized(
                content_image=content_image,
                style_description=style_description,
                style_strength=self.config.style_strength,
                content_preservation=self.config.content_strength,
                seed=seed,
            )

        return self._postprocess_image(result)

    def _neural_style_transfer(
        self,
        content_image: Image.Image,
        style_image: Image.Image | np.ndarray | str | Path,
    ) -> Image.Image:
        """Perform style transfer using neural style transfer."""
        if style_image is None:
            raise ValueError("style_image is required for neural style transfer")

        # This would implement traditional neural style transfer
        # For now, we'll raise an error as this requires additional implementation
        raise NotImplementedError(
            "Neural style transfer method not yet implemented. Use 'diffusion' method instead."
        )

    def _analyze_style_image(self, style_image: Image.Image) -> str:
        """Analyze style image to create descriptive text."""
        # Convert to numpy for analysis
        img_array = np.array(style_image)

        # Simple color and texture analysis
        mean_color = np.mean(img_array, axis=(0, 1))
        color_dominance = np.argmax(mean_color)
        brightness = np.mean(img_array) / 255.0

        # Basic edge detection for texture analysis
        gray = np.dot(img_array[..., :3], [0.2989, 0.5870, 0.1140])
        edges = np.gradient(gray)
        edge_strength = np.mean(np.abs(edges[0]) + np.abs(edges[1]))

        # Create style description
        style_terms = []

        # Color characteristics
        if color_dominance == 0:  # Red dominant
            style_terms.append("warm colors")
        elif color_dominance == 1:  # Green dominant
            style_terms.append("natural tones")
        else:  # Blue dominant
            style_terms.append("cool colors")

        # Brightness characteristics
        if brightness > 0.7:
            style_terms.append("bright")
        elif brightness < 0.3:
            style_terms.append("dark")
        else:
            style_terms.append("balanced lighting")

        # Texture characteristics
        if edge_strength > 50:
            style_terms.append("detailed textures")
        else:
            style_terms.append("smooth style")

        style_terms.append("artistic painting")

        return ", ".join(style_terms)

    def batch_transfer(
        self,
        content_images: list[Image.Image | np.ndarray | str | Path],
        style_image: Image.Image | np.ndarray | str | Path | None = None,
        style_description: str | None = None,
        output_dir: Path | None = None,
        save_images: bool = True,
    ) -> list[Image.Image]:
        """Process multiple content images with the same style."""
        results = []

        if output_dir and save_images:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        for i, content_image in enumerate(content_images):
            print(f"Processing image {i + 1}/{len(content_images)}")

            try:
                result = self.transfer_style(
                    content_image=content_image,
                    style_image=style_image,
                    style_description=style_description,
                )

                results.append(result)

                # Save if requested
                if output_dir and save_images:
                    output_path = (
                        output_dir
                        / f"styled_{i:03d}.{self.config.output_format.lower()}"
                    )
                    result.save(
                        output_path,
                        format=self.config.output_format,
                        quality=self.config.quality,
                    )

            except Exception as e:
                print(f"Failed to process image {i}: {e}")
                results.append(None)

        return results

    def get_pipeline_info(self) -> dict[str, any]:
        """Get information about the pipeline."""
        info = {
            "method": self.config.method,
            "device": self._device,
            "config": self.config.to_dict(),
        }

        if self.diffusion_pipeline:
            info["diffusion_pipeline"] = self.diffusion_pipeline.get_pipeline_info()

        return info

    def cleanup(self) -> None:
        """Clean up resources."""
        if self.diffusion_pipeline:
            self.diffusion_pipeline.unload_model()

        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
