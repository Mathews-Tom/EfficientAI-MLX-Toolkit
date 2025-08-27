"""Diffusion pipeline implementation for style transfer."""

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from .config import DiffusionConfig
from .model import StableDiffusionMLX


class DiffusionPipeline:
    """Enhanced diffusion pipeline for style transfer applications."""

    def __init__(self, config: DiffusionConfig):
        self.config = config
        self.model = StableDiffusionMLX(config)
        self.loaded = False

    def load_model(self) -> None:
        """Load the diffusion model."""
        self.model.load_model()
        self.loaded = True

    def unload_model(self) -> None:
        """Unload the diffusion model."""
        self.model.unload_model()
        self.loaded = False

    def _prepare_image(
        self,
        image: Image.Image | np.ndarray | str | Path,
        target_size: tuple | None = None,
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

        # Resize if target size specified
        if target_size:
            image = image.resize(target_size, Image.Resampling.LANCZOS)

        return image

    def _create_style_prompt(
        self,
        base_prompt: str,
        style_description: str,
        strength_modifiers: dict[str, float] | None = None,
    ) -> str:
        """Create an enhanced prompt for style transfer."""
        # Base style transfer prompt template
        style_prompt = f"{base_prompt}, in the style of {style_description}"

        # Add quality modifiers
        quality_terms = [
            "highly detailed",
            "professional quality",
            "artistic masterpiece",
            "vibrant colors",
            "sharp focus",
        ]

        if strength_modifiers:
            # Adjust prompt based on style strength
            detail_strength = strength_modifiers.get("detail", 1.0)
            if detail_strength > 0.8:
                style_prompt += ", " + ", ".join(quality_terms[:3])
            elif detail_strength > 0.5:
                style_prompt += ", " + ", ".join(quality_terms[:2])

        return style_prompt

    def _create_negative_prompt(
        self, base_negative: str | None = None, avoid_terms: list | None = None
    ) -> str:
        """Create negative prompt to avoid unwanted artifacts."""
        default_negative = [
            "blurry",
            "low quality",
            "distorted",
            "artifacts",
            "noise",
            "oversaturated",
            "watermark",
            "signature",
        ]

        negative_terms = default_negative.copy()
        if avoid_terms:
            negative_terms.extend(avoid_terms)
        if base_negative:
            negative_terms.insert(0, base_negative)

        return ", ".join(negative_terms)

    def text_to_image_stylized(
        self,
        prompt: str,
        style_description: str,
        width: int = 512,
        height: int = 512,
        num_inference_steps: int | None = None,
        guidance_scale: float | None = None,
        style_strength: float = 0.8,
        seed: int | None = None,
    ) -> Image.Image:
        """Generate stylized image from text prompt."""
        if not self.loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Create enhanced prompt
        enhanced_prompt = self._create_style_prompt(
            prompt, style_description, {"detail": style_strength}
        )

        # Create negative prompt
        negative_prompt = self._create_negative_prompt()

        # Set up generator for reproducibility
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.model.device).manual_seed(seed)

        return self.model.generate_image(
            prompt=enhanced_prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )

    def image_to_image_stylized(
        self,
        content_image: Image.Image | np.ndarray | str | Path,
        style_description: str,
        style_strength: float = 0.8,
        content_preservation: float = 0.7,
        num_inference_steps: int | None = None,
        guidance_scale: float | None = None,
        seed: int | None = None,
    ) -> Image.Image:
        """Apply style to existing image using img2img."""
        if not self.loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Prepare content image
        content_img = self._prepare_image(content_image)

        # Create style prompt based on image content
        content_prompt = "artistic painting, masterpiece artwork"
        enhanced_prompt = self._create_style_prompt(
            content_prompt, style_description, {"detail": style_strength}
        )

        negative_prompt = self._create_negative_prompt()

        # Calculate strength (higher = more style, lower = more content preservation)
        img2img_strength = 1.0 - content_preservation

        # Set up generator for reproducibility
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.model.device).manual_seed(seed)

        return self.model.img2img(
            prompt=enhanced_prompt,
            image=content_img,
            strength=img2img_strength,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )

    def dual_image_style_transfer(
        self,
        content_image: Image.Image | np.ndarray | str | Path,
        style_image: Image.Image | np.ndarray | str | Path,
        style_strength: float = 0.8,
        content_strength: float = 0.6,
        prompt_override: str | None = None,
        num_inference_steps: int | None = None,
        guidance_scale: float | None = None,
        seed: int | None = None,
    ) -> Image.Image:
        """Perform style transfer using both content and style images."""
        if not self.loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Prepare images
        content_img = self._prepare_image(content_image)
        style_img = self._prepare_image(style_image)

        # Analyze style image (simplified - could use more sophisticated analysis)
        style_description = self._analyze_style_image(style_img)

        # Use provided prompt or create one
        if prompt_override:
            base_prompt = prompt_override
        else:
            base_prompt = "detailed artistic composition"

        # Create enhanced prompt
        enhanced_prompt = self._create_style_prompt(
            base_prompt, style_description, {"detail": style_strength}
        )

        negative_prompt = self._create_negative_prompt()

        # Calculate img2img strength based on content preservation
        img2img_strength = 1.0 - content_strength

        # Set up generator
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.model.device).manual_seed(seed)

        return self.model.img2img(
            prompt=enhanced_prompt,
            image=content_img,
            strength=img2img_strength,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )

    def _analyze_style_image(self, style_image: Image.Image) -> str:
        """Analyze style image to create descriptive terms."""
        # Convert to numpy for analysis
        img_array = np.array(style_image)

        # Simple color analysis
        mean_color = np.mean(img_array, axis=(0, 1))
        color_dominance = np.argmax(mean_color)

        # Basic texture analysis using edge detection
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.mean(edges) / 255.0

        # Brightness analysis
        brightness = np.mean(img_array) / 255.0

        # Create style description based on analysis
        style_terms = []

        # Color characteristics
        if color_dominance == 0:  # Red dominant
            style_terms.append("warm tones")
        elif color_dominance == 1:  # Green dominant
            style_terms.append("natural colors")
        else:  # Blue dominant
            style_terms.append("cool tones")

        # Brightness characteristics
        if brightness > 0.7:
            style_terms.append("bright lighting")
        elif brightness < 0.3:
            style_terms.append("dramatic shadows")
        else:
            style_terms.append("balanced lighting")

        # Texture characteristics
        if edge_density > 0.3:
            style_terms.append("detailed textures")
        else:
            style_terms.append("smooth gradients")

        return ", ".join(style_terms)

    def batch_process(
        self, images: list, style_description: str, output_dir: Path, **kwargs
    ) -> list:
        """Process multiple images with the same style."""
        if not self.loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = []

        for i, image_path in enumerate(images):
            print(f"Processing image {i + 1}/{len(images)}: {image_path}")

            try:
                result_image = self.image_to_image_stylized(
                    content_image=image_path,
                    style_description=style_description,
                    **kwargs,
                )

                # Save result
                output_path = output_dir / f"stylized_{i:03d}.png"
                result_image.save(output_path)
                results.append(output_path)

            except Exception as e:
                print(f"Failed to process {image_path}: {e}")
                results.append(None)

        return results

    def get_pipeline_info(self) -> dict[str, any]:
        """Get information about the pipeline."""
        info = {"loaded": self.loaded, "config": self.config.to_dict()}

        if self.loaded:
            info.update(self.model.get_model_info())

        return info
