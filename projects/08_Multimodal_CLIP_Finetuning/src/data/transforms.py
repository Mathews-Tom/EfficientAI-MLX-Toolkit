#!/usr/bin/env python3
"""
Image and text transforms for CLIP fine-tuning.

Provides preprocessing and augmentation for images and text.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
import torchvision.transforms as T
from PIL import Image

if TYPE_CHECKING:
    from transformers import CLIPProcessor

    from config import CLIPFinetuningConfig

logger = logging.getLogger(__name__)


class ImageTransform:
    """Image preprocessing and augmentation for CLIP.

    Handles resizing, normalization, and optional augmentation.
    """

    def __init__(self, config: CLIPFinetuningConfig, augment: bool = False) -> None:
        """Initialize image transform.

        Args:
            config: CLIP fine-tuning configuration
            augment: Whether to apply data augmentation
        """
        self.config = config
        self.augment = augment

        # Build transform pipeline
        transforms: list = []

        # Resize to target resolution
        transforms.append(T.Resize((config.image_resolution, config.image_resolution)))

        if augment:
            # Data augmentation
            transforms.extend([
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(degrees=10),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            ])

        # Convert to tensor
        transforms.append(T.ToTensor())

        # Normalize (CLIP standard normalization)
        transforms.append(
            T.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            )
        )

        self.transform = T.Compose(transforms)

        logger.debug(f"Initialized ImageTransform (augment={augment})")

    def __call__(self, image: Image.Image) -> torch.Tensor:
        """Apply transform to image.

        Args:
            image: PIL Image

        Returns:
            Transformed image tensor [C, H, W]
        """
        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")

        return self.transform(image)


class TextTransform:
    """Text preprocessing and tokenization for CLIP.

    Handles tokenization, truncation, and padding using CLIP processor.
    """

    def __init__(self, processor: CLIPProcessor, config: CLIPFinetuningConfig) -> None:
        """Initialize text transform.

        Args:
            processor: CLIP processor for tokenization
            config: CLIP fine-tuning configuration
        """
        self.processor = processor
        self.config = config

        logger.debug("Initialized TextTransform")

    def __call__(self, text: str) -> dict[str, torch.Tensor]:
        """Apply transform to text.

        Args:
            text: Text string to tokenize

        Returns:
            Dictionary containing:
                - input_ids: Token IDs [sequence_length]
                - attention_mask: Attention mask [sequence_length]
        """
        # Tokenize text
        inputs = self.processor(
            text=[text],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.config.max_sequence_length,
        )

        # Remove batch dimension (will be re-added by DataLoader)
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
        }


class MultiResolutionImageTransform:
    """Multi-resolution image preprocessing for CLIP.

    Supports training with multiple resolutions to improve model robustness.
    """

    def __init__(
        self,
        config: CLIPFinetuningConfig,
        resolutions: list[int] | None = None,
        augment: bool = False,
    ) -> None:
        """Initialize multi-resolution image transform.

        Args:
            config: CLIP fine-tuning configuration
            resolutions: List of target resolutions (default: [224, 288, 384])
            augment: Whether to apply data augmentation
        """
        self.config = config
        self.resolutions = resolutions or [224, 288, 384]
        self.augment = augment

        # Create transforms for each resolution
        self.transforms = {
            res: self._build_transform(res) for res in self.resolutions
        }

        logger.debug(f"Initialized MultiResolutionImageTransform (resolutions={self.resolutions})")

    def _build_transform(self, resolution: int) -> T.Compose:
        """Build transform pipeline for a specific resolution.

        Args:
            resolution: Target resolution

        Returns:
            Composed transform
        """
        transforms: list = []

        # Resize to target resolution
        transforms.append(T.Resize((resolution, resolution)))

        if self.augment:
            # Data augmentation
            transforms.extend([
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(degrees=10),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            ])

        # Convert to tensor
        transforms.append(T.ToTensor())

        # Normalize (CLIP standard normalization)
        transforms.append(
            T.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            )
        )

        return T.Compose(transforms)

    def __call__(self, image: Image.Image, resolution: int | None = None) -> torch.Tensor:
        """Apply transform to image at specified resolution.

        Args:
            image: PIL Image
            resolution: Target resolution (if None, uses config default)

        Returns:
            Transformed image tensor [C, H, W]
        """
        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Use specified resolution or default
        if resolution is None:
            resolution = self.config.image_resolution

        # Apply transform
        if resolution in self.transforms:
            return self.transforms[resolution](image)
        else:
            # Fallback to closest resolution
            closest = min(self.transforms.keys(), key=lambda x: abs(x - resolution))
            logger.warning(f"Resolution {resolution} not available, using {closest}")
            return self.transforms[closest](image)
