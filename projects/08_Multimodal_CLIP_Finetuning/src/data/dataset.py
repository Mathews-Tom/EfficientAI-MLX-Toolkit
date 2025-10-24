#!/usr/bin/env python3
"""
Dataset class for CLIP fine-tuning.

Provides PyTorch Dataset for loading and preprocessing image-text pairs.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import torch
from PIL import Image
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from transformers import CLIPProcessor

    from config import CLIPFinetuningConfig

logger = logging.getLogger(__name__)


class CLIPDataset(Dataset):
    """PyTorch Dataset for image-text pairs.

    Loads images and captions, applies preprocessing, and returns
    batches suitable for CLIP fine-tuning.
    """

    def __init__(
        self,
        data: list[tuple[Path, str]],
        processor: CLIPProcessor,
        config: CLIPFinetuningConfig,
        image_transform: Callable[[Image.Image], torch.Tensor] | None = None,
        text_transform: Callable[[str], dict[str, torch.Tensor]] | None = None,
    ) -> None:
        """Initialize CLIP dataset.

        Args:
            data: List of (image_path, caption) tuples
            processor: CLIP processor for tokenization and image processing
            config: CLIP fine-tuning configuration
            image_transform: Optional custom image transform
            text_transform: Optional custom text transform
        """
        self.data = data
        self.processor = processor
        self.config = config
        self.image_transform = image_transform
        self.text_transform = text_transform

        # Validate data
        self._validate_data()

        logger.info(f"Initialized CLIPDataset with {len(self.data)} samples")

    def _validate_data(self) -> None:
        """Validate that all images exist and are accessible."""
        missing_images: list[Path] = []
        invalid_captions: list[int] = []

        for idx, (image_path, caption) in enumerate(self.data):
            # Check if image exists
            if not image_path.exists():
                missing_images.append(image_path)

            # Check if caption is valid
            if not caption or not caption.strip():
                invalid_captions.append(idx)

        if missing_images:
            logger.warning(f"Found {len(missing_images)} missing images")
            # Remove missing images from dataset
            self.data = [
                (img_path, caption)
                for img_path, caption in self.data
                if img_path not in missing_images
            ]

        if invalid_captions:
            logger.warning(f"Found {len(invalid_captions)} invalid captions")
            # Remove items with invalid captions
            self.data = [
                item for idx, item in enumerate(self.data)
                if idx not in invalid_captions
            ]

        if not self.data:
            raise ValueError("No valid image-text pairs found after validation")

        logger.info(f"Validated {len(self.data)} valid samples")

    def __len__(self) -> int:
        """Return dataset size.

        Returns:
            Number of samples in dataset
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a single sample.

        Args:
            idx: Sample index

        Returns:
            Dictionary containing:
                - pixel_values: Image tensor [C, H, W]
                - input_ids: Text token IDs [sequence_length]
                - attention_mask: Text attention mask [sequence_length]
                - image_path: Path to image (for debugging)
                - caption: Original caption text (for debugging)

        Raises:
            IndexError: If index out of range
            IOError: If image cannot be loaded
        """
        if idx < 0 or idx >= len(self.data):
            raise IndexError(f"Index {idx} out of range [0, {len(self.data)})")

        image_path, caption = self.data[idx]

        # Load image
        try:
            image = Image.open(image_path)
        except Exception as e:
            raise IOError(f"Failed to load image {image_path}: {e}")

        # Apply image preprocessing
        if self.image_transform is not None:
            # Use custom transform
            pixel_values = self.image_transform(image)
        else:
            # Use CLIP processor's default image preprocessing
            # Convert to RGB if needed
            if image.mode != "RGB":
                image = image.convert("RGB")

            # Process image
            processed = self.processor(images=image, return_tensors="pt")
            pixel_values = processed["pixel_values"].squeeze(0)

        # Apply text preprocessing
        if self.text_transform is not None:
            # Use custom transform
            text_inputs = self.text_transform(caption)
        else:
            # Use CLIP processor's default text preprocessing
            text_inputs = self.processor(
                text=[caption],
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.config.max_sequence_length,
            )
            # Remove batch dimension
            text_inputs = {
                "input_ids": text_inputs["input_ids"].squeeze(0),
                "attention_mask": text_inputs["attention_mask"].squeeze(0),
            }

        return {
            "pixel_values": pixel_values,
            "input_ids": text_inputs["input_ids"],
            "attention_mask": text_inputs["attention_mask"],
            "image_path": str(image_path),
            "caption": caption,
        }


class MultiCaptionCLIPDataset(CLIPDataset):
    """CLIP Dataset supporting multiple captions per image.

    Some datasets provide multiple captions for each image. This dataset
    handles that case by randomly selecting one caption per epoch.
    """

    def __init__(
        self,
        data: list[tuple[Path, list[str]]],
        processor: CLIPProcessor,
        config: CLIPFinetuningConfig,
        image_transform: Callable[[Image.Image], torch.Tensor] | None = None,
        text_transform: Callable[[str], dict[str, torch.Tensor]] | None = None,
    ) -> None:
        """Initialize multi-caption CLIP dataset.

        Args:
            data: List of (image_path, captions_list) tuples
            processor: CLIP processor
            config: CLIP fine-tuning configuration
            image_transform: Optional custom image transform
            text_transform: Optional custom text transform
        """
        # Convert to single-caption format for base class
        self.multi_caption_data = data
        single_caption_data = [
            (img_path, captions[0] if captions else "")
            for img_path, captions in data
        ]

        super().__init__(
            single_caption_data,
            processor,
            config,
            image_transform,
            text_transform,
        )

        logger.info(f"Initialized MultiCaptionCLIPDataset with {len(self.data)} samples")

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a single sample with random caption selection.

        Args:
            idx: Sample index

        Returns:
            Dictionary containing preprocessed image and text
        """
        if idx < 0 or idx >= len(self.multi_caption_data):
            raise IndexError(f"Index {idx} out of range [0, {len(self.multi_caption_data)})")

        image_path, captions = self.multi_caption_data[idx]

        # Randomly select a caption
        import random
        caption = random.choice(captions) if captions else ""

        # Update the single-caption data for this sample
        self.data[idx] = (image_path, caption)

        # Use parent class's __getitem__
        return super().__getitem__(idx)
