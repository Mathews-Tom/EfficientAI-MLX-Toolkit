#!/usr/bin/env python3
"""
CLIP Fine-tuning Controller.

Main controller for managing CLIP model loading, MPS device configuration,
and memory-efficient initialization for domain-specific fine-tuning.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from transformers import CLIPModel, CLIPProcessor

from config import CLIPFinetuningConfig
from device_manager import DeviceManager

if TYPE_CHECKING:
    from PIL import Image

logger = logging.getLogger(__name__)


class CLIPFinetuningController:
    """Main controller for CLIP domain-specific fine-tuning.

    Handles model loading from HuggingFace, device management (MPS/CPU),
    and memory-efficient initialization for training.
    """

    def __init__(self, config: CLIPFinetuningConfig) -> None:
        """Initialize CLIP fine-tuning controller.

        Args:
            config: Configuration for CLIP fine-tuning
        """
        self.config = config
        self.device_manager = DeviceManager(use_mps=config.use_mps)
        self.model: CLIPModel | None = None
        self.processor: CLIPProcessor | None = None

        logger.info("Initializing CLIP Fine-tuning Controller")
        logger.info(f"Model: {config.model_name}")
        logger.info(f"Domain: {config.domain}")

    def setup(self) -> None:
        """Set up the CLIP model and processor.

        Loads the model from HuggingFace and moves it to the appropriate device.
        Applies memory-efficient initialization and mixed precision if configured.
        """
        logger.info("Setting up CLIP model and processor")

        # Log device information
        self.device_manager.log_device_info()

        # Load processor (tokenizer + image processor)
        logger.info(f"Loading CLIP processor from {self.config.model_name}")
        self.processor = CLIPProcessor.from_pretrained(self.config.model_name)

        # Load model
        logger.info(f"Loading CLIP model from {self.config.model_name}")
        self.model = CLIPModel.from_pretrained(self.config.model_name)

        # Move model to device
        device = self.device_manager.device
        logger.info(f"Moving model to {device}")
        self.model.to(device)

        # Apply mixed precision if configured and device supports it
        if self.config.mixed_precision and device.type == "mps":
            logger.info("Enabling mixed precision training for MPS")
            # Note: MPS doesn't fully support torch.cuda.amp yet, but we can prepare for it
            # For now, we'll use float16 for forward passes manually during training
            self.model.half()

        # Apply device-specific optimizations
        self.device_manager.optimize_for_device()

        # Set model to training mode
        self.model.train()

        logger.info("CLIP model setup complete")
        self._log_model_info()

    def _log_model_info(self) -> None:
        """Log information about the loaded model."""
        if self.model is None:
            logger.warning("Model not loaded")
            return

        logger.info("=" * 50)
        logger.info("Model Information")
        logger.info("=" * 50)

        # Get first parameter for dtype and device info
        try:
            first_param = next(iter(self.model.parameters()))
            param_dtype = first_param.dtype
            param_device = first_param.device
        except StopIteration:
            logger.warning("Model has no parameters")
            param_dtype = "unknown"
            param_device = "unknown"

        # Count parameters (using separate iterations)
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Model dtype: {param_dtype}")
        logger.info(f"Model device: {param_device}")

        # Log memory info
        memory_info = self.device_manager.get_memory_info()
        logger.info(f"Memory info: {memory_info}")

        logger.info("=" * 50)

    def encode_text(self, text: list[str]) -> torch.Tensor:
        """Encode text using CLIP text encoder.

        Args:
            text: List of text strings to encode

        Returns:
            Text embeddings tensor

        Raises:
            RuntimeError: If model or processor not initialized
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("Model not initialized. Call setup() first.")

        # Tokenize text
        inputs = self.processor(
            text=text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_sequence_length,
        )

        # Move to device
        inputs = {k: v.to(self.device_manager.device) for k, v in inputs.items()}

        # Encode text
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)

        return text_features

    def encode_image(self, images: list[Image.Image]) -> torch.Tensor:
        """Encode images using CLIP image encoder.

        Args:
            images: List of PIL images to encode

        Returns:
            Image embeddings tensor

        Raises:
            RuntimeError: If model or processor not initialized
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("Model not initialized. Call setup() first.")

        # Process images
        inputs = self.processor(images=images, return_tensors="pt")

        # Move to device
        inputs = {k: v.to(self.device_manager.device) for k, v in inputs.items()}

        # Encode images
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)

        return image_features

    def compute_similarity(
        self, text: list[str], images: list[Image.Image]
    ) -> torch.Tensor:
        """Compute image-text similarity scores.

        Args:
            text: List of text strings
            images: List of PIL images

        Returns:
            Similarity matrix tensor [num_images x num_texts]

        Raises:
            RuntimeError: If model or processor not initialized
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("Model not initialized. Call setup() first.")

        # Encode text and images
        text_features = self.encode_text(text)
        image_features = self.encode_image(images)

        # Normalize features
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # Compute similarity (dot product)
        similarity = torch.matmul(image_features, text_features.T)

        return similarity

    def determine_batch_size(self) -> int:
        """Automatically determine optimal batch size based on available memory.

        Returns:
            Optimal batch size for training

        Note:
            This is a simple heuristic. For MPS, we use a conservative batch size
            since unified memory is shared with system RAM.
        """
        if self.config.batch_size is not None:
            return self.config.batch_size

        # Default batch sizes based on device
        if self.device_manager.device.type == "mps":
            # Conservative batch size for MPS
            batch_size = 16
        else:
            # CPU fallback - even smaller batch size
            batch_size = 8

        logger.info(f"Auto-determined batch size: {batch_size}")
        return batch_size

    def get_model_state(self) -> dict[str, object]:
        """Get current model state information.

        Returns:
            Dictionary containing model state information
        """
        if self.model is None:
            return {"status": "not_initialized"}

        return {
            "status": "initialized",
            "model_name": self.config.model_name,
            "domain": self.config.domain,
            "device": str(self.device_manager.device),
            "device_type": self.device_manager.device.type,
            "is_apple_silicon": self.device_manager.is_apple_silicon,
            "is_mps_available": self.device_manager.is_mps_available,
            "mixed_precision": self.config.mixed_precision,
            "training_mode": self.model.training,
        }
