#!/usr/bin/env python3
"""Tests for image and text transforms."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from PIL import Image

from config import CLIPFinetuningConfig
from data.transforms import (
    ImageTransform,
    MultiResolutionImageTransform,
    TextTransform,
)


@pytest.fixture
def config():
    """Create test configuration."""
    return CLIPFinetuningConfig(
        model_name="openai/clip-vit-base-patch32",
        image_resolution=224,
        max_sequence_length=77,
    )


@pytest.fixture
def mock_processor():
    """Create mock CLIP processor."""
    processor = MagicMock()

    def mock_text_process(text, **kwargs):
        # Return mock tokenized output
        seq_len = kwargs.get("max_length", 77)
        return {
            "input_ids": torch.ones(1, seq_len, dtype=torch.long),
            "attention_mask": torch.ones(1, seq_len, dtype=torch.long),
        }

    processor.side_effect = mock_text_process
    return processor


@pytest.fixture
def test_image():
    """Create test image."""
    return Image.new("RGB", (256, 256), color=(100, 150, 200))


class TestImageTransform:
    """Tests for ImageTransform class."""

    def test_init(self, config):
        """Test transform initialization."""
        transform = ImageTransform(config, augment=False)
        assert transform.config == config
        assert not transform.augment

    def test_init_with_augmentation(self, config):
        """Test transform initialization with augmentation."""
        transform = ImageTransform(config, augment=True)
        assert transform.augment

    def test_call(self, config, test_image):
        """Test applying transform to image."""
        transform = ImageTransform(config, augment=False)
        result = transform(test_image)

        # Check output is tensor
        assert isinstance(result, torch.Tensor)
        # Check shape [C, H, W]
        assert result.shape == (3, 224, 224)
        # Check dtype
        assert result.dtype == torch.float32

    def test_rgb_conversion(self, config):
        """Test that grayscale images are converted to RGB."""
        # Create grayscale image
        gray_image = Image.new("L", (256, 256), color=128)

        transform = ImageTransform(config, augment=False)
        result = transform(gray_image)

        # Should still have 3 channels
        assert result.shape[0] == 3

    def test_normalization(self, config, test_image):
        """Test that normalization is applied."""
        transform = ImageTransform(config, augment=False)
        result = transform(test_image)

        # Values should be normalized (not in [0, 255] range)
        assert result.min() < 0 or result.max() > 1

    def test_augmentation_determinism(self, config, test_image):
        """Test that augmentation produces different results."""
        transform = ImageTransform(config, augment=True)

        # Apply transform multiple times
        results = [transform(test_image) for _ in range(3)]

        # Results should be different due to random augmentation
        # (This might occasionally fail due to random chance, but unlikely)
        assert not all(torch.allclose(results[0], r) for r in results[1:])


class TestTextTransform:
    """Tests for TextTransform class."""

    def test_init(self, mock_processor, config):
        """Test transform initialization."""
        transform = TextTransform(mock_processor, config)
        assert transform.processor == mock_processor
        assert transform.config == config

    def test_call(self, config):
        """Test applying transform to text."""
        # Create real-ish mock processor
        processor = MagicMock()
        processor.return_value = {
            "input_ids": torch.ones(1, 77, dtype=torch.long),
            "attention_mask": torch.ones(1, 77, dtype=torch.long),
        }

        transform = TextTransform(processor, config)
        result = transform("Test caption")

        # Check output format
        assert isinstance(result, dict)
        assert "input_ids" in result
        assert "attention_mask" in result

        # Check shapes (batch dimension removed)
        assert result["input_ids"].shape == (77,)
        assert result["attention_mask"].shape == (77,)

    def test_tokenization_params(self, config):
        """Test that correct tokenization parameters are used."""
        processor = MagicMock()
        processor.return_value = {
            "input_ids": torch.ones(1, 77, dtype=torch.long),
            "attention_mask": torch.ones(1, 77, dtype=torch.long),
        }

        transform = TextTransform(processor, config)
        transform("Test caption")

        # Verify processor was called with correct parameters
        processor.assert_called_once()
        call_kwargs = processor.call_args[1]
        assert call_kwargs["padding"] == "max_length"
        assert call_kwargs["truncation"] is True
        assert call_kwargs["max_length"] == config.max_sequence_length


class TestMultiResolutionImageTransform:
    """Tests for MultiResolutionImageTransform class."""

    def test_init_default_resolutions(self, config):
        """Test initialization with default resolutions."""
        transform = MultiResolutionImageTransform(config)
        assert transform.resolutions == [224, 288, 384]

    def test_init_custom_resolutions(self, config):
        """Test initialization with custom resolutions."""
        resolutions = [192, 256, 320]
        transform = MultiResolutionImageTransform(config, resolutions=resolutions)
        assert transform.resolutions == resolutions

    def test_call_default_resolution(self, config, test_image):
        """Test applying transform with default resolution."""
        transform = MultiResolutionImageTransform(config)
        result = transform(test_image)

        # Should use config default (224)
        assert result.shape == (3, 224, 224)

    def test_call_specific_resolution(self, config, test_image):
        """Test applying transform with specific resolution."""
        transform = MultiResolutionImageTransform(config)
        result = transform(test_image, resolution=288)

        assert result.shape == (3, 288, 288)

    def test_call_unsupported_resolution(self, config, test_image):
        """Test applying transform with unsupported resolution."""
        transform = MultiResolutionImageTransform(config, resolutions=[224, 288])

        # Request resolution not in list
        result = transform(test_image, resolution=512)

        # Should fall back to closest resolution (288)
        assert result.shape == (3, 288, 288)

    def test_rgb_conversion(self, config):
        """Test that grayscale images are converted to RGB."""
        gray_image = Image.new("L", (256, 256), color=128)

        transform = MultiResolutionImageTransform(config)
        result = transform(gray_image)

        assert result.shape[0] == 3

    def test_augmentation(self, config, test_image):
        """Test that augmentation is applied when enabled."""
        transform = MultiResolutionImageTransform(config, augment=True)

        # Apply transform multiple times
        results = [transform(test_image) for _ in range(3)]

        # Results should be different due to random augmentation
        assert not all(torch.allclose(results[0], r) for r in results[1:])
