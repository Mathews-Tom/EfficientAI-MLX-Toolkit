#!/usr/bin/env python3
"""Tests for CLIP dataset."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from PIL import Image

from config import CLIPFinetuningConfig
from data.dataset import CLIPDataset, MultiCaptionCLIPDataset


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

    def mock_call(text=None, images=None, **kwargs):
        result = {}
        if text is not None:
            seq_len = kwargs.get("max_length", 77)
            result["input_ids"] = torch.ones(len(text) if isinstance(text, list) else 1, seq_len, dtype=torch.long)
            result["attention_mask"] = torch.ones(len(text) if isinstance(text, list) else 1, seq_len, dtype=torch.long)
        if images is not None:
            num_images = len(images) if isinstance(images, list) else 1
            result["pixel_values"] = torch.randn(num_images, 3, 224, 224)
        return result

    processor.side_effect = mock_call
    return processor


@pytest.fixture
def fixtures_dir():
    """Return path to test fixtures directory."""
    return Path(__file__).parent.parent / "fixtures"


@pytest.fixture
def sample_data(fixtures_dir):
    """Create sample data list."""
    return [
        (fixtures_dir / "test_image_0.jpg", "Caption 0"),
        (fixtures_dir / "test_image_1.jpg", "Caption 1"),
        (fixtures_dir / "test_image_2.jpg", "Caption 2"),
    ]


class TestCLIPDataset:
    """Tests for CLIPDataset class."""

    def test_init(self, sample_data, mock_processor, config):
        """Test dataset initialization."""
        dataset = CLIPDataset(sample_data, mock_processor, config)

        assert len(dataset) == 3
        assert dataset.data == sample_data
        assert dataset.processor == mock_processor
        assert dataset.config == config

    def test_len(self, sample_data, mock_processor, config):
        """Test dataset length."""
        dataset = CLIPDataset(sample_data, mock_processor, config)
        assert len(dataset) == 3

    def test_getitem(self, sample_data, mock_processor, config):
        """Test getting single item."""
        dataset = CLIPDataset(sample_data, mock_processor, config)
        item = dataset[0]

        # Check required keys
        assert "pixel_values" in item
        assert "input_ids" in item
        assert "attention_mask" in item
        assert "image_path" in item
        assert "caption" in item

        # Check types
        assert isinstance(item["pixel_values"], torch.Tensor)
        assert isinstance(item["input_ids"], torch.Tensor)
        assert isinstance(item["attention_mask"], torch.Tensor)
        assert isinstance(item["image_path"], str)
        assert isinstance(item["caption"], str)

        # Check shapes
        assert item["pixel_values"].shape == (3, 224, 224)
        assert item["input_ids"].shape == (77,)
        assert item["attention_mask"].shape == (77,)

    def test_getitem_out_of_range(self, sample_data, mock_processor, config):
        """Test getting item with invalid index."""
        dataset = CLIPDataset(sample_data, mock_processor, config)

        with pytest.raises(IndexError):
            _ = dataset[10]

        with pytest.raises(IndexError):
            _ = dataset[-10]

    def test_getitem_missing_image(self, mock_processor, config):
        """Test getting item with missing image."""
        data = [(Path("nonexistent.jpg"), "Caption")]
        # Dataset should validate and remove missing images, raising error if all invalid
        with pytest.raises(ValueError, match="No valid image-text pairs"):
            CLIPDataset(data, mock_processor, config)

    def test_validation_missing_images(self, mock_processor, config, fixtures_dir):
        """Test validation removes missing images."""
        data = [
            (fixtures_dir / "test_image_0.jpg", "Caption 0"),
            (Path("missing.jpg"), "Caption 1"),
            (fixtures_dir / "test_image_1.jpg", "Caption 2"),
        ]

        dataset = CLIPDataset(data, mock_processor, config)

        # Should have 2 valid samples (missing one removed)
        assert len(dataset) == 2

    def test_validation_invalid_captions(self, mock_processor, config, fixtures_dir):
        """Test validation removes items with invalid captions."""
        data = [
            (fixtures_dir / "test_image_0.jpg", "Valid caption"),
            (fixtures_dir / "test_image_1.jpg", ""),  # Empty caption
            (fixtures_dir / "test_image_2.jpg", "   "),  # Whitespace only
        ]

        dataset = CLIPDataset(data, mock_processor, config)

        # Should have 1 valid sample
        assert len(dataset) == 1

    def test_validation_all_invalid(self, mock_processor, config):
        """Test validation raises error when all samples invalid."""
        data = [(Path("missing.jpg"), "Caption")]

        with pytest.raises(ValueError, match="No valid image-text pairs"):
            CLIPDataset(data, mock_processor, config)

    def test_custom_image_transform(self, sample_data, mock_processor, config):
        """Test using custom image transform."""
        def custom_transform(image):
            return torch.zeros(3, 224, 224)

        dataset = CLIPDataset(
            sample_data,
            mock_processor,
            config,
            image_transform=custom_transform,
        )

        item = dataset[0]
        # Should use custom transform (all zeros)
        assert torch.allclose(item["pixel_values"], torch.zeros(3, 224, 224))

    def test_custom_text_transform(self, sample_data, mock_processor, config):
        """Test using custom text transform."""
        def custom_transform(text):
            return {
                "input_ids": torch.zeros(77, dtype=torch.long),
                "attention_mask": torch.zeros(77, dtype=torch.long),
            }

        dataset = CLIPDataset(
            sample_data,
            mock_processor,
            config,
            text_transform=custom_transform,
        )

        item = dataset[0]
        # Should use custom transform (all zeros)
        assert torch.allclose(item["input_ids"], torch.zeros(77, dtype=torch.long))


class TestMultiCaptionCLIPDataset:
    """Tests for MultiCaptionCLIPDataset class."""

    def test_init(self, mock_processor, config, fixtures_dir):
        """Test multi-caption dataset initialization."""
        data = [
            (fixtures_dir / "test_image_0.jpg", ["Caption 1", "Caption 2"]),
            (fixtures_dir / "test_image_1.jpg", ["Caption 3", "Caption 4", "Caption 5"]),
        ]

        dataset = MultiCaptionCLIPDataset(data, mock_processor, config)
        assert len(dataset) == 2

    def test_len(self, mock_processor, config, fixtures_dir):
        """Test dataset length."""
        data = [
            (fixtures_dir / "test_image_0.jpg", ["Caption 1", "Caption 2"]),
            (fixtures_dir / "test_image_1.jpg", ["Caption 3"]),
        ]

        dataset = MultiCaptionCLIPDataset(data, mock_processor, config)
        assert len(dataset) == 2

    def test_getitem(self, mock_processor, config, fixtures_dir):
        """Test getting single item."""
        data = [
            (fixtures_dir / "test_image_0.jpg", ["Caption 1", "Caption 2"]),
        ]

        dataset = MultiCaptionCLIPDataset(data, mock_processor, config)
        item = dataset[0]

        # Check required keys
        assert "pixel_values" in item
        assert "input_ids" in item
        assert "attention_mask" in item
        assert "caption" in item

        # Caption should be one of the provided captions
        assert item["caption"] in ["Caption 1", "Caption 2"]

    def test_random_caption_selection(self, mock_processor, config, fixtures_dir):
        """Test that captions are randomly selected."""
        data = [
            (fixtures_dir / "test_image_0.jpg", ["Caption A", "Caption B", "Caption C"]),
        ]

        dataset = MultiCaptionCLIPDataset(data, mock_processor, config)

        # Get item multiple times
        captions = [dataset[0]["caption"] for _ in range(10)]

        # Should have some variety (very unlikely to get same caption 10 times)
        assert len(set(captions)) > 1

    def test_empty_captions(self, mock_processor, config, fixtures_dir):
        """Test handling of empty caption lists."""
        data = [
            (fixtures_dir / "test_image_0.jpg", []),
        ]

        # Empty caption list results in empty string which fails validation
        with pytest.raises(ValueError, match="No valid image-text pairs"):
            MultiCaptionCLIPDataset(data, mock_processor, config)
