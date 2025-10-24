#!/usr/bin/env python3
"""Tests for DataLoader utilities."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from config import CLIPFinetuningConfig
from data.dataloader import (
    collate_fn,
    create_dataloader,
    create_train_val_dataloaders,
    PrefetchDataLoader,
)
from data.dataset import CLIPDataset


@pytest.fixture
def config():
    """Create test configuration."""
    return CLIPFinetuningConfig(
        model_name="openai/clip-vit-base-patch32",
        image_resolution=224,
        max_sequence_length=77,
        batch_size=4,
        num_workers=0,  # Use 0 workers for testing
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
def sample_dataset(fixtures_dir, mock_processor, config):
    """Create sample dataset."""
    data = [
        (fixtures_dir / "test_image_0.jpg", "Caption 0"),
        (fixtures_dir / "test_image_1.jpg", "Caption 1"),
        (fixtures_dir / "test_image_2.jpg", "Caption 2"),
        (fixtures_dir / "test_image_3.jpg", "Caption 3"),
    ]
    return CLIPDataset(data, mock_processor, config)


class TestCollateFn:
    """Tests for collate_fn."""

    def test_collate_batch(self):
        """Test collating a batch of samples."""
        batch = [
            {
                "pixel_values": torch.randn(3, 224, 224),
                "input_ids": torch.ones(77, dtype=torch.long),
                "attention_mask": torch.ones(77, dtype=torch.long),
                "image_path": "image0.jpg",
                "caption": "Caption 0",
            },
            {
                "pixel_values": torch.randn(3, 224, 224),
                "input_ids": torch.ones(77, dtype=torch.long),
                "attention_mask": torch.ones(77, dtype=torch.long),
                "image_path": "image1.jpg",
                "caption": "Caption 1",
            },
        ]

        result = collate_fn(batch)

        # Check keys
        assert "pixel_values" in result
        assert "input_ids" in result
        assert "attention_mask" in result
        assert "image_paths" in result
        assert "captions" in result

        # Check shapes
        assert result["pixel_values"].shape == (2, 3, 224, 224)
        assert result["input_ids"].shape == (2, 77)
        assert result["attention_mask"].shape == (2, 77)

        # Check metadata
        assert len(result["image_paths"]) == 2
        assert len(result["captions"]) == 2

    def test_collate_empty_batch(self):
        """Test collating an empty batch."""
        batch = []

        with pytest.raises((ValueError, RuntimeError)):
            collate_fn(batch)

    def test_collate_single_item(self):
        """Test collating a single item."""
        batch = [
            {
                "pixel_values": torch.randn(3, 224, 224),
                "input_ids": torch.ones(77, dtype=torch.long),
                "attention_mask": torch.ones(77, dtype=torch.long),
                "image_path": "image0.jpg",
                "caption": "Caption 0",
            }
        ]

        result = collate_fn(batch)

        # Check shapes (batch size 1)
        assert result["pixel_values"].shape == (1, 3, 224, 224)
        assert result["input_ids"].shape == (1, 77)


class TestCreateDataLoader:
    """Tests for create_dataloader."""

    def test_create_dataloader_default(self, sample_dataset, config):
        """Test creating DataLoader with default parameters."""
        loader = create_dataloader(sample_dataset, config)

        assert len(loader) > 0
        assert loader.batch_size == config.batch_size

    def test_create_dataloader_custom_batch_size(self, sample_dataset, config):
        """Test creating DataLoader with custom batch size."""
        loader = create_dataloader(sample_dataset, config, batch_size=2)

        assert loader.batch_size == 2

    def test_create_dataloader_shuffle(self, sample_dataset, config):
        """Test creating DataLoader with shuffle."""
        loader = create_dataloader(sample_dataset, config, shuffle=True)

        # Just check it doesn't error
        batch = next(iter(loader))
        assert batch is not None

    def test_create_dataloader_no_shuffle(self, sample_dataset, config):
        """Test creating DataLoader without shuffle."""
        loader = create_dataloader(sample_dataset, config, shuffle=False)

        # Just check it doesn't error
        batch = next(iter(loader))
        assert batch is not None

    def test_create_dataloader_auto_batch_size(self, sample_dataset, config):
        """Test creating DataLoader with auto-determined batch size."""
        config.batch_size = None
        loader = create_dataloader(sample_dataset, config)

        # Should auto-determine batch size
        assert loader.batch_size in [8, 16]  # Common auto values

    def test_create_dataloader_invalid_batch_size(self, sample_dataset, config):
        """Test creating DataLoader with invalid batch size."""
        with pytest.raises(ValueError):
            create_dataloader(sample_dataset, config, batch_size=0)

        with pytest.raises(ValueError):
            create_dataloader(sample_dataset, config, batch_size=-1)

    def test_create_dataloader_iterate(self, sample_dataset, config):
        """Test iterating over DataLoader."""
        loader = create_dataloader(sample_dataset, config, batch_size=2)

        batches = list(loader)
        assert len(batches) > 0

        # Check first batch
        batch = batches[0]
        assert "pixel_values" in batch
        assert "input_ids" in batch
        assert batch["pixel_values"].shape[0] <= 2  # Batch size


class TestCreateTrainValDataLoaders:
    """Tests for create_train_val_dataloaders."""

    def test_create_train_val_loaders(self, sample_dataset, config):
        """Test creating train and validation DataLoaders."""
        train_loader, val_loader = create_train_val_dataloaders(
            sample_dataset,
            sample_dataset,
            config,
        )

        assert train_loader is not None
        assert val_loader is not None

        # Train should shuffle, val should not
        assert len(train_loader) > 0
        assert len(val_loader) > 0

    def test_train_loader_drops_last(self, sample_dataset, config):
        """Test that train loader drops last incomplete batch."""
        # Create loader with batch size that doesn't divide evenly
        train_loader, _ = create_train_val_dataloaders(
            sample_dataset,
            sample_dataset,
            config,
            batch_size=3,
        )

        # Iterate through all batches
        batches = list(train_loader)

        # Last batch should be dropped if incomplete
        # (dataset has 4 samples, batch_size=3, so last batch of size 1 is dropped)
        if len(batches) > 0:
            for batch in batches:
                assert batch["pixel_values"].shape[0] == 3

    def test_val_loader_keeps_all(self, sample_dataset, config):
        """Test that validation loader keeps all samples."""
        _, val_loader = create_train_val_dataloaders(
            sample_dataset,
            sample_dataset,
            config,
            batch_size=3,
        )

        # Iterate through all batches
        batches = list(val_loader)
        total_samples = sum(batch["pixel_values"].shape[0] for batch in batches)

        # Should have all 4 samples
        assert total_samples == len(sample_dataset)


class TestPrefetchDataLoader:
    """Tests for PrefetchDataLoader."""

    def test_init(self, sample_dataset, config):
        """Test prefetch loader initialization."""
        base_loader = create_dataloader(sample_dataset, config, num_workers=0)
        device = torch.device("cpu")

        prefetch_loader = PrefetchDataLoader(base_loader, device)

        assert prefetch_loader.dataloader == base_loader
        assert prefetch_loader.device == device

    def test_len(self, sample_dataset, config):
        """Test prefetch loader length."""
        base_loader = create_dataloader(sample_dataset, config, num_workers=0)
        device = torch.device("cpu")

        prefetch_loader = PrefetchDataLoader(base_loader, device)

        assert len(prefetch_loader) == len(base_loader)

    def test_iteration(self, sample_dataset, config):
        """Test iterating over prefetch loader."""
        base_loader = create_dataloader(sample_dataset, config, batch_size=2, num_workers=0)
        device = torch.device("cpu")

        prefetch_loader = PrefetchDataLoader(base_loader, device)

        batches = list(prefetch_loader)
        assert len(batches) > 0

        # Check that data is on correct device
        for batch in batches:
            assert batch["pixel_values"].device == device

    def test_to_device(self, sample_dataset, config):
        """Test moving batch to device."""
        base_loader = create_dataloader(sample_dataset, config, num_workers=0)
        device = torch.device("cpu")

        prefetch_loader = PrefetchDataLoader(base_loader, device)

        # Get a batch
        batch = next(iter(base_loader))

        # Move to device
        moved_batch = prefetch_loader._to_device(batch)

        # Check tensors are on device
        assert moved_batch["pixel_values"].device == device
        assert moved_batch["input_ids"].device == device

        # Check non-tensors are preserved
        assert "image_paths" in moved_batch
        assert "captions" in moved_batch
