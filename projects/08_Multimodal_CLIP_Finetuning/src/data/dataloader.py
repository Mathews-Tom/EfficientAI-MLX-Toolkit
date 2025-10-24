#!/usr/bin/env python3
"""
DataLoader utilities for CLIP fine-tuning.

Provides collation functions and DataLoader factory for batching image-text pairs.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from config import CLIPFinetuningConfig

    from data.dataset import CLIPDataset

logger = logging.getLogger(__name__)


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate batch of image-text pairs.

    Stacks images and pads text sequences to the same length within the batch.

    Args:
        batch: List of dictionaries from CLIPDataset.__getitem__

    Returns:
        Dictionary containing batched tensors:
            - pixel_values: [batch_size, C, H, W]
            - input_ids: [batch_size, sequence_length]
            - attention_mask: [batch_size, sequence_length]
            - image_paths: List of image paths (for debugging)
            - captions: List of captions (for debugging)
    """
    # Stack images
    pixel_values = torch.stack([item["pixel_values"] for item in batch])

    # Stack text inputs (already padded by processor)
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])

    # Collect metadata
    image_paths = [item["image_path"] for item in batch]
    captions = [item["caption"] for item in batch]

    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "image_paths": image_paths,
        "captions": captions,
    }


def create_dataloader(
    dataset: CLIPDataset,
    config: CLIPFinetuningConfig,
    batch_size: int | None = None,
    shuffle: bool = True,
    num_workers: int | None = None,
    pin_memory: bool = True,
    drop_last: bool = False,
) -> DataLoader:
    """Create DataLoader for CLIP dataset.

    Args:
        dataset: CLIPDataset instance
        config: CLIP fine-tuning configuration
        batch_size: Batch size (if None, uses config.batch_size or auto-determines)
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes (if None, uses config.num_workers)
        pin_memory: Whether to pin memory for faster GPU transfer
        drop_last: Whether to drop last incomplete batch

    Returns:
        DataLoader instance

    Raises:
        ValueError: If batch_size cannot be determined
    """
    # Determine batch size
    if batch_size is None:
        if config.batch_size is not None:
            batch_size = config.batch_size
        else:
            # Auto-determine based on device (simple heuristic)
            if config.use_mps:
                batch_size = 16
            else:
                batch_size = 8

    if batch_size <= 0:
        raise ValueError(f"Batch size must be positive, got {batch_size}")

    # Determine number of workers
    if num_workers is None:
        # Default to config value or 4
        num_workers = getattr(config, "num_workers", 4)

    # Disable pin_memory for MPS (not supported)
    if config.use_mps:
        pin_memory = False

    logger.info(f"Creating DataLoader: batch_size={batch_size}, shuffle={shuffle}, num_workers={num_workers}")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_fn,
    )


def create_train_val_dataloaders(
    train_dataset: CLIPDataset,
    val_dataset: CLIPDataset,
    config: CLIPFinetuningConfig,
    batch_size: int | None = None,
    num_workers: int | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Create train and validation DataLoaders.

    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        config: CLIP fine-tuning configuration
        batch_size: Batch size (if None, uses config or auto-determines)
        num_workers: Number of workers (if None, uses config)

    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_loader = create_dataloader(
        train_dataset,
        config,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,  # Drop last incomplete batch for training
    )

    val_loader = create_dataloader(
        val_dataset,
        config,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle validation data
        num_workers=num_workers,
        drop_last=False,  # Keep all validation samples
    )

    logger.info(f"Created train DataLoader: {len(train_loader)} batches")
    logger.info(f"Created validation DataLoader: {len(val_loader)} batches")

    return train_loader, val_loader


class PrefetchDataLoader:
    """DataLoader with prefetching for improved performance.

    Prefetches next batch while current batch is being processed,
    reducing data loading bottleneck.
    """

    def __init__(
        self,
        dataloader: DataLoader,
        device: torch.device,
    ) -> None:
        """Initialize prefetch DataLoader.

        Args:
            dataloader: Base DataLoader to wrap
            device: Device to prefetch data to
        """
        self.dataloader = dataloader
        self.device = device
        self.stream = torch.cuda.Stream() if device.type == "cuda" else None

    def __iter__(self):
        """Iterate with prefetching."""
        loader_iter = iter(self.dataloader)

        # Prefetch first batch
        try:
            next_batch = next(loader_iter)
            next_batch = self._to_device(next_batch)
        except StopIteration:
            return

        for batch in loader_iter:
            # Current batch is already prefetched
            current_batch = next_batch

            # Prefetch next batch
            next_batch = self._to_device(batch)

            yield current_batch

        # Yield last batch
        yield next_batch

    def _to_device(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Move batch to device.

        Args:
            batch: Batch dictionary

        Returns:
            Batch with tensors moved to device
        """
        result = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                result[key] = value.to(self.device, non_blocking=True)
            else:
                result[key] = value
        return result

    def __len__(self) -> int:
        """Return number of batches."""
        return len(self.dataloader)
