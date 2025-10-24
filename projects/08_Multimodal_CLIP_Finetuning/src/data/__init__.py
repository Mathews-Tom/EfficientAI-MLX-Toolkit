#!/usr/bin/env python3
"""
Data pipeline for CLIP fine-tuning.

This module provides utilities for loading, preprocessing, and batching
image-text pairs for CLIP fine-tuning.
"""

from __future__ import annotations

from data.dataset import CLIPDataset
from data.dataloader import collate_fn, create_dataloader
from data.loaders import DatasetLoader
from data.transforms import ImageTransform, TextTransform

__all__ = [
    "CLIPDataset",
    "collate_fn",
    "create_dataloader",
    "DatasetLoader",
    "ImageTransform",
    "TextTransform",
]
