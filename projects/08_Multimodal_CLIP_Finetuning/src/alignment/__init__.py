#!/usr/bin/env python3
"""
Image-Text Alignment Module.

Provides utilities for computing image-text similarity, performing retrieval,
calculating alignment metrics, and visualizing alignment results.
"""

from __future__ import annotations

from alignment.metrics import AlignmentMetrics
from alignment.retrieval import ImageTextRetrieval
from alignment.similarity import SimilarityComputer
from alignment.visualization import AlignmentVisualizer

__all__ = [
    "SimilarityComputer",
    "ImageTextRetrieval",
    "AlignmentMetrics",
    "AlignmentVisualizer",
]
