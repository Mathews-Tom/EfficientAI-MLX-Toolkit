#!/usr/bin/env python3
"""
Similarity computation for image-text alignment.

Provides utilities for computing pairwise similarity between image and text embeddings
using various distance metrics (cosine, dot product, Euclidean distance).
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class SimilarityComputer:
    """Compute image-text similarity scores.

    Supports multiple similarity metrics:
    - cosine: Cosine similarity (normalized dot product)
    - dot: Raw dot product similarity
    - euclidean: Negative Euclidean distance

    Attributes:
        normalize: Whether to normalize embeddings before computing similarity
        metric: Similarity metric to use
    """

    def __init__(
        self,
        normalize: bool = True,
        metric: str = "cosine",
    ) -> None:
        """Initialize similarity computer.

        Args:
            normalize: Whether to normalize embeddings before computing similarity
            metric: Similarity metric ('cosine', 'dot', 'euclidean')

        Raises:
            ValueError: If metric is not supported
        """
        valid_metrics = {"cosine", "dot", "euclidean"}
        if metric not in valid_metrics:
            raise ValueError(
                f"Invalid metric '{metric}'. Must be one of {valid_metrics}"
            )

        self.normalize = normalize
        self.metric = metric

        logger.info(f"Initialized SimilarityComputer with metric={metric}, normalize={normalize}")

    def compute(
        self,
        image_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """Compute pairwise similarity between image and text embeddings.

        Args:
            image_embeds: Image embeddings [N_images, embed_dim]
            text_embeds: Text embeddings [N_texts, embed_dim]

        Returns:
            Similarity matrix [N_images, N_texts]

        Raises:
            ValueError: If embeddings have incompatible dimensions
        """
        if image_embeds.dim() != 2 or text_embeds.dim() != 2:
            raise ValueError(
                f"Expected 2D tensors, got image_embeds={image_embeds.dim()}D, "
                f"text_embeds={text_embeds.dim()}D"
            )

        if image_embeds.size(1) != text_embeds.size(1):
            raise ValueError(
                f"Embedding dimensions must match: image={image_embeds.size(1)}, "
                f"text={text_embeds.size(1)}"
            )

        # Normalize embeddings if requested
        if self.normalize:
            image_embeds = F.normalize(image_embeds, p=2, dim=1)
            text_embeds = F.normalize(text_embeds, p=2, dim=1)

        # Compute similarity based on metric
        if self.metric == "cosine":
            # Cosine similarity (normalized dot product)
            if not self.normalize:
                # Need to normalize for cosine similarity
                image_embeds = F.normalize(image_embeds, p=2, dim=1)
                text_embeds = F.normalize(text_embeds, p=2, dim=1)
            similarity = torch.matmul(image_embeds, text_embeds.T)

        elif self.metric == "dot":
            # Raw dot product
            similarity = torch.matmul(image_embeds, text_embeds.T)

        elif self.metric == "euclidean":
            # Negative Euclidean distance (higher = more similar)
            similarity = -torch.cdist(image_embeds, text_embeds, p=2)

        else:
            raise ValueError(f"Unsupported metric: {self.metric}")

        return similarity

    def top_k(
        self,
        similarity_matrix: torch.Tensor,
        k: int = 5,
        dim: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get top-k similarities and indices.

        Args:
            similarity_matrix: Similarity matrix [N_queries, N_candidates]
            k: Number of top results to return
            dim: Dimension to compute top-k over (0=column-wise, 1=row-wise)

        Returns:
            Tuple of (top_k_values, top_k_indices)

        Raises:
            ValueError: If k is invalid
        """
        if similarity_matrix.dim() != 2:
            raise ValueError(
                f"Expected 2D similarity matrix, got {similarity_matrix.dim()}D"
            )

        max_k = similarity_matrix.size(dim)
        if k <= 0 or k > max_k:
            raise ValueError(f"k must be in range [1, {max_k}], got {k}")

        # Get top-k values and indices
        top_k_values, top_k_indices = torch.topk(
            similarity_matrix, k=k, dim=dim, largest=True, sorted=True
        )

        return top_k_values, top_k_indices

    def batch_compute(
        self,
        image_embeds_list: list[torch.Tensor],
        text_embeds_list: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        """Compute similarity for multiple pairs of embedding sets.

        Args:
            image_embeds_list: List of image embedding tensors
            text_embeds_list: List of text embedding tensors

        Returns:
            List of similarity matrices

        Raises:
            ValueError: If lists have different lengths
        """
        if len(image_embeds_list) != len(text_embeds_list):
            raise ValueError(
                f"Lists must have same length: images={len(image_embeds_list)}, "
                f"texts={len(text_embeds_list)}"
            )

        similarities = []
        for image_embeds, text_embeds in zip(image_embeds_list, text_embeds_list):
            similarity = self.compute(image_embeds, text_embeds)
            similarities.append(similarity)

        return similarities

    def __repr__(self) -> str:
        """String representation of the similarity computer."""
        return f"SimilarityComputer(metric={self.metric}, normalize={self.normalize})"
