#!/usr/bin/env python3
"""
Alignment quality metrics.

Provides standard evaluation metrics for image-text alignment:
- Recall@K: Proportion of correct matches in top-K results
- Mean Reciprocal Rank (MRR): Average of reciprocal ranks of correct matches
- Median Rank: Median rank of correct matches
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


class AlignmentMetrics:
    """Compute alignment quality metrics.

    Computes standard retrieval metrics for image-text alignment evaluation:
    - Recall@K (bidirectional)
    - Mean Reciprocal Rank (MRR)
    - Median Rank

    All metrics are computed assuming diagonal ground truth (i.e., image[i] matches text[i]).
    """

    @staticmethod
    def recall_at_k(
        similarity_matrix: torch.Tensor,
        k: int = 5,
    ) -> dict[str, float]:
        """Compute Recall@K for both directions.

        Assumes diagonal ground truth: image[i] matches text[i].

        Args:
            similarity_matrix: Similarity matrix [N_images, N_texts]
            k: Number of top results to consider

        Returns:
            Dictionary with keys:
            - i2t_R@k: Image-to-text recall at k
            - t2i_R@k: Text-to-image recall at k

        Raises:
            ValueError: If similarity matrix is not 2D or k is invalid
        """
        if similarity_matrix.dim() != 2:
            raise ValueError(
                f"Expected 2D similarity matrix, got {similarity_matrix.dim()}D"
            )

        n_images, n_texts = similarity_matrix.shape

        if n_images != n_texts:
            raise ValueError(
                f"Recall@K requires square similarity matrix for diagonal ground truth, "
                f"got shape {similarity_matrix.shape}"
            )

        max_k = min(n_images, n_texts)
        if k <= 0 or k > max_k:
            raise ValueError(f"k must be in range [1, {max_k}], got {k}")

        # Image-to-text recall@k
        # For each image, check if correct text is in top-k
        i2t_top_k_indices = torch.topk(similarity_matrix, k=k, dim=1, largest=True).indices
        i2t_correct = torch.arange(n_images, device=similarity_matrix.device).unsqueeze(1)
        i2t_recall = (i2t_top_k_indices == i2t_correct).any(dim=1).float().mean().item()

        # Text-to-image recall@k
        # For each text, check if correct image is in top-k
        t2i_top_k_indices = torch.topk(similarity_matrix.T, k=k, dim=1, largest=True).indices
        t2i_correct = torch.arange(n_texts, device=similarity_matrix.device).unsqueeze(1)
        t2i_recall = (t2i_top_k_indices == t2i_correct).any(dim=1).float().mean().item()

        return {
            f"i2t_R@{k}": i2t_recall,
            f"t2i_R@{k}": t2i_recall,
        }

    @staticmethod
    def mean_reciprocal_rank(
        similarity_matrix: torch.Tensor,
    ) -> dict[str, float]:
        """Compute Mean Reciprocal Rank (MRR) for both directions.

        Assumes diagonal ground truth: image[i] matches text[i].

        Args:
            similarity_matrix: Similarity matrix [N_images, N_texts]

        Returns:
            Dictionary with keys:
            - i2t_MRR: Image-to-text MRR
            - t2i_MRR: Text-to-image MRR

        Raises:
            ValueError: If similarity matrix is not 2D or not square
        """
        if similarity_matrix.dim() != 2:
            raise ValueError(
                f"Expected 2D similarity matrix, got {similarity_matrix.dim()}D"
            )

        n_images, n_texts = similarity_matrix.shape

        if n_images != n_texts:
            raise ValueError(
                f"MRR requires square similarity matrix for diagonal ground truth, "
                f"got shape {similarity_matrix.shape}"
            )

        # Image-to-text MRR
        # For each image, find rank of correct text
        i2t_sorted_indices = torch.argsort(similarity_matrix, dim=1, descending=True)
        i2t_correct = torch.arange(n_images, device=similarity_matrix.device)
        i2t_ranks = (i2t_sorted_indices == i2t_correct.unsqueeze(1)).nonzero(as_tuple=True)[1]
        i2t_mrr = (1.0 / (i2t_ranks.float() + 1.0)).mean().item()

        # Text-to-image MRR
        # For each text, find rank of correct image
        t2i_sorted_indices = torch.argsort(similarity_matrix.T, dim=1, descending=True)
        t2i_correct = torch.arange(n_texts, device=similarity_matrix.device)
        t2i_ranks = (t2i_sorted_indices == t2i_correct.unsqueeze(1)).nonzero(as_tuple=True)[1]
        t2i_mrr = (1.0 / (t2i_ranks.float() + 1.0)).mean().item()

        return {
            "i2t_MRR": i2t_mrr,
            "t2i_MRR": t2i_mrr,
        }

    @staticmethod
    def median_rank(
        similarity_matrix: torch.Tensor,
    ) -> dict[str, float]:
        """Compute median rank for both directions.

        Assumes diagonal ground truth: image[i] matches text[i].

        Args:
            similarity_matrix: Similarity matrix [N_images, N_texts]

        Returns:
            Dictionary with keys:
            - i2t_median_rank: Image-to-text median rank
            - t2i_median_rank: Text-to-image median rank

        Raises:
            ValueError: If similarity matrix is not 2D or not square
        """
        if similarity_matrix.dim() != 2:
            raise ValueError(
                f"Expected 2D similarity matrix, got {similarity_matrix.dim()}D"
            )

        n_images, n_texts = similarity_matrix.shape

        if n_images != n_texts:
            raise ValueError(
                f"Median rank requires square similarity matrix for diagonal ground truth, "
                f"got shape {similarity_matrix.shape}"
            )

        # Image-to-text median rank
        # For each image, find rank of correct text
        i2t_sorted_indices = torch.argsort(similarity_matrix, dim=1, descending=True)
        i2t_correct = torch.arange(n_images, device=similarity_matrix.device)
        i2t_ranks = (i2t_sorted_indices == i2t_correct.unsqueeze(1)).nonzero(as_tuple=True)[1]
        i2t_median = torch.median(i2t_ranks.float()).item() + 1.0  # +1 for 1-indexed

        # Text-to-image median rank
        # For each text, find rank of correct image
        t2i_sorted_indices = torch.argsort(similarity_matrix.T, dim=1, descending=True)
        t2i_correct = torch.arange(n_texts, device=similarity_matrix.device)
        t2i_ranks = (t2i_sorted_indices == t2i_correct.unsqueeze(1)).nonzero(as_tuple=True)[1]
        t2i_median = torch.median(t2i_ranks.float()).item() + 1.0  # +1 for 1-indexed

        return {
            "i2t_median_rank": i2t_median,
            "t2i_median_rank": t2i_median,
        }

    def compute_all(
        self,
        similarity_matrix: torch.Tensor,
        k_values: list[int] | None = None,
    ) -> dict[str, float]:
        """Compute all metrics at multiple k values.

        Args:
            similarity_matrix: Similarity matrix [N_images, N_texts]
            k_values: List of k values for Recall@K (default: [1, 5, 10])

        Returns:
            Dictionary containing all metrics

        Raises:
            ValueError: If similarity matrix is invalid or k_values are invalid
        """
        if k_values is None:
            k_values = [1, 5, 10]

        if not k_values:
            raise ValueError("k_values cannot be empty")

        # Validate k values
        max_k = min(similarity_matrix.shape)
        for k in k_values:
            if k <= 0 or k > max_k:
                raise ValueError(
                    f"Invalid k={k}. Must be in range [1, {max_k}]"
                )

        metrics = {}

        # Compute Recall@K for each k
        for k in k_values:
            recall_metrics = self.recall_at_k(similarity_matrix, k=k)
            metrics.update(recall_metrics)

        # Compute MRR
        mrr_metrics = self.mean_reciprocal_rank(similarity_matrix)
        metrics.update(mrr_metrics)

        # Compute median rank
        median_metrics = self.median_rank(similarity_matrix)
        metrics.update(median_metrics)

        return metrics

    @staticmethod
    def compute_alignment_score(
        similarity_matrix: torch.Tensor,
    ) -> float:
        """Compute overall alignment score.

        Simple metric: average of diagonal elements (assuming diagonal ground truth).

        Args:
            similarity_matrix: Similarity matrix [N, N]

        Returns:
            Average diagonal similarity

        Raises:
            ValueError: If matrix is not square
        """
        if similarity_matrix.dim() != 2:
            raise ValueError(
                f"Expected 2D similarity matrix, got {similarity_matrix.dim()}D"
            )

        n_images, n_texts = similarity_matrix.shape

        if n_images != n_texts:
            raise ValueError(
                f"Alignment score requires square similarity matrix, "
                f"got shape {similarity_matrix.shape}"
            )

        # Return average of diagonal (correct pairs)
        diagonal = torch.diagonal(similarity_matrix)
        return diagonal.mean().item()

    def __repr__(self) -> str:
        """String representation of metrics computer."""
        return "AlignmentMetrics()"
