#!/usr/bin/env python3
"""
Alignment visualization utilities.

Provides visualization tools for image-text alignments:
- Similarity matrix heatmaps
- Retrieval results displays
- Alignment score distributions
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

if TYPE_CHECKING:
    from PIL import Image

logger = logging.getLogger(__name__)


class AlignmentVisualizer:
    """Visualize image-text alignments.

    Provides methods for creating visualizations of:
    - Similarity matrices (heatmaps)
    - Retrieval results (image with top-k texts)
    - Alignment score distributions (histograms)

    Attributes:
        output_dir: Directory for saving visualizations
    """

    def __init__(self, output_dir: Path) -> None:
        """Initialize alignment visualizer.

        Args:
            output_dir: Directory for saving visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set style
        sns.set_style("whitegrid")

        logger.info(f"Initialized AlignmentVisualizer with output_dir={output_dir}")

    def plot_similarity_matrix(
        self,
        similarity_matrix: torch.Tensor,
        image_labels: list[str] | None = None,
        text_labels: list[str] | None = None,
        save_path: Path | None = None,
        title: str = "Image-Text Similarity Matrix",
        figsize: tuple[int, int] = (12, 10),
    ) -> None:
        """Create heatmap of similarity matrix.

        Args:
            similarity_matrix: Similarity matrix [N_images, N_texts]
            image_labels: Labels for images (row labels)
            text_labels: Labels for texts (column labels)
            save_path: Path to save figure (if None, uses default)
            title: Plot title
            figsize: Figure size (width, height)

        Raises:
            ValueError: If similarity matrix is not 2D
        """
        if similarity_matrix.dim() != 2:
            raise ValueError(
                f"Expected 2D similarity matrix, got {similarity_matrix.dim()}D"
            )

        n_images, n_texts = similarity_matrix.shape

        # Convert to numpy
        sim_np = similarity_matrix.detach().cpu().numpy()

        # Create default labels if not provided
        if image_labels is None:
            image_labels = [f"Image {i}" for i in range(n_images)]
        if text_labels is None:
            text_labels = [f"Text {i}" for i in range(n_texts)]

        # Truncate long labels
        image_labels = [label[:30] + "..." if len(label) > 30 else label for label in image_labels]
        text_labels = [label[:30] + "..." if len(label) > 30 else label for label in text_labels]

        # Create heatmap
        fig, ax = plt.subplots(figsize=figsize)

        sns.heatmap(
            sim_np,
            xticklabels=text_labels,
            yticklabels=image_labels,
            cmap="RdYlGn",
            center=0.0,
            annot=n_images <= 10 and n_texts <= 10,  # Only annotate for small matrices
            fmt=".2f",
            cbar_kws={"label": "Similarity"},
            ax=ax,
        )

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Text", fontsize=12)
        ax.set_ylabel("Image", fontsize=12)

        plt.tight_layout()

        # Save figure
        if save_path is None:
            save_path = self.output_dir / "similarity_matrix.png"
        else:
            save_path = Path(save_path)

        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Saved similarity matrix plot to {save_path}")

    def plot_retrieval_results(
        self,
        query_image: Image.Image | None,
        retrieved_texts: list[tuple[str, float]],
        save_path: Path | None = None,
        title: str = "Image-to-Text Retrieval Results",
        figsize: tuple[int, int] = (10, 8),
    ) -> None:
        """Visualize top-k retrieved texts for an image.

        Args:
            query_image: Query image (optional, if None just shows text results)
            retrieved_texts: List of (text, score) tuples
            save_path: Path to save figure (if None, uses default)
            title: Plot title
            figsize: Figure size (width, height)
        """
        if not retrieved_texts:
            raise ValueError("No retrieved texts provided")

        fig = plt.figure(figsize=figsize)

        if query_image is not None:
            # Create grid: image on left, text results on right
            gs = fig.add_gridspec(1, 2, width_ratios=[1, 2])
            ax_img = fig.add_subplot(gs[0])
            ax_text = fig.add_subplot(gs[1])

            # Show image
            ax_img.imshow(query_image)
            ax_img.axis("off")
            ax_img.set_title("Query Image", fontsize=12, fontweight="bold")
        else:
            # Just show text results
            ax_text = fig.add_subplot(111)

        # Plot text results as horizontal bar chart
        texts = [text[:50] + "..." if len(text) > 50 else text for text, _ in retrieved_texts]
        scores = [score for _, score in retrieved_texts]

        y_pos = np.arange(len(texts))
        ax_text.barh(y_pos, scores, color="skyblue", edgecolor="navy")
        ax_text.set_yticks(y_pos)
        ax_text.set_yticklabels(texts, fontsize=10)
        ax_text.invert_yaxis()  # Top result at top
        ax_text.set_xlabel("Similarity Score", fontsize=12)
        ax_text.set_title(title, fontsize=12, fontweight="bold")
        ax_text.grid(axis="x", alpha=0.3)

        # Add score labels on bars
        for i, score in enumerate(scores):
            ax_text.text(score + 0.01, i, f"{score:.3f}", va="center", fontsize=9)

        plt.tight_layout()

        # Save figure
        if save_path is None:
            save_path = self.output_dir / "retrieval_results.png"
        else:
            save_path = Path(save_path)

        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Saved retrieval results plot to {save_path}")

    def plot_alignment_distribution(
        self,
        similarities: torch.Tensor,
        save_path: Path | None = None,
        title: str = "Alignment Score Distribution",
        figsize: tuple[int, int] = (10, 6),
        bins: int = 50,
    ) -> None:
        """Plot distribution of alignment scores.

        Args:
            similarities: Tensor of similarity scores (any shape, will be flattened)
            save_path: Path to save figure (if None, uses default)
            title: Plot title
            figsize: Figure size (width, height)
            bins: Number of histogram bins
        """
        # Flatten and convert to numpy
        sim_np = similarities.detach().cpu().flatten().numpy()

        if len(sim_np) == 0:
            raise ValueError("No similarities provided")

        fig, ax = plt.subplots(figsize=figsize)

        # Create histogram
        ax.hist(sim_np, bins=bins, color="skyblue", edgecolor="navy", alpha=0.7)

        # Add statistics
        mean_sim = np.mean(sim_np)
        median_sim = np.median(sim_np)
        std_sim = np.std(sim_np)

        ax.axvline(mean_sim, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean_sim:.3f}")
        ax.axvline(median_sim, color="green", linestyle="--", linewidth=2, label=f"Median: {median_sim:.3f}")

        ax.set_xlabel("Similarity Score", fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)

        # Add statistics text box
        stats_text = f"Mean: {mean_sim:.3f}\nMedian: {median_sim:.3f}\nStd: {std_sim:.3f}"
        ax.text(
            0.95,
            0.95,
            stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        plt.tight_layout()

        # Save figure
        if save_path is None:
            save_path = self.output_dir / "alignment_distribution.png"
        else:
            save_path = Path(save_path)

        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Saved alignment distribution plot to {save_path}")

    def plot_recall_curves(
        self,
        recall_metrics: dict[str, list[float]],
        k_values: list[int],
        save_path: Path | None = None,
        title: str = "Recall@K Curves",
        figsize: tuple[int, int] = (10, 6),
    ) -> None:
        """Plot Recall@K curves for image-to-text and text-to-image retrieval.

        Args:
            recall_metrics: Dictionary with keys 'i2t' and 't2i', values are lists of recall scores
            k_values: List of k values corresponding to recall scores
            save_path: Path to save figure (if None, uses default)
            title: Plot title
            figsize: Figure size (width, height)

        Raises:
            ValueError: If recall_metrics is missing required keys
        """
        if "i2t" not in recall_metrics or "t2i" not in recall_metrics:
            raise ValueError("recall_metrics must contain 'i2t' and 't2i' keys")

        if len(recall_metrics["i2t"]) != len(k_values):
            raise ValueError("Number of i2t recall values must match k_values")

        if len(recall_metrics["t2i"]) != len(k_values):
            raise ValueError("Number of t2i recall values must match k_values")

        fig, ax = plt.subplots(figsize=figsize)

        # Plot i2t recall
        ax.plot(k_values, recall_metrics["i2t"], marker="o", linewidth=2, label="Image-to-Text", color="blue")

        # Plot t2i recall
        ax.plot(k_values, recall_metrics["t2i"], marker="s", linewidth=2, label="Text-to-Image", color="green")

        ax.set_xlabel("K", fontsize=12)
        ax.set_ylabel("Recall@K", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_ylim([0, 1.05])

        plt.tight_layout()

        # Save figure
        if save_path is None:
            save_path = self.output_dir / "recall_curves.png"
        else:
            save_path = Path(save_path)

        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Saved recall curves plot to {save_path}")

    def __repr__(self) -> str:
        """String representation of visualizer."""
        return f"AlignmentVisualizer(output_dir={self.output_dir})"
