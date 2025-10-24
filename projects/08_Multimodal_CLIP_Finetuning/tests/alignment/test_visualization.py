#!/usr/bin/env python3
"""Tests for alignment visualization."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from PIL import Image

from alignment.visualization import AlignmentVisualizer


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create temporary output directory."""
    return tmp_path / "visualizations"


@pytest.fixture
def visualizer(temp_output_dir):
    """Create visualizer instance."""
    return AlignmentVisualizer(temp_output_dir)


class TestAlignmentVisualizer:
    """Test cases for AlignmentVisualizer."""

    def test_init(self, temp_output_dir) -> None:
        """Test initialization."""
        visualizer = AlignmentVisualizer(temp_output_dir)

        assert visualizer.output_dir == temp_output_dir
        assert temp_output_dir.exists()

    def test_init_creates_directory(self, tmp_path) -> None:
        """Test that initialization creates output directory."""
        output_dir = tmp_path / "new_dir"
        assert not output_dir.exists()

        visualizer = AlignmentVisualizer(output_dir)

        assert output_dir.exists()
        assert visualizer.output_dir == output_dir

    @patch("alignment.visualization.plt")
    @patch("alignment.visualization.sns")
    def test_plot_similarity_matrix(self, mock_sns, mock_plt, visualizer, temp_output_dir) -> None:
        """Test similarity matrix plotting."""
        # Create mock figure and axes
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        similarity_matrix = torch.rand(3, 3)
        image_labels = ["img1", "img2", "img3"]
        text_labels = ["txt1", "txt2", "txt3"]

        visualizer.plot_similarity_matrix(
            similarity_matrix,
            image_labels=image_labels,
            text_labels=text_labels,
        )

        # Verify heatmap was created
        mock_sns.heatmap.assert_called_once()

        # Verify figure was saved
        mock_fig.savefig.assert_called_once()

        # Verify figure was closed
        mock_plt.close.assert_called_once()

    @patch("alignment.visualization.plt")
    @patch("alignment.visualization.sns")
    def test_plot_similarity_matrix_default_labels(self, mock_sns, mock_plt, visualizer) -> None:
        """Test similarity matrix with default labels."""
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        similarity_matrix = torch.rand(2, 2)

        visualizer.plot_similarity_matrix(similarity_matrix)

        # Should create default labels
        mock_sns.heatmap.assert_called_once()

    @patch("alignment.visualization.plt")
    @patch("alignment.visualization.sns")
    def test_plot_similarity_matrix_custom_save_path(self, mock_sns, mock_plt, visualizer, temp_output_dir) -> None:
        """Test similarity matrix with custom save path."""
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        similarity_matrix = torch.rand(2, 2)
        custom_path = temp_output_dir / "custom.png"

        visualizer.plot_similarity_matrix(similarity_matrix, save_path=custom_path)

        # Verify saved to custom path
        call_args = mock_fig.savefig.call_args
        assert Path(call_args[0][0]) == custom_path

    @patch("alignment.visualization.plt")
    def test_plot_similarity_matrix_invalid_dims(self, mock_plt, visualizer) -> None:
        """Test error on non-2D similarity matrix."""
        invalid = torch.rand(5)

        with pytest.raises(ValueError, match="Expected 2D similarity matrix"):
            visualizer.plot_similarity_matrix(invalid)

    @patch("alignment.visualization.plt")
    def test_plot_retrieval_results_with_image(self, mock_plt, visualizer) -> None:
        """Test retrieval results plotting with image."""
        mock_fig = MagicMock()
        mock_gs = MagicMock()
        mock_ax_img = MagicMock()
        mock_ax_text = MagicMock()

        mock_fig.add_gridspec.return_value = mock_gs
        mock_fig.add_subplot.side_effect = [mock_ax_img, mock_ax_text]
        mock_plt.figure.return_value = mock_fig

        query_image = Image.new("RGB", (224, 224))
        retrieved_texts = [
            ("text A", 0.9),
            ("text B", 0.7),
            ("text C", 0.5),
        ]

        visualizer.plot_retrieval_results(query_image, retrieved_texts)

        # Verify image was displayed
        mock_ax_img.imshow.assert_called_once()

        # Verify bar chart was created
        mock_ax_text.barh.assert_called_once()

        # Verify figure was saved
        mock_fig.savefig.assert_called_once()

    @patch("alignment.visualization.plt")
    def test_plot_retrieval_results_without_image(self, mock_plt, visualizer) -> None:
        """Test retrieval results plotting without image."""
        mock_fig = MagicMock()
        mock_ax = MagicMock()

        mock_fig.add_subplot.return_value = mock_ax
        mock_plt.figure.return_value = mock_fig

        retrieved_texts = [
            ("text A", 0.9),
            ("text B", 0.7),
        ]

        visualizer.plot_retrieval_results(None, retrieved_texts)

        # Verify bar chart was created
        mock_ax.barh.assert_called_once()

        # Verify figure was saved
        mock_fig.savefig.assert_called_once()

    @patch("alignment.visualization.plt")
    def test_plot_retrieval_results_empty_texts(self, mock_plt, visualizer) -> None:
        """Test error on empty retrieved texts."""
        with pytest.raises(ValueError, match="No retrieved texts provided"):
            visualizer.plot_retrieval_results(None, [])

    @patch("alignment.visualization.plt")
    def test_plot_alignment_distribution(self, mock_plt, visualizer) -> None:
        """Test alignment distribution plotting."""
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        similarities = torch.rand(100)

        visualizer.plot_alignment_distribution(similarities)

        # Verify histogram was created
        mock_ax.hist.assert_called_once()

        # Verify vertical lines for mean and median
        assert mock_ax.axvline.call_count == 2

        # Verify figure was saved
        mock_fig.savefig.assert_called_once()

    @patch("alignment.visualization.plt")
    def test_plot_alignment_distribution_custom_bins(self, mock_plt, visualizer) -> None:
        """Test alignment distribution with custom bins."""
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        similarities = torch.rand(100)

        visualizer.plot_alignment_distribution(similarities, bins=30)

        # Verify histogram was created with custom bins
        call_args = mock_ax.hist.call_args
        assert call_args[1]["bins"] == 30

    @patch("alignment.visualization.plt")
    def test_plot_alignment_distribution_empty(self, mock_plt, visualizer) -> None:
        """Test error on empty similarities."""
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        empty = torch.tensor([])

        with pytest.raises(ValueError, match="No similarities provided"):
            visualizer.plot_alignment_distribution(empty)

    @patch("alignment.visualization.plt")
    def test_plot_recall_curves(self, mock_plt, visualizer) -> None:
        """Test recall curves plotting."""
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        recall_metrics = {
            "i2t": [0.5, 0.7, 0.9],
            "t2i": [0.4, 0.6, 0.8],
        }
        k_values = [1, 5, 10]

        visualizer.plot_recall_curves(recall_metrics, k_values)

        # Verify two plots were created (i2t and t2i)
        assert mock_ax.plot.call_count == 2

        # Verify figure was saved
        mock_fig.savefig.assert_called_once()

    @patch("alignment.visualization.plt")
    def test_plot_recall_curves_missing_keys(self, mock_plt, visualizer) -> None:
        """Test error on missing required keys."""
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        invalid_metrics = {"i2t": [0.5, 0.7]}  # Missing 't2i'
        k_values = [1, 5]

        with pytest.raises(ValueError, match="must contain 'i2t' and 't2i' keys"):
            visualizer.plot_recall_curves(invalid_metrics, k_values)

    @patch("alignment.visualization.plt")
    def test_plot_recall_curves_mismatched_lengths(self, mock_plt, visualizer) -> None:
        """Test error on mismatched lengths."""
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        recall_metrics = {
            "i2t": [0.5, 0.7],
            "t2i": [0.4, 0.6, 0.8],  # Different length
        }
        k_values = [1, 5]

        with pytest.raises(ValueError, match="must match k_values"):
            visualizer.plot_recall_curves(recall_metrics, k_values)

    def test_repr(self, visualizer, temp_output_dir) -> None:
        """Test string representation."""
        repr_str = repr(visualizer)

        assert "AlignmentVisualizer" in repr_str
        assert str(temp_output_dir) in repr_str

    @patch("alignment.visualization.plt")
    @patch("alignment.visualization.sns")
    def test_plot_similarity_matrix_truncates_long_labels(self, mock_sns, mock_plt, visualizer) -> None:
        """Test that long labels are truncated."""
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        similarity_matrix = torch.rand(2, 2)
        long_label = "a" * 50  # Very long label
        image_labels = [long_label, "short"]
        text_labels = [long_label, "short"]

        visualizer.plot_similarity_matrix(
            similarity_matrix,
            image_labels=image_labels,
            text_labels=text_labels,
        )

        # Verify heatmap was called (labels will be truncated internally)
        mock_sns.heatmap.assert_called_once()

    @patch("alignment.visualization.plt")
    def test_plot_retrieval_results_truncates_long_texts(self, mock_plt, visualizer) -> None:
        """Test that long texts are truncated."""
        mock_fig = MagicMock()
        mock_ax = MagicMock()

        mock_fig.add_subplot.return_value = mock_ax
        mock_plt.figure.return_value = mock_fig

        long_text = "a" * 100
        retrieved_texts = [(long_text, 0.9)]

        visualizer.plot_retrieval_results(None, retrieved_texts)

        # Verify bar chart was created (text will be truncated internally)
        mock_ax.barh.assert_called_once()
