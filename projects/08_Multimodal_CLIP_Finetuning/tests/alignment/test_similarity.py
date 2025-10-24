#!/usr/bin/env python3
"""Tests for similarity computation."""

from __future__ import annotations

import pytest
import torch

from alignment.similarity import SimilarityComputer


class TestSimilarityComputer:
    """Test cases for SimilarityComputer."""

    def test_init_valid_metrics(self) -> None:
        """Test initialization with valid metrics."""
        for metric in ["cosine", "dot", "euclidean"]:
            computer = SimilarityComputer(metric=metric)
            assert computer.metric == metric
            assert computer.normalize is True

    def test_init_invalid_metric(self) -> None:
        """Test initialization with invalid metric."""
        with pytest.raises(ValueError, match="Invalid metric"):
            SimilarityComputer(metric="invalid")

    def test_compute_cosine_similarity(self) -> None:
        """Test cosine similarity computation."""
        computer = SimilarityComputer(metric="cosine", normalize=True)

        # Create embeddings [N, D]
        image_embeds = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        text_embeds = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

        similarity = computer.compute(image_embeds, text_embeds)

        # Expected shape: [3, 2]
        assert similarity.shape == (3, 2)

        # Check cosine similarity values
        # image[0] = [1, 0] should match text[0] = [1, 0] perfectly
        assert pytest.approx(similarity[0, 0].item(), abs=1e-5) == 1.0
        # image[1] = [0, 1] should match text[1] = [0, 1] perfectly
        assert pytest.approx(similarity[1, 1].item(), abs=1e-5) == 1.0
        # image[0] and text[1] should be orthogonal
        assert pytest.approx(similarity[0, 1].item(), abs=1e-5) == 0.0

    def test_compute_dot_product(self) -> None:
        """Test dot product similarity computation."""
        computer = SimilarityComputer(metric="dot", normalize=False)

        image_embeds = torch.tensor([[2.0, 0.0], [0.0, 3.0]])
        text_embeds = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

        similarity = computer.compute(image_embeds, text_embeds)

        # Expected values: [2*1, 2*0], [0*1, 0*0]
        assert similarity.shape == (2, 2)
        assert pytest.approx(similarity[0, 0].item(), abs=1e-5) == 2.0
        assert pytest.approx(similarity[0, 1].item(), abs=1e-5) == 0.0
        assert pytest.approx(similarity[1, 0].item(), abs=1e-5) == 0.0
        assert pytest.approx(similarity[1, 1].item(), abs=1e-5) == 3.0

    def test_compute_euclidean_distance(self) -> None:
        """Test Euclidean distance similarity computation."""
        computer = SimilarityComputer(metric="euclidean", normalize=False)

        image_embeds = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        text_embeds = torch.tensor([[0.0, 0.0], [2.0, 2.0]])

        similarity = computer.compute(image_embeds, text_embeds)

        # Negative Euclidean distance (closer = higher score)
        assert similarity.shape == (2, 2)
        # Distance from [0,0] to [0,0] is 0
        assert pytest.approx(similarity[0, 0].item(), abs=1e-5) == 0.0
        # Distance from [1,1] to [0,0] is sqrt(2)
        assert similarity[1, 0].item() < 0.0  # Negative distance

    def test_compute_with_normalization(self) -> None:
        """Test that normalization is applied."""
        computer = SimilarityComputer(metric="cosine", normalize=True)

        # Non-normalized embeddings
        image_embeds = torch.tensor([[2.0, 0.0]])
        text_embeds = torch.tensor([[4.0, 0.0]])

        similarity = computer.compute(image_embeds, text_embeds)

        # After normalization, both become [1, 0], so cosine similarity = 1
        assert pytest.approx(similarity[0, 0].item(), abs=1e-5) == 1.0

    def test_compute_incompatible_dimensions(self) -> None:
        """Test error on incompatible embedding dimensions."""
        computer = SimilarityComputer(metric="cosine")

        image_embeds = torch.tensor([[1.0, 0.0]])
        text_embeds = torch.tensor([[1.0, 0.0, 0.0]])  # Different dimension

        with pytest.raises(ValueError, match="Embedding dimensions must match"):
            computer.compute(image_embeds, text_embeds)

    def test_compute_invalid_tensor_dims(self) -> None:
        """Test error on non-2D tensors."""
        computer = SimilarityComputer(metric="cosine")

        image_embeds = torch.tensor([1.0, 0.0])  # 1D
        text_embeds = torch.tensor([[1.0, 0.0]])

        with pytest.raises(ValueError, match="Expected 2D tensors"):
            computer.compute(image_embeds, text_embeds)

    def test_top_k(self) -> None:
        """Test top-k selection."""
        computer = SimilarityComputer(metric="cosine")

        # Create similarity matrix
        similarity_matrix = torch.tensor(
            [
                [0.9, 0.2, 0.5, 0.7],
                [0.1, 0.8, 0.3, 0.4],
            ]
        )

        # Get top-2 for each row
        top_k_values, top_k_indices = computer.top_k(similarity_matrix, k=2, dim=1)

        assert top_k_values.shape == (2, 2)
        assert top_k_indices.shape == (2, 2)

        # Row 0: top values should be 0.9, 0.7 (indices 0, 3)
        assert pytest.approx(top_k_values[0, 0].item(), abs=1e-5) == 0.9
        assert pytest.approx(top_k_values[0, 1].item(), abs=1e-5) == 0.7
        assert top_k_indices[0, 0].item() == 0
        assert top_k_indices[0, 1].item() == 3

        # Row 1: top values should be 0.8, 0.4 (indices 1, 3)
        assert pytest.approx(top_k_values[1, 0].item(), abs=1e-5) == 0.8
        assert pytest.approx(top_k_values[1, 1].item(), abs=1e-5) == 0.4
        assert top_k_indices[1, 0].item() == 1
        assert top_k_indices[1, 1].item() == 3

    def test_top_k_invalid_k(self) -> None:
        """Test error on invalid k value."""
        computer = SimilarityComputer(metric="cosine")

        similarity_matrix = torch.tensor([[0.9, 0.2], [0.1, 0.8]])

        with pytest.raises(ValueError, match="k must be in range"):
            computer.top_k(similarity_matrix, k=0)

        with pytest.raises(ValueError, match="k must be in range"):
            computer.top_k(similarity_matrix, k=5)

    def test_top_k_column_wise(self) -> None:
        """Test top-k selection column-wise."""
        computer = SimilarityComputer(metric="cosine")

        similarity_matrix = torch.tensor(
            [
                [0.9, 0.2],
                [0.1, 0.8],
                [0.5, 0.6],
            ]
        )

        # Get top-2 for each column (dim=0)
        top_k_values, top_k_indices = computer.top_k(similarity_matrix, k=2, dim=0)

        assert top_k_values.shape == (2, 2)

        # Column 0: top values should be 0.9, 0.5 (rows 0, 2)
        assert pytest.approx(top_k_values[0, 0].item(), abs=1e-5) == 0.9
        assert pytest.approx(top_k_values[1, 0].item(), abs=1e-5) == 0.5

    def test_batch_compute(self) -> None:
        """Test batch similarity computation."""
        computer = SimilarityComputer(metric="cosine")

        image_embeds_list = [
            torch.tensor([[1.0, 0.0]]),
            torch.tensor([[0.0, 1.0]]),
        ]
        text_embeds_list = [
            torch.tensor([[1.0, 0.0]]),
            torch.tensor([[0.0, 1.0]]),
        ]

        similarities = computer.batch_compute(image_embeds_list, text_embeds_list)

        assert len(similarities) == 2
        # First pair should have high similarity
        assert pytest.approx(similarities[0][0, 0].item(), abs=1e-5) == 1.0
        # Second pair should have high similarity
        assert pytest.approx(similarities[1][0, 0].item(), abs=1e-5) == 1.0

    def test_batch_compute_mismatched_lengths(self) -> None:
        """Test error on mismatched list lengths."""
        computer = SimilarityComputer(metric="cosine")

        image_embeds_list = [torch.tensor([[1.0, 0.0]])]
        text_embeds_list = [torch.tensor([[1.0, 0.0]]), torch.tensor([[0.0, 1.0]])]

        with pytest.raises(ValueError, match="Lists must have same length"):
            computer.batch_compute(image_embeds_list, text_embeds_list)

    def test_repr(self) -> None:
        """Test string representation."""
        computer = SimilarityComputer(metric="cosine", normalize=True)
        repr_str = repr(computer)

        assert "SimilarityComputer" in repr_str
        assert "cosine" in repr_str
        assert "True" in repr_str
