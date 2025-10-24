#!/usr/bin/env python3
"""Tests for image-text retrieval."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch
from PIL import Image

from alignment.retrieval import ImageTextRetrieval
from alignment.similarity import SimilarityComputer


@pytest.fixture
def mock_model():
    """Create mock CLIP model."""
    model = MagicMock()
    model.model = MagicMock()
    model.processor = MagicMock()
    model.config = MagicMock()
    model.config.model_name = "mock-clip"
    return model


@pytest.fixture
def similarity_computer():
    """Create similarity computer."""
    return SimilarityComputer(metric="cosine", normalize=True)


@pytest.fixture
def retrieval_system(mock_model, similarity_computer):
    """Create retrieval system."""
    return ImageTextRetrieval(mock_model, similarity_computer)


class TestImageTextRetrieval:
    """Test cases for ImageTextRetrieval."""

    def test_init_with_uninitialized_model(self, similarity_computer) -> None:
        """Test initialization with uninitialized model."""
        model = MagicMock()
        model.model = None
        model.processor = None

        with pytest.raises(RuntimeError, match="Model not initialized"):
            ImageTextRetrieval(model, similarity_computer)

    def test_retrieve_text(self, retrieval_system, mock_model) -> None:
        """Test text retrieval from image."""
        # Mock image embedding
        mock_model.encode_image.return_value = torch.tensor([[1.0, 0.0]])

        # Mock text embeddings (3 texts)
        mock_model.encode_text.return_value = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]]  # High similarity  # Low similarity  # Medium similarity
        )

        texts = ["text A", "text B", "text C"]

        # Create dummy image
        image = Image.new("RGB", (224, 224))

        # Retrieve top-2
        results = retrieval_system.retrieve_text(image, texts, top_k=2)

        assert len(results) == 2
        # First result should be "text A" with highest similarity
        assert results[0][0] == "text A"
        assert results[0][1] > results[1][1]  # First score > second score

    def test_retrieve_text_with_embedding(self, retrieval_system, mock_model) -> None:
        """Test text retrieval with pre-computed embedding."""
        # Use pre-computed embedding instead of image
        image_embed = torch.tensor([1.0, 0.0])

        # Mock text embeddings
        mock_model.encode_text.return_value = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

        texts = ["text A", "text B"]

        results = retrieval_system.retrieve_text(image_embed, texts, top_k=1)

        assert len(results) == 1
        # Should retrieve "text A" with high similarity
        assert results[0][0] == "text A"

    def test_retrieve_text_empty_texts(self, retrieval_system) -> None:
        """Test error on empty texts."""
        image = Image.new("RGB", (224, 224))

        with pytest.raises(ValueError, match="No texts provided"):
            retrieval_system.retrieve_text(image, [], top_k=5)

    def test_retrieve_text_invalid_k(self, retrieval_system, mock_model) -> None:
        """Test error on invalid k."""
        mock_model.encode_image.return_value = torch.tensor([[1.0, 0.0]])
        mock_model.encode_text.return_value = torch.tensor([[1.0, 0.0]])

        image = Image.new("RGB", (224, 224))
        texts = ["text A"]

        with pytest.raises(ValueError, match="top_k must be in range"):
            retrieval_system.retrieve_text(image, texts, top_k=5)

    def test_retrieve_image(self, retrieval_system, mock_model) -> None:
        """Test image retrieval from text."""
        # Mock text embedding
        mock_model.encode_text.return_value = torch.tensor([[1.0, 0.0]])

        # Mock image embeddings (3 images)
        mock_model.encode_image.return_value = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]]  # High similarity  # Low similarity  # Medium similarity
        )

        images = [Image.new("RGB", (224, 224)) for _ in range(3)]

        # Retrieve top-2
        results = retrieval_system.retrieve_image("query text", images, top_k=2)

        assert len(results) == 2
        # First result should be image index 0 with highest similarity
        assert results[0][0] == 0
        assert results[0][1] > results[1][1]  # First score > second score

    def test_retrieve_image_with_embeddings(self, retrieval_system, mock_model) -> None:
        """Test image retrieval with pre-computed embeddings."""
        # Mock text embedding
        mock_model.encode_text.return_value = torch.tensor([[1.0, 0.0]])

        # Use pre-computed image embeddings
        image_embeds = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

        results = retrieval_system.retrieve_image("query text", image_embeds, top_k=1)

        assert len(results) == 1
        # Should retrieve image index 0
        assert results[0][0] == 0

    def test_retrieve_image_empty_images(self, retrieval_system) -> None:
        """Test error on empty images."""
        with pytest.raises(ValueError, match="No images provided"):
            retrieval_system.retrieve_image("query", [], top_k=5)

    def test_retrieve_image_invalid_k(self, retrieval_system, mock_model) -> None:
        """Test error on invalid k."""
        mock_model.encode_text.return_value = torch.tensor([[1.0, 0.0]])
        mock_model.encode_image.return_value = torch.tensor([[1.0, 0.0]])

        images = [Image.new("RGB", (224, 224))]

        with pytest.raises(ValueError, match="top_k must be in range"):
            retrieval_system.retrieve_image("query", images, top_k=5)

    def test_batch_retrieve_text_to_text(self, retrieval_system, mock_model) -> None:
        """Test batch retrieval (text queries, text candidates)."""
        # Mock text embeddings for queries
        mock_model.encode_text.side_effect = [
            torch.tensor([[1.0, 0.0], [0.0, 1.0]]),  # Query embeddings
            torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]),  # Candidate embeddings
        ]

        queries = ["query 1", "query 2"]
        candidates = ["candidate 1", "candidate 2", "candidate 3"]

        results = retrieval_system.batch_retrieve(queries, candidates, top_k=2)

        assert len(results) == 2
        # Each query should have 2 results
        assert len(results[0]) == 2
        assert len(results[1]) == 2

    def test_batch_retrieve_image_to_text(self, retrieval_system, mock_model) -> None:
        """Test batch retrieval (image queries, text candidates)."""
        # Mock image embeddings for queries
        mock_model.encode_image.return_value = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

        # Mock text embeddings for candidates
        mock_model.encode_text.return_value = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

        queries = [Image.new("RGB", (224, 224)) for _ in range(2)]
        candidates = ["candidate 1", "candidate 2"]

        results = retrieval_system.batch_retrieve(queries, candidates, top_k=1)

        assert len(results) == 2
        assert len(results[0]) == 1
        assert len(results[1]) == 1

    def test_batch_retrieve_empty_queries(self, retrieval_system) -> None:
        """Test error on empty queries."""
        with pytest.raises(ValueError, match="must not be empty"):
            retrieval_system.batch_retrieve([], ["candidate"], top_k=1)

    def test_batch_retrieve_empty_candidates(self, retrieval_system) -> None:
        """Test error on empty candidates."""
        with pytest.raises(ValueError, match="must not be empty"):
            retrieval_system.batch_retrieve(["query"], [], top_k=1)

    def test_batch_retrieve_invalid_k(self, retrieval_system, mock_model) -> None:
        """Test error on invalid k."""
        mock_model.encode_text.side_effect = [
            torch.tensor([[1.0, 0.0]]),
            torch.tensor([[1.0, 0.0]]),
        ]

        with pytest.raises(ValueError, match="top_k must be in range"):
            retrieval_system.batch_retrieve(["query"], ["candidate"], top_k=5)

    def test_retrieve_mutual_nn(self, retrieval_system, mock_model) -> None:
        """Test mutual nearest neighbor retrieval."""
        # Mock image embeddings
        mock_model.encode_image.return_value = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]
        )

        # Mock text embeddings
        mock_model.encode_text.return_value = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [0.3, 0.7]]
        )

        images = [Image.new("RGB", (224, 224)) for _ in range(3)]
        texts = ["text 1", "text 2", "text 3"]

        mutual_nn = retrieval_system.retrieve_mutual_nn(images, texts)

        # Should find at least some mutual nearest neighbors
        assert isinstance(mutual_nn, list)
        # Each result should be (image_idx, text_idx, similarity)
        for result in mutual_nn:
            assert len(result) == 3
            assert isinstance(result[0], int)
            assert isinstance(result[1], int)
            assert isinstance(result[2], float)

    def test_retrieve_mutual_nn_empty_inputs(self, retrieval_system) -> None:
        """Test error on empty inputs for mutual NN."""
        with pytest.raises(ValueError, match="must not be empty"):
            retrieval_system.retrieve_mutual_nn([], ["text"])

        with pytest.raises(ValueError, match="must not be empty"):
            retrieval_system.retrieve_mutual_nn([Image.new("RGB", (224, 224))], [])

    def test_repr(self, retrieval_system) -> None:
        """Test string representation."""
        repr_str = repr(retrieval_system)

        assert "ImageTextRetrieval" in repr_str
        assert "mock-clip" in repr_str
