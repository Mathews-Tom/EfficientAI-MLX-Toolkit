"""Tests for task embedding modules.

Tests TaskEmbeddingNetwork, Task2VecEmbedding, TaskSimilarityMetric,
and TaskFeatureExtractor from learned_embeddings.py.
"""

from __future__ import annotations

import pytest
import mlx.core as mx
import mlx.nn as nn

from task_embedding.learned_embeddings import (
    TaskEmbeddingNetwork,
    Task2VecEmbedding,
    TaskSimilarityMetric,
    TaskFeatureExtractor,
)
from meta_learning.models import SimpleClassifier


class TestTaskEmbeddingNetwork:
    """Test TaskEmbeddingNetwork module."""

    def test_initialization(self):
        """Test TaskEmbeddingNetwork initialization."""
        embedder = TaskEmbeddingNetwork(
            input_dim=128,
            hidden_dim=256,
            embedding_dim=64,
            num_layers=3,
        )

        assert embedder.input_dim == 128
        assert embedder.hidden_dim == 256
        assert embedder.embedding_dim == 64
        assert embedder.num_layers == 3

    def test_forward_pass(self):
        """Test forward pass produces correct embedding shape."""
        embedder = TaskEmbeddingNetwork(
            input_dim=128,
            hidden_dim=256,
            embedding_dim=64,
        )

        # Test with single input
        task_features = mx.random.normal((128,))
        embedding = embedder(task_features)

        assert embedding.shape == (64,)

    def test_batch_forward_pass(self):
        """Test forward pass with batch of inputs."""
        embedder = TaskEmbeddingNetwork(
            input_dim=128,
            hidden_dim=256,
            embedding_dim=64,
        )

        # Test with batch input
        task_features = mx.random.normal((10, 128))
        embeddings = embedder(task_features)

        assert embeddings.shape == (10, 64)

    def test_embedding_consistency(self):
        """Test that same input produces same embedding."""
        embedder = TaskEmbeddingNetwork(
            input_dim=128,
            hidden_dim=256,
            embedding_dim=64,
        )

        task_features = mx.random.normal((128,))
        embedding1 = embedder(task_features)
        embedding2 = embedder(task_features)

        # Should be identical (no dropout in eval mode)
        assert mx.allclose(embedding1, embedding2)

    def test_different_inputs_different_embeddings(self):
        """Test different inputs produce different embeddings."""
        embedder = TaskEmbeddingNetwork(
            input_dim=128,
            hidden_dim=256,
            embedding_dim=64,
        )

        features1 = mx.random.normal((128,))
        features2 = mx.random.normal((128,))

        embedding1 = embedder(features1)
        embedding2 = embedder(features2)

        # Should be different
        assert not mx.allclose(embedding1, embedding2)


class TestTask2VecEmbedding:
    """Test Task2Vec embedding based on Fisher Information Matrix."""

    def test_initialization(self):
        """Test Task2Vec initialization."""
        model = SimpleClassifier(input_dim=10, hidden_dim=32, num_classes=5)

        task2vec = Task2VecEmbedding(
            model=model,
            embedding_method="fisher_diagonal",
        )

        assert task2vec.model == model
        assert task2vec.embedding_method == "fisher_diagonal"

    def test_compute_fisher_diagonal(self):
        """Test Fisher Information Matrix diagonal computation."""
        model = SimpleClassifier(input_dim=10, hidden_dim=32, num_classes=5)
        task2vec = Task2VecEmbedding(model)

        # Create synthetic task data
        support_x = mx.random.normal((20, 10))
        support_y = mx.random.randint(0, 5, (20,))

        fisher_diag = task2vec.compute_fisher_diagonal(support_x, support_y)

        # Verify it's a dictionary with all model parameters
        assert isinstance(fisher_diag, dict)
        assert len(fisher_diag) > 0

        # All values should be non-negative (squared gradients)
        for key, value in fisher_diag.items():
            assert mx.all(value >= 0)

    def test_embedding_from_task(self):
        """Test creating embedding from task data."""
        model = SimpleClassifier(input_dim=10, hidden_dim=32, num_classes=5)
        task2vec = Task2VecEmbedding(model)

        # Create task data
        support_x = mx.random.normal((20, 10))
        support_y = mx.random.randint(0, 5, (20,))

        embedding = task2vec(support_x, support_y)

        # Should be a vector
        assert len(embedding.shape) == 1
        assert embedding.shape[0] > 0

    def test_different_tasks_different_embeddings(self):
        """Test different tasks produce different Task2Vec embeddings."""
        model = SimpleClassifier(input_dim=10, hidden_dim=32, num_classes=5)
        task2vec = Task2VecEmbedding(model)

        # Task 1
        task1_x = mx.random.normal((20, 10))
        task1_y = mx.random.randint(0, 5, (20,))
        embedding1 = task2vec(task1_x, task1_y)

        # Task 2 (different distribution)
        task2_x = mx.random.normal((20, 10)) + 2.0
        task2_y = mx.random.randint(0, 5, (20,))
        embedding2 = task2vec(task2_x, task2_y)

        # Should be different
        assert not mx.allclose(embedding1, embedding2)


class TestTaskSimilarityMetric:
    """Test task similarity metrics."""

    def test_cosine_similarity(self):
        """Test cosine similarity computation."""
        # Identical embeddings
        emb1 = mx.array([1.0, 0.0, 0.0])
        emb2 = mx.array([1.0, 0.0, 0.0])
        sim = TaskSimilarityMetric.cosine_similarity(emb1, emb2)
        assert mx.isclose(sim, 1.0)

        # Orthogonal embeddings
        emb1 = mx.array([1.0, 0.0])
        emb2 = mx.array([0.0, 1.0])
        sim = TaskSimilarityMetric.cosine_similarity(emb1, emb2)
        assert mx.isclose(sim, 0.0, atol=1e-6)

        # Opposite embeddings
        emb1 = mx.array([1.0, 0.0])
        emb2 = mx.array([-1.0, 0.0])
        sim = TaskSimilarityMetric.cosine_similarity(emb1, emb2)
        assert mx.isclose(sim, -1.0)

    def test_euclidean_distance(self):
        """Test Euclidean distance computation."""
        # Identical embeddings
        emb1 = mx.array([1.0, 2.0, 3.0])
        emb2 = mx.array([1.0, 2.0, 3.0])
        dist = TaskSimilarityMetric.euclidean_distance(emb1, emb2)
        assert mx.isclose(dist, 0.0, atol=1e-6)

        # Known distance
        emb1 = mx.array([0.0, 0.0])
        emb2 = mx.array([3.0, 4.0])
        dist = TaskSimilarityMetric.euclidean_distance(emb1, emb2)
        assert mx.isclose(dist, 5.0)

    def test_manhattan_distance(self):
        """Test Manhattan distance computation."""
        # Identical embeddings
        emb1 = mx.array([1.0, 2.0, 3.0])
        emb2 = mx.array([1.0, 2.0, 3.0])
        dist = TaskSimilarityMetric.manhattan_distance(emb1, emb2)
        assert mx.isclose(dist, 0.0, atol=1e-6)

        # Known distance
        emb1 = mx.array([0.0, 0.0])
        emb2 = mx.array([3.0, 4.0])
        dist = TaskSimilarityMetric.manhattan_distance(emb1, emb2)
        assert mx.isclose(dist, 7.0)

    def test_similarity_matrix(self):
        """Test computing similarity matrix."""
        embeddings = [
            mx.array([1.0, 0.0]),
            mx.array([0.0, 1.0]),
            mx.array([1.0, 1.0]),
        ]

        sim_matrix = TaskSimilarityMetric.compute_similarity_matrix(
            embeddings, metric="cosine"
        )

        # Should be 3x3
        assert sim_matrix.shape == (3, 3)

        # Diagonal should be 1 (self-similarity)
        assert mx.allclose(mx.diag(sim_matrix), mx.ones(3))

        # Matrix should be symmetric
        assert mx.allclose(sim_matrix, sim_matrix.T)

    def test_find_most_similar_tasks(self):
        """Test finding most similar tasks."""
        query = mx.array([1.0, 0.0])
        task_embeddings = [
            mx.array([1.0, 0.0]),    # Most similar
            mx.array([0.5, 0.5]),    # Somewhat similar
            mx.array([0.0, 1.0]),    # Least similar
        ]

        similar_indices = TaskSimilarityMetric.find_most_similar_tasks(
            query, task_embeddings, top_k=2, metric="cosine"
        )

        # Should return top 2 indices
        assert len(similar_indices) == 2

        # First should be index 0 (identical)
        assert similar_indices[0] == 0

    def test_clustering_tasks(self):
        """Test task clustering by similarity."""
        embeddings = [
            mx.array([1.0, 0.0]),
            mx.array([1.1, 0.1]),
            mx.array([0.0, 1.0]),
            mx.array([0.1, 1.1]),
        ]

        clusters = TaskSimilarityMetric.cluster_tasks(
            embeddings, num_clusters=2, method="kmeans"
        )

        # Should assign to 2 clusters
        assert len(set(clusters)) == 2

        # Similar embeddings should be in same cluster
        assert clusters[0] == clusters[1]
        assert clusters[2] == clusters[3]


class TestTaskFeatureExtractor:
    """Test task feature extraction utilities."""

    def test_extract_statistical_features(self):
        """Test extracting statistical features from data."""
        # Create synthetic task data
        x_train = mx.random.normal((50, 10))
        y_train = mx.random.randint(0, 5, (50,))

        features = TaskFeatureExtractor.extract_statistical_features(
            x_train, y_train
        )

        # Should contain expected keys
        assert "data_mean" in features
        assert "data_std" in features
        assert "data_min" in features
        assert "data_max" in features
        assert "num_samples" in features
        assert "num_features" in features
        assert "num_classes" in features

        # Verify values
        assert features["num_samples"] == 50
        assert features["num_features"] == 10
        assert features["num_classes"] == 5

    def test_extract_model_features(self):
        """Test extracting model-based features."""
        model = SimpleClassifier(input_dim=10, hidden_dim=32, num_classes=5)
        x_train = mx.random.normal((50, 10))
        y_train = mx.random.randint(0, 5, (50,))

        features = TaskFeatureExtractor.extract_model_features(
            model, x_train, y_train
        )

        # Should contain expected keys
        assert "train_loss" in features
        assert "prediction_entropy" in features
        assert "confidence_mean" in features
        assert "confidence_std" in features

        # Values should be reasonable
        assert features["train_loss"] > 0
        assert features["prediction_entropy"] >= 0
        assert 0 <= features["confidence_mean"] <= 1

    def test_combine_features(self):
        """Test combining multiple feature types."""
        x_train = mx.random.normal((50, 10))
        y_train = mx.random.randint(0, 5, (50,))

        features = TaskFeatureExtractor.combine_features(x_train, y_train)

        # Should be a vector
        assert len(features.shape) == 1
        assert features.shape[0] > 0

    def test_feature_normalization(self):
        """Test feature normalization."""
        features = mx.array([1.0, 10.0, 100.0, 1000.0])

        normalized = TaskFeatureExtractor.normalize_features(features)

        # Should have mean ~0 and std ~1
        assert mx.isclose(mx.mean(normalized), 0.0, atol=0.1)
        assert mx.isclose(mx.std(normalized), 1.0, atol=0.1)


class TestTaskEmbeddingIntegration:
    """Integration tests for task embedding system."""

    def test_end_to_end_task_embedding_pipeline(self):
        """Test complete task embedding pipeline."""
        # Create model
        model = SimpleClassifier(input_dim=10, hidden_dim=32, num_classes=5)

        # Create task data
        task1_x = mx.random.normal((30, 10))
        task1_y = mx.random.randint(0, 5, (30,))
        task2_x = mx.random.normal((30, 10)) + 2.0
        task2_y = mx.random.randint(0, 5, (30,))

        # Method 1: Neural embedding
        neural_embedder = TaskEmbeddingNetwork(
            input_dim=128,
            hidden_dim=256,
            embedding_dim=64,
        )
        features1 = TaskFeatureExtractor.combine_features(task1_x, task1_y)
        features2 = TaskFeatureExtractor.combine_features(task2_x, task2_y)
        neural_emb1 = neural_embedder(features1)
        neural_emb2 = neural_embedder(features2)

        # Method 2: Task2Vec embedding
        task2vec = Task2VecEmbedding(model)
        fisher_emb1 = task2vec(task1_x, task1_y)
        fisher_emb2 = task2vec(task2_x, task2_y)

        # Both methods should produce embeddings
        assert neural_emb1.shape[0] > 0
        assert fisher_emb1.shape[0] > 0

        # Compute similarities
        neural_sim = TaskSimilarityMetric.cosine_similarity(
            neural_emb1, neural_emb2
        )
        fisher_sim = TaskSimilarityMetric.cosine_similarity(
            fisher_emb1, fisher_emb2
        )

        # Similarities should be reasonable values
        assert -1 <= float(neural_sim) <= 1
        assert -1 <= float(fisher_sim) <= 1

    def test_task_retrieval_with_embeddings(self):
        """Test using embeddings for task retrieval."""
        embedder = TaskEmbeddingNetwork(
            input_dim=128,
            hidden_dim=256,
            embedding_dim=64,
        )

        # Create multiple task embeddings
        task_embeddings = []
        for _ in range(10):
            features = mx.random.normal((128,))
            embedding = embedder(features)
            task_embeddings.append(embedding)

        # Query with new task
        query_features = mx.random.normal((128,))
        query_embedding = embedder(query_features)

        # Find similar tasks
        similar_indices = TaskSimilarityMetric.find_most_similar_tasks(
            query_embedding, task_embeddings, top_k=3
        )

        assert len(similar_indices) == 3
        assert all(0 <= idx < 10 for idx in similar_indices)

    def test_task_clustering_for_curriculum(self):
        """Test task clustering for curriculum learning."""
        embedder = TaskEmbeddingNetwork(
            input_dim=128,
            hidden_dim=256,
            embedding_dim=64,
        )

        # Create task embeddings
        embeddings = [embedder(mx.random.normal((128,))) for _ in range(20)]

        # Cluster into difficulty groups
        clusters = TaskSimilarityMetric.cluster_tasks(
            embeddings, num_clusters=3, method="kmeans"
        )

        # Should have 3 clusters
        assert len(set(clusters)) == 3

        # Each cluster should have tasks
        for cluster_id in range(3):
            cluster_size = sum(1 for c in clusters if c == cluster_id)
            assert cluster_size > 0
