"""Learned task embeddings for meta-learning.

This module implements neural network-based task embeddings that can be
learned during meta-training, as opposed to handcrafted features.

References:
    - Achille et al. (2019) "Task2Vec: Task Embedding for Meta-Learning"
    - Triantafillou et al. (2021) "Learning a Universal Template for Few-shot Dataset Generalization"
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn

from utils.logging import get_logger

logger = get_logger(__name__)


class TaskEmbeddingNetwork(nn.Module):
    """Neural network for learning task embeddings.

    This network takes task characteristics (dataset statistics, model
    outputs, etc.) and produces a fixed-size embedding that captures
    task-specific information useful for meta-learning.

    Attributes:
        embedding_dim: Dimension of output task embedding
        hidden_dim: Dimension of hidden layers
        num_layers: Number of hidden layers
    """

    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 256,
        embedding_dim: int = 64,
        num_layers: int = 3,
        activation: str = "relu",
    ):
        """Initialize task embedding network.

        Args:
            input_dim: Dimension of input features (task statistics)
            hidden_dim: Hidden layer dimension
            embedding_dim: Output embedding dimension
            num_layers: Number of hidden layers (default: 3)
            activation: Activation function ('relu', 'gelu', 'tanh')
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        # Build network layers
        layers = []

        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))

        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))

        # Output layer
        layers.append(nn.Linear(hidden_dim, embedding_dim))

        self.layers = layers

        # Activation function
        if activation == "relu":
            self.activation = nn.relu
        elif activation == "gelu":
            self.activation = nn.gelu
        elif activation == "tanh":
            self.activation = mx.tanh
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def __call__(self, task_features: mx.array) -> mx.array:
        """Compute task embedding from task features.

        Args:
            task_features: Task statistics/features of shape (batch_size, input_dim)
                          or (input_dim,) for single task

        Returns:
            Task embedding of shape (batch_size, embedding_dim) or (embedding_dim,)
        """
        x = task_features

        # Forward pass through all layers
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # Apply activation to all except last layer
            if i < len(self.layers) - 1:
                x = self.activation(x)

        # L2 normalize embeddings
        x = x / (mx.sqrt(mx.sum(x**2, axis=-1, keepdims=True)) + 1e-8)

        return x


class Task2VecEmbedding(nn.Module):
    """Task2Vec: embedding based on Fisher Information Matrix.

    Computes task embedding using the diagonal of the Fisher Information
    Matrix (FIM) of model parameters on the task support set.

    Reference:
        Achille et al. (2019) "Task2Vec: Task Embedding for Meta-Learning"
    """

    def __init__(self, embedding_dim: int = 64):
        """Initialize Task2Vec embedding.

        Args:
            embedding_dim: Target embedding dimension
        """
        super().__init__()
        self.embedding_dim = embedding_dim

    def compute_fisher_diagonal(
        self, model: nn.Module, support_x: mx.array, support_y: mx.array
    ) -> dict[str, mx.array]:
        """Compute diagonal of Fisher Information Matrix.

        Args:
            model: Neural network model
            support_x: Support set inputs (n_support, input_dim)
            support_y: Support set labels (n_support,)

        Returns:
            Dictionary of FIM diagonal values for each parameter
        """

        def loss_fn(params):
            model.update(params)
            logits = model(support_x)
            # Cross-entropy loss
            return mx.mean(nn.losses.cross_entropy(logits, support_y))

        # Compute gradients
        loss, grads = mx.value_and_grad(loss_fn)(model.parameters())

        # Square gradients to get FIM diagonal approximation
        fim_diag = {k: g**2 for k, g in grads.items()}

        return fim_diag

    def __call__(
        self,
        model: nn.Module,
        support_x: mx.array,
        support_y: mx.array,
        target_dim: int | None = None,
    ) -> mx.array:
        """Compute Task2Vec embedding.

        Args:
            model: Model to compute FIM on
            support_x: Support set inputs
            support_y: Support set labels
            target_dim: Target dimension (default: self.embedding_dim)

        Returns:
            Task embedding of shape (embedding_dim,)
        """
        if target_dim is None:
            target_dim = self.embedding_dim

        # Compute FIM diagonal
        fim_diag = self.compute_fisher_diagonal(model, support_x, support_y)

        # Flatten FIM values
        fim_flat = mx.concatenate([mx.flatten(v) for v in fim_diag.values()])

        # Reduce to target dimension using random projection
        # (In practice, use PCA or learned projection)
        if len(fim_flat) > target_dim:
            # Simple average pooling for dimension reduction
            chunk_size = len(fim_flat) // target_dim
            embedding = mx.array(
                [
                    mx.mean(fim_flat[i * chunk_size : (i + 1) * chunk_size])
                    for i in range(target_dim)
                ]
            )
        else:
            # Pad if needed
            padding = target_dim - len(fim_flat)
            embedding = mx.concatenate([fim_flat, mx.zeros(padding)])

        # L2 normalize
        embedding = embedding / (mx.sqrt(mx.sum(embedding**2)) + 1e-8)

        return embedding


class TaskSimilarityMetric:
    """Compute similarity between task embeddings.

    Provides various metrics for measuring task similarity, which can be
    used for:
        - Transfer learning (which task to transfer from?)
        - Task selection (which tasks are most useful for meta-training?)
        - Task clustering (group similar tasks)
    """

    @staticmethod
    def cosine_similarity(
        embedding1: mx.array, embedding2: mx.array
    ) -> mx.array:
        """Compute cosine similarity between embeddings.

        Args:
            embedding1: First embedding (embedding_dim,)
            embedding2: Second embedding (embedding_dim,)

        Returns:
            Cosine similarity in [-1, 1]
        """
        # Normalize embeddings
        emb1_norm = embedding1 / (mx.sqrt(mx.sum(embedding1**2)) + 1e-8)
        emb2_norm = embedding2 / (mx.sqrt(mx.sum(embedding2**2)) + 1e-8)

        # Dot product of normalized vectors
        similarity = mx.sum(emb1_norm * emb2_norm)

        return similarity

    @staticmethod
    def euclidean_distance(
        embedding1: mx.array, embedding2: mx.array
    ) -> mx.array:
        """Compute Euclidean distance between embeddings.

        Args:
            embedding1: First embedding
            embedding2: Second embedding

        Returns:
            Euclidean distance (lower = more similar)
        """
        diff = embedding1 - embedding2
        distance = mx.sqrt(mx.sum(diff**2))
        return distance

    @staticmethod
    def manhattan_distance(
        embedding1: mx.array, embedding2: mx.array
    ) -> mx.array:
        """Compute Manhattan (L1) distance between embeddings.

        Args:
            embedding1: First embedding
            embedding2: Second embedding

        Returns:
            Manhattan distance
        """
        diff = mx.abs(embedding1 - embedding2)
        distance = mx.sum(diff)
        return distance

    @staticmethod
    def similarity_matrix(
        embeddings: list[mx.array], metric: str = "cosine"
    ) -> mx.array:
        """Compute pairwise similarity matrix for multiple embeddings.

        Args:
            embeddings: List of task embeddings
            metric: Similarity metric ('cosine', 'euclidean', 'manhattan')

        Returns:
            Similarity matrix of shape (num_tasks, num_tasks)
        """
        num_tasks = len(embeddings)
        similarity_mat = mx.zeros((num_tasks, num_tasks))

        for i in range(num_tasks):
            for j in range(num_tasks):
                if metric == "cosine":
                    sim = TaskSimilarityMetric.cosine_similarity(
                        embeddings[i], embeddings[j]
                    )
                elif metric == "euclidean":
                    sim = -TaskSimilarityMetric.euclidean_distance(
                        embeddings[i], embeddings[j]
                    )
                elif metric == "manhattan":
                    sim = -TaskSimilarityMetric.manhattan_distance(
                        embeddings[i], embeddings[j]
                    )
                else:
                    raise ValueError(f"Unknown metric: {metric}")

                similarity_mat[i, j] = sim

        return similarity_mat

    @staticmethod
    def find_most_similar_tasks(
        query_embedding: mx.array,
        task_embeddings: list[mx.array],
        top_k: int = 5,
        metric: str = "cosine",
    ) -> list[tuple[int, float]]:
        """Find most similar tasks to query task.

        Args:
            query_embedding: Query task embedding
            task_embeddings: List of candidate task embeddings
            top_k: Number of most similar tasks to return
            metric: Similarity metric

        Returns:
            List of (task_index, similarity_score) tuples, sorted by similarity
        """
        similarities = []

        for i, task_emb in enumerate(task_embeddings):
            if metric == "cosine":
                sim = float(
                    TaskSimilarityMetric.cosine_similarity(
                        query_embedding, task_emb
                    )
                )
            elif metric == "euclidean":
                sim = -float(
                    TaskSimilarityMetric.euclidean_distance(
                        query_embedding, task_emb
                    )
                )
            elif metric == "manhattan":
                sim = -float(
                    TaskSimilarityMetric.manhattan_distance(
                        query_embedding, task_emb
                    )
                )
            else:
                raise ValueError(f"Unknown metric: {metric}")

            similarities.append((i, sim))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_k]


class TaskFeatureExtractor:
    """Extract features from tasks for embedding computation.

    Converts task data into fixed-size feature vectors that can be
    fed into TaskEmbeddingNetwork.
    """

    @staticmethod
    def extract_dataset_statistics(
        support_x: mx.array, support_y: mx.array, query_x: mx.array
    ) -> mx.array:
        """Extract basic dataset statistics as features.

        Args:
            support_x: Support set inputs (n_support, input_dim)
            support_y: Support set labels (n_support,)
            query_x: Query set inputs (n_query, input_dim)

        Returns:
            Feature vector containing dataset statistics
        """
        features = []

        # Dataset size features
        features.append(float(mx.log(float(len(support_x)) + 1)))
        features.append(float(mx.log(float(len(query_x)) + 1)))

        # Input dimension
        features.append(float(support_x.shape[-1]))

        # Number of classes
        num_classes = float(mx.max(support_y) + 1)
        features.append(num_classes)

        # Data statistics (mean, std)
        features.append(float(mx.mean(support_x)))
        features.append(float(mx.std(support_x)))
        features.append(float(mx.mean(query_x)))
        features.append(float(mx.std(query_x)))

        # Class balance
        for c in range(int(num_classes)):
            class_ratio = float(mx.sum(support_y == c)) / len(support_y)
            features.append(class_ratio)

        # Convert to array
        feature_vector = mx.array(features)

        return feature_vector

    @staticmethod
    def extract_model_based_features(
        model: nn.Module,
        support_x: mx.array,
        support_y: mx.array,
        num_adaptation_steps: int = 5,
    ) -> mx.array:
        """Extract features based on model behavior on task.

        Args:
            model: Neural network model
            support_x: Support set inputs
            support_y: Support set labels
            num_adaptation_steps: Steps to adapt model

        Returns:
            Feature vector based on model behavior
        """
        features = []

        # Initial loss and accuracy
        initial_logits = model(support_x)
        initial_loss = float(
            mx.mean(nn.losses.cross_entropy(initial_logits, support_y))
        )
        initial_acc = float(
            mx.mean(mx.argmax(initial_logits, axis=-1) == support_y)
        )

        features.append(initial_loss)
        features.append(initial_acc)

        # Gradient norm
        def loss_fn(params):
            model.update(params)
            logits = model(support_x)
            return mx.mean(nn.losses.cross_entropy(logits, support_y))

        _, grads = mx.value_and_grad(loss_fn)(model.parameters())
        grad_norm = float(
            mx.sqrt(sum(mx.sum(g**2) for g in grads.values()))
        )
        features.append(grad_norm)

        # Loss reduction rate (after adaptation)
        # (Simplified - in practice, run actual adaptation)
        features.append(initial_loss * 0.7)  # Placeholder

        return mx.array(features)


def combine_feature_extractors(
    support_x: mx.array,
    support_y: mx.array,
    query_x: mx.array,
    model: nn.Module | None = None,
) -> mx.array:
    """Combine multiple feature extractors into single feature vector.

    Args:
        support_x: Support set inputs
        support_y: Support set labels
        query_x: Query set inputs
        model: Optional model for model-based features

    Returns:
        Combined feature vector
    """
    # Dataset statistics
    dataset_features = TaskFeatureExtractor.extract_dataset_statistics(
        support_x, support_y, query_x
    )

    # Model-based features (if model provided)
    if model is not None:
        model_features = TaskFeatureExtractor.extract_model_based_features(
            model, support_x, support_y
        )
        # Concatenate
        combined = mx.concatenate([dataset_features, model_features])
    else:
        combined = dataset_features

    # Pad to fixed size (128-dim)
    target_dim = 128
    if len(combined) < target_dim:
        padding = mx.zeros(target_dim - len(combined))
        combined = mx.concatenate([combined, padding])
    elif len(combined) > target_dim:
        combined = combined[:target_dim]

    return combined
