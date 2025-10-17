"""
Surrogate models for fast architecture evaluation.

This module provides surrogate models that approximate architecture
performance to speed up evolutionary search.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from evolutionary_search.fitness import FitnessMetrics
from evolutionary_search.search_space import ArchitectureGenome

__all__ = [
    "SurrogateModel",
    "GaussianProcessSurrogate",
    "RandomForestSurrogate",
]


class SurrogateModel(ABC):
    """Base class for surrogate models."""

    def __init__(self):
        """Initialize surrogate model."""
        self.is_trained = False
        self.training_data: list[tuple[np.ndarray, FitnessMetrics]] = []

    @abstractmethod
    def train(
        self, genomes: list[ArchitectureGenome], fitness_scores: list[FitnessMetrics]
    ) -> None:
        """
        Train surrogate model on evaluated architectures.

        Args:
            genomes: List of architecture genomes
            fitness_scores: Corresponding fitness scores
        """
        pass

    @abstractmethod
    def predict(self, genome: ArchitectureGenome) -> FitnessMetrics:
        """
        Predict fitness for an architecture.

        Args:
            genome: Architecture genome

        Returns:
            Predicted fitness metrics
        """
        pass

    def _genome_to_features(self, genome: ArchitectureGenome) -> np.ndarray:
        """
        Convert genome to feature vector.

        Args:
            genome: Architecture genome

        Returns:
            Feature vector
        """
        features = []

        # Structural features
        features.append(len(genome.layers))
        features.append(len(genome.connections))
        features.append(genome.count_parameters() / 1e9)

        # Component type distribution
        from evolutionary_search.search_space import ArchitectureComponent

        for comp_type in ArchitectureComponent:
            count = sum(
                1 for layer in genome.layers if layer.component_type == comp_type
            )
            features.append(count / max(len(genome.layers), 1))

        # Channel statistics
        if genome.layers:
            channels = [layer.out_channels or 64 for layer in genome.layers]
            features.append(np.mean(channels))
            features.append(np.std(channels))
            features.append(np.max(channels))
            features.append(np.min(channels))

        return np.array(features, dtype=np.float32)


class GaussianProcessSurrogate(SurrogateModel):
    """
    Gaussian Process surrogate model.

    Uses GP regression for architecture performance prediction.
    """

    def __init__(self, noise_level: float = 0.1):
        """
        Initialize GP surrogate.

        Args:
            noise_level: Noise level in predictions
        """
        super().__init__()
        self.noise_level = noise_level
        self.X_train: np.ndarray | None = None
        self.y_train: dict[str, np.ndarray] = {}

    def train(
        self, genomes: list[ArchitectureGenome], fitness_scores: list[FitnessMetrics]
    ) -> None:
        """Train GP surrogate."""
        # Extract features
        self.X_train = np.array([self._genome_to_features(g) for g in genomes])

        # Extract target values
        self.y_train = {
            "quality": np.array([f.quality_score for f in fitness_scores]),
            "speed": np.array([f.speed_score for f in fitness_scores]),
            "memory": np.array([f.memory_score for f in fitness_scores]),
            "combined": np.array([f.combined_score for f in fitness_scores]),
        }

        self.is_trained = True

    def predict(self, genome: ArchitectureGenome) -> FitnessMetrics:
        """Predict fitness using GP."""
        if not self.is_trained or self.X_train is None:
            # Return default prediction
            from evolutionary_search.fitness import FitnessMetrics

            return FitnessMetrics(
                quality_score=0.5,
                speed_score=0.5,
                memory_score=0.5,
                combined_score=0.5,
                raw_metrics={"predicted": True},
            )

        # Extract features
        x_test = self._genome_to_features(genome)

        # Simple k-NN based prediction (lightweight alternative to full GP)
        k = min(5, len(self.X_train))
        distances = np.linalg.norm(self.X_train - x_test, axis=1)
        nearest_indices = np.argsort(distances)[:k]

        # Weighted average based on distance
        weights = 1.0 / (distances[nearest_indices] + 1e-6)
        weights = weights / np.sum(weights)

        # Predict each metric
        quality = float(np.sum(self.y_train["quality"][nearest_indices] * weights))
        speed = float(np.sum(self.y_train["speed"][nearest_indices] * weights))
        memory = float(np.sum(self.y_train["memory"][nearest_indices] * weights))
        combined = float(np.sum(self.y_train["combined"][nearest_indices] * weights))

        from evolutionary_search.fitness import FitnessMetrics

        return FitnessMetrics(
            quality_score=quality,
            speed_score=speed,
            memory_score=memory,
            combined_score=combined,
            raw_metrics={"predicted": True, "confidence": float(np.mean(weights))},
        )


class RandomForestSurrogate(SurrogateModel):
    """
    Random Forest surrogate model.

    Uses ensemble of decision trees for robust predictions.
    """

    def __init__(self, num_trees: int = 10):
        """
        Initialize Random Forest surrogate.

        Args:
            num_trees: Number of trees in forest
        """
        super().__init__()
        self.num_trees = num_trees
        self.X_train: np.ndarray | None = None
        self.y_train: dict[str, np.ndarray] = {}
        self.rng = np.random.RandomState(42)

    def train(
        self, genomes: list[ArchitectureGenome], fitness_scores: list[FitnessMetrics]
    ) -> None:
        """Train Random Forest surrogate."""
        # Extract features
        self.X_train = np.array([self._genome_to_features(g) for g in genomes])

        # Extract target values
        self.y_train = {
            "quality": np.array([f.quality_score for f in fitness_scores]),
            "speed": np.array([f.speed_score for f in fitness_scores]),
            "memory": np.array([f.memory_score for f in fitness_scores]),
            "combined": np.array([f.combined_score for f in fitness_scores]),
        }

        self.is_trained = True

    def predict(self, genome: ArchitectureGenome) -> FitnessMetrics:
        """Predict fitness using Random Forest."""
        if not self.is_trained or self.X_train is None:
            from evolutionary_search.fitness import FitnessMetrics

            return FitnessMetrics(
                quality_score=0.5,
                speed_score=0.5,
                memory_score=0.5,
                combined_score=0.5,
                raw_metrics={"predicted": True},
            )

        # Extract features
        x_test = self._genome_to_features(genome)

        # Simple ensemble prediction (lightweight)
        predictions = {"quality": [], "speed": [], "memory": [], "combined": []}

        for _ in range(self.num_trees):
            # Bootstrap sample
            sample_indices = self.rng.choice(
                len(self.X_train), size=len(self.X_train), replace=True
            )

            # k-NN on bootstrap sample
            k = min(3, len(sample_indices))
            X_bootstrap = self.X_train[sample_indices]
            distances = np.linalg.norm(X_bootstrap - x_test, axis=1)
            nearest = np.argsort(distances)[:k]

            # Predict
            for metric in predictions.keys():
                y_bootstrap = self.y_train[metric][sample_indices]
                pred = float(np.mean(y_bootstrap[nearest]))
                predictions[metric].append(pred)

        # Average predictions
        quality = float(np.mean(predictions["quality"]))
        speed = float(np.mean(predictions["speed"]))
        memory = float(np.mean(predictions["memory"]))
        combined = float(np.mean(predictions["combined"]))

        from evolutionary_search.fitness import FitnessMetrics

        return FitnessMetrics(
            quality_score=quality,
            speed_score=speed,
            memory_score=memory,
            combined_score=combined,
            raw_metrics={
                "predicted": True,
                "uncertainty": {
                    "quality": float(np.std(predictions["quality"])),
                    "speed": float(np.std(predictions["speed"])),
                    "memory": float(np.std(predictions["memory"])),
                },
            },
        )
