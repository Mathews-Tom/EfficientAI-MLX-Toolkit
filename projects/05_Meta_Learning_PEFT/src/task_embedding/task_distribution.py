"""Task distribution for meta-learning."""

import random
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import numpy as np


@dataclass
class TaskConfig:
    """Configuration for a single task."""

    task_id: str
    task_family: str
    num_classes: int
    input_dim: int
    support_size: int
    query_size: int
    domain: str = "synthetic"
    difficulty: str = "medium"
    metadata: dict[str, Any] | None = None


class Task:
    """Single task instance for meta-learning."""

    def __init__(self, config: TaskConfig, seed: int | None = None):
        """Initialize task.

        Args:
            config: Task configuration.
            seed: Random seed for reproducibility.
        """
        self.config = config
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        # Task-specific parameters (to be set by subclasses)
        self.task_params: dict[str, Any] = {}

    def sample_data(
        self, num_samples: int, sample_type: str = "support"
    ) -> tuple[mx.array, mx.array]:
        """Sample data from task distribution.

        Args:
            num_samples: Number of samples to generate.
            sample_type: Type of samples ("support" or "query").

        Returns:
            Tuple of (inputs, labels).
        """
        raise NotImplementedError("Subclasses must implement sample_data")

    def compute_task_features(self) -> dict[str, float]:
        """Compute handcrafted task features.

        Returns:
            Dictionary of task features.
        """
        features = {
            "num_classes": float(self.config.num_classes),
            "input_dim": float(self.config.input_dim),
            "support_size": float(self.config.support_size),
            "query_size": float(self.config.query_size),
            "difficulty_score": {"easy": 0.0, "medium": 0.5, "hard": 1.0}.get(
                self.config.difficulty, 0.5
            ),
        }
        return features

    def get_episode(self) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        """Get complete episode (support + query sets).

        Returns:
            Tuple of (support_x, support_y, query_x, query_y).
        """
        support_x, support_y = self.sample_data(
            self.config.support_size, sample_type="support"
        )
        query_x, query_y = self.sample_data(
            self.config.query_size, sample_type="query"
        )
        return support_x, support_y, query_x, query_y


class LinearTask(Task):
    """Linear classification task with transformations."""

    def __init__(
        self,
        config: TaskConfig,
        seed: int | None = None,
        rotation: float = 0.0,
        translation: tuple[float, float] = (0.0, 0.0),
        noise_level: float = 0.1,
    ):
        """Initialize linear task.

        Args:
            config: Task configuration.
            seed: Random seed.
            rotation: Rotation angle in radians.
            translation: Translation offset (x, y).
            noise_level: Gaussian noise standard deviation.
        """
        super().__init__(config, seed)
        self.rotation = rotation
        self.translation = translation
        self.noise_level = noise_level

        # Generate random decision boundary
        self.w = self.rng.randn(config.input_dim)
        self.w = self.w / np.linalg.norm(self.w)  # Normalize
        self.b = self.rng.randn()

        self.task_params = {
            "w": self.w,
            "b": self.b,
            "rotation": rotation,
            "translation": translation,
            "noise_level": noise_level,
        }

    def sample_data(
        self, num_samples: int, sample_type: str = "support"
    ) -> tuple[mx.array, mx.array]:
        """Sample linearly separable data with transformations.

        Args:
            num_samples: Number of samples.
            sample_type: Type of samples.

        Returns:
            Tuple of (inputs, labels).
        """
        # Generate random points
        x = self.rng.randn(num_samples, self.config.input_dim)

        # Apply rotation (for 2D)
        if self.config.input_dim == 2 and self.rotation != 0.0:
            cos_r = np.cos(self.rotation)
            sin_r = np.sin(self.rotation)
            rotation_matrix = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
            x = x @ rotation_matrix.T

        # Apply translation
        x[:, 0] += self.translation[0]
        if self.config.input_dim >= 2:
            x[:, 1] += self.translation[1]

        # Add noise
        x += self.rng.randn(num_samples, self.config.input_dim) * self.noise_level

        # Generate labels based on decision boundary
        y = (x @ self.w + self.b > 0).astype(np.int32)

        return mx.array(x.astype(np.float32)), mx.array(y)


class NonLinearTask(Task):
    """Non-linear classification task (XOR, circles, spiral)."""

    def __init__(
        self,
        config: TaskConfig,
        seed: int | None = None,
        pattern_type: str = "xor",
        noise_level: float = 0.1,
    ):
        """Initialize non-linear task.

        Args:
            config: Task configuration.
            seed: Random seed.
            pattern_type: Pattern type ("xor", "circles", "spiral").
            noise_level: Gaussian noise standard deviation.
        """
        super().__init__(config, seed)
        self.pattern_type = pattern_type
        self.noise_level = noise_level

        self.task_params = {
            "pattern_type": pattern_type,
            "noise_level": noise_level,
        }

    def sample_data(
        self, num_samples: int, sample_type: str = "support"
    ) -> tuple[mx.array, mx.array]:
        """Sample non-linearly separable data.

        Args:
            num_samples: Number of samples.
            sample_type: Type of samples.

        Returns:
            Tuple of (inputs, labels).
        """
        if self.pattern_type == "xor":
            x, y = self._sample_xor(num_samples)
        elif self.pattern_type == "circles":
            x, y = self._sample_circles(num_samples)
        elif self.pattern_type == "spiral":
            x, y = self._sample_spiral(num_samples)
        else:
            raise ValueError(f"Unknown pattern type: {self.pattern_type}")

        # Add noise
        x += self.rng.randn(*x.shape) * self.noise_level

        return mx.array(x.astype(np.float32)), mx.array(y)

    def _sample_xor(self, num_samples: int) -> tuple[np.ndarray, np.ndarray]:
        """Sample XOR pattern."""
        x = self.rng.uniform(-1, 1, size=(num_samples, 2))
        y = ((x[:, 0] > 0) ^ (x[:, 1] > 0)).astype(np.int32)
        return x, y

    def _sample_circles(self, num_samples: int) -> tuple[np.ndarray, np.ndarray]:
        """Sample concentric circles pattern."""
        r = self.rng.uniform(0, 1, size=(num_samples,))
        theta = self.rng.uniform(0, 2 * np.pi, size=(num_samples,))
        x = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        y = (r < 0.5).astype(np.int32)
        return x, y

    def _sample_spiral(self, num_samples: int) -> tuple[np.ndarray, np.ndarray]:
        """Sample spiral pattern."""
        n_per_class = num_samples // 2
        theta = np.linspace(0, 4 * np.pi, n_per_class)

        # Class 0 spiral
        r0 = theta / (4 * np.pi)
        x0 = np.column_stack([r0 * np.cos(theta), r0 * np.sin(theta)])
        y0 = np.zeros(n_per_class, dtype=np.int32)

        # Class 1 spiral (rotated)
        r1 = theta / (4 * np.pi)
        x1 = np.column_stack(
            [r1 * np.cos(theta + np.pi), r1 * np.sin(theta + np.pi)]
        )
        y1 = np.ones(n_per_class, dtype=np.int32)

        # Combine
        x = np.vstack([x0, x1])
        y = np.concatenate([y0, y1])

        # Shuffle
        perm = self.rng.permutation(len(y))
        return x[perm], y[perm]


class TaskDistribution:
    """Distribution over tasks for meta-learning."""

    def __init__(self, task_configs: list[TaskConfig], seed: int | None = None):
        """Initialize task distribution.

        Args:
            task_configs: List of task configurations.
            seed: Random seed for task sampling.
        """
        self.task_configs = task_configs
        self.seed = seed
        self.rng = random.Random(seed)

        # Group tasks by family
        self.task_families: dict[str, list[TaskConfig]] = {}
        for config in task_configs:
            if config.task_family not in self.task_families:
                self.task_families[config.task_family] = []
            self.task_families[config.task_family].append(config)

    def sample_task(self, task_family: str | None = None) -> Task:
        """Sample a random task from distribution.

        Args:
            task_family: Optional task family to sample from.

        Returns:
            Task instance.
        """
        if task_family is not None:
            if task_family not in self.task_families:
                raise ValueError(f"Unknown task family: {task_family}")
            config = self.rng.choice(self.task_families[task_family])
        else:
            config = self.rng.choice(self.task_configs)

        # Create task instance based on family
        task_seed = self.rng.randint(0, 2**31 - 1)

        if config.task_family == "linear_classification":
            rotation = self.rng.uniform(-np.pi, np.pi)
            translation = (self.rng.uniform(-2, 2), self.rng.uniform(-2, 2))
            noise_level = self.rng.uniform(0.05, 0.2)
            return LinearTask(
                config,
                seed=task_seed,
                rotation=rotation,
                translation=translation,
                noise_level=noise_level,
            )
        elif config.task_family == "nonlinear_classification":
            pattern_type = self.rng.choice(["xor", "circles", "spiral"])
            noise_level = self.rng.uniform(0.05, 0.2)
            return NonLinearTask(
                config, seed=task_seed, pattern_type=pattern_type, noise_level=noise_level
            )
        else:
            raise ValueError(f"Unknown task family: {config.task_family}")

    def sample_episode(
        self, task_family: str | None = None
    ) -> tuple[Task, mx.array, mx.array, mx.array, mx.array]:
        """Sample complete episode (task + support + query).

        Args:
            task_family: Optional task family to sample from.

        Returns:
            Tuple of (task, support_x, support_y, query_x, query_y).
        """
        task = self.sample_task(task_family)
        support_x, support_y, query_x, query_y = task.get_episode()
        return task, support_x, support_y, query_x, query_y

    def get_task_embedding(self, task: Task) -> mx.array:
        """Extract task embedding features.

        Args:
            task: Task instance.

        Returns:
            Task embedding vector.
        """
        features = task.compute_task_features()
        # Convert features dict to vector
        feature_vector = np.array(list(features.values()), dtype=np.float32)
        return mx.array(feature_vector)
