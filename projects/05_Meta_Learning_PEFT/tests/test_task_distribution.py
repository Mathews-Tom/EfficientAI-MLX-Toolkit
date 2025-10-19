"""Tests for task distribution."""

import pytest

try:
    import mlx.core as mx

    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from task_embedding.task_distribution import (
    LinearTask,
    NonLinearTask,
    Task,
    TaskConfig,
    TaskDistribution,
)


@pytest.mark.skipif(not MLX_AVAILABLE, reason="MLX not available")
class TestTaskConfig:
    """Test TaskConfig dataclass."""

    def test_task_config_creation(self) -> None:
        """Test creating TaskConfig."""
        config = TaskConfig(
            task_id="test_task_1",
            task_family="linear_classification",
            num_classes=2,
            input_dim=2,
            support_size=5,
            query_size=50,
        )

        assert config.task_id == "test_task_1"
        assert config.task_family == "linear_classification"
        assert config.num_classes == 2
        assert config.input_dim == 2
        assert config.support_size == 5
        assert config.query_size == 50
        assert config.domain == "synthetic"
        assert config.difficulty == "medium"


@pytest.mark.skipif(not MLX_AVAILABLE, reason="MLX not available")
class TestLinearTask:
    """Test LinearTask implementation."""

    def test_linear_task_creation(self) -> None:
        """Test creating LinearTask."""
        config = TaskConfig(
            task_id="linear_1",
            task_family="linear_classification",
            num_classes=2,
            input_dim=2,
            support_size=10,
            query_size=50,
        )

        task = LinearTask(config, seed=42)

        assert task.config == config
        assert task.seed == 42
        assert "w" in task.task_params
        assert "b" in task.task_params

    def test_linear_task_sample_data(self) -> None:
        """Test sampling data from LinearTask."""
        config = TaskConfig(
            task_id="linear_1",
            task_family="linear_classification",
            num_classes=2,
            input_dim=2,
            support_size=10,
            query_size=50,
        )

        task = LinearTask(config, seed=42)
        x, y = task.sample_data(10)

        assert x.shape == (10, 2)
        assert y.shape == (10,)
        assert y.dtype == mx.int32

    def test_linear_task_rotation(self) -> None:
        """Test LinearTask with rotation."""
        config = TaskConfig(
            task_id="linear_1",
            task_family="linear_classification",
            num_classes=2,
            input_dim=2,
            support_size=10,
            query_size=50,
        )

        task = LinearTask(config, seed=42, rotation=1.57)  # ~90 degrees
        x, y = task.sample_data(10)

        assert x.shape == (10, 2)
        assert task.rotation == 1.57

    def test_linear_task_episode(self) -> None:
        """Test getting complete episode."""
        config = TaskConfig(
            task_id="linear_1",
            task_family="linear_classification",
            num_classes=2,
            input_dim=2,
            support_size=5,
            query_size=20,
        )

        task = LinearTask(config, seed=42)
        support_x, support_y, query_x, query_y = task.get_episode()

        assert support_x.shape == (5, 2)
        assert support_y.shape == (5,)
        assert query_x.shape == (20, 2)
        assert query_y.shape == (20,)


@pytest.mark.skipif(not MLX_AVAILABLE, reason="MLX not available")
class TestNonLinearTask:
    """Test NonLinearTask implementation."""

    def test_nonlinear_task_xor(self) -> None:
        """Test XOR pattern generation."""
        config = TaskConfig(
            task_id="xor_1",
            task_family="nonlinear_classification",
            num_classes=2,
            input_dim=2,
            support_size=10,
            query_size=50,
        )

        task = NonLinearTask(config, seed=42, pattern_type="xor")
        x, y = task.sample_data(100)

        assert x.shape == (100, 2)
        assert y.shape == (100,)
        # XOR should have roughly balanced classes
        assert 30 <= int(mx.sum(y)) <= 70

    def test_nonlinear_task_circles(self) -> None:
        """Test circles pattern generation."""
        config = TaskConfig(
            task_id="circles_1",
            task_family="nonlinear_classification",
            num_classes=2,
            input_dim=2,
            support_size=10,
            query_size=50,
        )

        task = NonLinearTask(config, seed=42, pattern_type="circles")
        x, y = task.sample_data(100)

        assert x.shape == (100, 2)
        assert y.shape == (100,)

    def test_nonlinear_task_spiral(self) -> None:
        """Test spiral pattern generation."""
        config = TaskConfig(
            task_id="spiral_1",
            task_family="nonlinear_classification",
            num_classes=2,
            input_dim=2,
            support_size=10,
            query_size=50,
        )

        task = NonLinearTask(config, seed=42, pattern_type="spiral")
        x, y = task.sample_data(100)

        assert x.shape == (100, 2)
        assert y.shape == (100,)

    def test_nonlinear_task_invalid_pattern(self) -> None:
        """Test error handling for invalid pattern type."""
        config = TaskConfig(
            task_id="invalid_1",
            task_family="nonlinear_classification",
            num_classes=2,
            input_dim=2,
            support_size=10,
            query_size=50,
        )

        task = NonLinearTask(config, seed=42, pattern_type="invalid")

        with pytest.raises(ValueError, match="Unknown pattern type"):
            task.sample_data(10)


@pytest.mark.skipif(not MLX_AVAILABLE, reason="MLX not available")
class TestTaskDistribution:
    """Test TaskDistribution implementation."""

    def test_task_distribution_creation(self) -> None:
        """Test creating TaskDistribution."""
        configs = [
            TaskConfig(
                task_id=f"task_{i}",
                task_family="linear_classification",
                num_classes=2,
                input_dim=2,
                support_size=5,
                query_size=50,
            )
            for i in range(10)
        ]

        dist = TaskDistribution(configs, seed=42)

        assert len(dist.task_configs) == 10
        assert "linear_classification" in dist.task_families
        assert len(dist.task_families["linear_classification"]) == 10

    def test_sample_task(self) -> None:
        """Test sampling tasks."""
        configs = [
            TaskConfig(
                task_id=f"linear_{i}",
                task_family="linear_classification",
                num_classes=2,
                input_dim=2,
                support_size=5,
                query_size=50,
            )
            for i in range(5)
        ] + [
            TaskConfig(
                task_id=f"nonlinear_{i}",
                task_family="nonlinear_classification",
                num_classes=2,
                input_dim=2,
                support_size=5,
                query_size=50,
            )
            for i in range(5)
        ]

        dist = TaskDistribution(configs, seed=42)

        # Sample random task
        task = dist.sample_task()
        assert isinstance(task, Task)

        # Sample from specific family
        linear_task = dist.sample_task(task_family="linear_classification")
        assert isinstance(linear_task, LinearTask)

        nonlinear_task = dist.sample_task(task_family="nonlinear_classification")
        assert isinstance(nonlinear_task, NonLinearTask)

    def test_sample_episode(self) -> None:
        """Test sampling complete episodes."""
        configs = [
            TaskConfig(
                task_id=f"task_{i}",
                task_family="linear_classification",
                num_classes=2,
                input_dim=2,
                support_size=5,
                query_size=20,
            )
            for i in range(5)
        ]

        dist = TaskDistribution(configs, seed=42)

        task, support_x, support_y, query_x, query_y = dist.sample_episode()

        assert isinstance(task, Task)
        assert support_x.shape == (5, 2)
        assert support_y.shape == (5,)
        assert query_x.shape == (20, 2)
        assert query_y.shape == (20,)

    def test_get_task_embedding(self) -> None:
        """Test task embedding extraction."""
        config = TaskConfig(
            task_id="test_task",
            task_family="linear_classification",
            num_classes=2,
            input_dim=2,
            support_size=5,
            query_size=50,
        )

        dist = TaskDistribution([config], seed=42)
        task = dist.sample_task()
        embedding = dist.get_task_embedding(task)

        assert embedding.shape[0] > 0  # Has features
        assert embedding.dtype == mx.float32

    def test_task_family_grouping(self) -> None:
        """Test task families are correctly grouped."""
        configs = [
            TaskConfig(
                task_id=f"linear_{i}",
                task_family="linear_classification",
                num_classes=2,
                input_dim=2,
                support_size=5,
                query_size=50,
            )
            for i in range(3)
        ] + [
            TaskConfig(
                task_id=f"nonlinear_{i}",
                task_family="nonlinear_classification",
                num_classes=2,
                input_dim=2,
                support_size=5,
                query_size=50,
            )
            for i in range(2)
        ]

        dist = TaskDistribution(configs, seed=42)

        assert len(dist.task_families) == 2
        assert len(dist.task_families["linear_classification"]) == 3
        assert len(dist.task_families["nonlinear_classification"]) == 2
