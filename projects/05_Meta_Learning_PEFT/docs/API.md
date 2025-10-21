# Meta-Learning PEFT API Reference

**Version:** 0.1.0 (Research Phase)
**Status:** META-002 Active Development
**Last Updated:** 2025-10-21

---

## Overview

This document provides comprehensive API reference for the Meta-Learning Parameter-Efficient Fine-Tuning (PEFT) system. The API is organized into four main modules:

1. **Meta-Learning** - Core meta-learning algorithms (Reptile, MAML)
2. **Task Distribution** - Task generation and episode sampling
3. **Adapter Generation** - PEFT adapter creation and management
4. **Utilities** - Configuration, logging, and baseline tools

---

## Meta-Learning Module

### `meta_learning.reptile`

#### `ReptileLearner`

First-order meta-learning algorithm for fast task adaptation.

```python
class ReptileLearner:
    """
    Reptile meta-learning algorithm implementation.

    Learns an initialization that can be quickly adapted to new tasks
    through few-shot learning.

    Attributes:
        model: Neural network model to meta-train
        inner_lr: Learning rate for task adaptation (inner loop)
        outer_lr: Meta-learning rate (outer loop)
        num_inner_steps: Number of gradient steps per task
    """

    def __init__(
        self,
        model: nn.Module,
        inner_lr: float = 0.01,
        outer_lr: float = 0.001,
        num_inner_steps: int = 5
    ):
        """
        Initialize Reptile learner.

        Args:
            model: Neural network to meta-train
            inner_lr: Task adaptation learning rate (default: 0.01)
            outer_lr: Meta-learning rate (default: 0.001)
            num_inner_steps: Steps per task adaptation (default: 5)

        Example:
            >>> from src.meta_learning.models import SimpleClassifier
            >>> model = SimpleClassifier(input_dim=2, hidden_dim=64, num_classes=2)
            >>> learner = ReptileLearner(model, inner_lr=0.01, outer_lr=0.001)
        """
```

**Methods:**

##### `meta_train_step()`

```python
def meta_train_step(
    self,
    episodes: list[tuple[mx.array, mx.array, mx.array, mx.array]],
    loss_fn: Callable
) -> dict[str, float]:
    """
    Perform one meta-training step on a batch of tasks.

    Args:
        episodes: List of (support_x, support_y, query_x, query_y) tuples
        loss_fn: Loss function compatible with MLX

    Returns:
        Dictionary containing:
            - "query_loss": Average loss on query sets
            - "support_loss": Average loss on support sets
            - "param_diff": L2 norm of parameter update

    Example:
        >>> episodes = [dist.sample_episode()[1:] for _ in range(4)]
        >>> metrics = learner.meta_train_step(episodes, cross_entropy_loss)
        >>> print(f"Query loss: {metrics['query_loss']:.3f}")
    """
```

##### `evaluate()`

```python
def evaluate(
    self,
    episodes: list[tuple[mx.array, mx.array, mx.array, mx.array]],
    loss_fn: Callable
) -> dict[str, float]:
    """
    Evaluate meta-learned initialization on validation tasks.

    Args:
        episodes: Validation episodes
        loss_fn: Loss function

    Returns:
        Dictionary containing:
            - "step_1_acc", "step_5_acc": Accuracy after N adaptation steps
            - "final_loss": Average final loss

    Example:
        >>> val_episodes = [dist.sample_episode()[1:] for _ in range(10)]
        >>> results = learner.evaluate(val_episodes, cross_entropy_loss)
        >>> print(f"5-shot accuracy: {results['step_5_acc']:.3f}")
    """
```

##### `save()`

```python
def save(self, path: str | Path) -> None:
    """
    Save meta-learned parameters to file.

    Args:
        path: Path to save checkpoint (.npz format)

    Example:
        >>> learner.save("checkpoints/reptile_epoch_100.npz")
    """
```

##### `load()`

```python
def load(self, path: str | Path) -> None:
    """
    Load meta-learned parameters from file.

    Args:
        path: Path to checkpoint file

    Example:
        >>> learner.load("checkpoints/reptile_epoch_100.npz")
    """
```

---

### `meta_learning.models`

#### `SimpleClassifier`

```python
class SimpleClassifier(nn.Module):
    """
    Simple feedforward classifier for meta-learning tasks.

    Architecture:
        input -> linear -> relu -> linear -> relu -> output

    Attributes:
        input_dim: Input feature dimension
        hidden_dim: Hidden layer dimension
        num_classes: Number of output classes
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_classes: int = 2
    ):
        """
        Initialize classifier.

        Args:
            input_dim: Dimension of input features
            hidden_dim: Hidden layer size (default: 64)
            num_classes: Number of classes (default: 2)

        Example:
            >>> model = SimpleClassifier(input_dim=10, hidden_dim=128, num_classes=3)
            >>> x = mx.random.normal((32, 10))
            >>> logits = model(x)
            >>> print(logits.shape)  # (32, 3)
        """

    def __call__(self, x: mx.array) -> mx.array:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Logits of shape (batch_size, num_classes)
        """
```

#### Loss Functions

##### `cross_entropy_loss()`

```python
def cross_entropy_loss(
    logits: mx.array,
    targets: mx.array
) -> mx.array:
    """
    Cross-entropy loss for classification.

    Args:
        logits: Model predictions (batch_size, num_classes)
        targets: Ground truth labels (batch_size,) or (batch_size, num_classes)

    Returns:
        Scalar loss value

    Example:
        >>> logits = model(x)
        >>> loss = cross_entropy_loss(logits, y)
    """
```

##### `accuracy()`

```python
def accuracy(logits: mx.array, targets: mx.array) -> float:
    """
    Compute classification accuracy.

    Args:
        logits: Model predictions
        targets: Ground truth labels

    Returns:
        Accuracy as float in [0, 1]

    Example:
        >>> acc = accuracy(logits, targets)
        >>> print(f"Accuracy: {acc:.1%}")
    """
```

---

## Task Distribution Module

### `task_embedding.task_distribution`

#### `TaskConfig`

```python
@dataclass
class TaskConfig:
    """
    Configuration for a meta-learning task.

    Attributes:
        task_id: Unique task identifier
        task_family: Task category (e.g., "linear_classification")
        num_classes: Number of classes in classification task
        input_dim: Input feature dimension
        support_size: Number of support examples (K-shot)
        query_size: Number of query examples for evaluation
        task_params: Optional task-specific parameters
    """
    task_id: str
    task_family: str
    num_classes: int
    input_dim: int
    support_size: int
    query_size: int
    task_params: dict[str, Any] | None = None
```

#### `TaskDistribution`

```python
class TaskDistribution:
    """
    Manages distribution of meta-learning tasks.

    Provides task sampling, episode generation, and task embedding
    extraction for meta-learning experiments.

    Attributes:
        task_configs: List of TaskConfig objects
        seed: Random seed for reproducibility
    """

    def __init__(
        self,
        task_configs: list[TaskConfig],
        seed: int | None = None
    ):
        """
        Initialize task distribution.

        Args:
            task_configs: List of task configurations
            seed: Random seed (default: None)

        Example:
            >>> configs = [
            ...     TaskConfig(
            ...         task_id=f"task_{i}",
            ...         task_family="linear_classification",
            ...         num_classes=2,
            ...         input_dim=2,
            ...         support_size=5,
            ...         query_size=20
            ...     )
            ...     for i in range(20)
            ... ]
            >>> dist = TaskDistribution(configs, seed=42)
        """
```

**Methods:**

##### `sample_episode()`

```python
def sample_episode(
    self,
    task_id: str | None = None
) -> tuple[TaskConfig, mx.array, mx.array, mx.array, mx.array]:
    """
    Sample an episode (support + query sets) from a task.

    Args:
        task_id: Specific task to sample (default: random)

    Returns:
        Tuple of (task_config, support_x, support_y, query_x, query_y)

    Example:
        >>> config, support_x, support_y, query_x, query_y = dist.sample_episode()
        >>> print(f"Support: {support_x.shape}, Query: {query_x.shape}")
    """
```

##### `get_task_embedding()`

```python
def get_task_embedding(self, task_id: str) -> mx.array:
    """
    Extract handcrafted task embedding features.

    Features:
        - dataset_size: log(support_size + query_size)
        - k_shot: support_size
        - num_classes: number of classes
        - input_dim: feature dimension
        - complexity_score: estimated task difficulty

    Args:
        task_id: Task identifier

    Returns:
        Embedding vector of shape (embedding_dim,)

    Example:
        >>> embedding = dist.get_task_embedding("task_0")
        >>> print(f"Task embedding shape: {embedding.shape}")
    """
```

##### `split_tasks()`

```python
def split_tasks(
    self,
    train_ratio: float = 0.8
) -> tuple["TaskDistribution", "TaskDistribution"]:
    """
    Split tasks into train and validation sets.

    Args:
        train_ratio: Fraction of tasks for training (default: 0.8)

    Returns:
        (train_distribution, val_distribution)

    Example:
        >>> train_dist, val_dist = dist.split_tasks(train_ratio=0.8)
        >>> print(f"Train: {len(train_dist.task_configs)} tasks")
        >>> print(f"Val: {len(val_dist.task_configs)} tasks")
    """
```

---

## Adapter Generation Module

### `adapter_generation.adapter_factory`

#### `AdapterFactory`

```python
class AdapterFactory:
    """
    Factory for creating PEFT adapters based on task characteristics.

    Supports:
        - LoRA (Low-Rank Adaptation)
        - AdaLoRA (Adaptive LoRA)
        - Prefix Tuning
        - Prompt Tuning
    """

    @staticmethod
    def create_adapter(
        adapter_type: str,
        model: nn.Module,
        config: dict[str, Any]
    ) -> "BaseAdapter":
        """
        Create PEFT adapter for model.

        Args:
            adapter_type: Type of adapter ("lora", "adalora", etc.)
            model: Base model to adapt
            config: Adapter configuration

        Returns:
            Initialized adapter instance

        Example:
            >>> config = {"rank": 8, "alpha": 16, "dropout": 0.1}
            >>> adapter = AdapterFactory.create_adapter("lora", model, config)
        """
```

### `adapter_generation.peft_integration`

#### `LoRAAdapter`

```python
class LoRAAdapter:
    """
    Low-Rank Adaptation (LoRA) implementation for MLX.

    Decomposes weight updates as W + BA where:
        - B: (output_dim, rank)
        - A: (rank, input_dim)
        - rank << min(output_dim, input_dim)

    Attributes:
        rank: Low-rank decomposition rank
        alpha: Scaling factor
        dropout: Dropout probability
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0
    ):
        """
        Initialize LoRA adapter.

        Args:
            in_features: Input dimension
            out_features: Output dimension
            rank: Rank of low-rank matrices (default: 8)
            alpha: Scaling parameter (default: 16.0)
            dropout: Dropout rate (default: 0.0)

        Example:
            >>> adapter = LoRAAdapter(
            ...     in_features=768,
            ...     out_features=768,
            ...     rank=8,
            ...     alpha=16
            ... )
        """

    def __call__(self, x: mx.array) -> mx.array:
        """
        Apply LoRA adaptation.

        Args:
            x: Input tensor

        Returns:
            Adapted output
        """
```

---

## Utilities Module

### `utils.config`

#### `MetaLearningConfig`

```python
@dataclass
class MetaLearningConfig:
    """
    Configuration for meta-learning experiments.

    Attributes:
        algorithm: Meta-learning algorithm ("reptile" or "maml")
        meta_batch_size: Number of tasks per meta-update
        num_meta_iterations: Total meta-training iterations
        inner_lr: Task adaptation learning rate
        outer_lr: Meta-learning rate
        num_inner_steps: Gradient steps per task
        eval_interval: Iterations between evaluations
        save_interval: Iterations between checkpoints
    """
    algorithm: str = "reptile"
    meta_batch_size: int = 4
    num_meta_iterations: int = 1000
    inner_lr: float = 0.01
    outer_lr: float = 0.001
    num_inner_steps: int = 5
    eval_interval: int = 100
    save_interval: int = 500

    @classmethod
    def from_yaml(cls, path: str | Path) -> "MetaLearningConfig":
        """
        Load configuration from YAML file.

        Args:
            path: Path to YAML config file

        Returns:
            Loaded configuration

        Example:
            >>> config = MetaLearningConfig.from_yaml("configs/default.yaml")
        """

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
```

### `utils.logging`

#### `setup_logger()`

```python
def setup_logger(
    name: str,
    log_dir: str | Path | None = None,
    level: str = "INFO"
) -> logging.Logger:
    """
    Configure logger with file and console handlers.

    Args:
        name: Logger name
        log_dir: Directory for log files (default: None, console only)
        level: Logging level (default: "INFO")

    Returns:
        Configured logger instance

    Example:
        >>> logger = setup_logger("meta_learning", log_dir="logs", level="DEBUG")
        >>> logger.info("Starting meta-training")
    """
```

### `utils.baseline`

#### `BaselineComparison`

```python
class BaselineComparison:
    """
    Utilities for comparing meta-learning vs standard fine-tuning.

    Provides:
        - Standard fine-tuning baseline
        - Few-shot performance comparison
        - Adaptation speed metrics
        - Statistical significance testing
    """

    @staticmethod
    def evaluate_standard_finetuning(
        model: nn.Module,
        train_data: tuple[mx.array, mx.array],
        test_data: tuple[mx.array, mx.array],
        num_steps: int = 100,
        lr: float = 0.01
    ) -> dict[str, Any]:
        """
        Evaluate standard fine-tuning (no meta-learning).

        Args:
            model: Model to fine-tune
            train_data: (x_train, y_train)
            test_data: (x_test, y_test)
            num_steps: Training steps
            lr: Learning rate

        Returns:
            Dictionary with learning curve and final metrics

        Example:
            >>> results = BaselineComparison.evaluate_standard_finetuning(
            ...     model, (x_train, y_train), (x_test, y_test)
            ... )
            >>> print(f"Final accuracy: {results['final_acc']:.3f}")
        """

    @staticmethod
    def compare_few_shot_performance(
        meta_learned_model: nn.Module,
        standard_model: nn.Module,
        episodes: list,
        k_shots: list[int] = [1, 5, 10]
    ) -> dict[str, dict[str, float]]:
        """
        Compare meta-learning vs baseline on few-shot tasks.

        Args:
            meta_learned_model: Meta-learned initialization
            standard_model: Randomly initialized model
            episodes: List of test episodes
            k_shots: Shot sizes to evaluate

        Returns:
            Nested dictionary: {shot_size: {metric: value}}

        Example:
            >>> comparison = BaselineComparison.compare_few_shot_performance(
            ...     meta_model, baseline_model, test_episodes, k_shots=[1, 5, 10]
            ... )
            >>> for k, metrics in comparison.items():
            ...     print(f"{k}-shot: {metrics['meta_acc']:.3f} vs {metrics['baseline_acc']:.3f}")
        """
```

---

## Usage Examples

### Basic Meta-Training Loop

```python
import mlx.core as mx
from src.meta_learning.reptile import ReptileLearner
from src.meta_learning.models import SimpleClassifier, cross_entropy_loss
from src.task_embedding.task_distribution import TaskConfig, TaskDistribution
from src.utils.config import MetaLearningConfig
from src.utils.logging import setup_logger

# Setup
logger = setup_logger("meta_learning", log_dir="logs")
config = MetaLearningConfig.from_yaml("configs/default.yaml")

# Create task distribution
task_configs = [
    TaskConfig(
        task_id=f"task_{i}",
        task_family="linear_classification",
        num_classes=2,
        input_dim=2,
        support_size=5,
        query_size=20
    )
    for i in range(100)
]
dist = TaskDistribution(task_configs, seed=42)
train_dist, val_dist = dist.split_tasks(train_ratio=0.8)

# Initialize model and learner
model = SimpleClassifier(input_dim=2, hidden_dim=64, num_classes=2)
learner = ReptileLearner(
    model=model,
    inner_lr=config.inner_lr,
    outer_lr=config.outer_lr,
    num_inner_steps=config.num_inner_steps
)

# Meta-training loop
for iteration in range(config.num_meta_iterations):
    # Sample batch of tasks
    episodes = [train_dist.sample_episode()[1:] for _ in range(config.meta_batch_size)]

    # Meta-train step
    metrics = learner.meta_train_step(episodes, cross_entropy_loss)

    # Logging
    if iteration % config.eval_interval == 0:
        val_episodes = [val_dist.sample_episode()[1:] for _ in range(10)]
        val_results = learner.evaluate(val_episodes, cross_entropy_loss)
        logger.info(
            f"Iter {iteration}: "
            f"train_loss={metrics['query_loss']:.3f}, "
            f"val_acc={val_results['step_5_acc']:.3f}"
        )

    # Save checkpoint
    if iteration % config.save_interval == 0:
        learner.save(f"checkpoints/reptile_iter_{iteration}.npz")
```

### Baseline Comparison

```python
from src.utils.baseline import BaselineComparison

# Create baseline model
baseline_model = SimpleClassifier(input_dim=2, hidden_dim=64, num_classes=2)

# Compare few-shot performance
test_episodes = [val_dist.sample_episode()[1:] for _ in range(20)]
comparison = BaselineComparison.compare_few_shot_performance(
    meta_learned_model=learner.model,
    standard_model=baseline_model,
    episodes=test_episodes,
    k_shots=[1, 5, 10]
)

for k, metrics in comparison.items():
    print(f"{k}-shot accuracy:")
    print(f"  Meta-learned: {metrics['meta_acc']:.3f}")
    print(f"  Baseline: {metrics['baseline_acc']:.3f}")
    print(f"  Improvement: {metrics['meta_acc'] - metrics['baseline_acc']:.3f}")
```

---

## Type Annotations

All functions use modern Python type hints:

```python
from typing import Callable, Any
import mlx.core as mx
import mlx.nn as nn

# Preferred (built-in generics)
def process_batch(
    data: list[mx.array],
    config: dict[str, Any]
) -> tuple[float, dict[str, float]]:
    ...

# Avoided (deprecated typing module)
from typing import List, Dict, Tuple  # Don't use
```

---

## Error Handling

All functions raise appropriate exceptions:

```python
class MetaLearningError(Exception):
    """Base exception for meta-learning errors."""

class TaskSamplingError(MetaLearningError):
    """Error during task sampling."""

class AdapterCreationError(MetaLearningError):
    """Error during adapter creation."""

# Example usage
try:
    episode = dist.sample_episode(task_id="nonexistent")
except TaskSamplingError as e:
    logger.error(f"Failed to sample episode: {e}")
```

---

## Testing

All API functions have corresponding tests in `tests/`:

```bash
# Run API tests
uv run pytest tests/test_reptile.py -v
uv run pytest tests/test_task_distribution.py -v
uv run pytest tests/test_peft_integration.py -v
uv run pytest tests/test_baseline.py -v
```

---

## Version History

- **0.1.0** (2025-10-21): Initial API for META-002 research phase
  - Reptile meta-learning
  - Task distribution
  - Baseline comparison utilities

---

## Future API Additions

Planned for upcoming phases:

### META-003: Meta-Learning Framework
- `MAMLLearner` class
- Learned task embeddings
- Task similarity metrics

### META-004: PEFT Integration
- `AdaLoRAAdapter` class
- `PrefixTuningAdapter` class
- Adapter hyperparameter optimization

### META-005: Production Features
- Model serving API
- Batch inference
- MLOps integration

---

## Notes

- All array operations use **MLX** (`mlx.core.array`)
- Type hints follow **PEP 484/585** (built-in generics)
- Error handling follows **fail-fast** principle
- API stability: **Experimental** (subject to change)
