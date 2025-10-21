# Meta-Learning PEFT Usage Guide

**Version:** 0.1.0 (Research Phase)
**Status:** META-002 Active Development
**Last Updated:** 2025-10-21

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Installation](#installation)
3. [Basic Concepts](#basic-concepts)
4. [Common Workflows](#common-workflows)
5. [CLI Usage](#cli-usage)
6. [Python API Usage](#python-api-usage)
7. [Configuration](#configuration)
8. [Examples](#examples)
9. [Best Practices](#best-practices)
10. [FAQ](#faq)

---

## Quick Start

### 5-Minute Tutorial

```bash
# Navigate to project directory
cd projects/05_Meta_Learning_PEFT

# Install dependencies
uv sync

# Verify installation
uv run efficientai-toolkit meta-learning-peft:info

# Run simple meta-learning experiment (Python)
uv run python examples/quick_start.py
```

**Quick Start Python Script** (`examples/quick_start.py`):

```python
import mlx.core as mx
from src.meta_learning.reptile import ReptileLearner
from src.meta_learning.models import SimpleClassifier, cross_entropy_loss
from src.task_embedding.task_distribution import TaskConfig, TaskDistribution

# Create 20 simple tasks
configs = [
    TaskConfig(
        task_id=f"task_{i}",
        task_family="linear_classification",
        num_classes=2,
        input_dim=2,
        support_size=5,
        query_size=20
    )
    for i in range(20)
]
dist = TaskDistribution(configs, seed=42)

# Initialize model and meta-learner
model = SimpleClassifier(input_dim=2, hidden_dim=64, num_classes=2)
learner = ReptileLearner(model, inner_lr=0.01, outer_lr=0.001, num_inner_steps=5)

# Meta-train for 100 iterations
print("Starting meta-training...")
for iteration in range(100):
    episodes = [dist.sample_episode()[1:] for _ in range(4)]
    metrics = learner.meta_train_step(episodes, cross_entropy_loss)

    if iteration % 20 == 0:
        print(f"Iteration {iteration}: loss={metrics['query_loss']:.3f}")

# Evaluate on validation tasks
val_episodes = [dist.sample_episode()[1:] for _ in range(10)]
results = learner.evaluate(val_episodes, cross_entropy_loss)
print(f"\n5-shot accuracy: {results['step_5_acc']:.3f}")
print("Meta-learning complete!")
```

---

## Installation

### Prerequisites

- **Hardware**: Apple Silicon (M1/M2/M3) recommended
- **Python**: 3.11 or higher
- **Package Manager**: `uv`

### Step-by-Step Installation

#### Option 1: Install as Part of Toolkit (Recommended)

```bash
# Clone repository
git clone https://github.com/your-org/EfficientAI-MLX-Toolkit.git
cd EfficientAI-MLX-Toolkit

# Install all dependencies
uv sync

# Verify installation
uv run efficientai-toolkit meta-learning-peft:info
```

#### Option 2: Install Project Standalone

```bash
# Navigate to project
cd projects/05_Meta_Learning_PEFT

# Install dependencies
uv sync

# Verify installation
uv run python -c "import mlx.core as mx; print('MLX ready!')"
```

### Verify Installation

```bash
# Check MLX installation
uv run python -c "import mlx.core as mx; print(f'MLX version: {mx.__version__}')"

# Check all imports
uv run python -c "
from src.meta_learning.reptile import ReptileLearner
from src.task_embedding.task_distribution import TaskDistribution
print('All imports successful!')
"

# Run test suite
uv run pytest tests/ -v
```

---

## Basic Concepts

### What is Meta-Learning?

Meta-learning (learning to learn) trains models to adapt quickly to new tasks with minimal data.

**Key Idea:**
- Learn an **initialization** that can be fine-tuned rapidly
- Uses **task distribution** instead of single dataset
- Enables **few-shot learning** (learning from 1-10 examples)

### Reptile Algorithm

Reptile is a first-order meta-learning algorithm:

1. **Sample** a batch of tasks from task distribution
2. **Adapt** model to each task with SGD (K steps)
3. **Update** meta-parameters toward task-adapted parameters
4. **Repeat** until convergence

**Update Rule:**
```
θ_new = θ + β * mean(θ_task - θ)
```

Where:
- `θ`: Meta-parameters (initialization)
- `θ_task`: Task-adapted parameters
- `β`: Meta-learning rate

### Task Episodes

An **episode** consists of:
- **Support set**: K examples for adaptation (K-shot)
- **Query set**: Evaluation examples (not used for training)

Example:
```python
# 5-shot, 2-way classification episode
support_x: (5, 2)   # 5 examples, 2D input
support_y: (5,)     # 5 labels
query_x: (20, 2)    # 20 test examples
query_y: (20,)      # 20 test labels
```

### Task Distribution

A **task distribution** is a collection of related tasks:

```python
# Example: 20 linear classification tasks
task_configs = [
    TaskConfig(
        task_id=f"task_{i}",
        task_family="linear_classification",
        num_classes=2,
        input_dim=2,
        support_size=5,  # 5-shot
        query_size=20
    )
    for i in range(20)
]
```

---

## Common Workflows

### Workflow 1: Meta-Training from Scratch

```python
import mlx.core as mx
from src.meta_learning.reptile import ReptileLearner
from src.meta_learning.models import SimpleClassifier, cross_entropy_loss
from src.task_embedding.task_distribution import TaskConfig, TaskDistribution
from src.utils.config import MetaLearningConfig
from src.utils.logging import setup_logger

# 1. Setup logging
logger = setup_logger("meta_training", log_dir="logs")

# 2. Load configuration
config = MetaLearningConfig.from_yaml("configs/default.yaml")

# 3. Create task distribution
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

# 4. Initialize model and learner
model = SimpleClassifier(input_dim=2, hidden_dim=64, num_classes=2)
learner = ReptileLearner(
    model=model,
    inner_lr=config.inner_lr,
    outer_lr=config.outer_lr,
    num_inner_steps=config.num_inner_steps
)

# 5. Meta-training loop
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

logger.info("Meta-training complete!")
```

### Workflow 2: Evaluate Meta-Learned Model

```python
from src.meta_learning.reptile import ReptileLearner
from src.meta_learning.models import SimpleClassifier, cross_entropy_loss
from src.task_embedding.task_distribution import TaskDistribution

# 1. Load meta-learned checkpoint
model = SimpleClassifier(input_dim=2, hidden_dim=64, num_classes=2)
learner = ReptileLearner(model)
learner.load("checkpoints/reptile_iter_1000.npz")

# 2. Load validation tasks
val_dist = TaskDistribution.load("data/validation_tasks.pkl")

# 3. Evaluate few-shot performance
val_episodes = [val_dist.sample_episode()[1:] for _ in range(20)]
results = learner.evaluate(val_episodes, cross_entropy_loss)

print(f"1-shot accuracy: {results['step_1_acc']:.3f}")
print(f"5-shot accuracy: {results['step_5_acc']:.3f}")
print(f"Final loss: {results['final_loss']:.3f}")
```

### Workflow 3: Baseline Comparison

```python
from src.utils.baseline import BaselineComparison
from src.meta_learning.models import SimpleClassifier

# 1. Create baseline model (random init)
baseline_model = SimpleClassifier(input_dim=2, hidden_dim=64, num_classes=2)

# 2. Load meta-learned model
learner.load("checkpoints/reptile_iter_1000.npz")

# 3. Compare few-shot performance
test_episodes = [val_dist.sample_episode()[1:] for _ in range(20)]
comparison = BaselineComparison.compare_few_shot_performance(
    meta_learned_model=learner.model,
    standard_model=baseline_model,
    episodes=test_episodes,
    k_shots=[1, 5, 10]
)

# 4. Print results
for k, metrics in comparison.items():
    improvement = metrics['meta_acc'] - metrics['baseline_acc']
    print(f"{k}-shot:")
    print(f"  Meta-learned: {metrics['meta_acc']:.3f}")
    print(f"  Baseline: {metrics['baseline_acc']:.3f}")
    print(f"  Improvement: {improvement:.3f} ({improvement/metrics['baseline_acc']*100:.1f}%)")
```

### Workflow 4: Custom Task Distribution

```python
from src.task_embedding.task_distribution import TaskConfig, TaskDistribution
import mlx.core as mx
import numpy as np

# 1. Define custom task generator
def generate_rotation_tasks(num_tasks: int = 20):
    """Generate tasks with different 2D rotations."""
    configs = []
    for i in range(num_tasks):
        angle = np.random.uniform(0, 2*np.pi)
        configs.append(
            TaskConfig(
                task_id=f"rotation_{i}",
                task_family="linear_classification",
                num_classes=2,
                input_dim=2,
                support_size=5,
                query_size=20,
                task_params={"rotation_angle": angle}
            )
        )
    return configs

# 2. Create distribution
configs = generate_rotation_tasks(num_tasks=50)
custom_dist = TaskDistribution(configs, seed=42)

# 3. Use in meta-training
# (same as Workflow 1)
```

---

## CLI Usage

### Available Commands

```bash
# Display project information
uv run efficientai-toolkit meta-learning-peft:info

# Validate project setup
uv run efficientai-toolkit meta-learning-peft:validate

# Future commands (planned for META-003)
uv run efficientai-toolkit meta-learning-peft:train --config configs/default.yaml
uv run efficientai-toolkit meta-learning-peft:evaluate --checkpoint checkpoints/best.npz
```

### Command Details

#### `info`

```bash
uv run efficientai-toolkit meta-learning-peft:info
```

**Output:**
```
Meta-Learning PEFT System
Status: META-002 (Research Phase)
MLX: Available (version 0.0.8)
Python: 3.12.0
```

#### `validate`

```bash
uv run efficientai-toolkit meta-learning-peft:validate
```

**Checks:**
- MLX installation
- Required dependencies
- Configuration files
- Test suite passing

---

## Python API Usage

### Core API Patterns

#### Pattern 1: Initialize Learner

```python
from src.meta_learning.reptile import ReptileLearner
from src.meta_learning.models import SimpleClassifier

# Create model
model = SimpleClassifier(
    input_dim=10,
    hidden_dim=128,
    num_classes=3
)

# Create learner
learner = ReptileLearner(
    model=model,
    inner_lr=0.01,      # Task adaptation learning rate
    outer_lr=0.001,     # Meta-learning rate
    num_inner_steps=5   # Gradient steps per task
)
```

#### Pattern 2: Sample Episodes

```python
from src.task_embedding.task_distribution import TaskConfig, TaskDistribution

# Create task distribution
configs = [TaskConfig(...) for _ in range(100)]
dist = TaskDistribution(configs, seed=42)

# Sample single episode
config, support_x, support_y, query_x, query_y = dist.sample_episode()

# Sample batch of episodes
episodes = [dist.sample_episode()[1:] for _ in range(4)]
```

#### Pattern 3: Meta-Train Step

```python
from src.meta_learning.models import cross_entropy_loss

# Sample batch of tasks
episodes = [dist.sample_episode()[1:] for _ in range(4)]

# Perform meta-training step
metrics = learner.meta_train_step(episodes, cross_entropy_loss)

# Metrics available:
# - metrics['query_loss']: Loss on query sets
# - metrics['support_loss']: Loss on support sets
# - metrics['param_diff']: Parameter update magnitude
```

#### Pattern 4: Evaluate Model

```python
# Sample validation episodes
val_episodes = [val_dist.sample_episode()[1:] for _ in range(10)]

# Evaluate
results = learner.evaluate(val_episodes, cross_entropy_loss)

# Results available:
# - results['step_1_acc']: Accuracy after 1 gradient step
# - results['step_5_acc']: Accuracy after 5 gradient steps
# - results['final_loss']: Average final loss
```

#### Pattern 5: Save/Load Checkpoints

```python
# Save checkpoint
learner.save("checkpoints/reptile_epoch_100.npz")

# Load checkpoint
learner.load("checkpoints/reptile_epoch_100.npz")
```

---

## Configuration

### Configuration File Structure

**`configs/default.yaml`:**

```yaml
meta_learning:
  algorithm: "reptile"           # Meta-learning algorithm
  meta_batch_size: 4             # Tasks per meta-update
  num_meta_iterations: 1000      # Total iterations
  inner_lr: 0.01                 # Task adaptation LR
  outer_lr: 0.001                # Meta-learning LR
  num_inner_steps: 5             # Steps per task
  eval_interval: 100             # Evaluation frequency
  save_interval: 500             # Checkpoint frequency

task_distribution:
  task_families:
    - linear_classification
    - nonlinear_classification
  k_shot: 5                      # Support set size
  query_size: 50                 # Query set size
  num_train_tasks: 100           # Training tasks
  num_val_tasks: 20              # Validation tasks

model:
  backbone: "linear"             # Model architecture
  hidden_dim: 128                # Hidden layer size
  num_layers: 2                  # Number of layers

logging:
  log_dir: "logs"                # Log directory
  log_level: "INFO"              # Logging level
  wandb_enabled: false           # Weights & Biases integration

paths:
  checkpoint_dir: "checkpoints"  # Checkpoint directory
  data_dir: "data"               # Data directory
```

### Load Configuration

```python
from src.utils.config import MetaLearningConfig

# Load from YAML
config = MetaLearningConfig.from_yaml("configs/default.yaml")

# Access fields
print(config.inner_lr)  # 0.01
print(config.meta_batch_size)  # 4

# Convert to dict
config_dict = config.to_dict()
```

### Override Configuration

```python
# Load base config
config = MetaLearningConfig.from_yaml("configs/default.yaml")

# Override specific fields
config.inner_lr = 0.02
config.num_meta_iterations = 2000

# Use updated config
learner = ReptileLearner(
    model=model,
    inner_lr=config.inner_lr,
    outer_lr=config.outer_lr,
    num_inner_steps=config.num_inner_steps
)
```

---

## Examples

### Example 1: Simple 2D Classification

```python
import mlx.core as mx
from src.meta_learning.reptile import ReptileLearner
from src.meta_learning.models import SimpleClassifier, cross_entropy_loss
from src.task_embedding.task_distribution import TaskConfig, TaskDistribution

# Create 20 2D classification tasks
configs = [
    TaskConfig(
        task_id=f"task_{i}",
        task_family="linear_classification",
        num_classes=2,
        input_dim=2,
        support_size=5,
        query_size=20
    )
    for i in range(20)
]
dist = TaskDistribution(configs, seed=42)

# Meta-learn
model = SimpleClassifier(input_dim=2, hidden_dim=64, num_classes=2)
learner = ReptileLearner(model)

for iteration in range(100):
    episodes = [dist.sample_episode()[1:] for _ in range(4)]
    metrics = learner.meta_train_step(episodes, cross_entropy_loss)
    if iteration % 20 == 0:
        print(f"Iter {iteration}: loss={metrics['query_loss']:.3f}")

# Test
val_episodes = [dist.sample_episode()[1:] for _ in range(10)]
results = learner.evaluate(val_episodes, cross_entropy_loss)
print(f"5-shot accuracy: {results['step_5_acc']:.3f}")
```

### Example 2: Multi-Class Classification

```python
# Create 3-way classification tasks
configs = [
    TaskConfig(
        task_id=f"task_{i}",
        task_family="nonlinear_classification",
        num_classes=3,
        input_dim=10,
        support_size=10,  # 10-shot
        query_size=30
    )
    for i in range(50)
]
dist = TaskDistribution(configs, seed=42)

# Use larger model
model = SimpleClassifier(input_dim=10, hidden_dim=128, num_classes=3)
learner = ReptileLearner(model, inner_lr=0.01, outer_lr=0.001, num_inner_steps=10)

# Train with more inner steps
for iteration in range(500):
    episodes = [dist.sample_episode()[1:] for _ in range(4)]
    metrics = learner.meta_train_step(episodes, cross_entropy_loss)
    # ...
```

### Example 3: Baseline Comparison Study

```python
from src.utils.baseline import BaselineComparison

# Setup
meta_model = SimpleClassifier(input_dim=2, hidden_dim=64, num_classes=2)
baseline_model = SimpleClassifier(input_dim=2, hidden_dim=64, num_classes=2)
learner = ReptileLearner(meta_model)

# Meta-train
for iteration in range(1000):
    episodes = [train_dist.sample_episode()[1:] for _ in range(4)]
    learner.meta_train_step(episodes, cross_entropy_loss)

# Compare
test_episodes = [val_dist.sample_episode()[1:] for _ in range(20)]
comparison = BaselineComparison.compare_few_shot_performance(
    meta_learned_model=learner.model,
    standard_model=baseline_model,
    episodes=test_episodes,
    k_shots=[1, 5, 10]
)

# Visualize
import matplotlib.pyplot as plt
k_values = list(comparison.keys())
meta_accs = [comparison[k]['meta_acc'] for k in k_values]
baseline_accs = [comparison[k]['baseline_acc'] for k in k_values]

plt.plot(k_values, meta_accs, label='Meta-learned', marker='o')
plt.plot(k_values, baseline_accs, label='Baseline', marker='s')
plt.xlabel('K-shot')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Meta-Learning vs Baseline')
plt.savefig('comparison.png')
```

---

## Best Practices

### 1. Task Distribution Design

**Do:**
- Use 80-100 training tasks for stable meta-learning
- Keep support sets small (5-10 examples) to test few-shot learning
- Include diverse task variations

**Don't:**
- Use too few tasks (<20) - leads to overfitting
- Make tasks too similar - limits generalization
- Use too large support sets - defeats purpose of few-shot learning

### 2. Hyperparameter Selection

**Inner Learning Rate (`inner_lr`):**
- Start with `0.01`
- Increase if task adaptation is slow
- Decrease if training is unstable

**Outer Learning Rate (`outer_lr`):**
- Typically 10x smaller than `inner_lr`
- Start with `0.001`
- Use learning rate decay for longer runs

**Number of Inner Steps:**
- Start with `5` steps
- Increase for complex tasks (10-20)
- Monitor convergence on support set

### 3. Monitoring Training

```python
# Track key metrics
metrics_history = {
    'query_loss': [],
    'val_acc': [],
    'param_diff': []
}

for iteration in range(num_iterations):
    metrics = learner.meta_train_step(episodes, loss_fn)
    metrics_history['query_loss'].append(metrics['query_loss'])

    if iteration % eval_interval == 0:
        val_results = learner.evaluate(val_episodes, loss_fn)
        metrics_history['val_acc'].append(val_results['step_5_acc'])

        # Early stopping
        if val_results['step_5_acc'] > 0.95:
            print(f"Early stopping at iteration {iteration}")
            break
```

### 4. Checkpoint Management

```python
import os
from pathlib import Path

# Save best checkpoint
best_val_acc = 0.0
for iteration in range(num_iterations):
    # ... training ...

    if iteration % eval_interval == 0:
        val_results = learner.evaluate(val_episodes, loss_fn)
        if val_results['step_5_acc'] > best_val_acc:
            best_val_acc = val_results['step_5_acc']
            learner.save("checkpoints/best.npz")
            print(f"New best: {best_val_acc:.3f}")

    # Save periodic checkpoints
    if iteration % save_interval == 0:
        learner.save(f"checkpoints/iter_{iteration}.npz")
```

### 5. Memory Management

```python
# For large-scale experiments, clear gradients
import gc

for iteration in range(num_iterations):
    metrics = learner.meta_train_step(episodes, loss_fn)

    if iteration % 100 == 0:
        gc.collect()  # Free unused memory
```

---

## FAQ

### General Questions

**Q: What is the difference between Reptile and MAML?**

A: Reptile is a first-order algorithm (no second-order derivatives), making it simpler and more memory-efficient. MAML uses second-order derivatives for potentially faster adaptation but requires more computation.

**Q: How many tasks do I need for meta-learning?**

A: Minimum 20 tasks, but 80-100 recommended for stable training. More tasks = better generalization.

**Q: What is a good few-shot accuracy?**

A: Depends on task complexity. For simple 2D classification, >80% 5-shot accuracy is good. For complex tasks, >60% may be excellent.

### Technical Questions

**Q: Why is my meta-training loss not decreasing?**

A: Common causes:
1. `outer_lr` too low - try increasing to 0.005
2. `num_inner_steps` too few - try 10-20 steps
3. Tasks too diverse - reduce task variation
4. Model too small - increase `hidden_dim`

**Q: How do I speed up meta-training?**

A: Options:
1. Reduce `num_inner_steps` (5 is usually sufficient)
2. Use smaller `meta_batch_size` (4 is good)
3. Use simpler model architecture
4. Ensure MLX is using Apple Silicon acceleration

**Q: Can I use this with custom models?**

A: Yes! Any `mlx.nn.Module` works:

```python
import mlx.nn as nn

class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [nn.Linear(10, 64), nn.Linear(64, 2)]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

model = CustomModel()
learner = ReptileLearner(model)  # Works!
```

**Q: How do I save task distributions?**

A: Use pickle:

```python
import pickle

# Save
with open("data/task_dist.pkl", "wb") as f:
    pickle.dump(dist, f)

# Load
with open("data/task_dist.pkl", "rb") as f:
    loaded_dist = pickle.load(f)
```

### Troubleshooting

**Q: `ImportError: No module named 'mlx'`**

A: Install MLX:
```bash
uv add mlx
```

**Q: Tests failing with `AssertionError`**

A: Ensure you're on Apple Silicon:
```bash
uv run python -c "import mlx.core as mx; print(mx.metal.is_available())"
```

**Q: Out of memory errors**

A: Reduce batch size and model size:
```python
config.meta_batch_size = 2  # Reduce from 4
model = SimpleClassifier(hidden_dim=32)  # Reduce from 64
```

---

## Next Steps

### Learning Path

1. **Start Here**: Run Quick Start tutorial
2. **Experiment**: Modify hyperparameters in `configs/default.yaml`
3. **Custom Tasks**: Create your own task distribution
4. **Baseline Comparison**: Compare meta-learning vs standard fine-tuning
5. **Advanced**: Explore research papers in `research/literature_review.md`

### Additional Resources

- [API Reference](./API.md) - Complete API documentation
- [Architecture Guide](./ARCHITECTURE.md) - System design and internals
- [Integration Guide](./INTEGRATION.md) - MLOps and production deployment
- [Research Documentation](../research/literature_review.md) - Meta-learning papers

### Getting Help

- **Issues**: Check [Troubleshooting Guide](./TROUBLESHOOTING.md)
- **Examples**: See `examples/` directory (META-003+)
- **Tests**: Review `tests/` for usage patterns
- **Questions**: Refer to main toolkit repository

---

## Appendix

### Common Error Messages

| Error | Cause | Solution |
|-------|-------|----------|
| `TaskSamplingError` | Invalid task_id | Check task exists in distribution |
| `RuntimeError: Metal not available` | Not on Apple Silicon | Use CPU backend or Apple hardware |
| `ValueError: Invalid shape` | Mismatched dimensions | Verify input_dim matches data |
| `FileNotFoundError: config.yaml` | Config not found | Provide correct path to config |

### Performance Benchmarks

**Meta-Training Speed (M1 Max, 1000 iterations):**
- Simple model (64 hidden): ~2 minutes
- Large model (256 hidden): ~8 minutes

**Memory Usage:**
- Meta batch size 4: ~500 MB
- Meta batch size 8: ~1 GB

### Version Compatibility

- MLX: >= 0.0.8
- Python: >= 3.11
- NumPy: >= 1.24.0
- PyTorch: >= 2.0.0 (optional, for compatibility)

---

**End of Usage Guide**
