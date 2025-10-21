# Meta-Learning for Parameter-Efficient Fine-Tuning (PEFT) with MLX

**Phase:** META-002 Research and Prototyping
**Status:** Active Development
**Priority:** P2 (Future Research)

---

## Overview

This project implements meta-learning approaches for Parameter-Efficient Fine-Tuning (PEFT) methods using Apple's MLX framework. The goal is to enable rapid task adaptation with minimal training examples (<10 examples) through learned initialization strategies.

### Key Features

- **Reptile Meta-Learning**: First-order meta-learning algorithm for fast adaptation
- **Task Distribution**: Synthetic task generation for controlled experiments
- **Baseline Benchmarking**: Comparison utilities for meta-learning vs standard fine-tuning
- **MLX Optimized**: Leverages Apple Silicon for efficient meta-training

---

## Research Goals

### Phase 1: Research and Prototyping (META-002) - Current

**Objectives:**
- Literature review of meta-learning and PEFT approaches
- Task distribution design for meta-learning
- Minimal working prototype of Reptile algorithm
- Baseline performance measurement infrastructure

**Research Questions:**
1. Can meta-learning improve few-shot PEFT adaptation?
2. Does learned initialization outperform random initialization?
3. Can task embeddings guide adapter selection?

### Future Phases

- **META-003**: Full meta-learning framework (MAML, task embeddings)
- **META-004**: PEFT integration (LoRA, AdaLoRA, learned adapters)
- **META-005**: Production deployment and optimization

---

## Project Structure

```
projects/05_Meta_Learning_PEFT/
├── src/
│   ├── meta_learning/          # Meta-learning algorithms
│   │   ├── reptile.py          # Reptile implementation
│   │   └── models.py           # Neural network models
│   ├── task_embedding/         # Task distribution and embeddings
│   │   └── task_distribution.py
│   ├── utils/                  # Utilities
│   │   ├── config.py           # Configuration management
│   │   ├── logging.py          # Logging utilities
│   │   └── baseline.py         # Baseline benchmarking
│   └── cli.py                  # Command-line interface
├── tests/                      # Test suite
│   ├── test_task_distribution.py
│   ├── test_reptile.py
│   └── test_baseline.py
├── research/                   # Research documentation
│   ├── literature_review.md    # Meta-learning literature review
│   └── task_distribution_design.md
├── configs/                    # Configuration files
│   └── default.yaml
├── docs/                       # Additional documentation
└── README.md                   # This file
```

---

## Installation

### Prerequisites

- Python 3.11+
- MLX framework (Apple Silicon)
- uv package manager

### Setup

```bash
# Navigate to project directory
cd projects/05_Meta_Learning_PEFT

# Install dependencies (using uv from toolkit root)
uv sync

# Or install from project directory
uv add mlx numpy torch transformers learn2learn higher peft
```

---

## Usage

### CLI Commands

```bash
# Display project information
uv run efficientai-toolkit meta-learning-peft:info

# Validate project setup
uv run efficientai-toolkit meta-learning-peft:validate

# Future: Training (planned for META-003)
uv run efficientai-toolkit meta-learning-peft:train --config configs/default.yaml
```

### Python API Example

```python
import mlx.core as mx
from src.meta_learning.reptile import ReptileLearner
from src.meta_learning.models import SimpleClassifier, cross_entropy_loss
from src.task_embedding.task_distribution import TaskConfig, TaskDistribution

# Create task distribution
configs = [
    TaskConfig(
        task_id=f"task_{i}",
        task_family="linear_classification",
        num_classes=2,
        input_dim=2,
        support_size=5,
        query_size=20,
    )
    for i in range(20)
]
dist = TaskDistribution(configs, seed=42)

# Create model and meta-learner
model = SimpleClassifier(input_dim=2, hidden_dim=64, num_classes=2)
learner = ReptileLearner(
    model=model,
    inner_lr=0.01,
    outer_lr=0.001,
    num_inner_steps=5
)

# Meta-training loop
for iteration in range(100):
    # Sample batch of tasks
    episodes = []
    for _ in range(4):
        _, support_x, support_y, query_x, query_y = dist.sample_episode()
        episodes.append((support_x, support_y, query_x, query_y))

    # Meta-train step
    metrics = learner.meta_train_step(episodes, cross_entropy_loss)

    if iteration % 10 == 0:
        print(f"Iteration {iteration}: loss={metrics['query_loss']:.3f}")

# Evaluate on validation tasks
eval_episodes = [dist.sample_episode()[1:] for _ in range(10)]
results = learner.evaluate(eval_episodes, cross_entropy_loss)
print(f"5-shot accuracy: {results['step_5_acc']:.3f}")
```

---

## Research Documentation

### Literature Review

See [research/literature_review.md](research/literature_review.md) for comprehensive literature review covering:

- Meta-learning foundations (MAML, Reptile, Meta-SGD)
- Few-shot learning approaches
- PEFT methods (LoRA, AdaLoRA, Prompt Tuning)
- Task embeddings and similarity learning
- HyperNetworks for adapter generation

**Key Papers:**
1. Finn et al. (2017) - MAML
2. Nichol et al. (2018) - Reptile
3. Hu et al. (2021) - LoRA
4. Achille et al. (2019) - Task2Vec

### Task Distribution Design

See [research/task_distribution_design.md](research/task_distribution_design.md) for task distribution design covering:

- Task categories (synthetic, text classification, sequence labeling)
- Episode construction strategies
- Task embedding features
- Evaluation protocols
- Success metrics

---

## Implementation Details

### Reptile Algorithm

The Reptile algorithm learns an initialization that can be quickly adapted to new tasks:

**Update Rule:**
```
θ_new = θ + β * (mean(θ_task) - θ)
```

Where:
- `θ`: Meta-parameters (learned initialization)
- `θ_task`: Task-adapted parameters after K gradient steps
- `β`: Meta-learning rate (outer_lr)

**Advantages:**
- First-order algorithm (no second-order derivatives)
- Simpler than MAML, similar performance
- Efficient on Apple Silicon with MLX
- Lower memory requirements

### Task Distribution

**Current Implementation (META-002):**
- **Linear Classification**: 2D classification with rotation/translation
- **Non-Linear Classification**: XOR, circles, spiral patterns
- **Episode Sampling**: K-shot support + query sets
- **Task Embeddings**: Handcrafted features (dataset size, complexity)

**Future Additions (META-003+):**
- Text classification tasks
- Named entity recognition
- Learned task embeddings
- Cross-domain transfer tasks

---

## Testing

### Run All Tests

```bash
# From toolkit root
uv run pytest projects/05_Meta_Learning_PEFT/tests/ -v

# From project directory
uv run pytest tests/ -v
```

### Test Coverage

- **test_task_distribution.py**: Task generation and sampling (>95% coverage)
- **test_reptile.py**: Meta-learning algorithm validation (>90% coverage)
- **test_baseline.py**: Baseline evaluation utilities (>85% coverage)

### Test Categories

```bash
# Run fast tests only
uv run pytest -m "not slow"

# Run meta-learning tests
uv run pytest -m meta_learning

# Run with coverage
uv run pytest --cov=src --cov-report=term-missing
```

---

## Configuration

### Default Configuration (configs/default.yaml)

```yaml
meta_learning:
  algorithm: "reptile"
  meta_batch_size: 4
  num_meta_iterations: 1000
  inner_lr: 0.01
  outer_lr: 0.001
  num_inner_steps: 5

task_distribution:
  task_families: [linear_classification, nonlinear_classification]
  k_shot: 5
  query_size: 50
  num_train_tasks: 100

model:
  backbone: "linear"
  hidden_dim: 128
  num_layers: 2
```

---

## Performance Metrics

### Success Criteria (META-002)

- ✅ Reptile implementation complete
- ✅ Task distribution implemented
- ✅ Baseline benchmarking infrastructure
- ✅ Comprehensive test suite (>85% coverage)
- ✅ Research documentation complete

### Target Performance (META-003+)

- **Few-Shot Accuracy**: >80% with 10 examples
- **Adaptation Speed**: 5x faster than standard fine-tuning
- **Transfer Efficiency**: 90% task similarity accuracy
- **Meta-Training Efficiency**: <1 hour on M1/M2 for 1000 iterations

---

## Known Limitations

### Current Phase (META-002)

1. **Synthetic Tasks Only**: Limited to simple 2D classification tasks
2. **No PEFT Integration**: LoRA/AdaLoRA integration planned for META-004
3. **Handcrafted Features**: Task embeddings are manually designed
4. **No Production Pipeline**: Training/evaluation scripts planned for META-003

### Future Work

1. **MAML Implementation**: Second-order meta-learning for comparison
2. **Learned Task Embeddings**: Neural network-based task representations
3. **HyperNetworks**: Task-conditional adapter generation
4. **Production Deployment**: Integration with MLOps infrastructure

---

## Dependencies

### Core Dependencies

- `mlx>=0.0.8` - Apple Silicon ML framework
- `numpy>=1.24.0` - Numerical computing
- `torch>=2.0.0` - PyTorch for compatibility
- `learn2learn>=0.2.0` - Meta-learning utilities
- `higher>=0.2.1` - Differentiable optimization

### Development Dependencies

- `pytest>=7.4.0` - Testing framework
- `pytest-cov>=4.1.0` - Coverage reporting
- `black>=23.7.0` - Code formatting
- `mypy>=1.4.0` - Type checking

---

## Contributing

This is a research project in active development. Contributions should focus on:

1. **Literature Review**: Adding relevant papers and approaches
2. **Task Distributions**: Implementing new task families
3. **Algorithms**: Alternative meta-learning methods
4. **Experiments**: Validation studies and benchmarks

---

## References

### Meta-Learning

1. Finn, C., et al. (2017). "Model-Agnostic Meta-Learning for Fast Adaptation"
2. Nichol, A., et al. (2018). "On First-Order Meta-Learning Algorithms"
3. Antoniou, A., et al. (2019). "How to train your MAML"

### PEFT Methods

4. Hu, E. J., et al. (2021). "LoRA: Low-Rank Adaptation of Large Language Models"
5. Zhang, Q., et al. (2023). "Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning"

### Task Embeddings

6. Achille, A., et al. (2019). "Task2Vec: Task Embedding for Meta-Learning"
7. Ha, D., et al. (2016). "HyperNetworks"

---

## License

Part of EfficientAI-MLX-Toolkit. See repository root for license information.

---

## Contacts

For questions or collaboration on this research project, please refer to the main toolkit repository.
