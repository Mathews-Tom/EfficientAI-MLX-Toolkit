# Task Distribution Design for Meta-Learning PEFT

**Phase:** META-002 Research and Prototyping
**Date:** 2025-10-20
**Status:** Design Document

---

## Overview

This document defines the task distribution design for meta-learning PEFT methods. A well-designed task distribution is critical for effective meta-learning, enabling the model to learn generalizable adaptation strategies.

---

## 1. Task Distribution Principles

### 1.1 Diversity Requirements
- **Domain Coverage:** Multiple domains (text, code, dialogue)
- **Task Types:** Classification, generation, extraction
- **Difficulty Levels:** Simple to complex tasks
- **Data Characteristics:** Varying dataset sizes and distributions

### 1.2 Similarity Structure
- **Within-Family Similarity:** Tasks in same domain should be related
- **Cross-Family Transfer:** Some overlap between domains
- **Novel Task Adaptation:** Held-out tasks for validation

### 1.3 Computational Constraints
- **Task Complexity:** Manageable for rapid iteration
- **Dataset Size:** Small enough for meta-training efficiency
- **Evaluation Speed:** Fast task adaptation evaluation

---

## 2. Task Categories

### 2.1 Phase 1: Synthetic Tasks (META-002)

#### Binary Classification Tasks
**Purpose:** Controlled environment for initial validation

**Task Family 1: Linear Separable**
- 2D point classification with varying decision boundaries
- Rotation, translation, and scaling transformations
- 10 support examples, 50 query examples
- Easy baseline for debugging

**Task Family 2: Non-Linear Separable**
- XOR-like patterns with varying complexity
- Concentric circles, spirals, checkerboard
- Tests meta-learner's adaptation capacity

**Task Family 3: High-Dimensional**
- 100-dimensional synthetic data
- Sparse relevant features
- Tests feature selection and generalization

**Characteristics:**
```python
SyntheticTaskConfig:
    num_classes: 2
    input_dim: [2, 10, 100]
    support_size: [5, 10, 20]
    query_size: 50
    noise_level: [0.0, 0.1, 0.2]
    transformation_types: ["rotation", "scaling", "translation"]
```

### 2.2 Phase 2: Text Classification (META-003)

#### Sentiment Analysis Family
**Domains:** Product reviews, movie reviews, tweets
**Classes:** Binary (positive/negative) or 3-class (pos/neu/neg)
**Support Examples:** 5-20 per class
**Transfer:** Cross-domain sentiment transfer

**Task Examples:**
1. Electronics reviews → Book reviews
2. Movie reviews → Restaurant reviews
3. Twitter sentiment → Product feedback

#### Topic Classification Family
**Domains:** News articles, scientific papers, social media
**Classes:** 5-10 categories
**Support Examples:** 10-50 per class
**Transfer:** Domain-specific to general topics

**Task Examples:**
1. ArXiv categories (cs.AI, cs.CV, cs.LG)
2. News topics (politics, sports, technology)
3. Product categories (electronics, clothing, books)

### 2.3 Phase 3: Sequence Labeling (META-004)

#### Named Entity Recognition (NER) Family
**Entity Types:** Person, Organization, Location, Misc
**Domains:** News, biomedical, social media
**Support:** 50-200 annotated sentences
**Transfer:** Entity type and domain transfer

#### Intent Classification Family
**Intents:** 10-30 user intents
**Domains:** Customer service, virtual assistant, FAQ
**Support:** 10-50 examples per intent
**Transfer:** Cross-domain intent understanding

### 2.4 Phase 4: Generation Tasks (Future)

#### Summarization Family
**Styles:** Abstractive, extractive, bullet points
**Domains:** News, scientific papers, dialogue
**Metrics:** ROUGE, BLEU, human evaluation

#### Code Generation Family
**Languages:** Python, JavaScript, SQL
**Tasks:** Function completion, bug fixing, documentation
**Metrics:** Pass@k, code quality, execution success

---

## 3. Task Sampling Strategy

### 3.1 Episode Construction

**Meta-Training Episode:**
```python
Episode:
    task: T ~ p(T)  # Sample task from distribution
    support_set: S = {(x_i, y_i)} for i in 1..K  # K-shot support
    query_set: Q = {(x_j, y_j)} for j in 1..N  # Query examples
    task_embedding: e_T = embed(S)  # Optional task representation
```

**Episode Types:**
1. **Standard Episode:** Fixed K-shot across all tasks
2. **Variable-Shot Episode:** K varies per episode (1-20 examples)
3. **Imbalanced Episode:** Varying examples per class
4. **Domain-Shift Episode:** Support and query from different distributions

### 3.2 Task Curriculum

**Stage 1: Easy Tasks (Weeks 1-2)**
- Simple synthetic tasks
- High support shot count (20 examples)
- Low noise, clear patterns
- **Goal:** Validate meta-learning setup

**Stage 2: Moderate Tasks (Weeks 3-4)**
- Text classification with domain similarity
- Medium support (10 examples)
- Moderate noise and complexity
- **Goal:** Learn task adaptation

**Stage 3: Hard Tasks (Weeks 5-6)**
- Cross-domain transfer
- Low support (5 examples)
- High task diversity
- **Goal:** Test generalization

**Stage 4: Held-Out Validation (Week 7+)**
- Novel task families
- Extreme few-shot (1-3 examples)
- Distribution shift
- **Goal:** Measure meta-learning effectiveness

---

## 4. Task Embedding Features

### 4.1 Handcrafted Features (Phase 1)

**Dataset Statistics:**
- Number of examples
- Input dimensionality
- Number of classes
- Class imbalance ratio

**Data Characteristics:**
- Mean/std of features
- Sparsity level
- Correlation structure
- Signal-to-noise ratio

**Domain Information:**
- Domain type (text, vision, tabular)
- Language (for NLP)
- Vocabulary size
- Sequence length statistics

### 4.2 Learned Features (Phase 2+)

**Task2Vec Approach:**
- Fisher Information Matrix diagonal
- Model gradient statistics
- Task-specific representations

**Prototypical Features:**
- Class prototype embeddings
- Inter-class distances
- Intra-class variance

**HyperNetwork Input:**
- Concatenation of handcrafted + learned features
- Dimensionality: 128-256
- Normalized to unit sphere

---

## 5. Evaluation Protocol

### 5.1 Meta-Training Metrics

**Inner Loop (Task Adaptation):**
- Support set loss
- Query set loss after K gradient steps
- Adaptation speed (loss decrease per step)

**Outer Loop (Meta-Learning):**
- Meta-loss across task batch
- Meta-gradient norm
- Task diversity in batch

### 5.2 Meta-Validation Metrics

**Few-Shot Performance:**
- 1-shot accuracy
- 5-shot accuracy
- 10-shot accuracy
- 20-shot accuracy

**Transfer Quality:**
- Within-family transfer
- Cross-family transfer
- Novel task adaptation

**Efficiency Metrics:**
- Adaptation steps required
- Total gradient steps
- Wallclock time
- Memory usage

### 5.3 Meta-Testing Protocol

**Held-Out Tasks:**
- 20% of task families held out
- No overlap with meta-training
- Evaluate final performance

**Baselines:**
1. Random initialization
2. Pretrained model fine-tuning
3. Transfer from single task
4. Multi-task learning

**Success Criteria:**
- Meta-learned init > random init by >20%
- <10 adaptation steps for good performance
- >80% accuracy with 10 examples

---

## 6. Implementation Plan (META-002)

### 6.1 Task Generator Interface

```python
from dataclasses import dataclass
from typing import Callable, List, Tuple
import mlx.core as mx

@dataclass
class TaskConfig:
    """Configuration for a single task."""
    task_id: str
    task_family: str
    num_classes: int
    input_dim: int
    support_size: int
    query_size: int
    domain: str
    difficulty: str  # "easy", "medium", "hard"

class TaskDistribution:
    """Meta-learning task distribution."""

    def __init__(self, task_configs: List[TaskConfig]):
        self.task_configs = task_configs
        self.task_families = self._group_by_family()

    def sample_task(self) -> Task:
        """Sample a random task from distribution."""
        pass

    def sample_episode(
        self,
        support_size: int,
        query_size: int
    ) -> Tuple[mx.array, mx.array, mx.array, mx.array]:
        """Sample support and query sets for meta-training."""
        pass

    def get_task_embedding(self, task: Task) -> mx.array:
        """Extract task embedding features."""
        pass

class Task:
    """Single task instance."""

    def __init__(self, config: TaskConfig):
        self.config = config
        self.data_generator = self._create_generator()

    def sample_data(
        self,
        num_samples: int
    ) -> Tuple[mx.array, mx.array]:
        """Sample data from task distribution."""
        pass

    def compute_task_features(self) -> dict:
        """Compute handcrafted task features."""
        pass
```

### 6.2 Synthetic Task Generators

**Linear Classifier Generator:**
```python
class LinearTaskGenerator:
    """Generate linear classification tasks."""

    def __init__(self, input_dim: int = 2):
        self.input_dim = input_dim

    def generate_task(
        self,
        rotation: float = 0.0,
        translation: Tuple[float, float] = (0.0, 0.0)
    ) -> Task:
        """Generate task with specified transformations."""
        # Create random decision boundary
        # Apply rotation and translation
        # Return Task instance
        pass
```

**Non-Linear Task Generator:**
```python
class NonLinearTaskGenerator:
    """Generate non-linear classification tasks."""

    def __init__(self, pattern_type: str = "xor"):
        self.pattern_type = pattern_type

    def generate_task(
        self,
        noise_level: float = 0.1
    ) -> Task:
        """Generate non-linear pattern with noise."""
        # Create XOR, circle, or spiral pattern
        # Add Gaussian noise
        # Return Task instance
        pass
```

### 6.3 Episode Sampler

```python
class EpisodeSampler:
    """Sample episodes for meta-training."""

    def __init__(
        self,
        task_distribution: TaskDistribution,
        k_shot: int = 5,
        query_size: int = 50
    ):
        self.task_distribution = task_distribution
        self.k_shot = k_shot
        self.query_size = query_size

    def sample_batch(
        self,
        batch_size: int
    ) -> List[Tuple]:
        """Sample batch of episodes."""
        episodes = []
        for _ in range(batch_size):
            task = self.task_distribution.sample_task()
            support_x, support_y = task.sample_data(self.k_shot)
            query_x, query_y = task.sample_data(self.query_size)
            episodes.append((support_x, support_y, query_x, query_y))
        return episodes
```

---

## 7. Validation Experiments

### Experiment 1: Task Distribution Coverage
**Goal:** Validate task diversity and coverage
**Method:** Visualize task embeddings in 2D (t-SNE)
**Success:** Clear clustering by task family, coverage of feature space

### Experiment 2: Episode Sampling
**Goal:** Verify episode construction is correct
**Method:** Sample 1000 episodes, check statistics
**Success:** Support/query sizes correct, no data leakage

### Experiment 3: Baseline Performance
**Goal:** Measure performance without meta-learning
**Method:** Random init on each task, measure few-shot accuracy
**Success:** Establish baseline for meta-learning comparison

---

## 8. Next Steps

### Immediate (Week 1)
1. Implement TaskDistribution and Task classes
2. Create synthetic task generators (linear, non-linear)
3. Implement EpisodeSampler
4. Validate with visualization and statistics

### Short-term (Week 2-3)
1. Add handcrafted task embedding features
2. Implement baseline evaluation protocol
3. Create task curriculum stages
4. Set up experiment tracking

### Medium-term (Week 4+)
1. Expand to text classification tasks
2. Implement learned task embeddings
3. Add cross-domain transfer tasks
4. Production-ready task distribution

---

## References

1. Vinyals et al. (2016) - Episodic training for few-shot learning
2. Achille et al. (2019) - Task2Vec task embeddings
3. Finn et al. (2017) - MAML task distribution design
4. Triantafillou et al. (2020) - Meta-Dataset task diversity
