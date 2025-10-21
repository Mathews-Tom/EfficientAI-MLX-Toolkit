# Meta-Learning PEFT Integration Guide

**Version:** 0.1.0 (Research Phase)
**Status:** META-002 Active Development
**Last Updated:** 2025-10-21

---

## Table of Contents

1. [Overview](#overview)
2. [Toolkit Integration](#toolkit-integration)
3. [CLI Integration](#cli-integration)
4. [Shared Utilities Integration](#shared-utilities-integration)
5. [MLOps Integration (Future)](#mlops-integration-future)
6. [PEFT Methods Integration (Future)](#peft-methods-integration-future)
7. [External System Integration](#external-system-integration)
8. [Testing Integration](#testing-integration)

---

## Overview

### Integration Architecture

The Meta-Learning PEFT system integrates with the EfficientAI-MLX-Toolkit at multiple levels:

```
┌────────────────────────────────────────────────────────────┐
│ EfficientAI-MLX-Toolkit (Root)                             │
│  ├── Shared Utilities (utils/)                             │
│  │   ├── Logging                                           │
│  │   ├── Configuration                                     │
│  │   └── Benchmarking                                      │
│  │                                                          │
│  ├── Main CLI (efficientai_mlx_toolkit/cli.py)            │
│  │   └── Namespace:Command Dispatch                        │
│  │                                                          │
│  └── Projects                                              │
│      ├── 01_LoRA_Finetuning_MLX                           │
│      ├── 05_Meta_Learning_PEFT ← THIS PROJECT             │
│      └── ...                                               │
└────────────────────────────────────────────────────────────┘
```

### Integration Points

1. **CLI Integration**: Namespace command registration
2. **Shared Utilities**: Conditional imports for logging, config, benchmarking
3. **Testing**: Unified test infrastructure
4. **MLOps**: Experiment tracking and model registry (future)
5. **PEFT**: Adapter integration with LoRA project (future)

---

## Toolkit Integration

### Project Discovery

The toolkit automatically discovers the Meta-Learning PEFT project:

**Discovery Mechanism** (in `efficientai_mlx_toolkit/cli.py`):

```python
# Automatic project discovery
projects_dir = Path(__file__).parent.parent / "projects"
for project_path in projects_dir.iterdir():
    if project_path.is_dir() and (project_path / "src" / "cli.py").exists():
        # Extract namespace: 05_Meta_Learning_PEFT → meta-learning-peft
        namespace = project_path.name.lower()
        namespace = re.sub(r'^\d+_', '', namespace)  # Remove numeric prefix
        namespace = namespace.replace('_', '-').replace(' ', '-')

        # Import and register CLI
        # Result: "meta-learning-peft:info", "meta-learning-peft:validate"
```

**Result:**
```bash
# Commands are available as:
uv run efficientai-toolkit meta-learning-peft:info
uv run efficientai-toolkit meta-learning-peft:validate
```

### Standalone vs Unified Execution

The project supports both execution modes:

#### Unified Execution (via Toolkit CLI)

```bash
# From toolkit root
cd /path/to/EfficientAI-MLX-Toolkit
uv run efficientai-toolkit meta-learning-peft:info
```

**Advantages:**
- Access to all shared utilities
- Consistent CLI interface
- Integrated with other projects

#### Standalone Execution (Direct)

```bash
# From project directory
cd projects/05_Meta_Learning_PEFT
uv run python src/cli.py info
```

**Advantages:**
- Independent development
- Faster iteration
- No toolkit dependencies required

---

## CLI Integration

### Command Registration

**Project CLI** (`src/cli.py`):

```python
import typer

app = typer.Typer(
    name="meta-learning-peft",
    help="Meta-Learning PEFT System for rapid task adaptation"
)

@app.command()
def info():
    """Display project information."""
    typer.echo("Meta-Learning PEFT System")
    typer.echo("Status: META-002 (Research Phase)")
    # ...

@app.command()
def validate():
    """Validate project setup."""
    # Check dependencies, run tests, etc.
    # ...

# Future commands (META-003+)
@app.command()
def train(
    config: Path = typer.Option("configs/default.yaml", help="Config file"),
    num_iterations: int = typer.Option(1000, help="Meta-training iterations"),
    output_dir: Path = typer.Option("outputs", help="Output directory")
):
    """Run meta-training."""
    # ...
```

### Command Naming Convention

**Pattern**: `namespace:command`

```bash
# Project namespace (derived from directory name)
meta-learning-peft:info
meta-learning-peft:validate
meta-learning-peft:train       # Future

# Other projects follow same pattern
lora-finetuning-mlx:train
model-compression-mlx:quantize
```

### Help System Integration

```bash
# Global help
uv run efficientai-toolkit --help

# Project help
uv run efficientai-toolkit meta-learning-peft --help

# Command help
uv run efficientai-toolkit meta-learning-peft:info --help
```

---

## Shared Utilities Integration

### Conditional Import Pattern

The project uses conditional imports to support both unified and standalone execution:

**Pattern** (`src/cli.py`, `src/utils/logging.py`):

```python
# Try to import from shared toolkit utilities
try:
    from utils.logging_utils import get_logger, setup_logging
    from utils.config_manager import ConfigManager
    from utils.benchmark_runner import BenchmarkRunner
    SHARED_UTILS_AVAILABLE = True
except ImportError:
    # Fallback to local implementations for standalone execution
    import logging
    SHARED_UTILS_AVAILABLE = False

    def get_logger(name: str):
        """Local fallback logger."""
        return logging.getLogger(name)

    def setup_logging(name: str, log_dir: str | None = None):
        """Local fallback logging setup."""
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(name)
```

### Logging Integration

#### Using Shared Logging (Unified Mode)

```python
from utils.logging_utils import setup_logging

# Shared logger with Apple Silicon tracking
logger = setup_logging("meta_learning", log_dir="logs")
logger.info("Starting meta-training")
# → Logs to: logs/meta_learning_2025-10-21.log
# → Includes Apple Silicon hardware info
```

#### Using Local Logging (Standalone Mode)

```python
from src.utils.logging import setup_logger

# Local logger implementation
logger = setup_logger("meta_learning", log_dir="logs")
logger.info("Starting meta-training")
# → Basic logging without shared features
```

### Configuration Integration

#### Using Shared ConfigManager (Unified Mode)

```python
from utils.config_manager import ConfigManager

# Load config with profile support and validation
config_manager = ConfigManager("configs/default.yaml")
config = config_manager.get_config()

# Features:
# - Profile support (dev/prod)
# - Environment variable overrides
# - Schema validation
```

#### Using Local Config (Standalone Mode)

```python
from src.utils.config import MetaLearningConfig

# Simple YAML loading
config = MetaLearningConfig.from_yaml("configs/default.yaml")

# Features:
# - Dataclass-based configuration
# - Type safety
# - Default values
```

### Benchmarking Integration

#### Using Shared BenchmarkRunner (Future)

```python
from utils.benchmark_runner import BenchmarkRunner

# Hardware-aware benchmarking
runner = BenchmarkRunner("meta_learning_benchmark")
with runner.time_operation("meta_train_step"):
    metrics = learner.meta_train_step(episodes, loss_fn)

# Features:
# - Apple Silicon metrics
# - Memory profiling
# - Performance visualization
```

---

## MLOps Integration (Future)

### Experiment Tracking (META-005)

**Planned Integration** with toolkit MLOps infrastructure:

```python
from mlops.tracking import ExperimentTracker

# Initialize tracker
tracker = ExperimentTracker(
    experiment_name="meta_learning_reptile",
    run_name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
)

# Log configuration
tracker.log_params(config.to_dict())

# Log metrics during training
for iteration in range(num_iterations):
    metrics = learner.meta_train_step(episodes, loss_fn)
    tracker.log_metrics({
        "query_loss": metrics['query_loss'],
        "support_loss": metrics['support_loss'],
        "param_diff": metrics['param_diff']
    }, step=iteration)

# Log artifacts
tracker.log_artifact("checkpoints/best.npz", artifact_type="model")
tracker.log_artifact("configs/default.yaml", artifact_type="config")

tracker.end_run()
```

**Backend Support:**
- MLflow (default)
- Weights & Biases
- TensorBoard

### Model Registry (META-005)

**Planned Integration**:

```python
from mlops.registry import ModelRegistry

# Register meta-learned model
registry = ModelRegistry()
registry.register_model(
    name="reptile_meta_learner",
    version="v1.0",
    model_path="checkpoints/best.npz",
    metadata={
        "algorithm": "reptile",
        "num_tasks": 100,
        "inner_lr": 0.01,
        "outer_lr": 0.001,
        "5_shot_accuracy": 0.87
    },
    tags=["meta-learning", "reptile", "research"]
)

# Load registered model
model_info = registry.load_model("reptile_meta_learner", version="v1.0")
learner.load(model_info['path'])
```

### DVC Integration (META-005)

**Planned Integration** for data and model versioning:

```yaml
# .dvc/config
[remote "storage"]
    url = s3://efficientai-mlx-artifacts/meta-learning

# dvc.yaml
stages:
  meta_train:
    cmd: uv run efficientai-toolkit meta-learning-peft:train
    deps:
      - configs/default.yaml
      - src/meta_learning/reptile.py
      - data/task_distribution.pkl
    outs:
      - checkpoints/best.npz
    metrics:
      - metrics/train_metrics.json
```

---

## PEFT Methods Integration (Future)

### Integration with LoRA Project (META-004)

**Shared Adapter Interface**:

```python
# projects/01_LoRA_Finetuning_MLX/src/adapters/lora_adapter.py
class LoRAAdapter:
    """LoRA adapter shared across projects."""
    def __init__(self, rank, alpha, dropout):
        # ...

# projects/05_Meta_Learning_PEFT/src/adapter_generation/peft_integration.py
from projects.lora_finetuning_mlx.src.adapters import LoRAAdapter

class MetaLoRAAdapter(LoRAAdapter):
    """LoRA adapter with meta-learned initialization."""
    def __init__(self, rank, alpha, dropout, meta_init):
        super().__init__(rank, alpha, dropout)
        self.load_meta_initialization(meta_init)
```

**Cross-Project Workflow**:

```bash
# 1. Train LoRA adapter normally
uv run efficientai-toolkit lora-finetuning-mlx:train \
    --model-path mlx-community/Llama-3.2-1B \
    --dataset data/train.jsonl

# 2. Meta-learn optimal initialization
uv run efficientai-toolkit meta-learning-peft:meta-train-lora \
    --base-adapters outputs/lora_checkpoints \
    --task-distribution data/tasks.pkl

# 3. Use meta-learned initialization for new task
uv run efficientai-toolkit lora-finetuning-mlx:train \
    --model-path mlx-community/Llama-3.2-1B \
    --dataset data/new_task.jsonl \
    --meta-init outputs/meta_lora/best.npz  # Meta-learned init
```

---

## External System Integration

### Hugging Face Integration

**Model Loading**:

```python
from transformers import AutoTokenizer, AutoModel

# Load base model from Hugging Face
tokenizer = AutoTokenizer.from_pretrained("mlx-community/Llama-3.2-1B")
base_model = AutoModel.from_pretrained("mlx-community/Llama-3.2-1B")

# Wrap for meta-learning
from src.meta_learning.models import HuggingFaceWrapper
meta_model = HuggingFaceWrapper(base_model)

# Meta-train
learner = ReptileLearner(meta_model)
# ...
```

### Custom Dataset Integration

**Task Distribution from Custom Data**:

```python
from src.task_embedding.task_distribution import TaskConfig, TaskDistribution
import json

# Load tasks from JSON
with open("data/custom_tasks.json") as f:
    tasks_data = json.load(f)

# Create task configurations
configs = [
    TaskConfig(
        task_id=task['id'],
        task_family=task['family'],
        num_classes=task['num_classes'],
        input_dim=task['input_dim'],
        support_size=task['k_shot'],
        query_size=task['query_size'],
        task_params=task.get('params', {})
    )
    for task in tasks_data
]

# Create distribution
dist = TaskDistribution(configs, seed=42)
```

### MLflow Integration (Future)

```python
import mlflow

# Configure MLflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("meta_learning_reptile")

# Start run
with mlflow.start_run(run_name="reptile_experiment_1"):
    # Log parameters
    mlflow.log_params({
        "algorithm": "reptile",
        "inner_lr": config.inner_lr,
        "outer_lr": config.outer_lr,
        "num_tasks": len(train_dist.task_configs)
    })

    # Training loop
    for iteration in range(config.num_meta_iterations):
        metrics = learner.meta_train_step(episodes, loss_fn)

        # Log metrics
        mlflow.log_metrics({
            "query_loss": metrics['query_loss'],
            "support_loss": metrics['support_loss']
        }, step=iteration)

    # Log model
    mlflow.log_artifact("checkpoints/best.npz")
```

---

## Testing Integration

### Unified Test Infrastructure

**Test Discovery** (from toolkit root):

```bash
# Run all toolkit tests
uv run pytest

# Run meta-learning project tests specifically
uv run pytest projects/05_Meta_Learning_PEFT/tests/ -v

# Run via toolkit CLI
uv run efficientai-toolkit test meta-learning-peft
```

### Shared Test Utilities

**Using Shared Fixtures**:

```python
# tests/conftest.py (toolkit root)
import pytest

@pytest.fixture
def apple_silicon_available():
    """Check if Apple Silicon is available."""
    try:
        import mlx.core as mx
        return mx.metal.is_available()
    except:
        return False

# projects/05_Meta_Learning_PEFT/tests/test_reptile.py
def test_reptile_training(apple_silicon_available):
    """Test Reptile meta-training."""
    if not apple_silicon_available:
        pytest.skip("Apple Silicon not available")
    # ...
```

### Test Markers

**Shared Markers** (in `pyproject.toml`):

```toml
[tool.pytest.ini_options]
markers = [
    "apple_silicon: Tests requiring Apple Silicon hardware",
    "integration: Integration tests",
    "slow: Slow tests (>1 minute)",
    "meta_learning: Meta-learning specific tests",
    "requires_mlx: Tests requiring MLX framework"
]
```

**Usage**:

```bash
# Run only fast tests
uv run pytest -m "not slow"

# Run meta-learning tests
uv run pytest -m meta_learning

# Run integration tests
uv run pytest -m integration
```

### Coverage Integration

```bash
# Generate coverage report
uv run pytest --cov=src --cov-report=term-missing --cov-report=html

# Coverage output location
# → htmlcov/index.html
```

---

## Integration Checklist

### Pre-Integration Checklist

- [ ] Project has `src/cli.py` with Typer `app`
- [ ] Conditional imports for shared utilities implemented
- [ ] Tests pass in both standalone and unified modes
- [ ] Configuration files follow toolkit conventions
- [ ] Documentation includes integration instructions

### Post-Integration Verification

```bash
# 1. Verify CLI discovery
uv run efficientai-toolkit projects | grep meta-learning-peft

# 2. Verify commands work
uv run efficientai-toolkit meta-learning-peft:info
uv run efficientai-toolkit meta-learning-peft:validate

# 3. Verify standalone execution
cd projects/05_Meta_Learning_PEFT
uv run python src/cli.py info

# 4. Verify tests
uv run efficientai-toolkit test meta-learning-peft

# 5. Verify shared utilities
uv run python -c "
from utils.logging_utils import setup_logging
logger = setup_logging('test', log_dir='logs')
logger.info('Integration successful!')
"
```

---

## Common Integration Issues

### Issue 1: Import Errors in Unified Mode

**Problem:**
```
ImportError: cannot import name 'get_logger' from 'utils'
```

**Solution:**
```python
# Check conditional import pattern
try:
    from utils.logging_utils import get_logger
    SHARED_UTILS_AVAILABLE = True
except ImportError:
    SHARED_UTILS_AVAILABLE = False
    def get_logger(name):
        import logging
        return logging.getLogger(name)
```

### Issue 2: CLI Not Discovered

**Problem:**
```bash
uv run efficientai-toolkit meta-learning-peft:info
# Error: Unknown namespace 'meta-learning-peft'
```

**Solution:**
1. Verify `src/cli.py` exists
2. Ensure Typer `app` is defined
3. Check project directory name format: `##_Project_Name`
4. Run from toolkit root, not project directory

### Issue 3: Path Resolution Issues

**Problem:**
```
FileNotFoundError: configs/default.yaml
```

**Solution:**
```python
from pathlib import Path

# Resolve paths relative to project root
PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"
```

### Issue 4: Test Discovery Failure

**Problem:**
```bash
uv run pytest projects/05_Meta_Learning_PEFT/tests/
# No tests collected
```

**Solution:**
1. Ensure `tests/__init__.py` exists
2. Check test file naming: `test_*.py`
3. Verify test function naming: `test_*()`
4. Add project root to Python path:
   ```bash
   export PYTHONPATH="${PYTHONPATH}:$(pwd)"
   ```

---

## Migration Guide

### Migrating Standalone Project to Toolkit

**Step 1: Add CLI Integration**

```python
# src/cli.py
import typer

app = typer.Typer(name="meta-learning-peft", help="...")

# Convert existing scripts to commands
@app.command()
def train(...):
    # Existing training code
    pass
```

**Step 2: Implement Conditional Imports**

```python
# src/utils/logging.py
try:
    from utils.logging_utils import setup_logging
    SHARED_UTILS_AVAILABLE = True
except ImportError:
    SHARED_UTILS_AVAILABLE = False
    # Local fallback
```

**Step 3: Update Configuration**

```yaml
# configs/default.yaml
# Ensure paths are relative to project root
paths:
  checkpoint_dir: "checkpoints"  # Not /absolute/path
  data_dir: "data"
```

**Step 4: Verify Integration**

```bash
# From toolkit root
uv run efficientai-toolkit meta-learning-peft:info
```

---

## Best Practices

### 1. Use Conditional Imports

Always support both unified and standalone execution:

```python
try:
    from utils import shared_util
except ImportError:
    from src.utils import local_util as shared_util
```

### 2. Relative Path Resolution

Resolve paths relative to project root:

```python
PROJECT_ROOT = Path(__file__).parent.parent
CONFIG = PROJECT_ROOT / "configs" / "default.yaml"
```

### 3. Graceful Degradation

Provide fallback implementations for missing shared utilities:

```python
if SHARED_UTILS_AVAILABLE:
    logger = setup_logging("meta_learning", log_dir="logs")
else:
    import logging
    logger = logging.getLogger("meta_learning")
```

### 4. Test Both Modes

```bash
# Test unified mode
uv run efficientai-toolkit meta-learning-peft:validate

# Test standalone mode
cd projects/05_Meta_Learning_PEFT
uv run python src/cli.py validate
```

---

## Future Integration Plans

### META-005: Production Integration

- **Model Serving**: FastAPI integration for inference
- **Monitoring**: Prometheus metrics export
- **Deployment**: Docker containerization
- **CI/CD**: GitHub Actions workflows

### Cross-Project Integration

- **LoRA + Meta-Learning**: Shared adapter interface
- **Model Compression**: Meta-learned quantization strategies
- **MLOps**: Unified experiment tracking

---

**End of Integration Guide**
