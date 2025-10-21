# Meta-Learning PEFT Troubleshooting Guide

**Version:** 0.1.0 (Research Phase)
**Status:** META-002 Active Development
**Last Updated:** 2025-10-21

---

## Table of Contents

1. [Quick Diagnostics](#quick-diagnostics)
2. [Installation Issues](#installation-issues)
3. [Import Errors](#import-errors)
4. [Training Issues](#training-issues)
5. [Performance Issues](#performance-issues)
6. [MLX-Specific Issues](#mlx-specific-issues)
7. [Testing Issues](#testing-issues)
8. [Configuration Issues](#configuration-issues)
9. [Common Error Messages](#common-error-messages)
10. [Getting Help](#getting-help)

---

## Quick Diagnostics

### System Health Check

Run this script to diagnose common issues:

```bash
# Quick health check
uv run python -c "
import sys
print(f'Python: {sys.version}')

try:
    import mlx.core as mx
    print(f'MLX: {mx.__version__}')
    print(f'Metal available: {mx.metal.is_available()}')
except Exception as e:
    print(f'MLX error: {e}')

try:
    from src.meta_learning.reptile import ReptileLearner
    print('Project imports: OK')
except Exception as e:
    print(f'Project import error: {e}')
"
```

**Expected Output:**
```
Python: 3.12.0 (or 3.11+)
MLX: 0.0.8 (or higher)
Metal available: True
Project imports: OK
```

### Validation Command

```bash
# Comprehensive validation
uv run efficientai-toolkit meta-learning-peft:validate

# Or standalone
cd projects/05_Meta_Learning_PEFT
uv run python src/cli.py validate
```

---

## Installation Issues

### Issue: `uv` Command Not Found

**Error:**
```
bash: uv: command not found
```

**Solution:**
```bash
# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Add to PATH (if needed)
export PATH="$HOME/.cargo/bin:$PATH"

# Verify installation
uv --version
```

### Issue: Dependencies Not Installing

**Error:**
```
error: Failed to download `mlx`
```

**Solution 1**: Check Python version
```bash
python --version  # Must be 3.11+
uv python pin 3.12  # Pin to specific version
```

**Solution 2**: Clear cache and retry
```bash
uv cache clean
uv sync --refresh
```

**Solution 3**: Manual installation
```bash
uv add mlx numpy torch transformers learn2learn higher peft
```

### Issue: Apple Silicon Detection Fails

**Error:**
```
Platform not supported: requires Apple Silicon
```

**Solution:**
```bash
# Check architecture
uname -m
# Expected: arm64

# Check macOS version
sw_vers
# Expected: macOS 12.0+ (Monterey or later)

# If on Intel Mac, MLX will not work
# Use CPU-based alternatives or upgrade hardware
```

---

## Import Errors

### Issue: `ImportError: No module named 'mlx'`

**Error:**
```python
ImportError: No module named 'mlx'
```

**Solution:**
```bash
# Install MLX
uv add mlx

# Verify installation
uv run python -c "import mlx.core as mx; print(mx.__version__)"
```

### Issue: Shared Utilities Not Found

**Error:**
```python
ImportError: cannot import name 'setup_logging' from 'utils'
```

**Solution 1**: Running from wrong directory
```bash
# Must run from toolkit root for unified mode
cd /path/to/EfficientAI-MLX-Toolkit
uv run efficientai-toolkit meta-learning-peft:info
```

**Solution 2**: Use standalone mode
```bash
# Run from project directory
cd projects/05_Meta_Learning_PEFT
uv run python src/cli.py info
# Falls back to local implementations
```

**Solution 3**: Check conditional imports
```python
# Verify this pattern in src/cli.py
try:
    from utils.logging_utils import setup_logging
    SHARED_UTILS_AVAILABLE = True
except ImportError:
    SHARED_UTILS_AVAILABLE = False
    # Local fallback
```

### Issue: Circular Import Error

**Error:**
```python
ImportError: cannot import name 'ReptileLearner' from partially initialized module
```

**Solution:**
```python
# Check for circular dependencies
# BAD:
# reptile.py imports from models.py
# models.py imports from reptile.py

# GOOD:
# reptile.py imports from models.py
# models.py has no imports from reptile.py

# Fix by refactoring shared code to separate module
```

---

## Training Issues

### Issue: Loss Not Decreasing

**Symptom:**
```
Iteration 0: loss=2.345
Iteration 100: loss=2.340
Iteration 500: loss=2.335
# Loss barely changes
```

**Diagnosis:**
```python
# Check if gradients are flowing
def debug_training_step(learner, episodes, loss_fn):
    # Before update
    params_before = {k: v.copy() for k, v in learner.model.parameters().items()}

    # Training step
    metrics = learner.meta_train_step(episodes, loss_fn)

    # After update
    params_after = learner.model.parameters()

    # Check parameter changes
    for key in params_before:
        diff = mx.sum(mx.abs(params_after[key] - params_before[key]))
        print(f"{key}: Δ={float(diff):.6f}")

    return metrics
```

**Solutions:**

**1. Increase Learning Rates**
```yaml
# configs/default.yaml
meta_learning:
  inner_lr: 0.05    # Increase from 0.01
  outer_lr: 0.005   # Increase from 0.001
```

**2. Increase Inner Steps**
```yaml
meta_learning:
  num_inner_steps: 10  # Increase from 5
```

**3. Check Task Diversity**
```python
# Tasks may be too similar
# Add more variation:
configs = [
    TaskConfig(
        task_id=f"task_{i}",
        task_family="linear_classification",
        num_classes=2,
        input_dim=2,
        support_size=5,
        query_size=20,
        task_params={
            "rotation_angle": np.random.uniform(0, 2*np.pi),
            "translation": np.random.uniform(-2, 2, size=2)
        }
    )
    for i in range(100)  # More tasks
]
```

**4. Use Gradient Clipping**
```python
# In learner.meta_train_step()
def clip_gradients(grads, max_norm=1.0):
    """Clip gradients by global norm."""
    total_norm = mx.sqrt(sum(mx.sum(g**2) for g in grads.values()))
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
        grads = {k: v * clip_coef for k, v in grads.items()}
    return grads
```

### Issue: NaN Loss

**Error:**
```
Iteration 50: loss=nan
```

**Diagnosis:**
```python
# Check for numerical issues
def check_numerical_stability(model, x):
    logits = model(x)
    print(f"Logits min: {mx.min(logits)}")
    print(f"Logits max: {mx.max(logits)}")
    print(f"Logits mean: {mx.mean(logits)}")
    print(f"Contains NaN: {mx.any(mx.isnan(logits))}")
    print(f"Contains Inf: {mx.any(mx.isinf(logits))}")
```

**Solutions:**

**1. Reduce Learning Rate**
```yaml
meta_learning:
  inner_lr: 0.001   # Much smaller
  outer_lr: 0.0001
```

**2. Add Gradient Clipping**
```python
# Clip gradients before update
grads = clip_gradients(grads, max_norm=1.0)
```

**3. Check Input Data**
```python
# Normalize inputs
def normalize_data(x):
    mean = mx.mean(x, axis=0)
    std = mx.std(x, axis=0) + 1e-8
    return (x - mean) / std

# In task distribution
support_x = normalize_data(support_x)
query_x = normalize_data(query_x)
```

**4. Use Stable Loss Function**
```python
# Replace direct softmax + cross-entropy with stable version
def stable_cross_entropy_loss(logits, targets):
    """Numerically stable cross-entropy."""
    # MLX's built-in loss is already stable
    return mx.mean(nn.losses.cross_entropy(logits, targets))
```

### Issue: Overfitting to Training Tasks

**Symptom:**
```
Train loss: 0.05
Val loss: 1.50
```

**Solutions:**

**1. More Training Tasks**
```python
# Increase task diversity
configs = [
    TaskConfig(...)
    for i in range(200)  # Increase from 100
]
```

**2. Regularization**
```python
# Add L2 regularization
def l2_regularization(params, weight_decay=0.0001):
    return weight_decay * sum(mx.sum(p**2) for p in params.values())

# In loss computation
loss = cross_entropy_loss(logits, targets) + l2_regularization(model.parameters())
```

**3. Early Stopping**
```python
best_val_loss = float('inf')
patience = 50
patience_counter = 0

for iteration in range(num_iterations):
    # ... training ...

    if iteration % eval_interval == 0:
        val_results = learner.evaluate(val_episodes, loss_fn)
        if val_results['final_loss'] < best_val_loss:
            best_val_loss = val_results['final_loss']
            patience_counter = 0
            learner.save("checkpoints/best.npz")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at iteration {iteration}")
            break
```

---

## Performance Issues

### Issue: Slow Training Speed

**Symptom:**
```
Iteration 0: 5.2 seconds
Iteration 1: 5.3 seconds
# Too slow (should be <1 second per iteration)
```

**Diagnosis:**
```python
import time

def profile_training_step(learner, episodes, loss_fn):
    times = {}

    start = time.time()
    # Sample tasks
    times['sampling'] = time.time() - start

    start = time.time()
    # Meta-train step
    metrics = learner.meta_train_step(episodes, loss_fn)
    times['meta_step'] = time.time() - start

    for key, duration in times.items():
        print(f"{key}: {duration:.3f}s")

    return metrics
```

**Solutions:**

**1. Reduce Batch Size**
```yaml
meta_learning:
  meta_batch_size: 2  # Reduce from 4
```

**2. Reduce Inner Steps**
```yaml
meta_learning:
  num_inner_steps: 3  # Reduce from 5
```

**3. Use Smaller Model**
```python
model = SimpleClassifier(
    input_dim=2,
    hidden_dim=32,   # Reduce from 64
    num_classes=2
)
```

**4. Ensure MLX is Using Metal**
```python
import mlx.core as mx

# Check Metal is available
assert mx.metal.is_available(), "Metal not available!"

# Set default device (should be automatic)
mx.set_default_device(mx.gpu)
```

**5. Optimize Episode Sampling**
```python
# Pre-generate episodes instead of sampling on-the-fly
def pregenerate_episodes(dist, num_episodes):
    """Pre-generate episodes for faster training."""
    return [dist.sample_episode()[1:] for _ in range(num_episodes)]

# Before training
all_episodes = pregenerate_episodes(train_dist, num_iterations * meta_batch_size)

# During training
for iteration in range(num_iterations):
    start_idx = iteration * meta_batch_size
    end_idx = start_idx + meta_batch_size
    episodes = all_episodes[start_idx:end_idx]
    metrics = learner.meta_train_step(episodes, loss_fn)
```

### Issue: High Memory Usage

**Symptom:**
```
MemoryError: Cannot allocate memory
```

**Solutions:**

**1. Reduce Batch Size**
```yaml
meta_learning:
  meta_batch_size: 2
```

**2. Reduce Query Set Size**
```python
TaskConfig(
    # ...
    query_size=10  # Reduce from 20
)
```

**3. Clear Cache Periodically**
```python
import gc

for iteration in range(num_iterations):
    metrics = learner.meta_train_step(episodes, loss_fn)

    if iteration % 100 == 0:
        gc.collect()  # Free memory
```

**4. Use Gradient Accumulation**
```python
def meta_train_step_with_accumulation(learner, episodes, loss_fn, accumulation_steps=2):
    """Meta-train with gradient accumulation."""
    accumulated_params = {}

    for i in range(0, len(episodes), accumulation_steps):
        batch = episodes[i:i+accumulation_steps]
        metrics = learner.meta_train_step(batch, loss_fn)
        # Accumulate parameter updates
        # ...

    return metrics
```

---

## MLX-Specific Issues

### Issue: Metal Not Available

**Error:**
```python
RuntimeError: Metal is not available
```

**Diagnosis:**
```bash
# Check system
system_profiler SPDisplaysDataType | grep "Metal"

# Check MLX
uv run python -c "import mlx.core as mx; print(mx.metal.is_available())"
```

**Solutions:**

**1. Update macOS**
```bash
# Metal requires macOS 12.0+
sw_vers
# If < 12.0, update macOS
```

**2. Reinstall MLX**
```bash
uv remove mlx
uv add mlx --reinstall
```

**3. Use CPU Backend (Fallback)**
```python
import mlx.core as mx

if not mx.metal.is_available():
    print("Warning: Metal not available, using CPU")
    mx.set_default_device(mx.cpu)
```

### Issue: MLX Version Mismatch

**Error:**
```
AttributeError: module 'mlx.core' has no attribute 'metal'
```

**Solution:**
```bash
# Check MLX version
uv run python -c "import mlx; print(mlx.__version__)"

# Update MLX
uv add mlx --upgrade

# Verify version >= 0.0.8
```

---

## Testing Issues

### Issue: Tests Failing on Import

**Error:**
```
ModuleNotFoundError: No module named 'src'
```

**Solution:**
```bash
# Set PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or run from project root
cd projects/05_Meta_Learning_PEFT
uv run pytest tests/
```

### Issue: Tests Pass Individually but Fail Together

**Symptom:**
```bash
uv run pytest tests/test_reptile.py  # PASS
uv run pytest tests/test_task_distribution.py  # PASS
uv run pytest tests/  # FAIL
```

**Diagnosis:**
```python
# Check for shared state
# BAD:
GLOBAL_STATE = {}  # Shared across tests

# GOOD:
@pytest.fixture
def fresh_state():
    return {}  # New state per test
```

**Solution:**
```python
# Use pytest fixtures for isolation
@pytest.fixture
def learner():
    """Create fresh learner for each test."""
    model = SimpleClassifier(input_dim=2, hidden_dim=64, num_classes=2)
    return ReptileLearner(model)

def test_meta_train_step(learner):  # Inject fixture
    # Test with isolated learner
    # ...
```

### Issue: Flaky Tests

**Symptom:**
```
Sometimes PASS, sometimes FAIL
```

**Solution:**
```python
# Set random seeds
import numpy as np
import mlx.core as mx

@pytest.fixture(autouse=True)
def set_seeds():
    """Set seeds for reproducibility."""
    np.random.seed(42)
    mx.random.seed(42)

# Use deterministic operations
dist = TaskDistribution(configs, seed=42)  # Always use seed
```

---

## Configuration Issues

### Issue: Config File Not Found

**Error:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'configs/default.yaml'
```

**Solution:**
```python
from pathlib import Path

# Resolve path relative to project root
PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"

# Verify file exists
assert CONFIG_PATH.exists(), f"Config not found: {CONFIG_PATH}"
```

### Issue: Invalid Configuration

**Error:**
```
ValidationError: inner_lr must be positive
```

**Solution:**
```python
# Add validation to config
@dataclass
class MetaLearningConfig:
    inner_lr: float = 0.01

    def __post_init__(self):
        """Validate configuration."""
        if self.inner_lr <= 0:
            raise ValueError(f"inner_lr must be positive, got {self.inner_lr}")
        if self.outer_lr <= 0:
            raise ValueError(f"outer_lr must be positive, got {self.outer_lr}")
        # ...
```

---

## Common Error Messages

### `RuntimeError: Graph evaluation error`

**Cause:** MLX computation graph issue

**Solution:**
```python
# Ensure all operations are MLX-compatible
# Convert NumPy arrays to MLX
x_mlx = mx.array(x_numpy)

# Evaluate lazy computations
result = mx.eval(computation)
```

### `ValueError: Invalid shape`

**Cause:** Dimension mismatch

**Solution:**
```python
# Check tensor shapes
print(f"Input shape: {x.shape}")
print(f"Expected: (batch_size, input_dim)")

# Reshape if needed
x = mx.reshape(x, (batch_size, input_dim))
```

### `AssertionError: Expected accuracy > 0.5`

**Cause:** Model not learning

**Solution:**
1. Check learning rates (may be too low)
2. Verify loss is decreasing
3. Ensure gradients are non-zero
4. Check task distribution (may be too hard)

---

## Getting Help

### Self-Help Resources

1. **Check Logs**
   ```bash
   tail -f logs/meta_learning_*.log
   ```

2. **Run Diagnostics**
   ```bash
   uv run efficientai-toolkit meta-learning-peft:validate
   ```

3. **Review Tests**
   ```bash
   uv run pytest tests/ -v --tb=short
   ```

4. **Check Examples**
   - Review `examples/` directory (META-003+)
   - Read `tests/` for usage patterns

### Debug Mode

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Verbose MLX
import mlx.core as mx
mx.set_default_device(mx.gpu)
print(f"Default device: {mx.default_device()}")
```

### Minimal Reproducible Example

When reporting issues, provide:

```python
import mlx.core as mx
from src.meta_learning.reptile import ReptileLearner
from src.meta_learning.models import SimpleClassifier, cross_entropy_loss
from src.task_embedding.task_distribution import TaskConfig, TaskDistribution

# Minimal setup
configs = [
    TaskConfig(
        task_id=f"task_{i}",
        task_family="linear_classification",
        num_classes=2,
        input_dim=2,
        support_size=5,
        query_size=20
    )
    for i in range(5)  # Small number for testing
]
dist = TaskDistribution(configs, seed=42)

model = SimpleClassifier(input_dim=2, hidden_dim=64, num_classes=2)
learner = ReptileLearner(model)

# Reproduce issue
episodes = [dist.sample_episode()[1:] for _ in range(2)]
metrics = learner.meta_train_step(episodes, cross_entropy_loss)  # Error here?
```

### Community Support

- **GitHub Issues**: Main toolkit repository
- **Documentation**: `docs/` directory
- **Tests**: `tests/` for usage examples

---

## Appendix: Diagnostic Scripts

### Full System Diagnostic

```python
#!/usr/bin/env python
"""Comprehensive system diagnostic."""
import sys
from pathlib import Path

def check_python():
    """Check Python version."""
    print(f"Python: {sys.version}")
    assert sys.version_info >= (3, 11), "Python 3.11+ required"
    print("✓ Python version OK")

def check_mlx():
    """Check MLX installation."""
    try:
        import mlx.core as mx
        print(f"MLX: {mx.__version__}")
        print(f"Metal: {mx.metal.is_available()}")
        assert mx.metal.is_available(), "Metal not available"
        print("✓ MLX OK")
    except Exception as e:
        print(f"✗ MLX error: {e}")

def check_imports():
    """Check project imports."""
    try:
        from src.meta_learning.reptile import ReptileLearner
        from src.task_embedding.task_distribution import TaskDistribution
        print("✓ Project imports OK")
    except Exception as e:
        print(f"✗ Import error: {e}")

def check_config():
    """Check configuration files."""
    config_path = Path("configs/default.yaml")
    assert config_path.exists(), f"Config not found: {config_path}"
    print(f"✓ Config found: {config_path}")

def check_tests():
    """Check test suite."""
    import subprocess
    result = subprocess.run(
        ["uv", "run", "pytest", "tests/", "-v"],
        capture_output=True,
        text=True
    )
    if result.returncode == 0:
        print("✓ All tests passing")
    else:
        print(f"✗ Tests failing:\n{result.stdout}")

if __name__ == "__main__":
    print("=== Meta-Learning PEFT System Diagnostic ===\n")
    check_python()
    check_mlx()
    check_imports()
    check_config()
    check_tests()
    print("\n=== Diagnostic Complete ===")
```

Run with:
```bash
cd projects/05_Meta_Learning_PEFT
uv run python scripts/diagnostic.py
```

---

**End of Troubleshooting Guide**
