# Development Standards Compliance

This document outlines the compliance improvements made to the EfficientAI-MLX-Toolkit to follow all established development standards.

## ✅ Compliance Improvements Made

### 1. Import Standards Compliance

**Before (Non-compliant):**

```python
# Defensive imports for required dependencies
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None

def require_matplotlib():
    if not MATPLOTLIB_AVAILABLE:
        raise PlottingError("Matplotlib not available")
```

**After (Compliant):**

```python
# Direct imports for dependencies declared in pyproject.toml
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Note: Install with: uv sync --extra visualization
```

### 2. Modern Python Type Annotations

**Before (Non-compliant):**

```python
from typing import List, Dict, Optional, Union

def process_data(items: List[str]) -> Optional[Dict[str, Any]]:
    pass
```

**After (Compliant):**

```python
from collections.abc import Mapping, Sequence

def process_data(items: list[str]) -> dict[str, str | int | float] | None:
    pass
```

### 3. Dependency Management

**Updated pyproject.toml:**

- Added proper dependency groups for optional features
- Removed defensive import patterns
- Declared all required dependencies explicitly

```toml
[project.optional-dependencies]
visualization = [
    "matplotlib>=3.7.0",
    "seaborn>=0.12.0",
    "plotly>=5.15.0",
]
monitoring = [
    "mlflow>=2.8.0",
    "psutil>=5.9.0",
]
```

### 4. Error Handling Improvements

**Before:**

```python
except Exception as e:
    raise CustomError("Operation failed")
```

**After:**

```python
except Exception as e:
    raise CustomError("Operation failed") from e
```

### 5. Logging Standards

**Before:**

```python
logger.info(f"Processing {count} items")
```

**After:**

```python
logger.info("Processing %d items", count)
```

## Files Updated for Compliance

1. **`utils/plotting_utils.py`** - Removed defensive imports for matplotlib, seaborn, pandas
2. **`utils/config_manager.py`** - Removed defensive imports for yaml, tomli-w
3. **`utils/benchmark_runner.py`** - Removed defensive imports for psutil, kept MLX as truly optional
4. **`pyproject.toml`** - Added proper dependency groups, moved core packages to dependencies
5. **`utils/__init__.py`** - Fixed import structure and exports
6. **`efficientai_mlx_toolkit/__init__.py`** - Updated exports

### Key Dependency Changes

- **Moved to core dependencies**: `psutil`, `tomli-w`, `pandas` (commonly used)
- **Kept as optional**: `matplotlib`, `seaborn` (visualization group)
- **Truly optional**: `mlx` (Apple Silicon specific hardware)

## Installation Instructions

### Core Installation

```bash
uv sync
```

### With Visualization Support

```bash
uv sync --extra visualization
```

### With Monitoring Support

```bash
uv sync --extra monitoring
```

### Full Development Setup

```bash
uv sync --extra dev --extra visualization --extra monitoring --extra apple-silicon
```

## Key Principles Applied

1. **No Defensive Imports**: Required dependencies are declared in pyproject.toml and imported directly
2. **Modern Type Hints**: Use Python 3.11+ syntax throughout
3. **Proper Error Chaining**: Always use `from e` for exception chaining
4. **Pathlib Usage**: All file operations use pathlib.Path
5. **Lazy Logging**: Use `%` formatting in logging calls
6. **UV Package Management**: All package operations use uv instead of pip

## Benefits

- **Faster Startup**: No defensive import overhead
- **Better Error Messages**: Clear failure modes when dependencies are missing
- **Improved Type Safety**: Modern type annotations provide better IDE support
- **Consistent Code Style**: Follows all established development standards
- **Easier Maintenance**: Clear dependency management and error handling
