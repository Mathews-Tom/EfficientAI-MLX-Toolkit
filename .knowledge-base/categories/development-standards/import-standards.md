---
title: "Import Standards for DSPy Integration Framework"
category: "development-standards"
tags: ["python", "imports", "dependencies", "dspy", "standards"]
difficulty: "intermediate"
last_updated: "2025-08-15"
contributors: ["Tom Mathews"]
---

# Import Standards for DSPy Integration Framework

## Problem/Context

This knowledge applies when working with the DSPy Integration Framework to ensure clean, reliable, and maintainable code. Use these standards when:

- Writing new modules in the `dspy_toolkit` package
- Handling dependencies in Python code
- Setting up import statements for required vs optional dependencies
- Reviewing code that uses try/catch blocks for imports
- Ensuring proper dependency declaration in `pyproject.toml`

## Solution/Pattern

**Core Principle: Avoid defensive try/catch blocks for system dependencies that are declared in `pyproject.toml`.**

All required dependencies should be properly declared in the project's `pyproject.toml` file and installed as part of the system environment setup. Using try/catch blocks for these imports creates unnecessary complexity and can mask real configuration issues.

### Required Dependencies Pattern

✅ **Correct approach for required dependencies:**

```python
# Direct imports for required dependencies
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from dspy_toolkit.types import DSPyConfig
```

### Optional Dependencies Pattern

✅ **Acceptable defensive imports for optional dependencies:**

```python
# Optional Apple Silicon optimization
try:
    import mlx.core as mx
    import mlx.nn as nn
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    mx = None
    nn = None
    # Graceful fallback to CPU/MPS implementation

def require_mlx():
    """Ensure MLX is available for Apple Silicon optimization."""
    if not MLX_AVAILABLE:
        raise DSPyIntegrationError(
            "MLX is required for Apple Silicon optimization. "
            "Install with: uv add 'dspy-toolkit[apple-silicon]'"
        )
```

### Import Ordering Rules

Follow PEP 8 import ordering with these specific guidelines:

1. **Within each section**, imports should be alphabetically sorted
2. **Use absolute imports** for clarity and consistency
3. **Group related imports** from the same package on consecutive lines
4. **Separate each section** with exactly one blank line

```python
# ✅ Correct import ordering
import asyncio
import logging
import os
from collections.abc import Callable, Mapping
from datetime import datetime
from pathlib import Path

import dspy
import mlx.core as mx
import mlx.nn as nn
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field

from dspy_toolkit.exceptions import DSPyIntegrationError
from dspy_toolkit.providers.mlx_provider import MLXLLMProvider
from dspy_toolkit.types import DSPyConfig
```

```python
# ❌ Incorrect import ordering
from dspy_toolkit.types import DSPyConfig  # Local import mixed with others
import dspy
from pathlib import Path
import logging  # Standard library not grouped
from fastapi import FastAPI
from dspy_toolkit.exceptions import DSPyIntegrationError
import mlx.core as mx
```

## Code Example

### Dependency Declaration in pyproject.toml

```toml
[project]
dependencies = [
    # Core required dependencies
    "fastapi>=0.100.0",
    "uvicorn>=0.24.0",
    "pydantic>=2.0.0",
    "dspy-ai>=2.4.0",
    "litellm>=1.40.0",
    "mlx>=0.15.0",
    "mlx-lm>=0.15.0",
]

[project.optional-dependencies]
apple-silicon = [
    "mlx>=0.15.0",
    "mlx-lm>=0.15.0",
]
monitoring = [
    "mlflow>=2.8.0",
    "prometheus-client>=0.17.0",
]
dev = [
    "pytest>=8.4.1",
    "black>=23.0.0",
    "mypy>=1.4.0",
]
```

### Proper Import Structure

Imports must be organized in three distinct sections, separated by blank lines:

1. **Standard library imports** - Python built-in modules
2. **Third-party imports** - External packages (declared in pyproject.toml)
3. **Local/project imports** - Internal project modules

```python
"""
Example module showing proper import standards.
"""

# Standard library imports
import logging
import os
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path

# Third-party imports (declared in pyproject.toml)
import dspy
import mlx.core as mx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Local/project imports
from dspy_toolkit.exceptions import DSPyIntegrationError
from dspy_toolkit.types import DSPyConfig

# Optional dependency handling
try:
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    mx = None

logger = logging.getLogger(__name__)

class ExampleService:
    """Example service demonstrating proper import usage."""
    
    def __init__(self, config: DSPyConfig):
        self.config = config
        
    def require_mlx_optimization(self) -> bool:
        """Check if MLX optimization is available."""
        if not MLX_AVAILABLE:
            raise DSPyIntegrationError(
                "MLX optimization requires Apple Silicon. "
                "Install with: uv add 'dspy-toolkit[apple-silicon]'"
            )
        return True
```

## Gotchas/Pitfalls

- **Don't use defensive imports for required dependencies**: This masks configuration issues and makes debugging harder
- **Missing dependency declarations**: Always declare required dependencies in `pyproject.toml`
- **Inconsistent error messages**: Provide clear installation instructions when optional dependencies are missing
- **Mock object complexity**: Avoid creating complex mock objects for missing dependencies
- **Import order**: Follow PEP 8 import ordering (standard library, third-party, local)

## Performance Impact

- **Faster startup**: Direct imports fail fast if dependencies are missing
- **Reduced complexity**: No need to maintain availability flags throughout the codebase
- **Better testing**: Tests will fail immediately if environment is not properly set up
- **Cleaner code**: Eliminates conditional logic for required functionality

## Related Knowledge

- [Python Development Environment Setup](./python-development-environment-setup.md) - Setting up uv and dependencies
- [Python Code Quality Standards](./python-code-quality-standards.md) - General code quality guidelines
- [Modern Python Type Annotations Migration Guide](./modern-python-type-annotations-migration-guide.md) - Type hints for imports