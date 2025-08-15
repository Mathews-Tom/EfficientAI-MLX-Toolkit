---
title: "Python Development Environment Setup"
category: "development-standards"
tags: ['python', 'uv', 'environment']
difficulty: "intermediate"
last_updated: "2025-08-14"
contributors: ['Tom Mathews']
---

# Python Development Environment Setup

## Problem/Context

Setting up a consistent Python development environment is crucial for the EfficientAI-MLX-Toolkit project. This knowledge applies when:

- Starting development on the project for the first time
- Setting up CI/CD pipelines
- Onboarding new team members
- Ensuring consistent package management across the team
- Working with Apple Silicon optimizations that require specific Python versions

## Solution/Pattern

### Python Environment Requirements

- **Python Version**: 3.11+ required for optimal Apple Silicon support
- **Package Manager**: `uv` (Universal Versioning) - **NEVER use pip or other package managers**

### UV Package Management

UV is the mandated package manager for this project. It provides:

- Faster dependency resolution
- Better reproducible builds
- Improved virtual environment management
- Better compatibility with Apple Silicon

### Setup Steps

1. **Install UV**:

   ```bash
   # Using pip (one-time only)
   pip install uv
   
   # Or using homebrew on macOS
   brew install uv
   ```

2. **Project Setup**:

   ```bash
   # Clone the repository
   git clone https://github.com/Mathews-Tom/EfficientAI-MLX-Toolkit
   cd EfficientAI-MLX-Toolkit
   
   # Install dependencies
   uv sync
   
   # Install in development mode
   uv pip install -e .
   ```

3. **Daily Usage**:

   ```bash
   # Add new packages
   uv add <package-name>
   
   # Run Python modules
   uv run <module>
   
   # Run tests
   uv run pytest
   
   # Run CLI tools
   uv run kb --help
   ```

## Code Example

```bash
# Complete environment setup example
#!/bin/bash

# 1. Install UV (if not already installed)
if ! command -v uv &> /dev/null; then
    echo "Installing UV..."
    pip install uv
fi

# 2. Navigate to project directory
cd /path/to/EfficientAI-MLX-Toolkit

# 3. Create and activate virtual environment
uv venv --python 3.11

# 4. Install dependencies
uv sync

# 5. Install project in development mode
uv pip install -e .

# 6. Verify installation
uv run python --version
uv run kb --help

echo "✅ Development environment setup complete!"
```

```python
# Example pyproject.toml configuration
[project]
name = "efficientai-mlx-toolkit"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "mlx>=0.0.1",
    "typer>=0.16.0",
    "rich>=14.1.0",
    "pyyaml>=6.0.0",
]

[project.scripts]
kb = "knowledge_base.cli:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

## Gotchas/Pitfalls

- **Never use pip directly**: Always use `uv add` instead of `pip install`
- **Python version compatibility**: Ensure Python 3.11+ for Apple Silicon optimizations
- **Virtual environment confusion**: UV manages virtual environments automatically
- **Dependency conflicts**: Use `uv sync` to resolve conflicts, not manual pip commands
- **CI/CD setup**: Ensure CI systems use UV, not pip
- **Path issues**: Make sure UV's virtual environment is activated in your shell

## Performance Impact

UV provides significant performance improvements over traditional pip:

- **Dependency resolution**: 10-100x faster than pip
- **Installation speed**: 2-10x faster package installation
- **Virtual environment creation**: Near-instantaneous
- **Lock file generation**: Faster and more reliable than pip-tools
- **Apple Silicon compatibility**: Better native support for M1/M2 chips

## Related Knowledge

- [Python Code Quality Standards](python-code-quality-standards.md) - Code style and formatting guidelines
- [Modern Python Type Annotations Migration Guide](modern-python-type-annotations-migration-guide.md) - Type hints best practices
- [Git Version Control Standards](git-version-control-standards.md) - Version control workflow
