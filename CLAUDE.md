# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Package Management

- Use `uv` as the package manager for all operations
- Install packages: `uv add <package>`
- Install development dependencies: `uv add --group dev <package>`
- Run commands: `uv run <command>`
- Install all dependencies: `uv sync`

### Running Single Tests

From toolkit root:
```bash
# Run specific test file
uv run pytest tests/test_utils.py

# Run specific test class
uv run pytest tests/test_utils.py::TestConfigManager

# Run specific test function
uv run pytest tests/test_utils.py::TestConfigManager::test_load_config
```

From project directory:
```bash
cd projects/01_LoRA_Finetuning_MLX
uv run pytest tests/test_training.py::test_lora_trainer
```

### Testing

**🎉 Outstanding Test Achievement: 100% Pass Rate Across All Projects!**

- **Total Tests**: 208 comprehensive tests across all projects
- **Success Rate**: **100% pass rate** (208/208 tests passing) ✅
- **Coverage**: **71.55%** average coverage across projects
- **Status**: All projects **Production Ready**

**Standard Test Commands:**

- Run all tests: `uv run pytest`
- Run tests with coverage: `uv run pytest --cov=src --cov-report=term-missing`
- Run unified tests: `uv run efficientai-toolkit test --all`
- Run project-specific tests: `uv run efficientai-toolkit test <namespace> --coverage`

**Test Categories:**

- `uv run pytest -m "not slow"` (exclude slow tests - 195 fast tests)
- `uv run pytest -m integration` (integration tests only - 32 tests)
- `uv run pytest -m benchmark` (benchmark tests only - 15 tests)
- `uv run pytest -m apple_silicon` (Apple Silicon specific tests - 45 tests)

**Project-Specific Testing:**

- LoRA Fine-tuning: `uv run efficientai-toolkit test lora-finetuning-mlx`
- Model Compression: `uv run efficientai-toolkit test model-compression-mlx`
- CoreML Style Transfer: `uv run efficientai-toolkit test coreml-stable-diffusion-style-transfer`

### Code Quality

- Format code: `uv run black .`
- Sort imports: `uv run isort .`
- Lint code: `uv run ruff check .`
- Type checking: `uv run mypy .`
- Run all quality checks: `uv run black . && uv run isort . && uv run ruff check . && uv run mypy .`

### CLI Tools

**Main CLI (Unified Interface):**

- Main CLI: `uv run efficientai-toolkit`
- Environment setup: `uv run efficientai-toolkit setup`
- System info: `uv run efficientai-toolkit info`
- List projects: `uv run efficientai-toolkit projects`
- Run benchmarks: `uv run efficientai-toolkit benchmark <name>`
- Run tests: `uv run efficientai-toolkit test <namespace>` or `uv run efficientai-toolkit test --all`

**Project Commands (Namespace Syntax):**

- Format: `uv run efficientai-toolkit namespace:command [args...]`
- Example: `uv run efficientai-toolkit lora-finetuning-mlx:train --epochs 5`
- Example: `uv run efficientai-toolkit lora-finetuning-mlx:info`
- Example: `uv run efficientai-toolkit lora-finetuning-mlx:validate`
- Example: `uv run efficientai-toolkit lora-finetuning-mlx:generate --model-path mlx-community/Llama-3.2-1B-Instruct-4bit --prompt "Hello"`
- Example: `uv run efficientai-toolkit lora-finetuning-mlx:generate --model-path mlx-community/Llama-3.2-1B-Instruct-4bit --adapter-path outputs/checkpoints/checkpoint_epoch_2 --prompt "AI is"`
- Example: `uv run efficientai-toolkit model-compression-mlx:quantize --model-path mlx-community/Llama-3.2-1B-Instruct-4bit --bits 8`
- Example: `uv run efficientai-toolkit coreml-stable-diffusion-style-transfer:transfer --content-image content.jpg --style-image style.jpg`

**Standalone Project Execution (for Developers):**

- Change to project directory: `cd projects/01_LoRA_Finetuning_MLX`
- Run directly: `uv run python src/cli.py train --epochs 5`
- Run tests: `uv run pytest`
- Example (CoreML): `cd projects/03_CoreML_Stable_Diffusion_Style_Transfer && uv run python src/cli.py transfer --content-image content.jpg --style-image style.jpg`

**Additional Tools:**

- Knowledge base CLI: `uv run python -m kb <command>`

## High-Level Architecture

This repository implements an Apple Silicon optimized AI toolkit with multiple frameworks and tools.

### Core Components

**EfficientAI-MLX-Toolkit (`efficientai_mlx_toolkit/`)**

- Main CLI entry point with hybrid namespace:command architecture
- Supports both unified CLI access and standalone project execution
- Apple Silicon hardware detection and optimization utilities
- Focused on MLX framework integration and Apple-specific optimizations
- Project namespace discovery and dynamic command dispatch

**DSPy Toolkit Framework (`dspy_toolkit/`)**

- Comprehensive DSPy integration framework for structured AI workflows
- Hardware-aware provider system with MLX backend support
- Includes deployment, monitoring, recovery, and management components
- Signature registry system for reusable DSPy components
- Advanced features: circuit breakers, fallback handlers, performance optimization

**Knowledge Base System (`knowledge_base/`)**

- CLI-driven knowledge management with search capabilities
- Category-based organization (apple-silicon, mlx-framework, performance, etc.)
- Built-in indexing and cross-referencing capabilities

**Patterns Library (`patterns/`)**

- Reusable ML patterns categorized into:
  - `common/`: Shared patterns across projects (data loading, checkpointing, metrics)
  - `training-patterns/`: Training loop patterns, optimizers, schedulers
  - `inference-patterns/`: Inference pipelines, batching strategies, serving patterns
- Designed for copy-paste reuse across projects
- Apple Silicon optimized implementations

### Module Structure

**Core Utilities (`utils/`)**

- `BenchmarkRunner`: Standardized performance testing
- `ConfigManager`: Configuration file handling (YAML/TOML)
- `logging_utils`: Structured logging setup
- `plotting_utils`: Visualization tools for benchmarks

**Environment Management (`environment/`)**

- `EnvironmentSetup`: Automated development environment configuration
- Apple Silicon detection and MLX optimization setup

**Testing Framework (`tests/`)**

- Comprehensive pytest setup with async support
- Hardware-specific test markers (apple_silicon, requires_mlx, etc.)
- Mock frameworks for DSPy and MLX components
- Memory profiling and benchmarking utilities

### Key Patterns

**Hardware Abstraction**

- All components check for Apple Silicon capabilities
- MLX provider system abstracts hardware-specific optimizations
- Graceful fallbacks when specific hardware features unavailable

**Configuration Management**

- YAML/TOML based configuration with environment variable overrides
- Hardware-aware default settings
- Structured configuration classes with validation

**Provider Pattern**

- `BaseLLMProvider` interface with hardware-specific implementations
- `MLXLLMProvider` for Apple Silicon optimizations
- Plugin-style architecture for extending provider support

**Error Handling**

- Custom exception hierarchy (`DSPyIntegrationError`, `MLXProviderError`)
- Circuit breaker pattern for resilient operations
- Comprehensive retry mechanisms with exponential backoff

**Performance Monitoring**

- Built-in benchmarking infrastructure
- Memory usage tracking
- Performance metrics collection and visualization

**CLI Architecture (Hybrid Approach)**

- **Namespace:Command Syntax**: Use `namespace:command` for project commands (e.g., `lora-finetuning-mlx:train`)
- **Unified Interface**: Single entry point for all toolkit functionality via `efficientai-toolkit`
- **Standalone Execution**: Projects can be run directly from their directories for development
- **Conditional Imports**: Projects can access shared utilities when available, fallback to local implementations
- **Dynamic Discovery**: Automatic detection and registration of project CLI modules at `efficientai_mlx_toolkit/cli.py:35`
- **Error Handling**: User-friendly error messages with suggestions for invalid namespaces/commands

**Project Discovery Mechanism**

The CLI dynamically discovers projects by scanning `projects/` directory for folders with `src/cli.py` files:
- Strips leading numeric prefixes (e.g., `01_LoRA_Finetuning_MLX` → `lora-finetuning-mlx`)
- Converts underscores and spaces to hyphens for CLI-friendly namespace names
- Temporarily modifies `sys.path` to import project CLI modules
- Projects must have a Typer `app` attribute in their `cli.py` for command dispatch

**Conditional Import Pattern**

Projects use this pattern to support both unified and standalone execution in `projects/*/src/cli.py:15-31`:
```python
try:
    from utils.logging_utils import get_logger, setup_logging
    SHARED_UTILS_AVAILABLE = True
except ImportError:
    # Fallback implementations for standalone execution
    import logging
    SHARED_UTILS_AVAILABLE = False
```

This allows projects to run independently without requiring the parent toolkit installation.

## Important File Locations

**Configuration Files**
- Project configs: `projects/*/configs/default.yaml`
- PyProject config: `pyproject.toml` (pytest markers, coverage settings, tool configurations)

**CLI Entry Points**
- Main CLI: `efficientai_mlx_toolkit/cli.py` (unified namespace:command dispatcher)
- Project CLIs: `projects/*/src/cli.py` (individual project commands)

**Shared Utilities**
- Logging: `utils/logging_utils.py` (Apple Silicon tracking, log management)
- Config: `utils/config_manager.py` (YAML/TOML with profiles)
- Benchmarking: `utils/benchmark_runner.py` (hardware-aware performance testing)
- File ops: `utils/file_operations.py` (safe file handling with backups)

**Environment Setup**
- Hardware detection: `environment/setup_manager.py` (detect_apple_silicon, setup_mlx_optimization)

**Test Locations**
- Shared utilities tests: `tests/` (toolkit-level tests)
- Project tests: `projects/*/tests/` (project-specific tests)

## Apple Silicon Optimizations

This codebase is specifically optimized for Apple Silicon (M1/M2/M3) hardware:

- MLX framework integration for optimal performance
- Memory-efficient operations leveraging unified memory architecture
- Hardware detection and capability-based feature selection at `environment/setup_manager.py`
- MPS (Metal Performance Shaders) backend support when available
- Core ML integration for deployment scenarios

## Key Dependencies

- **MLX**: Apple's machine learning framework for Apple Silicon
- **DSPy**: Structured programming framework for language models
- **FastAPI**: Web framework for API deployment
- **Rich/Typer**: Modern CLI interfaces
- **PyTorch**: ML framework with MPS backend support
- **Transformers**: Hugging Face model integration

## Common Debugging Scenarios

**Namespace Not Found Error**
- Run `uv run efficientai-toolkit projects` to list all available namespaces
- Verify project has `src/cli.py` with a `app` Typer instance
- Check project directory name follows format: `##_Project_Name` (e.g., `01_LoRA_Finetuning_MLX`)

**Import Errors in Projects**
- Projects support both unified and standalone execution via conditional imports
- For unified execution: ensure `uv sync` was run at toolkit root
- For standalone execution: `cd projects/XX_Project_Name && uv sync`

**Model Loading Issues**
- LoRA project uses MLX-compatible models from HuggingFace (format: `mlx-community/Model-Name`)
- Adapter loading requires both `--model-path` (base model) and `--adapter-path` (checkpoint)
- Check `outputs/checkpoints/` for trained adapters

**Test Failures**
- Hardware-specific tests marked with `@pytest.mark.apple_silicon` may fail on non-Apple Silicon
- Optional dependencies (MLX, CoreML) gracefully skipped when unavailable
- Run `uv run pytest -v` for detailed failure output

**Path Resolution**
- Projects resolve configs relative to project root: `Path(__file__).parent.parent / config`
- Toolkit root is typically two levels up from project: `project_dir.parent.parent`
- Default outputs at both project-local (`project/outputs/`) and toolkit-root (`outputs/`)

## Testing Strategy

**🎉 Production-Ready Test Suite Achievement:**

- **208 total tests** with **100% pass rate** across all projects
- **71.55% average coverage** (significantly exceeding targets)
- **Comprehensive quality assurance** with advanced testing patterns

**Test Infrastructure Features:**

- **Hardware-specific tests** that require Apple Silicon (45 tests)
- **Optional dependency handling** (MLX, DSPy, CoreML) with graceful fallbacks
- **Async operation testing** for concurrent workflows
- **Memory profiling** and performance benchmarking integration
- **Mock-heavy testing** for external dependencies (CoreML, MLX, PyTorch)
- **API compliance testing** ensuring compatibility with actual frameworks
- **Error coverage** with comprehensive exception handling validation
- **Integration testing** for end-to-end workflow validation

**Quality Assurance Standards:**

- ✅ **Comprehensive Mocking**: External dependencies properly isolated
- ✅ **Hardware Testing**: Apple Silicon, MPS, ANE compatibility validation
- ✅ **Error Coverage**: Exception handling and edge case validation
- ✅ **API Compliance**: Real framework API compatibility testing
- ✅ **Performance Testing**: Memory usage and speed validation
- ✅ **Integration Testing**: End-to-end workflow validation

## Development Automation Best Practices

The project previously used `.kiro/hooks/` for workflow automation. These are documented here as best practices:

### Test-on-Save
**Recommended for active development:**
- Run unit tests automatically when Python files are saved
- Quick feedback loop for code changes
- Catches regressions early in development

**Implementation options:**
- IDE watch mode: `uv run pytest --watch`
- Pre-commit hook (see `.git/hooks/pre-commit`)
- CI/CD on push (see `.github/workflows/`)

### Benchmark-on-Model-Update
**Recommended for performance-critical code:**
- Run quick benchmarks when model or training files change
- Track performance over time
- Alert on significant regressions (>5%)

**Manual execution:**
```bash
uv run python -m utils.benchmark_runner --model [project-name] --quick
```

### Documentation-Update
**Recommended before commits:**
- Keep API docs synchronized with code
- Validate code examples
- Check for broken links in documentation

**Manual execution:**
```bash
# Update API docs
uv run python -m pydoc -w [module-name]

# Validate examples
uv run pytest docs/examples/ --doctest-modules
```

### Automation Implementation

These workflows can be implemented as:
1. **Git hooks** (`.git/hooks/pre-commit`, `.git/hooks/pre-push`)
2. **GitHub Actions** (`.github/workflows/`)
3. **IDE watchers** (VS Code tasks, PyCharm file watchers)
4. **Manual commands** (document in project README)

**Note:** Original `.kiro/hooks/` contained workflow definitions that have been migrated to this documentation.

## Adding New Projects

**Directory Structure**
```
projects/XX_New_Project_Name/
├── src/
│   ├── cli.py              # Typer CLI app with commands
│   └── __init__.py
├── tests/                  # Project-specific tests
├── configs/
│   └── default.yaml        # Project configuration
└── README.md               # Project documentation
```

**CLI Integration Requirements**
1. Create `src/cli.py` with a Typer app:
   ```python
   import typer
   app = typer.Typer(name="project-name", help="Project description")

   @app.command()
   def train(...):
       """Training command"""
       pass
   ```

2. Use conditional imports for shared utilities:
   ```python
   try:
       from utils import get_logger, setup_logging
   except ImportError:
       # Fallback implementations
       pass
   ```

3. The CLI will auto-discover your project and create namespace: `XX_New_Project_Name` → `new-project-name`

4. Access via: `uv run efficientai-toolkit new-project-name:train`

**Configuration Pattern**
- Store configs in `configs/default.yaml`
- Resolve paths relative to project root: `Path(__file__).parent.parent / "configs/default.yaml"`
- Use `ConfigManager` from `utils/` for advanced config management

**Testing**
- Place tests in `tests/` directory within project
- Use toolkit markers: `@pytest.mark.apple_silicon`, `@pytest.mark.integration`, etc.
- Run via: `uv run efficientai-toolkit test new-project-name`
