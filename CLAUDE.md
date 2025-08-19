# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Package Management
- Use `uv` as the package manager for all operations
- Install packages: `uv add <package>`
- Install development dependencies: `uv add --group dev <package>`
- Run commands: `uv run <command>`
- Install all dependencies: `uv sync`

### Testing
- Run all tests: `uv run pytest`
- Run tests with coverage: `uv run pytest --cov`
- Run specific test categories:
  - `uv run pytest -m "not slow"` (exclude slow tests)
  - `uv run pytest -m integration` (integration tests only)
  - `uv run pytest -m benchmark` (benchmark tests only)
  - `uv run pytest -m apple_silicon` (Apple Silicon specific tests)

### Code Quality
- Format code: `uv run black .`
- Sort imports: `uv run isort .`
- Lint code: `uv run ruff check .`
- Type checking: `uv run mypy .`
- Run all quality checks: `uv run black . && uv run isort . && uv run ruff check . && uv run mypy .`

### CLI Tools
- Main CLI: `uv run efficientai-toolkit`
- Environment setup: `uv run efficientai-toolkit setup`
- System info: `uv run efficientai-toolkit info`
- Run benchmarks: `uv run efficientai-toolkit benchmark <name>`
- Knowledge base CLI: `uv run python -m kb <command>`

## High-Level Architecture

This repository implements an Apple Silicon optimized AI toolkit with multiple frameworks and tools.

### Core Components

**EfficientAI-MLX-Toolkit (`efficientai_mlx_toolkit/`)**
- Main CLI entry point with setup, benchmarking, and info commands
- Apple Silicon hardware detection and optimization utilities
- Focused on MLX framework integration and Apple-specific optimizations

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

## Apple Silicon Optimizations

This codebase is specifically optimized for Apple Silicon (M1/M2/M3) hardware:

- MLX framework integration for optimal performance
- Memory-efficient operations leveraging unified memory architecture
- Hardware detection and capability-based feature selection
- MPS (Metal Performance Shaders) backend support when available
- Core ML integration for deployment scenarios

## Key Dependencies

- **MLX**: Apple's machine learning framework for Apple Silicon
- **DSPy**: Structured programming framework for language models  
- **FastAPI**: Web framework for API deployment
- **Rich/Typer**: Modern CLI interfaces
- **PyTorch**: ML framework with MPS backend support
- **Transformers**: Hugging Face model integration

## Testing Strategy

The test suite includes comprehensive coverage with special considerations for:
- Hardware-specific tests that require Apple Silicon
- Optional dependency handling (MLX, DSPy)
- Async operation testing
- Memory profiling and performance benchmarking
- Mock-heavy testing for external dependencies