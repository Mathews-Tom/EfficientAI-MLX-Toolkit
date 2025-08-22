# EfficientAI-MLX-Toolkit - Project Architecture

This document provides a comprehensive overview of the toolkit's architecture, project structure, and implementation details.

## High-Level Architecture

```
EfficientAI-MLX-Toolkit/
├── 🎯 Core Framework
│   ├── efficientai_mlx_toolkit/     # Unified CLI system
│   ├── utils/                       # Shared utilities
│   ├── dspy_toolkit/               # DSPy integration
│   ├── knowledge_base/             # Development knowledge
│   └── environment/                # Environment management
└── 🚀 AI/ML Projects
    ├── 01_LoRA_Finetuning_MLX/     # LoRA fine-tuning with MLX
    ├── 02_Model_Compression_MLX/   # Model quantization & compression
    └── 03_CoreML_Stable_Diffusion_Style_Transfer/ # Style transfer
```

## Project Status Matrix

| Component | Status | Tests | Coverage | Key Features |
|-----------|--------|-------|----------|--------------|
| **Core Utilities** | ✅ Complete | ✅ | 90%+ | Logging, config, benchmarking, file ops |
| **Unified CLI** | ✅ Complete | ✅ | 85%+ | Dynamic discovery, namespace routing |
| **LoRA MLX** | ✅ Complete | ✅ 56/56 | 85%+ | Training, optimization, serving, generation |
| **Model Compression** | ✅ Complete | ✅ 14/14 | 90%+ | Quantization, pruning, distillation |
| **CoreML Style Transfer** | ✅ Complete | ✅ 48/51 | 38%+ | Diffusion, Core ML conversion, benchmarking |
| **DSPy Framework** | ✅ Complete | ✅ | 80%+ | Structured AI workflows, MLX backend |
| **Knowledge Base** | ✅ Complete | ✅ | 75%+ | Development docs, search, indexing |

## Project Deep Dive

### 1. LoRA Fine-tuning MLX (`projects/01_LoRA_Finetuning_MLX/`)

**Purpose**: Apple Silicon optimized LoRA fine-tuning with production deployment

```
src/
├── lora/                   # LoRA implementation
│   ├── config.py          # LoRA configuration
│   ├── layers.py          # MLX-native LoRA layers
│   ├── model_adapter.py   # Model adaptation utilities
│   └── utils.py           # LoRA utilities
├── training/               # Training framework
│   ├── config.py          # Training configuration
│   ├── trainer.py         # MLX-optimized trainer
│   ├── optimizer.py       # Custom optimizers
│   └── callbacks.py       # Training callbacks
├── inference/              # Production inference
│   ├── config.py          # Inference configuration
│   ├── engine.py          # High-performance engine
│   └── serving.py         # FastAPI server
└── cli.py                 # Command-line interface
```

**Key Features**:
- MLX-native LoRA layers for Apple Silicon
- Hyperparameter optimization with Optuna
- Production-ready FastAPI serving
- Real-time text generation with streaming
- Comprehensive benchmarking and monitoring

**CLI Commands**:
```bash
uv run efficientai-toolkit lora-finetuning-mlx:train --epochs 3
uv run efficientai-toolkit lora-finetuning-mlx:optimize --trials 10
uv run efficientai-toolkit lora-finetuning-mlx:generate --prompt "Hello world"
uv run efficientai-toolkit lora-finetuning-mlx:serve --host 0.0.0.0
```

### 2. Model Compression MLX (`projects/02_Model_Compression_MLX/`)

**Purpose**: Production-ready model compression with quantization, pruning, and distillation

```
src/
├── compression/            # Core compression algorithms
│   ├── config.py          # Compression configuration
│   ├── quantization.py    # MLX-optimized quantization
│   ├── pruning.py         # Weight pruning algorithms
│   └── distillation.py    # Knowledge distillation
├── benchmarking/           # Performance measurement
│   ├── config.py          # Benchmark configuration
│   ├── runner.py          # Benchmark execution
│   └── metrics.py         # Performance metrics
├── utils/                  # Compression utilities
│   ├── model_utils.py     # Model manipulation
│   └── memory_profiling.py # Memory analysis
└── cli.py                 # Command-line interface
```

**Key Features**:
- Real MLX quantization with multiple bit precisions
- Advanced pruning algorithms (magnitude, gradual)
- Knowledge distillation for model efficiency
- Comprehensive benchmarking with memory profiling
- Apple Silicon optimized implementations

**CLI Commands**:
```bash
uv run efficientai-toolkit model-compression-mlx:quantize --bits 8
uv run efficientai-toolkit model-compression-mlx:prune --sparsity 0.5
uv run efficientai-toolkit model-compression-mlx:compress --methods quantization,pruning
uv run efficientai-toolkit model-compression-mlx:benchmark --output results/
```

### 3. CoreML Stable Diffusion Style Transfer (`projects/03_CoreML_Stable_Diffusion_Style_Transfer/`)

**Purpose**: Artistic style transfer using Stable Diffusion with Core ML optimization

```
src/
├── diffusion/              # Stable Diffusion implementation
│   ├── config.py          # Diffusion configuration
│   ├── model.py           # MLX-optimized Stable Diffusion
│   └── pipeline.py        # Diffusion inference pipeline
├── style_transfer/         # Style transfer framework
│   ├── config.py          # Style transfer configuration
│   ├── pipeline.py        # Main processing pipeline
│   └── engine.py          # Style processing engine
├── coreml/                 # Core ML optimization
│   ├── config.py          # Core ML configuration
│   ├── converter.py       # PyTorch to Core ML conversion
│   └── optimizer.py       # Apple Silicon optimization
├── training/               # Training framework
│   ├── config.py          # Training configuration
│   ├── trainer.py         # LoRA style training
│   └── callbacks.py       # Training monitoring
├── inference/              # Production inference
│   ├── config.py          # Inference configuration
│   └── engine.py          # High-performance inference
└── cli.py                 # Command-line interface
```

**Key Features**:
- Dual-method style transfer (diffusion and neural)
- MLX-accelerated Stable Diffusion models
- PyTorch to Core ML conversion pipeline
- Apple Neural Engine optimization
- Batch processing and real-time inference
- Comprehensive benchmarking tools

**CLI Commands**:
```bash
uv run efficientai-toolkit coreml-stable-diffusion-style-transfer:transfer \
  --content-image content.jpg --style-image style.jpg
uv run efficientai-toolkit coreml-stable-diffusion-style-transfer:convert \
  --model-path model.pth --output-path model.mlpackage
uv run efficientai-toolkit coreml-stable-diffusion-style-transfer:benchmark \
  --model-path model.mlpackage --test-images images/
```

## Unified CLI Architecture

### Namespace Discovery System

The toolkit uses a dynamic discovery system that automatically detects and registers project CLI modules:

```python
# efficientai_mlx_toolkit/cli.py
class ProjectDiscovery:
    def discover_projects(self) -> Dict[str, ProjectInfo]:
        """Dynamically discover and register project namespaces."""
        projects = {}
        
        for project_dir in self.projects_root.iterdir():
            if self._is_valid_project(project_dir):
                namespace = self._extract_namespace(project_dir.name)
                cli_module = self._load_cli_module(project_dir)
                projects[namespace] = ProjectInfo(
                    name=project_dir.name,
                    namespace=namespace,
                    cli_module=cli_module,
                    path=project_dir
                )
        
        return projects
```

### Command Routing

Commands are routed using the namespace:command syntax:

```bash
# Format: namespace:command [args...]
uv run efficientai-toolkit lora-finetuning-mlx:train --epochs 5
uv run efficientai-toolkit model-compression-mlx:quantize --bits 8
uv run efficientai-toolkit coreml-stable-diffusion-style-transfer:transfer --content-image img.jpg
```

### Error Handling and Help System

- **Invalid Namespace**: Shows available namespaces with suggestions
- **Invalid Command**: Shows available commands for the namespace
- **Help System**: Integrated help for all commands across projects

## Shared Utilities Framework

### Core Components

1. **Logging System (`utils/logging_utils.py`)**
   - Apple Silicon hardware tracking
   - Structured JSON logging
   - Performance monitoring integration

2. **Configuration Management (`utils/config_manager.py`)**
   - Multi-format support (YAML, TOML, JSON)
   - Environment variable overrides
   - Profile-based configurations
   - Type-safe configuration classes

3. **Benchmarking Framework (`utils/benchmark_runner.py`)**
   - Hardware-aware performance measurement
   - Memory profiling integration
   - Statistical analysis and reporting
   - Visualization and plotting

4. **File Operations (`utils/file_operations.py`)**
   - Safe file operations with atomic writes
   - Backup and recovery mechanisms
   - Cross-platform path handling
   - Compression and archiving

### Hardware Abstraction

All projects use a unified hardware detection system:

```python
from utils.hardware_utils import HardwareInfo

hw = HardwareInfo()
if hw.is_apple_silicon:
    # Use MLX optimizations
    device = "mps" if hw.mps_available else "cpu"
elif hw.cuda_available:
    device = "cuda"
else:
    device = "cpu"
```

## Testing Strategy

### Test Organization

Each project follows a consistent testing structure:

```
tests/
├── test_cli.py            # CLI command testing
├── test_config.py         # Configuration validation
├── test_[module].py       # Module-specific tests
├── conftest.py           # Shared fixtures
└── fixtures/             # Test data and mocks
```

### Test Categories

- **Unit Tests**: Individual component testing
- **Integration Tests**: Cross-component functionality
- **Apple Silicon Tests**: Hardware-specific tests
- **Benchmark Tests**: Performance validation
- **CLI Tests**: Command-line interface testing

### Coverage Requirements

- **Core Utilities**: 90%+ coverage required
- **Project Code**: 30%+ coverage minimum
- **Critical Paths**: 100% coverage required

## Development Workflow

### Code Standards

1. **Type Safety**: Full type annotations required
2. **Modern Python**: Use Python 3.10+ features (union types, match statements)
3. **Apple Silicon First**: Optimize for M1/M2/M3 hardware
4. **Testing**: Comprehensive test coverage
5. **Documentation**: Clear docstrings and README files

### Quality Assurance

```bash
# Code formatting and linting
uv run black . && uv run isort . && uv run ruff check . && uv run mypy .

# Testing
uv run efficientai-toolkit test --all

# Benchmarking
uv run efficientai-toolkit benchmark <project>
```

### Release Process

1. **Feature Development**: Feature branches with comprehensive tests
2. **Integration Testing**: Cross-project compatibility validation
3. **Performance Testing**: Benchmark regression testing
4. **Documentation**: Update all relevant documentation
5. **Release**: Tagged releases with changelog

This architecture enables scalable, maintainable, and high-performance AI/ML workflows optimized for Apple Silicon while maintaining cross-platform compatibility.