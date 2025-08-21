# EfficientAI-MLX-Toolkit

🚀 **Apple Silicon optimized AI toolkit for efficient machine learning workflows**

A comprehensive toolkit designed specifically for Apple Silicon (M1/M2/M3) that provides optimized implementations of various AI/ML techniques including LoRA fine-tuning, quantization, model compression, and deployment utilities.

## ✨ Features

- **🍎 Apple Silicon Optimized**: Native MLX framework integration with MPS and ANE support
- **🛠️ Shared Utilities**: Production-ready logging, configuration, benchmarking, and file operations
- **📊 Comprehensive Benchmarking**: Hardware-aware performance measurement and comparison
- **⚙️ Advanced Configuration**: Profile-based config with environment overrides and validation
- **🔧 Development Tools**: CLI toolkit for setup, benchmarking, and system information

## 🏗️ Architecture

```bash
EfficientAI-MLX-Toolkit/
├── utils/                     # ✅ Complete shared utilities
│   ├── logging_utils.py       # Apple Silicon tracking & log management
│   ├── config_manager.py      # Multi-format config with profiles
│   ├── file_operations.py     # Safe file ops with backup support
│   ├── benchmark_runner.py    # Hardware-aware benchmarking
│   └── plotting_utils.py      # Visualization and reporting
├── efficientai_mlx_toolkit/   # 🚧 Basic CLI (needs expansion)
├── dspy_toolkit/              # ✅ Complete DSPy integration framework
├── knowledge_base/            # ✅ Complete development knowledge system
├── environment/               # 🚧 Environment setup utilities
└── projects/                  # 🚧 Individual ML project implementations
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Mathews-Tom/EfficientAI-MLX-Toolkit.git
cd EfficientAI-MLX-Toolkit

# Install with uv (recommended)
uv sync

# Or install with pip
pip install -e .
```

### Unified CLI System with Namespace Architecture

The toolkit provides a **unified command-line interface** using a namespace:command syntax for seamless project integration:

```bash
# System information and hardware detection
uv run efficientai-toolkit info

# Environment setup for Apple Silicon
uv run efficientai-toolkit setup

# List all available projects
uv run efficientai-toolkit projects

# Unified testing across all projects
uv run efficientai-toolkit test --all
uv run efficientai-toolkit test <namespace>
```

### LoRA Fine-tuning Example (Namespace Syntax)

```bash
# Get LoRA project information
uv run efficientai-toolkit lora-finetuning-mlx:info

# Validate configuration
uv run efficientai-toolkit lora-finetuning-mlx:validate

# Train a LoRA model with namespace syntax
uv run efficientai-toolkit lora-finetuning-mlx:train \
  --epochs 3 --batch-size 2 --learning-rate 2e-4

# Run hyperparameter optimization
uv run efficientai-toolkit lora-finetuning-mlx:optimize \
  --model microsoft/DialoGPT-medium \
  --data projects/01_LoRA_Finetuning_MLX/data/samples/sample_conversations.jsonl \
  --trials 10

# Generate text with trained model (MLX-compatible models)
uv run efficientai-toolkit lora-finetuning-mlx:generate \
  --model-path mlx-community/Llama-3.2-1B-Instruct-4bit \
  --prompt "The future of AI is" \
  --max-length 100

# Generate with LoRA adapters (basic inference)
uv run efficientai-toolkit lora-finetuning-mlx:generate \
  --model-path mlx-community/Llama-3.2-1B-Instruct-4bit \
  --adapter-path outputs/checkpoints/checkpoint_epoch_2 \
  --prompt "Hello, how are you?" \
  --max-length 50

# Start inference server
uv run efficientai-toolkit lora-finetuning-mlx:serve \
  --model-path /path/to/model \
  --host 0.0.0.0 --port 8000
```

### Model Compression Example (Namespace Syntax)

```bash
# Get Model Compression project information
uv run efficientai-toolkit model-compression-mlx:info

# Validate configuration
uv run efficientai-toolkit model-compression-mlx:validate

# Apply quantization to reduce model size
uv run efficientai-toolkit model-compression-mlx:quantize \
  --model-path mlx-community/Llama-3.2-1B-Instruct-4bit \
  --bits 8 --method post_training

# Apply pruning to remove less important weights
uv run efficientai-toolkit model-compression-mlx:prune \
  --model-path mlx-community/Llama-3.2-1B-Instruct-4bit \
  --sparsity 0.5 --method magnitude

# Comprehensive compression with multiple techniques
uv run efficientai-toolkit model-compression-mlx:compress \
  --model-path mlx-community/Llama-3.2-1B-Instruct-4bit \
  --methods quantization,pruning

# Benchmark compression methods and measure performance
uv run efficientai-toolkit model-compression-mlx:benchmark \
  --model-path mlx-community/Llama-3.2-1B-Instruct-4bit \
  --output benchmark_results/
```

**Alternative: Direct Project Execution (Development)**

For standalone development, projects can also be executed directly:

```bash
cd projects/01_LoRA_Finetuning_MLX
uv run python src/cli.py train --epochs 3 --batch-size 2

cd projects/02_Model_Compression_MLX
uv run python src/cli.py quantize --model-path mlx-community/Llama-3.2-1B-Instruct-4bit --bits 8
```

### Using Shared Utilities

```python
from utils import setup_logging, ConfigManager, BenchmarkRunner
from pathlib import Path

# Setup Apple Silicon optimized logging
setup_logging(
    log_level="INFO",
    log_file=Path("logs/app.log"),
    enable_apple_silicon_tracking=True
)

# Configuration with profiles
config = ConfigManager(Path("config.yaml"), profile="development")
debug_mode = config.get_with_type("debug", bool, default=False)

# Hardware-aware benchmarking
runner = BenchmarkRunner()
if runner.hardware_info.mlx_available:
    result = runner.run_benchmark("my_optimization", benchmark_func)
```

## 📋 Project Status

| Component | Status | Description |
|-----------|--------|-------------|
| **Shared Utilities** | ✅ **Complete** | Production-ready foundational utilities |
| **DSPy Toolkit Framework** | ✅ **Complete** | Structured AI workflows with MLX backend |
| **Knowledge Base System** | ✅ **Complete** | Development knowledge management |
| **EfficientAI Unified CLI** | ✅ **Complete** | Dynamic project discovery and unified commands |
| **LoRA Fine-tuning MLX** | ✅ **Complete** | MLX-native LoRA with optimization & serving |
| **Model Compression MLX** | ✅ **Complete** | Quantization, pruning, distillation & benchmarking |
| **Deployment Tools** | 🚧 **Partial** | FastAPI serving implemented in LoRA project |

### Recent Achievements

- **🎯 Unified CLI System**: All projects accessible through single entry point
- **🧪 100% Test Coverage**: All LoRA framework tests passing (56/56) + Model Compression tests (14/14)
- **🔧 MLX Optimization**: Full Apple Silicon integration with unified memory
- **⚡ Dynamic Discovery**: Automatic project detection and registration
- **📊 Comprehensive Testing**: Unified test runner with per-project execution
- **🚀 LoRA Inference**: Working text generation with MLX-native models and LoRA adapters
- **📦 Production-Ready Compression**: Real MLX quantization, pruning, distillation & benchmarking

## 🧪 Development

### Testing

```bash
# Run tests for all projects
uv run efficientai-toolkit test --all

# Run tests for specific project
uv run efficientai-toolkit test lora-finetuning-mlx
uv run efficientai-toolkit test model-compression-mlx

# Run with coverage and verbose output
uv run efficientai-toolkit test lora-finetuning-mlx --coverage --verbose

# Run with specific pytest markers
uv run efficientai-toolkit test lora-finetuning-mlx --markers "not slow"

# Traditional pytest (still works)
uv run pytest
uv run pytest --cov
uv run pytest -m apple_silicon        # Apple Silicon specific tests
```

### Code Quality

```bash
# Format code
uv run black .

# Lint code
uv run ruff check .

# Type checking
uv run mypy .

# All quality checks
uv run black . && uv run isort . && uv run ruff check . && uv run mypy .
```

## 🎯 Planned Features

### 🔄 Recently Completed

- **LoRA Fine-tuning MLX**: ✅ Apple Silicon optimized LoRA implementation with optimization & serving
- **Model Compression MLX**: ✅ Production-ready quantization, pruning, distillation & benchmarking
- **Unified CLI System**: ✅ Dynamic project discovery with namespace architecture

### 📅 Roadmap

- **Multimodal CLIP Fine-tuning**: Vision-language model optimization
- **Core ML Diffusion**: Stable Diffusion for Apple Neural Engine
- **Federated Learning System**: Distributed training across Apple devices
- **MLOps Integration**: Complete deployment and monitoring solutions

## 📚 Documentation

- **[CLAUDE.md](CLAUDE.md)**: Developer guidance and architecture overview
- **[Development Knowledge Base](knowledge_base/)**: Comprehensive documentation system
- **[Project Specifications](.kiro/specs/)**: Detailed implementation plans
- **[API Documentation](docs/)**: Generated API documentation

## 🤝 Contributing

1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature/your-feature`
3. **Follow development guidelines** in [CLAUDE.md](CLAUDE.md)
4. **Add comprehensive tests** for new functionality
5. **Submit pull request** with detailed description

### Development Guidelines

- **Use `uv` for package management**: All dependencies and commands
- **Apple Silicon first**: Optimize for M1/M2/M3 hardware
- **Pathlib everywhere**: Modern file handling patterns
- **Comprehensive testing**: Maintain high test coverage
- **Type safety**: Full type annotations required

## 🔧 System Requirements

### Recommended (Apple Silicon)

- **macOS 12.0+** with Apple Silicon (M1/M2/M3)
- **Python 3.12+**
- **MLX framework** for optimal performance
- **16GB+ RAM** for model training/inference

### Supported

- **macOS/Linux/Windows** with fallback implementations
- **Intel/AMD processors** with CPU optimizations
- **CUDA GPUs** with PyTorch backend

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Apple MLX Team**: For the excellent MLX framework
- **DSPy Framework**: For structured AI programming patterns
- **Open Source Community**: For the tools and libraries that make this possible

---

**Built with ❤️ for Apple Silicon • Optimized for the future of AI**
