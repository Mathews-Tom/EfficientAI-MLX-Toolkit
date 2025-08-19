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

```
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

### Basic Usage

```bash
# System information and hardware detection
uv run efficientai-toolkit info

# Environment setup for Apple Silicon
uv run efficientai-toolkit setup

# Run benchmarks
uv run efficientai-toolkit benchmark
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
| **EfficientAI CLI** | 🚧 **Basic** | Core CLI exists, advanced features planned |
| **LoRA Fine-tuning MLX** | 📋 **Planned** | Next priority implementation |
| **Model Compression** | 📋 **Planned** | Quantization and pruning pipelines |
| **Deployment Tools** | 📋 **Planned** | FastAPI, Gradio, containerization |

## 🧪 Development

### Testing

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov

# Run specific test categories
uv run pytest -m "not slow"           # Exclude slow tests
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

### 🔄 In Development

- **LoRA Fine-tuning MLX**: Apple Silicon optimized LoRA implementation
- **Quantized Model Benchmarks**: 4-bit/8-bit quantization with MLX
- **Model Compression Pipeline**: Pruning and distillation for CPU deployment

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
