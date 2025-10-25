# P0 Projects MLOps Integration

Comprehensive MLOps integration for P0 toolkit projects, providing seamless experiment tracking, data versioning, model deployment, and performance monitoring.

## Projects Integrated

1. **[LoRA Fine-tuning MLX](lora_finetuning/)** - Experiment tracking, adapter deployment, inference monitoring
2. **[Model Compression MLX](model_compression/)** - Quantization/pruning tracking, benchmark comparison, compressed model deployment
3. **[CoreML Stable Diffusion](coreml_diffusion/)** - Style transfer tracking, CoreML conversion, performance benchmarking

## Features

### Unified MLOps Stack

All P0 projects connect to the same centralized MLOps infrastructure:

- **MLFlow Server** - Centralized experiment tracking and artifact storage
- **DVC Remote** - Shared data and model versioning storage
- **BentoML Registry** - Unified model deployment registry
- **Evidently Dashboard** - Centralized performance monitoring
- **Shared Workspaces** - Project-isolated but interconnected workspaces

### Automatic Integration

Each project provides a tracking class that handles:

- ✅ Automatic configuration based on project namespace
- ✅ Context-aware parameter and metric logging
- ✅ Apple Silicon optimization metrics
- ✅ Seamless data and model versioning
- ✅ One-command model deployment
- ✅ Real-time performance monitoring

### Zero-Configuration Setup

```python
# Just import and use - everything is pre-configured
from mlops.integrations.p0_projects.lora_finetuning import LoRAMLOpsTracker

tracker = LoRAMLOpsTracker()  # Auto-configured for LoRA project

with tracker.start_training_run(run_name="exp-001") as run:
    tracker.log_training_config(lora_config, training_config)
    # ... training code ...
    tracker.log_training_metrics(metrics, epoch=epoch)
```

## Quick Start

### 1. Choose Your Project

- [LoRA Fine-tuning](lora_finetuning/) - Language model fine-tuning with LoRA
- [Model Compression](model_compression/) - Model quantization and pruning
- [CoreML Diffusion](coreml_diffusion/) - Style transfer with CoreML

### 2. Follow Integration Guide

Each project has a detailed README with:
- Quick start examples
- Complete API reference
- Integration patterns
- Best practices
- Troubleshooting guide

### 3. Validate Integration

```bash
# Run integration tests
uv run pytest mlops/integrations/p0_projects/tests/ -v

# Check status for your project
uv run python -c "
from mlops.integrations.p0_projects.lora_finetuning import LoRAMLOpsTracker
tracker = LoRAMLOpsTracker()
import json
print(json.dumps(tracker.get_status(), indent=2))
"
```

## Integration Examples

### LoRA Fine-tuning

```python
from mlops.integrations.p0_projects.lora_finetuning import LoRAMLOpsTracker

tracker = LoRAMLOpsTracker()

with tracker.start_training_run(run_name="lora-exp-001") as run:
    # Log configurations
    tracker.log_training_config(lora_config, training_config)

    # Version dataset
    tracker.version_dataset("data/samples/train.jsonl", push_to_remote=True)

    # Training loop
    for epoch in range(num_epochs):
        metrics = train_epoch(...)
        tracker.log_training_metrics(metrics, epoch=epoch)

    # Save model
    tracker.save_model_artifact("outputs/checkpoints/final")

# Deploy adapter
tracker.deploy_adapter(
    adapter_path="outputs/checkpoints/final",
    model_name="lora_adapter_v1",
)
```

### Model Compression

```python
from mlops.integrations.p0_projects.model_compression import CompressionMLOpsTracker

tracker = CompressionMLOpsTracker()

with tracker.start_compression_run("quantize-8bit", "quantization") as run:
    # Log config
    tracker.log_quantization_config(quant_config)

    # Quantize model
    quantized_model = quantizer.quantize(model)

    # Log metrics
    tracker.log_compression_metrics({
        "compression_ratio": 3.8,
        "size_reduction_mb": 1250.5,
    })

    # Benchmark
    original_metrics = benchmark_model(original_model)
    compressed_metrics = benchmark_model(quantized_model)
    tracker.log_benchmark_metrics(original_metrics, compressed_metrics)

    # Save and deploy
    tracker.save_compressed_model("outputs/quantized/model_8bit")

tracker.deploy_compressed_model(
    model_path="outputs/quantized/model_8bit",
    model_name="llama_8bit_v1",
)
```

### CoreML Diffusion

```python
from mlops.integrations.p0_projects.coreml_diffusion import DiffusionMLOpsTracker

tracker = DiffusionMLOpsTracker()

# Style transfer
with tracker.start_transfer_run("artistic-001") as run:
    # Log config
    tracker.log_transfer_config(style_config)

    # Transfer style
    result_image = pipeline.transfer_style(content, style)

    # Log metrics
    tracker.log_transfer_metrics({
        "style_similarity": 0.87,
        "transfer_time_s": 3.5,
    })

    # Save output
    tracker.save_output_artifact("outputs/styled_image.png")

# CoreML conversion
with tracker.start_conversion_run("convert-001") as run:
    tracker.log_coreml_config(coreml_config)
    coreml_model = converter.convert_model("pytorch_model/")
    tracker.log_conversion_metrics({
        "conversion_time_s": 45.2,
        "ane_compatible": True,
    })
    tracker.save_coreml_model("outputs/coreml/model.mlpackage")

# Deploy
tracker.deploy_coreml_model(
    model_path="outputs/coreml/model.mlpackage",
    model_name="style_transfer_v1",
)
```

## Architecture

### Integration Structure

```
mlops/integrations/p0_projects/
├── lora_finetuning/
│   ├── __init__.py
│   ├── integration.py          # LoRAMLOpsTracker
│   └── README.md               # Complete documentation
├── model_compression/
│   ├── __init__.py
│   ├── integration.py          # CompressionMLOpsTracker
│   └── README.md
├── coreml_diffusion/
│   ├── __init__.py
│   ├── integration.py          # DiffusionMLOpsTracker
│   └── README.md
├── tests/
│   ├── test_lora_integration.py
│   ├── test_compression_integration.py
│   └── test_diffusion_integration.py
├── MIGRATION_GUIDE.md          # Step-by-step integration guide
└── README.md                   # This file
```

### Shared Infrastructure

All projects use the same underlying MLOps infrastructure:

```
mlops/
├── client/                     # Unified MLOps client
│   ├── mlops_client.py        # Main client (MLFlow + DVC + BentoML + Evidently)
│   ├── mlflow_client.py       # MLFlow integration
│   └── dvc_client.py          # DVC integration
├── tracking/                   # MLFlow tracking server
├── versioning/                 # DVC storage and config
├── serving/                    # BentoML deployment
├── monitoring/                 # Evidently monitoring
└── workspace/                  # Project workspaces
    ├── lora-finetuning-mlx/
    ├── model-compression-mlx/
    └── coreml-stable-diffusion-style-transfer/
```

### Workflow

```
┌─────────────────┐
│ P0 Project Code │
└────────┬────────┘
         │
         ▼
┌─────────────────────┐
│ Integration Tracker │  (e.g., LoRAMLOpsTracker)
└────────┬────────────┘
         │
         ▼
┌─────────────────┐
│ MLOps Client    │  (Unified client for all operations)
└────────┬────────┘
         │
    ┌────┴────┬──────────┬───────────┐
    ▼         ▼          ▼           ▼
┌────────┐ ┌───┐  ┌─────────┐  ┌──────────┐
│ MLFlow │ │DVC│  │ BentoML │  │ Evidently│
└────────┘ └───┘  └─────────┘  └──────────┘
```

## Testing

### Run All Integration Tests

```bash
# Run all P0 integration tests
uv run pytest mlops/integrations/p0_projects/tests/ -v

# Run tests for specific project
uv run pytest mlops/integrations/p0_projects/tests/test_lora_integration.py -v
uv run pytest mlops/integrations/p0_projects/tests/test_compression_integration.py -v
uv run pytest mlops/integrations/p0_projects/tests/test_diffusion_integration.py -v

# Run integration tests only
uv run pytest mlops/integrations/p0_projects/tests/ -v -m integration
```

### Test Coverage

- **45 tests** covering all three projects
- **Unit tests** for each tracker method
- **Integration tests** for complete workflows
- **Mock-based tests** for fast execution
- **100% pass rate** in validation

## Documentation

### Project-Specific Documentation

Each project has a comprehensive README:

- **Quick Start** - Get started in minutes
- **Features** - Detailed feature descriptions
- **Integration Examples** - Complete code examples
- **API Reference** - Full API documentation
- **Best Practices** - Recommended patterns
- **Troubleshooting** - Common issues and solutions

### General Documentation

- **[MIGRATION_GUIDE.md](MIGRATION_GUIDE.md)** - Step-by-step integration guide
- **[MLOps README](../../README.md)** - Main MLOps documentation
- **[Project READMEs](../../)** - Individual component documentation

## Best Practices

### 1. Consistent Naming

```python
# Use structured naming conventions
run_name = f"{model}_{method}_{timestamp}"
model_name = f"{model_type}_{version}"
```

### 2. Comprehensive Tagging

```python
# Add meaningful tags for filtering
tags = {
    "model": "llama-3.2-1b",
    "method": "lora",
    "dataset": "custom",
    "environment": "apple_silicon",
}
```

### 3. Data Versioning First

```python
# Always version data before training
tracker.version_dataset("data/train.jsonl", push_to_remote=True)

# Then train
with tracker.start_training_run(...):
    # ... training code ...
    pass
```

### 4. Monitoring Setup

```python
# Set up reference data during training
tracker.client.set_reference_data(train_df)

# Monitor during inference
results = tracker.monitor_inference(...)
if results.get("drift_detected"):
    alert("Data drift detected!")
```

### 5. Error Handling

```python
# Graceful degradation if MLOps unavailable
try:
    with tracker.start_training_run(...) as run:
        # ... training code ...
        pass
except MLOpsClientError as e:
    logger.warning(f"MLOps unavailable: {e}")
    # Continue without tracking
```

## Performance Impact

The integration adds minimal overhead:

| Operation | Overhead | Impact |
|-----------|----------|--------|
| Experiment tracking | 10-50ms per log | <0.1% |
| Data versioning | One-time | <1% |
| Model deployment | One-time | N/A |
| Monitoring | 50-100ms per batch | <1% |

**Total impact**: <1% on training time, <5% on inference time.

## Requirements

### Core Dependencies

Installed automatically with toolkit:
- `mlflow` - Experiment tracking
- `dvc` - Data versioning
- `bentoml` - Model deployment
- `evidently` - Performance monitoring
- `pandas` - Data handling

### Optional Dependencies

- `mlx` - Apple Silicon optimization (auto-detected)
- `coremltools` - CoreML conversion (CoreML project only)

## Troubleshooting

### Common Issues

#### MLFlow Connection Failed
```bash
# Check MLFlow server
curl http://localhost:5000/health

# View logs
docker logs mlflow-server
```

#### DVC Remote Not Configured
```bash
# Configure DVC remote
dvc remote add -d storage /path/to/remote
```

#### BentoML Not Available
```bash
# Install BentoML
uv add bentoml
```

### Getting Help

1. Check project-specific README
2. Review [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md)
3. Run integration tests to identify issues
4. Check MLOps status with `tracker.get_status()`

## Contributing

### Adding New Project Integration

1. Create project directory: `mlops/integrations/p0_projects/new_project/`
2. Implement tracker class: `integration.py`
3. Write comprehensive README
4. Add integration tests
5. Update this README

### Testing Guidelines

- Write unit tests for all tracker methods
- Add integration tests for complete workflows
- Use mocks for external dependencies
- Aim for >90% code coverage

## License

Same as parent project (see root LICENSE file).

## Related Documentation

- [MLOps Architecture](../../README.md)
- [MLFlow Client](../../client/mlflow_client.py)
- [DVC Client](../../client/dvc_client.py)
- [BentoML Packaging](../../serving/bentoml/)
- [Evidently Monitoring](../../monitoring/evidently/)
