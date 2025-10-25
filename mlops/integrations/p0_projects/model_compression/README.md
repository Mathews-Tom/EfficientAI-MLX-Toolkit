# Model Compression MLX MLOps Integration

Comprehensive MLOps integration for Model Compression project, providing experiment tracking, model versioning, deployment, and performance monitoring for quantization, pruning, and compression workflows.

## Quick Start

```python
from mlops.integrations.p0_projects.model_compression import CompressionMLOpsTracker

# Create tracker
tracker = CompressionMLOpsTracker()

# Track quantization run
with tracker.start_compression_run(
    run_name="quantize-8bit",
    compression_method="quantization"
) as run:
    # Log configuration
    tracker.log_quantization_config(quant_config)

    # Quantize model
    quantized_model = quantize_model(...)

    # Log metrics
    tracker.log_compression_metrics({
        "compression_ratio": 3.8,
        "size_reduction_mb": 1250.5,
    })

    # Save compressed model
    tracker.save_compressed_model("outputs/quantized/model_8bit")

# Deploy compressed model
tracker.deploy_compressed_model(
    model_path="outputs/quantized/model_8bit",
    model_name="llama_8bit",
)
```

## Features

### 1. Experiment Tracking

Track compression experiments with comprehensive metrics:

```python
with tracker.start_compression_run(
    run_name="quantize-8bit",
    compression_method="quantization"
) as run:
    # Log quantization config
    tracker.log_quantization_config(quant_config)

    # Log compression metrics
    tracker.log_compression_metrics({
        "compression_ratio": 3.8,
        "size_reduction_mb": 1250.5,
        "original_size_mb": 5000.0,
        "compressed_size_mb": 1315.8,
        "actual_bits": 7.9,
    })

    # Log benchmark comparison
    tracker.log_benchmark_metrics(
        original_metrics={
            "inference_time_s": 2.5,
            "memory_mb": 5000.0,
            "throughput_tokens_per_s": 45.2,
        },
        compressed_metrics={
            "inference_time_s": 1.1,
            "memory_mb": 1315.8,
            "throughput_tokens_per_s": 98.5,
        },
    )
```

**Tracked Parameters:**
- Quantization: target bits, method, symmetry, per-channel
- Pruning: target sparsity, method, structured/unstructured
- Distillation: temperature, alpha, teacher model
- MLX optimization settings

**Tracked Metrics:**
- Compression ratio
- Model size reduction (MB)
- Inference speedup
- Memory savings
- Accuracy degradation
- Throughput improvement
- Apple Silicon utilization

### 2. Model Versioning

Version compressed models with DVC:

```python
# Version quantized model
result = tracker.version_model(
    "outputs/quantized/model_8bit",
    push_to_remote=True
)

# Version pruned model
result = tracker.version_model(
    "outputs/pruned/model_sparse",
    push_to_remote=True
)
```

**Versioned Artifacts:**
- Quantized models
- Pruned models
- Distilled models
- Compression metadata
- Calibration datasets

### 3. Model Deployment

Deploy compressed models using BentoML:

```python
# Deploy quantized model
result = tracker.deploy_compressed_model(
    model_path="outputs/quantized/model_8bit",
    model_name="llama_8bit",
    model_version="v1.0",
    build_bento=True,
)

print(f"Deployed: {result['model_tag']}")
```

**Deployment Features:**
- Automatic BentoML packaging
- MLX-optimized inference
- Apple Silicon acceleration
- Version control
- API endpoint generation

### 4. Performance Monitoring

Monitor compressed model performance:

```python
# Monitor inference
results = tracker.monitor_inference(
    input_sizes=[128, 256, 512],
    latencies_ms=[45.2, 89.5, 178.3],
    memory_mb=1315.8,
    additional_data={
        "tokens_generated": [128, 256, 512],
        "quality_scores": [0.95, 0.94, 0.93],
    },
)
```

**Monitoring Capabilities:**
- Inference latency tracking
- Memory usage monitoring
- Throughput measurement
- Quality metrics
- Performance degradation detection

## Integration with Compression Code

### Quantization Integration

```python
from pathlib import Path
from quantization import MLXQuantizer, QuantizationConfig
from mlops.integrations.p0_projects.model_compression import CompressionMLOpsTracker

# Initialize tracker
tracker = CompressionMLOpsTracker()

# Configure quantization
quant_config = QuantizationConfig(
    target_bits=8,
    method="linear",
    use_mlx_quantization=True,
)

# Start tracking
with tracker.start_compression_run(
    run_name="quantize-llama-8bit",
    compression_method="quantization"
) as run:
    # Log configuration
    tracker.log_quantization_config(quant_config)

    # Load model
    model, tokenizer = load_model_and_tokenizer("mlx-community/Llama-3.2-1B-Instruct-4bit")

    # Quantize
    quantizer = MLXQuantizer(quant_config)
    quantized_model = quantizer.quantize(model_path=model)

    # Get quantization info
    info = quantizer.get_quantization_info()
    stats = info.get("stats", {})

    # Log metrics
    tracker.log_compression_metrics({
        "compression_ratio": stats.get("actual_compression_ratio", 0),
        "size_reduction_mb": stats.get("size_reduction_mb", 0),
        "original_size_mb": stats.get("original_size_mb", 0),
        "compressed_size_mb": stats.get("compressed_size_mb", 0),
    })

    # Benchmark
    original_metrics = benchmark_model(model)
    compressed_metrics = benchmark_model(quantized_model)
    tracker.log_benchmark_metrics(original_metrics, compressed_metrics)

    # Save model
    output_path = "outputs/quantized/llama_8bit"
    quantizer.save_quantized_model(output_path)
    tracker.save_compressed_model(output_path)

# Version and deploy
tracker.version_model(output_path, push_to_remote=True)
tracker.deploy_compressed_model(
    model_path=output_path,
    model_name="llama_8bit_v1",
)
```

### Pruning Integration

```python
from pruning import MLXPruner, PruningConfig
from mlops.integrations.p0_projects.model_compression import CompressionMLOpsTracker

# Initialize tracker
tracker = CompressionMLOpsTracker()

# Configure pruning
prune_config = PruningConfig(
    target_sparsity=0.5,
    method="magnitude",
    structured=False,
)

# Start tracking
with tracker.start_compression_run(
    run_name="prune-llama-50pct",
    compression_method="pruning"
) as run:
    # Log configuration
    tracker.log_pruning_config(prune_config)

    # Load and prune model
    model, tokenizer = load_model_and_tokenizer("mlx-community/Llama-3.2-1B-Instruct-4bit")
    pruner = MLXPruner(prune_config)
    pruned_model = pruner.prune(model)

    # Get pruning info
    info = pruner.get_pruning_info()
    stats = info.get("stats", {})

    # Log metrics
    tracker.log_compression_metrics({
        "actual_sparsity": stats.get("actual_sparsity", 0),
        "parameters_removed_percent": stats.get("parameters_removed_percent", 0),
        "model_size_reduction_mb": stats.get("size_reduction_mb", 0),
    })

    # Save pruned model
    output_path = "outputs/pruned/llama_sparse"
    pruner.save_pruned_model(output_path)
    tracker.save_compressed_model(output_path)
```

### Comprehensive Compression Integration

```python
from compression import ModelCompressor, CompressionConfig
from mlops.integrations.p0_projects.model_compression import CompressionMLOpsTracker

# Initialize tracker
tracker = CompressionMLOpsTracker()

# Configure compression pipeline
comp_config = CompressionConfig(
    enabled_methods=["quantization", "pruning"],
    model_name="mlx-community/Llama-3.2-1B-Instruct-4bit",
    sequential_compression=True,
)

# Start tracking
with tracker.start_compression_run(
    run_name="compress-llama-full",
    compression_method="comprehensive"
) as run:
    # Log configuration
    tracker.log_compression_config(comp_config)

    # Compress model
    compressor = ModelCompressor(comp_config)
    compressed_model = compressor.compress(comp_config.model_name)

    # Get results
    results = compressor.get_compression_results()

    # Log results for each method
    for method, result in results.items():
        tracker.log_compression_metrics({
            f"{method}_compression_ratio": result.get("compression_ratio", 0),
            f"{method}_size_reduction_mb": result.get("size_reduction_mb", 0),
        })

    # Save compressed model
    output_path = comp_config.output_dir / "final_compressed"
    tracker.save_compressed_model(output_path)
```

## CLI Integration

```bash
# Quantize and track with MLOps
uv run python -c "
from mlops.integrations.p0_projects.model_compression import CompressionMLOpsTracker
from quantization import MLXQuantizer, QuantizationConfig

tracker = CompressionMLOpsTracker()
quant_config = QuantizationConfig(target_bits=8)

with tracker.start_compression_run('quantize-8bit', 'quantization'):
    tracker.log_quantization_config(quant_config)
    # ... quantization code ...
"

# Check MLOps status
uv run python -c "
from mlops.integrations.p0_projects.model_compression import create_compression_mlops_client
client = create_compression_mlops_client()
import json
print(json.dumps(client.get_status(), indent=2))
"
```

## Configuration

The integration uses shared MLOps infrastructure:
- MLFlow experiment: `model-compression-mlx`
- DVC namespace: `model-compression-mlx`
- Workspace: `mlops/workspace/model-compression-mlx/`

## Best Practices

### 1. Experiment Organization
- Use descriptive run names: `quantize_{bits}bit_{date}`
- Tag by method: `{"method": "quantization", "bits": "8"}`
- Document compression settings in descriptions

### 2. Benchmarking
- Always benchmark before and after compression
- Track multiple metrics (speed, memory, quality)
- Monitor on representative data

### 3. Model Management
- Version all compressed models
- Push to remote storage for safety
- Deploy with clear version tags

### 4. Quality Assurance
- Monitor quality degradation
- Set acceptable thresholds
- Alert on significant drops

## Troubleshooting

### Compression Tracking Issues
```python
tracker = CompressionMLOpsTracker()
status = tracker.get_status()
print(f"MLFlow: {status['mlflow_available']}")
print(f"DVC: {status['dvc_available']}")
```

### Model Versioning Issues
```python
# Check DVC status
result = tracker.version_model("outputs/quantized/model")
print(f"DVC file: {result['dvc_file']}")
```

## API Reference

### CompressionMLOpsTracker

Main integration class for compression MLOps operations.

**Methods:**
- `start_compression_run()`: Start compression experiment
- `log_quantization_config()`: Log quantization settings
- `log_pruning_config()`: Log pruning settings
- `log_compression_config()`: Log comprehensive compression settings
- `log_compression_metrics()`: Log compression metrics
- `log_benchmark_metrics()`: Log benchmark comparisons
- `save_compressed_model()`: Save model as artifact
- `version_model()`: Version model with DVC
- `deploy_compressed_model()`: Deploy with BentoML
- `monitor_inference()`: Monitor performance
- `get_status()`: Get integration status

See integration.py for detailed API documentation.

## Examples

See `mlops/integrations/p0_projects/model_compression/examples/` for complete examples:
- `quantization.py`: Quantization with MLOps
- `pruning.py`: Pruning with MLOps
- `comprehensive.py`: Full compression pipeline
- `benchmarking.py`: Performance benchmarking
