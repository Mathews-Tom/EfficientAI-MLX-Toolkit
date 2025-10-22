# BentoML Model Packaging Usage Guide

## Overview

The BentoML integration provides Apple Silicon-optimized model packaging and serving for all toolkit projects. This guide covers how to package, deploy, and serve models using BentoML with MLX optimization.

## Features

- **Apple Silicon Optimization**: Automatic MPS, MLX, and unified memory support
- **Model Packaging**: Package MLX models, LoRA adapters, and PyTorch models
- **Service Generation**: Auto-generate BentoML services with health checks and metrics
- **Ray Serve Integration**: Hybrid deployment with BentoML packaging + Ray Serve runtime
- **Multi-Project Support**: Centralized model registry serving all toolkit projects

## Installation

BentoML is included in the toolkit dependencies:

```bash
uv sync
```

## Quick Start

### 1. Package a Model

```python
from mlops.serving.bentoml.packager import package_model
from mlops.serving.bentoml.config import ModelFramework

# Package a LoRA adapter
result = package_model(
    model_path="outputs/checkpoints/checkpoint_epoch_2",
    model_name="lora_adapter_v1",
    project_name="lora-finetuning-mlx",
    model_framework=ModelFramework.MLX,
    build_bento=True,
)

print(f"Model packaged: {result['model_tag']}")
print(f"Bento created: {result['bento_tag']}")
```

### 2. Create a Service

```python
from mlops.serving.bentoml.service import create_lora_service

# Create service for LoRA adapter
service = create_lora_service(
    model_path="outputs/checkpoints/checkpoint_epoch_2",
    service_name="my_lora_service",
    project_name="lora-finetuning-mlx",
)

# Service includes endpoints: /predict, /health, /metrics
```

### 3. Run Service Locally

```bash
# Start BentoML service
bentoml serve service:svc --reload

# Make predictions
curl -X POST http://localhost:3000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Explain artificial intelligence",
    "max_tokens": 100,
    "temperature": 0.7
  }'

# Check health
curl http://localhost:3000/health

# Get metrics
curl http://localhost:3000/metrics
```

## Configuration

### BentoML Configuration

```python
from mlops.serving.bentoml.config import BentoMLConfig, ModelFramework

# Create configuration
config = BentoMLConfig(
    service_name="my_service",
    project_name="my_project",
    model_framework=ModelFramework.MLX,
    workers=2,
    max_batch_size=32,
    enable_apple_silicon_optimization=True,
)

# Auto-detect optimal configuration
config = BentoMLConfig.detect(project_name="my_project")

# Configuration includes:
# - Apple Silicon optimization settings
# - Ray Serve integration
# - Resource allocation
# - API configuration
```

### Apple Silicon Optimization

```python
from mlops.serving.bentoml.config import AppleSiliconOptimization

optimization = AppleSiliconOptimization(
    enable_mps=True,              # Metal Performance Shaders
    enable_mlx=True,              # MLX framework
    enable_unified_memory=True,   # Unified memory architecture
    enable_ane=False,             # Apple Neural Engine
    thermal_aware=True,           # Thermal management
    max_batch_size=32,            # Batch size for unified memory
)
```

## Model Packaging Workflow

### 1. Package Configuration

```python
from mlops.serving.bentoml.packager import PackageConfig, ModelPackager
from mlops.serving.bentoml.config import ModelFramework

# Create package configuration
config = PackageConfig(
    model_path="models/my_model",
    model_name="my_model_v1",
    model_version="1.0.0",
    model_framework=ModelFramework.MLX,
    service_name="my_service",
    project_name="my_project",
    description="MLX-optimized model",
    labels={"env": "production", "team": "ml"},
    metadata={"training_date": "2025-10-22"},
)

# Create packager
packager = ModelPackager(config)
```

### 2. Validation and Collection

```python
# Validate model path
packager.validate_model_path()

# Collect model files (based on include/exclude patterns)
files = packager.collect_model_files()
print(f"Collected {len(files)} model files")
```

### 3. Create BentoML Model

```python
# Create model in BentoML store
model = packager.create_bento_model()
print(f"Model created: {model.tag}")
```

### 4. Build Bento Package

```python
# Build complete Bento package
result = packager.package_model(
    build_bento=True,
    create_docker=False,
)

if result["success"]:
    print(f"Success! Bento: {result['bento_tag']}")
```

## Model Runners

### MLX Model Runner

```python
from mlops.serving.bentoml.runner import MLXModelRunner

# Create runner
runner = MLXModelRunner(
    model_path="models/mlx_model",
    enable_mps=True,
    enable_unified_memory=True,
)

# Load model
runner.load_model()

# Run prediction
result = runner.predict({"input": "test"})

# Check memory usage
memory = runner.get_memory_usage()
print(f"Memory usage: {memory['total_mb']:.2f} MB")

# Unload model
runner.unload_model()
```

### LoRA Model Runner

```python
from mlops.serving.bentoml.runner import LoRAModelRunner

# Create LoRA runner (specialized for LoRA adapters)
runner = LoRAModelRunner(
    model_path="outputs/checkpoints/checkpoint_epoch_2",
    enable_mps=True,
)

# Load LoRA model
runner.load_model()

# Generate text
result = runner.predict({
    "text": "Explain machine learning",
    "max_tokens": 150,
    "temperature": 0.8,
})

print(result["prediction"])
```

## Service Endpoints

All BentoML services include these endpoints:

### 1. Prediction Endpoint

```bash
POST /predict

Request:
{
  "text": "Your input text",
  "max_tokens": 100,
  "temperature": 0.7
}

Response:
{
  "prediction": "Generated output",
  "mlx_optimized": true,
  "model_type": "lora",
  "service_name": "my_service",
  "project_name": "my_project"
}
```

### 2. Health Check Endpoint

```bash
GET /health

Response:
{
  "status": "healthy",
  "service_name": "my_service",
  "project_name": "my_project",
  "model_loaded": true,
  "memory_usage_mb": 1234.56,
  "mlx_available": true
}
```

### 3. Metrics Endpoint

```bash
GET /metrics

Response:
{
  "service_name": "my_service",
  "project_name": "my_project",
  "memory_usage": {
    "cache_mb": 512.0,
    "active_mb": 722.56,
    "total_mb": 1234.56,
    "mlx_available": true
  },
  "model_loaded": true
}
```

## Deployment

### Local Development

```bash
# Serve with hot-reload
bentoml serve service:svc --reload --port 3000

# Serve with production settings
bentoml serve service:svc --workers 2
```

### Docker Deployment

```bash
# Build Docker image
bentoml containerize my_service:latest

# Run container
docker run -p 3000:3000 my_service:latest
```

### Ray Serve Integration

```python
from mlops.serving.bentoml.config import BentoMLConfig, ServingBackend

# Configure for Ray Serve hybrid deployment
config = BentoMLConfig(
    service_name="my_service",
    serving_backend=ServingBackend.HYBRID,  # BentoML + Ray Serve
    ray_serve_enabled=True,
    ray_address="auto",  # Connect to Ray cluster
)

# Ray Serve will handle:
# - Auto-scaling based on load
# - Multi-replica deployment
# - Load balancing
# - Apple Silicon resource management
```

## Best Practices

### 1. Model Organization

```
models/
├── lora-finetuning-mlx/
│   ├── adapter_v1/
│   │   ├── model.safetensors
│   │   ├── config.json
│   │   └── tokenizer.json
│   └── adapter_v2/
├── model-compression-mlx/
│   └── quantized_model/
└── coreml-diffusion/
    └── style_transfer/
```

### 2. Version Management

```python
# Use semantic versioning
config = PackageConfig(
    model_name="lora_adapter",
    model_version="1.2.3",  # major.minor.patch
    labels={
        "stage": "production",
        "git_commit": "abc123",
    },
)
```

### 3. Resource Management

```python
# Configure based on hardware
config = BentoMLConfig.detect(project_name="my_project")

# For Apple Silicon M1/M2
if config.apple_silicon.chip_type in ["M1", "M2"]:
    config.workers = 1  # Conservative for thermal management
    config.max_batch_size = 16

# For Apple Silicon M3+
elif config.apple_silicon.chip_type == "M3":
    config.workers = 2
    config.max_batch_size = 32
```

### 4. Error Handling

```python
from mlops.serving.bentoml.service import BentoMLError

try:
    result = package_model(
        model_path="models/my_model",
        model_name="my_model",
    )
except BentoMLError as e:
    print(f"Packaging failed: {e}")
    print(f"Operation: {e.operation}")
    print(f"Details: {e.details}")
```

## Integration with MLOps Workflow

### 1. Training → Packaging

```python
from mlops.client import create_client
from mlops.serving.bentoml.packager import package_model

# After training
mlflow_client = create_client("my_project")

# Log model to MLFlow
mlflow_client.log_model(model, "trained_model")

# Package with BentoML
result = package_model(
    model_path="outputs/checkpoints/final",
    model_name="trained_model",
    project_name="my_project",
)

# Link in MLFlow
mlflow_client.log_param("bento_tag", result["bento_tag"])
```

### 2. Packaging → Deployment

```python
# 1. Package model
result = package_model(model_path, model_name, project_name)

# 2. Create service
service = create_bentoml_service(
    model_path,
    service_name=model_name,
    project_name=project_name,
)

# 3. Deploy to Ray Serve
# (Handled automatically with ServingBackend.HYBRID)
```

### 3. Monitoring

```python
# Use health and metrics endpoints for monitoring
import requests

health = requests.get("http://localhost:3000/health").json()
metrics = requests.get("http://localhost:3000/metrics").json()

# Log to MLFlow
mlflow_client.log_metrics({
    "serving_memory_mb": metrics["memory_usage"]["total_mb"],
    "model_loaded": int(metrics["model_loaded"]),
})
```

## Troubleshooting

### Issue: Model not loading

```python
# Check model path and files
from pathlib import Path

model_path = Path("models/my_model")
print(f"Path exists: {model_path.exists()}")
print(f"Files: {list(model_path.glob('*'))}")
```

### Issue: Memory errors on Apple Silicon

```python
# Check memory usage
from mlops.serving.bentoml.runner import MLXModelRunner

runner = MLXModelRunner(model_path)
runner.load_model()

memory = runner.get_memory_usage()
print(f"Active memory: {memory['active_mb']:.2f} MB")
print(f"Cached memory: {memory['cache_mb']:.2f} MB")

# Clear cache if needed
import mlx.core as mx
mx.metal.clear_cache()
```

### Issue: Service not responding

```bash
# Check service logs
bentoml logs my_service:latest

# Check Ray Serve status (if using hybrid mode)
ray status

# Check port availability
lsof -i :3000
```

## Advanced Usage

### Custom Model Types

```python
from mlops.serving.bentoml.runner import MLXModelRunner

class CustomModelRunner(MLXModelRunner):
    def _load_mlx_model(self):
        # Custom loading logic
        import mlx.core as mx
        # ... load your model
        return model

    def _mlx_predict(self, input_data):
        # Custom prediction logic
        # ... run inference
        return output

# Use custom runner
from mlops.serving.bentoml.service import create_bentoml_service

service = create_bentoml_service(
    model_path,
    model_type="custom",
    runner_class=CustomModelRunner,
)
```

### Batching Configuration

```python
config = BentoMLConfig(
    max_batch_size=32,
    max_latency_ms=1000,  # Max wait time before processing batch
)

# Bentofile configuration
"""
runner:
  batching:
    enabled: true
    max_batch_size: 32
    max_latency_ms: 1000
"""
```

## References

- [BentoML Documentation](https://docs.bentoml.org/)
- [MLX Framework](https://ml-explore.github.io/mlx/)
- [Ray Serve Documentation](https://docs.ray.io/en/latest/serve/index.html)
- MLOps Integration Spec: `docs/specs/mlops-integration/spec.md`
- Implementation Plan: `docs/specs/mlops-integration/plan.md`
