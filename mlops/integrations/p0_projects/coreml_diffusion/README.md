# CoreML Stable Diffusion Style Transfer MLOps Integration

Comprehensive MLOps integration for CoreML Stable Diffusion Style Transfer project, providing experiment tracking, model versioning, deployment, and performance monitoring for style transfer and CoreML conversion workflows.

## Quick Start

```python
from mlops.integrations.p0_projects.coreml_diffusion import DiffusionMLOpsTracker

# Create tracker
tracker = DiffusionMLOpsTracker()

# Track style transfer run
with tracker.start_transfer_run(run_name="artistic-style-001") as run:
    # Log configuration
    tracker.log_transfer_config(style_config)

    # Perform style transfer
    result_image = transfer_style(content_image, style_image)

    # Log metrics
    tracker.log_transfer_metrics({
        "style_similarity": 0.87,
        "content_preservation": 0.92,
        "transfer_time_s": 3.5,
    })

    # Save output
    tracker.save_output_artifact("outputs/styled_image.png")

# Deploy CoreML model
tracker.deploy_coreml_model(
    model_path="outputs/coreml/style_transfer.mlpackage",
    model_name="style_transfer_v1",
)
```

## Features

### 1. Experiment Tracking

Track style transfer experiments with comprehensive metrics:

```python
with tracker.start_transfer_run(run_name="artistic-style-001") as run:
    # Log style transfer config
    tracker.log_transfer_config(style_config)

    # Log diffusion config
    tracker.log_diffusion_config(diffusion_config)

    # Log style transfer metrics
    tracker.log_transfer_metrics({
        "style_similarity": 0.87,
        "content_preservation": 0.92,
        "transfer_time_s": 3.5,
        "memory_mb": 1024.5,
        "quality_score": 0.89,
    })

    # Log Apple Silicon metrics
    tracker.log_apple_silicon_metrics({
        "ane_utilization": 92.5,
        "memory_gb": 8.5,
        "thermal_state": "nominal",
    })
```

**Tracked Parameters:**
- Style transfer: strength, content preservation, resolution
- Diffusion model: steps, guidance scale, scheduler
- CoreML: optimization level, compute units, precision
- Hardware: MLX/MPS/ANE preferences

**Tracked Metrics:**
- Style similarity scores
- Content preservation metrics
- Transfer time
- Memory usage
- Image quality scores
- Apple Silicon utilization (ANE, MPS, unified memory)

### 2. Model and Image Versioning

Version CoreML models and image datasets with DVC:

```python
# Version CoreML model
result = tracker.version_model(
    "outputs/coreml/style_transfer.mlpackage",
    push_to_remote=True
)

# Version image datasets
result = tracker.version_images(
    "data/styles/",
    push_to_remote=True
)
```

**Versioned Artifacts:**
- CoreML models (.mlpackage, .mlmodel)
- PyTorch models (for comparison)
- Style images
- Content images
- Generated outputs
- Conversion metadata

### 3. Model Deployment

Deploy CoreML models using BentoML:

```python
# Deploy CoreML model
result = tracker.deploy_coreml_model(
    model_path="outputs/coreml/style_transfer.mlpackage",
    model_name="style_transfer",
    model_version="v1.0",
    build_bento=True,
)

print(f"Deployed: {result['model_tag']}")
```

**Deployment Features:**
- CoreML model packaging
- Apple Silicon optimization
- ANE acceleration
- API endpoint generation
- Batch inference support

### 4. Performance Monitoring

Monitor style transfer performance:

```python
# Monitor inference batch
results = tracker.monitor_inference(
    image_sizes=[(512, 512), (768, 768), (1024, 1024)],
    latencies_ms=[3500, 5200, 8900],
    memory_mb=1024.5,
    quality_scores=[0.87, 0.89, 0.91],
)
```

**Monitoring Capabilities:**
- Inference latency by image size
- Memory usage tracking
- Quality score monitoring
- Throughput measurement
- Performance degradation detection

## Integration with Style Transfer Code

### Basic Style Transfer Integration

```python
from pathlib import Path
from PIL import Image
from style_transfer import StyleTransferPipeline, StyleTransferConfig
from mlops.integrations.p0_projects.coreml_diffusion import DiffusionMLOpsTracker

# Initialize tracker
tracker = DiffusionMLOpsTracker()

# Configure style transfer
style_config = StyleTransferConfig(
    style_strength=0.8,
    content_strength=0.6,
    num_inference_steps=50,
    guidance_scale=7.5,
)

# Load images
content_image = Image.open("data/content/photo.jpg")
style_image = Image.open("data/styles/painting.jpg")

# Start tracking
with tracker.start_transfer_run(run_name="artistic-style-001") as run:
    # Log configuration
    tracker.log_transfer_config(style_config)

    # Version input images
    tracker.version_images("data/styles/", push_to_remote=False)

    # Perform style transfer
    import time
    start_time = time.time()

    pipeline = StyleTransferPipeline(style_config)
    result_image = pipeline.transfer_style(content_image, style_image)

    transfer_time = time.time() - start_time

    # Compute quality metrics (if available)
    quality_score = compute_quality_score(result_image, content_image, style_image)

    # Log metrics
    tracker.log_transfer_metrics({
        "style_strength": style_config.style_strength,
        "content_strength": style_config.content_strength,
        "transfer_time_s": transfer_time,
        "memory_mb": get_memory_usage(),
        "quality_score": quality_score,
        "image_width": result_image.width,
        "image_height": result_image.height,
    })

    # Save output
    output_path = "outputs/styled_image.png"
    result_image.save(output_path)
    tracker.save_output_artifact(output_path)
```

### CoreML Conversion Integration

```python
from coreml import CoreMLConverter, CoreMLConfig
from mlops.integrations.p0_projects.coreml_diffusion import DiffusionMLOpsTracker

# Initialize tracker
tracker = DiffusionMLOpsTracker()

# Configure CoreML conversion
coreml_config = CoreMLConfig(
    optimize_for_apple_silicon=True,
    compute_units="all",
    precision="float16",
    quantization="linear",
    use_ane=True,
)

# Start tracking
with tracker.start_conversion_run(run_name="coreml-conversion-001") as run:
    # Log configuration
    tracker.log_coreml_config(coreml_config)

    # Convert model
    import time
    start_time = time.time()

    converter = CoreMLConverter(coreml_config)
    coreml_model = converter.convert_model("pytorch_model/")

    conversion_time = time.time() - start_time

    # Get model info
    model_size_mb = get_model_size("outputs/coreml/model.mlpackage")
    ane_compatible = check_ane_compatibility(coreml_model)

    # Log metrics
    tracker.log_conversion_metrics({
        "conversion_time_s": conversion_time,
        "model_size_mb": model_size_mb,
        "ane_compatible": 1.0 if ane_compatible else 0.0,
        "precision": coreml_config.precision,
    })

    # Save CoreML model
    output_path = "outputs/coreml/style_transfer.mlpackage"
    converter.save_model(coreml_model, output_path)
    tracker.save_coreml_model(output_path)

# Version and deploy
tracker.version_model(output_path, push_to_remote=True)
tracker.deploy_coreml_model(
    model_path=output_path,
    model_name="style_transfer_v1",
)
```

### Benchmarking Integration

```python
from mlops.integrations.p0_projects.coreml_diffusion import DiffusionMLOpsTracker

# Initialize tracker
tracker = DiffusionMLOpsTracker()

# Benchmark PyTorch vs CoreML
with tracker.start_transfer_run(run_name="benchmark-pytorch-vs-coreml") as run:
    # Benchmark PyTorch model
    pytorch_metrics = benchmark_pytorch_model(
        model_path="pytorch_model/",
        test_images=["test1.jpg", "test2.jpg", "test3.jpg"],
    )

    # Benchmark CoreML model
    coreml_metrics = benchmark_coreml_model(
        model_path="outputs/coreml/style_transfer.mlpackage",
        test_images=["test1.jpg", "test2.jpg", "test3.jpg"],
    )

    # Log comparison
    tracker.log_benchmark_metrics(
        pytorch_metrics=pytorch_metrics,
        coreml_metrics=coreml_metrics,
    )

    # Log summary
    speedup = pytorch_metrics["inference_time_s"] / coreml_metrics["inference_time_s"]
    print(f"CoreML speedup: {speedup:.2f}x")
```

### Advanced Monitoring Integration

```python
from mlops.integrations.p0_projects.coreml_diffusion import DiffusionMLOpsTracker
from inference import InferenceEngine

# Initialize tracker and engine
tracker = DiffusionMLOpsTracker()
engine = InferenceEngine.from_coreml("outputs/coreml/style_transfer.mlpackage")

# Batch inference with monitoring
test_images = load_test_images("data/test/")
results = []

for i, (content, style) in enumerate(test_images):
    import time
    start = time.time()

    result = engine.transfer_style(content, style)
    latency = (time.time() - start) * 1000  # ms

    results.append({
        "image_size": (content.width, content.height),
        "latency_ms": latency,
        "quality_score": compute_quality(result),
    })

# Monitor batch
monitoring_results = tracker.monitor_inference(
    image_sizes=[r["image_size"] for r in results],
    latencies_ms=[r["latency_ms"] for r in results],
    memory_mb=get_memory_usage(),
    quality_scores=[r["quality_score"] for r in results],
)

# Check for issues
if monitoring_results.get("performance_degraded"):
    print("⚠️ Performance degradation detected!")
```

## CLI Integration

```bash
# Track style transfer with MLOps
uv run python -c "
from mlops.integrations.p0_projects.coreml_diffusion import DiffusionMLOpsTracker
from style_transfer import StyleTransferConfig

tracker = DiffusionMLOpsTracker()
config = StyleTransferConfig(style_strength=0.8)

with tracker.start_transfer_run('artistic-001'):
    tracker.log_transfer_config(config)
    # ... transfer code ...
"

# Check MLOps status
uv run python -c "
from mlops.integrations.p0_projects.coreml_diffusion import create_diffusion_mlops_client
client = create_diffusion_mlops_client()
import json
print(json.dumps(client.get_status(), indent=2))
"
```

## Configuration

The integration uses shared MLOps infrastructure:
- MLFlow experiment: `coreml-stable-diffusion-style-transfer`
- DVC namespace: `coreml-stable-diffusion-style-transfer`
- Workspace: `mlops/workspace/coreml-stable-diffusion-style-transfer/`

## Best Practices

### 1. Experiment Organization
- Use descriptive run names: `{style}_{date}`
- Tag by style type: `{"style": "artistic", "resolution": "1024"}`
- Document settings in descriptions

### 2. Model Management
- Version all CoreML models
- Track conversion settings
- Compare PyTorch vs CoreML performance

### 3. Quality Assurance
- Monitor output quality
- Track style similarity
- Alert on quality drops

### 4. Performance Optimization
- Benchmark different resolutions
- Test ANE acceleration
- Optimize for target hardware

## Troubleshooting

### CoreML Conversion Issues
```python
tracker = DiffusionMLOpsTracker()
with tracker.start_conversion_run("debug-conversion"):
    try:
        converter.convert_model(...)
    except Exception as e:
        tracker.log_params({"error": str(e)})
        raise
```

### ANE Compatibility Issues
```python
# Check ANE compatibility
coreml_model = load_coreml_model("model.mlpackage")
ane_compatible = check_ane_compatibility(coreml_model)
print(f"ANE compatible: {ane_compatible}")
```

## API Reference

### DiffusionMLOpsTracker

Main integration class for style transfer MLOps operations.

**Methods:**
- `start_transfer_run()`: Start style transfer experiment
- `start_conversion_run()`: Start CoreML conversion experiment
- `log_diffusion_config()`: Log diffusion settings
- `log_transfer_config()`: Log style transfer settings
- `log_coreml_config()`: Log CoreML settings
- `log_transfer_metrics()`: Log transfer metrics
- `log_conversion_metrics()`: Log conversion metrics
- `log_benchmark_metrics()`: Log benchmark comparisons
- `save_output_artifact()`: Save output image
- `save_coreml_model()`: Save CoreML model
- `version_model()`: Version model with DVC
- `version_images()`: Version image datasets with DVC
- `deploy_coreml_model()`: Deploy with BentoML
- `monitor_inference()`: Monitor performance
- `get_status()`: Get integration status

See integration.py for detailed API documentation.

## Examples

See `mlops/integrations/p0_projects/coreml_diffusion/examples/` for complete examples:
- `style_transfer.py`: Style transfer with MLOps
- `coreml_conversion.py`: CoreML conversion with MLOps
- `benchmarking.py`: Performance benchmarking
- `monitoring.py`: Production monitoring
