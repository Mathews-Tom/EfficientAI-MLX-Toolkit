# CoreML Stable Diffusion Style Transfer - Usage Guide

This guide provides detailed instructions for using the CoreML Stable Diffusion Style Transfer framework.

## Table of Contents

- [Getting Started](#getting-started)
- [Style Transfer Methods](#style-transfer-methods)
- [Model Management](#model-management)
- [Configuration](#configuration)
- [Performance Optimization](#performance-optimization)
- [Troubleshooting](#troubleshooting)

## Getting Started

### Prerequisites

- Apple Silicon Mac (M1/M2/M3) recommended for optimal performance
- Python 3.10 or higher
- MLX framework installed
- Core ML Tools for model conversion

### Installation

```bash
cd projects/03_CoreML_Stable_Diffusion_Style_Transfer
uv sync
```

### Basic Usage

1. **Validate your setup:**

   ```bash
   uv run python src/cli.py validate
   ```

2. **Get system information:**

   ```bash
   uv run python src/cli.py info
   ```

## Style Transfer Methods

### Diffusion-Based Style Transfer

The framework supports two diffusion-based approaches:

#### 1. Image-to-Image Style Transfer

```bash
uv run python src/cli.py transfer \
  --content-image path/to/content.jpg \
  --style-image path/to/style.jpg \
  --output path/to/output.png \
  --style-strength 0.8 \
  --content-strength 0.2 \
  --steps 50 \
  --guidance-scale 7.5
```

**Parameters:**

- `--style-strength`: How much style to apply (0.0-1.0)
- `--content-strength`: How much content to preserve (0.0-1.0)
- `--steps`: Number of diffusion steps (more steps = better quality, slower)
- `--guidance-scale`: How closely to follow the style guidance

#### 2. Text-Guided Style Transfer

```bash
uv run python src/cli.py transfer \
  --content-image path/to/content.jpg \
  --style-description "in the style of Van Gogh, impressionist painting" \
  --output path/to/output.png \
  --style-strength 0.7 \
  --steps 50
```

### Neural Style Transfer

For faster, real-time processing:

```bash
# Note: Neural style transfer requires a style image
uv run python src/cli.py transfer \
  --content-image path/to/content.jpg \
  --style-image path/to/style.jpg \
  --output path/to/output.png \
  --method neural_style
```

## Model Management

### Training Custom Style Models

Currently, the training functionality is under development. The framework includes:

- **LoRA Fine-tuning**: Lightweight adaptation of Stable Diffusion models
- **Style-specific Training**: Train models for specific artistic styles
- **Callbacks and Monitoring**: Track training progress

### Converting Models to Core ML

Convert trained PyTorch models to Core ML for Apple Silicon optimization:

```bash
uv run python src/cli.py convert \
  --model-path path/to/pytorch_model.pth \
  --output-path path/to/coreml_model.mlpackage \
  --optimize \
  --compute-units all
```

**Compute Units Options:**

- `all`: Use all available compute units (CPU, GPU, Neural Engine)
- `cpu_only`: CPU only
- `cpu_and_gpu`: CPU and GPU, but not Neural Engine

### Serving Models

The serving functionality is under development and will provide:

- REST API for style transfer
- Real-time processing endpoints
- Batch processing capabilities

## Configuration

### Configuration Files

The framework uses YAML configuration files located in `configs/`:

```yaml
# configs/default.yaml
diffusion:
  model_name: "runwayml/stable-diffusion-v1-5"
  num_inference_steps: 50
  guidance_scale: 7.5
  scheduler: "ddim"
  device: "auto"  # auto, mps, cuda, cpu

style_transfer:
  method: "diffusion"  # diffusion or neural_style
  style_strength: 0.7
  content_strength: 0.3
  output_resolution: [512, 512]
  preserve_aspect_ratio: true
  upscale_factor: 1.0
  output_format: "PNG"
  quality: 95

coreml:
  optimize_for_apple_silicon: true
  compute_units: "all"
  precision: "float16"

hardware:
  prefer_mlx: true
  use_mps: true
  memory_optimization: true

inference:
  batch_size: 1
  max_batch_size: 4
  use_attention_slicing: true
  attention_slice_size: "auto"
  use_cpu_offload: false
  workers: 1
  max_requests_per_worker: 100
  timeout: 30
```

### Environment Variables

You can override configuration values using environment variables:

```bash
export DIFFUSION_MODEL_NAME="stabilityai/stable-diffusion-2-1"
export STYLE_TRANSFER_METHOD="neural_style"
export COREML_COMPUTE_UNITS="cpu_and_gpu"
```

## Performance Optimization

### Apple Silicon Optimization

1. **Enable MLX Framework:**

   ```yaml
   hardware:
     prefer_mlx: true
     use_mps: true
   ```

2. **Use Core ML Models:**
   - Convert models to `.mlpackage` format
   - Enable Apple Silicon optimization during conversion
   - Use `float16` precision for better performance

3. **Optimize Memory Usage:**

   ```yaml
   diffusion:
     use_attention_slicing: true
     attention_slice_size: "auto"

   inference:
     use_cpu_offload: true
     memory_optimization: true
   ```

### Benchmarking

Run performance benchmarks to optimize your setup:

```bash
uv run python src/cli.py benchmark \
  --model-path path/to/model \
  --test-images test_images/ \
  --output benchmark_results/ \
  --iterations 10
```

The benchmark will provide:

- Average processing time per image
- Throughput (images per second)
- Memory usage statistics
- Hardware utilization metrics

## Troubleshooting

### Common Issues

1. **Memory Issues:**
   - Reduce `num_inference_steps`
   - Enable `use_attention_slicing`
   - Use smaller `output_resolution`
   - Enable `use_cpu_offload`

2. **Slow Performance:**
   - Ensure Apple Silicon optimization is enabled
   - Use Core ML models instead of PyTorch
   - Check `prefer_mlx` is set to `true`
   - Verify MPS is available and enabled

3. **Model Loading Errors:**
   - Check model path exists
   - Verify model format compatibility
   - Ensure sufficient disk space for model cache

4. **Import Errors:**
   - Run `uv sync` to install dependencies
   - Check Python version compatibility
   - Verify MLX installation on Apple Silicon

### Debug Mode

Enable debug logging for troubleshooting:

```bash
export LOG_LEVEL=DEBUG
uv run python src/cli.py transfer --content-image content.jpg --style-image style.jpg
```

### Getting Help

1. **Check Configuration:**

   ```bash
   uv run python src/cli.py validate
   ```

2. **System Information:**

   ```bash
   uv run python src/cli.py info
   ```

3. **Run Tests:**

   ```bash
   uv run pytest -v
   ```

## Advanced Usage

### Batch Processing

Process multiple images with the same style:

```python
from style_transfer import StyleTransferPipeline, StyleTransferConfig

config = StyleTransferConfig(
    method="diffusion",
    style_strength=0.8,
    output_resolution=(512, 512)
)

pipeline = StyleTransferPipeline(config)

# Process multiple images
content_images = ["img1.jpg", "img2.jpg", "img3.jpg"]
results = pipeline.batch_transfer(
    content_images=content_images,
    style_image="style.jpg",
    output_dir="results/",
    save_images=True
)
```

### Custom Schedulers

Configure different diffusion schedulers:

```yaml
diffusion:
  scheduler: "ddim"  # ddim, pndm, lms, euler, euler_ancestral
```

### Memory Management

For large images or limited memory:

```yaml
style_transfer:
  output_resolution: [256, 256]  # Smaller output size

diffusion:
  use_attention_slicing: true
  attention_slice_size: 1  # Smaller slice size

inference:
  use_cpu_offload: true
  batch_size: 1  # Process one at a time
```

This completes the usage guide. For more advanced topics, refer to the API documentation and source code examples.
