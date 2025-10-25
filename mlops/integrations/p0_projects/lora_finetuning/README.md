# LoRA Fine-tuning MLX MLOps Integration

Comprehensive MLOps integration for LoRA Fine-tuning project, providing experiment tracking, data versioning, model deployment, and performance monitoring.

## Quick Start

```python
from mlops.integrations.p0_projects.lora_finetuning import LoRAMLOpsTracker

# Create tracker
tracker = LoRAMLOpsTracker()

# Track training run
with tracker.start_training_run(run_name="experiment-001") as run:
    # Log configuration
    tracker.log_training_config(lora_config, training_config)

    # Training loop
    for epoch in range(num_epochs):
        metrics = train_epoch(...)
        tracker.log_training_metrics(metrics, epoch=epoch)

    # Save model
    tracker.save_model_artifact("outputs/checkpoints/checkpoint_epoch_2")

# Version dataset
tracker.version_dataset("data/samples/train.jsonl", push_to_remote=True)

# Deploy model
tracker.deploy_adapter(
    adapter_path="outputs/checkpoints/checkpoint_epoch_2",
    model_name="lora_adapter_v1",
)
```

## Features

### 1. Experiment Tracking

Track LoRA training experiments with comprehensive metrics:

```python
with tracker.start_training_run(run_name="lora-exp-001") as run:
    # Log hyperparameters
    tracker.log_training_config(lora_config, training_config)

    # Log training metrics
    tracker.log_training_metrics({
        "train_loss": 0.45,
        "learning_rate": 0.0001,
        "grad_norm": 1.2,
    }, epoch=5)

    # Log Apple Silicon metrics
    tracker.log_apple_silicon_metrics({
        "mps_utilization": 87.5,
        "memory_gb": 14.2,
        "thermal_state": "nominal",
    })
```

**Tracked Parameters:**
- LoRA configuration (rank, alpha, dropout, target modules)
- Training configuration (epochs, batch size, learning rate, optimizer)
- MLX optimization settings (precision, unified memory)
- Model architecture details

**Tracked Metrics:**
- Training loss per epoch
- Validation metrics (if available)
- Learning rate schedule
- Gradient norms
- Inference speed (tokens/second)
- Memory usage
- Apple Silicon utilization (MPS, ANE, unified memory)

### 2. Data Versioning

Version training datasets with DVC for reproducibility:

```python
# Version single dataset
result = tracker.version_dataset(
    "data/samples/train.jsonl",
    push_to_remote=True
)
print(f"DVC file: {result['dvc_file']}")

# Version entire dataset directory
result = tracker.version_dataset(
    "data/datasets/",
    push_to_remote=True
)
```

**Versioned Artifacts:**
- Training datasets (JSONL, CSV, Parquet)
- Validation datasets
- Test datasets
- Tokenizer configurations

### 3. Model Deployment

Deploy LoRA adapters using BentoML:

```python
# Deploy adapter
result = tracker.deploy_adapter(
    adapter_path="outputs/checkpoints/checkpoint_epoch_2",
    model_name="lora_adapter",
    model_version="v1.0",
    build_bento=True,
)

print(f"Deployed: {result['model_tag']}")
print(f"Bento: {result.get('bento_tag')}")
```

**Deployment Features:**
- Automatic BentoML packaging
- Model versioning and tagging
- Inference service generation
- API endpoint creation
- MLX optimization for Apple Silicon

### 4. Performance Monitoring

Monitor inference performance with Evidently:

```python
# Monitor inference batch
results = tracker.monitor_inference(
    prompts=["Hello", "AI is amazing"],
    generated_texts=["Hello world!", "AI is amazing technology"],
    latencies_ms=[150.2, 142.8],
    memory_mb=512.5,
)

# Check for drift
if results.get("drift_detected"):
    print("Data drift detected!")

# Check performance
if results.get("performance_degraded"):
    print("Performance degradation detected!")
```

**Monitoring Capabilities:**
- Inference latency tracking
- Token generation speed
- Memory usage monitoring
- Quality metrics (if ground truth available)
- Data drift detection
- Performance degradation alerts

## Integration with Training Code

### Basic Integration

```python
from pathlib import Path
from lora import LoRAConfig, TrainingConfig, LoRATrainer
from mlops.integrations.p0_projects.lora_finetuning import LoRAMLOpsTracker

# Initialize MLOps tracker
tracker = LoRAMLOpsTracker()

# Load configurations
lora_config = LoRAConfig(rank=16, alpha=32, dropout=0.1)
training_config = TrainingConfig(
    model_name="mlx-community/Llama-3.2-1B-Instruct-4bit",
    num_epochs=3,
    batch_size=4,
    learning_rate=1e-4,
)

# Start tracking
with tracker.start_training_run(run_name="lora-training-001") as run:
    # Log configurations
    tracker.log_training_config(lora_config, training_config)

    # Version dataset
    dataset_path = "data/samples/train.jsonl"
    tracker.version_dataset(dataset_path)

    # Create trainer
    trainer = LoRATrainer(model, lora_config, training_config)

    # Training loop with metrics logging
    for epoch in range(training_config.num_epochs):
        # Train epoch
        metrics = trainer.train_epoch(tokenizer, train_data_path)

        # Log metrics
        tracker.log_training_metrics({
            "train_loss": metrics["loss"],
            "learning_rate": metrics["lr"],
            "tokens_per_second": metrics["speed"],
        }, epoch=epoch)

        # Log Apple Silicon metrics if available
        if "mps_utilization" in metrics:
            tracker.log_apple_silicon_metrics({
                "mps_utilization": metrics["mps_utilization"],
                "memory_gb": metrics["memory_gb"],
            }, step=epoch)

    # Save final model
    checkpoint_path = "outputs/checkpoints/final"
    trainer.save_checkpoint(checkpoint_path)
    tracker.save_model_artifact(checkpoint_path)

# Deploy trained adapter
tracker.deploy_adapter(
    adapter_path=checkpoint_path,
    model_name="lora_adapter_v1",
    model_version="1.0.0",
)
```

### Advanced Integration with Inference Monitoring

```python
from lora import LoRAInferenceEngine
from mlops.integrations.p0_projects.lora_finetuning import LoRAMLOpsTracker

# Initialize tracker
tracker = LoRAMLOpsTracker()

# Load inference engine
engine = LoRAInferenceEngine.from_pretrained(
    model_path="mlx-community/Llama-3.2-1B-Instruct-4bit",
    adapter_path="outputs/checkpoints/final",
)

# Monitor inference
prompts = ["Hello", "AI is", "The future of"]
generated_texts = []
latencies = []

for prompt in prompts:
    import time
    start = time.time()
    result = engine.generate(prompt, max_length=50)
    latency = (time.time() - start) * 1000  # Convert to ms

    generated_texts.append(result.generated_text)
    latencies.append(latency)

# Log monitoring results
monitoring_results = tracker.monitor_inference(
    prompts=prompts,
    generated_texts=generated_texts,
    latencies_ms=latencies,
    memory_mb=512.0,
)

print(f"Monitoring: {monitoring_results}")
```

## CLI Integration

The integration can also be accessed via CLI commands:

```bash
# Version dataset
uv run python -c "
from mlops.integrations.p0_projects.lora_finetuning import LoRAMLOpsTracker
tracker = LoRAMLOpsTracker()
tracker.version_dataset('data/samples/train.jsonl', push_to_remote=True)
"

# Check MLOps status
uv run python -c "
from mlops.integrations.p0_projects.lora_finetuning import create_lora_mlops_client
client = create_lora_mlops_client()
import json
print(json.dumps(client.get_status(), indent=2))
"
```

## Configuration

The integration automatically uses the shared MLOps infrastructure:

- **MLFlow Server**: Centralized experiment tracking
- **DVC Remote**: Shared data storage
- **BentoML Registry**: Model deployment registry
- **Evidently Monitoring**: Performance monitoring dashboard

Configuration is managed through:
- `mlops/config/mlflow_config.py`
- `mlops/config/dvc_config.py`
- Project-specific workspace in `mlops/workspace/lora-finetuning-mlx/`

## Best Practices

### 1. Version Control
- Always version datasets before training
- Push versioned data to remote storage
- Tag experiments with meaningful names

### 2. Experiment Organization
- Use descriptive run names: `{model}_{method}_{date}`
- Add tags for easy filtering: `{"model": "llama", "method": "lora"}`
- Add descriptions for complex experiments

### 3. Model Management
- Save checkpoints at regular intervals
- Log final model as artifact
- Deploy production models with version tags

### 4. Monitoring
- Set up reference data from training set
- Monitor production inference regularly
- Set up alerts for drift/degradation

## Troubleshooting

### MLFlow Connection Issues
```python
tracker = LoRAMLOpsTracker()
status = tracker.get_status()
print(f"MLFlow available: {status['mlflow_available']}")
```

### DVC Remote Issues
```python
# Check DVC connection
tracker = LoRAMLOpsTracker()
status = tracker.get_status()
print(f"DVC available: {status['dvc_available']}")
```

### BentoML Deployment Issues
```python
# Check BentoML availability
tracker = LoRAMLOpsTracker()
status = tracker.get_status()
print(f"BentoML available: {status['bentoml_available']}")
```

## API Reference

### LoRAMLOpsTracker

Main integration class for LoRA fine-tuning MLOps operations.

**Methods:**
- `start_training_run()`: Start MLFlow experiment run
- `log_training_config()`: Log LoRA and training configurations
- `log_training_metrics()`: Log training metrics
- `log_apple_silicon_metrics()`: Log Apple Silicon metrics
- `save_model_artifact()`: Save checkpoint as artifact
- `version_dataset()`: Version dataset with DVC
- `deploy_adapter()`: Deploy LoRA adapter with BentoML
- `monitor_inference()`: Monitor inference performance
- `get_status()`: Get MLOps integration status

See integration.py for detailed API documentation.

## Examples

See `mlops/integrations/p0_projects/lora_finetuning/examples/` for complete examples:
- `basic_training.py`: Basic training with MLOps tracking
- `advanced_training.py`: Advanced training with all features
- `inference_monitoring.py`: Inference monitoring example
- `deployment.py`: Model deployment example
