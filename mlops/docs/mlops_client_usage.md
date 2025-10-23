# MLOps Client Usage Guide

## Overview

The MLOps Client provides a unified interface for all MLOps operations across the EfficientAI-MLX-Toolkit. It integrates:

- **Experiment Tracking** (MLFlow)
- **Data Versioning** (DVC)
- **Model Deployment** (BentoML)
- **Performance Monitoring** (Evidently)
- **Apple Silicon Metrics** collection

## Quick Start

### Basic Usage

```python
from mlops.client import MLOpsClient

# Auto-configure from project namespace
client = MLOpsClient.from_project("lora-finetuning-mlx")

# Start experiment run
with client.start_run(run_name="experiment-001"):
    # Log parameters
    client.log_params({"lr": 0.0001, "epochs": 10})

    # Log metrics
    client.log_metrics({"loss": 0.42, "accuracy": 0.95})

    # Log Apple Silicon metrics
    client.log_apple_silicon_metrics({"mps_utilization": 87.5})
```

### Complete Workflow Example

```python
from mlops.client import create_client
import pandas as pd

# Create client
client = create_client("lora-finetuning-mlx")

# 1. EXPERIMENT TRACKING
with client.start_run(
    run_name="training-run-001",
    tags={"model": "llama", "task": "finetuning"},
    description="Fine-tuning Llama 3.2 1B on custom dataset"
):
    # Log training parameters
    client.log_params({
        "learning_rate": 0.0001,
        "batch_size": 4,
        "epochs": 10,
        "lora_rank": 8,
    })

    # Training loop (simulated)
    for epoch in range(10):
        loss = 0.5 - (epoch * 0.04)
        accuracy = 0.7 + (epoch * 0.02)

        # Log metrics per epoch
        client.log_metrics({
            "loss": loss,
            "accuracy": accuracy,
        }, step=epoch)

        # Log Apple Silicon metrics
        client.log_apple_silicon_metrics({
            "mps_utilization": 85.0 + epoch,
            "memory_gb": 12.5,
            "thermal_state": "nominal",
        }, step=epoch)

    # Log final model artifact
    client.log_artifact("outputs/model.safetensors", artifact_path="model")

# 2. DATA VERSIONING
# Add dataset to DVC tracking
result = client.dvc_add("datasets/train.jsonl")
print(f"DVC file: {result['dvc_file']}")

# Push to remote storage
client.dvc_push()

# Pull from remote (on another machine)
# client.dvc_pull()

# 3. MODEL DEPLOYMENT
deployment = client.deploy_model(
    model_path="outputs/checkpoints/checkpoint_epoch_10",
    model_name="lora_adapter",
    model_version="v1.0",
)
print(f"Deployed: {deployment['model_tag']}")

# 4. MONITORING
# Set reference data for monitoring
train_data = pd.DataFrame({
    "feature1": [1.0, 2.0, 3.0],
    "feature2": [0.5, 1.5, 2.5],
    "target": [0, 1, 0],
    "prediction": [0, 1, 0],
})

client.set_reference_data(
    train_data,
    target_column="target",
    prediction_column="prediction"
)

# Monitor production predictions
prod_data = pd.DataFrame({
    "feature1": [1.5, 2.5, 3.5],
    "feature2": [0.7, 1.7, 2.7],
    "target": [0, 1, 1],
    "prediction": [0, 1, 0],
})

results = client.monitor_predictions(
    prod_data,
    target_column="target",
    prediction_column="prediction",
    latency_ms=45.2,
    memory_mb=256.0,
)

if results.get("retraining_suggested"):
    print("⚠️ Model retraining recommended!")
```

## API Reference

### Initialization

#### `MLOpsClient.from_project(project_name, repo_root=None)`

Auto-configure client for a project.

**Parameters:**
- `project_name` (str): Project namespace identifier
- `repo_root` (str | Path, optional): Repository root directory

**Returns:**
- `MLOpsClient`: Configured client instance

**Example:**
```python
client = MLOpsClient.from_project("lora-finetuning-mlx")
```

#### `create_client(project_name, repo_root=None)`

Convenience function to create a client.

**Example:**
```python
from mlops.client import create_client
client = create_client("model-compression-mlx")
```

### Experiment Tracking

#### `start_run(run_name=None, nested=False, tags=None, description=None)`

Start an experiment run (context manager).

**Parameters:**
- `run_name` (str, optional): Run identifier
- `nested` (bool): Whether this is a nested run
- `tags` (dict, optional): Run tags
- `description` (str, optional): Run description

**Example:**
```python
with client.start_run(run_name="test-run", tags={"env": "dev"}):
    client.log_params({"param1": "value1"})
```

#### `log_params(params)`

Log parameters to current run.

**Parameters:**
- `params` (dict): Parameter dictionary

**Example:**
```python
client.log_params({
    "learning_rate": 0.001,
    "batch_size": 32,
})
```

#### `log_metrics(metrics, step=None)`

Log metrics to current run.

**Parameters:**
- `metrics` (dict): Metric dictionary
- `step` (int, optional): Step number for time series

**Example:**
```python
client.log_metrics({"loss": 0.5, "accuracy": 0.95}, step=100)
```

#### `log_apple_silicon_metrics(metrics, step=None)`

Log Apple Silicon specific metrics.

**Parameters:**
- `metrics` (dict): Apple Silicon metrics
- `step` (int, optional): Step number

**Example:**
```python
client.log_apple_silicon_metrics({
    "mps_utilization": 87.5,
    "memory_gb": 14.2,
    "thermal_state": "nominal",
})
```

#### `log_artifact(local_path, artifact_path=None)`

Log a file artifact.

**Parameters:**
- `local_path` (str | Path): Path to local file
- `artifact_path` (str, optional): Subdirectory in artifact store

**Example:**
```python
client.log_artifact("model.safetensors", artifact_path="models")
```

#### `log_model(model, artifact_path, registered_model_name=None, **kwargs)`

Log a model object.

**Parameters:**
- `model`: Model object
- `artifact_path` (str): Path within artifacts
- `registered_model_name` (str, optional): Name for model registry

**Example:**
```python
client.log_model(model, artifact_path="model", registered_model_name="my-model")
```

### Data Versioning

#### `dvc_add(path, recursive=False)`

Add file or directory to DVC tracking.

**Parameters:**
- `path` (str | Path): Path to track
- `recursive` (bool): Add directory recursively

**Returns:**
- dict: Tracking information

**Example:**
```python
result = client.dvc_add("datasets/train.csv")
print(result["dvc_file"])  # "datasets/train.csv.dvc"
```

#### `dvc_push(targets=None, remote=None)`

Push tracked data to remote storage.

**Parameters:**
- `targets` (list, optional): Specific targets to push
- `remote` (str, optional): Remote name

**Returns:**
- dict: Push status

**Example:**
```python
# Push all
client.dvc_push()

# Push specific files
client.dvc_push(targets=["datasets/train.csv"])
```

#### `dvc_pull(targets=None, remote=None, force=False)`

Pull tracked data from remote storage.

**Parameters:**
- `targets` (list, optional): Specific targets to pull
- `remote` (str, optional): Remote name
- `force` (bool): Force download

**Returns:**
- dict: Pull status

**Example:**
```python
client.dvc_pull(targets=["datasets/train.csv"])
```

### Model Deployment

#### `deploy_model(model_path, model_name, model_version=None, build_bento=True)`

Deploy model using BentoML.

**Parameters:**
- `model_path` (str | Path): Path to model files
- `model_name` (str): Model identifier
- `model_version` (str, optional): Model version
- `build_bento` (bool): Build Bento package

**Returns:**
- dict: Deployment information

**Example:**
```python
result = client.deploy_model(
    model_path="outputs/checkpoints/final",
    model_name="lora_adapter",
    model_version="v1.0",
)
print(result["model_tag"])
```

### Monitoring

#### `set_reference_data(reference_data, target_column=None, prediction_column=None)`

Set reference data for drift and performance monitoring.

**Parameters:**
- `reference_data` (DataFrame): Reference dataset
- `target_column` (str, optional): Target column name
- `prediction_column` (str, optional): Prediction column name

**Example:**
```python
client.set_reference_data(
    train_df,
    target_column="label",
    prediction_column="prediction"
)
```

#### `monitor_predictions(current_data, target_column=None, prediction_column=None, latency_ms=None, memory_mb=None)`

Monitor predictions for drift and performance.

**Parameters:**
- `current_data` (DataFrame): Current data to monitor
- `target_column` (str, optional): Target column name
- `prediction_column` (str, optional): Prediction column name
- `latency_ms` (float, optional): Inference latency
- `memory_mb` (float, optional): Memory usage

**Returns:**
- dict: Monitoring results with drift, performance, and alerts

**Example:**
```python
results = client.monitor_predictions(
    prod_df,
    target_column="label",
    prediction_column="prediction",
    latency_ms=50.0,
)

if results["retraining_suggested"]:
    print("Retraining needed!")
```

### Workspace Management

#### `get_workspace_path(subdir=None)`

Get workspace path for outputs.

**Parameters:**
- `subdir` (str, optional): Subdirectory name

**Returns:**
- Path: Workspace or subdirectory path

**Example:**
```python
workspace = client.get_workspace_path()
models_dir = client.get_workspace_path("models")
```

#### `get_status()`

Get MLOps client status.

**Returns:**
- dict: Status information

**Example:**
```python
status = client.get_status()
print(f"Project: {status['project_name']}")
print(f"MLFlow available: {status['mlflow_available']}")
print(f"DVC available: {status['dvc_available']}")
```

## Configuration

### Custom MLFlow Configuration

```python
from mlops.config.mlflow_config import MLFlowConfig
from mlops.client import MLOpsClient

mlflow_config = MLFlowConfig(
    tracking_uri="http://mlflow-server:5000",
    experiment_name="my-experiment",
    environment="production",
    enable_apple_silicon_metrics=True,
)

client = MLOpsClient(
    project_name="my-project",
    mlflow_config=mlflow_config,
)
```

### Custom DVC Configuration

```python
from mlops.config.dvc_config import DVCConfig
from mlops.client import MLOpsClient

dvc_config = DVCConfig(
    storage_backend="s3",
    remote_url="s3://my-bucket/dvc-storage",
    project_namespace="my-project",
    s3_region="us-west-2",
)

client = MLOpsClient(
    project_name="my-project",
    dvc_config=dvc_config,
)
```

## Error Handling

All client operations raise `MLOpsClientError` on failure:

```python
from mlops.client import MLOpsClient, MLOpsClientError

client = MLOpsClient.from_project("my-project")

try:
    client.dvc_push()
except MLOpsClientError as e:
    print(f"Operation failed: {e}")
    print(f"Component: {e.component}")
    print(f"Operation: {e.operation}")
    print(f"Details: {e.details}")
```

## Best Practices

### 1. Use Context Managers for Runs

Always use `start_run()` as a context manager to ensure proper cleanup:

```python
# ✅ Good
with client.start_run(run_name="training"):
    client.log_params({"lr": 0.001})
    client.log_metrics({"loss": 0.5})

# ❌ Bad (no cleanup on error)
client.mlflow_client.start_run(run_name="training")
client.log_params({"lr": 0.001})
client.mlflow_client.end_run()
```

### 2. Version Data Before Training

Always version datasets before starting experiments:

```python
# Version data first
client.dvc_add("datasets/train.jsonl")
client.dvc_push()

# Then start experiment
with client.start_run(run_name="training"):
    # Training code...
    pass
```

### 3. Log Apple Silicon Metrics

Take advantage of Apple Silicon hardware monitoring:

```python
with client.start_run(run_name="training"):
    for epoch in range(epochs):
        # Training...

        # Log both standard and Apple Silicon metrics
        client.log_metrics({"loss": loss}, step=epoch)
        client.log_apple_silicon_metrics({
            "mps_utilization": get_mps_utilization(),
            "memory_gb": get_memory_usage(),
        }, step=epoch)
```

### 4. Monitor Production Models

Set up monitoring for deployed models:

```python
# During training: set reference data
client.set_reference_data(train_df, "target", "prediction")

# In production: monitor predictions
results = client.monitor_predictions(prod_df, "target", "prediction")

if results.get("retraining_suggested"):
    # Trigger retraining pipeline
    trigger_retraining()
```

### 5. Check Component Availability

Handle optional components gracefully:

```python
client = MLOpsClient.from_project("my-project")
status = client.get_status()

if not status["dvc_available"]:
    print("⚠️ DVC not configured, skipping data versioning")

if status["bentoml_available"]:
    client.deploy_model(model_path, model_name)
```

## Integration Examples

### LoRA Fine-tuning Project

```python
from mlops.client import create_client

def train_lora_model():
    client = create_client("lora-finetuning-mlx")

    # Version training data
    client.dvc_add("datasets/train.jsonl")
    client.dvc_push()

    # Start experiment
    with client.start_run(run_name="lora-training"):
        # Log hyperparameters
        client.log_params({
            "lora_rank": 8,
            "lora_alpha": 16,
            "learning_rate": 0.0001,
        })

        # Training loop
        for epoch in range(10):
            loss = train_epoch()

            client.log_metrics({"loss": loss}, step=epoch)
            client.log_apple_silicon_metrics({
                "mps_utilization": 87.5,
            }, step=epoch)

        # Log model
        client.log_artifact(
            "outputs/adapter.safetensors",
            artifact_path="adapters"
        )

    # Deploy
    client.deploy_model(
        model_path="outputs/adapter",
        model_name="lora_adapter",
    )
```

### Model Compression Project

```python
def compress_model():
    client = create_client("model-compression-mlx")

    with client.start_run(run_name="quantization"):
        # Log compression parameters
        client.log_params({
            "bits": 8,
            "method": "quantization",
        })

        # Compress
        compressed_size = compress()

        # Log results
        client.log_metrics({
            "compression_ratio": 4.0,
            "accuracy_drop": 0.02,
        })

        # Version compressed model
        client.dvc_add("outputs/compressed_model")
        client.dvc_push()
```

## Troubleshooting

### MLFlow Connection Issues

```python
# Check MLFlow status
status = client.get_status()
if not status["mlflow_available"]:
    print("MLFlow not available")
    print(status.get("mlflow_error"))
```

### DVC Remote Issues

```python
# Check DVC configuration
status = client.get_status()
print(status["dvc_connection"])
```

### Component Not Available

```python
try:
    client.deploy_model(model_path, model_name)
except MLOpsClientError as e:
    if "BentoML is not available" in str(e):
        print("Install BentoML: uv add bentoml")
```

## See Also

- [MLFlow Configuration](/mlops/config/mlflow_config.py)
- [DVC Configuration](/mlops/config/dvc_config.py)
- [BentoML Packaging](/mlops/serving/bentoml/packager.py)
- [Evidently Monitoring](/mlops/monitoring/evidently/monitor.py)
