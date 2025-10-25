# MLOps Complete Guide

**Comprehensive documentation for the Apple Silicon-optimized MLOps integration**

Version: 1.0.0
Last Updated: 2025-10-24

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Getting Started](#getting-started)
4. [Core Components](#core-components)
5. [Integration Patterns](#integration-patterns)
6. [Apple Silicon Optimization](#apple-silicon-optimization)
7. [Security](#security)
8. [Operations](#operations)
9. [Troubleshooting](#troubleshooting)
10. [Best Practices](#best-practices)
11. [API Reference](#api-reference)
12. [Examples](#examples)

---

## Overview

### What is This System?

The EfficientAI MLOps integration provides a complete, production-ready MLOps stack specifically optimized for Apple Silicon hardware. It integrates industry-standard tools (MLFlow, DVC, BentoML, Evidently) with Apple Silicon-specific optimizations for maximum performance.

### Key Features

- **Unified MLOps Client**: Single interface for all MLOps operations
- **Experiment Tracking**: Comprehensive MLFlow integration with Apple Silicon metrics
- **Data Versioning**: DVC integration for datasets and models
- **Model Deployment**: BentoML packaging and serving
- **Performance Monitoring**: Evidently integration with drift detection
- **Apple Silicon Optimization**: Hardware-aware configuration and monitoring
- **Thermal-Aware Scheduling**: Airflow integration with thermal management
- **Dashboard**: Unified monitoring and visualization

### System Requirements

- **Hardware**: Apple Silicon (M1/M2/M3 series)
- **OS**: macOS 12.0+
- **Python**: 3.10+
- **Memory**: 16GB+ recommended
- **Storage**: 50GB+ for artifacts and data

### Installation

```bash
# Install toolkit with MLOps dependencies
cd /path/to/EfficientAI-MLX-Toolkit
uv sync

# Verify installation
uv run python -c "from mlops.client.mlops_client import MLOpsClient; print('MLOps installed successfully')"
```

---

## Architecture

### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Application Layer                       │
│  (LoRA Fine-tuning, Model Compression, CoreML Diffusion)    │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│                    Integration Layer                         │
│  (Project-specific MLOps Trackers)                          │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│                   MLOps Client Layer                         │
│  (Unified client coordinating all operations)               │
└───────────────────────────┬─────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
┌───────▼────────┐  ┌──────▼──────┐  ┌─────────▼────────┐
│   Tracking     │  │  Versioning │  │    Serving       │
│   (MLFlow)     │  │    (DVC)    │  │   (BentoML)      │
└────────────────┘  └─────────────┘  └──────────────────┘
        │                   │                   │
┌───────▼────────┐  ┌──────▼──────┐  ┌─────────▼────────┐
│  Monitoring    │  │    Apple    │  │  Orchestration   │
│ (Evidently)    │  │   Silicon   │  │   (Airflow)      │
└────────────────┘  └─────────────┘  └──────────────────┘
```

### Component Overview

| Component | Purpose | Technology |
|-----------|---------|------------|
| **MLFlow** | Experiment tracking and model registry | MLFlow Server |
| **DVC** | Data and model versioning | DVC + Cloud Storage |
| **BentoML** | Model packaging and deployment | BentoML + Ray Serve |
| **Evidently** | Model monitoring and drift detection | Evidently AI |
| **Airflow** | Workflow orchestration | Apache Airflow |
| **Apple Silicon Monitor** | Hardware metrics collection | Custom |
| **Dashboard** | Unified visualization | FastAPI + Plotly |

### Directory Structure

```
mlops/
├── client/                    # Unified MLOps client
│   ├── mlops_client.py       # Main client interface
│   ├── mlflow_client.py      # MLFlow integration
│   └── dvc_client.py         # DVC integration
├── tracking/                  # Experiment tracking
│   └── apple_silicon_metrics.py
├── versioning/                # Data versioning
│   ├── dvc_operations.py
│   └── remote_manager.py
├── serving/                   # Model serving
│   ├── bentoml/              # BentoML integration
│   ├── model_wrapper.py
│   └── ray_serve.py
├── monitoring/                # Performance monitoring
│   └── evidently/            # Evidently integration
├── orchestration/             # Workflow orchestration
│   ├── dag_templates.py
│   └── resource_manager.py
├── silicon/                   # Apple Silicon optimization
│   ├── detector.py
│   ├── monitor.py
│   └── optimizer.py
├── airflow/                   # Airflow integration
│   ├── thermal_scheduler.py
│   └── operators.py
├── dashboard/                 # Visualization
│   ├── server.py
│   └── data_aggregator.py
├── workspace/                 # Project workspaces
├── integrations/              # Project integrations
│   └── p0_projects/
├── tests/                     # Test suite
├── benchmarks/                # Performance benchmarks
├── examples/                  # Example workflows
└── docs/                      # Documentation
```

---

## Getting Started

### Quick Start (5 Minutes)

1. **Initialize MLOps Client**

```python
from mlops.client.mlops_client import MLOpsClient

# Auto-configured for your project
client = MLOpsClient(project_namespace="my-project")
```

2. **Track an Experiment**

```python
# Create experiment
experiment_id = client.create_experiment("first-experiment")

# Start run
with client.start_run(run_name="run-001", experiment_id=experiment_id):
    # Log parameters
    client.log_params({"learning_rate": 0.001, "batch_size": 32})

    # Training loop
    for epoch in range(10):
        metrics = {"loss": 1.0 - epoch * 0.1, "accuracy": 0.7 + epoch * 0.03}
        client.log_metrics(metrics, step=epoch)

    # Save model
    client.log_artifact("outputs/model.bin")
```

3. **View Results**

```bash
# Start MLFlow UI
mlflow ui --backend-store-uri mlops/workspace/my-project/mlruns

# Open browser
open http://localhost:5000
```

### Complete Workflow Example

```python
import pandas as pd
from mlops.client.mlops_client import MLOpsClient

# Initialize client
client = MLOpsClient(project_namespace="complete-example")

# 1. Version dataset
dataset = pd.DataFrame({"feature": [1, 2, 3], "label": [0, 1, 0]})
dataset.to_csv("data/train.csv", index=False)
client.version_dataset("data/train.csv", push_to_remote=True)

# 2. Track experiment
experiment_id = client.create_experiment("training-experiment")

with client.start_run(run_name="experiment-001", experiment_id=experiment_id):
    # Log config
    client.log_params({"model": "example", "version": "1.0"})

    # Collect Apple Silicon metrics
    silicon_metrics = client.collect_apple_silicon_metrics()
    print(f"Running on {silicon_metrics['chip_type']}")

    # Train (simulated)
    for epoch in range(5):
        client.log_metrics({"train_loss": 0.5 - epoch * 0.05}, step=epoch)

    # Save model
    client.log_artifact("outputs/model")

# 3. Deploy model
deployment_result = client.deploy_model(
    model_path="outputs/model",
    model_name="example_model",
    version="v1"
)
print(f"Deployment status: {deployment_result['status']}")

# 4. Monitor performance
reference_data = pd.DataFrame({"feature": [1, 2, 3, 4, 5]})
client.set_reference_data(reference_data)

current_data = pd.DataFrame({"feature": [6, 7, 8, 9, 10]})
drift_report = client.evidently_monitor.generate_drift_report(
    current_data, reference_data
)
if drift_report["drift_detected"]:
    print("Warning: Data drift detected!")
```

---

## Core Components

### MLOps Client

**Purpose**: Unified interface for all MLOps operations

**Key Methods**:
- `create_experiment()` - Create MLFlow experiment
- `start_run()` - Start tracking run
- `log_params()` - Log hyperparameters
- `log_metrics()` - Log training/inference metrics
- `log_artifact()` - Save model artifacts
- `version_dataset()` - Version data with DVC
- `deploy_model()` - Deploy model with BentoML
- `collect_apple_silicon_metrics()` - Collect hardware metrics

**Configuration**:
```python
client = MLOpsClient(
    project_namespace="my-project",        # Project identifier
    mlflow_tracking_uri="http://localhost:5000",  # MLFlow server
    dvc_remote_name="storage",             # DVC remote
    workspace_root="mlops/workspace"       # Workspace directory
)
```

### MLFlow Integration

**Features**:
- Experiment management and organization
- Run tracking with parameters and metrics
- Model registry and versioning
- Artifact storage
- Apple Silicon metrics integration
- Tag-based filtering and search

**Usage**:
```python
from mlops.client.mlflow_client import create_client

client = create_client(project_name="my-project")

# Create experiment
experiment_id = client.create_experiment("training", tags={"env": "production"})

# Track run
with client.start_run(run_name="run-001", experiment_id=experiment_id) as run:
    client.log_params({"lr": 0.001})
    client.log_metrics({"loss": 0.5}, step=0)
    client.log_artifact("model.bin")
```

### DVC Integration

**Features**:
- Data and model versioning
- Remote storage (S3, GCS, Azure, local)
- Efficient deduplication
- Pipeline tracking
- Reproducibility

**Usage**:
```python
from mlops.client.dvc_client import create_client

client = create_client(project_name="my-project")

# Initialize DVC
client.init()

# Version data
client.add("data/train.csv")
client.push()

# Pull data
client.pull()

# Check status
status = client.status()
```

### BentoML Integration

**Features**:
- Model packaging
- API serving
- Auto-scaling
- Docker containerization
- Ray Serve backend

**Usage**:
```python
from mlops.serving.bentoml.packager import package_model

result = package_model(
    model_path="outputs/model",
    model_name="my_model",
    model_framework="mlx",
    build_docker=True
)

print(f"Bento: {result['bento_name']}")
```

### Evidently Integration

**Features**:
- Data drift detection
- Model performance monitoring
- Statistical tests
- Interactive reports
- Alerting

**Usage**:
```python
from mlops.monitoring.evidently.monitor import EvidentlyMonitor

monitor = EvidentlyMonitor(project_name="my-project")

# Generate drift report
report = monitor.generate_drift_report(current_data, reference_data)

if report["drift_detected"]:
    print(f"Drift score: {report['drift_score']}")
```

---

## Integration Patterns

### Pattern 1: Project-Specific Tracker

**Use Case**: Integrate MLOps into existing project

**Implementation**:
```python
from mlops.client.mlops_client import MLOpsClient

class MyProjectTracker:
    """Custom tracker for my project."""

    def __init__(self):
        self.client = MLOpsClient(project_namespace="my-project")

    def start_training(self, config):
        """Start training with tracking."""
        experiment_id = self.client.create_experiment("training")
        return self.client.start_run(
            run_name=f"train_{config['model']}",
            experiment_id=experiment_id
        )

    def log_training_metrics(self, metrics, epoch):
        """Log training metrics."""
        self.client.log_metrics(metrics, step=epoch)
```

### Pattern 2: Decorator-Based Tracking

**Use Case**: Add tracking to existing functions

**Implementation**:
```python
from functools import wraps
from mlops.client.mlops_client import MLOpsClient

def track_experiment(project_namespace, run_name):
    """Decorator for automatic experiment tracking."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            client = MLOpsClient(project_namespace=project_namespace)
            experiment_id = client.create_experiment(func.__name__)

            with client.start_run(run_name=run_name, experiment_id=experiment_id):
                result = func(*args, **kwargs)
                return result
        return wrapper
    return decorator

@track_experiment(project_namespace="my-project", run_name="training")
def train_model(config):
    # Training code here
    pass
```

### Pattern 3: Context Manager

**Use Case**: Clean resource management

**Implementation**:
```python
class MLOpsContext:
    """Context manager for MLOps operations."""

    def __init__(self, project_namespace, run_name):
        self.client = MLOpsClient(project_namespace=project_namespace)
        self.run_name = run_name
        self.experiment_id = None

    def __enter__(self):
        self.experiment_id = self.client.create_experiment("training")
        self.run = self.client.start_run(
            run_name=self.run_name,
            experiment_id=self.experiment_id
        )
        self.run.__enter__()
        return self.client

    def __exit__(self, *args):
        self.run.__exit__(*args)

# Usage
with MLOpsContext("my-project", "run-001") as client:
    client.log_params({"lr": 0.001})
    # Training code
```

---

## Apple Silicon Optimization

### Hardware Detection

**Automatic Detection**:
```python
from mlops.silicon.detector import AppleSiliconDetector

detector = AppleSiliconDetector()
info = detector.detect()

print(f"Chip: {info['chip_type']}")
print(f"Memory: {info['unified_memory_gb']} GB")
print(f"MPS Available: {info['mps_available']}")
print(f"ANE Available: {info['ane_available']}")
```

### Metrics Collection

**Hardware Metrics**:
```python
from mlops.silicon.monitor import AppleSiliconMonitor

monitor = AppleSiliconMonitor()
metrics = monitor.get_metrics()

# Metrics include:
# - chip_type
# - unified_memory_gb
# - mps_available
# - ane_available
# - performance_cores
# - efficiency_cores
# - thermal_state
# - power_mode
# - memory_pressure
```

### Thermal-Aware Scheduling

**Airflow Integration**:
```python
from mlops.airflow.thermal_scheduler import ThermalAwareScheduler

scheduler = ThermalAwareScheduler()

# Check if task should run
decision = scheduler.should_run_task(
    task_type="training",
    priority="high",
    estimated_duration_minutes=60
)

if decision.should_run:
    # Run task
    pass
else:
    # Wait or reschedule
    print(f"Reason: {decision.reason}")
```

### Optimization Patterns

**Memory-Efficient Training**:
```python
# Use unified memory effectively
config = {
    "batch_size": 32,  # Optimized for M3
    "gradient_accumulation_steps": 4,
    "mixed_precision": True,  # Use MPS mixed precision
}
```

**MPS Backend**:
```python
import mlx.core as mx

# Automatically uses MPS when available
x = mx.array([1, 2, 3])
y = mx.array([4, 5, 6])
result = x + y  # Runs on MPS
```

---

## Security

### Authentication Setup

**MLFlow Authentication**:
```bash
# Set environment variables
export MLFLOW_TRACKING_USERNAME=admin
export MLFLOW_TRACKING_PASSWORD=secure_password

# Or use config file
cat > mlops/config/mlflow_auth.yaml <<EOF
basic_auth:
  username: admin
  password: ${MLFLOW_PASSWORD}
EOF
```

**DVC Remote Credentials**:
```bash
# AWS S3
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret

# Or configure in DVC
dvc remote modify storage access_key_id your_key
dvc remote modify storage secret_access_key your_secret --local
```

### Secrets Management

**Use Environment Variables**:
```python
import os
from pathlib import Path

# Good: Environment variables
api_key = os.getenv("API_KEY")

# Good: Read from secure file
with open(Path.home() / ".secrets" / "api_key") as f:
    api_key = f.read().strip()

# Bad: Hardcoded
api_key = "sk-123456789"  # DON'T DO THIS
```

**Use .env Files**:
```bash
# .env (add to .gitignore)
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_TRACKING_USERNAME=admin
MLFLOW_TRACKING_PASSWORD=secure_password
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
```

### Security Checklist

See [SECURITY_CHECKLIST.md](SECURITY_CHECKLIST.md) for complete checklist.

**Critical Items**:
- [ ] All credentials in environment variables
- [ ] HTTPS enabled for MLFlow
- [ ] DVC remote access restricted
- [ ] API authentication enabled
- [ ] Secrets not in git history
- [ ] Regular security audits

---

## Operations

See [OPERATIONS_GUIDE.md](OPERATIONS_GUIDE.md) for detailed operations guide.

### Starting Services

**MLFlow Server**:
```bash
# Start MLFlow
mlflow server \
    --backend-store-uri sqlite:///mlops/workspace/mlflow.db \
    --default-artifact-root mlops/workspace/artifacts \
    --host 0.0.0.0 \
    --port 5000
```

**Dashboard**:
```bash
# Start dashboard
uv run python mlops/dashboard/server.py
```

**BentoML Serving**:
```bash
# Serve model
bentoml serve my_model:latest --port 3000
```

### Monitoring

**Health Checks**:
```bash
# MLFlow
curl http://localhost:5000/health

# BentoML
curl http://localhost:3000/healthz

# Dashboard
curl http://localhost:8000/health
```

**Log Monitoring**:
```bash
# View MLFlow logs
tail -f mlops/workspace/logs/mlflow.log

# View dashboard logs
tail -f mlops/workspace/logs/dashboard.log
```

### Backup & Recovery

**Backup MLFlow Data**:
```bash
# Backup experiments
tar -czf mlflow_backup_$(date +%Y%m%d).tar.gz mlops/workspace/mlruns/

# Backup database
sqlite3 mlops/workspace/mlflow.db ".backup 'mlflow_backup.db'"
```

**Backup DVC Data**:
```bash
# Push to remote
dvc push

# Verify remote
dvc list-url s3://your-bucket/dvc-storage
```

---

## Troubleshooting

### Common Issues

#### MLFlow Connection Failed

**Symptom**: `ConnectionError: Cannot connect to MLFlow server`

**Solutions**:
1. Check if server is running: `curl http://localhost:5000/health`
2. Check firewall: `sudo lsof -i :5000`
3. Verify tracking URI: `echo $MLFLOW_TRACKING_URI`
4. Check logs: `tail -f mlops/workspace/logs/mlflow.log`

#### DVC Remote Not Configured

**Symptom**: `DVC remote 'storage' not found`

**Solutions**:
1. List remotes: `dvc remote list`
2. Add remote: `dvc remote add -d storage s3://bucket/path`
3. Configure credentials: Set AWS environment variables
4. Test connection: `dvc remote list-url storage`

#### BentoML Import Error

**Symptom**: `ModuleNotFoundError: No module named 'bentoml'`

**Solutions**:
1. Install BentoML: `uv add bentoml`
2. Verify installation: `uv run python -c "import bentoml; print(bentoml.__version__)"`
3. Check dependencies: `uv pip list | grep bentoml`

#### Apple Silicon Metrics Not Collected

**Symptom**: Empty or missing hardware metrics

**Solutions**:
1. Verify Apple Silicon: `sysctl machdep.cpu.brand_string`
2. Check MPS availability: `python -c "import torch; print(torch.backends.mps.is_available())"`
3. Update system: Software Update in System Preferences
4. Check logs: Monitor output for warnings

### Debug Mode

**Enable Debug Logging**:
```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("mlops")
logger.setLevel(logging.DEBUG)
```

**Verbose Client**:
```python
client = MLOpsClient(
    project_namespace="debug-project",
    verbose=True  # Enable verbose output
)
```

---

## Best Practices

### Experiment Organization

**Naming Conventions**:
```python
# Good
run_name = f"{model_name}_{method}_{timestamp}"
experiment_name = f"{project}_training_{stage}"
tag_dict = {"model": "llama", "method": "lora", "stage": "dev"}

# Bad
run_name = "run1"
experiment_name = "experiment"
```

**Hierarchical Structure**:
```
experiments/
├── baseline/
│   ├── run-001
│   ├── run-002
│   └── run-003
├── optimization/
│   ├── lr-tuning/
│   └── batch-size/
└── production/
    └── final-model/
```

### Data Versioning

**Version Early, Version Often**:
```python
# Before training
client.version_dataset("data/train.csv", push_to_remote=True)
client.version_dataset("data/val.csv", push_to_remote=True)

# After training
client.version_dataset("outputs/model", push_to_remote=True)
```

**Use Tags**:
```python
client.dvc_client.add("data/train.csv")
# Tag with git
import subprocess
subprocess.run(["git", "tag", "-a", "data-v1.0", "-m", "Training data v1.0"])
```

### Model Deployment

**Incremental Rollout**:
```python
# Deploy canary
client.deploy_model(model_path="model", model_name="prod_model", version="canary")

# Monitor performance
monitor_metrics(duration_minutes=60)

# Full rollout
client.deploy_model(model_path="model", model_name="prod_model", version="v1.0")
```

### Performance Optimization

**Batch Logging**:
```python
# Good: Batch metrics
metrics_batch = []
for epoch in range(100):
    metrics_batch.append({"loss": 0.5, "accuracy": 0.85})
    if epoch % 10 == 0:
        client.log_metrics_batch(metrics_batch)
        metrics_batch = []

# Bad: Log every iteration
for epoch in range(100):
    client.log_metrics({"loss": 0.5}, step=epoch)  # 100 API calls
```

**Async Operations**:
```python
import asyncio

async def log_metrics_async(client, metrics, step):
    await client.log_metrics_async(metrics, step)

# Use async for long-running operations
asyncio.run(log_metrics_async(client, metrics, 0))
```

---

## API Reference

### MLOpsClient

```python
class MLOpsClient:
    """Unified MLOps client."""

    def __init__(
        self,
        project_namespace: str,
        mlflow_tracking_uri: str | None = None,
        dvc_remote_name: str = "storage",
        workspace_root: str | Path | None = None
    ):
        """Initialize MLOps client."""

    def create_experiment(
        self,
        experiment_name: str,
        tags: dict[str, str] | None = None
    ) -> str:
        """Create MLFlow experiment."""

    def start_run(
        self,
        run_name: str,
        experiment_id: str | None = None,
        tags: dict[str, str] | None = None
    ):
        """Start tracking run (context manager)."""

    def log_params(self, params: dict[str, Any]):
        """Log parameters."""

    def log_metrics(self, metrics: dict[str, float], step: int | None = None):
        """Log metrics."""

    def log_artifact(self, local_path: str):
        """Log artifact."""

    def version_dataset(self, path: str, push_to_remote: bool = False) -> bool:
        """Version dataset with DVC."""

    def deploy_model(
        self,
        model_path: str,
        model_name: str,
        version: str
    ) -> dict:
        """Deploy model with BentoML."""

    def collect_apple_silicon_metrics(self) -> dict:
        """Collect Apple Silicon metrics."""

    def set_reference_data(self, data: pd.DataFrame):
        """Set reference data for monitoring."""
```

See individual component documentation for detailed API reference.

---

## Examples

See [examples/](../examples/) directory for complete examples:

1. **[Training Workflow](../examples/training_workflow.py)** - End-to-end training
2. **[Deployment Workflow](../examples/deployment_workflow.py)** - Model deployment
3. **[Monitoring Workflow](../examples/monitoring_workflow.py)** - Performance monitoring
4. **[Data Versioning](../examples/data_versioning.py)** - Dataset management
5. **[Hyperparameter Tuning](../examples/hyperparameter_tuning.py)** - Experiment tuning

---

## Additional Documentation

- **[MLFlow Setup](mlflow-setup.md)** - Detailed MLFlow configuration
- **[BentoML Usage](bentoml_usage.md)** - Model deployment guide
- **[Evidently Usage](evidently_usage.md)** - Monitoring setup
- **[Apple Silicon Implementation](apple_silicon_implementation.md)** - Hardware optimization
- **[Thermal-Aware Scheduling](thermal_aware_scheduling.md)** - Airflow integration
- **[MLOps Client Usage](mlops_client_usage.md)** - Client API guide
- **[Security Checklist](SECURITY_CHECKLIST.md)** - Security validation
- **[Operations Guide](OPERATIONS_GUIDE.md)** - Production operations
- **[Migration Guide](../integrations/p0_projects/MIGRATION_GUIDE.md)** - Project integration

---

## Support

### Getting Help

1. **Documentation**: Check this guide and component docs
2. **Examples**: Review example workflows
3. **Tests**: Examine test cases for usage patterns
4. **Issues**: Check existing issues in repository

### Contributing

Contributions welcome! Please:
1. Follow existing code style
2. Add tests for new features
3. Update documentation
4. Submit pull request

---

## License

Same as parent project (see root LICENSE file).

---

**End of Complete Guide**
