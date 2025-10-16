# MLFlow Tracking Infrastructure Setup Guide

This guide provides instructions for setting up and using the MLFlow tracking infrastructure for the EfficientAI-MLX-Toolkit.

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Docker Compose Setup](#docker-compose-setup)
4. [Configuration](#configuration)
5. [Usage Examples](#usage-examples)
6. [Apple Silicon Metrics](#apple-silicon-metrics)
7. [Troubleshooting](#troubleshooting)
8. [Production Deployment](#production-deployment)

## Overview

The MLFlow tracking infrastructure provides:

- **Experiment Tracking**: Log parameters, metrics, and artifacts
- **Model Registry**: Version and manage trained models
- **Apple Silicon Integration**: Collect and log Apple Silicon-specific metrics
- **Centralized Storage**: PostgreSQL backend with S3-compatible artifact storage

### Architecture

```
┌─────────────────────────────────────────────────────┐
│ EfficientAI-MLX-Toolkit Projects                    │
│ (LoRA, Compression, Diffusion, etc.)                │
└────────────────┬────────────────────────────────────┘
                 │
                 │ MLFlow Client API
                 ▼
┌─────────────────────────────────────────────────────┐
│ MLFlow Tracking Server (Port 5000)                  │
│  - REST API for experiment tracking                 │
│  - Model registry management                        │
│  - Apple Silicon metrics collection                 │
└─────────┬──────────────────────┬────────────────────┘
          │                      │
          ▼                      ▼
┌──────────────────┐   ┌──────────────────────┐
│ PostgreSQL DB    │   │ MinIO S3 Storage     │
│ (Metadata)       │   │ (Artifacts & Models) │
└──────────────────┘   └──────────────────────┘
```

## Quick Start

### Prerequisites

- Docker and Docker Compose installed
- Python 3.11+ with uv package manager
- EfficientAI-MLX-Toolkit installed

### 1. Start MLFlow Services

```bash
# Navigate to MLOps docker directory
cd mlops/docker

# Start all services
docker-compose -f mlflow-compose.yml up -d

# Check services status
docker-compose -f mlflow-compose.yml ps

# View logs
docker-compose -f mlflow-compose.yml logs -f mlflow-server
```

### 2. Verify Installation

```bash
# Check MLFlow server health
curl http://localhost:5000/health

# Open MLFlow UI in browser
open http://localhost:5000
```

### 3. Basic Usage

```python
from mlops.client import create_client

# Create MLFlow client
client = create_client()

# Track an experiment
with client.run(run_name="my-experiment"):
    # Log parameters
    client.log_params({
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 10
    })

    # Log metrics
    for epoch in range(10):
        loss = train_epoch()
        client.log_metric("loss", loss, step=epoch)

    # Log model
    client.log_model(model, "model")
```

## Docker Compose Setup

### Services Overview

The Docker Compose stack includes:

1. **mlflow-db** (PostgreSQL 15)
   - Stores experiment metadata, parameters, and metrics
   - Port: 5432
   - Credentials: mlflow/mlflow_password

2. **mlflow-minio** (MinIO S3)
   - S3-compatible object storage for artifacts
   - API Port: 9000
   - Console Port: 9001
   - Credentials: minioadmin/minioadmin

3. **mlflow-server** (MLFlow Tracking Server)
   - Main tracking server
   - Port: 5000
   - Connects to PostgreSQL and MinIO

### Service Management

```bash
# Start services
docker-compose -f mlflow-compose.yml up -d

# Stop services
docker-compose -f mlflow-compose.yml down

# Stop and remove volumes (WARNING: deletes all data)
docker-compose -f mlflow-compose.yml down -v

# Restart specific service
docker-compose -f mlflow-compose.yml restart mlflow-server

# View service logs
docker-compose -f mlflow-compose.yml logs -f [service-name]

# Scale services (if needed)
docker-compose -f mlflow-compose.yml up -d --scale mlflow-server=2
```

### Accessing Services

- **MLFlow UI**: http://localhost:5000
- **MinIO Console**: http://localhost:9001 (minioadmin/minioadmin)
- **PostgreSQL**: localhost:5432 (mlflow/mlflow_password)

## Configuration

### Environment-Based Configuration

The MLFlow client supports three environments:

#### Development (Default)
```python
from mlops.config import MLFlowConfig

config = MLFlowConfig.from_environment("development")
# Uses: http://localhost:5000, SQLite backend
```

#### Testing
```python
config = MLFlowConfig.from_environment("testing")
# Uses: file:///tmp/mlflow-test, in-memory backend
```

#### Production
```python
config = MLFlowConfig.from_environment("production")
# Requires proper tracking URI and backend configuration
```

### Configuration File

Create `configs/mlflow.yaml`:

```yaml
mlflow:
  tracking_uri: http://localhost:5000
  experiment_name: my-project
  artifact_location: ./mlruns
  environment: development
  enable_system_metrics: true
  enable_apple_silicon_metrics: true
  log_models: true
  log_artifacts: true
  tags:
    project: efficientai-mlx-toolkit
    version: "1.0"
```

Load configuration:

```python
from pathlib import Path
from mlops.config import load_config_from_file

config = load_config_from_file(Path("configs/mlflow.yaml"))
client = MLFlowClient(config=config)
```

### Environment Variables

Override configuration with environment variables:

```bash
export MLFLOW_TRACKING_URI=http://mlflow-server:5000
export MLFLOW_EXPERIMENT_NAME=production-experiment
export MLFLOW_ENABLE_APPLE_SILICON_METRICS=true
export MLFLOW_ENVIRONMENT=production
```

## Usage Examples

### Example 1: Basic Experiment Tracking

```python
from mlops.client import create_client

client = create_client()

with client.run(run_name="basic-training"):
    # Log hyperparameters
    client.log_params({
        "model": "transformer",
        "optimizer": "adam",
        "lr": 0.001
    })

    # Training loop
    for epoch in range(10):
        train_loss = train_model()
        val_loss = validate_model()

        client.log_metrics({
            "train_loss": train_loss,
            "val_loss": val_loss
        }, step=epoch)
```

### Example 2: Model Versioning

```python
from mlops.client import create_client
import torch

client = create_client()

with client.run(run_name="model-v1"):
    # Train model
    model = train_model()

    # Log model with registry
    client.log_model(
        model,
        artifact_path="model",
        registered_model_name="my-transformer"
    )

    # Log model metadata as tags
    client.set_tags({
        "model_version": "1.0",
        "architecture": "transformer",
        "parameters": "110M"
    })
```

### Example 3: Artifact Management

```python
from mlops.client import create_client
from pathlib import Path

client = create_client()

with client.run(run_name="artifact-demo"):
    # Train and save model
    model = train_model()
    model_path = Path("checkpoints/model.pth")
    torch.save(model.state_dict(), model_path)

    # Log single artifact
    client.log_artifact(model_path, artifact_path="models")

    # Log entire directory
    client.log_artifacts("checkpoints", artifact_path="checkpoints")

    # Log configuration file
    client.log_artifact("config.yaml", artifact_path="configs")
```

### Example 4: Run Comparison

```python
from mlops.client import create_client

client = create_client()

# Search for best runs
runs = client.search_runs(
    filter_string="metrics.val_loss < 0.5",
    order_by=["metrics.val_loss ASC"],
    max_results=10
)

# Analyze results
for run in runs:
    print(f"Run: {run['run_id']}")
    print(f"Loss: {run['metrics.val_loss']}")
    print(f"Params: {run['params']}")
```

## Apple Silicon Metrics

### Automatic Collection

```python
from mlops.client import create_client
from mlops.tracking import log_metrics_to_mlflow

client = create_client()

with client.run(run_name="apple-silicon-test"):
    # Automatically collect and log all Apple Silicon metrics
    log_metrics_to_mlflow(client)

    # Metrics logged:
    # - chip_type (M1/M2/M3 variant)
    # - memory_total_gb, memory_used_gb, memory_available_gb
    # - memory_utilization_percent
    # - mps_available, mps_utilization_percent
    # - ane_available
    # - thermal_state (0-3)
    # - power_mode (low_power, normal, high_performance)
    # - core_count, performance_core_count, efficiency_core_count
```

### Manual Collection

```python
from mlops.tracking import collect_metrics

# Collect metrics
metrics = collect_metrics()

print(f"Chip: {metrics.chip_type}")
print(f"Memory Used: {metrics.memory_used_gb} GB")
print(f"MPS Available: {metrics.mps_available}")

# Convert to MLFlow format
mlflow_metrics = metrics.to_mlflow_metrics()

# Log manually
client.log_metrics(mlflow_metrics)
```

### Custom Metrics

```python
client.log_apple_silicon_metrics({
    "custom_memory_metric": memory_usage,
    "mps_efficiency": mps_utilization,
    "thermal_throttling": throttle_count
})
```

## Troubleshooting

### Services Not Starting

```bash
# Check service logs
docker-compose -f mlflow-compose.yml logs mlflow-server

# Check all service status
docker-compose -f mlflow-compose.yml ps

# Restart services
docker-compose -f mlflow-compose.yml restart
```

### Connection Refused

```bash
# Verify MLFlow server is running
curl http://localhost:5000/health

# Check network connectivity
docker network inspect mlflow_mlflow-network

# Check firewall settings
```

### Database Connection Issues

```bash
# Check PostgreSQL logs
docker-compose -f mlflow-compose.yml logs mlflow-db

# Test database connection
docker exec -it mlflow-postgres psql -U mlflow -d mlflow

# Reset database (WARNING: deletes all data)
docker-compose -f mlflow-compose.yml down -v
docker-compose -f mlflow-compose.yml up -d
```

### Artifact Storage Issues

```bash
# Check MinIO logs
docker-compose -f mlflow-compose.yml logs mlflow-minio

# Access MinIO console
open http://localhost:9001

# Verify bucket exists
docker exec -it mlflow-minio mc ls minio/mlflow-artifacts
```

### Python Client Issues

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Test connection
from mlops.client import create_client
client = create_client()
print(client.get_experiment_info())
```

## Production Deployment

### Security Considerations

1. **Change Default Credentials**
   ```yaml
   # Update in mlflow-compose.yml
   POSTGRES_PASSWORD: <strong-password>
   MINIO_ROOT_PASSWORD: <strong-password>
   ```

2. **Use HTTPS**
   - Add reverse proxy (nginx/traefik)
   - Configure SSL certificates
   - Update tracking URI to https://

3. **Network Security**
   - Use private networks for database
   - Restrict port access
   - Enable firewall rules

4. **Authentication**
   - Configure MLFlow authentication
   - Use API tokens
   - Implement RBAC

### Scaling

1. **Horizontal Scaling**
   ```bash
   # Scale MLFlow servers
   docker-compose -f mlflow-compose.yml up -d --scale mlflow-server=3

   # Add load balancer
   # Configure nginx/traefik for load balancing
   ```

2. **Database Scaling**
   - Use PostgreSQL replication
   - Configure read replicas
   - Implement connection pooling

3. **Storage Scaling**
   - Use S3/GCS/Azure Blob for production
   - Configure CDN for artifact delivery
   - Implement lifecycle policies

### Backup and Recovery

```bash
# Backup PostgreSQL
docker exec mlflow-postgres pg_dump -U mlflow mlflow > mlflow-backup.sql

# Restore PostgreSQL
docker exec -i mlflow-postgres psql -U mlflow mlflow < mlflow-backup.sql

# Backup MinIO
mc mirror minio/mlflow-artifacts /backup/mlflow-artifacts

# Restore MinIO
mc mirror /backup/mlflow-artifacts minio/mlflow-artifacts
```

### Monitoring

```bash
# Monitor service health
docker-compose -f mlflow-compose.yml ps

# Check resource usage
docker stats

# Monitor logs
docker-compose -f mlflow-compose.yml logs -f --tail=100

# Set up alerts
# Configure Prometheus + Grafana for production monitoring
```

## Additional Resources

- [MLFlow Documentation](https://mlflow.org/docs/latest/index.html)
- [MinIO Documentation](https://min.io/docs/minio/linux/index.html)
- [PostgreSQL Documentation](https://www.postgresql.org/docs/)
- [EfficientAI-MLX-Toolkit GitHub](https://github.com/your-org/efficientai-mlx-toolkit)

## Support

For issues and questions:

1. Check the troubleshooting section
2. Review service logs
3. Open an issue on GitHub
4. Contact the team

---

**Last Updated**: 2025-10-17
**Version**: 1.0.0
