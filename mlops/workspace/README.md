# Project Workspace Management

The workspace management system provides isolated workspaces for organizing MLOps data across all toolkit projects.

## Overview

Each project gets a dedicated workspace with isolated directories for:

- **MLFlow experiments** - Experiment tracking and artifacts
- **DVC data** - Data versioning and remote storage
- **BentoML models** - Model registry and deployment
- **Evidently monitoring** - Performance and drift monitoring
- **Project outputs** - Training checkpoints and results

## Quick Start

### Using WorkspaceManager

```python
from mlops.workspace import WorkspaceManager

# Initialize manager
mgr = WorkspaceManager()

# Create workspace for a project
workspace = mgr.create_workspace("lora-finetuning-mlx")

# Access workspace paths
print(workspace.mlflow_path)        # MLFlow artifacts directory
print(workspace.dvc_path)           # DVC data directory
print(workspace.monitoring_path)    # Monitoring data
print(workspace.models_path)        # Models registry
print(workspace.outputs_path)       # Project outputs

# List all workspaces
workspaces = mgr.list_workspaces()
for ws in workspaces:
    print(f"{ws.project_name}: {ws.mlflow_experiment_id}")

# Get workspace status
status = mgr.get_workspace_status("lora-finetuning-mlx")
print(status["directory_stats"])
```

### Integration with MLOpsClient

The `MLOpsClient` automatically uses `WorkspaceManager` for workspace management:

```python
from mlops.client import MLOpsClient

# Create client (automatically creates workspace)
client = MLOpsClient.from_project("lora-finetuning-mlx")

# Workspace is available
print(client.workspace.mlflow_path)
print(client.workspace.mlflow_experiment_id)

# Get comprehensive status (includes workspace info)
status = client.get_status()
print(status["workspace"])
```

## Key Features

### 1. Automatic Configuration

Workspaces are automatically configured with appropriate defaults:

```python
workspace = mgr.create_workspace(
    project_name="lora-finetuning-mlx",
    mlflow_experiment_id="exp-123",
    mlflow_tracking_uri="http://localhost:5000",
    dvc_remote_path="/data/dvc-remote",
    bentoml_tag_prefix="lora-finetuning-mlx:",
    metadata={"description": "LoRA fine-tuning project"},
)
```

### 2. Project Isolation

Each project has completely isolated directories:

```python
ws1 = mgr.create_workspace("project-1")
ws2 = mgr.create_workspace("project-2")

# Write to project-1
(ws1.mlflow_path / "exp1.txt").write_text("data")

# project-2 does not see the file
assert not (ws2.mlflow_path / "exp1.txt").exists()
```

### 3. Metadata Persistence

Workspace metadata is automatically saved to YAML:

```python
# Create workspace
workspace = mgr.create_workspace("test-project")

# Load workspace in new manager instance
mgr2 = WorkspaceManager()
loaded_workspace = mgr2.get_workspace("test-project")

# Metadata is preserved
assert loaded_workspace.mlflow_experiment_id == workspace.mlflow_experiment_id
```

### 4. Status and Analytics

Get comprehensive status for all workspaces:

```python
# Single workspace status
status = mgr.get_workspace_status("lora-finetuning-mlx")
print(f"MLFlow files: {status['directory_stats']['mlflow_files']}")
print(f"Models: {status['directory_stats']['models_files']}")

# All workspaces status
all_status = mgr.get_all_workspaces_status()
for status in all_status:
    print(f"{status['project_name']}: {status['mlflow_experiment_id']}")
```

### 5. Workspace Lifecycle

```python
# Create
workspace = mgr.create_workspace("test-project")

# Update metadata
mgr.update_workspace_metadata(
    "test-project",
    mlflow_experiment_id="exp-new",
    metadata={"stage": "production"},
)

# List
workspaces = mgr.list_workspaces()

# Get or create (safe)
workspace = mgr.get_or_create_workspace("test-project")

# Delete (requires force flag for safety)
mgr.delete_workspace("test-project", force=True)
```

## Directory Structure

Each workspace has the following structure:

```
mlops/workspace/<project-name>/
├── workspace.yaml           # Workspace metadata
├── mlflow/                  # MLFlow artifacts
├── dvc/                     # DVC data
├── monitoring/              # Evidently monitoring data
├── models/                  # Model registry
└── outputs/                 # Project outputs
```

## Integration Points

### With MLFlow

```python
client = MLOpsClient.from_project("lora-finetuning-mlx")

with client.start_run(run_name="experiment-001"):
    # Experiments stored in workspace.mlflow_path
    client.log_params({"lr": 0.001})
    client.log_metrics({"loss": 0.5})
    client.log_artifact("model.pth")
```

### With DVC

```python
client = MLOpsClient.from_project("lora-finetuning-mlx")

# Data tracked in workspace.dvc_path
result = client.dvc_add("datasets/train.csv")
client.dvc_push()
```

### With BentoML

```python
client = MLOpsClient.from_project("lora-finetuning-mlx")

# Models deployed with workspace-specific tags
result = client.deploy_model(
    model_path="outputs/model",
    model_name="lora_adapter",
    model_version="v1.0",
)
# Tag: lora-finetuning-mlx:lora_adapter:v1.0
```

### With Evidently

```python
client = MLOpsClient.from_project("lora-finetuning-mlx")

# Monitoring data in workspace.monitoring_path
client.set_reference_data(reference_df)
results = client.monitor_predictions(current_df)
```

## Multi-Project Workflows

### Dashboard-Style Analytics

```python
mgr = WorkspaceManager()

# Get all projects
workspaces = mgr.list_workspaces()

# Aggregate metrics across projects
for workspace in workspaces:
    status = mgr.get_workspace_status(workspace.project_name)
    print(f"{workspace.project_name}:")
    print(f"  Experiments: {status['mlflow_experiment_id']}")
    print(f"  Models: {status['directory_stats']['models_files']}")
    print(f"  Data files: {status['directory_stats']['dvc_files']}")
```

### Project Comparison

```python
# Compare workspace sizes
all_status = mgr.get_all_workspaces_status()
for status in all_status:
    total_files = sum(
        status["directory_stats"][f"{path}_files"]
        for path in ["mlflow", "dvc", "monitoring", "models", "outputs"]
        if f"{path}_files" in status["directory_stats"]
    )
    print(f"{status['project_name']}: {total_files} files")
```

## Error Handling

```python
from mlops.workspace import WorkspaceError

try:
    workspace = mgr.get_workspace("nonexistent")
except WorkspaceError as e:
    print(f"Error: {e}")
    print(f"Operation: {e.operation}")
    print(f"Workspace: {e.workspace}")
    print(f"Details: {e.details}")
```

## Best Practices

1. **Use from_project for clients**: Let MLOpsClient handle workspace management automatically
2. **Use get_or_create_workspace**: Safe pattern for workspace access
3. **Store metadata**: Use workspace metadata for project-specific configuration
4. **Check status**: Use status methods to monitor workspace health
5. **Isolate projects**: Never share workspace directories between projects

## Advanced Usage

### Custom Base Path

```python
# Use custom base directory
mgr = WorkspaceManager(base_path="/data/mlops/workspaces")
```

### Workspace Templates

```python
def create_workspace_with_defaults(project_name: str) -> ProjectWorkspace:
    """Create workspace with standard configuration"""
    return mgr.create_workspace(
        project_name=project_name,
        mlflow_tracking_uri="http://mlflow-server:5000",
        dvc_remote_path=f"s3://dvc-storage/{project_name}",
        metadata={
            "created_by": "automation",
            "environment": "production",
        },
    )
```

### Workspace Migration

```python
# Export workspace metadata
workspace = mgr.get_workspace("old-project")
metadata = workspace.to_dict()

# Create new workspace with same config
new_workspace = mgr.create_workspace(
    project_name="new-project",
    mlflow_experiment_id=metadata["mlflow_experiment_id"],
    mlflow_tracking_uri=metadata["mlflow_tracking_uri"],
    metadata=metadata["metadata"],
)
```

## API Reference

See the module docstrings for complete API documentation:

- `ProjectWorkspace`: Dataclass representing a project workspace
- `WorkspaceManager`: Manager for creating and managing workspaces
- `WorkspaceError`: Exception raised for workspace operation failures
