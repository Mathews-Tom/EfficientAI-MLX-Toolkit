# MLOps Unified Dashboard

Unified web dashboard for visualizing experiments, models, monitoring, and alerts across all EfficientAI-MLX-Toolkit projects with Apple Silicon metrics integration.

## Overview

The MLOps Dashboard provides a centralized interface for:

- **Project Overview**: Summary of all projects and their status
- **Experiment Tracking**: MLFlow experiments and runs by project
- **Model Registry**: Unified view of all registered models
- **Monitoring**: Evidently drift detection and performance monitoring
- **Alert Management**: Active alerts with severity tracking
- **Hardware Metrics**: Real-time Apple Silicon performance metrics

## Features

### ✅ Implemented Features

1. **Unified Data Aggregation**
   - Pulls data from MLFlow, Evidently, workspaces, and Apple Silicon metrics
   - Cross-project statistics and analytics
   - Real-time data updates

2. **Project Filtering**
   - Filter all views by project namespace
   - Project-specific detail pages
   - Quick navigation between projects

3. **Experiment Visualization**
   - View all experiments grouped by project
   - Recent runs with status and duration
   - Links to MLFlow tracking server

4. **Model Registry**
   - List all models with project tags
   - Model size and creation time
   - Direct path to model files

5. **Monitoring Dashboard**
   - Drift detection status
   - Performance monitoring status
   - Apple Silicon metrics availability
   - Alert configuration

6. **Alert Management**
   - View active alerts by severity
   - Alert details and metadata
   - Filter by project and severity

7. **Apple Silicon Metrics**
   - Chip type and variant
   - Memory usage and CPU utilization
   - Thermal state and power mode
   - Framework availability (MLX, MPS, ANE)

## Installation

### Dependencies

The dashboard requires these additional packages:

```bash
uv add fastapi uvicorn jinja2 python-multipart
```

All other dependencies (MLFlow, Evidently, etc.) are already part of the toolkit.

## Usage

### Starting the Dashboard

**Method 1: Using DashboardServer class**

```python
from mlops.dashboard import DashboardServer

# Create and run server
server = DashboardServer(
    repo_root="/path/to/repo",  # Optional, defaults to current directory
    host="0.0.0.0",             # Server host
    port=8000,                  # Server port
)

server.run()
```

**Method 2: Using FastAPI app directly**

```python
from mlops.dashboard import create_dashboard_app
import uvicorn

app = create_dashboard_app(repo_root="/path/to/repo")
uvicorn.run(app, host="0.0.0.0", port=8000)
```

**Method 3: CLI entry point**

```bash
# From mlops/dashboard directory
python server.py --host 0.0.0.0 --port 8000 --repo-root /path/to/repo

# With auto-reload for development
python server.py --reload
```

### Accessing the Dashboard

Once the server is running, access the dashboard at:

```
http://localhost:8000
```

### Dashboard Routes

- **`/`** - Home page (overview)
- **`/overview`** - Project overview with statistics
- **`/experiments`** - Experiment tracking page
- **`/experiments?project=NAME`** - Filtered by project
- **`/models`** - Model registry
- **`/models?project=NAME`** - Filtered by project
- **`/monitoring`** - Monitoring dashboard
- **`/monitoring?project=NAME`** - Filtered by project
- **`/alerts`** - Alert management
- **`/alerts?project=NAME`** - Filtered by project
- **`/hardware`** - Apple Silicon metrics
- **`/hardware?project=NAME`** - Filtered by project
- **`/project/{name}`** - Project detail page

### API Endpoints

All pages have corresponding API endpoints that return JSON:

- **`/api/data`** - All dashboard data
- **`/api/experiments`** - Experiments data
- **`/api/experiments?project=NAME`** - Filtered experiments
- **`/api/models`** - Models data
- **`/api/monitoring`** - Monitoring status
- **`/api/alerts`** - Alerts data
- **`/api/hardware`** - Apple Silicon metrics
- **`/api/project/{name}`** - Project-specific data
- **`/health`** - Health check

## Architecture

### Components

1. **DashboardDataAggregator** (`data_aggregator.py`)
   - Aggregates data from all MLOps components
   - Provides unified data access layer
   - Handles cross-project statistics

2. **DashboardServer** (`server.py`)
   - FastAPI application with routes
   - HTML template rendering
   - API endpoint handlers

3. **Templates** (`templates/`)
   - Base template with navigation
   - Page-specific templates
   - Inline CSS styling

### Data Flow

```
Workspaces → WorkspaceManager
                    ↓
MLFlow Experiments → search_experiments()
                    ↓
Models → models_path scanning
                    ↓
Alerts → AlertManager
                    ↓
Monitoring → EvidentlyMonitor
                    ↓
Apple Silicon → AppleSiliconMetricsCollector
                    ↓
         DashboardDataAggregator
                    ↓
              FastAPI Routes
                    ↓
            HTML Templates
                    ↓
               Browser
```

## Configuration

### Workspace Configuration

The dashboard automatically discovers projects through the WorkspaceManager:

```python
from mlops.workspace.manager import WorkspaceManager

manager = WorkspaceManager(repo_root="/path/to/repo")
workspace = manager.get_or_create_workspace(
    project_name="lora-finetuning-mlx",
    mlflow_tracking_uri="file:///tmp/mlflow",
)
```

### MLFlow Configuration

Experiments are tracked using MLFlow:

```python
from mlops.client import MLOpsClient

client = MLOpsClient.from_project("lora-finetuning-mlx")

with client.start_run(run_name="experiment-001"):
    client.log_params({"lr": 0.0001, "epochs": 10})
    client.log_metrics({"loss": 0.42, "accuracy": 0.95})
```

### Monitoring Configuration

Enable monitoring for projects:

```python
from mlops.monitoring.evidently.monitor import create_monitor

monitor = create_monitor(
    project_name="lora-finetuning-mlx",
    enable_drift_detection=True,
    enable_performance_monitoring=True,
    enable_apple_silicon_metrics=True,
    enable_alerts=True,
)

monitor.set_reference_data(train_data, "target", "prediction")
results = monitor.monitor(test_data, "target", "prediction")
```

## Deployment

### Development

```bash
# Run with auto-reload
python server.py --reload --host 127.0.0.1 --port 8000
```

### Production

```bash
# Run with Gunicorn (more workers)
gunicorn mlops.dashboard.server:create_dashboard_app --workers 4 --bind 0.0.0.0:8000
```

### Docker

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY . /app

RUN pip install -e .
RUN pip install fastapi uvicorn jinja2

EXPOSE 8000

CMD ["python", "-m", "mlops.dashboard.server", "--host", "0.0.0.0", "--port", "8000"]
```

## Troubleshooting

### Issue: No projects showing

**Solution**: Ensure workspaces are created:

```python
from mlops.workspace.manager import WorkspaceManager

manager = WorkspaceManager()
workspace = manager.create_workspace("my-project")
```

### Issue: No experiments showing

**Solution**: Ensure MLFlow tracking is configured:

```python
from mlops.client import MLOpsClient

client = MLOpsClient.from_project("my-project")
with client.start_run():
    client.log_metrics({"loss": 0.5})
```

### Issue: No Apple Silicon metrics

**Solution**: This is expected on non-Apple Silicon hardware. The dashboard will show "Not Available".

### Issue: Templates not found

**Solution**: Ensure templates directory exists at `mlops/dashboard/templates/`:

```bash
ls mlops/dashboard/templates/
# Should show: base.html, overview.html, experiments.html, etc.
```

### Issue: API returns 500 errors

**Solution**: Check logs for detailed error messages:

```bash
# Run with debug logging
python server.py --log-level debug
```

## Examples

### Example 1: Basic Dashboard Usage

```python
from mlops.dashboard import DashboardServer

# Start server
server = DashboardServer()
server.run()

# Access at http://localhost:8000
```

### Example 2: Custom Port and Host

```python
from mlops.dashboard import DashboardServer

server = DashboardServer(
    repo_root="/custom/path",
    host="0.0.0.0",
    port=9000,
)
server.run()
```

### Example 3: Using Data Aggregator Directly

```python
from mlops.dashboard import DashboardDataAggregator

aggregator = DashboardDataAggregator(repo_root="/path/to/repo")

# Get all data
data = aggregator.get_all_data()
print(f"Total projects: {data.cross_project_stats['total_projects']}")
print(f"Total experiments: {data.cross_project_stats['total_experiments']}")

# Get project-specific data
project_data = aggregator.get_project_data("lora-finetuning-mlx")
print(f"Experiments: {len(project_data['experiments'])}")
print(f"Models: {len(project_data['models'])}")
```

### Example 4: API Integration

```python
import requests

# Get all data as JSON
response = requests.get("http://localhost:8000/api/data")
data = response.json()

# Get experiments for specific project
response = requests.get("http://localhost:8000/api/experiments?project=lora-finetuning-mlx")
experiments = response.json()

# Get active alerts
response = requests.get("http://localhost:8000/api/alerts")
alerts = response.json()
```

## Testing

Run tests with pytest:

```bash
# Run all dashboard tests
uv run pytest mlops/tests/test_dashboard.py -v

# Run with coverage
uv run pytest mlops/tests/test_dashboard.py --cov=mlops.dashboard --cov-report=term-missing

# Run specific test
uv run pytest mlops/tests/test_dashboard.py::TestDashboardServer::test_health_endpoint -v
```

## Performance Considerations

1. **Data Caching**: The dashboard fetches fresh data on each request. For large deployments, consider implementing caching.

2. **Pagination**: Currently shows all data. For many experiments/models, implement pagination.

3. **Background Updates**: Consider WebSocket for real-time updates instead of polling.

4. **Database**: For production, consider storing aggregated data in a database instead of scanning filesystems.

## Future Enhancements

- [ ] Real-time updates via WebSockets
- [ ] User authentication and authorization
- [ ] Custom dashboard layouts
- [ ] Export reports as PDF
- [ ] Scheduled monitoring reports
- [ ] Integration with Slack/Email for alerts
- [ ] Model comparison views
- [ ] Experiment comparison views
- [ ] Custom metrics visualization
- [ ] Data caching for performance

## Related Documentation

- [MLOps Client](../client/README.md)
- [Workspace Manager](../workspace/README.md)
- [Evidently Monitoring](../monitoring/evidently/README.md)
- [Apple Silicon Metrics](../silicon/README.md)

## Support

For issues or questions:
1. Check logs for detailed error messages
2. Verify all dependencies are installed
3. Ensure workspaces and experiments are configured
4. Review the test suite for usage examples
