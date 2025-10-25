# MLOps Dashboard - Quick Start Guide

## 60-Second Start

```bash
# Start dashboard (default: http://0.0.0.0:8000)
uv run efficientai-toolkit dashboard
```

Open browser: http://localhost:8000

## Common Commands

```bash
# Custom port
uv run efficientai-toolkit dashboard --port 9000

# Development mode with auto-reload
uv run efficientai-toolkit dashboard --reload

# Custom repository location
uv run efficientai-toolkit dashboard --repo-root /path/to/repo

# Help
uv run efficientai-toolkit dashboard --help
```

## Quick Test

```bash
# Test dashboard functionality
uv run pytest mlops/tests/test_dashboard.py -v

# Check health endpoint
curl http://localhost:8000/health
```

## Available Pages

- **Overview**: http://localhost:8000/overview
- **Experiments**: http://localhost:8000/experiments
- **Models**: http://localhost:8000/models
- **Monitoring**: http://localhost:8000/monitoring
- **Alerts**: http://localhost:8000/alerts
- **Hardware**: http://localhost:8000/hardware

## API Endpoints

```bash
# All data
curl http://localhost:8000/api/data

# Experiments
curl http://localhost:8000/api/experiments

# Models
curl http://localhost:8000/api/models

# Monitoring
curl http://localhost:8000/api/monitoring

# Alerts
curl http://localhost:8000/api/alerts

# Hardware metrics
curl http://localhost:8000/api/hardware
```

## Filter by Project

Add `?project=PROJECT_NAME` to any URL:

```bash
# Experiments for specific project
curl "http://localhost:8000/api/experiments?project=lora-finetuning-mlx"

# Models for specific project
curl "http://localhost:8000/api/models?project=lora-finetuning-mlx"
```

## Python Usage

```python
from mlops.dashboard import DashboardServer

# Start server
server = DashboardServer(
    host="0.0.0.0",
    port=8000,
)
server.run()
```

## Data Access

```python
from mlops.dashboard import DashboardDataAggregator

# Get all data
aggregator = DashboardDataAggregator()
data = aggregator.get_all_data()

# Access specific components
print(f"Projects: {len(data.workspaces)}")
print(f"Experiments: {len(data.experiments)}")
print(f"Models: {len(data.models)}")
print(f"Alerts: {len(data.alerts)}")
```

## Troubleshooting

### Port in use
```bash
# Use different port
uv run efficientai-toolkit dashboard --port 8001
```

### No data showing
```python
# Create workspace first
from mlops.workspace.manager import WorkspaceManager
manager = WorkspaceManager()
manager.create_workspace("my-project")
```

### Import errors
```bash
# Install dependencies
uv add fastapi uvicorn jinja2 python-multipart
```

## Production Deployment

```bash
# With gunicorn (4 workers)
gunicorn mlops.dashboard.server:create_dashboard_app \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000
```

## Docker

```bash
# Build
docker build -t efficientai-dashboard .

# Run
docker run -d -p 8000:8000 efficientai-dashboard
```

## Documentation

- **README.md**: Comprehensive guide (432 lines)
- **DEPLOYMENT.md**: Production deployment (498 lines)
- **IMPLEMENTATION_SUMMARY.md**: Technical details

## Support

For issues:
1. Check logs for detailed errors
2. Review test suite: `mlops/tests/test_dashboard.py`
3. Consult full documentation in README.md
