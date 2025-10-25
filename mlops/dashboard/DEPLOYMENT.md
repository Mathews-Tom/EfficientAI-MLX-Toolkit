# MLOps Dashboard Deployment Guide

Comprehensive guide for deploying the unified MLOps dashboard in various environments.

## Quick Start

### Local Development

```bash
# Start dashboard on default port (8000)
uv run efficientai-toolkit dashboard

# Custom host and port
uv run efficientai-toolkit dashboard --host 127.0.0.1 --port 9000

# Enable auto-reload for development
uv run efficientai-toolkit dashboard --reload

# Custom repository root
uv run efficientai-toolkit dashboard --repo-root /path/to/repo
```

Access at: http://localhost:8000

### Programmatic Usage

```python
from mlops.dashboard import DashboardServer

# Create and run server
server = DashboardServer(
    repo_root="/path/to/repo",
    host="0.0.0.0",
    port=8000,
)

server.run()
```

## Production Deployment

### Option 1: Direct Uvicorn

```bash
# Production server with multiple workers
uvicorn mlops.dashboard.server:create_dashboard_app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 4 \
    --log-level info
```

### Option 2: Gunicorn with Uvicorn Workers

```bash
# Install gunicorn
uv add gunicorn

# Run with gunicorn
gunicorn mlops.dashboard.server:create_dashboard_app \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000 \
    --access-logfile - \
    --error-logfile -
```

### Option 3: systemd Service

Create `/etc/systemd/system/mlops-dashboard.service`:

```ini
[Unit]
Description=EfficientAI MLOps Dashboard
After=network.target

[Service]
Type=simple
User=mlops
WorkingDirectory=/opt/efficientai
Environment="PATH=/opt/efficientai/.venv/bin"
ExecStart=/opt/efficientai/.venv/bin/uvicorn mlops.dashboard.server:create_dashboard_app --host 0.0.0.0 --port 8000 --workers 4
Restart=on-failure
RestartSec=5s

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl enable mlops-dashboard
sudo systemctl start mlops-dashboard
sudo systemctl status mlops-dashboard
```

## Docker Deployment

### Dockerfile

Create `Dockerfile` in repository root:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -e .
RUN pip install --no-cache-dir fastapi uvicorn jinja2 python-multipart

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run dashboard
CMD ["python", "-m", "uvicorn", "mlops.dashboard.server:create_dashboard_app", "--host", "0.0.0.0", "--port", "8000"]
```

### Build and Run

```bash
# Build image
docker build -t efficientai-dashboard:latest .

# Run container
docker run -d \
    --name mlops-dashboard \
    -p 8000:8000 \
    -v $(pwd):/app \
    efficientai-dashboard:latest

# View logs
docker logs -f mlops-dashboard

# Stop container
docker stop mlops-dashboard
```

### Docker Compose

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  dashboard:
    build: .
    image: efficientai-dashboard:latest
    container_name: mlops-dashboard
    ports:
      - "8000:8000"
    volumes:
      - .:/app
      - ./mlflow_data:/app/mlflow_data
      - ./workspace_data:/app/workspace_data
    environment:
      - MLFLOW_TRACKING_URI=file:///app/mlflow_data
      - PYTHONUNBUFFERED=1
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 5s

  mlflow:
    image: ghcr.io/mlflow/mlflow:v2.8.1
    container_name: mlflow-server
    ports:
      - "5000:5000"
    volumes:
      - ./mlflow_data:/mlflow
    command: >
      mlflow server
      --backend-store-uri sqlite:///mlflow/mlflow.db
      --default-artifact-root /mlflow/artifacts
      --host 0.0.0.0
      --port 5000
    restart: unless-stopped
```

Run with Docker Compose:

```bash
docker-compose up -d
docker-compose logs -f
docker-compose down
```

## Kubernetes Deployment

### Deployment YAML

Create `k8s/dashboard-deployment.yaml`:

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: mlops

---
apiVersion: v1
kind: ConfigMap
metadata:
  name: dashboard-config
  namespace: mlops
data:
  MLFLOW_TRACKING_URI: "http://mlflow-service:5000"

---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mlops-dashboard
  namespace: mlops
spec:
  replicas: 3
  selector:
    matchLabels:
      app: mlops-dashboard
  template:
    metadata:
      labels:
        app: mlops-dashboard
    spec:
      containers:
      - name: dashboard
        image: efficientai-dashboard:latest
        ports:
        - containerPort: 8000
        envFrom:
        - configMapRef:
            name: dashboard-config
        resources:
          requests:
            cpu: "500m"
            memory: "512Mi"
          limits:
            cpu: "1000m"
            memory: "1Gi"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 10

---
apiVersion: v1
kind: Service
metadata:
  name: dashboard-service
  namespace: mlops
spec:
  selector:
    app: mlops-dashboard
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer

---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: dashboard-ingress
  namespace: mlops
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
  - hosts:
    - dashboard.example.com
    secretName: dashboard-tls
  rules:
  - host: dashboard.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: dashboard-service
            port:
              number: 80
```

Deploy to Kubernetes:

```bash
kubectl apply -f k8s/dashboard-deployment.yaml
kubectl get pods -n mlops
kubectl logs -f deployment/mlops-dashboard -n mlops
```

## Reverse Proxy Setup

### Nginx

Create `/etc/nginx/sites-available/mlops-dashboard`:

```nginx
server {
    listen 80;
    server_name dashboard.example.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # WebSocket support (if needed in future)
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }

    # Static files (if any)
    location /static {
        alias /opt/efficientai/mlops/dashboard/static;
        expires 30d;
        add_header Cache-Control "public, immutable";
    }
}
```

Enable and reload:

```bash
sudo ln -s /etc/nginx/sites-available/mlops-dashboard /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

### Caddy

Create `Caddyfile`:

```
dashboard.example.com {
    reverse_proxy localhost:8000

    # Automatic HTTPS
    tls your-email@example.com

    # Access logs
    log {
        output file /var/log/caddy/dashboard-access.log
    }
}
```

Run Caddy:

```bash
caddy run --config Caddyfile
```

## SSL/TLS Configuration

### Let's Encrypt with Certbot

```bash
# Install certbot
sudo apt-get install certbot python3-certbot-nginx

# Obtain certificate
sudo certbot --nginx -d dashboard.example.com

# Auto-renewal is configured by default
sudo certbot renew --dry-run
```

### Self-Signed Certificate (Development)

```bash
# Generate certificate
openssl req -x509 -newkey rsa:4096 -nodes \
    -keyout key.pem -out cert.pem -days 365 \
    -subj "/CN=localhost"

# Run dashboard with HTTPS
uvicorn mlops.dashboard.server:create_dashboard_app \
    --host 0.0.0.0 \
    --port 8443 \
    --ssl-keyfile key.pem \
    --ssl-certfile cert.pem
```

## Environment Variables

Configure dashboard behavior with environment variables:

```bash
# MLFlow tracking URI
export MLFLOW_TRACKING_URI=http://mlflow-server:5000

# Workspace root
export WORKSPACE_ROOT=/data/mlops

# Logging level
export LOG_LEVEL=INFO

# Dashboard host and port
export DASHBOARD_HOST=0.0.0.0
export DASHBOARD_PORT=8000
```

## Performance Tuning

### Worker Configuration

```bash
# Calculate optimal workers: (2 x $num_cores) + 1
WORKERS=$((2 * $(nproc) + 1))

uvicorn mlops.dashboard.server:create_dashboard_app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers $WORKERS
```

### Resource Limits

For Docker:

```bash
docker run -d \
    --name mlops-dashboard \
    --cpus="2.0" \
    --memory="2g" \
    -p 8000:8000 \
    efficientai-dashboard:latest
```

For systemd:

```ini
[Service]
MemoryLimit=2G
CPUQuota=200%
```

## Monitoring and Logging

### Access Logs

Configure Uvicorn logging:

```bash
uvicorn mlops.dashboard.server:create_dashboard_app \
    --host 0.0.0.0 \
    --port 8000 \
    --access-log \
    --log-level info \
    --log-config logging.yaml
```

### Health Checks

The dashboard exposes a health endpoint:

```bash
# Check health
curl http://localhost:8000/health

# Expected response
{"status":"ok"}
```

### Metrics Collection

Integrate with Prometheus:

```python
from prometheus_fastapi_instrumentator import Instrumentator

app = create_dashboard_app()
Instrumentator().instrument(app).expose(app)
```

## Security Considerations

### Authentication

Add authentication middleware:

```python
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBasic, HTTPBasicCredentials

security = HTTPBasic()

def authenticate(credentials: HTTPBasicCredentials = Depends(security)):
    if credentials.username != "admin" or credentials.password != "secret":
        raise HTTPException(status_code=401, detail="Unauthorized")
    return credentials.username
```

### CORS Configuration

Enable CORS for API access:

```python
from fastapi.middleware.cors import CORSMiddleware

app = create_dashboard_app()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://example.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Rate Limiting

Install slowapi:

```bash
uv add slowapi
```

Add rate limiting:

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.get("/api/data")
@limiter.limit("10/minute")
async def get_data(request: Request):
    ...
```

## Backup and Recovery

### Backup Dashboard Data

```bash
# Backup workspace data
tar -czf workspace-backup-$(date +%Y%m%d).tar.gz workspace_data/

# Backup MLFlow data
tar -czf mlflow-backup-$(date +%Y%m%d).tar.gz mlflow_data/

# Upload to S3
aws s3 cp workspace-backup-*.tar.gz s3://backups/dashboard/
```

### Automated Backup Script

Create `backup.sh`:

```bash
#!/bin/bash
BACKUP_DIR=/backups/mlops
DATE=$(date +%Y%m%d_%H%M%S)

mkdir -p $BACKUP_DIR

# Backup workspace
tar -czf $BACKUP_DIR/workspace-$DATE.tar.gz workspace_data/

# Backup MLFlow
tar -czf $BACKUP_DIR/mlflow-$DATE.tar.gz mlflow_data/

# Retain only last 7 days
find $BACKUP_DIR -type f -mtime +7 -delete

echo "Backup completed: $DATE"
```

Add to crontab:

```bash
# Run daily at 2 AM
0 2 * * * /opt/efficientai/backup.sh
```

## Troubleshooting

### Dashboard Not Starting

Check logs:

```bash
# systemd
journalctl -u mlops-dashboard -f

# Docker
docker logs -f mlops-dashboard

# Direct
uvicorn mlops.dashboard.server:create_dashboard_app --log-level debug
```

### Port Already in Use

Find and kill process:

```bash
lsof -i :8000
kill -9 <PID>
```

Or use a different port:

```bash
uv run efficientai-toolkit dashboard --port 8001
```

### Templates Not Found

Verify template directory:

```bash
ls mlops/dashboard/templates/
# Should show: base.html, overview.html, etc.
```

### No Data Showing

Ensure workspaces are configured:

```python
from mlops.workspace.manager import WorkspaceManager

manager = WorkspaceManager()
manager.create_workspace("test-project")
```

## Migration Guide

### From Standalone Dashboard

If you have an existing dashboard setup, migrate with:

```bash
# Export current configuration
python -m mlops.dashboard.export_config > config.yaml

# Import to new deployment
python -m mlops.dashboard.import_config config.yaml
```

## Support and Maintenance

### Update Dashboard

```bash
# Pull latest changes
git pull origin main

# Reinstall dependencies
uv sync

# Restart service
sudo systemctl restart mlops-dashboard
```

### Check Version

```bash
uv run efficientai-toolkit --version
```

### Contact

For issues and support:
- Check logs for detailed error messages
- Review test suite for usage examples
- Consult main README and component documentation
