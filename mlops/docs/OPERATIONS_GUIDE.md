# MLOps Operations & Deployment Guide

**Production operations handbook for MLOps infrastructure**

Version: 1.0.0
Last Updated: 2025-10-24

---

## Table of Contents

1. [Deployment](#deployment)
2. [Operations](#operations)
3. [Monitoring](#monitoring)
4. [Backup & Recovery](#backup--recovery)
5. [Scaling](#scaling)
6. [Troubleshooting](#troubleshooting)
7. [Maintenance](#maintenance)

---

## Deployment

### Pre-Deployment Checklist

- [ ] System requirements met
- [ ] Dependencies installed
- [ ] Configuration reviewed
- [ ] Security checklist completed
- [ ] Backup strategy defined
- [ ] Monitoring configured
- [ ] Documentation updated

### Environment Setup

#### Development Environment

```bash
# Install toolkit
cd /path/to/EfficientAI-MLX-Toolkit
uv sync

# Configure development settings
export MLOPS_ENVIRONMENT=development
export MLFLOW_TRACKING_URI=http://localhost:5000
export DVC_REMOTE_URL=file:///tmp/dvc-storage

# Initialize workspace
mkdir -p mlops/workspace/my-project
cd mlops/workspace/my-project
dvc init
```

#### Production Environment

```bash
# Production configuration
export MLOPS_ENVIRONMENT=production
export MLFLOW_TRACKING_URI=https://mlflow.production.example.com
export DVC_REMOTE_URL=s3://production-ml-data/dvc-storage
export AWS_ACCESS_KEY_ID=${AWS_KEY}
export AWS_SECRET_ACCESS_KEY=${AWS_SECRET}

# Security
export MLFLOW_TRACKING_USERNAME=admin
export MLFLOW_TRACKING_PASSWORD=${SECURE_PASSWORD}

# Enable HTTPS
export MLFLOW_TRACKING_INSECURE_TLS=false
```

### Service Deployment

#### MLFlow Server

**Development**:
```bash
# Local SQLite backend
mlflow server \
    --backend-store-uri sqlite:///mlops/workspace/mlflow.db \
    --default-artifact-root mlops/workspace/artifacts \
    --host 0.0.0.0 \
    --port 5000
```

**Production**:
```bash
# PostgreSQL backend with S3 artifacts
mlflow server \
    --backend-store-uri postgresql://user:password@localhost:5432/mlflow \
    --default-artifact-root s3://production-ml-artifacts/ \
    --host 0.0.0.0 \
    --port 5000 \
    --workers 4 \
    --gunicorn-opts "--timeout 120"
```

**Docker Deployment**:
```dockerfile
# Dockerfile
FROM python:3.11-slim

RUN pip install mlflow psycopg2-binary boto3

EXPOSE 5000

CMD ["mlflow", "server", \
     "--backend-store-uri", "postgresql://user:password@db:5432/mlflow", \
     "--default-artifact-root", "s3://artifacts/", \
     "--host", "0.0.0.0", \
     "--port", "5000"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  mlflow:
    build: .
    ports:
      - "5000:5000"
    environment:
      - AWS_ACCESS_KEY_ID=${AWS_KEY}
      - AWS_SECRET_ACCESS_KEY=${AWS_SECRET}
    depends_on:
      - db

  db:
    image: postgres:15
    environment:
      - POSTGRES_USER=mlflow
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=mlflow
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
```

#### BentoML Serving

**Local Deployment**:
```bash
# Serve model
bentoml serve my_model:latest --port 3000
```

**Production Deployment**:
```bash
# Build container
bentoml containerize my_model:latest -t my-registry/my_model:v1.0

# Push to registry
docker push my-registry/my_model:v1.0

# Deploy with Kubernetes
kubectl apply -f deployment.yaml
```

**Kubernetes Deployment**:
```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-model-deployment
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ml-model
  template:
    metadata:
      labels:
        app: ml-model
    spec:
      containers:
      - name: model-server
        image: my-registry/my_model:v1.0
        ports:
        - containerPort: 3000
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
---
apiVersion: v1
kind: Service
metadata:
  name: ml-model-service
spec:
  selector:
    app: ml-model
  ports:
  - protocol: TCP
    port: 80
    targetPort: 3000
  type: LoadBalancer
```

#### Airflow Orchestration

**Local Development**:
```bash
# Initialize Airflow
export AIRFLOW_HOME=mlops/airflow
airflow db init
airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com

# Start services
airflow webserver --port 8080 &
airflow scheduler &
```

**Production Deployment**:
```bash
# Use Celery executor
export AIRFLOW__CELERY__BROKER_URL=redis://redis:6379/0
export AIRFLOW__CELERY__RESULT_BACKEND=db+postgresql://user:password@db:5432/airflow

# Start services
airflow webserver &
airflow scheduler &
airflow celery worker &
```

**Docker Compose**:
```yaml
# airflow-docker-compose.yml
version: '3.8'

services:
  postgres:
    image: postgres:13
    environment:
      POSTGRES_USER: airflow
      POSTGRES_PASSWORD: airflow
      POSTGRES_DB: airflow

  redis:
    image: redis:latest

  airflow-webserver:
    image: apache/airflow:2.7.0
    command: webserver
    ports:
      - "8080:8080"
    depends_on:
      - postgres
      - redis
    environment:
      - AIRFLOW__CORE__EXECUTOR=CeleryExecutor
      - AIRFLOW__CELERY__BROKER_URL=redis://redis:6379/0

  airflow-scheduler:
    image: apache/airflow:2.7.0
    command: scheduler
    depends_on:
      - postgres
      - redis

  airflow-worker:
    image: apache/airflow:2.7.0
    command: celery worker
    depends_on:
      - postgres
      - redis
```

#### Dashboard Deployment

```bash
# Start dashboard
cd mlops/dashboard
uv run python server.py --host 0.0.0.0 --port 8000

# Or with gunicorn
gunicorn -w 4 -b 0.0.0.0:8000 "server:create_app()"
```

---

## Operations

### Service Management

#### Start Services

```bash
# Start all services
./scripts/start_mlops_services.sh

# Or individually
mlflow server &
bentoml serve model:latest &
uv run python mlops/dashboard/server.py &
```

#### Stop Services

```bash
# Stop all services
./scripts/stop_mlops_services.sh

# Or individually
pkill -f "mlflow server"
pkill -f "bentoml serve"
pkill -f "dashboard/server.py"
```

#### Service Status

```bash
# Check service status
./scripts/check_mlops_health.sh

# Individual health checks
curl http://localhost:5000/health      # MLFlow
curl http://localhost:3000/healthz     # BentoML
curl http://localhost:8000/health      # Dashboard
```

### Configuration Management

#### Environment-Specific Configs

```python
# config/production.yaml
mlflow:
  tracking_uri: "https://mlflow.prod.example.com"
  artifact_location: "s3://prod-artifacts/"
  auth_enabled: true

dvc:
  remote_url: "s3://prod-ml-data/"
  cache_dir: "/data/dvc-cache"
  auto_push: true

bentoml:
  max_replicas: 10
  min_replicas: 2
  target_cpu_percent: 75

monitoring:
  alert_threshold: 0.85
  check_interval_minutes: 5
```

#### Loading Configurations

```python
import yaml
from pathlib import Path

def load_config(environment="production"):
    config_path = Path(f"config/{environment}.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)

config = load_config("production")
```

### User Management

#### MLFlow Users

```bash
# Create user
mlflow users create --username john --password secure_pass

# Update user
mlflow users update --username john --password new_pass

# List users
mlflow users list

# Delete user
mlflow users delete --username john
```

#### Airflow Users

```bash
# Create admin user
airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com

# Create viewer user
airflow users create \
    --username viewer \
    --firstname Viewer \
    --lastname User \
    --role Viewer \
    --email viewer@example.com
```

---

## Monitoring

### Service Monitoring

#### Health Checks

```bash
#!/bin/bash
# health_check.sh

# MLFlow
if curl -s http://localhost:5000/health > /dev/null; then
    echo "MLFlow: OK"
else
    echo "MLFlow: DOWN"
fi

# BentoML
if curl -s http://localhost:3000/healthz > /dev/null; then
    echo "BentoML: OK"
else
    echo "BentoML: DOWN"
fi

# Dashboard
if curl -s http://localhost:8000/health > /dev/null; then
    echo "Dashboard: OK"
else
    echo "Dashboard: DOWN"
fi
```

#### Resource Monitoring

```bash
# Monitor resource usage
ps aux | grep -E "mlflow|bentoml|airflow" | awk '{print $2, $3, $4, $11}'

# Disk usage
df -h mlops/workspace/

# Memory usage
free -h

# GPU/MPS usage (Apple Silicon)
sudo powermetrics --samplers gpu_power -i 1000 -n 1
```

#### Log Monitoring

```bash
# Tail all logs
tail -f mlops/workspace/logs/*.log

# Search for errors
grep -r "ERROR" mlops/workspace/logs/

# Count errors by service
for log in mlops/workspace/logs/*.log; do
    echo "$log: $(grep -c ERROR $log) errors"
done
```

### Performance Metrics

#### MLFlow Metrics

```python
from mlops.client.mlflow_client import create_client

client = create_client(project_name="monitoring")

# Query recent runs
runs = client.search_runs(
    experiment_ids=["1"],
    filter_string="metrics.accuracy > 0.9",
    max_results=100
)

# Calculate statistics
accuracies = [run.data.metrics["accuracy"] for run in runs]
avg_accuracy = sum(accuracies) / len(accuracies)
print(f"Average accuracy: {avg_accuracy:.4f}")
```

#### System Metrics

```python
from mlops.silicon.monitor import AppleSiliconMonitor

monitor = AppleSiliconMonitor()
metrics = monitor.get_metrics()

print(f"Thermal State: {metrics['thermal_state']}")
print(f"Memory Pressure: {metrics['memory_pressure']}")
print(f"CPU Usage: {metrics['cpu_usage_percent']}%")
```

### Alerting

#### Configure Alerts

```python
# alerts_config.yaml
alerts:
  - name: "high_error_rate"
    condition: "error_rate > 0.05"
    severity: "critical"
    channels: ["slack", "email"]

  - name: "drift_detected"
    condition: "drift_score > 0.8"
    severity: "warning"
    channels: ["slack"]

  - name: "low_disk_space"
    condition: "disk_usage_percent > 90"
    severity: "critical"
    channels: ["email", "pagerduty"]
```

#### Alert Handler

```python
class AlertManager:
    def check_alerts(self):
        # Check error rate
        if self.get_error_rate() > 0.05:
            self.send_alert("high_error_rate", severity="critical")

        # Check drift
        if self.get_drift_score() > 0.8:
            self.send_alert("drift_detected", severity="warning")

        # Check disk space
        if self.get_disk_usage() > 90:
            self.send_alert("low_disk_space", severity="critical")

    def send_alert(self, alert_name, severity):
        # Send to configured channels
        pass
```

---

## Backup & Recovery

### Backup Strategy

#### MLFlow Backups

```bash
#!/bin/bash
# backup_mlflow.sh

BACKUP_DIR="backups/mlflow"
DATE=$(date +%Y%m%d_%H%M%S)

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Backup database
sqlite3 mlops/workspace/mlflow.db ".backup '$BACKUP_DIR/mlflow_$DATE.db'"

# Backup experiments
tar -czf "$BACKUP_DIR/mlruns_$DATE.tar.gz" mlops/workspace/mlruns/

# Backup artifacts
tar -czf "$BACKUP_DIR/artifacts_$DATE.tar.gz" mlops/workspace/artifacts/

echo "Backup completed: $BACKUP_DIR"
```

#### DVC Backups

```bash
# Push all data to remote
dvc push

# Verify remote storage
dvc list-url s3://your-bucket/dvc-storage

# Backup DVC metadata
tar -czf dvc_backup_$(date +%Y%m%d).tar.gz .dvc/
```

#### Configuration Backups

```bash
# Backup configurations
tar -czf config_backup_$(date +%Y%m%d).tar.gz \
    mlops/config/ \
    .env \
    .dvc/config
```

### Automated Backups

```bash
# crontab entry
# Daily backups at 2 AM
0 2 * * * /path/to/backup_mlflow.sh

# Weekly full backups on Sunday
0 3 * * 0 /path/to/full_backup.sh

# Monthly archive to S3
0 4 1 * * /path/to/archive_to_s3.sh
```

### Recovery Procedures

#### Restore MLFlow

```bash
# Stop MLFlow
pkill -f "mlflow server"

# Restore database
cp backups/mlflow/mlflow_20251024_020000.db mlops/workspace/mlflow.db

# Restore experiments
tar -xzf backups/mlflow/mlruns_20251024_020000.tar.gz -C mlops/workspace/

# Restart MLFlow
mlflow server --backend-store-uri sqlite:///mlops/workspace/mlflow.db &
```

#### Restore DVC Data

```bash
# Pull from remote
dvc pull

# Restore from specific version
git checkout <commit-hash>
dvc checkout
```

### Disaster Recovery

**Recovery Time Objective (RTO)**: 1 hour
**Recovery Point Objective (RPO)**: 24 hours

**DR Procedure**:
1. Verify backup integrity
2. Provision new infrastructure
3. Restore database
4. Restore artifacts
5. Verify data integrity
6. Resume services
7. Validate functionality

---

## Scaling

### Horizontal Scaling

#### MLFlow

```bash
# Load balanced MLFlow servers
for i in {1..3}; do
    mlflow server \
        --backend-store-uri postgresql://user:pass@db:5432/mlflow \
        --default-artifact-root s3://artifacts/ \
        --host 0.0.0.0 \
        --port 500$i &
done

# Configure load balancer
# nginx.conf
upstream mlflow_backend {
    server 127.0.0.1:5001;
    server 127.0.0.1:5002;
    server 127.0.0.1:5003;
}
```

#### BentoML

```yaml
# Kubernetes HPA
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ml-model-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ml-model-deployment
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

### Vertical Scaling

#### Resource Allocation

```yaml
# docker-compose.yml
services:
  mlflow:
    deploy:
      resources:
        limits:
          cpus: '4.0'
          memory: 8G
        reservations:
          cpus: '2.0'
          memory: 4G
```

### Database Scaling

#### PostgreSQL Optimization

```sql
-- Optimize MLFlow queries
CREATE INDEX idx_experiments_name ON experiments(name);
CREATE INDEX idx_runs_experiment_id ON runs(experiment_id);
CREATE INDEX idx_metrics_run_id ON metrics(run_id);

-- Regular maintenance
VACUUM ANALYZE experiments;
VACUUM ANALYZE runs;
VACUUM ANALYZE metrics;
```

---

## Troubleshooting

### Common Issues

#### High Memory Usage

**Diagnosis**:
```bash
# Check memory usage
ps aux --sort=-%mem | head -10

# Monitor over time
while true; do
    free -h
    sleep 5
done
```

**Solutions**:
- Restart services
- Increase memory limits
- Enable swap
- Optimize batch sizes

#### Slow Performance

**Diagnosis**:
```bash
# Check CPU usage
top -o %CPU

# Check I/O wait
iostat -x 1

# Check network
netstat -i
```

**Solutions**:
- Scale horizontally
- Optimize queries
- Cache frequently accessed data
- Use CDN for artifacts

#### Connection Timeouts

**Diagnosis**:
```bash
# Test connections
telnet localhost 5000
curl -v http://localhost:5000/health

# Check firewall
sudo iptables -L

# Check network
ping -c 4 mlflow.example.com
```

**Solutions**:
- Increase timeout settings
- Check firewall rules
- Verify network connectivity
- Check DNS resolution

---

## Maintenance

### Regular Maintenance

#### Daily Tasks
- [ ] Check service health
- [ ] Monitor resource usage
- [ ] Review error logs
- [ ] Verify backups

#### Weekly Tasks
- [ ] Review performance metrics
- [ ] Clean up old experiments
- [ ] Update documentation
- [ ] Test disaster recovery

#### Monthly Tasks
- [ ] Security audit
- [ ] Dependency updates
- [ ] Performance optimization
- [ ] Capacity planning

### Cleanup Procedures

#### MLFlow Cleanup

```python
# cleanup_old_experiments.py
from mlops.client.mlflow_client import create_client
from datetime import datetime, timedelta

client = create_client(project_name="cleanup")

# Delete experiments older than 90 days
cutoff_date = datetime.now() - timedelta(days=90)

experiments = client.list_experiments()
for exp in experiments:
    if exp.creation_time < cutoff_date.timestamp():
        client.delete_experiment(exp.experiment_id)
        print(f"Deleted experiment: {exp.name}")
```

#### DVC Cleanup

```bash
# Remove unused cache
dvc gc --workspace --cloud

# Clean temporary files
dvc cache clean
```

### Updates & Patches

```bash
# Update dependencies
uv sync --upgrade

# Apply security patches
uv add package@latest

# Test after updates
uv run pytest mlops/tests/ -v
```

---

## Operations Automation

### Ansible Playbook

```yaml
# deploy_mlops.yml
---
- name: Deploy MLOps Infrastructure
  hosts: mlops_servers
  become: yes

  tasks:
    - name: Install dependencies
      apt:
        name:
          - python3
          - postgresql
          - redis
        state: present

    - name: Start MLFlow
      systemd:
        name: mlflow
        state: started
        enabled: yes

    - name: Configure firewall
      ufw:
        rule: allow
        port: '5000'
        proto: tcp
```

### Terraform Configuration

```hcl
# main.tf
resource "aws_instance" "mlflow_server" {
  ami           = "ami-12345678"
  instance_type = "t3.large"

  tags = {
    Name = "MLFlow Server"
  }
}

resource "aws_s3_bucket" "artifacts" {
  bucket = "ml-artifacts-${var.environment}"

  versioning {
    enabled = true
  }
}
```

---

**End of Operations Guide**
