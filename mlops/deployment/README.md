# MLOps Production Deployment Guide

Complete production deployment infrastructure for the MLOps platform.

## Overview

This deployment package provides production-ready configurations for deploying the complete MLOps stack including:

- **MLFlow** - Experiment tracking and model registry
- **DVC** - Data version control
- **Airflow** - Workflow orchestration
- **Ray** - Distributed computing
- **Evidently** - Model monitoring
- **Dashboard** - Unified MLOps interface
- **Prometheus** - Metrics collection
- **Grafana** - Visualization

## Deployment Options

### 1. Docker Compose (Development/Small Production)

**Best for**: Local development, small teams, single-server deployments

```bash
# Copy environment configuration
cp .env.example .env
# Edit .env with your configuration
vim .env

# Deploy all services
docker-compose up -d

# Check service health
./scripts/health-check.sh

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

**Services Access**:
- MLFlow: http://localhost:5000
- Airflow: http://localhost:8080
- Dashboard: http://localhost:8000
- Grafana: http://localhost:3000
- Prometheus: http://localhost:9090
- MinIO Console: http://localhost:9001

### 2. Kubernetes (Production)

**Best for**: Production deployments, high availability, auto-scaling

```bash
# Create namespace
kubectl apply -f kubernetes/namespace.yaml

# Configure secrets (update with your values)
kubectl apply -f kubernetes/secrets.yaml

# Deploy configuration
kubectl apply -f kubernetes/configmap.yaml

# Deploy PostgreSQL
kubectl apply -f kubernetes/postgres-deployment.yaml

# Wait for PostgreSQL to be ready
kubectl wait --for=condition=ready pod -l app=postgres -n mlops --timeout=300s

# Deploy MLFlow
kubectl apply -f kubernetes/mlflow-deployment.yaml

# Configure ingress (update domains in ingress.yaml)
kubectl apply -f kubernetes/ingress.yaml

# Check deployment status
kubectl get all -n mlops

# Check pod logs
kubectl logs -f deployment/mlflow -n mlops
```

### 3. Terraform (Infrastructure as Code)

**Best for**: Cloud infrastructure provisioning (AWS)

```bash
cd terraform/

# Initialize Terraform
terraform init

# Create variables file
cat > terraform.tfvars <<EOF
aws_region          = "us-east-1"
environment         = "production"
project_name        = "mlops"
db_username         = "mlops"
db_password         = "secure_password_here"
enable_eks          = true
EOF

# Plan deployment
terraform plan -out=tfplan

# Apply infrastructure
terraform apply tfplan

# View outputs
terraform output
```

## Architecture

### Network Architecture
```
┌─────────────────────────────────────────────────────────┐
│                      Load Balancer                       │
│                     (Nginx/Ingress)                      │
└─────────────────────┬───────────────────────────────────┘
                      │
        ┌─────────────┼─────────────┐
        │             │             │
┌───────▼────┐ ┌──────▼──────┐ ┌───▼────────┐
│   MLFlow   │ │   Airflow   │ │  Dashboard │
│  (Port 5000)│ │  (Port 8080)│ │ (Port 8000)│
└────────┬───┘ └──────┬──────┘ └───┬────────┘
         │            │            │
         └────────────┼────────────┘
                      │
         ┌────────────┼────────────┐
         │            │            │
    ┌────▼──┐    ┌───▼───┐   ┌────▼────┐
    │Postgres│    │ Redis │   │  MinIO  │
    │(5432) │    │(6379) │   │(9000)   │
    └────────┘    └───────┘   └─────────┘
```

### Component Responsibilities

**MLFlow**:
- Experiment tracking
- Model registry
- Artifact storage
- Metrics logging

**Airflow**:
- Pipeline orchestration
- Scheduled workflows
- DAG management
- Task execution

**PostgreSQL**:
- MLFlow metadata
- Airflow metadata
- Persistent storage

**Redis**:
- Caching layer
- Task queue (Celery)
- Session storage

**MinIO/S3**:
- Artifact storage
- Model storage
- DVC remote storage

**Prometheus**:
- Metrics collection
- Alert management
- Service monitoring

**Grafana**:
- Metrics visualization
- Dashboard creation
- Alerting interface

## Configuration

### Environment Variables

Key environment variables in `.env`:

```bash
# Database
POSTGRES_USER=mlops
POSTGRES_PASSWORD=secure_password
POSTGRES_DB=mlops

# Redis
REDIS_PASSWORD=redis_password

# MinIO
MINIO_ROOT_USER=minioadmin
MINIO_ROOT_PASSWORD=minioadmin_password

# MLFlow
MLFLOW_TRACKING_USERNAME=admin
MLFLOW_TRACKING_PASSWORD=mlflow_password

# Airflow
AIRFLOW_ADMIN_USER=admin
AIRFLOW_ADMIN_PASSWORD=airflow_password
```

### Resource Requirements

**Minimum** (Docker Compose):
- CPU: 8 cores
- RAM: 16 GB
- Disk: 100 GB SSD

**Recommended** (Production):
- CPU: 16+ cores
- RAM: 32+ GB
- Disk: 500 GB+ SSD

**Kubernetes Node Requirements**:
- Instance Type: t3.xlarge or larger
- Nodes: 3+ for high availability
- Storage: 100 GB per node

## Monitoring

### Prometheus Metrics

Access Prometheus: http://localhost:9090

Key metrics:
- `up` - Service availability
- `node_cpu_seconds_total` - CPU usage
- `node_memory_MemAvailable_bytes` - Memory usage
- `mlflow_requests_total` - MLFlow request count
- `airflow_dag_run_duration_seconds` - DAG execution time

### Grafana Dashboards

Access Grafana: http://localhost:3000

Pre-configured dashboards:
- MLOps Platform Overview
- Service Health
- Resource Usage
- Application Metrics

### Alerts

Alert rules defined in `monitoring/alerts.yml`:
- Service down
- High CPU/memory usage
- Disk space low
- Database connection issues
- High error rates

## Security

### Best Practices

1. **Change all default passwords** in `.env` and `secrets.yaml`
2. **Enable HTTPS** for all public endpoints
3. **Configure firewall rules** to restrict access
4. **Use secrets management** (AWS Secrets Manager, Vault)
5. **Enable audit logging** for compliance
6. **Regular security updates** for all components
7. **Network segmentation** using VPCs and security groups

### SSL/TLS Configuration

For production with SSL:

```bash
# Generate self-signed certificate (development)
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout nginx/ssl/key.pem \
  -out nginx/ssl/cert.pem

# Or use Let's Encrypt (production)
# Configure cert-manager in Kubernetes
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml
```

## Backup and Recovery

### Automated Backups

```bash
# PostgreSQL backup
docker-compose exec postgres pg_dump -U mlops mlops > backup_$(date +%Y%m%d).sql

# MinIO backup
mc mirror minio/mlflow-artifacts /backups/artifacts/

# DVC cache backup
dvc push --all-branches --all-tags
```

### Restore Procedures

```bash
# Restore PostgreSQL
docker-compose exec -T postgres psql -U mlops mlops < backup_20251024.sql

# Restore MinIO
mc mirror /backups/artifacts/ minio/mlflow-artifacts

# Restore DVC
dvc pull
```

## Scaling

### Horizontal Scaling

**Docker Compose**:
```bash
# Scale specific service
docker-compose up -d --scale mlflow=3
```

**Kubernetes**:
```bash
# Scale deployment
kubectl scale deployment mlflow --replicas=5 -n mlops

# Enable auto-scaling
kubectl autoscale deployment mlflow \
  --cpu-percent=70 \
  --min=2 \
  --max=10 \
  -n mlops
```

### Vertical Scaling

Update resource limits in:
- Docker Compose: `resources.limits` section
- Kubernetes: `resources.requests` and `resources.limits`

## Troubleshooting

### Common Issues

**Services not starting**:
```bash
# Check logs
docker-compose logs service-name
kubectl logs deployment/service-name -n mlops

# Check health
./scripts/health-check.sh
```

**Database connection errors**:
```bash
# Verify database is running
docker-compose ps postgres
kubectl get pods -l app=postgres -n mlops

# Test connection
docker-compose exec postgres psql -U mlops -d mlops -c "SELECT 1;"
```

**Storage issues**:
```bash
# Check disk usage
df -h

# Clean up Docker
docker system prune -a

# Expand volumes (Kubernetes)
kubectl edit pvc postgres-storage -n mlops
```

### Debug Mode

Enable debug logging:
```bash
# Docker Compose
export DEBUG=true
docker-compose up

# Kubernetes
kubectl set env deployment/mlflow DEBUG=true -n mlops
```

## CI/CD Integration

### GitHub Actions

See `ci-cd/github-actions.yml` for complete CI/CD pipeline:
- Automated testing
- Security scanning
- Docker image building
- Kubernetes deployment
- Smoke tests

### GitLab CI

Example `.gitlab-ci.yml`:
```yaml
stages:
  - test
  - build
  - deploy

test:
  stage: test
  script:
    - uv run pytest mlops/tests/

build:
  stage: build
  script:
    - docker build -t mlops/mlflow:$CI_COMMIT_SHA .

deploy:
  stage: deploy
  script:
    - kubectl apply -f kubernetes/
```

## Maintenance

### Regular Tasks

**Daily**:
- Check service health
- Monitor resource usage
- Review error logs

**Weekly**:
- Performance metrics review
- Security updates
- Backup verification

**Monthly**:
- Capacity planning
- Cost optimization
- Disaster recovery drill

### Update Procedure

```bash
# Pull latest changes
git pull origin main

# Update Docker images
docker-compose pull
docker-compose up -d

# Or for Kubernetes
kubectl set image deployment/mlflow mlflow=new-image:tag -n mlops
kubectl rollout status deployment/mlflow -n mlops
```

## Support

### Resources

- [Operations Guide](../docs/OPERATIONS_GUIDE.md)
- [Security Checklist](../docs/SECURITY_CHECKLIST.md)
- [Complete Guide](../docs/COMPLETE_GUIDE.md)
- [Production Checklist](PRODUCTION_CHECKLIST.md)

### Contacts

- Platform Team: platform@example.com
- On-Call: oncall@example.com
- Security: security@example.com

## License

See repository root for license information.
