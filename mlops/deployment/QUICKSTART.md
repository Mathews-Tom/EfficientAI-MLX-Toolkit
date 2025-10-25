# MLOps Deployment - Quick Start Guide

**Get your MLOps platform running in under 10 minutes!**

---

## Prerequisites

- Docker and Docker Compose installed
- 16GB RAM minimum
- 100GB disk space
- macOS, Linux, or WSL2

---

## Step 1: Configure Environment (2 minutes)

```bash
cd mlops/deployment

# Copy environment template
cp .env.example .env

# Edit configuration (use your favorite editor)
vim .env  # or nano, code, etc.
```

**Required changes in `.env`**:
- Change all passwords (look for `_change_me`)
- Update `POSTGRES_PASSWORD`
- Update `REDIS_PASSWORD`
- Update `MINIO_ROOT_PASSWORD`
- Update service passwords

---

## Step 2: Start Services (5 minutes)

```bash
# Start all services
docker-compose up -d

# Watch logs (optional)
docker-compose logs -f
```

Services will start in this order:
1. PostgreSQL → Redis → MinIO
2. MinIO initialization
3. MLFlow → Airflow → Dashboard
4. Prometheus → Grafana → Nginx

---

## Step 3: Verify Health (1 minute)

```bash
# Run health check
./scripts/health-check.sh
```

Expected output:
```
[INFO] MLFlow: OK
[INFO] Postgres: OK
[INFO] Redis: OK
[INFO] Dashboard: OK
[INFO] All services are healthy
```

---

## Step 4: Access Services (1 minute)

Open your browser and visit:

- **MLFlow**: http://localhost:5000
- **Airflow**: http://localhost:8080
- **Dashboard**: http://localhost:8000
- **Grafana**: http://localhost:3000
- **Prometheus**: http://localhost:9090

**Default credentials**:
- Grafana: `admin` / `admin123` (from your `.env`)
- Airflow: `admin` / `airflow_pass` (from your `.env`)

---

## Quick Commands

### View all services
```bash
docker-compose ps
```

### View logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f mlflow
```

### Restart a service
```bash
docker-compose restart mlflow
```

### Stop all services
```bash
docker-compose down
```

### Stop and remove volumes (clean slate)
```bash
docker-compose down -v
```

---

## Troubleshooting

### Services won't start
```bash
# Check for port conflicts
docker-compose ps
netstat -an | grep -E "5000|5432|6379|8080|8000"

# Restart services
docker-compose restart
```

### Health checks failing
```bash
# Wait a bit longer (services need time to initialize)
sleep 30
./scripts/health-check.sh

# Check specific service logs
docker-compose logs service-name
```

### Out of disk space
```bash
# Clean up Docker
docker system prune -a --volumes

# Remove old containers
docker-compose down -v
```

---

## Next Steps

1. **Configure MLFlow**:
   - Visit http://localhost:5000
   - Create your first experiment

2. **Set up Airflow DAGs**:
   - Place DAGs in `mlops/airflow/dags/`
   - Visit http://localhost:8080

3. **View Metrics**:
   - Visit http://localhost:3000 (Grafana)
   - Check pre-configured dashboards

4. **Read Documentation**:
   - [Full README](README.md)
   - [Production Checklist](PRODUCTION_CHECKLIST.md)
   - [Operations Guide](../docs/OPERATIONS_GUIDE.md)

---

## Production Deployment

For production deployment:
1. Read [PRODUCTION_CHECKLIST.md](PRODUCTION_CHECKLIST.md)
2. Use Kubernetes: `kubectl apply -f kubernetes/`
3. Or Terraform: `cd terraform && terraform apply`

---

## Get Help

- Check logs: `docker-compose logs -f`
- Run health check: `./scripts/health-check.sh`
- Validate config: `./scripts/validate-deployment.sh`
- Read docs: [README.md](README.md)

---

**You're all set! Happy MLOps! 🚀**
