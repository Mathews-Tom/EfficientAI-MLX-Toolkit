#!/bin/bash

# Health Check Script for MLOps Services
# Usage: ./health-check.sh [namespace]

NAMESPACE="${1:-mlops}"
TIMEOUT=300
INTERVAL=5

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_http_endpoint() {
    local service=$1
    local port=$2
    local path=$3
    local expected_code=${4:-200}

    log_info "Checking $service at http://localhost:$port$path"

    if curl -s -o /dev/null -w "%{http_code}" "http://localhost:$port$path" | grep -q "$expected_code"; then
        log_info "$service is healthy"
        return 0
    else
        log_error "$service is not responding correctly"
        return 1
    fi
}

check_kubernetes_service() {
    local deployment=$1

    log_info "Checking Kubernetes deployment: $deployment"

    if kubectl rollout status deployment/"$deployment" -n "$NAMESPACE" --timeout="${TIMEOUT}s" &> /dev/null; then
        log_info "$deployment is ready"
        return 0
    else
        log_error "$deployment is not ready"
        return 1
    fi
}

check_docker_compose_service() {
    local service=$1

    log_info "Checking Docker Compose service: $service"

    if docker-compose ps | grep "$service" | grep -q "Up (healthy)"; then
        log_info "$service is healthy"
        return 0
    elif docker-compose ps | grep "$service" | grep -q "Up"; then
        log_warn "$service is running but not healthy yet"
        return 1
    else
        log_error "$service is not running"
        return 1
    fi
}

wait_for_service() {
    local service=$1
    local check_func=$2
    local max_attempts=$((TIMEOUT / INTERVAL))
    local attempt=1

    log_info "Waiting for $service to be ready (timeout: ${TIMEOUT}s)..."

    while [ $attempt -le $max_attempts ]; do
        if $check_func "$service"; then
            return 0
        fi

        log_info "Attempt $attempt/$max_attempts - waiting ${INTERVAL}s..."
        sleep $INTERVAL
        ((attempt++))
    done

    log_error "$service failed to become ready within ${TIMEOUT}s"
    return 1
}

check_all_docker_services() {
    log_info "Checking all Docker Compose services..."

    local services=("mlflow" "postgres" "redis" "minio" "airflow-webserver" "airflow-scheduler" "dashboard" "prometheus" "grafana")
    local all_healthy=1

    for service in "${services[@]}"; do
        if ! wait_for_service "$service" check_docker_compose_service; then
            all_healthy=0
        fi
    done

    return $all_healthy
}

check_all_kubernetes_services() {
    log_info "Checking all Kubernetes services in namespace: $NAMESPACE"

    local deployments=("mlflow" "postgres" "dashboard")
    local all_ready=1

    for deployment in "${deployments[@]}"; do
        if ! check_kubernetes_service "$deployment"; then
            all_ready=0
        fi
    done

    # Check pod status
    log_info "Pod status:"
    kubectl get pods -n "$NAMESPACE"

    # Check service endpoints
    log_info "Service endpoints:"
    kubectl get services -n "$NAMESPACE"

    return $all_ready
}

check_endpoint_connectivity() {
    log_info "Checking endpoint connectivity..."

    local endpoints=(
        "mlflow:5000:/health"
        "dashboard:8000:/health"
        "grafana:3000:/api/health"
        "prometheus:9090:/-/healthy"
    )

    local all_connected=1

    for endpoint in "${endpoints[@]}"; do
        IFS=':' read -r service port path <<< "$endpoint"

        if ! check_http_endpoint "$service" "$port" "$path"; then
            all_connected=0
        fi
    done

    return $all_connected
}

main() {
    log_info "Starting health check..."

    # Detect deployment type
    if command -v kubectl &> /dev/null && kubectl get namespace "$NAMESPACE" &> /dev/null; then
        log_info "Detected Kubernetes deployment"
        if check_all_kubernetes_services; then
            log_info "All Kubernetes services are healthy"
            exit 0
        else
            log_error "Some Kubernetes services are not healthy"
            exit 1
        fi
    elif command -v docker-compose &> /dev/null && docker-compose ps &> /dev/null; then
        log_info "Detected Docker Compose deployment"
        if check_all_docker_services && check_endpoint_connectivity; then
            log_info "All Docker Compose services are healthy"
            exit 0
        else
            log_error "Some Docker Compose services are not healthy"
            exit 1
        fi
    else
        log_error "Could not detect deployment type"
        log_info "Make sure you are in the deployment directory or have kubectl configured"
        exit 1
    fi
}

main
