#!/bin/bash

# MLOps Deployment Script
# Usage: ./deploy.sh [environment] [component]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOYMENT_DIR="$(dirname "$SCRIPT_DIR")"
ENVIRONMENT="${1:-production}"
COMPONENT="${2:-all}"

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

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."

    local missing=0

    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        missing=1
    fi

    if ! command -v kubectl &> /dev/null; then
        log_warn "kubectl is not installed (required for Kubernetes deployment)"
    fi

    if ! command -v terraform &> /dev/null; then
        log_warn "Terraform is not installed (required for infrastructure provisioning)"
    fi

    if [ $missing -eq 1 ]; then
        log_error "Missing required dependencies"
        exit 1
    fi

    log_info "Prerequisites check passed"
}

# Deploy with Docker Compose
deploy_docker_compose() {
    log_info "Deploying with Docker Compose..."

    cd "$DEPLOYMENT_DIR"

    # Check if .env file exists
    if [ ! -f .env ]; then
        log_warn ".env file not found, copying from .env.example"
        cp .env.example .env
        log_error "Please update .env file with your configuration and run again"
        exit 1
    fi

    # Pull latest images
    log_info "Pulling latest images..."
    docker-compose pull

    # Start services
    log_info "Starting services..."
    docker-compose up -d

    # Wait for services to be healthy
    log_info "Waiting for services to be healthy..."
    sleep 10

    # Check service health
    check_service_health

    log_info "Docker Compose deployment completed"
}

# Deploy to Kubernetes
deploy_kubernetes() {
    log_info "Deploying to Kubernetes..."

    cd "$DEPLOYMENT_DIR/kubernetes"

    # Create namespace
    log_info "Creating namespace..."
    kubectl apply -f namespace.yaml

    # Apply secrets (if not exists)
    if ! kubectl get secret mlops-secrets -n mlops &> /dev/null; then
        log_warn "Creating secrets from template..."
        kubectl apply -f secrets.yaml
        log_warn "Please update secrets with actual values: kubectl edit secret mlops-secrets -n mlops"
    fi

    # Apply configmap
    log_info "Applying configuration..."
    kubectl apply -f configmap.yaml

    # Deploy PostgreSQL
    log_info "Deploying PostgreSQL..."
    kubectl apply -f postgres-deployment.yaml

    # Wait for PostgreSQL to be ready
    kubectl wait --for=condition=ready pod -l app=postgres -n mlops --timeout=300s

    # Deploy MLFlow
    log_info "Deploying MLFlow..."
    kubectl apply -f mlflow-deployment.yaml

    # Deploy Ingress
    log_info "Configuring Ingress..."
    kubectl apply -f ingress.yaml

    # Check deployment status
    kubectl get deployments -n mlops
    kubectl get pods -n mlops

    log_info "Kubernetes deployment completed"
}

# Deploy infrastructure with Terraform
deploy_terraform() {
    log_info "Deploying infrastructure with Terraform..."

    cd "$DEPLOYMENT_DIR/terraform"

    # Initialize Terraform
    log_info "Initializing Terraform..."
    terraform init

    # Validate configuration
    log_info "Validating Terraform configuration..."
    terraform validate

    # Plan deployment
    log_info "Planning Terraform deployment..."
    terraform plan -out=tfplan

    # Confirm before applying
    read -p "Apply Terraform plan? (yes/no): " confirm
    if [ "$confirm" != "yes" ]; then
        log_warn "Terraform deployment cancelled"
        exit 0
    fi

    # Apply infrastructure
    log_info "Applying Terraform configuration..."
    terraform apply tfplan

    # Output important values
    log_info "Infrastructure outputs:"
    terraform output

    log_info "Terraform deployment completed"
}

# Check service health
check_service_health() {
    log_info "Checking service health..."

    local services=("mlflow:5000" "postgres:5432" "redis:6379" "dashboard:8000")
    local all_healthy=1

    for service in "${services[@]}"; do
        IFS=':' read -r name port <<< "$service"

        if docker-compose ps | grep "$name" | grep -q "Up"; then
            log_info "$name is running"
        else
            log_error "$name is not running"
            all_healthy=0
        fi
    done

    if [ $all_healthy -eq 1 ]; then
        log_info "All services are healthy"
    else
        log_error "Some services are not healthy"
        exit 1
    fi
}

# Main deployment logic
main() {
    log_info "Starting MLOps deployment..."
    log_info "Environment: $ENVIRONMENT"
    log_info "Component: $COMPONENT"

    check_prerequisites

    case "$COMPONENT" in
        docker|docker-compose)
            deploy_docker_compose
            ;;
        kubernetes|k8s)
            deploy_kubernetes
            ;;
        terraform|infra)
            deploy_terraform
            ;;
        all)
            log_info "Deploying all components..."
            deploy_terraform
            deploy_kubernetes
            ;;
        *)
            log_error "Unknown component: $COMPONENT"
            log_info "Valid components: docker, kubernetes, terraform, all"
            exit 1
            ;;
    esac

    log_info "Deployment completed successfully!"
    log_info ""
    log_info "Access services at:"
    log_info "  MLFlow:    http://localhost:5000"
    log_info "  Airflow:   http://localhost:8080"
    log_info "  Dashboard: http://localhost:8000"
    log_info "  Grafana:   http://localhost:3000"
}

main
