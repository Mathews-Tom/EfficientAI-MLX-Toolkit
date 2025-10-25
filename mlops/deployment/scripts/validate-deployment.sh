#!/bin/bash

# Deployment Validation Script
# Validates all deployment configurations before deployment

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOYMENT_DIR="$(dirname "$SCRIPT_DIR")"

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

VALIDATION_PASSED=1

validate_docker_compose() {
    log_info "Validating Docker Compose configuration..."

    if [ ! -f "$DEPLOYMENT_DIR/docker-compose.yml" ]; then
        log_error "docker-compose.yml not found"
        VALIDATION_PASSED=0
        return
    fi

    if command -v docker-compose &> /dev/null; then
        if docker-compose -f "$DEPLOYMENT_DIR/docker-compose.yml" config > /dev/null 2>&1; then
            log_info "Docker Compose configuration is valid"
        else
            log_error "Docker Compose configuration is invalid"
            VALIDATION_PASSED=0
        fi
    else
        log_warn "docker-compose not installed, skipping validation"
    fi
}

validate_kubernetes() {
    log_info "Validating Kubernetes manifests..."

    if [ ! -d "$DEPLOYMENT_DIR/kubernetes" ]; then
        log_error "kubernetes directory not found"
        VALIDATION_PASSED=0
        return
    fi

    if command -v kubectl &> /dev/null; then
        for file in "$DEPLOYMENT_DIR/kubernetes"/*.yaml; do
            if kubectl apply --dry-run=client -f "$file" > /dev/null 2>&1; then
                log_info "$(basename "$file") is valid"
            else
                log_error "$(basename "$file") is invalid"
                VALIDATION_PASSED=0
            fi
        done
    else
        log_warn "kubectl not installed, skipping validation"
    fi
}

validate_terraform() {
    log_info "Validating Terraform configuration..."

    if [ ! -d "$DEPLOYMENT_DIR/terraform" ]; then
        log_error "terraform directory not found"
        VALIDATION_PASSED=0
        return
    fi

    if command -v terraform &> /dev/null; then
        cd "$DEPLOYMENT_DIR/terraform"

        if terraform init -backend=false > /dev/null 2>&1; then
            if terraform validate > /dev/null 2>&1; then
                log_info "Terraform configuration is valid"
            else
                log_error "Terraform configuration is invalid"
                VALIDATION_PASSED=0
            fi
        else
            log_error "Terraform initialization failed"
            VALIDATION_PASSED=0
        fi

        cd - > /dev/null
    else
        log_warn "terraform not installed, skipping validation"
    fi
}

validate_environment_file() {
    log_info "Validating environment file..."

    if [ ! -f "$DEPLOYMENT_DIR/.env" ]; then
        log_warn ".env file not found (this is expected for first run)"
        log_info "Copy .env.example to .env and configure before deployment"
    else
        # Check for default passwords
        if grep -q "change_me" "$DEPLOYMENT_DIR/.env"; then
            log_error "Found default passwords in .env file - please change them"
            VALIDATION_PASSED=0
        else
            log_info ".env file looks good (no default passwords found)"
        fi
    fi
}

validate_scripts() {
    log_info "Validating deployment scripts..."

    local scripts=("deploy.sh" "health-check.sh" "validate-deployment.sh")

    for script in "${scripts[@]}"; do
        if [ ! -f "$SCRIPT_DIR/$script" ]; then
            log_error "Script $script not found"
            VALIDATION_PASSED=0
        elif [ ! -x "$SCRIPT_DIR/$script" ]; then
            log_warn "Script $script is not executable"
            chmod +x "$SCRIPT_DIR/$script"
            log_info "Made $script executable"
        else
            log_info "Script $script is valid and executable"
        fi
    done
}

validate_monitoring_config() {
    log_info "Validating monitoring configuration..."

    # Check Prometheus config
    if [ ! -f "$DEPLOYMENT_DIR/monitoring/prometheus.yml" ]; then
        log_error "prometheus.yml not found"
        VALIDATION_PASSED=0
    else
        log_info "Prometheus configuration found"
    fi

    # Check alerts config
    if [ ! -f "$DEPLOYMENT_DIR/monitoring/alerts.yml" ]; then
        log_error "alerts.yml not found"
        VALIDATION_PASSED=0
    else
        log_info "Alert rules configuration found"
    fi

    # Check Grafana datasources
    if [ ! -f "$DEPLOYMENT_DIR/monitoring/grafana/datasources/prometheus.yml" ]; then
        log_error "Grafana datasource configuration not found"
        VALIDATION_PASSED=0
    else
        log_info "Grafana datasource configuration found"
    fi
}

validate_nginx_config() {
    log_info "Validating Nginx configuration..."

    if [ ! -f "$DEPLOYMENT_DIR/nginx/nginx.conf" ]; then
        log_error "nginx.conf not found"
        VALIDATION_PASSED=0
    else
        if command -v nginx &> /dev/null; then
            if nginx -t -c "$DEPLOYMENT_DIR/nginx/nginx.conf" > /dev/null 2>&1; then
                log_info "Nginx configuration is valid"
            else
                log_error "Nginx configuration is invalid"
                VALIDATION_PASSED=0
            fi
        else
            log_info "Nginx configuration found (nginx not installed for validation)"
        fi
    fi
}

print_summary() {
    echo ""
    echo "======================================"
    echo "Validation Summary"
    echo "======================================"
    echo ""

    if [ $VALIDATION_PASSED -eq 1 ]; then
        log_info "All validations passed!"
        echo ""
        log_info "Next steps:"
        echo "  1. Copy .env.example to .env and configure"
        echo "  2. Update secrets in kubernetes/secrets.yaml (if using K8s)"
        echo "  3. Update domain names in nginx/nginx.conf and kubernetes/ingress.yaml"
        echo "  4. Review PRODUCTION_CHECKLIST.md"
        echo "  5. Run ./scripts/deploy.sh <environment> <component>"
        echo ""
        exit 0
    else
        log_error "Validation failed!"
        echo ""
        log_error "Please fix the errors above before deploying"
        echo ""
        exit 1
    fi
}

main() {
    log_info "Starting deployment validation..."
    echo ""

    validate_docker_compose
    validate_kubernetes
    validate_terraform
    validate_environment_file
    validate_scripts
    validate_monitoring_config
    validate_nginx_config

    print_summary
}

main
