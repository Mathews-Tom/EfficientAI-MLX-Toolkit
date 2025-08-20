# Makefile for LoRA Fine-tuning Framework
# Provides convenient commands for development, testing, and deployment

.PHONY: help install test test-fast test-integration test-benchmark test-coverage
.PHONY: lint format type-check quality-check clean serve demo
.PHONY: build-docs serve-docs install-dev setup-dev

# Default target
help: ## Show this help message
	@echo "🚀 LoRA Fine-tuning Framework - Available Commands"
	@echo "=================================================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Installation and Setup
install: ## Install dependencies using uv
	@echo "📦 Installing dependencies with uv..."
	uv sync --extra dev --extra visualization --extra apple-silicon

install-dev: ## Install development dependencies
	@echo "🛠️  Installing development dependencies..."
	uv sync --extra dev --extra visualization --extra apple-silicon
	uv run pre-commit install

setup-dev: install-dev ## Complete development environment setup
	@echo "🔧 Setting up development environment..."
	mkdir -p logs outputs data/samples
	@echo "✅ Development environment ready!"

# Testing
test: ## Run all tests
	@echo "🧪 Running all tests..."
	python run_tests.py

test-fast: ## Run only fast tests
	@echo "⚡ Running fast tests only..."
	python run_tests.py --fast

test-integration: ## Run integration tests
	@echo "🔗 Running integration tests..."
	python run_tests.py --integration

test-benchmark: ## Run benchmark tests
	@echo "📊 Running benchmark tests..."
	python run_tests.py --benchmark

test-coverage: ## Run tests with coverage report
	@echo "📋 Running tests with coverage..."
	python run_tests.py --coverage
	@echo "📊 Coverage report: coverage_html/index.html"

test-parallel: ## Run tests in parallel
	@echo "🚀 Running tests in parallel..."
	python run_tests.py --parallel 4

test-verbose: ## Run tests with verbose output
	@echo "🔍 Running verbose tests..."
	python run_tests.py --verbose

# Code Quality
lint: ## Run linting with ruff
	@echo "🔍 Linting code with ruff..."
	uv run ruff check src/ tests/ --fix

format: ## Format code with black and isort
	@echo "✨ Formatting code..."
	uv run black src/ tests/ --line-length 100
	uv run isort src/ tests/ --profile black

type-check: ## Run type checking with mypy
	@echo "🔍 Type checking with mypy..."
	uv run mypy src/ --ignore-missing-imports

quality-check: lint format type-check ## Run all code quality checks
	@echo "✅ All quality checks completed!"

# Development Commands
clean: ## Clean temporary files and caches
	@echo "🧹 Cleaning temporary files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .coverage coverage_html/ coverage.xml
	rm -rf dist/ build/
	@echo "✅ Cleanup completed!"

# CLI Commands
validate-config: ## Validate default configuration
	@echo "✅ Validating configuration..."
	uv run python src/cli.py validate --config configs/default.yaml

info: ## Show framework information
	@echo "ℹ️  Framework Information:"
	uv run python src/cli.py info --config configs/default.yaml

# Training and Inference
train-demo: ## Run demo training (mock)
	@echo "🎯 Running demo training..."
	uv run python src/cli.py train \
		--config configs/default.yaml \
		--model microsoft/DialoGPT-medium \
		--epochs 1 \
		--batch-size 1

optimize-demo: ## Run demo hyperparameter optimization
	@echo "🔍 Running demo optimization..."
	uv run python src/cli.py optimize \
		--model microsoft/DialoGPT-medium \
		--data data/samples/ \
		--trials 3 \
		--max-epochs 1

serve-demo: ## Start demo inference server
	@echo "🌐 Starting demo server..."
	uv run python src/cli.py serve \
		--model-path outputs/best_model \
		--host 127.0.0.1 \
		--port 8000

generate-demo: ## Generate demo text
	@echo "✨ Generating demo text..."
	uv run python src/cli.py generate \
		--model-path outputs/best_model \
		--prompt "Hello, how are you?" \
		--max-length 50

# Development Utilities
jupyter: ## Start Jupyter for development
	@echo "📓 Starting Jupyter notebook..."
	uv run jupyter lab --ip=0.0.0.0 --port=8888 --allow-root

profile-memory: ## Run memory profiling
	@echo "🧠 Profiling memory usage..."
	uv run python -m memory_profiler src/cli.py info

profile-performance: ## Run performance profiling  
	@echo "⚡ Profiling performance..."
	uv run python -m cProfile -o profile_output.prof src/cli.py info
	@echo "📊 Profile saved to profile_output.prof"

# Docker Commands
docker-build: ## Build Docker image
	@echo "🐳 Building Docker image..."
	docker build -t lora-framework .

docker-run: ## Run Docker container
	@echo "🐳 Running Docker container..."
	docker run -p 8000:8000 lora-framework

docker-test: ## Run tests in Docker
	@echo "🐳 Running tests in Docker..."
	docker run --rm lora-framework python run_tests.py

# Documentation
build-docs: ## Build documentation
	@echo "📚 Building documentation..."
	mkdir -p docs/
	uv run python -c "
import os
from pathlib import Path
readme_content = Path('README.md').read_text()
with open('docs/index.html', 'w') as f:
    f.write(f'<html><body><pre>{readme_content}</pre></body></html>')
"
	@echo "📖 Documentation built in docs/"

serve-docs: build-docs ## Serve documentation locally
	@echo "📚 Serving documentation..."
	uv run python -m http.server 8080 -d docs/

# Deployment
package: clean ## Package for deployment
	@echo "📦 Packaging for deployment..."
	uv build
	@echo "✅ Package built in dist/"

publish: package ## Publish to PyPI (test)
	@echo "🚀 Publishing to TestPyPI..."
	uv run twine upload --repository testpypi dist/*

# Environment Management  
check-env: ## Check development environment
	@echo "🔍 Checking environment..."
	@echo "Python: $$(python --version)"
	@echo "UV: $$(uv --version)"
	@python -c "
try:
    import mlx.core as mx
    print('✅ MLX: Available')
    if hasattr(mx, 'metal'):
        print(f'🍎 Metal: {mx.metal.is_available()}')
except ImportError:
    print('❌ MLX: Not available')
"
	@python -c "
try:
    import torch
    print('✅ PyTorch: Available')
    if hasattr(torch.backends, 'mps'):
        print(f'🍎 MPS: {torch.backends.mps.is_available()}')
except ImportError:
    print('❌ PyTorch: Not available')
"

# Quick Start
quick-start: setup-dev validate-config test-fast ## Quick start for new developers
	@echo "🎉 Quick start completed!"
	@echo ""
	@echo "Next steps:"
	@echo "1. Run 'make train-demo' to test training"
	@echo "2. Run 'make test' to run full test suite"
	@echo "3. Check 'make help' for more commands"

# All-in-one commands
full-test: clean test-coverage quality-check ## Run comprehensive testing
	@echo "🏆 Full test suite completed!"

ci: clean install test-coverage quality-check ## Continuous Integration pipeline
	@echo "🤖 CI pipeline completed!"

# Default development workflow
dev: setup-dev test-fast lint ## Standard development workflow
	@echo "💻 Development workflow completed!"

# Show current status
status: ## Show project status
	@echo "📊 Project Status"
	@echo "================"
	@echo "📁 Directory: $$(pwd)"
	@echo "🐍 Python: $$(python --version)"
	@echo "📦 UV: $$(uv --version 2>/dev/null || echo 'Not installed')"
	@echo "🧪 Tests: $$(find tests/ -name '*.py' | wc -l) test files"
	@echo "📝 Source: $$(find src/ -name '*.py' | wc -l) source files"
	@echo "📋 Config: $$(ls configs/*.yaml | wc -l) config files"
	@echo "📊 Data: $$(find data/ -name '*' -type f 2>/dev/null | wc -l) data files"