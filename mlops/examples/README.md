# MLOps Example Workflows and Templates

Complete, production-ready examples for common MLOps workflows.

## Available Examples

1. **[Training Workflow](training_workflow.py)** - Complete training pipeline with tracking
2. **[Deployment Workflow](deployment_workflow.py)** - Model packaging and deployment
3. **[Monitoring Workflow](monitoring_workflow.py)** - Production monitoring and alerting
4. **[Data Versioning](data_versioning.py)** - Dataset and model versioning
5. **[Hyperparameter Tuning](hyperparameter_tuning.py)** - MLFlow-tracked experiment tuning
6. **[A/B Testing](ab_testing.py)** - Model comparison and validation
7. **[CI/CD Pipeline](ci_cd_example.py)** - Automated model deployment pipeline

## Quick Start

```bash
# Run an example
cd mlops/examples
uv run python training_workflow.py

# With custom config
uv run python training_workflow.py --config custom_config.yaml

# Run all examples
for example in *.py; do
    uv run python "$example" --dry-run
done
```

## Example Structure

Each example includes:
- **Purpose**: What the example demonstrates
- **Prerequisites**: Required setup and dependencies
- **Configuration**: Configurable parameters
- **Usage**: How to run the example
- **Output**: Expected results and artifacts
- **Next Steps**: How to extend the example

## Integration with Toolkit

All examples use the MLOps client for seamless integration:

```python
from mlops.client.mlops_client import MLOpsClient

# Auto-configured for your project
client = MLOpsClient(project_namespace="my-project")

# Use in your workflow
with client.start_run(run_name="experiment-001"):
    # Your training code here
    pass
```

## Customization

Modify examples for your use case:
1. Copy example to your project
2. Update configuration
3. Adapt training/inference code
4. Run and iterate

## Testing Examples

```bash
# Validate examples
uv run pytest mlops/examples/tests/ -v

# Run in dry-run mode
uv run python training_workflow.py --dry-run

# Check generated artifacts
ls mlops/workspace/your-project/
```

## Templates

Pre-configured templates for common scenarios:
- `template_training.py` - Training pipeline template
- `template_inference.py` - Inference service template
- `template_batch.py` - Batch processing template
- `template_streaming.py` - Streaming inference template

## Contributing

Add your own examples:
1. Create new file following naming convention
2. Include comprehensive docstrings
3. Add configuration options
4. Write tests
5. Update this README
