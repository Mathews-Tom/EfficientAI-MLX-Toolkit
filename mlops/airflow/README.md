# Thermal-Aware Airflow Scheduling

Intelligent Airflow task scheduling for Apple Silicon that respects thermal constraints, memory pressure, and power mode.

## Quick Start

```python
from mlops.airflow import ThermalAwareScheduler, ThermalAwareTrainingOperator

# Create scheduler
scheduler = ThermalAwareScheduler()

# Check if task should run
decision = scheduler.should_run_task(
    task_id="train_model",
    thermal_threshold=2,
    memory_required_gb=16.0,
    priority="high"
)

# Create thermal-aware task
task = ThermalAwareTrainingOperator(
    task_id="train_llm",
    model_name="llama-7b",
    dataset_path="/data/train.jsonl",
    thermal_threshold=2,
    priority="high",
)
```

## Components

### ThermalAwareScheduler
Core scheduling logic that monitors thermal state and makes intelligent task scheduling decisions.

**Key Methods:**
- `should_run_task()`: Check if task should run based on thermal state
- `wait_for_thermal_clearance()`: Wait for thermal conditions to improve
- `get_scheduling_stats()`: Get real-time scheduling statistics
- `suggest_task_configuration()`: Get optimal task config based on thermal state

### Operators

**ThermalAwareMLXOperator**: General-purpose thermal-aware operator
- Checks thermal state before execution
- Applies throttling based on thermal conditions
- Logs thermal metrics to XCom

**ThermalAwareTrainingOperator**: Optimized for model training
- Higher thermal thresholds (default: 2)
- Higher memory requirements (default: 16GB)
- High priority execution
- Waits for thermal clearance

**ThermalAwareInferenceOperator**: Optimized for model inference
- Lower thermal thresholds (default: 1)
- Lower memory requirements (default: 8GB)
- Normal priority execution
- Fast retry cycles

### Configuration

**Task Profiles:**
- `training`: High resource, high priority (thermal=2, memory=16GB)
- `inference`: Moderate resource, normal priority (thermal=1, memory=8GB)
- `preprocessing`: Low resource, normal priority (thermal=1, memory=4GB)
- `evaluation`: Moderate resource, normal priority (thermal=1, memory=8GB)
- `deployment`: Critical priority, can run in any thermal state (thermal=3)

## Directory Structure

```
mlops/airflow/
├── __init__.py                    # Package exports
├── README.md                      # This file
├── thermal_scheduler.py           # Core scheduler logic
├── operators.py                   # Custom Airflow operators
├── config.py                      # Configuration classes
└── dags/
    ├── example_training_dag.py    # Training workflow example
    └── example_inference_dag.py   # Batch inference example
```

## Examples

### Training DAG

```python
from airflow import DAG
from mlops.airflow.operators import ThermalAwareTrainingOperator

with DAG("training_dag", ...) as dag:
    train = ThermalAwareTrainingOperator(
        task_id="train_model",
        model_name="llama-7b",
        dataset_path="/data/train.jsonl",
        thermal_threshold=2,
        priority="high",
        wait_for_clearance=True,
    )
```

### Inference DAG

```python
from mlops.airflow.operators import ThermalAwareInferenceOperator

with DAG("inference_dag", ...) as dag:
    inference = ThermalAwareInferenceOperator(
        task_id="run_inference",
        model_path="/models/llama-7b",
        input_data=prompts,
        thermal_threshold=1,
    )
```

## Testing

All components have comprehensive test coverage:

```bash
# Run all thermal-aware tests
uv run pytest mlops/tests/test_airflow_thermal_scheduler.py
uv run pytest mlops/tests/test_airflow_operators.py
uv run pytest mlops/tests/test_thermal_config.py

# Run with coverage
uv run pytest mlops/tests/test_airflow_*.py --cov=mlops/airflow
```

**Test Results:**
- 62 total tests
- 100% pass rate
- Comprehensive coverage of scheduler, operators, and configuration

## Documentation

Full documentation available at:
- [Thermal-Aware Scheduling Guide](../docs/thermal_aware_scheduling.md)
- [API Reference](thermal_scheduler.py)
- [Example DAGs](dags/)

## Features

- Real-time thermal monitoring (0-3 scale)
- Memory pressure detection
- Health score calculation
- Automatic task throttling
- Priority-based execution
- Clearance waiting
- Retry logic with exponential backoff
- XCom integration for metrics

## Best Practices

1. **Choose appropriate thermal thresholds**
   - Training: `thermal_threshold=2`
   - Inference: `thermal_threshold=1`
   - Critical: `thermal_threshold=3`

2. **Set accurate memory requirements**
   ```python
   memory_required_gb=16.0  # Realistic estimate
   ```

3. **Use priority levels wisely**
   - `normal`: Most tasks
   - `high`: Important training
   - `critical`: Must-complete tasks only

4. **Enable clearance waiting for long tasks**
   ```python
   wait_for_clearance=True
   clearance_timeout_seconds=600
   ```

## Integration with Apple Silicon Monitor

The thermal-aware scheduler integrates seamlessly with `AppleSiliconMonitor` (from MLOP-010) to track:
- Thermal state (0-3)
- Memory utilization
- CPU usage
- Power mode
- Health score

This integration ensures optimal task scheduling while preventing thermal throttling and system instability.
