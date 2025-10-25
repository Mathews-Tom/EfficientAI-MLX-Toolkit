# Thermal-Aware Airflow Scheduling for Apple Silicon

## Overview

The thermal-aware scheduling system integrates with Apache Airflow to provide intelligent task scheduling that respects Apple Silicon thermal constraints, memory pressure, and power mode. This ensures optimal performance while preventing thermal throttling and system instability.

## Features

- **Real-time Thermal Monitoring**: Continuous monitoring of Apple Silicon thermal state (0-3 scale)
- **Intelligent Task Scheduling**: Tasks are scheduled based on thermal conditions, memory availability, and health score
- **Automatic Throttling**: Tasks run with reduced resource usage during high thermal states
- **Priority-Based Execution**: Critical tasks can bypass thermal constraints with heavy throttling
- **Clearance Waiting**: Tasks can wait for thermal conditions to improve before execution
- **Configurable Profiles**: Predefined task profiles for training, inference, preprocessing, etc.

## Architecture

### Components

1. **ThermalAwareScheduler**: Core scheduling logic with thermal monitoring
2. **ThermalAwareOperatorMixin**: Base mixin for thermal-aware operators
3. **ThermalAwareMLXOperator**: General-purpose thermal-aware operator
4. **ThermalAwareTrainingOperator**: Optimized for model training workloads
5. **ThermalAwareInferenceOperator**: Optimized for model inference workloads
6. **Configuration System**: Task profiles and thermal thresholds

### Integration Flow

```
┌─────────────────────────────────────────────────────────────┐
│                      Airflow Task                            │
│                                                               │
│  ┌────────────────────────────────────────────────────────┐  │
│  │         ThermalAwareOperator                           │  │
│  │                                                          │  │
│  │  1. Pre-Execute Check                                   │  │
│  │     ├─> Check thermal state via AppleSiliconMonitor    │  │
│  │     ├─> Check memory availability                       │  │
│  │     ├─> Check health score                              │  │
│  │     └─> Determine throttle level                        │  │
│  │                                                          │  │
│  │  2. Execute Task (with throttling if needed)            │  │
│  │                                                          │  │
│  │  3. Post-Execute Logging                                │  │
│  │     └─> Log thermal metrics to XCom                     │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                               │
│  ┌────────────────────────────────────────────────────────┐  │
│  │         ThermalAwareScheduler                          │  │
│  │                                                          │  │
│  │  • Thermal state: 0 (nominal) → 3 (critical)           │  │
│  │  • Memory utilization monitoring                        │  │
│  │  • Health score calculation (0-100)                     │  │
│  │  • Retry delay calculation                              │  │
│  │  • Throttle level determination                         │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                               │
│  ┌────────────────────────────────────────────────────────┐  │
│  │         AppleSiliconMonitor                            │  │
│  │                                                          │  │
│  │  • Thermal state detection (0-3)                        │  │
│  │  • Memory metrics (vm_stat)                             │  │
│  │  • CPU utilization                                       │  │
│  │  • Power mode                                            │  │
│  └────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Thermal States

The system uses a 0-3 thermal state scale:

| State | Level | Description | Task Behavior |
|-------|-------|-------------|---------------|
| 0 | Nominal | System running cool | Full speed (100%) |
| 1 | Fair | Slight warmth | Slight throttle (90%) |
| 2 | Serious | High temperature | Moderate throttle (70%) |
| 3 | Critical | Thermal throttling | Heavy throttle (50%) |

## Usage Guide

### Basic Usage

```python
from mlops.airflow import ThermalAwareScheduler, create_thermal_aware_task

# Create scheduler
scheduler = ThermalAwareScheduler()

# Check if task should run
decision = scheduler.should_run_task(
    task_id="train_model",
    thermal_threshold=2,  # Max serious
    memory_required_gb=16.0,
    priority="high"
)

if decision.should_run:
    print(f"Running with throttle: {decision.throttle_level}")
else:
    print(f"Delayed: {decision.reason}")
```

### Creating Thermal-Aware Tasks

#### Simple Task

```python
from mlops.airflow.operators import create_thermal_aware_task

def my_function():
    # Your task logic
    return "success"

task = create_thermal_aware_task(
    task_id="my_task",
    python_callable=my_function,
    thermal_threshold=2,
    priority="normal",
    memory_required_gb=8.0,
)
```

#### Training Task

```python
from mlops.airflow.operators import ThermalAwareTrainingOperator

task = ThermalAwareTrainingOperator(
    task_id="train_llm",
    model_name="llama-7b",
    dataset_path="/data/train.jsonl",
    output_path="/models/output",
    thermal_threshold=2,
    priority="high",
    memory_required_gb=16.0,
    wait_for_clearance=True,
    clearance_timeout_seconds=600,
    training_config={
        "epochs": 3,
        "batch_size": 32,
        "learning_rate": 1e-4,
    },
)
```

#### Inference Task

```python
from mlops.airflow.operators import ThermalAwareInferenceOperator

task = ThermalAwareInferenceOperator(
    task_id="run_inference",
    model_path="/models/llama-7b",
    input_data=["prompt1", "prompt2"],
    thermal_threshold=1,
    priority="normal",
    memory_required_gb=8.0,
    inference_config={
        "batch_size": 64,
        "max_tokens": 512,
    },
)
```

### Complete DAG Example

```python
from datetime import datetime, timedelta
from airflow import DAG
from mlops.airflow.operators import (
    ThermalAwareTrainingOperator,
    ThermalAwareInferenceOperator,
    create_thermal_aware_task,
)

default_args = {
    "owner": "mlops",
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    "thermal_aware_training",
    default_args=default_args,
    schedule_interval=timedelta(days=1),
    start_date=datetime(2024, 1, 1),
    catchup=False,
) as dag:

    # Data preparation (low thermal requirements)
    prep = create_thermal_aware_task(
        task_id="prepare_data",
        python_callable=prepare_dataset,
        thermal_threshold=1,
        priority="normal",
    )

    # Model training (high thermal requirements)
    train = ThermalAwareTrainingOperator(
        task_id="train_model",
        model_name="llama-7b",
        dataset_path="/data/train.jsonl",
        thermal_threshold=2,
        priority="high",
        wait_for_clearance=True,
    )

    # Model validation (moderate requirements)
    validate = ThermalAwareInferenceOperator(
        task_id="validate_model",
        model_path="/models/output",
        input_data=validation_prompts,
        thermal_threshold=1,
    )

    prep >> train >> validate
```

## Task Configuration

### Task Profiles

Predefined profiles for common task types:

```python
from mlops.airflow.config import create_task_config

# Training profile
config = create_task_config("train_task", profile="training")
# thermal_threshold=2, memory=16GB, priority=high

# Inference profile
config = create_task_config("inference_task", profile="inference")
# thermal_threshold=1, memory=8GB, priority=normal

# Preprocessing profile
config = create_task_config("prep_task", profile="preprocessing")
# thermal_threshold=1, memory=4GB, priority=normal

# Deployment profile
config = create_task_config("deploy_task", profile="deployment")
# thermal_threshold=3, memory=2GB, priority=critical
```

### Custom Configuration

```python
from mlops.airflow.config import create_task_config

config = create_task_config(
    "custom_task",
    profile="training",  # Start with training profile
    thermal_threshold=1,  # Override to lower threshold
    memory_required_gb=32.0,  # Override memory
)
```

## Advanced Features

### Waiting for Thermal Clearance

Tasks can wait for thermal conditions to improve:

```python
task = ThermalAwareTrainingOperator(
    task_id="train_model",
    model_name="llama-7b",
    dataset_path="/data/train.jsonl",
    wait_for_clearance=True,  # Enable waiting
    clearance_timeout_seconds=600,  # Wait up to 10 minutes
    thermal_threshold=2,
)
```

### Priority-Based Execution

Critical tasks bypass thermal constraints:

```python
# Critical task runs even during thermal throttling
task = create_thermal_aware_task(
    task_id="critical_deploy",
    python_callable=deploy_model,
    thermal_threshold=3,  # Can run in critical state
    priority="critical",  # Bypass health checks
)
```

### Dynamic Configuration Suggestions

Get optimal configuration based on current thermal state:

```python
scheduler = ThermalAwareScheduler()
suggestion = scheduler.suggest_task_configuration("training")

print(f"Batch size: {suggestion['config']['batch_size']}")
print(f"Workers: {suggestion['config']['workers']}")
print(f"Memory: {suggestion['config']['memory_gb']}GB")
print(f"Throttle: {suggestion['throttle_level']}")
```

### Monitoring and Stats

Get real-time scheduling statistics:

```python
scheduler = ThermalAwareScheduler()
stats = scheduler.get_scheduling_stats()

print(f"Thermal state: {stats['thermal_state']}")
print(f"Health score: {stats['health_score']}")
print(f"Memory utilization: {stats['memory_utilization']}%")
print(f"Can schedule normal: {stats['can_schedule_normal']}")
print(f"Recommendations: {stats['recommendations']}")
```

## Configuration Reference

### ThermalSchedulingConfig

```python
from mlops.airflow.config import ThermalSchedulingConfig

config = ThermalSchedulingConfig(
    thermal_threshold_nominal=0,       # Nominal threshold
    thermal_threshold_fair=1,          # Fair threshold
    thermal_threshold_serious=2,       # Serious threshold
    thermal_threshold_critical=3,      # Critical threshold
    memory_threshold_percent=85.0,     # Memory pressure threshold
    min_health_score=40.0,             # Minimum health to run
    clearance_timeout_seconds=300,     # Default clearance timeout
    check_interval_seconds=10,         # Thermal check interval
)
```

### TaskConfig

```python
from mlops.airflow.config import TaskConfig

config = TaskConfig(
    task_id="my_task",
    thermal_threshold=2,               # Max thermal state
    memory_required_gb=16.0,           # Memory requirement
    priority="high",                   # Task priority
    retry_on_thermal=True,             # Retry on thermal throttle
    wait_for_clearance=True,           # Wait for clearance
    max_retries=3,                     # Max retry attempts
    retry_delay_seconds=60,            # Retry delay
)
```

## Best Practices

### 1. Choose Appropriate Thermal Thresholds

- **Training tasks**: `thermal_threshold=2` (requires good cooling)
- **Inference tasks**: `thermal_threshold=1` (can run in most conditions)
- **Preprocessing**: `thermal_threshold=1` (lightweight)
- **Critical deployments**: `thermal_threshold=3` (must complete)

### 2. Set Realistic Memory Requirements

```python
# Accurate memory requirements prevent OOM
task = ThermalAwareTrainingOperator(
    task_id="train_large_model",
    memory_required_gb=32.0,  # Realistic estimate
    ...
)
```

### 3. Use Priority Levels Wisely

```python
# Normal priority for most tasks
normal_task = create_thermal_aware_task(..., priority="normal")

# High priority for important training
train_task = create_thermal_aware_task(..., priority="high")

# Critical only for must-complete tasks
deploy_task = create_thermal_aware_task(..., priority="critical")
```

### 4. Enable Clearance Waiting for Long Tasks

```python
# Training can wait for good conditions
train = ThermalAwareTrainingOperator(
    wait_for_clearance=True,
    clearance_timeout_seconds=600,
    ...
)

# Quick inference doesn't wait
inference = ThermalAwareInferenceOperator(
    wait_for_clearance=False,
    ...
)
```

### 5. Monitor Thermal Metrics

```python
# Access thermal metrics from XCom
thermal_metrics = context['task_instance'].xcom_pull(
    key='thermal_metrics',
    task_ids='previous_task'
)

print(f"Previous task thermal: {thermal_metrics['thermal_state']}")
```

## Troubleshooting

### Tasks Not Running

**Problem**: Tasks stuck waiting for thermal clearance

**Solution**:
- Reduce `thermal_threshold` if tasks are too strict
- Increase `clearance_timeout_seconds`
- Use higher priority for important tasks
- Check system cooling

### Thermal Throttling

**Problem**: System consistently in thermal throttling state

**Solution**:
- Reduce concurrent task count
- Lower batch sizes in training config
- Enable `wait_for_clearance` for heavy tasks
- Improve system cooling

### Memory Errors

**Problem**: Tasks fail with OOM despite thermal checks

**Solution**:
- Set accurate `memory_required_gb` for tasks
- Lower `memory_threshold_percent` to leave more headroom
- Reduce batch sizes in task configs

### Performance Issues

**Problem**: Tasks running slower than expected

**Solution**:
- Check `throttle_level` in scheduling decisions
- Review thermal state trends
- Consider task scheduling during cooler times
- Optimize task resource requirements

## API Reference

See full API documentation:
- [ThermalAwareScheduler API](../mlops/airflow/thermal_scheduler.py)
- [Operators API](../mlops/airflow/operators.py)
- [Configuration API](../mlops/airflow/config.py)

## Examples

Full example DAGs available at:
- [Training DAG](../mlops/airflow/dags/example_training_dag.py)
- [Batch Inference DAG](../mlops/airflow/dags/example_inference_dag.py)

## Support

For issues or questions:
1. Check thermal state with `scheduler.get_scheduling_stats()`
2. Review task logs for thermal decisions
3. Adjust thermal thresholds and priorities
4. Consult Apple Silicon monitoring docs
