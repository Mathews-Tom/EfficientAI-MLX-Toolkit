

# Apple Silicon Optimization Module

Centralized Apple Silicon detection, optimization, and monitoring for the MLOps infrastructure. This module consolidates all Apple Silicon-related functionality to avoid duplication across components.

## Overview

The `mlops.silicon` module provides:

- **Hardware Detection**: Comprehensive Apple Silicon chip detection with capabilities checking
- **Performance Optimization**: Configuration recommendations based on hardware capabilities
- **Real-time Monitoring**: Continuous metrics collection including memory, CPU, thermal, and power state
- **Integration Helpers**: Backward-compatible wrappers for easy migration

## Architecture

```
mlops/silicon/
├── __init__.py           # Public API exports
├── detector.py           # Hardware detection (AppleSiliconDetector)
├── metrics.py            # Metrics data structures (AppleSiliconMetrics)
├── monitor.py            # Real-time monitoring (AppleSiliconMonitor)
├── optimizer.py          # Configuration optimization (AppleSiliconOptimizer)
├── integration.py        # Backward-compatible helpers
└── README.md            # This file
```

## Quick Start

### Basic Detection

```python
from mlops.silicon import AppleSiliconDetector

# Detect hardware
detector = AppleSiliconDetector()

if detector.is_apple_silicon:
    info = detector.get_hardware_info()
    print(f"Chip: {info.chip_type} {info.chip_variant}")
    print(f"Memory: {info.memory_total_gb}GB")
    print(f"Cores: {info.core_count} ({info.performance_cores}P + {info.efficiency_cores}E)")
    print(f"MLX: {info.mlx_available}, MPS: {info.mps_available}, ANE: {info.ane_available}")
```

### Optimization Recommendations

```python
from mlops.silicon import AppleSiliconDetector, AppleSiliconOptimizer

# Get hardware info
detector = AppleSiliconDetector()
info = detector.get_hardware_info()

# Get optimization recommendations
optimizer = AppleSiliconOptimizer(info)

# For inference workload
inference_config = optimizer.get_optimal_config(workload_type="inference")
print(f"Workers: {inference_config.workers}")
print(f"Batch size: {inference_config.batch_size}")
print(f"Memory limit: {inference_config.memory_limit_gb}GB")
print(f"Use MLX: {inference_config.use_mlx}")

# For training workload
training_config = optimizer.get_training_config(memory_intensive=True)
print(f"Batch size: {training_config['batch_size']}")
print(f"Workers: {training_config['num_workers']}")
print(f"Memory limit: {training_config['memory_limit_gb']}GB")

# For deployment
deployment_config = optimizer.get_deployment_config()
print(f"Workers: {deployment_config['workers']}")
print(f"Max batch size: {deployment_config['max_batch_size']}")
```

### Real-time Monitoring

```python
from mlops.silicon import AppleSiliconMonitor

# Initialize monitor
monitor = AppleSiliconMonitor(project_name="my_project")

# Collect current metrics
metrics = monitor.collect()
print(f"CPU: {metrics.cpu_percent}%")
print(f"Memory: {metrics.memory_used_gb}GB / {metrics.memory_total_gb}GB")
print(f"Thermal: {metrics.thermal_state} (0=nominal, 3=critical)")
print(f"Power: {metrics.power_mode}")

# Check system health
health = monitor.check_health()
print(f"Health score: {health['score']}/100")
print(f"Throttling: {health['thermal_throttling']}")
print(f"Recommendations: {health['recommendations']}")
```

### MLFlow Integration

```python
from mlops.silicon import AppleSiliconMonitor
from mlops.client import MLOpsClient

# Initialize client and monitor
client = MLOpsClient.from_project("my_project")
monitor = AppleSiliconMonitor()

with client.start_run(run_name="training_run"):
    # Log training params
    client.log_params({"learning_rate": 0.001, "batch_size": 32})

    # Collect and log Apple Silicon metrics
    metrics = monitor.collect()
    client.log_apple_silicon_metrics(metrics.to_mlflow_metrics())

    # Training loop...
    for epoch in range(10):
        # Check health during training
        health = monitor.check_health()
        if health["thermal_throttling"]:
            print("WARNING: Thermal throttling detected!")

        # Log epoch metrics
        client.log_metrics({"epoch": epoch, "loss": 0.5})
```

## API Reference

### AppleSiliconDetector

Centralized hardware detection with caching.

**Methods:**

- `is_apple_silicon` (property): Returns `bool` indicating Apple Silicon
- `get_hardware_info()`: Returns `HardwareInfo` with complete hardware details
- `refresh()`: Re-detect hardware (updates dynamic properties)

**HardwareInfo attributes:**

- `chip_type`: Chip generation (M1, M2, M3, M4)
- `chip_variant`: Chip variant (Base, Pro, Max, Ultra)
- `memory_total_gb`: Total unified memory
- `core_count`: Total CPU cores
- `performance_cores`: Number of P-cores
- `efficiency_cores`: Number of E-cores
- `mlx_available`: MLX framework availability
- `mps_available`: MPS backend availability
- `ane_available`: Apple Neural Engine availability
- `thermal_state`: Current thermal state (0-3)
- `power_mode`: Power mode (low_power, normal, high_performance)

### AppleSiliconOptimizer

Configuration optimization based on hardware capabilities.

**Methods:**

- `get_optimal_config(workload_type, memory_intensive)`: Get optimal configuration
- `get_deployment_config()`: Get BentoML/Ray Serve deployment config
- `get_training_config(memory_intensive)`: Get training configuration

**OptimalConfig attributes:**

- `workers`: Recommended worker processes
- `batch_size`: Recommended batch size
- `memory_limit_gb`: Recommended memory limit
- `use_mlx`: Whether to use MLX
- `use_mps`: Whether to use MPS
- `use_ane`: Whether to use ANE
- `prefetch_batches`: Number of batches to prefetch
- `cpu_threads`: CPU threads per worker
- `recommendations`: List of optimization recommendations

### AppleSiliconMonitor

Real-time performance monitoring.

**Methods:**

- `collect()`: Collect current metrics (returns `AppleSiliconMetrics`)
- `is_apple_silicon()`: Check if running on Apple Silicon
- `get_hardware_summary()`: Get hardware information dict
- `check_health()`: Check system health with recommendations

**AppleSiliconMetrics attributes:**

- `timestamp`: When metrics were collected
- `chip_type`, `chip_variant`: Chip information
- `memory_*`: Memory metrics (total, used, available, utilization)
- `cpu_percent`: CPU utilization
- `thermal_state`: Thermal state (0-3)
- `power_mode`: Power mode
- `mlx_available`, `mps_available`, `ane_available`: Framework availability

**Health check returns:**

- `score`: Health score (0-100)
- `thermal_throttling`: Whether system is throttling
- `memory_constrained`: Whether memory is constrained
- `recommendations`: List of recommendations

## Integration Patterns

### BentoML Integration

```python
from mlops.silicon.integration import get_optimal_config_for_bentoml

config = get_optimal_config_for_bentoml(project_name="my_model")

# Use config for BentoML service
bentoml_config = {
    "service": {
        "name": "my_service",
        "workers": config["workers"],
    },
    "runner": {
        "batching": {
            "max_batch_size": config["max_batch_size"],
        },
    },
}
```

### Training Loop Integration

```python
from mlops.silicon.integration import get_optimal_config_for_training

config = get_optimal_config_for_training(memory_intensive=True)

# Configure data loader
train_loader = DataLoader(
    dataset,
    batch_size=config["batch_size"],
    num_workers=config["num_workers"],
    prefetch_factor=config["prefetch_factor"],
    persistent_workers=config["persistent_workers"],
)

# Select backend
if config["use_mlx"]:
    import mlx.core as mx
    device = mx.gpu
elif config["use_mps"]:
    import torch
    device = torch.device("mps")
```

### Monitoring Dashboard

```python
from mlops.silicon import AppleSiliconMonitor
import time

monitor = AppleSiliconMonitor()

while True:
    metrics = monitor.collect()
    health = monitor.check_health()

    print(f"CPU: {metrics.cpu_percent}% | "
          f"Memory: {metrics.memory_utilization_percent}% | "
          f"Health: {health['score']}/100")

    if health["thermal_throttling"]:
        print("⚠️  Thermal throttling - reducing workload")

    time.sleep(5)
```

## Best Practices

### 1. Use Centralized Detection

**Do:**
```python
from mlops.silicon import AppleSiliconDetector

detector = AppleSiliconDetector()  # Centralized, cached
info = detector.get_hardware_info()
```

**Don't:**
```python
import platform

# Duplicated detection logic
if platform.system() == "Darwin" and platform.machine() == "arm64":
    # Custom detection...
```

### 2. Apply Optimization Recommendations

**Do:**
```python
from mlops.silicon import AppleSiliconDetector, AppleSiliconOptimizer

detector = AppleSiliconDetector()
optimizer = AppleSiliconOptimizer(detector.get_hardware_info())
config = optimizer.get_optimal_config(workload_type="training")

# Use recommended config
batch_size = config.batch_size
workers = config.workers
```

**Don't:**
```python
# Hardcoded values
batch_size = 32  # May not be optimal for hardware
workers = 4      # May cause thermal throttling
```

### 3. Monitor System Health

**Do:**
```python
from mlops.silicon import AppleSiliconMonitor

monitor = AppleSiliconMonitor()

for epoch in range(100):
    health = monitor.check_health()

    if health["thermal_throttling"]:
        # Adjust workload
        reduce_batch_size()

    train_epoch()
```

**Don't:**
```python
# Ignore thermal state
for epoch in range(100):
    train_epoch()  # May throttle and slow down
```

### 4. Log Metrics for Analysis

**Do:**
```python
from mlops.silicon import AppleSiliconMonitor

monitor = AppleSiliconMonitor()

with mlops_client.start_run():
    metrics = monitor.collect()
    mlops_client.log_apple_silicon_metrics(metrics.to_mlflow_metrics())
```

**Don't:**
```python
# Missing hardware metrics
with mlops_client.start_run():
    mlops_client.log_metrics({"loss": 0.5})
    # No visibility into hardware state
```

## Backward Compatibility

The module provides backward-compatible helpers for existing code:

```python
# Old code using monitoring/evidently module
from mlops.monitoring.evidently.apple_silicon_metrics import (
    AppleSiliconMetricsCollector
)

collector = AppleSiliconMetricsCollector(project_name="test")
metrics = collector.collect()
```

Can be migrated to:

```python
# New centralized module
from mlops.silicon import AppleSiliconMonitor

monitor = AppleSiliconMonitor(project_name="test")
metrics = monitor.collect()
```

Or use compatibility wrapper:

```python
from mlops.silicon.integration import AppleSiliconMetricsCollector

# Same interface as before
collector = AppleSiliconMetricsCollector(project_name="test")
metrics = collector.collect()
```

## Performance Considerations

1. **Detection Caching**: Hardware detection is cached on initialization. Call `detector.refresh()` only when needed.

2. **Monitoring Frequency**: Collect metrics at reasonable intervals (5-10 seconds) to avoid overhead.

3. **Memory Overhead**: Monitor uses minimal memory (~1MB). Safe for continuous monitoring.

4. **CPU Overhead**: Metrics collection takes <10ms on average.

## Troubleshooting

### "Not running on Apple Silicon" but I have M1/M2/M3

Check system detection:
```python
from mlops.silicon import AppleSiliconDetector

detector = AppleSiliconDetector()
info = detector.get_hardware_info()
print(f"System: {info.system}, Machine: {info.machine}")
```

### "MLX not available" but it's installed

Verify MLX installation:
```python
from mlops.silicon import AppleSiliconDetector

detector = AppleSiliconDetector()
info = detector.get_hardware_info()
print(f"MLX available: {info.mlx_available}")

# Try importing directly
import mlx.core as mx
print(mx.array([1, 2, 3]))
```

### Thermal throttling warnings

1. Check current state:
```python
monitor = AppleSiliconMonitor()
health = monitor.check_health()
print(f"Thermal state: {health['thermal_state']}")
```

2. Apply optimization:
```python
optimizer = AppleSiliconOptimizer(detector.get_hardware_info())
config = optimizer.get_optimal_config()
# Use recommended workers/batch size
```

## Testing

Run tests:
```bash
# All tests
uv run pytest mlops/tests/test_silicon_*.py

# Specific component
uv run pytest mlops/tests/test_silicon_detector.py -v

# With coverage
uv run pytest mlops/tests/test_silicon_*.py --cov=mlops.silicon
```

## Support

For issues or questions:

1. Check hardware compatibility with `AppleSiliconDetector`
2. Review optimization recommendations from `AppleSiliconOptimizer`
3. Monitor system health with `AppleSiliconMonitor`
4. Check MLOps documentation for integration examples
