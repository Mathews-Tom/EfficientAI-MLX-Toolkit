# Evidently Monitoring Dashboard Usage Guide

## Overview

The Evidently monitoring infrastructure provides comprehensive model monitoring for all toolkit projects with:

- **Data Drift Detection**: Monitor input distribution changes
- **Performance Monitoring**: Track model accuracy, precision, recall, F1, latency
- **Apple Silicon Metrics**: Hardware-specific metrics (MPS, MLX, thermal state)
- **Alert Management**: Automated alerts for drift and degradation
- **Retraining Suggestions**: Automated recommendations based on monitoring results

## Quick Start

### Basic Monitoring Setup

```python
from mlops.monitoring import create_monitor
import pandas as pd

# Create monitor for your project
monitor = create_monitor("lora-finetuning-mlx")

# Set reference data (training or baseline data)
reference_data = pd.read_csv("train_data.csv")
monitor.set_reference_data(
    reference_data,
    target_column="label",
    prediction_column="prediction"
)

# Monitor current predictions
current_data = pd.read_csv("production_data.csv")
results = monitor.monitor(
    current_data,
    target_column="label",
    prediction_column="prediction",
    latency_ms=45.0,
    memory_mb=512.0
)

# Check results
if results["retraining_suggested"]:
    print("Retraining recommended!")
    print(f"Drift detected: {results['drift_report']['dataset_drift']}")
    print(f"Performance degraded: {results['performance_metrics']['degraded']}")
```

## Components

### 1. Drift Detection

Monitor data distribution changes:

```python
from mlops.monitoring.evidently import DriftDetector, DriftDetectionConfig

# Configure drift detection
config = DriftDetectionConfig(
    stattest="wasserstein",  # Statistical test
    stattest_threshold=0.1,  # Threshold for drift detection
    drift_share=0.5,  # Share of drifted features to trigger alert
)

# Create detector
detector = DriftDetector(
    project_name="my-project",
    config=config,
)

# Set reference data
detector.set_reference_data(reference_df)

# Detect drift
report = detector.detect_drift(current_df)

print(f"Dataset drift: {report.dataset_drift}")
print(f"Drifted features: {report.drifted_features}")
print(f"Drift share: {report.drift_share:.2%}")
```

### 2. Performance Monitoring

Track model performance over time:

```python
from mlops.monitoring.evidently import PerformanceMonitor, PerformanceThresholds

# Set thresholds
thresholds = PerformanceThresholds(
    accuracy_threshold=0.85,
    precision_threshold=0.80,
    recall_threshold=0.80,
    f1_threshold=0.80,
    latency_threshold_ms=100.0,
    memory_threshold_mb=2048.0,
)

# Create monitor
monitor = PerformanceMonitor(
    project_name="my-project",
    task_type="classification",  # or "regression"
    thresholds=thresholds,
)

# Set reference data
monitor.set_reference_data(
    reference_data,
    target_column="target",
    prediction_column="prediction"
)

# Monitor performance
metrics = monitor.monitor_performance(
    current_data,
    target_column="target",
    prediction_column="prediction",
    latency_ms=75.0,
    memory_mb=1024.0,
)

if metrics.degraded:
    print("Performance degradation detected!")
    for reason in metrics.degradation_reasons:
        print(f"  - {reason}")
```

### 3. Apple Silicon Metrics

Collect Apple Silicon-specific metrics:

```python
from mlops.monitoring.evidently import AppleSiliconMetricsCollector

# Create collector
collector = AppleSiliconMetricsCollector(project_name="my-project")

# Collect metrics
metrics = collector.collect()

print(f"Chip type: {metrics.chip_type}")
print(f"Memory usage: {metrics.memory_used_gb:.2f} GB / {metrics.memory_total_gb:.2f} GB")
print(f"MLX available: {metrics.mlx_available}")
print(f"MPS available: {metrics.mps_available}")
print(f"ANE available: {metrics.ane_available}")
print(f"Thermal state: {metrics.thermal_state}")
print(f"CPU usage: {metrics.cpu_percent:.1f}%")
```

### 4. Alert Management

Configure and manage alerts:

```python
from mlops.monitoring.evidently import AlertManager, AlertConfig, AlertSeverity, AlertType

# Configure alerts
config = AlertConfig(
    enabled=True,
    drift_threshold=0.5,
    performance_degradation_enabled=True,
    apple_silicon_monitoring_enabled=True,
    notification_channels=["log", "email"],  # Custom channels can be added
)

# Create manager
manager = AlertManager(
    project_name="my-project",
    config=config,
)

# Create custom alert
alert = manager.create_alert(
    alert_type=AlertType.THRESHOLD_EXCEEDED,
    severity=AlertSeverity.WARNING,
    title="Memory usage high",
    message="Memory usage exceeded 90%",
    metadata={"memory_percent": 92.5},
)

# Get all unresolved alerts
unresolved_alerts = manager.get_all_alerts(unresolved_only=True)

# Acknowledge and resolve alerts
manager.acknowledge_alert(alert.alert_id)
manager.resolve_alert(alert.alert_id)
```

### Custom Notification Channels

Add custom notification handlers:

```python
def email_handler(alert):
    """Send email notification"""
    send_email(
        subject=f"Alert: {alert.title}",
        body=alert.message,
        to="team@example.com"
    )

def slack_handler(alert):
    """Send Slack notification"""
    post_to_slack(
        channel="#ml-alerts",
        message=f"{alert.severity.value.upper()}: {alert.title}\n{alert.message}"
    )

# Register handlers
manager.register_notification_handler("email", email_handler)
manager.register_notification_handler("slack", slack_handler)

# Update config to use new channels
config.notification_channels = ["log", "email", "slack"]
```

## Integration with Projects

### LoRA Fine-tuning Example

```python
from mlops.monitoring import create_monitor
import pandas as pd

# Create monitor for LoRA project
monitor = create_monitor("lora-finetuning-mlx")

# Set reference data from training
train_predictions = pd.read_csv("outputs/train_predictions.csv")
monitor.set_reference_data(
    train_predictions,
    target_column="label",
    prediction_column="prediction"
)

# Monitor production predictions
production_data = pd.read_csv("production_predictions.csv")
results = monitor.monitor(
    production_data,
    target_column="label",
    prediction_column="prediction",
    latency_ms=inference_latency,
    memory_mb=memory_usage,
)

# Check for retraining trigger
if results["retraining_suggested"]:
    # Trigger retraining workflow
    trigger_retraining_job(
        project="lora-finetuning-mlx",
        reason="drift_detected" if results["drift_report"]["dataset_drift"] else "performance_degraded"
    )
```

### Model Compression Example

```python
# Monitor compression quality over time
monitor = create_monitor("model-compression-mlx")

# Set reference from original model
reference_predictions = get_original_model_predictions()
monitor.set_reference_data(
    reference_predictions,
    target_column="label",
    prediction_column="prediction"
)

# Monitor compressed model
compressed_predictions = get_compressed_model_predictions()
results = monitor.monitor(
    compressed_predictions,
    target_column="label",
    prediction_column="prediction",
    latency_ms=compressed_latency,
    memory_mb=compressed_memory,
)

# Validate compression didn't degrade quality
if results["performance_metrics"]["degraded"]:
    print("Compression quality issue detected!")
    for reason in results["performance_metrics"]["degradation_reasons"]:
        print(f"  - {reason}")
```

## Configuration Files

### YAML Configuration

```yaml
# config/monitoring.yaml
evidently:
  project_name: "my-project"
  workspace_path: "mlops/monitoring/workspace"
  monitoring_enabled: true
  drift_detection_enabled: true
  performance_monitoring_enabled: true
  apple_silicon_metrics_enabled: true
  alert_enabled: true
  dashboard_port: 8000
  retention_days: 30

drift_detection:
  stattest: "wasserstein"
  stattest_threshold: 0.1
  drift_share: 0.5
  window_size: 1000

performance_thresholds:
  accuracy_threshold: 0.85
  precision_threshold: 0.80
  recall_threshold: 0.80
  f1_threshold: 0.80
  latency_threshold_ms: 100.0
  memory_threshold_mb: 2048.0

alerts:
  enabled: true
  drift_threshold: 0.5
  performance_degradation_enabled: true
  apple_silicon_monitoring_enabled: true
  notification_channels:
    - log
    - email
  alert_retention_days: 30
```

### Loading Configuration

```python
from mlops.monitoring.evidently.config import EvidentlyConfig, DriftDetectionConfig, PerformanceThresholds
from mlops.monitoring.evidently.alert_manager import AlertConfig
import yaml

# Load config
with open("config/monitoring.yaml") as f:
    config = yaml.safe_load(f)

# Create components
evidently_config = EvidentlyConfig.from_dict(config["evidently"])
drift_config = DriftDetectionConfig.from_dict(config["drift_detection"])
performance_thresholds = PerformanceThresholds.from_dict(config["performance_thresholds"])
alert_config = AlertConfig.from_dict(config["alerts"])

# Create monitor
monitor = EvidentlyMonitor(
    project_name=evidently_config.project_name,
    config=evidently_config,
    drift_config=drift_config,
    performance_thresholds=performance_thresholds,
    alert_config=alert_config,
)
```

## Dashboard

### Starting the Dashboard

```bash
# Start Evidently dashboard
evidently ui --workspace mlops/monitoring/workspace --host 0.0.0.0 --port 8000
```

### Programmatic Dashboard Access

```python
from mlops.monitoring.evidently.dashboard import start_dashboard, get_dashboard_url

# Start dashboard
start_dashboard(
    workspace_path="mlops/monitoring/workspace",
    port=8000
)

# Get dashboard URL
url = get_dashboard_url(host="localhost", port=8000)
print(f"Dashboard available at: {url}")
```

## Best Practices

### 1. Reference Data Selection

- Use representative training data as reference
- Update reference data periodically (e.g., monthly)
- Ensure reference data quality and balance

### 2. Threshold Configuration

- Start with conservative thresholds
- Adjust based on business requirements
- Monitor false positive rate

### 3. Alert Management

- Review alerts regularly
- Acknowledge and resolve promptly
- Tune notification channels based on severity

### 4. Monitoring Frequency

- Real-time monitoring for critical models
- Batch monitoring for less critical models
- Balance monitoring overhead vs. detection speed

### 5. Apple Silicon Optimization

- Monitor MPS availability for GPU acceleration
- Track thermal state during heavy inference
- Optimize based on unified memory usage

## Troubleshooting

### Issue: High False Positive Drift Alerts

**Solution**: Adjust drift detection thresholds

```python
config = DriftDetectionConfig(
    stattest_threshold=0.2,  # Increase threshold
    drift_share=0.7,  # Require more features to drift
)
```

### Issue: Performance Monitoring Shows Unexpected Degradation

**Solution**: Verify reference data quality

```python
# Check reference data distribution
reference_data = monitor.performance_monitor.get_reference_data()
print(reference_data.describe())

# Update reference data if needed
monitor.set_reference_data(new_reference_data, "target", "prediction")
```

### Issue: Alerts Not Being Sent

**Solution**: Check alert configuration

```python
# Verify alert manager is enabled
status = monitor.get_monitoring_status()
print(f"Alerts enabled: {status['alert_enabled']}")

# Check notification channels
print(f"Channels: {manager.config.notification_channels}")
```

## API Reference

See individual component documentation:

- [`EvidentlyMonitor`](../monitoring/evidently/monitor.py) - Unified monitoring interface
- [`DriftDetector`](../monitoring/evidently/drift_detector.py) - Data drift detection
- [`PerformanceMonitor`](../monitoring/evidently/performance_monitor.py) - Performance monitoring
- [`AppleSiliconMetricsCollector`](../monitoring/evidently/apple_silicon_metrics.py) - Hardware metrics
- [`AlertManager`](../monitoring/evidently/alert_manager.py) - Alert management

## Examples

See [examples/monitoring/](../../examples/monitoring/) for complete working examples:

- `basic_monitoring.py` - Simple monitoring setup
- `advanced_drift_detection.py` - Advanced drift detection patterns
- `performance_tracking.py` - Performance monitoring over time
- `alert_integration.py` - Custom alert handlers
- `dashboard_setup.py` - Dashboard configuration
