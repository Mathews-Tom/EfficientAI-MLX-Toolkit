"""Example Batch Inference DAG with Thermal-Aware Scheduling

This DAG demonstrates thermal-aware batch inference processing with
dynamic parallelism based on thermal conditions.
"""

from datetime import datetime, timedelta

try:
    from airflow import DAG
    from airflow.operators.python import BranchPythonOperator
except ImportError:
    class DAG:
        def __init__(self, *args, **kwargs):
            pass

    class BranchPythonOperator:
        def __init__(self, *args, **kwargs):
            pass

from mlops.airflow.operators import ThermalAwareInferenceOperator
from mlops.airflow.thermal_scheduler import ThermalAwareScheduler


default_args = {
    "owner": "mlops",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=2),
}


def check_thermal_state(**context):
    """Check thermal state and decide parallelism level"""
    scheduler = ThermalAwareScheduler()
    stats = scheduler.get_scheduling_stats()

    thermal_state = stats["thermal_state"]

    if thermal_state == 0:  # Nominal
        return "parallel_high"
    elif thermal_state == 1:  # Fair
        return "parallel_medium"
    else:  # Serious or Critical
        return "sequential"


def load_batch_data(batch_id: int, **context):
    """Load batch data for inference"""
    print(f"Loading batch {batch_id}...")
    # Batch loading logic
    return {"batch_id": batch_id, "size": 100}


def run_inference_batch(batch_id: int, **context):
    """Run inference on a batch"""
    print(f"Running inference on batch {batch_id}...")
    # Inference logic
    return {"batch_id": batch_id, "results": []}


def aggregate_results(**context):
    """Aggregate all batch results"""
    print("Aggregating results...")
    # Aggregation logic
    return {"total_batches": 10, "total_results": 1000}


# Create DAG
with DAG(
    "thermal_aware_batch_inference",
    default_args=default_args,
    description="Batch inference with thermal-aware parallelism",
    schedule_interval=timedelta(hours=1),
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["mlx", "inference", "thermal-aware", "batch"],
) as dag:

    # Check thermal state to decide parallelism
    thermal_check = BranchPythonOperator(
        task_id="check_thermal_state",
        python_callable=check_thermal_state,
    )

    # High parallelism path (nominal thermal state)
    parallel_high_tasks = []
    for i in range(10):
        task = ThermalAwareInferenceOperator(
            task_id=f"inference_batch_{i}_high",
            model_path="/models/llama-7b",
            input_data=f"batch_{i}",
            thermal_threshold=1,
            priority="normal",
            memory_required_gb=6.0,
        )
        parallel_high_tasks.append(task)

    # Medium parallelism path (fair thermal state)
    parallel_medium_tasks = []
    for i in range(5):
        task = ThermalAwareInferenceOperator(
            task_id=f"inference_batch_{i}_medium",
            model_path="/models/llama-7b",
            input_data=f"batch_{i}",
            thermal_threshold=2,
            priority="normal",
            memory_required_gb=8.0,
        )
        parallel_medium_tasks.append(task)

    # Sequential path (serious/critical thermal state)
    sequential_task = ThermalAwareInferenceOperator(
        task_id="inference_sequential",
        model_path="/models/llama-7b",
        input_data="all_batches",
        thermal_threshold=2,
        priority="high",
        memory_required_gb=10.0,
        wait_for_clearance=True,
        clearance_timeout_seconds=300,
    )

    # Aggregation task (runs after any path completes)
    from mlops.airflow.operators import create_thermal_aware_task
    aggregate = create_thermal_aware_task(
        task_id="aggregate_results",
        python_callable=aggregate_results,
        thermal_threshold=1,
        priority="normal",
        memory_required_gb=4.0,
    )

    # Define dependencies
    thermal_check >> parallel_high_tasks >> aggregate
    thermal_check >> parallel_medium_tasks >> aggregate
    thermal_check >> sequential_task >> aggregate
