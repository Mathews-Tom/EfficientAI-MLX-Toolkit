"""Example Training DAG with Thermal-Aware Scheduling

This DAG demonstrates how to use thermal-aware operators for model training
workflows on Apple Silicon hardware.
"""

from datetime import datetime, timedelta

# Note: This example assumes Airflow is installed
# For testing without Airflow, these imports would be mocked
try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
except ImportError:
    # Fallback for environments without Airflow
    class DAG:
        def __init__(self, *args, **kwargs):
            pass

    class PythonOperator:
        def __init__(self, *args, **kwargs):
            pass

from mlops.airflow.operators import (
    ThermalAwareInferenceOperator,
    ThermalAwareTrainingOperator,
    create_thermal_aware_task,
)
from mlops.airflow.thermal_scheduler import ThermalAwareScheduler


# Default arguments for all tasks
default_args = {
    "owner": "mlops",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
}


def prepare_dataset(**context):
    """Prepare training dataset"""
    print("Preparing dataset...")
    # Dataset preparation logic here
    return {"dataset_path": "/tmp/train_data.jsonl", "num_samples": 10000}


def validate_model(**context):
    """Validate trained model"""
    print("Validating model...")
    # Model validation logic here
    return {"accuracy": 0.95, "loss": 0.05}


def publish_model(**context):
    """Publish trained model to serving"""
    print("Publishing model...")
    # Model publishing logic here
    return {"model_url": "mlx-models/llama-7b-finetuned"}


# Create DAG
with DAG(
    "thermal_aware_training",
    default_args=default_args,
    description="Train model with thermal-aware scheduling",
    schedule_interval=timedelta(days=1),
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["mlx", "training", "thermal-aware"],
) as dag:

    # Task 1: Prepare dataset (low thermal threshold)
    prep_task = create_thermal_aware_task(
        task_id="prepare_dataset",
        python_callable=prepare_dataset,
        thermal_threshold=1,  # Can run in most conditions
        priority="normal",
        memory_required_gb=4.0,
    )

    # Task 2: Train model (high thermal threshold, needs good cooling)
    train_task = ThermalAwareTrainingOperator(
        task_id="train_model",
        model_name="llama-7b",
        dataset_path="/tmp/train_data.jsonl",
        output_path="/tmp/models/llama-7b-finetuned",
        thermal_threshold=2,  # Requires serious or better
        priority="high",
        memory_required_gb=16.0,
        wait_for_clearance=True,
        clearance_timeout_seconds=600,  # Wait up to 10 minutes
        training_config={
            "epochs": 3,
            "batch_size": 32,
            "learning_rate": 1e-4,
        },
    )

    # Task 3: Validate model (moderate thermal requirements)
    validate_task = create_thermal_aware_task(
        task_id="validate_model",
        python_callable=validate_model,
        thermal_threshold=1,
        priority="normal",
        memory_required_gb=8.0,
    )

    # Task 4: Run inference for testing (low thermal requirements)
    inference_task = ThermalAwareInferenceOperator(
        task_id="test_inference",
        model_path="/tmp/models/llama-7b-finetuned",
        input_data=["Test prompt 1", "Test prompt 2"],
        thermal_threshold=1,
        priority="normal",
        memory_required_gb=8.0,
        inference_config={
            "batch_size": 64,
            "max_tokens": 512,
        },
    )

    # Task 5: Publish model (critical priority, must complete)
    publish_task = create_thermal_aware_task(
        task_id="publish_model",
        python_callable=publish_model,
        thermal_threshold=3,  # Can run even in critical thermal state
        priority="critical",
        memory_required_gb=2.0,
    )

    # Define task dependencies
    prep_task >> train_task >> validate_task >> inference_task >> publish_task
