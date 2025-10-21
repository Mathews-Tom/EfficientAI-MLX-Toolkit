"""DAG Template Generator for ML Workflows

This module provides templates for generating Airflow DAGs optimized for ML workflows
on Apple Silicon. It includes pre-configured templates for common ML tasks like
training, evaluation, inference, and deployment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

from mlops.config import AirflowConfig, AppleSiliconConfig


class MLWorkflowType(Enum):
    """Types of ML workflow templates"""
    TRAINING = "training"
    EVALUATION = "evaluation"
    INFERENCE = "inference"
    DATA_PIPELINE = "data_pipeline"
    MODEL_DEPLOYMENT = "model_deployment"


@dataclass
class TaskConfig:
    """Configuration for a single task in a DAG"""
    task_id: str
    operator: str  # PythonOperator, BashOperator, etc.
    python_callable: str | None = None
    bash_command: str | None = None
    dependencies: list[str] = field(default_factory=list)
    retries: int = 3
    retry_delay: timedelta = field(default_factory=lambda: timedelta(minutes=5))
    execution_timeout: timedelta | None = None
    pool: str | None = None
    queue: str | None = None

    # Apple Silicon specific
    requires_apple_silicon: bool = False
    min_cores: int = 1
    min_memory_gb: float = 1.0


@dataclass
class DAGTemplate:
    """Template for generating an Airflow DAG"""
    dag_id: str
    description: str
    workflow_type: MLWorkflowType
    schedule_interval: str | None = None
    start_date: datetime = field(default_factory=lambda: datetime(2024, 1, 1))
    catchup: bool = False
    max_active_runs: int = 1
    tags: list[str] = field(default_factory=list)
    default_args: dict[str, Any] = field(default_factory=dict)
    tasks: list[TaskConfig] = field(default_factory=list)

    # Apple Silicon optimization
    apple_silicon_config: AppleSiliconConfig | None = None
    enable_thermal_throttling: bool = True

    def __post_init__(self):
        """Initialize default args if not provided"""
        if not self.default_args:
            self.default_args = {
                'owner': 'mlops',
                'depends_on_past': False,
                'email_on_failure': False,
                'email_on_retry': False,
                'retries': 3,
                'retry_delay': timedelta(minutes=5),
            }

        # Add workflow type to tags
        if self.workflow_type.value not in self.tags:
            self.tags.append(self.workflow_type.value)

    def add_task(self, task: TaskConfig) -> DAGTemplate:
        """Add a task to the DAG template"""
        self.tasks.append(task)
        return self

    def to_python_code(self) -> str:
        """Generate Python code for the DAG"""
        code_lines = [
            '"""',
            f'{self.description}',
            '',
            'Generated DAG for Airflow orchestration with Apple Silicon optimization.',
            '"""',
            '',
            'from datetime import datetime, timedelta',
            'from airflow import DAG',
            'from airflow.operators.python import PythonOperator',
            'from airflow.operators.bash import BashOperator',
            '',
            f'# DAG Configuration',
            f'dag_id = "{self.dag_id}"',
            f'description = "{self.description}"',
            f'schedule_interval = {repr(self.schedule_interval)}',
            '',
            f'# Default arguments',
            f'default_args = {self._format_dict(self.default_args)}',
            '',
            f'# Create DAG',
            f'with DAG(',
            f'    dag_id=dag_id,',
            f'    description=description,',
            f'    default_args=default_args,',
            f'    schedule_interval=schedule_interval,',
            f'    start_date={self._format_datetime(self.start_date)},',
            f'    catchup={self.catchup},',
            f'    max_active_runs={self.max_active_runs},',
            f'    tags={self.tags},',
            f') as dag:',
            '',
        ]

        # Add tasks
        for task in self.tasks:
            code_lines.extend(self._generate_task_code(task))

        # Add dependencies
        code_lines.append('')
        code_lines.append('    # Task dependencies')
        for task in self.tasks:
            if task.dependencies:
                deps = ' >> '.join(task.dependencies)
                code_lines.append(f'    {deps} >> {task.task_id}')

        return '\n'.join(code_lines)

    def _generate_task_code(self, task: TaskConfig) -> list[str]:
        """Generate code for a single task"""
        lines = []

        if task.operator == 'PythonOperator':
            lines.extend([
                f'    # Task: {task.task_id}',
                f'    {task.task_id} = PythonOperator(',
                f'        task_id="{task.task_id}",',
                f'        python_callable={task.python_callable},',
            ])
        elif task.operator == 'BashOperator':
            lines.extend([
                f'    # Task: {task.task_id}',
                f'    {task.task_id} = BashOperator(',
                f'        task_id="{task.task_id}",',
                f'        bash_command="{task.bash_command}",',
            ])

        # Add optional parameters
        if task.execution_timeout:
            lines.append(f'        execution_timeout=timedelta(seconds={task.execution_timeout.total_seconds()}),')
        if task.pool:
            lines.append(f'        pool="{task.pool}",')
        if task.queue:
            lines.append(f'        queue="{task.queue}",')

        lines.append('    )')
        lines.append('')

        return lines

    def _format_dict(self, d: dict) -> str:
        """Format dictionary for Python code"""
        items = []
        for k, v in d.items():
            if isinstance(v, timedelta):
                items.append(f'    "{k}": timedelta(seconds={v.total_seconds()})')
            elif isinstance(v, bool):
                items.append(f'    "{k}": {v}')
            elif isinstance(v, str):
                items.append(f'    "{k}": "{v}"')
            else:
                items.append(f'    "{k}": {v}')

        return '{\n' + ',\n'.join(items) + '\n}'

    def _format_datetime(self, dt: datetime) -> str:
        """Format datetime for Python code"""
        return f'datetime({dt.year}, {dt.month}, {dt.day})'


class DAGTemplateFactory:
    """Factory for creating pre-configured DAG templates"""

    def __init__(self, airflow_config: AirflowConfig | None = None):
        """Initialize factory with Airflow configuration

        Args:
            airflow_config: Airflow configuration (auto-detected if not provided)
        """
        from mlops.config import get_airflow_config
        self.config = airflow_config or get_airflow_config()

    def create_training_pipeline(
        self,
        model_name: str,
        data_source: str,
        output_path: str | Path,
    ) -> DAGTemplate:
        """Create a training pipeline DAG template

        Args:
            model_name: Name of the model to train
            data_source: Path or URI to training data
            output_path: Path to save trained model

        Returns:
            DAGTemplate configured for training workflow
        """
        dag_id = f"train_{model_name.lower().replace(' ', '_')}"

        template = DAGTemplate(
            dag_id=dag_id,
            description=f"Training pipeline for {model_name}",
            workflow_type=MLWorkflowType.TRAINING,
            schedule_interval="@daily",
            apple_silicon_config=self.config.apple_silicon,
            tags=["training", "ml", model_name],
        )

        # Data validation task
        template.add_task(TaskConfig(
            task_id="validate_data",
            operator="PythonOperator",
            python_callable="validate_training_data",
            requires_apple_silicon=False,
        ))

        # Preprocessing task
        template.add_task(TaskConfig(
            task_id="preprocess_data",
            operator="PythonOperator",
            python_callable="preprocess_data",
            dependencies=["validate_data"],
            requires_apple_silicon=True,
            min_cores=4,
            min_memory_gb=8.0,
        ))

        # Training task
        template.add_task(TaskConfig(
            task_id="train_model",
            operator="PythonOperator",
            python_callable="train_model",
            dependencies=["preprocess_data"],
            requires_apple_silicon=True,
            min_cores=self.config.apple_silicon.cores,
            min_memory_gb=self.config.apple_silicon.memory_gb,
            execution_timeout=timedelta(hours=24),
        ))

        # Evaluation task
        template.add_task(TaskConfig(
            task_id="evaluate_model",
            operator="PythonOperator",
            python_callable="evaluate_model",
            dependencies=["train_model"],
            requires_apple_silicon=True,
            min_cores=2,
            min_memory_gb=4.0,
        ))

        # Model registration task
        template.add_task(TaskConfig(
            task_id="register_model",
            operator="PythonOperator",
            python_callable="register_model",
            dependencies=["evaluate_model"],
            requires_apple_silicon=False,
        ))

        return template

    def create_data_pipeline(
        self,
        pipeline_name: str,
        source: str,
        destination: str,
    ) -> DAGTemplate:
        """Create a data pipeline DAG template

        Args:
            pipeline_name: Name of the data pipeline
            source: Data source path or URI
            destination: Destination path or URI

        Returns:
            DAGTemplate configured for data pipeline workflow
        """
        dag_id = f"data_{pipeline_name.lower().replace(' ', '_')}"

        template = DAGTemplate(
            dag_id=dag_id,
            description=f"Data pipeline: {pipeline_name}",
            workflow_type=MLWorkflowType.DATA_PIPELINE,
            schedule_interval="@hourly",
            apple_silicon_config=self.config.apple_silicon,
            tags=["data", "pipeline", pipeline_name],
        )

        # Extract task
        template.add_task(TaskConfig(
            task_id="extract_data",
            operator="PythonOperator",
            python_callable="extract_data",
        ))

        # Transform task
        template.add_task(TaskConfig(
            task_id="transform_data",
            operator="PythonOperator",
            python_callable="transform_data",
            dependencies=["extract_data"],
            requires_apple_silicon=True,
            min_cores=4,
            min_memory_gb=8.0,
        ))

        # Validate task
        template.add_task(TaskConfig(
            task_id="validate_data",
            operator="PythonOperator",
            python_callable="validate_data",
            dependencies=["transform_data"],
        ))

        # Load task
        template.add_task(TaskConfig(
            task_id="load_data",
            operator="PythonOperator",
            python_callable="load_data",
            dependencies=["validate_data"],
        ))

        return template

    def create_model_deployment(
        self,
        model_name: str,
        model_version: str,
        deployment_target: str,
    ) -> DAGTemplate:
        """Create a model deployment DAG template

        Args:
            model_name: Name of the model to deploy
            model_version: Version of the model
            deployment_target: Target deployment environment

        Returns:
            DAGTemplate configured for deployment workflow
        """
        dag_id = f"deploy_{model_name.lower().replace(' ', '_')}"

        template = DAGTemplate(
            dag_id=dag_id,
            description=f"Deploy {model_name} v{model_version} to {deployment_target}",
            workflow_type=MLWorkflowType.MODEL_DEPLOYMENT,
            schedule_interval=None,  # Manual trigger
            apple_silicon_config=self.config.apple_silicon,
            tags=["deployment", "ml", model_name, deployment_target],
        )

        # Fetch model task
        template.add_task(TaskConfig(
            task_id="fetch_model",
            operator="PythonOperator",
            python_callable="fetch_model_from_registry",
        ))

        # Validate model task
        template.add_task(TaskConfig(
            task_id="validate_model",
            operator="PythonOperator",
            python_callable="validate_model",
            dependencies=["fetch_model"],
            requires_apple_silicon=True,
        ))

        # Build container task
        template.add_task(TaskConfig(
            task_id="build_container",
            operator="BashOperator",
            bash_command=f"docker build -t {model_name}:{model_version} .",
            dependencies=["validate_model"],
        ))

        # Deploy task
        template.add_task(TaskConfig(
            task_id="deploy_model",
            operator="PythonOperator",
            python_callable="deploy_to_target",
            dependencies=["build_container"],
        ))

        # Health check task
        template.add_task(TaskConfig(
            task_id="health_check",
            operator="PythonOperator",
            python_callable="check_deployment_health",
            dependencies=["deploy_model"],
            retries=5,
            retry_delay=timedelta(minutes=2),
        ))

        return template

    def create_evaluation_pipeline(
        self,
        model_name: str,
        test_dataset: str,
    ) -> DAGTemplate:
        """Create a model evaluation DAG template

        Args:
            model_name: Name of the model to evaluate
            test_dataset: Path to test dataset

        Returns:
            DAGTemplate configured for evaluation workflow
        """
        dag_id = f"eval_{model_name.lower().replace(' ', '_')}"

        template = DAGTemplate(
            dag_id=dag_id,
            description=f"Evaluation pipeline for {model_name}",
            workflow_type=MLWorkflowType.EVALUATION,
            schedule_interval="@weekly",
            apple_silicon_config=self.config.apple_silicon,
            tags=["evaluation", "ml", model_name],
        )

        # Load model task
        template.add_task(TaskConfig(
            task_id="load_model",
            operator="PythonOperator",
            python_callable="load_model",
        ))

        # Load test data task
        template.add_task(TaskConfig(
            task_id="load_test_data",
            operator="PythonOperator",
            python_callable="load_test_data",
        ))

        # Run evaluation task
        template.add_task(TaskConfig(
            task_id="run_evaluation",
            operator="PythonOperator",
            python_callable="run_evaluation",
            dependencies=["load_model", "load_test_data"],
            requires_apple_silicon=True,
            min_cores=4,
            min_memory_gb=8.0,
        ))

        # Generate report task
        template.add_task(TaskConfig(
            task_id="generate_report",
            operator="PythonOperator",
            python_callable="generate_evaluation_report",
            dependencies=["run_evaluation"],
        ))

        return template
