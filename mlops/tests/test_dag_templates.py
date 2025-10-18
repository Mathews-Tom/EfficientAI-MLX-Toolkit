"""Tests for DAG Template Generator

Tests for ML workflow templates with Apple Silicon optimization.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from mlops.config import AirflowConfig, AppleSiliconConfig
from mlops.orchestration import (
    DAGTemplate,
    DAGTemplateFactory,
    MLWorkflowType,
    TaskConfig,
)


class TestTaskConfig:
    """Tests for TaskConfig"""

    def test_basic_task_config(self):
        """Test basic task configuration"""
        task = TaskConfig(
            task_id="test_task",
            operator="PythonOperator",
            python_callable="test_function",
        )

        assert task.task_id == "test_task"
        assert task.operator == "PythonOperator"
        assert task.python_callable == "test_function"
        assert task.retries == 3
        assert task.dependencies == []

    def test_task_with_dependencies(self):
        """Test task with dependencies"""
        task = TaskConfig(
            task_id="dependent_task",
            operator="PythonOperator",
            python_callable="process_data",
            dependencies=["extract_data", "validate_data"],
        )

        assert task.dependencies == ["extract_data", "validate_data"]

    def test_apple_silicon_requirements(self):
        """Test Apple Silicon resource requirements"""
        task = TaskConfig(
            task_id="ml_task",
            operator="PythonOperator",
            python_callable="train_model",
            requires_apple_silicon=True,
            min_cores=8,
            min_memory_gb=16.0,
        )

        assert task.requires_apple_silicon is True
        assert task.min_cores == 8
        assert task.min_memory_gb == 16.0

    def test_bash_operator_task(self):
        """Test BashOperator task configuration"""
        task = TaskConfig(
            task_id="build_task",
            operator="BashOperator",
            bash_command="docker build -t model:latest .",
        )

        assert task.operator == "BashOperator"
        assert task.bash_command == "docker build -t model:latest ."
        assert task.python_callable is None


class TestDAGTemplate:
    """Tests for DAGTemplate"""

    def test_basic_dag_template(self):
        """Test basic DAG template creation"""
        template = DAGTemplate(
            dag_id="test_dag",
            description="Test DAG",
            workflow_type=MLWorkflowType.TRAINING,
        )

        assert template.dag_id == "test_dag"
        assert template.description == "Test DAG"
        assert template.workflow_type == MLWorkflowType.TRAINING
        assert template.catchup is False
        assert template.max_active_runs == 1

    def test_default_args_initialization(self):
        """Test default args are initialized"""
        template = DAGTemplate(
            dag_id="test_dag",
            description="Test DAG",
            workflow_type=MLWorkflowType.TRAINING,
        )

        assert "owner" in template.default_args
        assert "retries" in template.default_args
        assert template.default_args["owner"] == "mlops"

    def test_workflow_type_in_tags(self):
        """Test workflow type is added to tags"""
        template = DAGTemplate(
            dag_id="test_dag",
            description="Test DAG",
            workflow_type=MLWorkflowType.EVALUATION,
            tags=["ml", "test"],
        )

        assert "evaluation" in template.tags
        assert "ml" in template.tags
        assert "test" in template.tags

    def test_add_task(self):
        """Test adding tasks to template"""
        template = DAGTemplate(
            dag_id="test_dag",
            description="Test DAG",
            workflow_type=MLWorkflowType.TRAINING,
        )

        task1 = TaskConfig(
            task_id="task1",
            operator="PythonOperator",
            python_callable="func1",
        )

        task2 = TaskConfig(
            task_id="task2",
            operator="PythonOperator",
            python_callable="func2",
            dependencies=["task1"],
        )

        template.add_task(task1).add_task(task2)

        assert len(template.tasks) == 2
        assert template.tasks[0].task_id == "task1"
        assert template.tasks[1].task_id == "task2"

    def test_to_python_code_generation(self):
        """Test Python code generation from template"""
        template = DAGTemplate(
            dag_id="test_training_dag",
            description="Test training pipeline",
            workflow_type=MLWorkflowType.TRAINING,
            schedule_interval="@daily",
        )

        task = TaskConfig(
            task_id="train_model",
            operator="PythonOperator",
            python_callable="train_function",
        )
        template.add_task(task)

        code = template.to_python_code()

        assert "test_training_dag" in code
        assert "Test training pipeline" in code
        assert "@daily" in code
        assert "train_model" in code
        assert "PythonOperator" in code

    def test_apple_silicon_config(self):
        """Test DAG template with Apple Silicon config"""
        apple_config = AppleSiliconConfig(
            chip_type="M1",
            cores=8,
            memory_gb=16.0,
        )

        template = DAGTemplate(
            dag_id="optimized_dag",
            description="Apple Silicon optimized DAG",
            workflow_type=MLWorkflowType.TRAINING,
            apple_silicon_config=apple_config,
            enable_thermal_throttling=True,
        )

        assert template.apple_silicon_config == apple_config
        assert template.enable_thermal_throttling is True


class TestDAGTemplateFactory:
    """Tests for DAGTemplateFactory"""

    @pytest.fixture
    def factory(self):
        """Create DAGTemplateFactory instance"""
        config = AirflowConfig.detect()
        return DAGTemplateFactory(config)

    def test_factory_initialization(self, factory):
        """Test factory initialization"""
        assert factory.config is not None
        assert isinstance(factory.config, AirflowConfig)

    def test_create_training_pipeline(self, factory):
        """Test creating training pipeline template"""
        template = factory.create_training_pipeline(
            model_name="BERT Classifier",
            data_source="s3://data/training",
            output_path="/models/bert",
        )

        assert template.workflow_type == MLWorkflowType.TRAINING
        assert "training" in template.tags
        assert "BERT Classifier" in template.tags
        assert len(template.tasks) == 5  # validate, preprocess, train, evaluate, register

        # Check task IDs
        task_ids = [task.task_id for task in template.tasks]
        assert "validate_data" in task_ids
        assert "preprocess_data" in task_ids
        assert "train_model" in task_ids
        assert "evaluate_model" in task_ids
        assert "register_model" in task_ids

    def test_create_data_pipeline(self, factory):
        """Test creating data pipeline template"""
        template = factory.create_data_pipeline(
            pipeline_name="ETL Pipeline",
            source="s3://raw-data",
            destination="s3://processed-data",
        )

        assert template.workflow_type == MLWorkflowType.DATA_PIPELINE
        assert "data" in template.tags
        assert "pipeline" in template.tags
        assert len(template.tasks) == 4  # extract, transform, validate, load

        # Check task IDs
        task_ids = [task.task_id for task in template.tasks]
        assert "extract_data" in task_ids
        assert "transform_data" in task_ids
        assert "validate_data" in task_ids
        assert "load_data" in task_ids

    def test_create_model_deployment(self, factory):
        """Test creating model deployment template"""
        template = factory.create_model_deployment(
            model_name="GPT Model",
            model_version="v1.0",
            deployment_target="production",
        )

        assert template.workflow_type == MLWorkflowType.MODEL_DEPLOYMENT
        assert "deployment" in template.tags
        assert template.schedule_interval is None  # Manual trigger
        assert len(template.tasks) == 5  # fetch, validate, build, deploy, health_check

        # Check task IDs
        task_ids = [task.task_id for task in template.tasks]
        assert "fetch_model" in task_ids
        assert "validate_model" in task_ids
        assert "build_container" in task_ids
        assert "deploy_model" in task_ids
        assert "health_check" in task_ids

    def test_create_evaluation_pipeline(self, factory):
        """Test creating evaluation pipeline template"""
        template = factory.create_evaluation_pipeline(
            model_name="ResNet50",
            test_dataset="s3://test-data",
        )

        assert template.workflow_type == MLWorkflowType.EVALUATION
        assert "evaluation" in template.tags
        assert len(template.tasks) == 4  # load_model, load_test_data, run_eval, report

        # Check task IDs
        task_ids = [task.task_id for task in template.tasks]
        assert "load_model" in task_ids
        assert "load_test_data" in task_ids
        assert "run_evaluation" in task_ids
        assert "generate_report" in task_ids

    def test_training_pipeline_dependencies(self, factory):
        """Test training pipeline task dependencies"""
        template = factory.create_training_pipeline(
            model_name="Test Model",
            data_source="test_source",
            output_path="test_output",
        )

        # Find preprocess task
        preprocess_task = next(
            t for t in template.tasks if t.task_id == "preprocess_data"
        )
        assert "validate_data" in preprocess_task.dependencies

        # Find train task
        train_task = next(t for t in template.tasks if t.task_id == "train_model")
        assert "preprocess_data" in train_task.dependencies

    def test_apple_silicon_requirements_in_training(self, factory):
        """Test Apple Silicon requirements in training pipeline"""
        template = factory.create_training_pipeline(
            model_name="Large Model",
            data_source="data",
            output_path="output",
        )

        # Training task should require Apple Silicon
        train_task = next(t for t in template.tasks if t.task_id == "train_model")
        assert train_task.requires_apple_silicon is True
        assert train_task.min_cores == factory.config.apple_silicon.cores


class TestMLWorkflowType:
    """Tests for MLWorkflowType enum"""

    def test_workflow_types(self):
        """Test workflow type values"""
        assert MLWorkflowType.TRAINING.value == "training"
        assert MLWorkflowType.EVALUATION.value == "evaluation"
        assert MLWorkflowType.INFERENCE.value == "inference"
        assert MLWorkflowType.DATA_PIPELINE.value == "data_pipeline"
        assert MLWorkflowType.MODEL_DEPLOYMENT.value == "model_deployment"
