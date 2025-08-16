"""
Base DSPy signatures for common ML workflows.
"""

# Standard library imports
from typing import Any

# Third-party imports
import dspy


class BaseOptimizationSignature(dspy.Signature):
    """Base signature for optimization tasks."""

    task_description = dspy.InputField(desc="Description of the optimization task")
    current_config = dspy.InputField(desc="Current configuration parameters")
    performance_metrics = dspy.InputField(desc="Current performance metrics")
    constraints = dspy.InputField(desc="Optimization constraints and requirements")

    optimized_config = dspy.OutputField(desc="Optimized configuration parameters")
    expected_improvement = dspy.OutputField(desc="Expected performance improvement")
    optimization_rationale = dspy.OutputField(desc="Explanation of optimization decisions")


class BaseTrainingSignature(dspy.Signature):
    """Base signature for training strategy generation."""

    model_details = dspy.InputField(desc="Model configuration and architecture details")
    dataset_info = dspy.InputField(desc="Dataset characteristics and statistics")
    training_objectives = dspy.InputField(desc="Training goals and target metrics")
    hardware_constraints = dspy.InputField(desc="Available hardware and resource constraints")

    training_strategy = dspy.OutputField(desc="Detailed training strategy and plan")
    hyperparameters = dspy.OutputField(desc="Recommended hyperparameters")
    monitoring_plan = dspy.OutputField(desc="Training monitoring and evaluation plan")


class BaseEvaluationSignature(dspy.Signature):
    """Base signature for model evaluation tasks."""

    model_info = dspy.InputField(desc="Model information and configuration")
    evaluation_data = dspy.InputField(desc="Evaluation dataset and metrics")
    evaluation_criteria = dspy.InputField(desc="Evaluation criteria and requirements")

    evaluation_results = dspy.OutputField(desc="Comprehensive evaluation results")
    performance_analysis = dspy.OutputField(desc="Performance analysis and insights")
    recommendations = dspy.OutputField(desc="Recommendations for improvement")


class BaseDeploymentSignature(dspy.Signature):
    """Base signature for model deployment planning."""

    model_details = dspy.InputField(desc="Model details and requirements")
    deployment_environment = dspy.InputField(desc="Target deployment environment")
    performance_requirements = dspy.InputField(desc="Performance and scalability requirements")

    deployment_plan = dspy.OutputField(desc="Detailed deployment strategy")
    infrastructure_requirements = dspy.OutputField(desc="Required infrastructure and resources")
    monitoring_setup = dspy.OutputField(desc="Monitoring and observability configuration")
