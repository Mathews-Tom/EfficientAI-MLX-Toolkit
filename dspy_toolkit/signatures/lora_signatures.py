"""
LoRA fine-tuning specific DSPy signatures.
"""

# Third-party imports
import dspy

# Local imports
from .base_signatures import BaseOptimizationSignature, BaseTrainingSignature


class LoRAOptimizationSignature(BaseOptimizationSignature):
    """Optimize LoRA hyperparameters for given dataset and model."""

    model_name = dspy.InputField(desc="Base model name and architecture")
    dataset_info = dspy.InputField(desc="Dataset characteristics, size, and complexity metrics")
    hardware_constraints = dspy.InputField(
        desc="Available memory, compute, and Apple Silicon capabilities"
    )
    performance_targets = dspy.InputField(
        desc="Target metrics, quality requirements, and efficiency goals"
    )

    optimal_lora_rank = dspy.OutputField(
        desc="Optimized LoRA rank based on dataset complexity and model size"
    )
    optimal_lora_alpha = dspy.OutputField(desc="Optimized LoRA alpha scaling parameter")
    optimal_learning_rate = dspy.OutputField(desc="Optimized learning rate schedule")
    optimal_batch_size = dspy.OutputField(
        desc="Optimized batch size for Apple Silicon memory constraints"
    )
    training_config = dspy.OutputField(desc="Complete optimized training configuration")
    expected_performance = dspy.OutputField(desc="Expected performance metrics and training time")


class LoRATrainingSignature(BaseTrainingSignature):
    """Generate comprehensive training strategy for LoRA fine-tuning."""

    model_details = dspy.InputField(desc="Model architecture and LoRA configuration")
    dataset_path = dspy.InputField(desc="Training dataset location and preprocessing requirements")
    training_objectives = dspy.InputField(
        desc="Training goals, target metrics, and success criteria"
    )
    apple_silicon_config = dspy.InputField(
        desc="Apple Silicon hardware configuration and optimization level"
    )

    training_plan = dspy.OutputField(desc="Step-by-step training strategy with MLX optimizations")
    memory_optimization = dspy.OutputField(desc="Memory management strategy for Apple Silicon")
    monitoring_strategy = dspy.OutputField(
        desc="Training monitoring, logging, and checkpoint strategy"
    )
    evaluation_protocol = dspy.OutputField(desc="Evaluation methodology and validation approach")


class LoRAComparisonSignature(dspy.Signature):
    """Compare different LoRA configurations and methods."""

    base_model = dspy.InputField(desc="Base model for comparison")
    dataset_characteristics = dspy.InputField(desc="Dataset properties and requirements")
    comparison_methods = dspy.InputField(
        desc="LoRA methods to compare (LoRA, QLoRA, full fine-tuning)"
    )
    evaluation_criteria = dspy.InputField(desc="Comparison criteria and metrics")

    method_comparison = dspy.OutputField(desc="Detailed comparison of LoRA methods")
    performance_analysis = dspy.OutputField(desc="Performance analysis across different methods")
    recommendation = dspy.OutputField(
        desc="Recommended method based on requirements and constraints"
    )
    trade_off_analysis = dspy.OutputField(desc="Analysis of trade-offs between methods")


class LoRAHyperparameterSearchSignature(dspy.Signature):
    """Intelligent hyperparameter search for LoRA fine-tuning."""

    search_space = dspy.InputField(desc="Hyperparameter search space and bounds")
    optimization_objective = dspy.InputField(desc="Optimization objective and metrics to maximize")
    search_budget = dspy.InputField(desc="Search budget in terms of trials and time")
    prior_knowledge = dspy.InputField(desc="Prior knowledge and previous optimization results")

    search_strategy = dspy.OutputField(desc="Intelligent search strategy (Bayesian, grid, random)")
    parameter_priorities = dspy.OutputField(
        desc="Parameter importance ranking and search priorities"
    )
    early_stopping_criteria = dspy.OutputField(desc="Early stopping criteria for efficient search")
    expected_optimal_config = dspy.OutputField(
        desc="Expected optimal configuration based on priors"
    )


class LoRAMemoryOptimizationSignature(dspy.Signature):
    """Optimize memory usage for LoRA training on Apple Silicon."""

    model_size = dspy.InputField(desc="Model size and memory requirements")
    available_memory = dspy.InputField(desc="Available Apple Silicon unified memory")
    training_config = dspy.InputField(desc="Current training configuration")
    performance_requirements = dspy.InputField(desc="Performance and speed requirements")

    memory_strategy = dspy.OutputField(desc="Memory optimization strategy for Apple Silicon")
    batch_size_recommendation = dspy.OutputField(desc="Optimal batch size for memory constraints")
    gradient_accumulation = dspy.OutputField(desc="Gradient accumulation strategy")
    checkpoint_strategy = dspy.OutputField(desc="Checkpointing strategy for memory efficiency")


class LoRADatasetAnalysisSignature(dspy.Signature):
    """Analyze dataset for optimal LoRA configuration."""

    dataset_sample = dspy.InputField(desc="Representative sample of the training dataset")
    task_type = dspy.InputField(desc="Type of task (classification, generation, etc.)")
    domain_characteristics = dspy.InputField(
        desc="Domain-specific characteristics and requirements"
    )

    dataset_complexity = dspy.OutputField(desc="Dataset complexity analysis and metrics")
    recommended_lora_rank = dspy.OutputField(desc="Recommended LoRA rank based on dataset analysis")
    training_recommendations = dspy.OutputField(
        desc="Training recommendations based on dataset properties"
    )
    potential_challenges = dspy.OutputField(
        desc="Potential training challenges and mitigation strategies"
    )
