"""
Diffusion model optimization specific DSPy signatures.
"""

# Third-party imports
import dspy

# Local imports
from .base_signatures import BaseOptimizationSignature


class DiffusionOptimizationSignature(BaseOptimizationSignature):
    """Optimize diffusion model architecture and training for Apple Silicon."""

    base_architecture = dspy.InputField(desc="Base diffusion model architecture (U-Net, DiT, etc.)")
    target_domain = dspy.InputField(desc="Target domain and use case (images, audio, text)")
    quality_requirements = dspy.InputField(desc="Quality requirements and performance targets")
    apple_silicon_config = dspy.InputField(
        desc="Apple Silicon hardware configuration and constraints"
    )

    optimized_architecture = dspy.OutputField(
        desc="Optimized diffusion architecture for Apple Silicon"
    )
    training_optimizations = dspy.OutputField(
        desc="Training optimizations and MLX-specific improvements"
    )
    memory_optimizations = dspy.OutputField(
        desc="Memory optimization strategies for unified memory"
    )
    performance_predictions = dspy.OutputField(
        desc="Expected performance improvements and benchmarks"
    )


class SamplingScheduleSignature(dspy.Signature):
    """Generate adaptive sampling schedule for diffusion models."""

    model_complexity = dspy.InputField(
        desc="Diffusion model architecture complexity and parameters"
    )
    content_type = dspy.InputField(desc="Type of content being generated and its characteristics")
    quality_speed_tradeoff = dspy.InputField(desc="Quality vs speed preferences and requirements")
    hardware_capabilities = dspy.InputField(
        desc="Apple Silicon capabilities and optimization level"
    )

    sampling_schedule = dspy.OutputField(desc="Optimized denoising schedule with step counts")
    noise_schedule = dspy.OutputField(desc="Optimized noise schedule for Apple Silicon")
    adaptive_strategy = dspy.OutputField(
        desc="Adaptive sampling strategy based on content complexity"
    )
    performance_estimates = dspy.OutputField(desc="Expected generation time and quality metrics")


class DiffusionArchitectureSearchSignature(dspy.Signature):
    """Search for optimal diffusion model architecture variants."""

    base_architecture = dspy.InputField(desc="Starting architecture configuration")
    search_constraints = dspy.InputField(desc="Architecture search constraints and requirements")
    optimization_objectives = dspy.InputField(
        desc="Objectives to optimize (speed, quality, memory)"
    )
    apple_silicon_features = dspy.InputField(desc="Apple Silicon specific features to leverage")

    architecture_variants = dspy.OutputField(desc="Promising architecture variants to explore")
    search_strategy = dspy.OutputField(desc="Architecture search strategy and methodology")
    evaluation_protocol = dspy.OutputField(desc="Protocol for evaluating architecture variants")
    expected_improvements = dspy.OutputField(
        desc="Expected improvements from architecture optimization"
    )


class DiffusionDistillationSignature(dspy.Signature):
    """Design progressive distillation strategy for diffusion models."""

    teacher_model = dspy.InputField(desc="Teacher diffusion model configuration and capabilities")
    distillation_objectives = dspy.InputField(
        desc="Distillation objectives and quality requirements"
    )
    compression_targets = dspy.InputField(
        desc="Target compression ratio and performance requirements"
    )
    hardware_constraints = dspy.InputField(
        desc="Hardware constraints and optimization requirements"
    )

    distillation_strategy = dspy.OutputField(desc="Multi-stage progressive distillation strategy")
    student_architecture = dspy.OutputField(desc="Optimized student model architecture")
    training_protocol = dspy.OutputField(desc="Distillation training protocol and schedule")
    quality_preservation = dspy.OutputField(desc="Strategy for preserving generation quality")


class DiffusionMemoryOptimizationSignature(dspy.Signature):
    """Optimize memory usage for diffusion model training and inference."""

    model_architecture = dspy.InputField(
        desc="Diffusion model architecture and memory requirements"
    )
    available_memory = dspy.InputField(desc="Available Apple Silicon unified memory")
    generation_requirements = dspy.InputField(desc="Generation requirements and batch sizes")
    quality_constraints = dspy.InputField(desc="Quality constraints and acceptable trade-offs")

    memory_strategy = dspy.OutputField(desc="Memory optimization strategy for Apple Silicon")
    attention_optimization = dspy.OutputField(
        desc="Attention mechanism optimization for memory efficiency"
    )
    gradient_checkpointing = dspy.OutputField(desc="Gradient checkpointing strategy")
    inference_optimization = dspy.OutputField(desc="Inference-time memory optimization techniques")


class DiffusionConsistencySignature(dspy.Signature):
    """Integrate consistency model capabilities for faster generation."""

    base_diffusion_model = dspy.InputField(desc="Base diffusion model to enhance with consistency")
    speed_requirements = dspy.InputField(desc="Speed requirements and generation time targets")
    quality_preservation = dspy.InputField(desc="Quality preservation requirements")

    consistency_integration = dspy.OutputField(
        desc="Strategy for integrating consistency model capabilities"
    )
    training_modifications = dspy.OutputField(desc="Required training modifications and additions")
    inference_acceleration = dspy.OutputField(
        desc="Inference acceleration techniques and expected speedup"
    )
    quality_analysis = dspy.OutputField(desc="Analysis of quality preservation and trade-offs")


class DiffusionMultiResolutionSignature(dspy.Signature):
    """Optimize diffusion models for multi-resolution training and generation."""

    target_resolutions = dspy.InputField(desc="Target resolutions for training and generation")
    model_architecture = dspy.InputField(desc="Base model architecture and scalability")
    training_strategy = dspy.InputField(desc="Current training approach and limitations")
    hardware_optimization = dspy.InputField(desc="Apple Silicon optimization requirements")

    multi_resolution_strategy = dspy.OutputField(
        desc="Multi-resolution training and generation strategy"
    )
    architecture_adaptations = dspy.OutputField(
        desc="Architecture adaptations for resolution scalability"
    )
    training_schedule = dspy.OutputField(desc="Progressive resolution training schedule")
    memory_management = dspy.OutputField(desc="Memory management across different resolutions")


class DiffusionDomainAdaptationSignature(dspy.Signature):
    """Adapt diffusion models for specific domains and use cases."""

    source_domain = dspy.InputField(desc="Source domain and pre-trained model characteristics")
    target_domain = dspy.InputField(desc="Target domain requirements and characteristics")
    available_data = dspy.InputField(desc="Available target domain data and constraints")
    adaptation_objectives = dspy.InputField(
        desc="Domain adaptation objectives and success criteria"
    )

    adaptation_strategy = dspy.OutputField(desc="Domain adaptation strategy and methodology")
    fine_tuning_protocol = dspy.OutputField(desc="Fine-tuning protocol for domain adaptation")
    data_augmentation = dspy.OutputField(desc="Data augmentation strategies for target domain")
    evaluation_methodology = dspy.OutputField(
        desc="Evaluation methodology for domain adaptation success"
    )
