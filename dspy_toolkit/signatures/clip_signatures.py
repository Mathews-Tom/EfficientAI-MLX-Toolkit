"""
CLIP fine-tuning specific DSPy signatures.
"""

# Third-party imports
import dspy

# Local imports
from .base_signatures import BaseOptimizationSignature


class CLIPDomainAdaptationSignature(BaseOptimizationSignature):
    """Adapt CLIP model for specific domain with optimal strategy."""

    source_domain = dspy.InputField(desc="Original CLIP training domain and characteristics")
    target_domain = dspy.InputField(
        desc="Target domain for adaptation (medical, industrial, scientific)"
    )
    available_data = dspy.InputField(desc="Available domain-specific image-text pairs and quality")
    adaptation_objectives = dspy.InputField(
        desc="Specific adaptation objectives and success metrics"
    )

    adaptation_strategy = dspy.OutputField(
        desc="Optimal domain adaptation strategy and methodology"
    )
    fine_tuning_approach = dspy.OutputField(desc="Fine-tuning approach (full, partial, LoRA-based)")
    data_requirements = dspy.OutputField(desc="Data requirements and augmentation strategies")
    expected_performance = dspy.OutputField(
        desc="Expected performance improvements in target domain"
    )


class ContrastiveLossSignature(dspy.Signature):
    """Design optimal contrastive loss for CLIP fine-tuning."""

    domain_characteristics = dspy.InputField(
        desc="Target domain characteristics and data properties"
    )
    similarity_requirements = dspy.InputField(desc="Required similarity metrics and precision")
    training_objectives = dspy.InputField(desc="Training objectives and optimization goals")
    hardware_constraints = dspy.InputField(
        desc="Apple Silicon hardware constraints and MPS optimization"
    )

    loss_configuration = dspy.OutputField(
        desc="Optimized contrastive loss configuration and parameters"
    )
    similarity_metrics = dspy.OutputField(desc="Optimal similarity metrics and distance functions")
    temperature_scaling = dspy.OutputField(
        desc="Temperature scaling strategy for contrastive learning"
    )
    negative_sampling = dspy.OutputField(desc="Negative sampling strategy and hard negative mining")


class CLIPMultiModalOptimizationSignature(dspy.Signature):
    """Optimize CLIP for multi-modal understanding and performance."""

    modality_requirements = dspy.InputField(desc="Multi-modal requirements (image, text, audio)")
    performance_targets = dspy.InputField(desc="Performance targets for each modality")
    cross_modal_objectives = dspy.InputField(desc="Cross-modal understanding objectives")
    apple_silicon_config = dspy.InputField(desc="Apple Silicon configuration and MPS capabilities")

    optimization_strategy = dspy.OutputField(desc="Multi-modal optimization strategy")
    architecture_modifications = dspy.OutputField(
        desc="Architecture modifications for multi-modal performance"
    )
    training_protocol = dspy.OutputField(desc="Training protocol for multi-modal optimization")
    evaluation_framework = dspy.OutputField(
        desc="Evaluation framework for multi-modal capabilities"
    )


class CLIPMemoryOptimizationSignature(dspy.Signature):
    """Optimize CLIP training for memory efficiency on Apple Silicon."""

    model_size = dspy.InputField(desc="CLIP model size and memory requirements")
    batch_size_requirements = dspy.InputField(desc="Desired batch size and training efficiency")
    available_memory = dspy.InputField(desc="Available Apple Silicon unified memory")
    quality_constraints = dspy.InputField(desc="Quality constraints and acceptable trade-offs")

    memory_strategy = dspy.OutputField(desc="Memory optimization strategy for Apple Silicon")
    attention_slicing = dspy.OutputField(
        desc="Attention slicing configuration for memory efficiency"
    )
    gradient_accumulation = dspy.OutputField(
        desc="Gradient accumulation strategy for larger effective batches"
    )
    mixed_precision = dspy.OutputField(desc="Mixed precision training configuration")


class CLIPDataAugmentationSignature(dspy.Signature):
    """Design data augmentation strategy for CLIP fine-tuning."""

    domain_data = dspy.InputField(desc="Available domain-specific data and characteristics")
    augmentation_objectives = dspy.InputField(desc="Data augmentation objectives and requirements")
    quality_preservation = dspy.InputField(
        desc="Quality preservation requirements for augmented data"
    )

    augmentation_strategy = dspy.OutputField(desc="Comprehensive data augmentation strategy")
    image_augmentations = dspy.OutputField(desc="Image augmentation techniques and parameters")
    text_augmentations = dspy.OutputField(desc="Text augmentation and paraphrasing strategies")
    cross_modal_augmentations = dspy.OutputField(desc="Cross-modal augmentation techniques")


class CLIPEvaluationSignature(dspy.Signature):
    """Design comprehensive evaluation protocol for CLIP fine-tuning."""

    evaluation_objectives = dspy.InputField(desc="Evaluation objectives and success criteria")
    target_tasks = dspy.InputField(desc="Target tasks and downstream applications")
    benchmark_requirements = dspy.InputField(desc="Benchmark requirements and comparison baselines")

    evaluation_protocol = dspy.OutputField(desc="Comprehensive evaluation protocol and methodology")
    benchmark_suite = dspy.OutputField(desc="Benchmark suite for domain-specific evaluation")
    metrics_framework = dspy.OutputField(desc="Metrics framework for multi-modal evaluation")
    performance_analysis = dspy.OutputField(
        desc="Performance analysis and interpretation guidelines"
    )


class CLIPZeroShotOptimizationSignature(dspy.Signature):
    """Optimize CLIP for zero-shot performance in target domain."""

    target_domain = dspy.InputField(desc="Target domain for zero-shot performance")
    zero_shot_objectives = dspy.InputField(desc="Zero-shot performance objectives and requirements")
    prompt_engineering = dspy.InputField(desc="Prompt engineering requirements and constraints")

    zero_shot_strategy = dspy.OutputField(desc="Zero-shot optimization strategy")
    prompt_optimization = dspy.OutputField(desc="Prompt optimization and engineering approach")
    few_shot_enhancement = dspy.OutputField(desc="Few-shot enhancement strategies")
    performance_expectations = dspy.OutputField(desc="Expected zero-shot performance improvements")


class CLIPScalabilitySignature(dspy.Signature):
    """Design scalability strategy for CLIP deployment and inference."""

    deployment_requirements = dspy.InputField(desc="Deployment requirements and scale expectations")
    performance_targets = dspy.InputField(
        desc="Performance targets for inference speed and throughput"
    )
    resource_constraints = dspy.InputField(desc="Resource constraints and hardware limitations")

    scalability_strategy = dspy.OutputField(desc="Scalability strategy for CLIP deployment")
    inference_optimization = dspy.OutputField(desc="Inference optimization techniques")
    caching_strategy = dspy.OutputField(desc="Caching strategy for embeddings and features")
    load_balancing = dspy.OutputField(desc="Load balancing and distributed inference approach")
