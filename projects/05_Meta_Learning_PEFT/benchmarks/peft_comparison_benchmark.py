"""Comprehensive benchmark for PEFT method comparison.

Compares LoRA, AdaLoRA, Prompt Tuning, Prefix Tuning, and Task-Conditional
adapters across multiple dimensions:
- Accuracy
- Parameter efficiency
- Training speed
- Inference speed
- Memory footprint
- Adaptation capability
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from adapter_generation import (
    AdapterFactory,
    PEFTMethod,
    LoRAMetaLearner,
    AdaLoRAMetaLearner,
    PromptTuningMetaLearner,
)
from task_embedding.task_distribution import TaskDistribution, TaskConfig
from utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PEFTBenchmarkResult:
    """Results from PEFT method benchmark."""

    method: str
    metric: str
    value: float
    unit: str
    metadata: dict[str, Any] | None = None


@dataclass
class PEFTBenchmarkConfig:
    """Configuration for PEFT benchmarks."""

    input_dim: int = 10
    hidden_dim: int = 32
    output_dim: int = 5

    # LoRA/AdaLoRA config
    lora_rank: int = 8
    lora_alpha: float = 16.0

    # Prompt tuning config
    num_prompts: int = 10

    # Training config
    num_meta_iterations: int = 50
    meta_batch_size: int = 4
    inner_lr: float = 0.01
    outer_lr: float = 0.001
    num_inner_steps: int = 5

    # Evaluation config
    num_eval_tasks: int = 10
    num_support: int = 5
    num_query: int = 15


class PEFTComparisonBenchmark:
    """Benchmark suite for comparing PEFT methods."""

    def __init__(self, config: PEFTBenchmarkConfig | None = None):
        """Initialize PEFT comparison benchmark.

        Args:
            config: Benchmark configuration
        """
        self.config = config or PEFTBenchmarkConfig()
        self.results: list[PEFTBenchmarkResult] = []

    def create_task_distribution(self) -> TaskDistribution:
        """Create task distribution for benchmarking."""
        task_config = TaskConfig(
            name="peft_benchmark_tasks",
            num_classes=self.config.output_dim,
            input_dim=self.config.input_dim,
            num_support=self.config.num_support,
            num_query=self.config.num_query,
        )

        return TaskDistribution(task_config)

    def create_learner(self, method: str) -> Any:
        """Create PEFT meta-learner for specified method.

        Args:
            method: PEFT method name

        Returns:
            Configured meta-learner
        """
        if method == "lora":
            return AdapterFactory.create_lora_meta_learner(
                input_dim=self.config.input_dim,
                hidden_dim=self.config.hidden_dim,
                output_dim=self.config.output_dim,
                lora_rank=self.config.lora_rank,
                lora_alpha=self.config.lora_alpha,
                inner_lr=self.config.inner_lr,
                outer_lr=self.config.outer_lr,
                num_inner_steps=self.config.num_inner_steps,
            )
        elif method == "adalora":
            return AdapterFactory.create_adalora_meta_learner(
                input_dim=self.config.input_dim,
                hidden_dim=self.config.hidden_dim,
                output_dim=self.config.output_dim,
                lora_rank=self.config.lora_rank,
                lora_alpha=self.config.lora_alpha,
                inner_lr=self.config.inner_lr,
                outer_lr=self.config.outer_lr,
                num_inner_steps=self.config.num_inner_steps,
            )
        elif method == "prompt_tuning":
            return AdapterFactory.create_prompt_tuning_meta_learner(
                input_dim=self.config.input_dim,
                hidden_dim=self.config.hidden_dim,
                output_dim=self.config.output_dim,
                num_prompts=self.config.num_prompts,
                inner_lr=self.config.inner_lr,
                outer_lr=self.config.outer_lr,
                num_inner_steps=self.config.num_inner_steps,
            )
        else:
            raise ValueError(f"Unsupported method: {method}")

    def loss_fn(self, logits: mx.array, labels: mx.array) -> mx.array:
        """Cross-entropy loss function."""
        return nn.losses.cross_entropy(logits, labels, reduction="mean")

    def benchmark_parameter_efficiency(
        self, method: str
    ) -> PEFTBenchmarkResult:
        """Benchmark parameter efficiency.

        Args:
            method: PEFT method to benchmark

        Returns:
            Parameter efficiency result
        """
        logger.info(f"Benchmarking parameter efficiency for {method}")

        learner = self.create_learner(method)

        # Count trainable parameters
        if hasattr(learner, "count_trainable_parameters"):
            param_info = learner.count_trainable_parameters()
            trainable_params = param_info["trainable"]
            total_params = param_info["total"]
            reduction_ratio = param_info["reduction_ratio"]
        else:
            # Fallback: count all model parameters
            model_params = learner.model.parameters()
            total_params = sum(p.size for p in model_params.values())
            trainable_params = total_params
            reduction_ratio = 1.0

        return PEFTBenchmarkResult(
            method=method,
            metric="parameter_efficiency",
            value=float(reduction_ratio),
            unit="reduction_ratio",
            metadata={
                "trainable_params": trainable_params,
                "total_params": total_params,
            },
        )

    def benchmark_training_speed(self, method: str) -> PEFTBenchmarkResult:
        """Benchmark training speed.

        Args:
            method: PEFT method to benchmark

        Returns:
            Training speed result
        """
        logger.info(f"Benchmarking training speed for {method}")

        task_dist = self.create_task_distribution()
        learner = self.create_learner(method)

        # Measure time per iteration
        start_time = time.time()

        for _ in range(self.config.num_meta_iterations):
            episodes = [
                task_dist.sample_episode()
                for _ in range(self.config.meta_batch_size)
            ]
            learner.meta_train_step(episodes, self.loss_fn)

        total_time = time.time() - start_time
        time_per_iteration = total_time / self.config.num_meta_iterations

        return PEFTBenchmarkResult(
            method=method,
            metric="training_speed",
            value=time_per_iteration,
            unit="seconds_per_iteration",
            metadata={"total_time": total_time},
        )

    def benchmark_inference_speed(self, method: str) -> PEFTBenchmarkResult:
        """Benchmark inference speed.

        Args:
            method: PEFT method to benchmark

        Returns:
            Inference speed result
        """
        logger.info(f"Benchmarking inference speed for {method}")

        learner = self.create_learner(method)

        # Generate test data
        batch_size = 32
        test_x = mx.random.normal((batch_size, self.config.input_dim))

        # Warm-up
        for _ in range(5):
            _ = learner.model(test_x)

        # Measure inference time
        num_runs = 100
        start_time = time.time()

        for _ in range(num_runs):
            _ = learner.model(test_x)

        total_time = time.time() - start_time
        time_per_batch = total_time / num_runs
        time_per_sample = time_per_batch / batch_size

        return PEFTBenchmarkResult(
            method=method,
            metric="inference_speed",
            value=time_per_sample * 1000,  # Convert to milliseconds
            unit="ms_per_sample",
            metadata={
                "batch_size": batch_size,
                "time_per_batch": time_per_batch,
            },
        )

    def benchmark_accuracy(self, method: str) -> list[PEFTBenchmarkResult]:
        """Benchmark final accuracy after meta-training.

        Args:
            method: PEFT method to benchmark

        Returns:
            Accuracy results
        """
        logger.info(f"Benchmarking accuracy for {method}")

        results = []
        task_dist = self.create_task_distribution()
        learner = self.create_learner(method)

        # Meta-training
        for _ in range(self.config.num_meta_iterations):
            episodes = [
                task_dist.sample_episode()
                for _ in range(self.config.meta_batch_size)
            ]
            learner.meta_train_step(episodes, self.loss_fn)

        # Evaluation
        eval_episodes = [
            task_dist.sample_episode()
            for _ in range(self.config.num_eval_tasks)
        ]

        eval_results = learner.evaluate(eval_episodes, self.loss_fn)

        # Record accuracy
        results.append(PEFTBenchmarkResult(
            method=method,
            metric="final_accuracy",
            value=float(eval_results.get("avg_acc", 0.0)),
            unit="accuracy",
            metadata={"avg_loss": float(eval_results.get("avg_loss", 0.0))},
        ))

        return results

    def benchmark_adaptation_capability(
        self, method: str
    ) -> list[PEFTBenchmarkResult]:
        """Benchmark adaptation capability (few-shot learning).

        Args:
            method: PEFT method to benchmark

        Returns:
            Adaptation capability results
        """
        logger.info(f"Benchmarking adaptation capability for {method}")

        results = []
        task_dist = self.create_task_distribution()
        learner = self.create_learner(method)

        # Meta-training
        for _ in range(self.config.num_meta_iterations):
            episodes = [
                task_dist.sample_episode()
                for _ in range(self.config.meta_batch_size)
            ]
            learner.meta_train_step(episodes, self.loss_fn)

        # Test different K-shot settings
        for k in [1, 3, 5]:
            # Create k-shot task distribution
            k_shot_config = TaskConfig(
                name=f"{k}_shot_tasks",
                num_classes=self.config.output_dim,
                input_dim=self.config.input_dim,
                num_support=k,
                num_query=self.config.num_query,
            )
            k_shot_dist = TaskDistribution(k_shot_config)

            # Evaluate
            k_shot_episodes = [
                k_shot_dist.sample_episode()
                for _ in range(self.config.num_eval_tasks)
            ]

            eval_results = learner.evaluate(k_shot_episodes, self.loss_fn)

            results.append(PEFTBenchmarkResult(
                method=method,
                metric=f"{k}_shot_accuracy",
                value=float(eval_results.get("avg_acc", 0.0)),
                unit="accuracy",
                metadata={"num_shots": k},
            ))

        return results

    def benchmark_memory_footprint(self, method: str) -> PEFTBenchmarkResult:
        """Benchmark memory footprint.

        Args:
            method: PEFT method to benchmark

        Returns:
            Memory footprint result
        """
        logger.info(f"Benchmarking memory footprint for {method}")

        learner = self.create_learner(method)

        # Count all parameters (trainable and frozen)
        model_params = learner.model.parameters()
        total_params = sum(p.size for p in model_params.values())

        # Estimate memory (4 bytes per float32)
        memory_mb = (total_params * 4) / (1024 * 1024)

        return PEFTBenchmarkResult(
            method=method,
            metric="memory_footprint",
            value=memory_mb,
            unit="MB",
            metadata={"total_parameters": total_params},
        )

    def run_full_benchmark(
        self,
        methods: list[str] = ["lora", "adalora", "prompt_tuning"],
    ) -> dict[str, list[PEFTBenchmarkResult]]:
        """Run comprehensive PEFT comparison benchmark.

        Args:
            methods: List of PEFT methods to benchmark

        Returns:
            Dictionary mapping method names to results
        """
        logger.info("Starting comprehensive PEFT comparison benchmark")

        all_results = {}

        for method in methods:
            logger.info(f"\n{'='*60}")
            logger.info(f"Benchmarking: {method.upper()}")
            logger.info(f"{'='*60}")

            results = []

            # Parameter efficiency
            results.append(self.benchmark_parameter_efficiency(method))

            # Training speed
            results.append(self.benchmark_training_speed(method))

            # Inference speed
            results.append(self.benchmark_inference_speed(method))

            # Accuracy
            results.extend(self.benchmark_accuracy(method))

            # Adaptation capability
            results.extend(self.benchmark_adaptation_capability(method))

            # Memory footprint
            results.append(self.benchmark_memory_footprint(method))

            all_results[method] = results

            # Log summary
            logger.info(f"\n{method.upper()} Summary:")
            for result in results:
                logger.info(
                    f"  {result.metric:30s}: {result.value:8.4f} {result.unit}"
                )

        self.results = [r for results in all_results.values() for r in results]

        return all_results

    def generate_comparison_report(self) -> str:
        """Generate comparison report.

        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 80)
        report.append("PEFT METHOD COMPARISON REPORT")
        report.append("=" * 80)
        report.append("")

        # Group results by method
        by_method: dict[str, list[PEFTBenchmarkResult]] = {}
        for result in self.results:
            if result.method not in by_method:
                by_method[result.method] = []
            by_method[result.method].append(result)

        # Create comparison table for key metrics
        methods = list(by_method.keys())
        key_metrics = [
            "parameter_efficiency",
            "training_speed",
            "inference_speed",
            "final_accuracy",
            "memory_footprint",
        ]

        report.append("\nKEY METRICS COMPARISON")
        report.append("-" * 80)
        report.append(f"{'Metric':<30s} " + " ".join(f"{m:>12s}" for m in methods))
        report.append("-" * 80)

        for metric in key_metrics:
            values = []
            unit = ""
            for method in methods:
                method_results = by_method[method]
                matching = [r for r in method_results if r.metric == metric]
                if matching:
                    values.append(f"{matching[0].value:12.4f}")
                    unit = matching[0].unit
                else:
                    values.append(f"{'N/A':>12s}")

            report.append(f"{metric:<30s} " + " ".join(values) + f"  ({unit})")

        # Detailed results per method
        report.append("\n\nDETAILED RESULTS")
        report.append("=" * 80)

        for method, results in by_method.items():
            report.append(f"\n{method.upper()}")
            report.append("-" * 80)

            for result in results:
                report.append(
                    f"  {result.metric:40s}: {result.value:10.4f} {result.unit}"
                )

        report.append("\n" + "=" * 80)

        return "\n".join(report)


def run_quick_peft_benchmark() -> None:
    """Run quick PEFT benchmark for development."""
    config = PEFTBenchmarkConfig(
        num_meta_iterations=20,
        num_eval_tasks=5,
    )

    benchmark = PEFTComparisonBenchmark(config)
    results = benchmark.run_full_benchmark(methods=["lora", "prompt_tuning"])

    print(benchmark.generate_comparison_report())


def run_full_peft_benchmark() -> None:
    """Run comprehensive PEFT benchmark suite."""
    config = PEFTBenchmarkConfig(
        num_meta_iterations=50,
        num_eval_tasks=10,
    )

    benchmark = PEFTComparisonBenchmark(config)
    results = benchmark.run_full_benchmark()

    print(benchmark.generate_comparison_report())

    # Save results
    import json
    from pathlib import Path

    output_dir = Path(__file__).parent.parent / "outputs" / "benchmarks"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert results to JSON
    results_json = {
        method: [
            {
                "metric": r.metric,
                "value": r.value,
                "unit": r.unit,
                "metadata": r.metadata,
            }
            for r in results_list
        ]
        for method, results_list in results.items()
    }

    output_file = output_dir / "peft_comparison_benchmark.json"
    with open(output_file, "w") as f:
        json.dump(results_json, f, indent=2)

    logger.info(f"Results saved to {output_file}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        run_quick_peft_benchmark()
    else:
        run_full_peft_benchmark()
