"""Comprehensive benchmark suite for meta-learning algorithms.

Benchmarks MAML, Reptile, FOMAML, and Meta-SGD across various metrics:
- Adaptation speed (few-shot learning)
- Final accuracy
- Training time
- Memory usage
- Parameter efficiency
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

import mlx.core as mx
import mlx.nn as nn

from meta_learning.maml import MAMLLearner, FOMAMLLearner
from meta_learning.reptile import ReptileLearner
from meta_learning.meta_sgd import MetaSGDLearner
from meta_learning.models import SimpleClassifier
from task_embedding.task_distribution import TaskDistribution, TaskConfig
from utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""

    algorithm: str
    metric: str
    value: float
    unit: str
    metadata: dict[str, Any] | None = None


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""

    input_dim: int = 10
    hidden_dim: int = 32
    output_dim: int = 5
    num_classes: int = 5

    # Meta-training config
    num_meta_iterations: int = 100
    meta_batch_size: int = 4
    inner_lr: float = 0.01
    outer_lr: float = 0.001
    num_inner_steps: int = 5

    # Task config
    num_support: int = 5  # K-shot
    num_query: int = 15

    # Evaluation config
    num_eval_tasks: int = 20


class MetaLearningBenchmark:
    """Benchmark suite for meta-learning algorithms."""

    def __init__(self, config: BenchmarkConfig | None = None):
        """Initialize benchmark suite.

        Args:
            config: Benchmark configuration
        """
        self.config = config or BenchmarkConfig()
        self.results: list[BenchmarkResult] = []

    def create_task_distribution(self) -> TaskDistribution:
        """Create task distribution for benchmarking.

        Returns:
            TaskDistribution for synthetic tasks
        """
        task_config = TaskConfig(
            name="benchmark_tasks",
            num_classes=self.config.num_classes,
            input_dim=self.config.input_dim,
            num_support=self.config.num_support,
            num_query=self.config.num_query,
        )

        return TaskDistribution(task_config)

    def create_learner(self, algorithm: str) -> Any:
        """Create meta-learner for specified algorithm.

        Args:
            algorithm: Algorithm name (maml, fomaml, reptile, meta_sgd)

        Returns:
            Configured meta-learner
        """
        model = SimpleClassifier(
            input_dim=self.config.input_dim,
            hidden_dim=self.config.hidden_dim,
            num_classes=self.config.num_classes,
        )

        if algorithm == "maml":
            return MAMLLearner(
                model=model,
                inner_lr=self.config.inner_lr,
                outer_lr=self.config.outer_lr,
                num_inner_steps=self.config.num_inner_steps,
            )
        elif algorithm == "fomaml":
            return FOMAMLLearner(
                model=model,
                inner_lr=self.config.inner_lr,
                outer_lr=self.config.outer_lr,
                num_inner_steps=self.config.num_inner_steps,
            )
        elif algorithm == "reptile":
            return ReptileLearner(
                model=model,
                inner_lr=self.config.inner_lr,
                outer_lr=self.config.outer_lr,
                num_inner_steps=self.config.num_inner_steps,
            )
        elif algorithm == "meta_sgd":
            return MetaSGDLearner(
                model=model,
                meta_lr=self.config.outer_lr,
                alpha_lr=self.config.outer_lr,
                num_inner_steps=self.config.num_inner_steps,
            )
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

    def loss_fn(self, logits: mx.array, labels: mx.array) -> mx.array:
        """Cross-entropy loss function.

        Args:
            logits: Model predictions
            labels: Ground truth labels

        Returns:
            Loss value
        """
        return nn.losses.cross_entropy(logits, labels, reduction="mean")

    def benchmark_adaptation_speed(
        self,
        algorithm: str,
        num_shots: list[int] = [1, 5, 10],
    ) -> list[BenchmarkResult]:
        """Benchmark few-shot adaptation speed.

        Args:
            algorithm: Algorithm to benchmark
            num_shots: List of K-shot settings to test

        Returns:
            List of benchmark results
        """
        logger.info(f"Benchmarking adaptation speed for {algorithm}")

        results = []
        task_dist = self.create_task_distribution()
        learner = self.create_learner(algorithm)

        # Meta-train
        for _ in range(self.config.num_meta_iterations):
            episodes = [
                task_dist.sample_episode()
                for _ in range(self.config.meta_batch_size)
            ]
            learner.meta_train_step(episodes, self.loss_fn)

        # Evaluate at different K-shot settings
        for k in num_shots:
            # Update task config for K-shot
            task_config = TaskConfig(
                name=f"{k}_shot_tasks",
                num_classes=self.config.num_classes,
                input_dim=self.config.input_dim,
                num_support=k,
                num_query=self.config.num_query,
            )
            eval_dist = TaskDistribution(task_config)

            # Sample evaluation tasks
            eval_episodes = [
                eval_dist.sample_episode()
                for _ in range(self.config.num_eval_tasks)
            ]

            # Evaluate
            eval_results = learner.evaluate(
                eval_episodes,
                self.loss_fn,
                num_adaptation_steps=[0, 1, 3, 5],
            )

            # Record results
            for step, acc_key in [
                (0, "step_0_acc"),
                (1, "step_1_acc"),
                (3, "step_3_acc"),
                (5, "step_5_acc"),
            ]:
                if acc_key in eval_results:
                    results.append(BenchmarkResult(
                        algorithm=algorithm,
                        metric=f"{k}_shot_acc_step_{step}",
                        value=float(eval_results[acc_key]),
                        unit="accuracy",
                        metadata={"num_shots": k, "adaptation_steps": step},
                    ))

        return results

    def benchmark_training_time(self, algorithm: str) -> BenchmarkResult:
        """Benchmark meta-training time.

        Args:
            algorithm: Algorithm to benchmark

        Returns:
            Training time result
        """
        logger.info(f"Benchmarking training time for {algorithm}")

        task_dist = self.create_task_distribution()
        learner = self.create_learner(algorithm)

        # Measure training time
        start_time = time.time()

        for _ in range(self.config.num_meta_iterations):
            episodes = [
                task_dist.sample_episode()
                for _ in range(self.config.meta_batch_size)
            ]
            learner.meta_train_step(episodes, self.loss_fn)

        training_time = time.time() - start_time

        return BenchmarkResult(
            algorithm=algorithm,
            metric="training_time",
            value=training_time,
            unit="seconds",
            metadata={"num_iterations": self.config.num_meta_iterations},
        )

    def benchmark_memory_usage(self, algorithm: str) -> BenchmarkResult:
        """Benchmark memory usage.

        Args:
            algorithm: Algorithm to benchmark

        Returns:
            Memory usage result
        """
        logger.info(f"Benchmarking memory usage for {algorithm}")

        learner = self.create_learner(algorithm)

        # Count parameters
        model_params = learner.model.parameters()
        total_params = sum(p.size for p in model_params.values())

        # Estimate memory (4 bytes per float32 parameter)
        memory_mb = (total_params * 4) / (1024 * 1024)

        return BenchmarkResult(
            algorithm=algorithm,
            metric="memory_usage",
            value=memory_mb,
            unit="MB",
            metadata={"total_parameters": total_params},
        )

    def benchmark_convergence_rate(self, algorithm: str) -> list[BenchmarkResult]:
        """Benchmark convergence rate during meta-training.

        Args:
            algorithm: Algorithm to benchmark

        Returns:
            Convergence rate results
        """
        logger.info(f"Benchmarking convergence rate for {algorithm}")

        results = []
        task_dist = self.create_task_distribution()
        learner = self.create_learner(algorithm)

        # Track loss at intervals
        eval_intervals = [10, 25, 50, 75, 100]

        for iteration in range(self.config.num_meta_iterations):
            # Training step
            episodes = [
                task_dist.sample_episode()
                for _ in range(self.config.meta_batch_size)
            ]
            learner.meta_train_step(episodes, self.loss_fn)

            # Evaluate at intervals
            if (iteration + 1) in eval_intervals:
                eval_episodes = [
                    task_dist.sample_episode()
                    for _ in range(10)
                ]
                eval_results = learner.evaluate(eval_episodes, self.loss_fn)

                results.append(BenchmarkResult(
                    algorithm=algorithm,
                    metric=f"convergence_iter_{iteration + 1}",
                    value=float(eval_results["avg_acc"]),
                    unit="accuracy",
                    metadata={"iteration": iteration + 1},
                ))

        return results

    def run_full_benchmark(
        self,
        algorithms: list[str] = ["maml", "fomaml", "reptile", "meta_sgd"],
    ) -> dict[str, list[BenchmarkResult]]:
        """Run comprehensive benchmark suite.

        Args:
            algorithms: List of algorithms to benchmark

        Returns:
            Dictionary mapping algorithm names to results
        """
        logger.info("Starting comprehensive meta-learning benchmark")

        all_results = {}

        for algorithm in algorithms:
            logger.info(f"\n{'='*60}")
            logger.info(f"Benchmarking: {algorithm.upper()}")
            logger.info(f"{'='*60}")

            results = []

            # Adaptation speed
            results.extend(self.benchmark_adaptation_speed(algorithm))

            # Training time
            results.append(self.benchmark_training_time(algorithm))

            # Memory usage
            results.append(self.benchmark_memory_usage(algorithm))

            # Convergence rate
            results.extend(self.benchmark_convergence_rate(algorithm))

            all_results[algorithm] = results

            # Log summary
            logger.info(f"\n{algorithm.upper()} Summary:")
            for result in results[-5:]:  # Show last 5 results
                logger.info(
                    f"  {result.metric}: {result.value:.4f} {result.unit}"
                )

        self.results = [r for results in all_results.values() for r in results]

        return all_results

    def generate_report(self) -> str:
        """Generate benchmark report.

        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 80)
        report.append("META-LEARNING BENCHMARK REPORT")
        report.append("=" * 80)
        report.append("")

        # Group results by algorithm
        by_algorithm: dict[str, list[BenchmarkResult]] = {}
        for result in self.results:
            if result.algorithm not in by_algorithm:
                by_algorithm[result.algorithm] = []
            by_algorithm[result.algorithm].append(result)

        # Report for each algorithm
        for algorithm, results in by_algorithm.items():
            report.append(f"\n{algorithm.upper()}")
            report.append("-" * 80)

            # Group by metric type
            adaptation_results = [
                r for r in results if "shot_acc" in r.metric
            ]
            time_results = [r for r in results if "time" in r.metric]
            memory_results = [r for r in results if "memory" in r.metric]
            convergence_results = [
                r for r in results if "convergence" in r.metric
            ]

            # Adaptation speed
            if adaptation_results:
                report.append("\n  Few-Shot Adaptation:")
                for result in adaptation_results:
                    report.append(
                        f"    {result.metric:40s}: {result.value:8.4f} {result.unit}"
                    )

            # Training time
            if time_results:
                report.append("\n  Training Time:")
                for result in time_results:
                    report.append(
                        f"    {result.metric:40s}: {result.value:8.2f} {result.unit}"
                    )

            # Memory usage
            if memory_results:
                report.append("\n  Memory Usage:")
                for result in memory_results:
                    report.append(
                        f"    {result.metric:40s}: {result.value:8.2f} {result.unit}"
                    )

            # Convergence
            if convergence_results:
                report.append("\n  Convergence Rate:")
                for result in convergence_results[-5:]:  # Last 5 points
                    report.append(
                        f"    {result.metric:40s}: {result.value:8.4f} {result.unit}"
                    )

        report.append("\n" + "=" * 80)

        return "\n".join(report)


def run_quick_benchmark() -> None:
    """Run quick benchmark for development/testing."""
    config = BenchmarkConfig(
        num_meta_iterations=20,
        num_eval_tasks=5,
    )

    benchmark = MetaLearningBenchmark(config)
    results = benchmark.run_full_benchmark(algorithms=["maml", "reptile"])

    print(benchmark.generate_report())


def run_full_benchmark() -> None:
    """Run comprehensive benchmark suite."""
    config = BenchmarkConfig(
        num_meta_iterations=100,
        num_eval_tasks=20,
    )

    benchmark = MetaLearningBenchmark(config)
    results = benchmark.run_full_benchmark()

    print(benchmark.generate_report())

    # Save results
    import json
    from pathlib import Path

    output_dir = Path(__file__).parent.parent / "outputs" / "benchmarks"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert results to JSON
    results_json = {
        algorithm: [
            {
                "metric": r.metric,
                "value": r.value,
                "unit": r.unit,
                "metadata": r.metadata,
            }
            for r in results_list
        ]
        for algorithm, results_list in results.items()
    }

    output_file = output_dir / "meta_learning_benchmark.json"
    with open(output_file, "w") as f:
        json.dump(results_json, f, indent=2)

    logger.info(f"Results saved to {output_file}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        run_quick_benchmark()
    else:
        run_full_benchmark()
