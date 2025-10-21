"""Comprehensive validation suite for meta-learning PEFT system.

Validates correctness, robustness, and performance of all components:
- Meta-learning algorithms
- PEFT methods
- Task embeddings
- End-to-end workflows
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from adapter_generation import AdapterFactory
from meta_learning.maml import MAMLLearner
from meta_learning.reptile import ReptileLearner
from meta_learning.evaluation import (
    FewShotEvaluator,
    CrossTaskEvaluator,
    BaselineComparator,
)
from task_embedding.learned_embeddings import (
    TaskEmbeddingNetwork,
    Task2VecEmbedding,
    TaskSimilarityMetric,
)
from task_embedding.task_distribution import TaskDistribution, TaskConfig
from utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ValidationResult:
    """Result from a validation check."""

    component: str
    test_name: str
    passed: bool
    message: str
    metadata: dict[str, Any] | None = None


class ValidationSuite:
    """Comprehensive validation suite."""

    def __init__(self):
        """Initialize validation suite."""
        self.results: list[ValidationResult] = []

    def loss_fn(self, logits: mx.array, labels: mx.array) -> mx.array:
        """Cross-entropy loss function."""
        return nn.losses.cross_entropy(logits, labels, reduction="mean")

    def validate_meta_learning_convergence(self) -> ValidationResult:
        """Validate that meta-learning algorithms converge.

        Returns:
            Validation result
        """
        logger.info("Validating meta-learning convergence")

        try:
            # Create simple task distribution
            task_config = TaskConfig(
                name="convergence_test",
                num_classes=3,
                input_dim=5,
                num_support=5,
                num_query=10,
            )
            task_dist = TaskDistribution(task_config)

            # Create MAML learner
            from meta_learning.models import SimpleClassifier

            model = SimpleClassifier(input_dim=5, hidden_dim=16, num_classes=3)
            learner = MAMLLearner(
                model=model,
                inner_lr=0.01,
                outer_lr=0.01,
                num_inner_steps=3,
            )

            # Track loss over iterations
            losses = []
            for _ in range(20):
                episodes = [task_dist.sample_episode() for _ in range(4)]
                metrics = learner.meta_train_step(episodes, self.loss_fn)
                losses.append(metrics["meta_loss"])

            # Check if loss decreased
            initial_loss = sum(losses[:5]) / 5
            final_loss = sum(losses[-5:]) / 5

            converged = final_loss < initial_loss * 0.9

            return ValidationResult(
                component="meta_learning",
                test_name="convergence",
                passed=converged,
                message=f"Loss decreased from {initial_loss:.4f} to {final_loss:.4f}"
                if converged
                else f"Loss did not converge: {initial_loss:.4f} -> {final_loss:.4f}",
                metadata={"initial_loss": initial_loss, "final_loss": final_loss},
            )

        except Exception as e:
            return ValidationResult(
                component="meta_learning",
                test_name="convergence",
                passed=False,
                message=f"Error during convergence test: {str(e)}",
            )

    def validate_few_shot_learning(self) -> ValidationResult:
        """Validate few-shot learning capability.

        Returns:
            Validation result
        """
        logger.info("Validating few-shot learning")

        try:
            # Create task distribution
            task_config = TaskConfig(
                name="few_shot_test",
                num_classes=5,
                input_dim=10,
                num_support=5,
                num_query=15,
            )
            task_dist = TaskDistribution(task_config)

            # Create LoRA meta-learner
            learner = AdapterFactory.create_lora_meta_learner(
                input_dim=10,
                hidden_dim=32,
                output_dim=5,
                lora_rank=4,
                inner_lr=0.01,
                outer_lr=0.01,
                num_inner_steps=5,
            )

            # Meta-train
            for _ in range(30):
                episodes = [task_dist.sample_episode() for _ in range(4)]
                learner.meta_train_step(episodes, self.loss_fn)

            # Evaluate few-shot
            evaluator = FewShotEvaluator()
            test_episodes = [task_dist.sample_episode() for _ in range(10)]

            results = evaluator.evaluate_k_shot(
                learner, test_episodes, self.loss_fn, k_shots=[1, 5]
            )

            # Check if 5-shot is better than 1-shot
            one_shot_acc = results["1_shot"]["avg_acc"]
            five_shot_acc = results["5_shot"]["avg_acc"]

            passed = five_shot_acc > one_shot_acc

            return ValidationResult(
                component="few_shot_learning",
                test_name="k_shot_improvement",
                passed=passed,
                message=f"5-shot ({five_shot_acc:.4f}) > 1-shot ({one_shot_acc:.4f})"
                if passed
                else f"5-shot ({five_shot_acc:.4f}) <= 1-shot ({one_shot_acc:.4f})",
                metadata={"1_shot_acc": one_shot_acc, "5_shot_acc": five_shot_acc},
            )

        except Exception as e:
            return ValidationResult(
                component="few_shot_learning",
                test_name="k_shot_improvement",
                passed=False,
                message=f"Error during few-shot test: {str(e)}",
            )

    def validate_peft_parameter_efficiency(self) -> ValidationResult:
        """Validate that PEFT methods reduce parameters.

        Returns:
            Validation result
        """
        logger.info("Validating PEFT parameter efficiency")

        try:
            # Create LoRA learner
            lora_learner = AdapterFactory.create_lora_meta_learner(
                input_dim=10,
                hidden_dim=32,
                output_dim=5,
                lora_rank=8,
            )

            # Count parameters
            param_info = lora_learner.count_trainable_parameters()
            reduction_ratio = param_info["reduction_ratio"]

            # PEFT should reduce parameters significantly
            passed = reduction_ratio > 2.0  # At least 2x reduction

            return ValidationResult(
                component="peft",
                test_name="parameter_efficiency",
                passed=passed,
                message=f"Parameter reduction: {reduction_ratio:.1f}x"
                if passed
                else f"Insufficient reduction: {reduction_ratio:.1f}x (expected > 2x)",
                metadata=param_info,
            )

        except Exception as e:
            return ValidationResult(
                component="peft",
                test_name="parameter_efficiency",
                passed=False,
                message=f"Error during parameter efficiency test: {str(e)}",
            )

    def validate_task_embeddings(self) -> ValidationResult:
        """Validate task embedding generation.

        Returns:
            Validation result
        """
        logger.info("Validating task embeddings")

        try:
            # Create task embedding network
            embedder = TaskEmbeddingNetwork(
                input_dim=128,
                hidden_dim=256,
                embedding_dim=64,
            )

            # Generate embeddings for different inputs
            features1 = mx.random.normal((128,))
            features2 = mx.random.normal((128,))

            embedding1 = embedder(features1)
            embedding2 = embedder(features2)

            # Check shapes
            correct_shape = (
                embedding1.shape == (64,) and embedding2.shape == (64,)
            )

            # Check that different inputs give different embeddings
            different = not mx.allclose(embedding1, embedding2)

            passed = correct_shape and different

            return ValidationResult(
                component="task_embeddings",
                test_name="embedding_generation",
                passed=passed,
                message="Task embeddings generated correctly"
                if passed
                else "Task embedding generation failed",
                metadata={
                    "correct_shape": correct_shape,
                    "different_embeddings": different,
                },
            )

        except Exception as e:
            return ValidationResult(
                component="task_embeddings",
                test_name="embedding_generation",
                passed=False,
                message=f"Error during task embedding test: {str(e)}",
            )

    def validate_task_similarity(self) -> ValidationResult:
        """Validate task similarity computation.

        Returns:
            Validation result
        """
        logger.info("Validating task similarity")

        try:
            # Create embeddings
            emb1 = mx.array([1.0, 0.0, 0.0])
            emb2 = mx.array([1.0, 0.0, 0.0])  # Identical
            emb3 = mx.array([0.0, 1.0, 0.0])  # Orthogonal

            # Compute similarities
            sim_identical = TaskSimilarityMetric.cosine_similarity(emb1, emb2)
            sim_orthogonal = TaskSimilarityMetric.cosine_similarity(emb1, emb3)

            # Check: identical should be ~1, orthogonal should be ~0
            identical_correct = abs(float(sim_identical) - 1.0) < 0.01
            orthogonal_correct = abs(float(sim_orthogonal) - 0.0) < 0.01

            passed = identical_correct and orthogonal_correct

            return ValidationResult(
                component="task_similarity",
                test_name="cosine_similarity",
                passed=passed,
                message="Task similarity computed correctly"
                if passed
                else f"Similarity incorrect: identical={sim_identical:.4f}, orthogonal={sim_orthogonal:.4f}",
                metadata={
                    "sim_identical": float(sim_identical),
                    "sim_orthogonal": float(sim_orthogonal),
                },
            )

        except Exception as e:
            return ValidationResult(
                component="task_similarity",
                test_name="cosine_similarity",
                passed=False,
                message=f"Error during similarity test: {str(e)}",
            )

    def validate_cross_task_transfer(self) -> ValidationResult:
        """Validate cross-task knowledge transfer.

        Returns:
            Validation result
        """
        logger.info("Validating cross-task transfer")

        try:
            # Source task distribution
            source_config = TaskConfig(
                name="source_tasks",
                num_classes=3,
                input_dim=10,
                num_support=10,
                num_query=10,
            )
            source_dist = TaskDistribution(source_config)

            # Target task distribution (similar)
            target_config = TaskConfig(
                name="target_tasks",
                num_classes=3,
                input_dim=10,
                num_support=10,
                num_query=10,
            )
            target_dist = TaskDistribution(target_config)

            # Create learner
            from meta_learning.models import SimpleClassifier

            model = SimpleClassifier(input_dim=10, hidden_dim=16, num_classes=3)
            learner = ReptileLearner(
                model=model,
                inner_lr=0.01,
                outer_lr=0.01,
                num_inner_steps=3,
            )

            # Meta-train on source tasks
            for _ in range(20):
                episodes = [source_dist.sample_episode() for _ in range(4)]
                learner.meta_train_step(episodes, self.loss_fn)

            # Evaluate on target tasks
            target_episodes = [target_dist.sample_episode() for _ in range(10)]
            results = learner.evaluate(target_episodes, self.loss_fn)

            # Should achieve reasonable accuracy (better than random)
            random_acc = 1.0 / 3.0  # 3 classes
            achieved_acc = results["avg_acc"]

            passed = achieved_acc > random_acc * 1.5

            return ValidationResult(
                component="cross_task_transfer",
                test_name="transfer_learning",
                passed=passed,
                message=f"Transfer accuracy: {achieved_acc:.4f} (random: {random_acc:.4f})"
                if passed
                else f"Poor transfer: {achieved_acc:.4f} <= {random_acc * 1.5:.4f}",
                metadata={
                    "achieved_acc": achieved_acc,
                    "random_acc": random_acc,
                },
            )

        except Exception as e:
            return ValidationResult(
                component="cross_task_transfer",
                test_name="transfer_learning",
                passed=False,
                message=f"Error during transfer test: {str(e)}",
            )

    def validate_save_load_functionality(self) -> ValidationResult:
        """Validate model saving and loading.

        Returns:
            Validation result
        """
        logger.info("Validating save/load functionality")

        try:
            import tempfile

            # Create learner
            learner = AdapterFactory.create_lora_meta_learner(
                input_dim=10,
                hidden_dim=16,
                output_dim=5,
                lora_rank=4,
            )

            # Get initial parameters
            initial_params = {
                k: v.copy() for k, v in learner.model.parameters().items()
            }

            # Save
            with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
                checkpoint_path = f.name

            learner.save(checkpoint_path)

            # Modify parameters
            for key in learner.model.parameters().keys():
                # Perturb parameters
                learner.model.parameters()[key] = (
                    learner.model.parameters()[key] + mx.random.normal(
                        learner.model.parameters()[key].shape
                    ) * 0.1
                )

            # Load
            learner.load(checkpoint_path)

            # Check parameters match
            loaded_params = learner.model.parameters()
            params_match = all(
                mx.allclose(loaded_params[k], initial_params[k])
                for k in initial_params.keys()
            )

            # Cleanup
            Path(checkpoint_path).unlink()

            return ValidationResult(
                component="save_load",
                test_name="checkpoint_persistence",
                passed=params_match,
                message="Save/load works correctly"
                if params_match
                else "Parameters did not match after load",
            )

        except Exception as e:
            return ValidationResult(
                component="save_load",
                test_name="checkpoint_persistence",
                passed=False,
                message=f"Error during save/load test: {str(e)}",
            )

    def run_full_validation(self) -> dict[str, list[ValidationResult]]:
        """Run full validation suite.

        Returns:
            Dictionary of validation results by component
        """
        logger.info("Starting comprehensive validation suite")

        # Run all validation tests
        self.results = [
            self.validate_meta_learning_convergence(),
            self.validate_few_shot_learning(),
            self.validate_peft_parameter_efficiency(),
            self.validate_task_embeddings(),
            self.validate_task_similarity(),
            self.validate_cross_task_transfer(),
            self.validate_save_load_functionality(),
        ]

        # Group by component
        by_component: dict[str, list[ValidationResult]] = {}
        for result in self.results:
            if result.component not in by_component:
                by_component[result.component] = []
            by_component[result.component].append(result)

        return by_component

    def generate_validation_report(self) -> str:
        """Generate validation report.

        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 80)
        report.append("META-LEARNING PEFT VALIDATION REPORT")
        report.append("=" * 80)
        report.append("")

        # Summary
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.passed)
        failed_tests = total_tests - passed_tests

        report.append("SUMMARY")
        report.append("-" * 80)
        report.append(f"Total tests:  {total_tests}")
        report.append(f"Passed:       {passed_tests} ({passed_tests/total_tests*100:.1f}%)")
        report.append(f"Failed:       {failed_tests}")
        report.append("")

        # Detailed results
        report.append("DETAILED RESULTS")
        report.append("=" * 80)

        for result in self.results:
            status = "✓ PASS" if result.passed else "✗ FAIL"
            report.append(f"\n[{status}] {result.component} - {result.test_name}")
            report.append(f"      {result.message}")

        report.append("\n" + "=" * 80)

        return "\n".join(report)


def run_validation() -> None:
    """Run validation suite."""
    suite = ValidationSuite()
    results = suite.run_full_validation()

    print(suite.generate_validation_report())

    # Save results
    import json
    from pathlib import Path

    output_dir = Path(__file__).parent.parent / "outputs" / "validation"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert results to JSON
    results_json = {
        component: [
            {
                "test_name": r.test_name,
                "passed": r.passed,
                "message": r.message,
                "metadata": r.metadata,
            }
            for r in results_list
        ]
        for component, results_list in results.items()
    }

    output_file = output_dir / "validation_results.json"
    with open(output_file, "w") as f:
        json.dump(results_json, f, indent=2)

    logger.info(f"Validation results saved to {output_file}")


if __name__ == "__main__":
    run_validation()
