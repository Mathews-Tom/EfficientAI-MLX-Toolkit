"""Comprehensive evaluation utilities for meta-learning.

This module provides utilities for evaluating meta-learned models including:
    - Few-shot performance evaluation
    - Learning curve analysis
    - Cross-task transfer evaluation
    - Comparison with baselines
    - Statistical significance testing
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class EvaluationMetrics:
    """Evaluation metrics for meta-learning.

    Attributes:
        accuracy: Classification accuracy
        loss: Loss value
        adaptation_steps: Number of adaptation steps taken
        adaptation_time: Time taken for adaptation (seconds)
        query_samples: Number of query samples evaluated
    """

    accuracy: float
    loss: float
    adaptation_steps: int
    adaptation_time: float | None = None
    query_samples: int | None = None


class FewShotEvaluator:
    """Evaluator for few-shot learning performance."""

    @staticmethod
    def evaluate_k_shot(
        learner: Any,
        episodes: list[tuple[mx.array, mx.array, mx.array, mx.array]],
        loss_fn: Callable,
        k_shots: list[int] = [1, 5, 10],
    ) -> dict[int, EvaluationMetrics]:
        """Evaluate performance at different K-shot settings.

        Args:
            learner: Meta-learning learner
            episodes: Test episodes
            loss_fn: Loss function
            k_shots: List of K values to evaluate

        Returns:
            Dictionary mapping K to evaluation metrics
        """
        results = {}

        for k in k_shots:
            # Filter episodes to use only K support examples
            k_shot_episodes = []
            for support_x, support_y, query_x, query_y in episodes:
                # Take first K examples from support set
                k_support_x = support_x[:k]
                k_support_y = support_y[:k]
                k_shot_episodes.append((k_support_x, k_support_y, query_x, query_y))

            # Evaluate
            eval_results = learner.evaluate(k_shot_episodes, loss_fn)

            # Extract metrics
            acc = eval_results.get(
                "step_5_acc",
                eval_results.get("acc_after_adaptation", 0.0),
            )
            loss = eval_results.get("final_loss", eval_results.get("avg_loss", 0.0))

            results[k] = EvaluationMetrics(
                accuracy=acc,
                loss=loss,
                adaptation_steps=learner.num_inner_steps,
                query_samples=len(query_x) if len(k_shot_episodes) > 0 else None,
            )

            logger.info(f"{k}-shot: acc={acc:.4f}, loss={loss:.4f}")

        return results

    @staticmethod
    def learning_curve(
        learner: Any,
        episode: tuple[mx.array, mx.array, mx.array, mx.array],
        loss_fn: Callable,
        max_steps: int = 20,
    ) -> dict[str, list[float]]:
        """Compute learning curve showing adaptation over steps.

        Args:
            learner: Meta-learning learner
            episode: Single test episode
            loss_fn: Loss function
            max_steps: Maximum adaptation steps

        Returns:
            Dictionary with 'steps', 'accuracy', 'loss' lists
        """
        support_x, support_y, query_x, query_y = episode

        steps = []
        accuracies = []
        losses = []

        # Evaluate before adaptation
        logits = learner.model(query_x)
        acc_0 = float(mx.mean(mx.argmax(logits, axis=-1) == query_y))
        loss_0 = float(loss_fn(logits, query_y))

        steps.append(0)
        accuracies.append(acc_0)
        losses.append(loss_0)

        # Adapt incrementally
        import copy

        adapted_params = copy.deepcopy(learner.model.parameters())

        for step in range(1, max_steps + 1):
            # Single adaptation step
            def compute_loss(params):
                learner.model.update(params)
                logits = learner.model(support_x)
                return loss_fn(logits, support_y)

            loss, grads = mx.value_and_grad(compute_loss)(adapted_params)

            # Update parameters
            adapted_params = {
                key: adapted_params[key] - learner.inner_lr * grads[key]
                for key in adapted_params.keys()
            }

            # Evaluate on query set
            learner.model.update(adapted_params)
            query_logits = learner.model(query_x)
            query_acc = float(mx.mean(mx.argmax(query_logits, axis=-1) == query_y))
            query_loss = float(loss_fn(query_logits, query_y))

            steps.append(step)
            accuracies.append(query_acc)
            losses.append(query_loss)

        return {"steps": steps, "accuracy": accuracies, "loss": losses}


class CrossTaskEvaluator:
    """Evaluator for cross-task transfer."""

    @staticmethod
    def evaluate_transfer(
        learner: Any,
        source_episodes: list[tuple[mx.array, mx.array, mx.array, mx.array]],
        target_episodes: list[tuple[mx.array, mx.array, mx.array, mx.array]],
        loss_fn: Callable,
    ) -> dict[str, float]:
        """Evaluate transfer from source to target tasks.

        Args:
            learner: Meta-learning learner
            source_episodes: Source task episodes for meta-training
            target_episodes: Target task episodes for evaluation
            loss_fn: Loss function

        Returns:
            Transfer evaluation metrics
        """
        # Meta-train on source tasks
        for _ in range(10):  # Quick meta-training
            learner.meta_train_step(source_episodes, loss_fn)

        # Evaluate on target tasks
        target_results = learner.evaluate(target_episodes, loss_fn)

        # Evaluate random init on target tasks
        import copy

        random_model = copy.deepcopy(learner.model)
        # Re-initialize parameters randomly
        for param in random_model.parameters().values():
            param = mx.random.normal(param.shape)

        # Create temporary learner with random init
        temp_learner = copy.deepcopy(learner)
        temp_learner.model = random_model
        random_results = temp_learner.evaluate(target_episodes, loss_fn)

        # Compute transfer gain
        transfer_gain = target_results.get("step_5_acc", 0.0) - random_results.get(
            "step_5_acc", 0.0
        )

        return {
            "target_acc_with_transfer": target_results.get("step_5_acc", 0.0),
            "target_acc_without_transfer": random_results.get("step_5_acc", 0.0),
            "transfer_gain": transfer_gain,
        }


class BaselineComparator:
    """Compare meta-learning with baselines."""

    @staticmethod
    def compare_with_scratch(
        meta_learner: Any,
        episodes: list[tuple[mx.array, mx.array, mx.array, mx.array]],
        loss_fn: Callable,
        scratch_training_steps: int = 100,
    ) -> dict[str, Any]:
        """Compare meta-learning with training from scratch.

        Args:
            meta_learner: Meta-learned model
            episodes: Test episodes
            loss_fn: Loss function
            scratch_training_steps: Training steps for scratch model

        Returns:
            Comparison metrics
        """
        # Evaluate meta-learned model
        meta_results = meta_learner.evaluate(episodes, loss_fn)
        meta_acc = meta_results.get("step_5_acc", meta_results.get("acc_after_adaptation", 0.0))

        # Train from scratch
        import copy

        scratch_accs = []

        for support_x, support_y, query_x, query_y in episodes:
            # Initialize random model
            scratch_model = copy.deepcopy(meta_learner.model)

            # Train on support set
            params = scratch_model.parameters()
            for step in range(scratch_training_steps):

                def loss_fn_scratch(p):
                    scratch_model.update(p)
                    logits = scratch_model(support_x)
                    return loss_fn(logits, support_y)

                loss, grads = mx.value_and_grad(loss_fn_scratch)(params)
                params = {
                    key: params[key] - 0.01 * grads[key] for key in params.keys()
                }

            # Evaluate
            scratch_model.update(params)
            logits = scratch_model(query_x)
            acc = float(mx.mean(mx.argmax(logits, axis=-1) == query_y))
            scratch_accs.append(acc)

        avg_scratch_acc = np.mean(scratch_accs)

        return {
            "meta_learning_acc": meta_acc,
            "scratch_acc": avg_scratch_acc,
            "improvement": meta_acc - avg_scratch_acc,
            "speedup": scratch_training_steps / meta_learner.num_inner_steps,
        }


class StatisticalTester:
    """Statistical significance testing for meta-learning."""

    @staticmethod
    def paired_t_test(
        method1_scores: list[float], method2_scores: list[float], alpha: float = 0.05
    ) -> dict[str, Any]:
        """Perform paired t-test between two methods.

        Args:
            method1_scores: Scores from method 1
            method2_scores: Scores from method 2
            alpha: Significance level

        Returns:
            Test results
        """
        from scipy import stats

        # Compute differences
        differences = np.array(method1_scores) - np.array(method2_scores)

        # Paired t-test
        t_stat, p_value = stats.ttest_rel(method1_scores, method2_scores)

        is_significant = p_value < alpha

        return {
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "is_significant": is_significant,
            "alpha": alpha,
            "mean_difference": float(np.mean(differences)),
            "std_difference": float(np.std(differences)),
        }

    @staticmethod
    def bootstrap_confidence_interval(
        scores: list[float], confidence: float = 0.95, num_bootstrap: int = 1000
    ) -> dict[str, float]:
        """Compute bootstrap confidence interval.

        Args:
            scores: Performance scores
            confidence: Confidence level (default: 0.95)
            num_bootstrap: Number of bootstrap samples

        Returns:
            Confidence interval bounds
        """
        scores_array = np.array(scores)
        bootstrap_means = []

        for _ in range(num_bootstrap):
            # Resample with replacement
            sample = np.random.choice(scores_array, size=len(scores_array), replace=True)
            bootstrap_means.append(np.mean(sample))

        # Compute percentiles
        alpha = 1 - confidence
        lower = np.percentile(bootstrap_means, 100 * alpha / 2)
        upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))

        return {
            "mean": float(np.mean(scores)),
            "lower_bound": float(lower),
            "upper_bound": float(upper),
            "confidence": confidence,
        }


def comprehensive_evaluation(
    learner: Any,
    train_episodes: list[tuple[mx.array, mx.array, mx.array, mx.array]],
    val_episodes: list[tuple[mx.array, mx.array, mx.array, mx.array]],
    test_episodes: list[tuple[mx.array, mx.array, mx.array, mx.array]],
    loss_fn: Callable,
) -> dict[str, Any]:
    """Run comprehensive evaluation suite.

    Args:
        learner: Meta-learning learner
        train_episodes: Training episodes
        val_episodes: Validation episodes
        test_episodes: Test episodes
        loss_fn: Loss function

    Returns:
        Comprehensive evaluation results
    """
    logger.info("Running comprehensive evaluation...")

    results = {}

    # 1. Few-shot performance
    logger.info("Evaluating few-shot performance...")
    few_shot_results = FewShotEvaluator.evaluate_k_shot(
        learner, test_episodes, loss_fn, k_shots=[1, 5, 10]
    )
    results["few_shot"] = {k: v.__dict__ for k, v in few_shot_results.items()}

    # 2. Learning curves
    logger.info("Computing learning curves...")
    if len(test_episodes) > 0:
        learning_curve = FewShotEvaluator.learning_curve(
            learner, test_episodes[0], loss_fn, max_steps=20
        )
        results["learning_curve"] = learning_curve

    # 3. Cross-task transfer
    logger.info("Evaluating cross-task transfer...")
    if len(train_episodes) > 0 and len(test_episodes) > 0:
        transfer_results = CrossTaskEvaluator.evaluate_transfer(
            learner, train_episodes[:5], test_episodes[:5], loss_fn
        )
        results["transfer"] = transfer_results

    # 4. Baseline comparison
    logger.info("Comparing with baseline...")
    baseline_results = BaselineComparator.compare_with_scratch(
        learner, test_episodes[:10], loss_fn
    )
    results["baseline_comparison"] = baseline_results

    logger.info("Comprehensive evaluation complete!")

    return results
