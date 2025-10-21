"""Baseline performance measurement utilities."""

import time
from typing import Any, Callable

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from meta_learning.models import cross_entropy_loss
from utils.logging import get_logger

logger = get_logger(__name__)


class BaselineEvaluator:
    """Evaluate baseline few-shot performance without meta-learning."""

    def __init__(
        self,
        model_factory: Callable[[], nn.Module],
        learning_rate: float = 0.01,
        num_steps: int = 100,
    ):
        """Initialize baseline evaluator.

        Args:
            model_factory: Factory function to create fresh model instances.
            learning_rate: Learning rate for baseline training.
            num_steps: Number of training steps per task.
        """
        self.model_factory = model_factory
        self.learning_rate = learning_rate
        self.num_steps = num_steps

    def evaluate_task(
        self,
        support_x: mx.array,
        support_y: mx.array,
        query_x: mx.array,
        query_y: mx.array,
        loss_fn: Callable[[mx.array, mx.array], mx.array] = cross_entropy_loss,
    ) -> dict[str, Any]:
        """Evaluate baseline performance on a single task.

        Args:
            support_x: Support set inputs.
            support_y: Support set labels.
            query_x: Query set inputs.
            query_y: Query set labels.
            loss_fn: Loss function.

        Returns:
            Dictionary of metrics.
        """
        # Create fresh model
        model = self.model_factory()
        optimizer = optim.SGD(learning_rate=self.learning_rate)

        # Track metrics
        start_time = time.time()
        train_losses = []
        query_losses = []
        query_accs = []

        # Training loop
        for step in range(self.num_steps):
            # Forward pass and gradient computation
            def loss_and_grad_fn(model: nn.Module) -> tuple[mx.array, dict]:
                logits = model(support_x)
                loss = loss_fn(logits, support_y)
                return loss, dict(model.parameters())

            (train_loss, _), grads = mx.value_and_grad(loss_and_grad_fn)(
                model
            )

            # Update parameters
            optimizer.update(model, grads)
            mx.eval(model.parameters())

            train_losses.append(float(train_loss))

            # Evaluate on query set (every 10 steps)
            if step % 10 == 0 or step == self.num_steps - 1:
                query_logits = model(query_x)
                query_loss = loss_fn(query_logits, query_y)
                query_preds = mx.argmax(query_logits, axis=-1)
                query_acc = mx.mean((query_preds == query_y).astype(mx.float32))

                query_losses.append(float(query_loss))
                query_accs.append(float(query_acc))

        elapsed_time = time.time() - start_time

        # Final evaluation
        final_query_logits = model(query_x)
        final_query_loss = loss_fn(final_query_logits, query_y)
        final_query_preds = mx.argmax(final_query_logits, axis=-1)
        final_query_acc = mx.mean(
            (final_query_preds == query_y).astype(mx.float32)
        )

        return {
            "final_query_loss": float(final_query_loss),
            "final_query_acc": float(final_query_acc),
            "train_losses": train_losses,
            "query_losses": query_losses,
            "query_accs": query_accs,
            "elapsed_time": elapsed_time,
            "num_steps": self.num_steps,
        }

    def evaluate_tasks(
        self,
        episodes: list[tuple[mx.array, mx.array, mx.array, mx.array]],
        loss_fn: Callable[[mx.array, mx.array], mx.array] = cross_entropy_loss,
    ) -> dict[str, Any]:
        """Evaluate baseline on multiple tasks.

        Args:
            episodes: List of (support_x, support_y, query_x, query_y) tuples.
            loss_fn: Loss function.

        Returns:
            Aggregated metrics.
        """
        results = []

        logger.info(f"Evaluating baseline on {len(episodes)} tasks...")

        for i, (support_x, support_y, query_x, query_y) in enumerate(episodes):
            task_results = self.evaluate_task(
                support_x, support_y, query_x, query_y, loss_fn
            )
            results.append(task_results)

            if (i + 1) % 10 == 0:
                logger.info(f"Evaluated {i + 1}/{len(episodes)} tasks")

        # Aggregate results
        aggregated = {
            "mean_query_loss": sum(r["final_query_loss"] for r in results)
            / len(results),
            "mean_query_acc": sum(r["final_query_acc"] for r in results)
            / len(results),
            "mean_elapsed_time": sum(r["elapsed_time"] for r in results)
            / len(results),
            "std_query_acc": self._std(
                [r["final_query_acc"] for r in results]
            ),
            "num_tasks": len(episodes),
        }

        logger.info(
            f"Baseline results: "
            f"acc={aggregated['mean_query_acc']:.3f} ± {aggregated['std_query_acc']:.3f}, "
            f"loss={aggregated['mean_query_loss']:.3f}"
        )

        return aggregated

    def _std(self, values: list[float]) -> float:
        """Compute standard deviation."""
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance**0.5


def compare_meta_vs_baseline(
    meta_results: dict[str, float],
    baseline_results: dict[str, float],
) -> dict[str, Any]:
    """Compare meta-learning vs baseline results.

    Args:
        meta_results: Meta-learning evaluation results.
        baseline_results: Baseline evaluation results.

    Returns:
        Comparison metrics.
    """
    comparison = {
        "meta_acc": meta_results.get("step_5_acc", 0.0),
        "baseline_acc": baseline_results.get("mean_query_acc", 0.0),
        "meta_loss": meta_results.get("step_5_loss", 0.0),
        "baseline_loss": baseline_results.get("mean_query_loss", 0.0),
    }

    # Compute improvement
    if baseline_results.get("mean_query_acc", 0.0) > 0:
        acc_improvement = (
            comparison["meta_acc"] - comparison["baseline_acc"]
        ) / comparison["baseline_acc"]
        comparison["acc_improvement_pct"] = acc_improvement * 100
    else:
        comparison["acc_improvement_pct"] = 0.0

    # Speed improvement (if available)
    if "mean_elapsed_time" in baseline_results and "adaptation_time" in meta_results:
        time_ratio = baseline_results["mean_elapsed_time"] / meta_results.get(
            "adaptation_time", 1.0
        )
        comparison["speedup"] = time_ratio
    else:
        comparison["speedup"] = None

    logger.info(
        f"Meta-learning vs Baseline: "
        f"acc improvement = {comparison['acc_improvement_pct']:.1f}%"
    )

    return comparison
