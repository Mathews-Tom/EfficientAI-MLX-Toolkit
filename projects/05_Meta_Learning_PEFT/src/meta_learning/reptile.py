"""Reptile meta-learning algorithm implementation."""

import copy
from typing import Any, Callable

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from meta_learning.maml import flatten_params, unflatten_params
from utils.logging import get_logger

logger = get_logger(__name__)


class ReptileLearner:
    """Reptile meta-learning algorithm.

    Reptile is a first-order meta-learning algorithm that learns an initialization
    that can be quickly adapted to new tasks with minimal gradient steps.

    Reference: Nichol et al. (2018) "On First-Order Meta-Learning Algorithms"
    """

    def __init__(
        self,
        model: nn.Module,
        inner_lr: float = 0.01,
        outer_lr: float = 0.001,
        num_inner_steps: int = 5,
        meta_batch_size: int = 4,
    ):
        """Initialize Reptile learner.

        Args:
            model: Neural network model to meta-learn.
            inner_lr: Learning rate for task-specific adaptation.
            outer_lr: Meta-learning rate for outer loop.
            num_inner_steps: Number of gradient steps per task.
            meta_batch_size: Number of tasks per meta-update.
        """
        self.model = model
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.num_inner_steps = num_inner_steps
        self.meta_batch_size = meta_batch_size

        # Store initial parameters
        self.meta_params = copy.deepcopy(dict(model.parameters()))

        logger.info(
            f"Initialized Reptile learner: inner_lr={inner_lr}, "
            f"outer_lr={outer_lr}, num_inner_steps={num_inner_steps}"
        )

    def inner_loop(
        self,
        support_x: mx.array,
        support_y: mx.array,
        loss_fn: Callable[[mx.array, mx.array], mx.array],
    ) -> dict[str, mx.array]:
        """Perform inner loop adaptation on a single task.

        Args:
            support_x: Support set inputs.
            support_y: Support set labels.
            loss_fn: Loss function.

        Returns:
            Task-adapted parameters.
        """
        # Create task-specific model (copy of meta-model)
        task_model = copy.deepcopy(self.model)

        # Task-specific optimizer
        task_optimizer = optim.SGD(learning_rate=self.inner_lr)

        # Inner loop adaptation
        for step in range(self.num_inner_steps):
            # Forward pass
            def loss_and_grad_fn(model: nn.Module) -> tuple[mx.array, dict]:
                logits = model(support_x)
                loss = loss_fn(logits, support_y)
                return loss, dict(model.parameters())

            # Compute loss and gradients
            (loss, _), grads = mx.value_and_grad(loss_and_grad_fn)(
                task_model
            )

            # Update task-specific parameters
            task_optimizer.update(task_model, grads)
            mx.eval(task_model.parameters())

        return dict(task_model.parameters())

    def outer_loop_update(self, task_params_list: list[dict[str, mx.array]]) -> None:
        """Perform outer loop meta-update using Reptile.

        The Reptile update rule is:
        θ_new = θ + β * (θ_task - θ)

        Where θ is meta-parameters and θ_task is task-adapted parameters.

        Args:
            task_params_list: List of task-adapted parameter dictionaries.
        """
        # Flatten all parameter dicts for easier manipulation
        flat_meta_params = flatten_params(self.meta_params)
        flat_task_params_list = [flatten_params(tp) for tp in task_params_list]

        # Compute mean of task-adapted parameters
        mean_task_params = {}
        for key in flat_meta_params.keys():
            task_param_stack = mx.stack(
                [task_params[key] for task_params in flat_task_params_list]
            )
            mean_task_params[key] = mx.mean(task_param_stack, axis=0)

        # Reptile update: θ_new = θ + β * (mean(θ_task) - θ)
        flat_new_params = {}
        for key in flat_meta_params.keys():
            flat_new_params[key] = flat_meta_params[key] + self.outer_lr * (
                mean_task_params[key] - flat_meta_params[key]
            )

        # Unflatten and update model parameters
        self.meta_params = unflatten_params(flat_new_params)
        self.model.update(self.meta_params)
        mx.eval(self.model.parameters())

    def meta_train_step(
        self,
        episodes: list[tuple[mx.array, mx.array, mx.array, mx.array]],
        loss_fn: Callable[[mx.array, mx.array], mx.array],
    ) -> dict[str, float]:
        """Perform one meta-training step.

        Args:
            episodes: List of (support_x, support_y, query_x, query_y) tuples.
            loss_fn: Loss function.

        Returns:
            Dictionary of metrics.
        """
        task_params_list = []
        inner_losses = []
        query_losses = []
        query_accuracies = []

        # Inner loop for each task
        for support_x, support_y, query_x, query_y in episodes:
            # Adapt to task
            task_params = self.inner_loop(support_x, support_y, loss_fn)
            task_params_list.append(task_params)

            # Evaluate on query set (for monitoring)
            task_model = copy.deepcopy(self.model)
            task_model.update(task_params)
            query_logits = task_model(query_x)
            query_loss = loss_fn(query_logits, query_y)
            query_losses.append(float(query_loss))

            # Track query accuracy
            query_preds = mx.argmax(query_logits, axis=-1)
            query_acc = mx.mean((query_preds == query_y).astype(mx.float32))
            query_accuracies.append(float(query_acc))

            # Track inner loop final loss
            support_logits = task_model(support_x)
            support_loss = loss_fn(support_logits, support_y)
            inner_losses.append(float(support_loss))

        # Outer loop meta-update
        self.outer_loop_update(task_params_list)

        # Return metrics
        metrics = {
            "inner_loss": sum(inner_losses) / len(inner_losses),
            "query_loss": sum(query_losses) / len(query_losses),
            "query_accuracy": sum(query_accuracies) / len(query_accuracies),
        }

        return metrics

    def evaluate(
        self,
        episodes: list[tuple[mx.array, mx.array, mx.array, mx.array]],
        loss_fn: Callable[[mx.array, mx.array], mx.array],
        num_adaptation_steps: list[int] | None = None,
    ) -> dict[str, Any]:
        """Evaluate meta-learned model on validation tasks.

        Args:
            episodes: List of validation episodes.
            loss_fn: Loss function.
            num_adaptation_steps: List of adaptation steps to evaluate.

        Returns:
            Dictionary of evaluation metrics.
        """
        if num_adaptation_steps is None:
            num_adaptation_steps = [0, 1, 3, 5, 10]

        results: dict[str, Any] = {f"step_{k}": [] for k in num_adaptation_steps}

        for support_x, support_y, query_x, query_y in episodes:
            # Evaluate at different adaptation steps
            for k in num_adaptation_steps:
                if k == 0:
                    # Zero-shot (no adaptation)
                    query_logits = self.model(query_x)
                else:
                    # k-step adaptation
                    task_model = copy.deepcopy(self.model)
                    task_optimizer = optim.SGD(learning_rate=self.inner_lr)

                    for _ in range(k):
                        def loss_and_grad_fn(model: nn.Module) -> tuple[mx.array, dict]:
                            logits = model(support_x)
                            loss = loss_fn(logits, support_y)
                            return loss, dict(model.parameters())

                        (_, _), grads = mx.value_and_grad(
                            loss_and_grad_fn
                        )(task_model)
                        task_optimizer.update(task_model, grads)
                        mx.eval(task_model.parameters())

                    query_logits = task_model(query_x)

                # Compute loss and accuracy
                query_loss = loss_fn(query_logits, query_y)
                query_preds = mx.argmax(query_logits, axis=-1)
                query_acc = mx.mean((query_preds == query_y).astype(mx.float32))

                results[f"step_{k}"].append(
                    {"loss": float(query_loss), "accuracy": float(query_acc)}
                )

        # Aggregate results
        aggregated = {}
        for k in num_adaptation_steps:
            step_results = results[f"step_{k}"]
            aggregated[f"step_{k}_loss"] = sum(r["loss"] for r in step_results) / len(
                step_results
            )
            aggregated[f"step_{k}_acc"] = sum(
                r["accuracy"] for r in step_results
            ) / len(step_results)

        return aggregated

    def save(self, path: str) -> None:
        """Save meta-learned parameters.

        Args:
            path: Path to save parameters.
        """
        mx.save_safetensors(path, self.meta_params)
        logger.info(f"Saved meta-learned parameters to {path}")

    def load(self, path: str) -> None:
        """Load meta-learned parameters.

        Args:
            path: Path to load parameters from.
        """
        self.meta_params = mx.load(path)
        self.model.update(self.meta_params)
        mx.eval(self.model.parameters())
        logger.info(f"Loaded meta-learned parameters from {path}")
