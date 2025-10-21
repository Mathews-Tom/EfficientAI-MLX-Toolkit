"""MAML (Model-Agnostic Meta-Learning) algorithm implementation.

This module implements the MAML algorithm with second-order gradients for
meta-learning. MAML learns an initialization that can be quickly adapted to
new tasks with minimal gradient steps.

Reference: Finn et al. (2017) "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks"
"""

import copy
from typing import Any, Callable

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from utils.logging import get_logger

logger = get_logger(__name__)


def flatten_params(params: dict) -> dict:
    """Flatten nested parameter dictionary to flat dict.

    Args:
        params: Nested dict of parameters.

    Returns:
        Flat dict with dot-separated keys.
    """
    flat = {}

    def _flatten(d, prefix=""):
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                _flatten(v, key)
            else:
                flat[key] = v

    _flatten(params)
    return flat


def unflatten_params(flat_params: dict) -> dict:
    """Unflatten flat parameter dict to nested structure.

    Args:
        flat_params: Flat dict with dot-separated keys.

    Returns:
        Nested dict matching original structure.
    """
    nested = {}

    for key, value in flat_params.items():
        parts = key.split(".")
        current = nested

        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]

        current[parts[-1]] = value

    return nested


class MAMLLearner:
    """MAML meta-learning algorithm.

    MAML is a second-order meta-learning algorithm that learns an initialization
    that can be quickly adapted to new tasks through gradient descent. Unlike
    Reptile (first-order), MAML computes gradients through the optimization process
    itself, leading to better performance at the cost of higher computation.

    Key differences from Reptile:
    - Second-order gradients through inner loop optimization
    - Requires differentiable optimization (higher-order autodiff)
    - Better task adaptation performance
    - Higher memory and computation cost

    Reference: Finn et al. (2017) "Model-Agnostic Meta-Learning"
    """

    def __init__(
        self,
        model: nn.Module,
        inner_lr: float = 0.01,
        outer_lr: float = 0.001,
        num_inner_steps: int = 5,
        meta_batch_size: int = 4,
        first_order: bool = False,
    ):
        """Initialize MAML learner.

        Args:
            model: Neural network model to meta-learn.
            inner_lr: Learning rate for task-specific adaptation (alpha in paper).
            outer_lr: Meta-learning rate for outer loop (beta in paper).
            num_inner_steps: Number of gradient steps per task (K in paper).
            meta_batch_size: Number of tasks per meta-update.
            first_order: If True, use first-order approximation (FOMAML).
        """
        self.model = model
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.num_inner_steps = num_inner_steps
        self.meta_batch_size = meta_batch_size
        self.first_order = first_order

        # Meta-optimizer for outer loop
        self.meta_optimizer = optim.Adam(learning_rate=outer_lr)

        # Store meta-parameters
        self.meta_params = copy.deepcopy(dict(model.parameters()))

        logger.info(
            f"Initialized {'FOMAML' if first_order else 'MAML'} learner: "
            f"inner_lr={inner_lr}, outer_lr={outer_lr}, "
            f"num_inner_steps={num_inner_steps}"
        )

    def inner_loop_adaptation(
        self,
        model: nn.Module,
        support_x: mx.array,
        support_y: mx.array,
        loss_fn: Callable[[mx.array, mx.array], mx.array],
    ) -> nn.Module:
        """Perform inner loop adaptation on support set.

        Args:
            model: Model to adapt (starts from meta-parameters).
            support_x: Support set inputs.
            support_y: Support set labels.
            loss_fn: Loss function.

        Returns:
            Adapted model after K gradient steps.
        """
        # Create copy of model for task adaptation
        adapted_model = copy.deepcopy(model)

        # Inner loop: K gradient steps on support set
        for step in range(self.num_inner_steps):
            # Compute loss and gradients on support set
            def loss_and_grad_fn(m: nn.Module) -> tuple[mx.array, dict]:
                logits = m(support_x)
                loss = loss_fn(logits, support_y)
                return loss, dict(m.parameters())

            (loss, params), grads = mx.value_and_grad(loss_and_grad_fn)(
                adapted_model
            )

            # Flatten nested dicts for easier manipulation
            flat_params = flatten_params(params)
            flat_grads = flatten_params(grads)

            # Manual SGD update: θ' = θ - α * ∇L
            new_flat_params = {}
            for key in flat_params:
                new_flat_params[key] = flat_params[key] - self.inner_lr * flat_grads[key]

            # Unflatten and update model
            new_params = unflatten_params(new_flat_params)
            adapted_model.update(new_params)
            mx.eval(adapted_model.parameters())

        return adapted_model

    def compute_meta_loss(
        self,
        episodes: list[tuple[mx.array, mx.array, mx.array, mx.array]],
        loss_fn: Callable[[mx.array, mx.array], mx.array],
    ) -> tuple[mx.array, dict[str, float]]:
        """Compute meta-loss across batch of tasks.

        The meta-loss is the average query loss across all tasks after
        adaptation on their respective support sets.

        Args:
            episodes: List of (support_x, support_y, query_x, query_y) tuples.
            loss_fn: Loss function.

        Returns:
            Tuple of (meta_loss, metrics_dict).
        """
        task_losses = []
        task_accuracies = []
        support_losses = []

        for support_x, support_y, query_x, query_y in episodes:
            # Adapt to task using support set
            adapted_model = self.inner_loop_adaptation(
                self.model, support_x, support_y, loss_fn
            )

            # Compute query loss (this is what we meta-optimize)
            query_logits = adapted_model(query_x)
            query_loss = loss_fn(query_logits, query_y)
            task_losses.append(query_loss)

            # Track metrics (not part of computation graph)
            query_preds = mx.argmax(query_logits, axis=-1)
            query_acc = mx.mean((query_preds == query_y).astype(mx.float32))
            task_accuracies.append(float(query_acc))

            # Track support loss for monitoring
            support_logits = adapted_model(support_x)
            support_loss = loss_fn(support_logits, support_y)
            support_losses.append(float(support_loss))

        # Meta-loss is mean of task losses
        meta_loss = mx.mean(mx.stack(task_losses))

        # Aggregate metrics
        metrics = {
            "support_loss": sum(support_losses) / len(support_losses),
            "query_loss": float(meta_loss),
            "query_accuracy": sum(task_accuracies) / len(task_accuracies),
        }

        return meta_loss, metrics

    def meta_train_step(
        self,
        episodes: list[tuple[mx.array, mx.array, mx.array, mx.array]],
        loss_fn: Callable[[mx.array, mx.array], mx.array],
    ) -> dict[str, float]:
        """Perform one meta-training step.

        This implements the MAML outer loop update:
        1. For each task: adapt model on support set
        2. Compute query loss for each adapted model
        3. Compute meta-gradients through adaptation process
        4. Update meta-parameters

        Args:
            episodes: List of (support_x, support_y, query_x, query_y) tuples.
            loss_fn: Loss function.

        Returns:
            Dictionary of training metrics.
        """

        # Define meta-loss function for gradient computation
        def meta_loss_fn(model: nn.Module) -> tuple[mx.array, dict]:
            meta_loss, metrics = self.compute_meta_loss(episodes, loss_fn)
            return meta_loss, metrics

        # Compute meta-loss and meta-gradients
        (meta_loss, metrics), meta_grads = mx.value_and_grad(
            meta_loss_fn
        )(self.model)

        # Update meta-parameters using optimizer
        self.meta_optimizer.update(self.model, meta_grads)

        # Force evaluation of all parameters
        for k, v in self.model.parameters().items():
            if isinstance(v, dict):
                for k2, v2 in v.items():
                    mx.eval(v2)
            else:
                mx.eval(v)

        # Update stored meta-parameters
        self.meta_params = copy.deepcopy(dict(self.model.parameters()))

        # Add meta-loss to metrics
        metrics["meta_loss"] = float(meta_loss)

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
                                   Default: [0, 1, 3, 5, 10]

        Returns:
            Dictionary of evaluation metrics at different adaptation steps.
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
                    adapted_model = copy.deepcopy(self.model)

                    for _ in range(k):
                        # Compute gradients on support set
                        def loss_and_grad_fn(m: nn.Module) -> tuple[mx.array, dict]:
                            logits = m(support_x)
                            loss = loss_fn(logits, support_y)
                            return loss, dict(m.parameters())

                        (_, params), grads = mx.value_and_grad(
                            loss_and_grad_fn
                        )(adapted_model)

                        # Flatten nested dicts for easier manipulation
                        flat_params = flatten_params(params)
                        flat_grads = flatten_params(grads)

                        # Manual SGD update
                        new_flat_params = {}
                        for key in flat_params:
                            new_flat_params[key] = flat_params[key] - self.inner_lr * flat_grads[key]

                        # Unflatten and update model
                        new_params = unflatten_params(new_flat_params)
                        adapted_model.update(new_params)
                        mx.eval(adapted_model.parameters())

                    query_logits = adapted_model(query_x)

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
        # Flatten parameters for saving (mx.savez requires flat dict)
        flat_params = flatten_params(self.meta_params)
        mx.savez(path, **flat_params)
        logger.info(f"Saved MAML meta-learned parameters to {path}")

    def load(self, path: str) -> None:
        """Load meta-learned parameters.

        Args:
            path: Path to load parameters from.
        """
        # Load flat parameters and unflatten
        loaded_flat = mx.load(path)
        self.meta_params = unflatten_params(dict(loaded_flat))
        self.model.update(self.meta_params)
        mx.eval(self.model.parameters())
        logger.info(f"Loaded MAML meta-learned parameters from {path}")


class FOMAMLLearner(MAMLLearner):
    """First-Order MAML (FOMAML) learner.

    FOMAML is a simplified version of MAML that only uses first-order gradients,
    ignoring the gradient through the inner loop optimization. This makes it
    computationally cheaper while maintaining similar performance.

    The key difference is that FOMAML stops gradients after the inner loop
    adaptation, treating adapted parameters as constants when computing meta-gradients.
    """

    def __init__(
        self,
        model: nn.Module,
        inner_lr: float = 0.01,
        outer_lr: float = 0.001,
        num_inner_steps: int = 5,
        meta_batch_size: int = 4,
    ):
        """Initialize FOMAML learner.

        Args:
            model: Neural network model to meta-learn.
            inner_lr: Learning rate for task-specific adaptation.
            outer_lr: Meta-learning rate for outer loop.
            num_inner_steps: Number of gradient steps per task.
            meta_batch_size: Number of tasks per meta-update.
        """
        super().__init__(
            model=model,
            inner_lr=inner_lr,
            outer_lr=outer_lr,
            num_inner_steps=num_inner_steps,
            meta_batch_size=meta_batch_size,
            first_order=True,
        )

    def compute_meta_loss(
        self,
        episodes: list[tuple[mx.array, mx.array, mx.array, mx.array]],
        loss_fn: Callable[[mx.array, mx.array], mx.array],
    ) -> tuple[mx.array, dict[str, float]]:
        """Compute meta-loss with first-order approximation.

        FOMAML stops gradients after inner loop adaptation, making the
        computation more efficient.

        Args:
            episodes: List of (support_x, support_y, query_x, query_y) tuples.
            loss_fn: Loss function.

        Returns:
            Tuple of (meta_loss, metrics_dict).
        """
        task_losses = []
        task_accuracies = []
        support_losses = []

        for support_x, support_y, query_x, query_y in episodes:
            # Adapt to task using support set
            adapted_model = self.inner_loop_adaptation(
                self.model, support_x, support_y, loss_fn
            )

            # Stop gradients through adaptation (first-order approximation)
            # Flatten, stop gradients, and unflatten
            flat_adapted_params = flatten_params(dict(adapted_model.parameters()))
            flat_stopped_params = {
                key: mx.stop_gradient(param)
                for key, param in flat_adapted_params.items()
            }
            adapted_params = unflatten_params(flat_stopped_params)
            adapted_model.update(adapted_params)
            mx.eval(adapted_model.parameters())

            # Compute query loss
            query_logits = adapted_model(query_x)
            query_loss = loss_fn(query_logits, query_y)
            task_losses.append(query_loss)

            # Track metrics
            query_preds = mx.argmax(query_logits, axis=-1)
            query_acc = mx.mean((query_preds == query_y).astype(mx.float32))
            task_accuracies.append(float(query_acc))

            # Track support loss
            support_logits = adapted_model(support_x)
            support_loss = loss_fn(support_logits, support_y)
            support_losses.append(float(support_loss))

        # Meta-loss is mean of task losses
        meta_loss = mx.mean(mx.stack(task_losses))

        # Aggregate metrics
        metrics = {
            "support_loss": sum(support_losses) / len(support_losses),
            "query_loss": float(meta_loss),
            "query_accuracy": sum(task_accuracies) / len(task_accuracies),
        }

        return meta_loss, metrics
