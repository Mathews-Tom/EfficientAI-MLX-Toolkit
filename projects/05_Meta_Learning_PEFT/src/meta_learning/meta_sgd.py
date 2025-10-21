"""Meta-SGD: Meta-Learning with Learnable Learning Rates.

This module implements Meta-SGD, an extension of MAML that learns both the
initialization AND the learning rates for fast adaptation.

Reference:
    Li et al. (2017) "Meta-SGD: Learning to Learn Quickly for Few-Shot Learning"
"""

from __future__ import annotations

import copy
from typing import Any, Callable

import mlx.core as mx
import mlx.nn as nn

from utils.logging import get_logger

logger = get_logger(__name__)


class MetaSGDLearner:
    """Meta-SGD learner with learnable learning rates.

    Unlike MAML which uses a fixed inner learning rate, Meta-SGD learns
    a separate learning rate (alpha) for each parameter, allowing for
    more flexible adaptation.

    Attributes:
        model: Neural network model to meta-train
        meta_lr: Meta-learning rate (outer loop)
        alpha_lr: Learning rate for learning rate parameters
        num_inner_steps: Number of gradient steps per task
    """

    def __init__(
        self,
        model: nn.Module,
        meta_lr: float = 0.001,
        alpha_lr: float = 0.01,
        num_inner_steps: int = 5,
        init_inner_lr: float = 0.01,
    ):
        """Initialize Meta-SGD learner.

        Args:
            model: Neural network to meta-train
            meta_lr: Meta-learning rate for parameters (default: 0.001)
            alpha_lr: Learning rate for alpha parameters (default: 0.01)
            num_inner_steps: Steps per task adaptation (default: 5)
            init_inner_lr: Initial value for learnable learning rates (default: 0.01)
        """
        self.model = model
        self.meta_lr = meta_lr
        self.alpha_lr = alpha_lr
        self.num_inner_steps = num_inner_steps

        # Initialize learnable learning rates (alphas) for each parameter
        self.alphas = {}
        for key, param in model.parameters().items():
            # Initialize alpha with same shape as parameter
            self.alphas[key] = mx.full(param.shape, init_inner_lr)

        logger.info(
            f"Initialized Meta-SGD with {len(self.alphas)} learnable learning rates"
        )

    def inner_loop_adaptation(
        self,
        support_x: mx.array,
        support_y: mx.array,
        loss_fn: Callable,
    ) -> dict[str, mx.array]:
        """Adapt model to task using learnable learning rates.

        Args:
            support_x: Support set inputs (n_support, input_dim)
            support_y: Support set labels (n_support,)
            loss_fn: Loss function

        Returns:
            Task-adapted parameters
        """
        # Clone current parameters
        adapted_params = copy.deepcopy(self.model.parameters())

        # Inner loop: adapt with learnable learning rates
        for step in range(self.num_inner_steps):

            def compute_loss(params):
                self.model.update(params)
                logits = self.model(support_x)
                return loss_fn(logits, support_y)

            # Compute gradients
            loss, grads = mx.value_and_grad(compute_loss)(adapted_params)

            # Update with learnable learning rates (element-wise)
            adapted_params = {
                key: adapted_params[key] - self.alphas[key] * grads[key]
                for key in adapted_params.keys()
            }

        return adapted_params

    def compute_meta_loss(
        self,
        adapted_params: dict[str, mx.array],
        query_x: mx.array,
        query_y: mx.array,
        loss_fn: Callable,
    ) -> mx.array:
        """Compute loss on query set with adapted parameters.

        Args:
            adapted_params: Task-adapted parameters
            query_x: Query set inputs
            query_y: Query set labels
            loss_fn: Loss function

        Returns:
            Query loss
        """
        # Temporarily set adapted parameters
        self.model.update(adapted_params)

        # Forward pass on query set
        query_logits = self.model(query_x)
        query_loss = loss_fn(query_logits, query_y)

        return query_loss

    def meta_train_step(
        self,
        episodes: list[tuple[mx.array, mx.array, mx.array, mx.array]],
        loss_fn: Callable,
    ) -> dict[str, Any]:
        """Perform one meta-training step with learnable learning rates.

        Args:
            episodes: List of (support_x, support_y, query_x, query_y) tuples
            loss_fn: Loss function

        Returns:
            Dictionary of metrics
        """
        total_query_loss = 0.0
        param_grads = {key: mx.zeros_like(param) for key, param in self.model.parameters().items()}
        alpha_grads = {key: mx.zeros_like(alpha) for key, alpha in self.alphas.items()}

        # Process each task in batch
        for support_x, support_y, query_x, query_y in episodes:
            # Inner loop: adapt to task
            adapted_params = self.inner_loop_adaptation(
                support_x, support_y, loss_fn
            )

            # Compute meta-loss on query set
            def query_loss_fn(params, alphas):
                # Recompute adaptation with current alphas
                temp_params = copy.deepcopy(self.model.parameters())

                for step in range(self.num_inner_steps):

                    def support_loss_fn(p):
                        self.model.update(p)
                        logits = self.model(support_x)
                        return loss_fn(logits, support_y)

                    _, grads = mx.value_and_grad(support_loss_fn)(temp_params)

                    # Update with learnable alphas
                    temp_params = {
                        key: temp_params[key] - alphas[key] * grads[key]
                        for key in temp_params.keys()
                    }

                # Evaluate on query set
                self.model.update(temp_params)
                query_logits = self.model(query_x)
                return loss_fn(query_logits, query_y)

            # Compute gradients w.r.t. both parameters and alphas
            query_loss = query_loss_fn(self.model.parameters(), self.alphas)
            total_query_loss += float(query_loss)

            # Compute gradients (simplified - in practice use second-order)
            # For parameters
            def param_loss_fn(params):
                return query_loss_fn(params, self.alphas)

            _, p_grads = mx.value_and_grad(param_loss_fn)(
                self.model.parameters()
            )

            # Accumulate parameter gradients
            for key in param_grads.keys():
                param_grads[key] += p_grads[key]

            # For alphas (learning rates)
            def alpha_loss_fn(alphas):
                return query_loss_fn(self.model.parameters(), alphas)

            _, a_grads = mx.value_and_grad(alpha_loss_fn)(self.alphas)

            # Accumulate alpha gradients
            for key in alpha_grads.keys():
                alpha_grads[key] += a_grads[key]

        # Average gradients over batch
        num_tasks = len(episodes)
        for key in param_grads.keys():
            param_grads[key] /= num_tasks
        for key in alpha_grads.keys():
            alpha_grads[key] /= num_tasks

        # Meta-update: update both parameters and alphas
        updated_params = {
            key: self.model.parameters()[key] - self.meta_lr * param_grads[key]
            for key in param_grads.keys()
        }
        self.model.update(updated_params)

        # Update alphas
        self.alphas = {
            key: self.alphas[key] - self.alpha_lr * alpha_grads[key]
            for key in alpha_grads.keys()
        }

        # Clip alphas to reasonable range
        self.alphas = {
            key: mx.clip(alpha, 1e-5, 1.0) for key, alpha in self.alphas.items()
        }

        # Return metrics
        avg_query_loss = total_query_loss / num_tasks
        avg_alpha = float(
            mx.mean(
                mx.concatenate([mx.flatten(alpha) for alpha in self.alphas.values()])
            )
        )

        return {
            "query_loss": avg_query_loss,
            "avg_alpha": avg_alpha,
            "min_alpha": float(
                mx.min(
                    mx.concatenate(
                        [mx.flatten(alpha) for alpha in self.alphas.values()]
                    )
                )
            ),
            "max_alpha": float(
                mx.max(
                    mx.concatenate(
                        [mx.flatten(alpha) for alpha in self.alphas.values()]
                    )
                )
            ),
        }

    def evaluate(
        self,
        episodes: list[tuple[mx.array, mx.array, mx.array, mx.array]],
        loss_fn: Callable,
    ) -> dict[str, float]:
        """Evaluate meta-learned initialization on validation tasks.

        Args:
            episodes: Validation episodes
            loss_fn: Loss function

        Returns:
            Dictionary of evaluation metrics
        """
        total_loss = 0.0
        total_acc_before = 0.0
        total_acc_after = 0.0

        for support_x, support_y, query_x, query_y in episodes:
            # Accuracy before adaptation
            logits_before = self.model(query_x)
            acc_before = float(
                mx.mean(mx.argmax(logits_before, axis=-1) == query_y)
            )
            total_acc_before += acc_before

            # Adapt to task
            adapted_params = self.inner_loop_adaptation(
                support_x, support_y, loss_fn
            )

            # Evaluate adapted model
            self.model.update(adapted_params)
            logits_after = self.model(query_x)
            loss = float(loss_fn(logits_after, query_y))
            acc_after = float(
                mx.mean(mx.argmax(logits_after, axis=-1) == query_y)
            )

            total_loss += loss
            total_acc_after += acc_after

        num_episodes = len(episodes)

        return {
            "avg_loss": total_loss / num_episodes,
            "acc_before_adaptation": total_acc_before / num_episodes,
            "acc_after_adaptation": total_acc_after / num_episodes,
            "adaptation_improvement": (total_acc_after - total_acc_before)
            / num_episodes,
        }

    def get_learned_learning_rates(self) -> dict[str, mx.array]:
        """Get the learned learning rates (alphas).

        Returns:
            Dictionary of learned learning rates for each parameter
        """
        return copy.deepcopy(self.alphas)

    def save(self, path: str) -> None:
        """Save meta-learned parameters and learning rates.

        Args:
            path: Path to save checkpoint
        """
        checkpoint = {
            "model_params": self.model.parameters(),
            "alphas": self.alphas,
            "meta_lr": self.meta_lr,
            "alpha_lr": self.alpha_lr,
            "num_inner_steps": self.num_inner_steps,
        }

        mx.savez(path, **{k: v for k, v in checkpoint.items() if isinstance(v, (mx.array, dict))})
        logger.info(f"Saved Meta-SGD checkpoint to {path}")

    def load(self, path: str) -> None:
        """Load meta-learned parameters and learning rates.

        Args:
            path: Path to checkpoint file
        """
        checkpoint = mx.load(path)

        # Load model parameters
        if "model_params" in checkpoint:
            self.model.update(checkpoint["model_params"])

        # Load alphas
        if "alphas" in checkpoint:
            self.alphas = checkpoint["alphas"]

        logger.info(f"Loaded Meta-SGD checkpoint from {path}")


def compare_meta_sgd_vs_maml(
    meta_sgd_learner: MetaSGDLearner,
    maml_learner: Any,  # MAMLLearner
    episodes: list[tuple[mx.array, mx.array, mx.array, mx.array]],
    loss_fn: Callable,
) -> dict[str, Any]:
    """Compare Meta-SGD and MAML performance.

    Args:
        meta_sgd_learner: Meta-SGD learner
        maml_learner: MAML learner
        episodes: Test episodes
        loss_fn: Loss function

    Returns:
        Comparison metrics
    """
    # Evaluate Meta-SGD
    meta_sgd_results = meta_sgd_learner.evaluate(episodes, loss_fn)

    # Evaluate MAML
    maml_results = maml_learner.evaluate(episodes, loss_fn)

    # Compute differences
    comparison = {
        "meta_sgd_acc": meta_sgd_results["acc_after_adaptation"],
        "maml_acc": maml_results.get("step_5_acc", maml_results.get("final_acc", 0.0)),
        "meta_sgd_loss": meta_sgd_results["avg_loss"],
        "maml_loss": maml_results.get("final_loss", 0.0),
        "improvement": meta_sgd_results["acc_after_adaptation"]
        - maml_results.get("step_5_acc", 0.0),
    }

    return comparison
