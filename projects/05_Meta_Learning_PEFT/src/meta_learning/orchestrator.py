"""Meta-training orchestrator for unified meta-learning workflows.

This module provides a high-level orchestrator for meta-training that handles:
    - Multiple meta-learning algorithms (Reptile, MAML, Meta-SGD)
    - Task distribution management
    - Training loop with logging and checkpointing
    - Evaluation and early stopping
    - Experiment tracking
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable

import mlx.core as mx
import mlx.nn as nn

from utils.logging import get_logger

logger = get_logger(__name__)


class MetaLearningAlgorithm(Enum):
    """Supported meta-learning algorithms."""

    REPTILE = "reptile"
    MAML = "maml"
    FOMAML = "fomaml"
    META_SGD = "meta_sgd"


@dataclass
class MetaTrainingConfig:
    """Configuration for meta-training.

    Attributes:
        algorithm: Meta-learning algorithm to use
        num_iterations: Total meta-training iterations
        meta_batch_size: Number of tasks per meta-update
        inner_lr: Inner loop learning rate (task adaptation)
        outer_lr: Outer loop learning rate (meta-update)
        num_inner_steps: Gradient steps per task
        eval_interval: Iterations between evaluations
        save_interval: Iterations between checkpoints
        early_stopping_patience: Patience for early stopping (None to disable)
        checkpoint_dir: Directory for checkpoints
        log_dir: Directory for logs
    """

    algorithm: str = "reptile"
    num_iterations: int = 1000
    meta_batch_size: int = 4
    inner_lr: float = 0.01
    outer_lr: float = 0.001
    num_inner_steps: int = 5
    eval_interval: int = 100
    save_interval: int = 500
    early_stopping_patience: int | None = None
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"


class MetaLearningOrchestrator:
    """Orchestrator for meta-training workflows.

    Handles the complete meta-training pipeline including:
        - Learner initialization
        - Training loop management
        - Logging and checkpointing
        - Evaluation and early stopping
    """

    def __init__(
        self,
        model: nn.Module,
        train_task_distribution: Any,  # TaskDistribution
        val_task_distribution: Any,  # TaskDistribution
        loss_fn: Callable,
        config: MetaTrainingConfig,
    ):
        """Initialize orchestrator.

        Args:
            model: Neural network model to meta-train
            train_task_distribution: Training task distribution
            val_task_distribution: Validation task distribution
            loss_fn: Loss function
            config: Meta-training configuration
        """
        self.model = model
        self.train_dist = train_task_distribution
        self.val_dist = val_task_distribution
        self.loss_fn = loss_fn
        self.config = config

        # Initialize learner based on algorithm
        self.learner = self._create_learner()

        # Training state
        self.best_val_metric = float("inf")
        self.patience_counter = 0
        self.training_history = {
            "train_loss": [],
            "val_loss": [],
            "val_acc": [],
            "iteration": [],
        }

        # Create directories
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        if config.log_dir:
            Path(config.log_dir).mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized MetaLearningOrchestrator with {config.algorithm}")

    def _create_learner(self):
        """Create meta-learning learner based on config."""
        from src.meta_learning.reptile import ReptileLearner
        from src.meta_learning.maml import MAMLLearner, FOMAMLLearner
        from src.meta_learning.meta_sgd import MetaSGDLearner

        if self.config.algorithm == "reptile":
            return ReptileLearner(
                model=self.model,
                inner_lr=self.config.inner_lr,
                outer_lr=self.config.outer_lr,
                num_inner_steps=self.config.num_inner_steps,
            )
        elif self.config.algorithm == "maml":
            return MAMLLearner(
                model=self.model,
                inner_lr=self.config.inner_lr,
                meta_lr=self.config.outer_lr,
                num_inner_steps=self.config.num_inner_steps,
            )
        elif self.config.algorithm == "fomaml":
            return FOMAMLLearner(
                model=self.model,
                inner_lr=self.config.inner_lr,
                meta_lr=self.config.outer_lr,
                num_inner_steps=self.config.num_inner_steps,
            )
        elif self.config.algorithm == "meta_sgd":
            return MetaSGDLearner(
                model=self.model,
                meta_lr=self.config.outer_lr,
                alpha_lr=self.config.inner_lr,
                num_inner_steps=self.config.num_inner_steps,
            )
        else:
            raise ValueError(f"Unknown algorithm: {self.config.algorithm}")

    def sample_task_batch(
        self, task_distribution: Any, batch_size: int
    ) -> list[tuple[mx.array, mx.array, mx.array, mx.array]]:
        """Sample batch of tasks from distribution.

        Args:
            task_distribution: Task distribution to sample from
            batch_size: Number of tasks to sample

        Returns:
            List of (support_x, support_y, query_x, query_y) tuples
        """
        episodes = []
        for _ in range(batch_size):
            _, support_x, support_y, query_x, query_y = task_distribution.sample_episode()
            episodes.append((support_x, support_y, query_x, query_y))
        return episodes

    def train_step(self, iteration: int) -> dict[str, Any]:
        """Perform single meta-training step.

        Args:
            iteration: Current iteration number

        Returns:
            Training metrics
        """
        # Sample batch of tasks
        episodes = self.sample_task_batch(
            self.train_dist, self.config.meta_batch_size
        )

        # Meta-train step
        start_time = time.time()
        metrics = self.learner.meta_train_step(episodes, self.loss_fn)
        step_time = time.time() - start_time

        metrics["step_time"] = step_time
        metrics["iteration"] = iteration

        return metrics

    def evaluate(self, num_eval_tasks: int = 20) -> dict[str, float]:
        """Evaluate on validation tasks.

        Args:
            num_eval_tasks: Number of validation tasks to evaluate on

        Returns:
            Evaluation metrics
        """
        # Sample validation episodes
        val_episodes = self.sample_task_batch(self.val_dist, num_eval_tasks)

        # Evaluate
        eval_results = self.learner.evaluate(val_episodes, self.loss_fn)

        return eval_results

    def save_checkpoint(self, iteration: int, is_best: bool = False) -> None:
        """Save checkpoint.

        Args:
            iteration: Current iteration
            is_best: Whether this is the best checkpoint so far
        """
        checkpoint_path = Path(self.config.checkpoint_dir) / f"checkpoint_iter_{iteration}.npz"
        self.learner.save(str(checkpoint_path))

        if is_best:
            best_path = Path(self.config.checkpoint_dir) / "best.npz"
            self.learner.save(str(best_path))
            logger.info(f"Saved best checkpoint at iteration {iteration}")

    def check_early_stopping(self, val_metric: float) -> bool:
        """Check if early stopping criteria met.

        Args:
            val_metric: Current validation metric

        Returns:
            True if should stop training
        """
        if self.config.early_stopping_patience is None:
            return False

        if val_metric < self.best_val_metric:
            self.best_val_metric = val_metric
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.config.early_stopping_patience:
                logger.info(
                    f"Early stopping triggered after {self.patience_counter} iterations without improvement"
                )
                return True
            return False

    def train(self) -> dict[str, Any]:
        """Run complete meta-training loop.

        Returns:
            Training summary with final metrics and history
        """
        logger.info(f"Starting meta-training for {self.config.num_iterations} iterations")
        logger.info(f"Algorithm: {self.config.algorithm}")
        logger.info(f"Meta batch size: {self.config.meta_batch_size}")

        start_time = time.time()

        for iteration in range(self.config.num_iterations):
            # Training step
            train_metrics = self.train_step(iteration)

            # Log training metrics
            if iteration % 10 == 0:
                logger.info(
                    f"Iter {iteration}/{self.config.num_iterations}: "
                    f"loss={train_metrics.get('query_loss', 0.0):.4f}, "
                    f"time={train_metrics['step_time']:.3f}s"
                )

            # Evaluation
            if iteration % self.config.eval_interval == 0:
                eval_metrics = self.evaluate()

                # Extract validation metric (use loss or accuracy)
                val_metric = eval_metrics.get(
                    "final_loss", eval_metrics.get("avg_loss", 0.0)
                )
                val_acc = eval_metrics.get(
                    "step_5_acc", eval_metrics.get("acc_after_adaptation", 0.0)
                )

                logger.info(
                    f"Evaluation at iter {iteration}: "
                    f"val_loss={val_metric:.4f}, "
                    f"val_acc={val_acc:.4f}"
                )

                # Update history
                self.training_history["train_loss"].append(
                    train_metrics.get("query_loss", 0.0)
                )
                self.training_history["val_loss"].append(val_metric)
                self.training_history["val_acc"].append(val_acc)
                self.training_history["iteration"].append(iteration)

                # Check early stopping
                is_best = val_metric < self.best_val_metric
                if self.check_early_stopping(val_metric):
                    logger.info(f"Stopping early at iteration {iteration}")
                    self.save_checkpoint(iteration, is_best=True)
                    break

                # Save checkpoint if best
                if is_best:
                    self.save_checkpoint(iteration, is_best=True)

            # Periodic checkpointing
            if iteration % self.config.save_interval == 0 and iteration > 0:
                self.save_checkpoint(iteration)

        # Final evaluation
        final_metrics = self.evaluate(num_eval_tasks=50)
        total_time = time.time() - start_time

        logger.info(f"Meta-training complete in {total_time:.2f}s")
        logger.info(f"Final validation metrics: {final_metrics}")

        # Save final checkpoint
        self.save_checkpoint(self.config.num_iterations)

        return {
            "final_metrics": final_metrics,
            "training_history": self.training_history,
            "total_time": total_time,
            "best_val_metric": self.best_val_metric,
        }

    def load_best_checkpoint(self) -> None:
        """Load best checkpoint from training."""
        best_path = Path(self.config.checkpoint_dir) / "best.npz"
        if best_path.exists():
            self.learner.load(str(best_path))
            logger.info(f"Loaded best checkpoint from {best_path}")
        else:
            logger.warning(f"Best checkpoint not found at {best_path}")


def create_orchestrator_from_config(
    model: nn.Module,
    train_dist: Any,
    val_dist: Any,
    loss_fn: Callable,
    config_dict: dict[str, Any],
) -> MetaLearningOrchestrator:
    """Create orchestrator from configuration dictionary.

    Args:
        model: Neural network model
        train_dist: Training task distribution
        val_dist: Validation task distribution
        loss_fn: Loss function
        config_dict: Configuration dictionary

    Returns:
        Initialized orchestrator
    """
    config = MetaTrainingConfig(**config_dict)
    return MetaLearningOrchestrator(
        model=model,
        train_task_distribution=train_dist,
        val_task_distribution=val_dist,
        loss_fn=loss_fn,
        config=config,
    )


def quick_meta_train(
    model: nn.Module,
    train_dist: Any,
    val_dist: Any,
    loss_fn: Callable,
    algorithm: str = "reptile",
    num_iterations: int = 1000,
) -> dict[str, Any]:
    """Quick meta-training with default settings.

    Args:
        model: Neural network model
        train_dist: Training task distribution
        val_dist: Validation task distribution
        loss_fn: Loss function
        algorithm: Meta-learning algorithm (default: "reptile")
        num_iterations: Number of iterations (default: 1000)

    Returns:
        Training summary
    """
    config = MetaTrainingConfig(
        algorithm=algorithm, num_iterations=num_iterations
    )

    orchestrator = MetaLearningOrchestrator(
        model=model,
        train_task_distribution=train_dist,
        val_task_distribution=val_dist,
        loss_fn=loss_fn,
        config=config,
    )

    results = orchestrator.train()

    return results
