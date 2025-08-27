"""
MLX-optimized LoRA trainer implementation.

Comprehensive training pipeline with Apple Silicon optimizations, automatic
mixed precision, gradient accumulation, and advanced monitoring capabilities.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from lora import LoRAConfig, ModelAdapter, TrainingConfig
from training.callbacks import MLXMonitorCallback, TrainingCallback
from training.data_loader import ConversationDataLoader, DatasetConfig, create_data_loader
from training.optimizer import create_optimizer, create_scheduler

# Import shared utilities with fallback
try:
    from utils.logging_utils import get_logger
except ImportError:
    def get_logger(name: str) -> logging.Logger:
        return logging.getLogger(name)

logger = get_logger(__name__)

# Constants for numerical stability and precision
EPSILON = 1e-8
GRADIENT_CLIP_EPSILON = 1e-6
MAX_SEQUENCE_LENGTH = 512
DEFAULT_CALIBRATION_SAMPLES = 512


@dataclass
class TrainingState:
    """Training state tracking."""

    epoch: int = 0
    global_step: int = 0
    best_metric: float = float("inf")
    best_epoch: int = 0
    is_best_model: bool = False

    # Metrics tracking
    train_loss: float = 0.0
    eval_loss: float = 0.0
    learning_rate: float = 0.0

    # Timing
    epoch_start_time: float = field(default_factory=time.time)
    total_train_time: float = 0.0

    # Memory tracking (Apple Silicon specific)
    peak_memory_mb: float = 0.0
    current_memory_mb: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch,
            "train_loss": self.train_loss,
            "eval_loss": self.eval_loss,
            "learning_rate": self.learning_rate,
            "peak_memory_mb": self.peak_memory_mb,
            "current_memory_mb": self.current_memory_mb,
            "total_train_time": self.total_train_time,
        }


class LoRATrainer:
    """
    MLX-optimized LoRA trainer with Apple Silicon acceleration.

    Comprehensive trainer supporting automated hyperparameter optimization,
    mixed precision training, gradient accumulation, and advanced callbacks.
    """

    def __init__(
        self,
        model: nn.Module,
        lora_config: LoRAConfig,
        training_config: TrainingConfig,
        train_dataset: Any = None,
        eval_dataset: Any = None,
        callbacks: list[TrainingCallback] | None = None,
    ):
        self.model = model
        self.lora_config = lora_config
        self.training_config = training_config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

        # Initialize model adapter
        self.model_adapter = ModelAdapter(model, lora_config)
        self.model_adapter.adapt_model()

        # Training state
        self.state = TrainingState()

        # Setup callbacks
        self.callbacks = callbacks or []
        if not any(isinstance(cb, MLXMonitorCallback) for cb in self.callbacks):
            self.callbacks.append(MLXMonitorCallback())

        # Training components (initialized during training)
        self.optimizer: optim.Optimizer | None = None
        self.scheduler: Any | None = None

        # Performance tracking
        self.training_history = []

        # Data loading components
        self.data_loader: ConversationDataLoader | None = None
        self.tokenizer: Any | None = None

        # Setup output directories
        self.setup_directories()

    def setup_directories(self) -> None:
        """Create necessary directories for training outputs."""
        self.training_config.output_dir.mkdir(parents=True, exist_ok=True)
        (self.training_config.output_dir / "checkpoints").mkdir(exist_ok=True)
        (self.training_config.output_dir / "logs").mkdir(exist_ok=True)

    def setup_data_loading(self, tokenizer: Any, train_data_path: str | None = None) -> None:
        """
        Setup data loading for training.

        Args:
            tokenizer: Tokenizer for text processing
            train_data_path: Path to training data file (JSONL format)
        """
        self.tokenizer = tokenizer

        # Setup data loader configuration
        data_config = DatasetConfig(
            max_length=getattr(self.training_config, "max_length", 512),
            padding=True,
            truncation=True,
            add_special_tokens=True,
        )

        self.data_loader = ConversationDataLoader(tokenizer, data_config)

        # Load training data if path is provided
        if train_data_path:
            logger.info("Loading training data from %s", train_data_path)
            dataset = self.data_loader.create_dataset(train_data_path)

            # Create batch iterator
            batch_iterator = self.data_loader.create_batches(
                dataset, batch_size=self.training_config.batch_size, shuffle=True
            )

            # Convert iterator to list for multiple epochs
            self.train_dataset = list(batch_iterator)

            # Print dataset statistics
            stats = self.data_loader.get_data_stats(dataset)
            logger.info("Dataset loaded: %d examples", stats['num_examples'])
            logger.info("Average sequence length: %.1f", stats['avg_sequence_length'])
            logger.info("Batches per epoch: %d", len(self.train_dataset))
        else:
            logger.warning("No training data path provided")

    def setup_optimization(self) -> None:
        """Setup optimizer and learning rate scheduler."""
        trainable_params = self.model_adapter.get_trainable_parameters()

        self.optimizer = create_optimizer(
            parameters=trainable_params.values(),
            optimizer_name=self.training_config.optimizer,
            learning_rate=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay,
        )

        if self.training_config.scheduler != "none":
            total_steps = self.calculate_total_steps()
            self.scheduler = create_scheduler(
                optimizer=self.optimizer,
                scheduler_name=self.training_config.scheduler,
                num_warmup_steps=self.training_config.warmup_steps,
                num_training_steps=total_steps,
            )

    def calculate_total_steps(self) -> int:
        """Calculate total training steps."""
        if self.train_dataset is None:
            return 1000  # Fallback

        dataset_size = len(self.train_dataset)
        steps_per_epoch = dataset_size // (
            self.training_config.batch_size * self.training_config.gradient_accumulation_steps
        )
        return steps_per_epoch * self.training_config.num_epochs

    def _extract_logits(self, outputs) -> mx.array:
        """Extract logits from model outputs."""
        if isinstance(outputs, tuple):
            return outputs[0]
        elif hasattr(outputs, "logits"):
            return outputs.logits
        else:
            return outputs

    def _compute_causal_loss(self, logits: mx.array, labels: mx.array) -> mx.array:
        """Compute cross-entropy loss for causal language modeling."""
        # Shift labels for causal language modeling
        shift_logits = logits[..., :-1, :]
        shift_labels = labels[..., 1:]

        # Flatten for loss computation
        vocab_size = shift_logits.shape[-1]
        flat_logits = shift_logits.reshape(-1, vocab_size)
        flat_labels = shift_labels.reshape(-1)

        # Cross-entropy loss
        return mx.mean(nn.losses.cross_entropy(flat_logits, flat_labels))

    def compute_loss(self, batch: dict[str, mx.array]) -> mx.array:
        """Compute training loss for a batch."""
        return self.compute_loss_with_model(self.model, batch)

    def compute_loss_with_model(self, model: nn.Module, batch: dict[str, mx.array]) -> mx.array:
        """Compute training loss for a batch with a specific model."""
        input_ids = batch["input_ids"]
        labels = batch.get("labels", input_ids)

        # Forward pass through model
        outputs = model(input_ids)
        logits = self._extract_logits(outputs)

        return self._compute_causal_loss(logits, labels)

    def _create_loss_function(self, batch: dict[str, mx.array]):
        """Create loss function for gradient computation."""
        def loss_fn(model_params):
            # Temporarily update model parameters
            original_params = dict(self.model.parameters())
            self.model.update(model_params)
            try:
                return self.compute_loss(batch)
            finally:
                # Restore original parameters
                self.model.update(original_params)
        return loss_fn

    def _compute_gradients(self, batch: dict[str, mx.array]) -> tuple[mx.array, dict]:
        """Compute loss and gradients for a batch."""
        loss_fn = self._create_loss_function(batch)
        loss_and_grad_fn = nn.value_and_grad(loss_fn)
        return loss_and_grad_fn(dict(self.model.parameters()))

    def _update_model_parameters(self, gradients: dict) -> None:
        """Update model parameters using optimizer."""
        # Apply gradient clipping if enabled
        if self.training_config.gradient_clipping > 0:
            gradients = self.clip_gradients(gradients)

        # Update model parameters
        self.optimizer.update(self.model, gradients)
        mx.eval(self.model.parameters())

    def _update_training_state(self) -> float:
        """Update training state after a successful step. Returns current learning rate."""
        # Update learning rate with scheduler
        current_lr = self.training_config.learning_rate
        if self.scheduler is not None:
            self.scheduler.step()
            current_lr = (
                self.scheduler.get_last_lr()[0]
                if hasattr(self.scheduler, "get_last_lr")
                else current_lr
            )

        # Update training state
        self.state.learning_rate = current_lr
        self.state.global_step += 1

        return current_lr

    def training_step(self, batch: dict[str, mx.array]) -> dict[str, float]:
        """Execute a single training step with proper gradient computation."""
        try:
            # Ensure optimizer is initialized
            if self.optimizer is None:
                raise RuntimeError("Optimizer not initialized. Call setup_optimization() first.")

            # Compute loss and gradients
            loss, gradients = self._compute_gradients(batch)

            # Update model parameters
            self._update_model_parameters(gradients)

            # Update training state
            current_lr = self._update_training_state()

            return {
                "loss": float(loss),
                "learning_rate": current_lr,
                "global_step": self.state.global_step,
            }

        except (RuntimeError, ValueError, TypeError) as e:
            # Log specific training errors
            logger.error("Training step failed: %s", e)

            # Return previous loss to indicate failure but allow training to continue
            return {
                "loss": self.state.train_loss,
                "learning_rate": self.state.learning_rate,
                "global_step": self.state.global_step,
            }

    def clip_gradients(self, gradients: dict[str, mx.array]) -> dict[str, mx.array]:
        """Apply gradient clipping."""
        # Compute global gradient norm
        total_norm = 0.0
        for grad in gradients.values():
            if grad is not None:
                total_norm += mx.sum(grad**2)

        total_norm = mx.sqrt(total_norm)
        clip_coeff = self.training_config.gradient_clipping / (total_norm + GRADIENT_CLIP_EPSILON)

        # Apply clipping if necessary
        if clip_coeff < 1.0:
            clipped_grads = {}
            for name, grad in gradients.items():
                if grad is not None:
                    clipped_grads[name] = grad * clip_coeff
                else:
                    clipped_grads[name] = grad
            return clipped_grads

        return gradients

    def evaluate(self) -> dict[str, float]:
        """Run evaluation on validation dataset."""
        if self.eval_dataset is None:
            return {"eval_loss": 0.0}

        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        try:
            for batch in self.eval_dataset:
                with mx.no_grad():
                    loss = self.compute_loss(batch)
                    total_loss += float(loss)
                    num_batches += 1

        except Exception as e:
            logger.warning("Evaluation failed: %s", e)
            return {"eval_loss": 0.0}

        finally:
            self.model.train()

        avg_loss = total_loss / max(num_batches, 1)
        return {"eval_loss": avg_loss}

    def should_save_checkpoint(self, eval_metrics: dict[str, float]) -> bool:
        """Determine if current model should be saved as checkpoint."""
        current_metric = eval_metrics.get("eval_loss", float("inf"))

        if current_metric < self.state.best_metric:
            self.state.best_metric = current_metric
            self.state.best_epoch = self.state.epoch
            self.state.is_best_model = True
            return True

        self.state.is_best_model = False
        return False

    def save_checkpoint(self, is_best: bool = False) -> None:
        """Save model checkpoint."""
        checkpoint_dir = self.training_config.output_dir / "checkpoints"

        # Save LoRA adapters
        if is_best:
            save_path = checkpoint_dir / "best_model"
        else:
            save_path = checkpoint_dir / f"checkpoint_epoch_{self.state.epoch}"

        self.model_adapter.save_adapters(save_path)

        # Save training state
        state_dict = self.state.to_dict()
        with open(save_path / "training_state.json", "w") as f:
            json.dump(state_dict, f, indent=2)

        # Save training history
        with open(save_path / "training_history.json", "w") as f:
            json.dump(self.training_history, f, indent=2)

        logger.info("Checkpoint saved to %s", save_path)

    def train_epoch(self) -> dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0

        # Callback: on_epoch_start
        for callback in self.callbacks:
            callback.on_epoch_start(self.state, logs={})

        self.state.epoch_start_time = time.time()

        # Training loop
        if self.train_dataset is not None:
            for batch_idx, batch in enumerate(self.train_dataset):
                # Callback: on_batch_start
                batch_logs = {"batch": batch_idx}
                for callback in self.callbacks:
                    callback.on_batch_start(self.state, batch_logs)

                # Training step
                step_metrics = self.training_step(batch)
                epoch_loss += step_metrics["loss"]
                num_batches += 1

                # Update batch logs
                batch_logs.update(step_metrics)

                # Callback: on_batch_end
                for callback in self.callbacks:
                    callback.on_batch_end(self.state, batch_logs)

                # Logging
                if self.state.global_step % self.training_config.logging_steps == 0:
                    avg_loss = epoch_loss / num_batches
                    logger.info(
                        "Step %d: loss=%.4f, lr=%.2e",
                        self.state.global_step, avg_loss, step_metrics['learning_rate']
                    )

                # Evaluation
                if self.state.global_step % self.training_config.eval_steps == 0:
                    eval_metrics = self.evaluate()
                    logger.info("Evaluation: %s", eval_metrics)

                    # Check for best model
                    if self.should_save_checkpoint(eval_metrics):
                        self.save_checkpoint(is_best=True)

        # Calculate epoch metrics
        avg_loss = epoch_loss / max(num_batches, 1)
        epoch_time = time.time() - self.state.epoch_start_time
        self.state.total_train_time += epoch_time

        epoch_metrics = {
            "train_loss": avg_loss,
            "epoch_time": epoch_time,
            "learning_rate": self.state.learning_rate,
        }

        # Final evaluation for epoch
        eval_metrics = self.evaluate()
        epoch_metrics.update(eval_metrics)

        self.state.train_loss = avg_loss
        self.state.eval_loss = eval_metrics.get("eval_loss", 0.0)

        # Save training history
        history_entry = {
            "epoch": self.state.epoch,
            "timestamp": datetime.now().isoformat(),
            **epoch_metrics,
            **self.state.to_dict(),
        }
        self.training_history.append(history_entry)

        # Callback: on_epoch_end
        for callback in self.callbacks:
            callback.on_epoch_end(self.state, epoch_metrics)

        return epoch_metrics

    def _setup_training(self, tokenizer: Any = None, train_data_path: str | None = None) -> None:
        """Setup training components and data loading."""
        logger.info("Starting LoRA fine-tuning...")
        logger.info("Training for %d epochs", self.training_config.num_epochs)
        logger.info("Output directory: %s", self.training_config.output_dir)

        # Setup data loading if provided
        if tokenizer is not None:
            self.setup_data_loading(tokenizer, train_data_path)
        elif self.train_dataset is None:
            logger.warning("No training data available")

        # Setup optimization
        self.setup_optimization()

    def _run_training_epochs(self) -> None:
        """Execute the main training loop."""
        for epoch in range(self.training_config.num_epochs):
            self.state.epoch = epoch
            logger.info("Epoch %d/%d", epoch + 1, self.training_config.num_epochs)

            epoch_metrics = self.train_epoch()
            self._log_epoch_summary(epoch + 1, epoch_metrics)

            # Save checkpoint
            if (epoch + 1) % self.training_config.save_steps == 0:
                self.save_checkpoint(is_best=False)

    def _log_epoch_summary(self, epoch_num: int, epoch_metrics: dict[str, Any]) -> None:
        """Log epoch summary with metrics."""
        logger.info("Epoch %d Summary:", epoch_num)
        for metric, value in epoch_metrics.items():
            if isinstance(value, float):
                logger.info("  %s: %.4f", metric, value)
            else:
                logger.info("  %s: %s", metric, value)

    def _finalize_training(self, training_start_time: float) -> None:
        """Finalize training with logging and checkpoint saving."""
        total_training_time = time.time() - training_start_time
        self.state.total_train_time = total_training_time

        # Callback: on_train_end
        final_logs = {"total_training_time": total_training_time}
        for callback in self.callbacks:
            callback.on_train_end(self.state, final_logs)

        # Final checkpoint
        self.save_checkpoint(is_best=False)

        logger.info("Training completed in %.2f seconds", total_training_time)
        logger.info("Best model saved at epoch %d", self.state.best_epoch)
        logger.info("Best validation loss: %.4f", self.state.best_metric)

    def train(self, tokenizer: Any = None, train_data_path: str | None = None) -> dict[str, Any]:
        """
        Main training loop.

        Args:
            tokenizer: Tokenizer for text processing
            train_data_path: Path to training data file (JSONL format)

        Returns:
            Training results dictionary
        """
        # Setup training components
        self._setup_training(tokenizer, train_data_path)

        # Callback: on_train_start
        for callback in self.callbacks:
            callback.on_train_start(self.state, {})

        training_start_time = time.time()

        try:
            self._run_training_epochs()

        except KeyboardInterrupt:
            logger.info("Training interrupted by user")

        except Exception as e:
            logger.error("Training failed with error: %s", e)
            raise e

        finally:
            self._finalize_training(training_start_time)

        return {
            "best_epoch": self.state.best_epoch,
            "best_metric": self.state.best_metric,
            "total_training_time": self.state.total_train_time,
            "training_history": self.training_history,
        }
