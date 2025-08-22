"""
Knowledge distillation implementation for MLX models.

Provides teacher-student model training for knowledge transfer,
enabling smaller models to achieve comparable performance to larger ones.
Optimized for Apple Silicon with MLX backend.
"""

import logging
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx_lm import generate, load
from mlx_lm.utils import load as load_model_and_tokenizer

# Import utility classes with fallback for standalone execution
try:
    from ..utils import MemoryProfiler, ModelSizeCalculator, PerformanceProfiler
except ImportError:
    # Mock classes for standalone execution
    class ModelSizeCalculator:
        def calculate_model_size(self, model):
            return 100.0

    class MemoryProfiler:
        def profile_memory_usage(self, func):
            func()
            return 50.0

    class PerformanceProfiler:
        def profile_performance(self, func):
            func()
            return {"time": 1.0}


logger = logging.getLogger(__name__)


class KnowledgeDistiller:
    """
    Knowledge distillation implementation for MLX models.

    Implements teacher-student training where a smaller student model
    learns to mimic the behavior of a larger teacher model.
    """

    def __init__(
        self,
        teacher_model_path: str,
        student_model_path: str,
        temperature: float = 4.0,
        alpha: float = 0.7,
        beta: float = 0.3,
        device: str | None = None,
    ):
        """
        Initialize knowledge distiller.

        Args:
            teacher_model_path: Path to teacher model
            student_model_path: Path to student model
            temperature: Temperature for softmax distillation
            alpha: Weight for distillation loss
            beta: Weight for student loss
            device: MLX device to use
        """
        self.teacher_model_path = teacher_model_path
        self.student_model_path = student_model_path
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta

        # Load models
        self.teacher_model, self.teacher_tokenizer = self._load_teacher_model()
        self.student_model, self.student_tokenizer = self._load_student_model()

        # Set teacher to eval mode
        self.teacher_model.eval()

        # Initialize components
        self.size_calculator = ModelSizeCalculator()
        self.memory_profiler = MemoryProfiler()
        self.performance_profiler = PerformanceProfiler()

        logger.info(f"Knowledge distiller initialized")
        logger.info(f"Teacher: {teacher_model_path}")
        logger.info(f"Student: {student_model_path}")
        logger.info(f"Temperature: {temperature}, Alpha: {alpha}, Beta: {beta}")

    def _load_teacher_model(self) -> tuple[Any, Any]:
        """Load teacher model."""
        try:
            model, tokenizer = load_model_and_tokenizer(self.teacher_model_path)
            logger.info(f"Teacher model loaded from {self.teacher_model_path}")
            return model, tokenizer
        except Exception as e:
            logger.error(f"Failed to load teacher model: {e}")
            raise

    def _load_student_model(self) -> tuple[Any, Any]:
        """Load student model."""
        try:
            model, tokenizer = load_model_and_tokenizer(self.student_model_path)
            logger.info(f"Student model loaded from {self.student_model_path}")
            return model, tokenizer
        except Exception as e:
            logger.error(f"Failed to load student model: {e}")
            raise

    def distillation_loss(
        self,
        student_logits: mx.array,
        teacher_logits: mx.array,
        targets: mx.array,
        temperature: float = None,
    ) -> mx.array:
        """
        Compute knowledge distillation loss.

        Args:
            student_logits: Student model logits
            teacher_logits: Teacher model logits
            targets: Ground truth targets
            temperature: Distillation temperature

        Returns:
            Combined distillation loss
        """
        if temperature is None:
            temperature = self.temperature

        # Soft targets from teacher
        teacher_probs = mx.softmax(teacher_logits / temperature, axis=-1)
        student_log_probs = mx.log_softmax(student_logits / temperature, axis=-1)

        # KL divergence loss
        kl_loss = -mx.sum(teacher_probs * student_log_probs, axis=-1)
        kl_loss = mx.mean(kl_loss) * (temperature**2)

        # Student loss (cross-entropy with hard targets)
        student_loss = mx.mean(nn.losses.cross_entropy(student_logits, targets))

        # Combined loss
        total_loss = self.alpha * kl_loss + self.beta * student_loss

        return total_loss, kl_loss, student_loss

    def distill(
        self,
        train_data: list[dict[str, Any]],
        val_data: list[dict[str, Any]] | None = None,
        epochs: int = 3,
        learning_rate: float = 1e-4,
        batch_size: int = 4,
        output_dir: str | None = None,
        save_steps: int = 500,
        eval_steps: int = 100,
    ) -> dict[str, Any]:
        """
        Perform knowledge distillation training.

        Args:
            train_data: Training dataset
            val_data: Validation dataset
            epochs: Number of training epochs
            learning_rate: Learning rate
            batch_size: Training batch size
            output_dir: Directory to save checkpoints
            save_steps: Steps between saves
            eval_steps: Steps between evaluations

        Returns:
            Training results and metrics
        """
        logger.info("Starting knowledge distillation training")

        # Setup optimizer
        optimizer = optim.Adam(learning_rate=learning_rate)

        # Training metrics
        training_stats = {
            "epochs": epochs,
            "total_steps": 0,
            "losses": [],
            "kl_losses": [],
            "student_losses": [],
            "eval_results": [],
        }

        # Create output directory
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

        step = 0
        start_time = time.time()

        for epoch in range(epochs):
            logger.info(f"Epoch {epoch + 1}/{epochs}")
            epoch_losses = []
            epoch_kl_losses = []
            epoch_student_losses = []

            # Process batches
            for batch_idx in range(0, len(train_data), batch_size):
                batch = train_data[batch_idx : batch_idx + batch_size]

                # Prepare batch data
                batch_losses = []
                batch_kl_losses = []
                batch_student_losses = []

                for item in batch:
                    # Get inputs and targets
                    input_text = item.get("input", "")
                    target_text = item.get("target", "")

                    # Tokenize
                    teacher_inputs = self._tokenize_text(input_text, self.teacher_tokenizer)
                    student_inputs = self._tokenize_text(input_text, self.student_tokenizer)
                    targets = self._tokenize_text(target_text, self.student_tokenizer)

                    if teacher_inputs is None or student_inputs is None or targets is None:
                        continue

                    # Forward pass through teacher (no gradients)
                    with mx.no_grad():
                        teacher_logits = self._forward_pass(self.teacher_model, teacher_inputs)

                    # Forward pass through student
                    student_logits = self._forward_pass(self.student_model, student_inputs)

                    # Compute losses
                    total_loss, kl_loss, student_loss = self.distillation_loss(
                        student_logits, teacher_logits, targets
                    )

                    batch_losses.append(total_loss)
                    batch_kl_losses.append(kl_loss)
                    batch_student_losses.append(student_loss)

                if batch_losses:
                    # Average batch losses
                    avg_loss = mx.mean(mx.stack(batch_losses))
                    avg_kl_loss = mx.mean(mx.stack(batch_kl_losses))
                    avg_student_loss = mx.mean(mx.stack(batch_student_losses))

                    # Backward pass
                    loss_and_grad_fn = nn.value_and_grad(self.student_model, avg_loss)
                    loss, grads = loss_and_grad_fn(self.student_model.parameters())

                    # Update parameters
                    optimizer.update(self.student_model, grads)
                    mx.eval(self.student_model.parameters())

                    # Track metrics
                    epoch_losses.append(float(avg_loss))
                    epoch_kl_losses.append(float(avg_kl_loss))
                    epoch_student_losses.append(float(avg_student_loss))

                    step += 1

                    # Logging
                    if step % 10 == 0:
                        logger.info(
                            f"Step {step}: Loss={float(avg_loss):.4f}, "
                            f"KL={float(avg_kl_loss):.4f}, "
                            f"Student={float(avg_student_loss):.4f}"
                        )

                    # Evaluation
                    if val_data and step % eval_steps == 0:
                        eval_results = self._evaluate(val_data)
                        training_stats["eval_results"].append(
                            {"step": step, "epoch": epoch, **eval_results}
                        )
                        logger.info(f"Evaluation at step {step}: {eval_results}")

                    # Save checkpoint
                    if output_dir and step % save_steps == 0:
                        checkpoint_path = output_path / f"checkpoint_step_{step}"
                        self._save_model(checkpoint_path)
                        logger.info(f"Checkpoint saved at step {step}")

            # Epoch statistics
            if epoch_losses:
                avg_epoch_loss = np.mean(epoch_losses)
                avg_epoch_kl = np.mean(epoch_kl_losses)
                avg_epoch_student = np.mean(epoch_student_losses)

                training_stats["losses"].append(avg_epoch_loss)
                training_stats["kl_losses"].append(avg_epoch_kl)
                training_stats["student_losses"].append(avg_epoch_student)

                logger.info(
                    f"Epoch {epoch + 1} completed: "
                    f"Avg Loss={avg_epoch_loss:.4f}, "
                    f"Avg KL={avg_epoch_kl:.4f}, "
                    f"Avg Student={avg_epoch_student:.4f}"
                )

        # Final evaluation
        if val_data:
            final_eval = self._evaluate(val_data)
            training_stats["final_evaluation"] = final_eval
            logger.info(f"Final evaluation: {final_eval}")

        # Save final model
        if output_dir:
            final_path = output_path / "final_model"
            self._save_model(final_path)
            logger.info(f"Final model saved to {final_path}")

        # Calculate training time
        training_time = time.time() - start_time
        training_stats["training_time"] = training_time
        training_stats["total_steps"] = step

        logger.info(f"Knowledge distillation completed in {training_time:.2f}s")
        return training_stats

    def _tokenize_text(self, text: str, tokenizer: Any) -> mx.array | None:
        """Tokenize text with error handling."""
        try:
            tokens = tokenizer.encode(text)
            return mx.array(tokens)
        except Exception as e:
            logger.warning(f"Tokenization failed: {e}")
            return None

    def _forward_pass(self, model: Any, inputs: mx.array) -> mx.array:
        """Perform forward pass through model."""
        try:
            # Ensure inputs are properly shaped
            if inputs.ndim == 1:
                inputs = mx.expand_dims(inputs, 0)

            # Forward pass
            outputs = model(inputs)

            # Extract logits
            if hasattr(outputs, "logits"):
                return outputs.logits
            elif isinstance(outputs, tuple):
                return outputs[0]
            else:
                return outputs
        except Exception as e:
            logger.error(f"Forward pass failed: {e}")
            raise

    def _evaluate(self, val_data: list[dict[str, Any]]) -> dict[str, float]:
        """Evaluate student model on validation data."""
        self.student_model.eval()

        total_loss = 0.0
        total_kl_loss = 0.0
        total_student_loss = 0.0
        num_samples = 0

        with mx.no_grad():
            for item in val_data[:100]:  # Limit evaluation samples
                input_text = item.get("input", "")
                target_text = item.get("target", "")

                # Tokenize
                teacher_inputs = self._tokenize_text(input_text, self.teacher_tokenizer)
                student_inputs = self._tokenize_text(input_text, self.student_tokenizer)
                targets = self._tokenize_text(target_text, self.student_tokenizer)

                if teacher_inputs is None or student_inputs is None or targets is None:
                    continue

                # Forward passes
                teacher_logits = self._forward_pass(self.teacher_model, teacher_inputs)
                student_logits = self._forward_pass(self.student_model, student_inputs)

                # Compute losses
                total_loss_item, kl_loss_item, student_loss_item = self.distillation_loss(
                    student_logits, teacher_logits, targets
                )

                total_loss += float(total_loss_item)
                total_kl_loss += float(kl_loss_item)
                total_student_loss += float(student_loss_item)
                num_samples += 1

        self.student_model.train()

        if num_samples == 0:
            return {"eval_loss": 0.0, "eval_kl_loss": 0.0, "eval_student_loss": 0.0}

        return {
            "eval_loss": total_loss / num_samples,
            "eval_kl_loss": total_kl_loss / num_samples,
            "eval_student_loss": total_student_loss / num_samples,
            "num_eval_samples": num_samples,
        }

    def _save_model(self, output_path: Path) -> None:
        """Save student model to disk."""
        try:
            output_path.mkdir(parents=True, exist_ok=True)

            # Save model weights
            weights_path = output_path / "model.npz"
            weights = {k: v for k, v in self.student_model.parameters().items()}
            mx.savez(str(weights_path), **weights)

            # Save tokenizer if available
            if hasattr(self.student_tokenizer, "save_pretrained"):
                self.student_tokenizer.save_pretrained(str(output_path))

            logger.info(f"Model saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise

    def benchmark_distillation(
        self, test_data: list[dict[str, Any]], metrics: list[str] | None = None
    ) -> dict[str, Any]:
        """
        Benchmark distilled model performance.

        Args:
            test_data: Test dataset
            metrics: Metrics to compute

        Returns:
            Benchmark results
        """
        if metrics is None:
            metrics = ["accuracy", "perplexity", "speed", "memory"]

        logger.info("Benchmarking distilled model")

        results = {
            "teacher_model": self.teacher_model_path,
            "student_model": self.student_model_path,
            "distillation_config": {
                "temperature": self.temperature,
                "alpha": self.alpha,
                "beta": self.beta,
            },
        }

        # Model size comparison
        teacher_size = self.size_calculator.calculate_model_size(self.teacher_model)
        student_size = self.size_calculator.calculate_model_size(self.student_model)

        results["model_sizes"] = {
            "teacher_mb": teacher_size,
            "student_mb": student_size,
            "compression_ratio": teacher_size / student_size if student_size > 0 else 0,
        }

        # Performance benchmarks
        if "speed" in metrics:
            results["speed_comparison"] = self._benchmark_inference_speed(test_data[:10])

        if "memory" in metrics:
            results["memory_comparison"] = self._benchmark_memory_usage(test_data[:5])

        if "accuracy" in metrics or "perplexity" in metrics:
            eval_results = self._evaluate(test_data[:50])
            results["evaluation"] = eval_results

        logger.info("Distillation benchmarking completed")
        return results

    def _benchmark_inference_speed(self, test_data: list[dict[str, Any]]) -> dict[str, float]:
        """Benchmark inference speed comparison."""
        teacher_times = []
        student_times = []

        for item in test_data:
            input_text = item.get("input", "")

            # Teacher inference
            start_time = time.time()
            teacher_inputs = self._tokenize_text(input_text, self.teacher_tokenizer)
            if teacher_inputs is not None:
                with mx.no_grad():
                    _ = self._forward_pass(self.teacher_model, teacher_inputs)
                mx.eval(_)
            teacher_time = time.time() - start_time
            teacher_times.append(teacher_time)

            # Student inference
            start_time = time.time()
            student_inputs = self._tokenize_text(input_text, self.student_tokenizer)
            if student_inputs is not None:
                with mx.no_grad():
                    _ = self._forward_pass(self.student_model, student_inputs)
                mx.eval(_)
            student_time = time.time() - start_time
            student_times.append(student_time)

        avg_teacher_time = np.mean(teacher_times)
        avg_student_time = np.mean(student_times)

        return {
            "teacher_avg_time": avg_teacher_time,
            "student_avg_time": avg_student_time,
            "speedup": avg_teacher_time / avg_student_time if avg_student_time > 0 else 0,
        }

    def _benchmark_memory_usage(self, test_data: list[dict[str, Any]]) -> dict[str, float]:
        """Benchmark memory usage comparison."""
        # Teacher memory
        teacher_memory = self.memory_profiler.profile_memory_usage(
            lambda: self._run_inference_batch(test_data, self.teacher_model, self.teacher_tokenizer)
        )

        # Student memory
        student_memory = self.memory_profiler.profile_memory_usage(
            lambda: self._run_inference_batch(test_data, self.student_model, self.student_tokenizer)
        )

        return {
            "teacher_memory_mb": teacher_memory,
            "student_memory_mb": student_memory,
            "memory_reduction": (teacher_memory - student_memory) / teacher_memory
            if teacher_memory > 0
            else 0,
        }

    def _run_inference_batch(
        self, test_data: list[dict[str, Any]], model: Any, tokenizer: Any
    ) -> None:
        """Run inference on a batch of data."""
        with mx.no_grad():
            for item in test_data:
                input_text = item.get("input", "")
                inputs = self._tokenize_text(input_text, tokenizer)
                if inputs is not None:
                    outputs = self._forward_pass(model, inputs)
                    mx.eval(outputs)


class TeacherStudentPair:
    """
    Manages teacher-student model pairs for distillation.
    """

    def __init__(self, teacher_path: str, student_path: str):
        self.teacher_path = teacher_path
        self.student_path = student_path
        self.distiller = None

    def create_distiller(
        self, temperature: float = 4.0, alpha: float = 0.7, beta: float = 0.3
    ) -> KnowledgeDistiller:
        """Create knowledge distiller for this pair."""
        self.distiller = KnowledgeDistiller(
            teacher_model_path=self.teacher_path,
            student_model_path=self.student_path,
            temperature=temperature,
            alpha=alpha,
            beta=beta,
        )
        return self.distiller

    def distill(self, **kwargs) -> dict[str, Any]:
        """Run distillation training."""
        if self.distiller is None:
            self.create_distiller()
        return self.distiller.distill(**kwargs)

    def benchmark(self, test_data: list[dict[str, Any]]) -> dict[str, Any]:
        """Benchmark the distilled model."""
        if self.distiller is None:
            self.create_distiller()
        return self.distiller.benchmark_distillation(test_data)
