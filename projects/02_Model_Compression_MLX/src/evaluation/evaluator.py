"""
Comprehensive evaluation tools for compressed models.
"""

import logging
import math
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np

# Import shared utilities with fallback
try:
    from utils.logging_utils import get_logger
except ImportError:

    def get_logger(name: str) -> logging.Logger:
        return logging.getLogger(name)


logger = get_logger(__name__)


class CompressionEvaluator:
    """
    Comprehensive evaluator for compressed models.

    Provides evaluation of:
    - Model accuracy/perplexity
    - Inference performance
    - Memory usage
    - Model size reduction
    - Hardware efficiency
    """

    def __init__(self):
        """Initialize the evaluator."""
        self.device = mx.default_device()
        self.evaluation_results = {}

    def evaluate_compressed_model(
        self,
        original_model: Any,
        compressed_model: Any,
        tokenizer: Any,
        eval_dataset: Any | None = None,
        metrics: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Comprehensive evaluation of compressed vs original model.

        Args:
            original_model: Original uncompressed model
            compressed_model: Compressed model
            tokenizer: Tokenizer for text processing
            eval_dataset: Evaluation dataset
            metrics: List of metrics to evaluate

        Returns:
            Dictionary containing all evaluation results
        """
        logger.info("Starting comprehensive model evaluation")

        if metrics is None:
            metrics = ["perplexity", "inference_speed", "memory_usage", "model_size", "accuracy"]

        results = {
            "original": {},
            "compressed": {},
            "comparison": {},
            "metrics": metrics,
        }

        try:
            # Evaluate original model
            logger.info("Evaluating original model")
            results["original"] = self._evaluate_single_model(
                original_model, tokenizer, eval_dataset, "original"
            )

            # Evaluate compressed model
            logger.info("Evaluating compressed model")
            results["compressed"] = self._evaluate_single_model(
                compressed_model, tokenizer, eval_dataset, "compressed"
            )

            # Calculate comparison metrics
            logger.info("Calculating comparison metrics")
            results["comparison"] = self._calculate_comparison_metrics(
                results["original"], results["compressed"]
            )

            self.evaluation_results = results
            logger.info("Comprehensive evaluation completed")

            return results

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise

    def _evaluate_single_model(
        self, model: Any, tokenizer: Any, eval_dataset: Any | None, model_type: str
    ) -> dict[str, Any]:
        """Evaluate a single model."""
        results = {}

        # Model size metrics
        results.update(self._calculate_model_size_metrics(model))

        # Performance metrics
        results.update(self._benchmark_inference_performance(model, tokenizer))

        # Memory usage metrics
        results.update(self._measure_memory_usage(model))

        # Accuracy metrics (if dataset provided)
        if eval_dataset:
            results.update(self._calculate_accuracy_metrics(model, tokenizer, eval_dataset))

        return results

    def _calculate_model_size_metrics(self, model: Any) -> dict[str, Any]:
        """Calculate model size and parameter metrics."""
        logger.debug("Calculating model size metrics")

        metrics = {}

        try:
            total_params = 0
            trainable_params = 0
            total_bytes = 0

            if hasattr(model, "parameters"):
                for param in model.parameters():
                    if isinstance(param, mx.array):
                        param_size = param.size
                        total_params += param_size
                        trainable_params += param_size  # Assume all are trainable for MLX
                        total_bytes += param.nbytes
                    elif hasattr(param, "shape"):
                        param_size = np.prod(param.shape)
                        total_params += param_size
                        trainable_params += param_size
                        # Estimate bytes (assume float32)
                        total_bytes += param_size * 4

            metrics.update(
                {
                    "total_parameters": total_params,
                    "trainable_parameters": trainable_params,
                    "model_size_mb": total_bytes / (1024 * 1024),
                    "model_size_gb": total_bytes / (1024 * 1024 * 1024),
                }
            )

        except Exception as e:
            logger.warning(f"Could not calculate model size metrics: {e}")

        return metrics

    def _benchmark_inference_performance(
        self, model: Any, tokenizer: Any, num_runs: int = 100
    ) -> dict[str, Any]:
        """Benchmark inference performance."""
        logger.debug("Benchmarking inference performance")

        metrics = {}

        try:
            # Test inputs
            test_prompts = [
                "The future of artificial intelligence is",
                "Machine learning enables",
                "Deep neural networks can",
            ]

            # Tokenize test inputs
            test_tokens = []
            for prompt in test_prompts:
                if tokenizer:
                    tokens = tokenizer.encode(prompt)
                    if len(tokens) > 128:  # Limit length
                        tokens = tokens[:128]
                    test_tokens.append(tokens)
                else:
                    # Dummy tokens if no tokenizer
                    test_tokens.append(list(range(50)))

            # Warmup runs
            for _ in range(10):
                try:
                    for tokens in test_tokens:
                        input_ids = mx.array([tokens])
                        with mx.stream(mx.default_stream()):
                            _ = model(input_ids)
                            mx.eval(_)
                except Exception:
                    break

            # Actual benchmark runs
            start_time = time.time()
            successful_runs = 0

            for run in range(num_runs):
                try:
                    for tokens in test_tokens:
                        input_ids = mx.array([tokens])
                        with mx.stream(mx.default_stream()):
                            _ = model(input_ids)
                            mx.eval(_)
                    successful_runs += 1
                except Exception as e:
                    logger.warning(f"Benchmark run {run} failed: {e}")
                    continue

            total_time = time.time() - start_time

            if successful_runs > 0:
                avg_time_per_run = total_time / successful_runs
                throughput = successful_runs / total_time

                metrics.update(
                    {
                        "avg_inference_time_ms": avg_time_per_run * 1000,
                        "throughput_samples_per_sec": throughput,
                        "successful_runs": successful_runs,
                        "total_runs": num_runs,
                        "success_rate": successful_runs / num_runs,
                    }
                )
            else:
                metrics.update(
                    {
                        "avg_inference_time_ms": float("inf"),
                        "throughput_samples_per_sec": 0.0,
                        "successful_runs": 0,
                        "total_runs": num_runs,
                        "success_rate": 0.0,
                    }
                )

        except Exception as e:
            logger.warning(f"Could not benchmark inference performance: {e}")

        return metrics

    def _measure_memory_usage(self, model: Any) -> dict[str, Any]:
        """Measure memory usage of the model."""
        logger.debug("Measuring memory usage")

        metrics = {}

        try:
            # Calculate theoretical memory usage
            total_bytes = 0
            if hasattr(model, "parameters"):
                for param in model.parameters():
                    if isinstance(param, mx.array):
                        total_bytes += param.nbytes
                    elif hasattr(param, "shape"):
                        # Estimate (assume float32)
                        total_bytes += np.prod(param.shape) * 4

            metrics.update(
                {
                    "memory_usage_mb": total_bytes / (1024 * 1024),
                    "memory_usage_gb": total_bytes / (1024 * 1024 * 1024),
                }
            )

            # Try to get actual GPU memory usage if available
            try:
                # This would be platform specific
                # For MLX on Apple Silicon, memory usage is unified
                pass
            except Exception:
                pass

        except Exception as e:
            logger.warning(f"Could not measure memory usage: {e}")

        return metrics

    def _calculate_accuracy_metrics(
        self, model: Any, tokenizer: Any, eval_dataset: Any
    ) -> dict[str, Any]:
        """Calculate accuracy metrics on evaluation dataset."""
        logger.debug("Calculating accuracy metrics")

        metrics = {}

        try:
            if isinstance(eval_dataset, (list, str)):
                # Text evaluation - calculate perplexity
                perplexity = self._calculate_perplexity(model, tokenizer, eval_dataset)
                metrics["perplexity"] = perplexity

                # Calculate cross-entropy loss
                ce_loss = self._calculate_cross_entropy_loss(model, tokenizer, eval_dataset)
                metrics["cross_entropy_loss"] = ce_loss

            # Additional metrics could be added here for specific tasks
            # (classification accuracy, BLEU scores, etc.)

        except Exception as e:
            logger.warning(f"Could not calculate accuracy metrics: {e}")

        return metrics

    def _calculate_perplexity(
        self, model: Any, tokenizer: Any, eval_data: str | list[str], max_samples: int = 100
    ) -> float:
        """Calculate perplexity on evaluation data."""
        if isinstance(eval_data, str):
            eval_texts = [eval_data]
        else:
            eval_texts = eval_data[:max_samples] if len(eval_data) > max_samples else eval_data

        total_log_likelihood = 0.0
        total_tokens = 0

        try:
            for text in eval_texts:
                if not tokenizer:
                    continue

                try:
                    # Tokenize text
                    tokens = tokenizer.encode(text)
                    if len(tokens) < 2:  # Need at least 2 tokens for loss calculation
                        continue

                    # Limit length
                    if len(tokens) > 512:
                        tokens = tokens[:512]

                    # Convert to MLX array
                    input_ids = mx.array([tokens[:-1]])  # Input
                    target_ids = mx.array([tokens[1:]])  # Targets (shifted by 1)

                    # Forward pass
                    with mx.stream(mx.default_stream()):
                        logits = model(input_ids)
                        mx.eval(logits)

                    # Calculate log likelihood
                    # This is a simplified calculation
                    log_probs = mx.log_softmax(logits, axis=-1)
                    target_log_probs = mx.take_along_axis(
                        log_probs, mx.expand_dims(target_ids, axis=-1), axis=-1
                    )

                    total_log_likelihood += mx.sum(target_log_probs)
                    total_tokens += len(tokens) - 1

                except Exception as e:
                    logger.warning(f"Failed to process text for perplexity: {e}")
                    continue

            if total_tokens > 0:
                avg_log_likelihood = total_log_likelihood / total_tokens
                perplexity = mx.exp(-avg_log_likelihood)
                return float(perplexity)
            else:
                return float("inf")

        except Exception as e:
            logger.warning(f"Could not calculate perplexity: {e}")
            return float("inf")

    def _calculate_cross_entropy_loss(
        self, model: Any, tokenizer: Any, eval_data: str | list[str], max_samples: int = 100
    ) -> float:
        """Calculate cross-entropy loss on evaluation data."""
        if isinstance(eval_data, str):
            eval_texts = [eval_data]
        else:
            eval_texts = eval_data[:max_samples] if len(eval_data) > max_samples else eval_data

        total_loss = 0.0
        total_samples = 0

        try:
            for text in eval_texts:
                if not tokenizer:
                    continue

                try:
                    # Tokenize text
                    tokens = tokenizer.encode(text)
                    if len(tokens) < 2:
                        continue

                    if len(tokens) > 512:
                        tokens = tokens[:512]

                    # Convert to MLX array
                    input_ids = mx.array([tokens[:-1]])
                    target_ids = mx.array([tokens[1:]])

                    # Forward pass
                    with mx.stream(mx.default_stream()):
                        logits = model(input_ids)
                        mx.eval(logits)

                    # Calculate cross-entropy loss
                    loss = mx.mean(mx.cross_entropy(logits, target_ids))
                    total_loss += float(loss)
                    total_samples += 1

                except Exception as e:
                    logger.warning(f"Failed to calculate CE loss for text: {e}")
                    continue

            if total_samples > 0:
                return total_loss / total_samples
            else:
                return float("inf")

        except Exception as e:
            logger.warning(f"Could not calculate cross-entropy loss: {e}")
            return float("inf")

    def _calculate_comparison_metrics(
        self, original_results: dict[str, Any], compressed_results: dict[str, Any]
    ) -> dict[str, Any]:
        """Calculate comparison metrics between original and compressed models."""
        comparison = {}

        try:
            # Size reduction metrics
            if "model_size_mb" in original_results and "model_size_mb" in compressed_results:
                orig_size = original_results["model_size_mb"]
                comp_size = compressed_results["model_size_mb"]

                if orig_size > 0:
                    compression_ratio = orig_size / comp_size
                    size_reduction_percent = ((orig_size - comp_size) / orig_size) * 100

                    comparison.update(
                        {
                            "compression_ratio": compression_ratio,
                            "size_reduction_percent": size_reduction_percent,
                            "size_reduction_mb": orig_size - comp_size,
                        }
                    )

            # Performance comparison
            if (
                "avg_inference_time_ms" in original_results
                and "avg_inference_time_ms" in compressed_results
            ):
                orig_time = original_results["avg_inference_time_ms"]
                comp_time = compressed_results["avg_inference_time_ms"]

                if orig_time > 0 and comp_time > 0:
                    speedup = orig_time / comp_time
                    comparison["inference_speedup"] = speedup

            # Memory usage comparison
            if "memory_usage_mb" in original_results and "memory_usage_mb" in compressed_results:
                orig_mem = original_results["memory_usage_mb"]
                comp_mem = compressed_results["memory_usage_mb"]

                if orig_mem > 0:
                    memory_reduction_percent = ((orig_mem - comp_mem) / orig_mem) * 100
                    comparison["memory_reduction_percent"] = memory_reduction_percent

            # Accuracy degradation
            if "perplexity" in original_results and "perplexity" in compressed_results:
                orig_ppl = original_results["perplexity"]
                comp_ppl = compressed_results["perplexity"]

                if orig_ppl > 0 and comp_ppl > 0:
                    perplexity_increase = ((comp_ppl - orig_ppl) / orig_ppl) * 100
                    comparison["perplexity_increase_percent"] = perplexity_increase

            if (
                "cross_entropy_loss" in original_results
                and "cross_entropy_loss" in compressed_results
            ):
                orig_loss = original_results["cross_entropy_loss"]
                comp_loss = compressed_results["cross_entropy_loss"]

                if orig_loss > 0 and comp_loss > 0:
                    loss_increase = ((comp_loss - orig_loss) / orig_loss) * 100
                    comparison["ce_loss_increase_percent"] = loss_increase

        except Exception as e:
            logger.warning(f"Could not calculate comparison metrics: {e}")

        return comparison

    def generate_evaluation_report(self, output_path: Path | None = None) -> str:
        """Generate a comprehensive evaluation report."""
        if not self.evaluation_results:
            return "No evaluation results available. Run evaluate_compressed_model() first."

        report_lines = []
        results = self.evaluation_results

        # Header
        report_lines.append("=" * 80)
        report_lines.append("MODEL COMPRESSION EVALUATION REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")

        # Original model metrics
        report_lines.append("ORIGINAL MODEL:")
        report_lines.append("-" * 40)
        orig = results["original"]
        if "total_parameters" in orig:
            report_lines.append(f"Parameters: {orig['total_parameters']:,}")
        if "model_size_mb" in orig:
            report_lines.append(f"Size: {orig['model_size_mb']:.2f} MB")
        if "avg_inference_time_ms" in orig:
            report_lines.append(f"Avg Inference Time: {orig['avg_inference_time_ms']:.2f} ms")
        if "perplexity" in orig:
            report_lines.append(f"Perplexity: {orig['perplexity']:.2f}")
        report_lines.append("")

        # Compressed model metrics
        report_lines.append("COMPRESSED MODEL:")
        report_lines.append("-" * 40)
        comp = results["compressed"]
        if "total_parameters" in comp:
            report_lines.append(f"Parameters: {comp['total_parameters']:,}")
        if "model_size_mb" in comp:
            report_lines.append(f"Size: {comp['model_size_mb']:.2f} MB")
        if "avg_inference_time_ms" in comp:
            report_lines.append(f"Avg Inference Time: {comp['avg_inference_time_ms']:.2f} ms")
        if "perplexity" in comp:
            report_lines.append(f"Perplexity: {comp['perplexity']:.2f}")
        report_lines.append("")

        # Comparison metrics
        report_lines.append("COMPRESSION IMPACT:")
        report_lines.append("-" * 40)
        comparison = results["comparison"]
        if "compression_ratio" in comparison:
            report_lines.append(f"Compression Ratio: {comparison['compression_ratio']:.2f}x")
        if "size_reduction_percent" in comparison:
            report_lines.append(f"Size Reduction: {comparison['size_reduction_percent']:.1f}%")
        if "inference_speedup" in comparison:
            report_lines.append(f"Inference Speedup: {comparison['inference_speedup']:.2f}x")
        if "perplexity_increase_percent" in comparison:
            report_lines.append(
                f"Perplexity Increase: {comparison['perplexity_increase_percent']:.1f}%"
            )

        report_lines.append("")
        report_lines.append("=" * 80)

        report = "\n".join(report_lines)

        # Save to file if requested
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                f.write(report)
            logger.info(f"Evaluation report saved to: {output_path}")

        return report
