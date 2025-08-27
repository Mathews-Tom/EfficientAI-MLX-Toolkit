"""
Comprehensive benchmarking for model compression methods.
"""

import logging
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
import yaml

# Import shared utilities with fallback
try:
    from utils.logging_utils import get_logger
except ImportError:

    def get_logger(name: str) -> logging.Logger:
        return logging.getLogger(name)


from evaluation import CompressionEvaluator
from pruning import MLXPruner, PruningConfig

# Import compression modules
from quantization import MLXQuantizer, QuantizationConfig

logger = get_logger(__name__)


class CompressionBenchmark:
    """Comprehensive benchmarking for model compression methods."""

    def __init__(self, config_path: Path):
        """Initialize benchmark with configuration."""
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.evaluator = CompressionEvaluator()
        self.benchmark_results = {}

    def _load_config(self) -> dict[str, Any]:
        """Load benchmark configuration."""
        try:
            with open(self.config_path, "r") as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            logger.warning(f"Could not load config from {self.config_path}: {e}")
            return {}

    def run_comprehensive_benchmark(
        self, model_path: str, methods: list[str], output_dir: Path | None = None
    ) -> dict[str, Any]:
        """
        Run comprehensive benchmarking of compression methods.

        Args:
            model_path: Path to model to benchmark
            methods: List of compression methods to test
            output_dir: Output directory for results

        Returns:
            Dictionary of benchmark results
        """
        logger.info(f"Starting comprehensive benchmark for methods: {methods}")
        logger.info(f"Model: {model_path}")

        results = {
            "benchmark_info": {
                "model_path": model_path,
                "methods": methods,
                "timestamp": time.time(),
                "config_path": str(self.config_path),
            },
            "method_results": {},
            "summary": {},
        }

        try:
            # Load original model
            from mlx_lm.utils import load as load_model_and_tokenizer

            original_model, tokenizer = load_model_and_tokenizer(model_path)
            logger.info("Original model loaded successfully")

            # Baseline evaluation
            logger.info("Running baseline evaluation")
            baseline_results = self._benchmark_baseline(original_model, tokenizer)
            results["baseline"] = baseline_results

            # Benchmark each compression method
            for method in methods:
                logger.info(f"Benchmarking compression method: {method}")

                try:
                    method_results = self._benchmark_compression_method(
                        method, original_model, tokenizer, model_path
                    )
                    results["method_results"][method] = method_results

                except Exception as e:
                    logger.error(f"Failed to benchmark method {method}: {e}")
                    results["method_results"][method] = {"error": str(e), "status": "failed"}

            # Generate summary
            results["summary"] = self._generate_benchmark_summary(results)

            # Save results if output directory specified
            if output_dir:
                self._save_benchmark_results(results, output_dir)

            self.benchmark_results = results
            logger.info("Comprehensive benchmark completed")

            return results

        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
            raise

    def _benchmark_baseline(self, model: Any, tokenizer: Any) -> dict[str, Any]:
        """Benchmark the baseline (uncompressed) model."""
        logger.info("Benchmarking baseline model")

        baseline_results = {}

        try:
            # Model size and parameter count
            baseline_results.update(self._measure_model_metrics(model))

            # Performance benchmarking
            baseline_results.update(
                self._benchmark_inference_performance(model, tokenizer, "baseline")
            )

            # Memory usage
            baseline_results.update(self._measure_memory_usage(model))

            # Accuracy metrics (if eval data available)
            eval_data = self._get_evaluation_data()
            if eval_data:
                baseline_results.update(self._measure_accuracy_metrics(model, tokenizer, eval_data))

        except Exception as e:
            logger.error(f"Baseline benchmarking failed: {e}")
            baseline_results["error"] = str(e)

        return baseline_results

    def _benchmark_compression_method(
        self, method: str, original_model: Any, tokenizer: Any, model_path: str
    ) -> dict[str, Any]:
        """Benchmark a specific compression method."""
        method_results = {
            "method": method,
            "status": "started",
            "compression_time": 0.0,
        }

        try:
            start_time = time.time()

            # Apply compression method
            if method == "quantization":
                compressed_model = self._apply_quantization(original_model, model_path)
            elif method == "pruning":
                compressed_model = self._apply_pruning(original_model)
            elif method == "combined":
                compressed_model = self._apply_combined_compression(original_model, model_path)
            else:
                raise ValueError(f"Unknown compression method: {method}")

            compression_time = time.time() - start_time
            method_results["compression_time"] = compression_time
            method_results["status"] = "compressed"

            # Benchmark compressed model
            logger.info(f"Evaluating {method} compressed model")

            # Model metrics
            method_results.update(self._measure_model_metrics(compressed_model))

            # Performance benchmarking
            method_results.update(
                self._benchmark_inference_performance(compressed_model, tokenizer, method)
            )

            # Memory usage
            method_results.update(self._measure_memory_usage(compressed_model))

            # Accuracy metrics
            eval_data = self._get_evaluation_data()
            if eval_data:
                method_results.update(
                    self._measure_accuracy_metrics(compressed_model, tokenizer, eval_data)
                )

            method_results["status"] = "completed"

        except Exception as e:
            logger.error(f"Failed to benchmark {method}: {e}")
            method_results["error"] = str(e)
            method_results["status"] = "failed"

        return method_results

    def _apply_quantization(self, model: Any, model_path: str) -> Any:
        """Apply quantization compression."""
        logger.info("Applying quantization compression")

        # Create quantization config from benchmark config
        quant_config_dict = self.config.get("quantization", {})
        quant_config = QuantizationConfig.from_dict(quant_config_dict)

        # Create quantizer and apply
        quantizer = MLXQuantizer(quant_config)
        quantized_model = quantizer.quantize(model_path=model_path)

        return quantized_model

    def _apply_pruning(self, model: Any) -> Any:
        """Apply pruning compression."""
        logger.info("Applying pruning compression")

        # Create pruning config from benchmark config
        prune_config_dict = self.config.get("pruning", {})
        prune_config = PruningConfig.from_dict(prune_config_dict)

        # Create pruner and apply
        pruner = MLXPruner(prune_config)
        pruned_model = pruner.prune(model)

        return pruned_model

    def _apply_combined_compression(self, model: Any, model_path: str) -> Any:
        """Apply combined quantization and pruning."""
        logger.info("Applying combined compression (quantization + pruning)")

        # First apply pruning
        pruned_model = self._apply_pruning(model)

        # Then apply quantization (would need to save/reload in practice)
        # For now, just return pruned model
        logger.warning("Combined compression: quantization after pruning not fully implemented")
        return pruned_model

    def _measure_model_metrics(self, model: Any) -> dict[str, Any]:
        """Measure basic model metrics."""
        metrics = {}

        try:
            total_params = 0
            total_bytes = 0

            if hasattr(model, "parameters"):
                for param in model.parameters():
                    if isinstance(param, mx.array):
                        total_params += param.size
                        total_bytes += param.nbytes
                    elif hasattr(param, "shape"):
                        param_size = np.prod(param.shape)
                        total_params += param_size
                        total_bytes += param_size * 4  # Assume float32

            metrics.update(
                {
                    "total_parameters": total_params,
                    "model_size_mb": total_bytes / (1024 * 1024),
                    "model_size_bytes": total_bytes,
                }
            )

        except Exception as e:
            logger.warning(f"Could not measure model metrics: {e}")

        return metrics

    def _benchmark_inference_performance(
        self, model: Any, tokenizer: Any, model_type: str, num_runs: int = 50
    ) -> dict[str, Any]:
        """Benchmark inference performance."""
        logger.debug(f"Benchmarking inference performance for {model_type}")

        performance_metrics = {}

        try:
            # Test prompts
            test_prompts = [
                "The future of artificial intelligence",
                "Machine learning enables computers to",
                "Deep neural networks are capable of",
            ]

            # Prepare test inputs
            test_inputs = []
            for prompt in test_prompts:
                if tokenizer:
                    tokens = tokenizer.encode(prompt)[:100]  # Limit length
                    test_inputs.append(tokens)
                else:
                    test_inputs.append(list(range(50)))  # Dummy tokens

            # Warmup
            for _ in range(5):
                try:
                    for tokens in test_inputs:
                        input_ids = mx.array([tokens])
                        with mx.stream(mx.default_stream()):
                            _ = model(input_ids)
                            mx.eval(_)
                except Exception:
                    break

            # Actual benchmarking
            start_time = time.time()
            successful_runs = 0

            for run in range(num_runs):
                try:
                    for tokens in test_inputs:
                        input_ids = mx.array([tokens])
                        with mx.stream(mx.default_stream()):
                            _ = model(input_ids)
                            mx.eval(_)
                    successful_runs += 1
                except Exception:
                    continue

            total_time = time.time() - start_time

            if successful_runs > 0:
                avg_time = total_time / successful_runs
                throughput = successful_runs / total_time

                performance_metrics.update(
                    {
                        f"{model_type}_avg_inference_time_ms": avg_time * 1000,
                        f"{model_type}_throughput_samples_per_sec": throughput,
                        f"{model_type}_successful_runs": successful_runs,
                        f"{model_type}_total_runs": num_runs,
                    }
                )
            else:
                performance_metrics.update(
                    {
                        f"{model_type}_avg_inference_time_ms": float("inf"),
                        f"{model_type}_throughput_samples_per_sec": 0.0,
                        f"{model_type}_successful_runs": 0,
                        f"{model_type}_total_runs": num_runs,
                    }
                )

        except Exception as e:
            logger.warning(f"Could not benchmark inference performance: {e}")

        return performance_metrics

    def _measure_memory_usage(self, model: Any) -> dict[str, Any]:
        """Measure memory usage."""
        memory_metrics = {}

        try:
            total_bytes = 0

            if hasattr(model, "parameters"):
                for param in model.parameters():
                    if isinstance(param, mx.array):
                        total_bytes += param.nbytes
                    elif hasattr(param, "shape"):
                        total_bytes += np.prod(param.shape) * 4  # Assume float32

            memory_metrics.update(
                {
                    "memory_usage_mb": total_bytes / (1024 * 1024),
                    "memory_usage_bytes": total_bytes,
                }
            )

        except Exception as e:
            logger.warning(f"Could not measure memory usage: {e}")

        return memory_metrics

    def _measure_accuracy_metrics(
        self, model: Any, tokenizer: Any, eval_data: list[str]
    ) -> dict[str, Any]:
        """Measure accuracy metrics."""
        accuracy_metrics = {}

        try:
            # Calculate perplexity
            perplexity = self._calculate_perplexity(model, tokenizer, eval_data[:20])
            accuracy_metrics["perplexity"] = perplexity

            # Calculate cross-entropy loss
            ce_loss = self._calculate_cross_entropy_loss(model, tokenizer, eval_data[:20])
            accuracy_metrics["cross_entropy_loss"] = ce_loss

        except Exception as e:
            logger.warning(f"Could not measure accuracy metrics: {e}")

        return accuracy_metrics

    def _calculate_perplexity(self, model: Any, tokenizer: Any, texts: list[str]) -> float:
        """Calculate perplexity (simplified implementation)."""
        try:
            total_log_likelihood = 0.0
            total_tokens = 0

            for text in texts:
                if not tokenizer:
                    continue

                tokens = tokenizer.encode(text)[:200]  # Limit length
                if len(tokens) < 2:
                    continue

                input_ids = mx.array([tokens[:-1]])
                target_ids = mx.array([tokens[1:]])

                with mx.stream(mx.default_stream()):
                    logits = model(input_ids)
                    mx.eval(logits)

                log_probs = mx.log_softmax(logits, axis=-1)
                # Simplified perplexity calculation
                total_log_likelihood += mx.sum(log_probs)
                total_tokens += len(tokens) - 1

            if total_tokens > 0:
                avg_log_likelihood = total_log_likelihood / total_tokens
                perplexity = mx.exp(-avg_log_likelihood)
                return float(perplexity)

        except Exception as e:
            logger.warning(f"Could not calculate perplexity: {e}")

        return float("inf")

    def _calculate_cross_entropy_loss(self, model: Any, tokenizer: Any, texts: list[str]) -> float:
        """Calculate cross-entropy loss (simplified implementation)."""
        try:
            total_loss = 0.0
            total_samples = 0

            for text in texts:
                if not tokenizer:
                    continue

                tokens = tokenizer.encode(text)[:200]
                if len(tokens) < 2:
                    continue

                input_ids = mx.array([tokens[:-1]])
                target_ids = mx.array([tokens[1:]])

                with mx.stream(mx.default_stream()):
                    logits = model(input_ids)
                    mx.eval(logits)

                loss = mx.mean(mx.cross_entropy(logits, target_ids))
                total_loss += float(loss)
                total_samples += 1

            if total_samples > 0:
                return total_loss / total_samples

        except Exception as e:
            logger.warning(f"Could not calculate cross-entropy loss: {e}")

        return float("inf")

    def _get_evaluation_data(self) -> list[str] | None:
        """Get evaluation data from config or create dummy data."""
        eval_config = self.config.get("evaluation", {})

        # Try to load evaluation dataset
        eval_dataset_path = eval_config.get("eval_dataset")
        if eval_dataset_path:
            try:
                eval_path = Path(eval_dataset_path)
                if eval_path.exists():
                    with open(eval_path, "r") as f:
                        return [line.strip() for line in f if line.strip()][:100]
            except Exception as e:
                logger.warning(f"Could not load evaluation dataset: {e}")

        # Return dummy evaluation data
        return [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is a subset of artificial intelligence.",
            "Apple Silicon processors provide excellent performance for AI workloads.",
            "Model compression techniques include quantization and pruning.",
            "MLX framework enables efficient computation on Apple hardware.",
        ]

    def _generate_benchmark_summary(self, results: dict[str, Any]) -> dict[str, Any]:
        """Generate summary statistics from benchmark results."""
        summary = {}

        try:
            baseline = results.get("baseline", {})
            method_results = results.get("method_results", {})

            # Summary for each method
            for method, method_data in method_results.items():
                if method_data.get("status") != "completed":
                    continue

                method_summary = {}

                # Compression ratio
                if "model_size_mb" in baseline and "model_size_mb" in method_data:
                    orig_size = baseline["model_size_mb"]
                    comp_size = method_data["model_size_mb"]
                    if comp_size > 0:
                        compression_ratio = orig_size / comp_size
                        method_summary["compression_ratio"] = compression_ratio
                        method_summary["size_reduction_percent"] = (
                            (orig_size - comp_size) / orig_size
                        ) * 100

                # Performance comparison
                baseline_time_key = f"baseline_avg_inference_time_ms"
                method_time_key = f"{method}_avg_inference_time_ms"

                if baseline_time_key in baseline and method_time_key in method_data:
                    baseline_time = baseline[baseline_time_key]
                    method_time = method_data[method_time_key]
                    if method_time > 0 and baseline_time > 0:
                        speedup = baseline_time / method_time
                        method_summary["inference_speedup"] = speedup

                # Accuracy degradation
                if "perplexity" in baseline and "perplexity" in method_data:
                    baseline_ppl = baseline["perplexity"]
                    method_ppl = method_data["perplexity"]
                    if baseline_ppl > 0:
                        ppl_increase = ((method_ppl - baseline_ppl) / baseline_ppl) * 100
                        method_summary["perplexity_increase_percent"] = ppl_increase

                summary[method] = method_summary

        except Exception as e:
            logger.warning(f"Could not generate benchmark summary: {e}")

        return summary

    def _save_benchmark_results(self, results: dict[str, Any], output_dir: Path) -> None:
        """Save benchmark results to files."""
        try:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save full results as YAML
            results_path = output_dir / "benchmark_results.yaml"
            with open(results_path, "w") as f:
                yaml.dump(results, f, default_flow_style=False)

            # Save summary as JSON for easy parsing
            summary_path = output_dir / "benchmark_summary.json"
            import json

            with open(summary_path, "w") as f:
                json.dump(results["summary"], f, indent=2)

            # Generate and save text report
            report = self._generate_text_report(results)
            report_path = output_dir / "benchmark_report.txt"
            with open(report_path, "w") as f:
                f.write(report)

            logger.info(f"Benchmark results saved to: {output_dir}")

        except Exception as e:
            logger.error(f"Failed to save benchmark results: {e}")

    def _generate_text_report(self, results: dict[str, Any]) -> str:
        """Generate a human-readable text report."""
        report_lines = []

        report_lines.append("=" * 80)
        report_lines.append("COMPRESSION BENCHMARK REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")

        # Benchmark info
        info = results.get("benchmark_info", {})
        report_lines.append(f"Model: {info.get('model_path', 'Unknown')}")
        report_lines.append(f"Methods: {', '.join(info.get('methods', []))}")
        report_lines.append("")

        # Baseline results
        baseline = results.get("baseline", {})
        report_lines.append("BASELINE MODEL:")
        report_lines.append("-" * 40)
        if "total_parameters" in baseline:
            report_lines.append(f"Parameters: {baseline['total_parameters']:,}")
        if "model_size_mb" in baseline:
            report_lines.append(f"Size: {baseline['model_size_mb']:.2f} MB")
        if "baseline_avg_inference_time_ms" in baseline:
            report_lines.append(
                f"Avg Inference: {baseline['baseline_avg_inference_time_ms']:.2f} ms"
            )
        if "perplexity" in baseline:
            report_lines.append(f"Perplexity: {baseline['perplexity']:.2f}")
        report_lines.append("")

        # Method results
        summary = results.get("summary", {})
        for method, method_summary in summary.items():
            report_lines.append(f"{method.upper()} COMPRESSION:")
            report_lines.append("-" * 40)

            if "compression_ratio" in method_summary:
                report_lines.append(
                    f"Compression Ratio: {method_summary['compression_ratio']:.2f}x"
                )
            if "size_reduction_percent" in method_summary:
                report_lines.append(
                    f"Size Reduction: {method_summary['size_reduction_percent']:.1f}%"
                )
            if "inference_speedup" in method_summary:
                report_lines.append(
                    f"Inference Speedup: {method_summary['inference_speedup']:.2f}x"
                )
            if "perplexity_increase_percent" in method_summary:
                report_lines.append(
                    f"Perplexity Increase: {method_summary['perplexity_increase_percent']:.1f}%"
                )

            report_lines.append("")

        report_lines.append("=" * 80)

        return "\n".join(report_lines)
