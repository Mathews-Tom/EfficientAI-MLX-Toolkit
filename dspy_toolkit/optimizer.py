"""
Intelligent optimizer engine for DSPy Integration Framework.
"""

# Standard library imports
import logging
import time
from collections.abc import Callable
from datetime import datetime

# Third-party imports
import dspy

# Optional third-party imports
try:
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Local imports
from .exceptions import DSPyIntegrationError, OptimizerFailureError
from .interfaces import OptimizerEngineInterface
from .types import OptimizationResult

logger = logging.getLogger(__name__)


class OptimizerEngine(OptimizerEngineInterface):
    """Intelligent optimizer selection and configuration for DSPy modules."""

    def __init__(self):
        """Initialize the optimizer engine."""
        self.optimizers = {
            "bootstrap": dspy.BootstrapFewShot,
            "mipro": getattr(dspy, "MIPROv2", None),  # May not be available in all versions
            "gepa": getattr(dspy, "GEPA", None),  # May not be available in all versions
        }

        # Remove None optimizers (not available in current DSPy version)
        self.optimizers = {k: v for k, v in self.optimizers.items() if v is not None}

        self.optimization_history: list[OptimizationResult] = []

        # Default optimizer configurations
        self.optimizer_configs = {
            "bootstrap": {
                "max_bootstrapped_demos": 8,
                "max_labeled_demos": 16,
                "max_rounds": 3,
            },
            "mipro": (
                {
                    "num_candidates": 10,
                    "init_temperature": 1.0,
                }
                if "mipro" in self.optimizers
                else {}
            ),
            "gepa": (
                {
                    "max_iters": 100,
                    "patience": 10,
                }
                if "gepa" in self.optimizers
                else {}
            ),
        }

        logger.info(
            "Optimizer engine initialized with optimizers: %s", list(self.optimizers.keys())
        )

    def select_optimizer(self, task_type: str, dataset_size: int, complexity: str) -> str:
        """Intelligently select optimizer based on task characteristics."""
        try:
            # Rule-based optimizer selection
            if dataset_size < 50:
                # Small datasets - use bootstrap
                if "bootstrap" in self.optimizers:
                    return "bootstrap"
            elif dataset_size < 200:
                # Medium datasets - prefer GEPA if available, otherwise bootstrap
                if "gepa" in self.optimizers and complexity in ["medium", "high"]:
                    return "gepa"
                elif "bootstrap" in self.optimizers:
                    return "bootstrap"
            else:
                # Large datasets - prefer MIPRO for complex tasks
                if "mipro" in self.optimizers and complexity == "high":
                    return "mipro"
                elif "gepa" in self.optimizers:
                    return "gepa"
                elif "bootstrap" in self.optimizers:
                    return "bootstrap"

            # Fallback to first available optimizer
            if self.optimizers:
                return list(self.optimizers.keys())[0]
            else:
                raise OptimizerFailureError("No optimizers available")

        except Exception as e:
            logger.error("Optimizer selection failed: %s", e)
            # Return bootstrap as safe fallback if available
            if "bootstrap" in self.optimizers:
                return "bootstrap"
            raise OptimizerFailureError("Optimizer selection failed") from e

    def optimize(
        self,
        module: dspy.Module,
        dataset: list[dict],
        metrics: list[str],
        task_type: str = "general",
    ) -> dspy.Module:
        """Optimize a DSPy module with best optimizer for task."""

        if not dataset:
            logger.warning("Empty dataset provided, returning unoptimized module")
            return module

        start_time = time.time()

        try:
            # Assess dataset complexity
            complexity = self._assess_complexity(dataset)

            # Select optimizer
            optimizer_name = self.select_optimizer(task_type, len(dataset), complexity)

            logger.info(
                "Selected optimizer %s for task_type=%s, dataset_size=%d, complexity=%s",
                optimizer_name,
                task_type,
                len(dataset),
                complexity,
            )

            # Get baseline performance
            original_performance = self._evaluate_module(
                module, dataset[:10], metrics
            )  # Sample for speed

            # Configure and run optimizer
            optimized_module = self._run_optimizer(
                optimizer_name, module, dataset, metrics, task_type
            )

            # Evaluate optimized performance
            optimized_performance = self._evaluate_module(optimized_module, dataset[:10], metrics)

            # Record optimization result
            optimization_time = time.time() - start_time
            result = OptimizationResult(
                optimizer_used=optimizer_name,
                original_performance=original_performance,
                optimized_performance=optimized_performance,
                optimization_time=optimization_time,
                num_examples_used=len(dataset),
                metadata={
                    "task_type": task_type,
                    "complexity": complexity,
                    "metrics": metrics,
                    "dataset_size": len(dataset),
                },
            )

            self.optimization_history.append(result)

            logger.info(
                "Optimization completed in %.2fs using %s",
                optimization_time,
                optimizer_name,
            )
            return optimized_module

        except Exception as e:
            logger.error("Optimization failed: %s", e)
            raise OptimizerFailureError("Module optimization failed") from e

    def _run_optimizer(
        self,
        optimizer_name: str,
        module: dspy.Module,
        dataset: list[dict],
        metrics: list[str],
        task_type: str,
    ) -> dspy.Module:
        """Run the specified optimizer on the module."""

        if optimizer_name not in self.optimizers:
            raise OptimizerFailureError(f"Optimizer {optimizer_name} not available")

        optimizer_class = self.optimizers[optimizer_name]
        config = self.optimizer_configs.get(optimizer_name, {})

        try:
            # Create metric function
            metric_func = self._create_metric_function(metrics)

            # Prepare dataset for DSPy (convert to DSPy format if needed)
            dspy_dataset = self._prepare_dataset(dataset)

            # Configure optimizer based on type
            if optimizer_name == "bootstrap":
                optimizer = optimizer_class(
                    metric=metric_func,
                    max_bootstrapped_demos=min(
                        config.get("max_bootstrapped_demos", 8), len(dataset) // 4
                    ),
                    max_labeled_demos=min(config.get("max_labeled_demos", 16), len(dataset) // 2),
                    max_rounds=config.get("max_rounds", 3),
                )
            elif optimizer_name == "mipro" and "mipro" in self.optimizers:
                optimizer = optimizer_class(
                    metric=metric_func,
                    num_candidates=config.get("num_candidates", 10),
                    init_temperature=config.get("init_temperature", 1.0),
                )
            elif optimizer_name == "gepa" and "gepa" in self.optimizers:
                optimizer = optimizer_class(
                    metric=metric_func,
                    max_iters=config.get("max_iters", 100),
                    patience=config.get("patience", 10),
                )
            else:
                # Fallback configuration
                optimizer = optimizer_class(metric=metric_func)

            # Run optimization
            logger.info("Running %s optimizer on %d examples", optimizer_name, len(dspy_dataset))
            optimized_module = optimizer.compile(module, trainset=dspy_dataset)

            return optimized_module

        except Exception as e:
            logger.error("Optimizer %s execution failed: %s", optimizer_name, e)
            raise OptimizerFailureError("Optimizer execution failed") from e

    def _create_metric_function(self, metrics: list[str]) -> Callable:
        """Create composite metric function from metric names."""

        def composite_metric(gold, pred, trace=None):
            """Composite metric function for DSPy optimization."""
            try:
                scores = {}

                # Handle different metric types
                for metric in metrics:
                    if metric == "accuracy":
                        if SKLEARN_AVAILABLE:
                            scores[metric] = accuracy_score([gold], [pred])
                        else:
                            scores[metric] = 1.0 if gold == pred else 0.0

                    elif metric == "f1" and SKLEARN_AVAILABLE:
                        scores[metric] = f1_score(
                            [gold], [pred], average="weighted", zero_division=0
                        )

                    elif metric == "precision" and SKLEARN_AVAILABLE:
                        scores[metric] = precision_score(
                            [gold], [pred], average="weighted", zero_division=0
                        )

                    elif metric == "recall" and SKLEARN_AVAILABLE:
                        scores[metric] = recall_score(
                            [gold], [pred], average="weighted", zero_division=0
                        )

                    elif metric == "exact_match":
                        scores[metric] = 1.0 if str(gold).strip() == str(pred).strip() else 0.0

                    elif metric == "contains":
                        scores[metric] = 1.0 if str(gold).lower() in str(pred).lower() else 0.0

                    else:
                        # Default to exact match for unknown metrics
                        scores[metric] = 1.0 if gold == pred else 0.0

                # Return average of all metrics
                if scores:
                    return sum(scores.values()) / len(scores)
                else:
                    return 0.0

            except Exception as e:
                logger.warning("Metric calculation failed: %s", e)
                return 0.0

        return composite_metric

    def _prepare_dataset(self, dataset: list[dict]) -> list[dspy.Example]:
        """Prepare dataset for DSPy optimization."""
        try:
            # Convert to DSPy Example format if needed
            dspy_examples = []

            for item in dataset:
                if isinstance(item, dict):
                    # Create DSPy Example from dict
                    example = dspy.Example(**item)
                    dspy_examples.append(example)
                else:
                    # Assume it's already in correct format
                    dspy_examples.append(item)

            return dspy_examples

        except Exception as e:
            logger.warning("Dataset preparation failed: %s, using original dataset", e)
            return dataset

    def _evaluate_module(
        self, module: dspy.Module, dataset: list[dict], metrics: list[str]
    ) -> dict[str, float]:
        """Evaluate module performance on dataset."""
        try:
            if not dataset:
                return {}

            metric_func = self._create_metric_function(metrics)
            scores = []

            # Sample a few examples for evaluation (to avoid long evaluation times)
            eval_dataset = dataset[: min(10, len(dataset))]

            for item in eval_dataset:
                try:
                    # Get prediction from module
                    if isinstance(item, dict):
                        # Extract input and expected output
                        inputs = {k: v for k, v in item.items() if not k.startswith("expected_")}
                        expected = item.get("expected_output", item.get("output", ""))

                        # Run module
                        result = module(**inputs)

                        # Extract prediction (handle different result formats)
                        if hasattr(result, "answer"):
                            prediction = result.answer
                        elif hasattr(result, "output"):
                            prediction = result.output
                        elif isinstance(result, dict):
                            prediction = result.get("answer", result.get("output", str(result)))
                        else:
                            prediction = str(result)

                        # Calculate score
                        score = metric_func(expected, prediction)
                        scores.append(score)

                except Exception as e:
                    logger.warning("Evaluation failed for item: %s", e)
                    scores.append(0.0)

            # Calculate aggregate metrics
            if scores:
                return {
                    "average_score": sum(scores) / len(scores),
                    "max_score": max(scores),
                    "min_score": min(scores),
                    "num_evaluated": len(scores),
                }
            else:
                return {"average_score": 0.0, "num_evaluated": 0}

        except Exception as e:
            logger.error("Module evaluation failed: %s", e)
            return {"error": str(e), "average_score": 0.0}

    def _assess_complexity(self, dataset: list[dict]) -> str:
        """Assess dataset complexity for optimizer selection."""
        try:
            if not dataset:
                return "low"

            # Simple heuristics for complexity assessment
            sample_size = min(10, len(dataset))
            sample = dataset[:sample_size]

            complexity_score = 0

            for item in sample:
                if isinstance(item, dict):
                    # Check input complexity
                    for key, value in item.items():
                        if isinstance(value, str):
                            # Text length indicates complexity
                            if len(value) > 500:
                                complexity_score += 2
                            elif len(value) > 100:
                                complexity_score += 1
                        elif isinstance(value, (list, dict)):
                            # Structured data indicates complexity
                            complexity_score += 1

            # Normalize by sample size
            avg_complexity = complexity_score / sample_size if sample_size > 0 else 0

            if avg_complexity > 2:
                return "high"
            elif avg_complexity > 1:
                return "medium"
            else:
                return "low"

        except Exception as e:
            logger.warning("Complexity assessment failed: %s", e)
            return "medium"  # Safe default

    def get_optimization_history(self) -> list[OptimizationResult]:
        """Get history of optimization results."""
        return self.optimization_history.copy()

    def get_optimizer_stats(self) -> dict[str, str | int | float | bool]:
        """Get statistics about the optimizer engine."""
        try:
            total_optimizations = len(self.optimization_history)

            if total_optimizations == 0:
                return {
                    "total_optimizations": 0,
                    "available_optimizers": list(self.optimizers.keys()),
                    "sklearn_available": SKLEARN_AVAILABLE,
                }

            # Calculate statistics from history
            avg_time = (
                sum(r.optimization_time for r in self.optimization_history) / total_optimizations
            )

            optimizer_usage = {}
            for result in self.optimization_history:
                optimizer = result.optimizer_used
                optimizer_usage[optimizer] = optimizer_usage.get(optimizer, 0) + 1

            # Calculate improvement statistics
            improvements = []
            for result in self.optimization_history:
                if result.original_performance and result.optimized_performance:
                    orig_score = result.original_performance.get("average_score", 0)
                    opt_score = result.optimized_performance.get("average_score", 0)
                    if orig_score > 0:
                        improvement = (opt_score - orig_score) / orig_score
                        improvements.append(improvement)

            avg_improvement = sum(improvements) / len(improvements) if improvements else 0

            return {
                "total_optimizations": total_optimizations,
                "available_optimizers": list(self.optimizers.keys()),
                "optimizer_usage": optimizer_usage,
                "average_optimization_time": avg_time,
                "average_improvement": avg_improvement,
                "sklearn_available": SKLEARN_AVAILABLE,
                "last_optimization": (
                    self.optimization_history[-1].metadata if self.optimization_history else None
                ),
            }

        except Exception as e:
            logger.error("Failed to get optimizer stats: %s", e)
            return {"error": str(e)}

    def clear_history(self) -> None:
        """Clear optimization history."""
        self.optimization_history.clear()
        logger.info("Optimization history cleared")

    def export_history(self, export_path: str) -> None:
        """Export optimization history to file."""
        try:
            import json

            # Convert OptimizationResult objects to dicts
            history_data = []
            for result in self.optimization_history:
                history_data.append(
                    {
                        "optimizer_used": result.optimizer_used,
                        "original_performance": result.original_performance,
                        "optimized_performance": result.optimized_performance,
                        "optimization_time": result.optimization_time,
                        "num_examples_used": result.num_examples_used,
                        "metadata": result.metadata,
                    }
                )

            export_data = {
                "optimization_history": history_data,
                "stats": self.get_optimizer_stats(),
                "exported_at": datetime.now().isoformat(),
            }

            with open(export_path, "w") as f:
                json.dump(export_data, f, indent=2)

            logger.info("Optimization history exported to %s", export_path)

        except Exception as e:
            logger.error("Failed to export optimization history: %s", e)
            raise DSPyIntegrationError("History export failed") from e
