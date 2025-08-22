"""
Automated hyperparameter tuner for LoRA fine-tuning.

Intelligent hyperparameter optimization with Bayesian optimization,
early stopping, multi-objective optimization, and Apple Silicon awareness.
"""

import json
import logging
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from lora import LoRAConfig, ModelAdapter, TrainingConfig
from optimization.objectives import OptimizationObjective, PerplexityObjective
from optimization.search import BayesianOptimization, SearchStrategy
from training import LoRATrainer

# Import shared utilities with fallback
try:
    from utils.logging_utils import get_logger
except ImportError:

    def get_logger(name: str) -> logging.Logger:
        return logging.getLogger(name)


logger = get_logger(__name__)


@dataclass
class HyperparameterSpace:
    """Definition of hyperparameter search space."""

    # LoRA parameters
    rank: tuple[int, int] = (8, 64)
    alpha: tuple[float, float] = (8.0, 64.0)
    dropout: tuple[float, float] = (0.0, 0.3)

    # Training parameters
    learning_rate: tuple[float, float] = (1e-5, 5e-4)
    batch_size: list[int] = field(default_factory=lambda: [1, 2, 4])
    warmup_steps: tuple[int, int] = (50, 200)
    weight_decay: tuple[float, float] = (0.0, 0.1)

    # Optimization parameters
    optimizer: list[str] = field(default_factory=lambda: ["adamw", "adam"])
    scheduler: list[str] = field(default_factory=lambda: ["linear", "cosine"])

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "rank": self.rank,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "warmup_steps": self.warmup_steps,
            "weight_decay": self.weight_decay,
            "optimizer": self.optimizer,
            "scheduler": self.scheduler,
        }

    def sample(self) -> dict[str, Any]:
        """Sample a random configuration from the space."""
        config = {}

        # Continuous parameters
        config["rank"] = random.randint(self.rank[0], self.rank[1])
        config["alpha"] = random.uniform(self.alpha[0], self.alpha[1])
        config["dropout"] = random.uniform(self.dropout[0], self.dropout[1])
        config["learning_rate"] = random.uniform(self.learning_rate[0], self.learning_rate[1])
        config["warmup_steps"] = random.randint(self.warmup_steps[0], self.warmup_steps[1])
        config["weight_decay"] = random.uniform(self.weight_decay[0], self.weight_decay[1])

        # Categorical parameters
        config["batch_size"] = random.choice(self.batch_size)
        config["optimizer"] = random.choice(self.optimizer)
        config["scheduler"] = random.choice(self.scheduler)

        return config


@dataclass
class OptimizationResult:
    """Result of a single optimization trial."""

    trial_id: str
    hyperparameters: dict[str, Any]
    objective_value: float
    metrics: dict[str, float]

    # Training information
    training_time: float
    epochs_completed: int
    early_stopped: bool = False

    # Resource usage
    peak_memory_mb: float = 0.0

    # Timestamps
    start_time: str = field(default_factory=lambda: datetime.now().isoformat())
    end_time: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "trial_id": self.trial_id,
            "hyperparameters": self.hyperparameters,
            "objective_value": self.objective_value,
            "metrics": self.metrics,
            "training_time": self.training_time,
            "epochs_completed": self.epochs_completed,
            "early_stopped": self.early_stopped,
            "peak_memory_mb": self.peak_memory_mb,
            "start_time": self.start_time,
            "end_time": self.end_time,
        }


class AutoTuner:
    """
    Automated hyperparameter tuner for LoRA fine-tuning.

    Provides intelligent hyperparameter search with multiple optimization
    strategies, early stopping, and Apple Silicon performance awareness.
    """

    def __init__(
        self,
        base_model: Any,
        train_dataset: Any,
        eval_dataset: Any,
        hyperparameter_space: HyperparameterSpace | None = None,
        search_strategy: SearchStrategy | None = None,
        objective: OptimizationObjective | None = None,
        n_trials: int = 20,
        max_epochs_per_trial: int = 3,
        early_stopping_patience: int = 2,
        output_dir: Path = Path("optimization_results/"),
        resource_limits: dict[str, Any] | None = None,
    ):
        self.base_model = base_model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.hyperparameter_space = hyperparameter_space or HyperparameterSpace()
        self.search_strategy = search_strategy or BayesianOptimization(self.hyperparameter_space)
        self.objective = objective or PerplexityObjective()

        self.n_trials = n_trials
        self.max_epochs_per_trial = max_epochs_per_trial
        self.early_stopping_patience = early_stopping_patience
        self.output_dir = Path(output_dir)
        self.resource_limits = resource_limits or {}

        # Optimization state
        self.results: list[OptimizationResult] = []
        self.best_result: OptimizationResult | None = None
        self.optimization_history = []

        # Setup output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def optimize(self) -> OptimizationResult:
        """
        Run hyperparameter optimization.

        Returns:
            Best optimization result found
        """
        logger.info("Starting hyperparameter optimization with %d trials", self.n_trials)
        logger.info("Output directory: %s", self.output_dir)
        logger.info("Search strategy: %s", self.search_strategy.__class__.__name__)
        logger.info("Objective: %s", self.objective.__class__.__name__)

        optimization_start_time = time.time()

        try:
            for trial_idx in range(self.n_trials):
                logger.info("Trial %d/%d", trial_idx + 1, self.n_trials)

                # Get next hyperparameter configuration
                hyperparameters = self.search_strategy.suggest(trial_history=self.results)

                logger.info("Trying hyperparameters: %s", hyperparameters)

                # Run single trial
                result = self._run_trial(
                    trial_id=f"trial_{trial_idx:03d}",
                    hyperparameters=hyperparameters,
                )

                # Update results
                self.results.append(result)

                # Update best result
                if self.best_result is None or self._is_better_result(result, self.best_result):
                    self.best_result = result
                    logger.info("New best result! Objective: %.4f", result.objective_value)

                # Save intermediate results
                self._save_results()

                # Report progress
                logger.info(
                    "Trial completed: objective=%.4f, time=%.2fs",
                    result.objective_value,
                    result.training_time,
                )
                logger.info("Best so far: %.4f", self.best_result.objective_value)

                # Early termination check
                if self._should_terminate_optimization():
                    logger.info("Early termination triggered")
                    break

        except KeyboardInterrupt:
            logger.info("Optimization interrupted by user")

        except Exception as e:
            logger.error("Optimization failed with error: %s", e)
            raise e

        finally:
            total_optimization_time = time.time() - optimization_start_time

            # Final results summary
            logger.info("Optimization Complete")
            logger.info("Total trials: %d", len(self.results))
            logger.info("Total time: %.2f seconds", total_optimization_time)

            if self.best_result:
                logger.info("Best objective: %.4f", self.best_result.objective_value)
                logger.info("Best hyperparameters: %s", self.best_result.hyperparameters)

            # Save final results
            self._save_results()
            self._generate_optimization_report(total_optimization_time)

        return self.best_result

    def _run_trial(self, trial_id: str, hyperparameters: dict[str, Any]) -> OptimizationResult:
        """Run a single optimization trial."""
        trial_start_time = time.time()

        try:
            # Create configurations from hyperparameters
            lora_config, training_config = self._create_configs_from_hyperparameters(
                hyperparameters
            )

            # Create trainer
            trainer = LoRATrainer(
                model=self.base_model,
                lora_config=lora_config,
                training_config=training_config,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
            )

            # Run training
            training_result = trainer.train()

            # Calculate objective value
            objective_value = self.objective.calculate(training_result, trainer.state)

            # Extract metrics
            metrics = self._extract_metrics(training_result, trainer.state)

            # Create result
            result = OptimizationResult(
                trial_id=trial_id,
                hyperparameters=hyperparameters,
                objective_value=objective_value,
                metrics=metrics,
                training_time=time.time() - trial_start_time,
                epochs_completed=trainer.state.epoch,
                early_stopped=training_result.get("early_stopped", False),
                peak_memory_mb=trainer.state.peak_memory_mb,
                end_time=datetime.now().isoformat(),
            )

            return result

        except Exception as e:
            logger.error("Trial %s failed: %s", trial_id, e)

            # Create failed result
            return OptimizationResult(
                trial_id=trial_id,
                hyperparameters=hyperparameters,
                objective_value=float("inf"),  # Worst possible value
                metrics={"error": str(e)},
                training_time=time.time() - trial_start_time,
                epochs_completed=0,
                early_stopped=True,
                end_time=datetime.now().isoformat(),
            )

    def _create_configs_from_hyperparameters(
        self, hyperparameters: dict[str, Any]
    ) -> tuple[LoRAConfig, TrainingConfig]:
        """Create LoRA and training configs from hyperparameters."""
        # Create LoRA config
        lora_config = LoRAConfig(
            rank=int(hyperparameters["rank"]),
            alpha=float(hyperparameters["alpha"]),
            dropout=float(hyperparameters["dropout"]),
        )

        # Create training config
        training_config = TrainingConfig(
            learning_rate=float(hyperparameters["learning_rate"]),
            batch_size=int(hyperparameters["batch_size"]),
            warmup_steps=int(hyperparameters["warmup_steps"]),
            weight_decay=float(hyperparameters["weight_decay"]),
            optimizer=hyperparameters["optimizer"],
            scheduler=hyperparameters["scheduler"],
            num_epochs=self.max_epochs_per_trial,
            early_stopping_patience=self.early_stopping_patience,
            output_dir=self.output_dir / "trial_outputs",
        )

        return lora_config, training_config

    def _extract_metrics(
        self, training_result: dict[str, Any], training_state: Any
    ) -> dict[str, float]:
        """Extract metrics from training result."""
        metrics = {
            "best_metric": training_result.get("best_metric", float("inf")),
            "best_epoch": training_result.get("best_epoch", 0),
            "total_training_time": training_result.get("total_training_time", 0.0),
            "peak_memory_mb": training_state.peak_memory_mb,
            "final_train_loss": training_state.train_loss,
            "final_eval_loss": training_state.eval_loss,
        }

        return metrics

    def _is_better_result(self, result1: OptimizationResult, result2: OptimizationResult) -> bool:
        """Check if result1 is better than result2."""
        return self.objective.is_better(result1.objective_value, result2.objective_value)

    def _should_terminate_optimization(self) -> bool:
        """Check if optimization should be terminated early."""
        if len(self.results) < 5:  # Need at least 5 trials
            return False

        # Check if recent trials are not improving
        recent_objectives = [r.objective_value for r in self.results[-5:]]
        best_recent = (
            min(recent_objectives)
            if self.objective.direction == "minimize"
            else max(recent_objectives)
        )

        if self.objective.direction == "minimize":
            improvement = self.best_result.objective_value - best_recent
        else:
            improvement = best_recent - self.best_result.objective_value

        # Terminate if no significant improvement
        return improvement < 0.001

    def _save_results(self) -> None:
        """Save optimization results to files."""
        # Save all results
        results_data = [result.to_dict() for result in self.results]
        with open(self.output_dir / "optimization_results.json", "w") as f:
            json.dump(results_data, f, indent=2)

        # Save best result
        if self.best_result:
            with open(self.output_dir / "best_result.json", "w") as f:
                json.dump(self.best_result.to_dict(), f, indent=2)

        # Save hyperparameter space
        with open(self.output_dir / "hyperparameter_space.json", "w") as f:
            json.dump(self.hyperparameter_space.to_dict(), f, indent=2)

    def _generate_optimization_report(self, total_time: float) -> None:
        """Generate comprehensive optimization report."""
        report = {
            "optimization_summary": {
                "total_trials": len(self.results),
                "successful_trials": len(
                    [r for r in self.results if r.objective_value != float("inf")]
                ),
                "total_optimization_time": total_time,
                "search_strategy": self.search_strategy.__class__.__name__,
                "objective": self.objective.__class__.__name__,
            },
            "best_result": self.best_result.to_dict() if self.best_result else None,
            "hyperparameter_space": self.hyperparameter_space.to_dict(),
            "trial_results": [result.to_dict() for result in self.results],
        }

        with open(self.output_dir / "optimization_report.json", "w") as f:
            json.dump(report, f, indent=2)

        logger.info("Optimization report saved to %s", self.output_dir / "optimization_report.json")

    def get_best_config(self) -> tuple[LoRAConfig, TrainingConfig]:
        """Get the best LoRA and training configurations found."""
        if self.best_result is None:
            raise RuntimeError("No optimization results available")

        return self._create_configs_from_hyperparameters(self.best_result.hyperparameters)

    def plot_optimization_history(self, save_path: Path | None = None) -> None:
        """Plot optimization history (requires matplotlib)."""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            logger.warning("matplotlib not available for plotting")
            return

        if not self.results:
            logger.warning("No results to plot")
            return

        # Extract data
        trial_numbers = list(range(1, len(self.results) + 1))
        objective_values = [
            r.objective_value for r in self.results if r.objective_value != float("inf")
        ]
        training_times = [r.training_time for r in self.results]

        # Create subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # Plot objective values
        ax1.plot(trial_numbers[: len(objective_values)], objective_values, "b-o", markersize=4)
        ax1.set_xlabel("Trial Number")
        ax1.set_ylabel("Objective Value")
        ax1.set_title("Optimization Progress")
        ax1.grid(True, alpha=0.3)

        # Plot best objective so far
        if self.objective.direction == "minimize":
            best_so_far = np.minimum.accumulate(objective_values)
        else:
            best_so_far = np.maximum.accumulate(objective_values)

        ax1.plot(
            trial_numbers[: len(best_so_far)], best_so_far, "r--", linewidth=2, label="Best So Far"
        )
        ax1.legend()

        # Plot training times
        ax2.bar(trial_numbers, training_times, alpha=0.7, color="green")
        ax2.set_xlabel("Trial Number")
        ax2.set_ylabel("Training Time (seconds)")
        ax2.set_title("Training Time per Trial")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        if save_path is None:
            save_path = self.output_dir / "optimization_history.png"

        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info("Optimization history plot saved to %s", save_path)


def run_optimization(
    model_name: str,
    dataset_path: Path,
    output_dir: Path,
    n_trials: int = 20,
    max_epochs_per_trial: int = 3,
) -> OptimizationResult:
    """
    Convenience function to run hyperparameter optimization.

    Args:
        model_name: Name or path of base model
        dataset_path: Path to training dataset
        output_dir: Directory to save results
        n_trials: Number of optimization trials
        max_epochs_per_trial: Maximum epochs per trial

    Returns:
        Best optimization result
    """
    logger.info("Running optimization for model: %s", model_name)
    logger.info("Dataset: %s", dataset_path)

    # This would load actual model and dataset in a real implementation
    # For now, create placeholder
    base_model = None  # load_model(model_name)
    train_dataset = None  # load_dataset(dataset_path / "train")
    eval_dataset = None  # load_dataset(dataset_path / "eval")

    # Create tuner
    tuner = AutoTuner(
        base_model=base_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        n_trials=n_trials,
        max_epochs_per_trial=max_epochs_per_trial,
        output_dir=output_dir,
    )

    # Run optimization
    best_result = tuner.optimize()

    # Generate plots
    tuner.plot_optimization_history()

    return best_result
