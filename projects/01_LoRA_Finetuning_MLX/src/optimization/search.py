"""
Search strategies for hyperparameter optimization.

Provides various search algorithms for finding optimal hyperparameters
including random search, Bayesian optimization, and grid search.
"""

import random
from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class SearchStrategy(ABC):
    """Base class for hyperparameter search strategies."""

    def __init__(self, hyperparameter_space: Any):
        self.hyperparameter_space = hyperparameter_space

    @abstractmethod
    def suggest(self, trial_history: list[Any]) -> dict[str, Any]:
        """Suggest next hyperparameter configuration."""
        pass


class RandomSearch(SearchStrategy):
    """Random search strategy."""

    def suggest(self, trial_history: list[Any]) -> dict[str, Any]:
        """Suggest random hyperparameters from the space."""
        return self.hyperparameter_space.sample()


class BayesianOptimization(SearchStrategy):
    """Bayesian optimization strategy (simplified implementation)."""

    def __init__(self, hyperparameter_space: Any, n_random_starts: int = 5):
        super().__init__(hyperparameter_space)
        self.n_random_starts = n_random_starts

    def suggest(self, trial_history: list[Any]) -> dict[str, Any]:
        """Suggest hyperparameters using Bayesian optimization."""
        # For first few trials, use random search
        if len(trial_history) < self.n_random_starts:
            return self.hyperparameter_space.sample()

        # Simplified Bayesian optimization - in practice would use proper GP
        # For now, use random search with slight bias toward good regions
        best_trials = sorted(trial_history, key=lambda x: x.objective_value)[:3]

        if best_trials:
            # Sample near best configurations with some randomness
            base_config = best_trials[0].hyperparameters
            config = self.hyperparameter_space.sample()

            # Blend with best configuration (simplified)
            for key in config:
                if key in base_config and random.random() < 0.3:
                    if isinstance(config[key], (int, float)):
                        # Add noise to best value
                        noise = random.gauss(0, 0.1) * abs(base_config[key])
                        config[key] = max(0, base_config[key] + noise)

            return config

        return self.hyperparameter_space.sample()


class GridSearch(SearchStrategy):
    """Grid search strategy."""

    def __init__(self, hyperparameter_space: Any, grid_density: int = 3):
        super().__init__(hyperparameter_space)
        self.grid_density = grid_density
        self.grid_points = self._generate_grid()
        self.current_index = 0

    def _generate_grid(self) -> list[dict[str, Any]]:
        """Generate grid points for search."""
        # Simplified grid generation
        grid_points = []

        # For continuous parameters, create grid
        space_dict = self.hyperparameter_space.to_dict()

        # Create combinations (simplified - only handles a few parameters)
        ranks = np.linspace(
            space_dict["rank"][0], space_dict["rank"][1], self.grid_density, dtype=int
        )
        alphas = np.linspace(space_dict["alpha"][0], space_dict["alpha"][1], self.grid_density)
        lrs = np.logspace(
            np.log10(space_dict["learning_rate"][0]),
            np.log10(space_dict["learning_rate"][1]),
            self.grid_density,
        )

        for rank in ranks:
            for alpha in alphas:
                for lr in lrs:
                    config = {
                        "rank": int(rank),
                        "alpha": float(alpha),
                        "learning_rate": float(lr),
                        "dropout": 0.1,  # Fixed for grid search
                        "batch_size": random.choice(space_dict["batch_size"]),
                        "warmup_steps": random.randint(*space_dict["warmup_steps"]),
                        "weight_decay": random.uniform(*space_dict["weight_decay"]),
                        "optimizer": random.choice(space_dict["optimizer"]),
                        "scheduler": random.choice(space_dict["scheduler"]),
                    }
                    grid_points.append(config)

        return grid_points

    def suggest(self, trial_history: list[Any]) -> dict[str, Any]:
        """Suggest next point from grid."""
        if self.current_index >= len(self.grid_points):
            # Fallback to random if grid is exhausted
            return self.hyperparameter_space.sample()

        config = self.grid_points[self.current_index]
        self.current_index += 1
        return config


class HalvingRandomSearch(SearchStrategy):
    """Successive halving with random search."""

    def __init__(self, hyperparameter_space: Any, reduction_factor: int = 2):
        super().__init__(hyperparameter_space)
        self.reduction_factor = reduction_factor

    def suggest(self, trial_history: list[Any]) -> dict[str, Any]:
        """Suggest hyperparameters using successive halving logic."""
        # Simplified implementation - in practice would track cohorts
        return self.hyperparameter_space.sample()
