"""
Optimization objectives for hyperparameter tuning.

Provides various objective functions for evaluating hyperparameter
configurations including perplexity, BLEU, ROUGE, and multi-objective.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List
import numpy as np


class OptimizationObjective(ABC):
    """Base class for optimization objectives."""
    
    def __init__(self, direction: str = "minimize"):
        if direction not in ["minimize", "maximize"]:
            raise ValueError("Direction must be 'minimize' or 'maximize'")
        self.direction = direction
    
    @abstractmethod
    def calculate(self, training_result: Dict[str, Any], training_state: Any) -> float:
        """Calculate objective value from training results."""
        pass
    
    def is_better(self, value1: float, value2: float) -> bool:
        """Check if value1 is better than value2."""
        if self.direction == "minimize":
            return value1 < value2
        else:
            return value1 > value2


class PerplexityObjective(OptimizationObjective):
    """Perplexity-based objective (minimize)."""
    
    def __init__(self):
        super().__init__(direction="minimize")
    
    def calculate(self, training_result: Dict[str, Any], training_state: Any) -> float:
        """Calculate perplexity from training results."""
        # Use validation loss as proxy for perplexity
        eval_loss = training_state.eval_loss
        if eval_loss <= 0:
            return float('inf')
        
        # Convert loss to perplexity
        import math
        perplexity = math.exp(eval_loss)
        return perplexity


class BLEUObjective(OptimizationObjective):
    """BLEU score objective (maximize)."""
    
    def __init__(self):
        super().__init__(direction="maximize")
    
    def calculate(self, training_result: Dict[str, Any], training_state: Any) -> float:
        """Calculate BLEU score."""
        # Placeholder implementation - would need actual BLEU calculation
        # For now, use inverse of loss as proxy
        eval_loss = training_state.eval_loss
        if eval_loss <= 0:
            return 0.0
        
        # Convert loss to approximate BLEU score
        bleu_score = max(0, 1.0 - eval_loss)
        return bleu_score


class ROUGEObjective(OptimizationObjective):
    """ROUGE score objective (maximize)."""
    
    def __init__(self):
        super().__init__(direction="maximize")
    
    def calculate(self, training_result: Dict[str, Any], training_state: Any) -> float:
        """Calculate ROUGE score."""
        # Placeholder implementation - would need actual ROUGE calculation
        eval_loss = training_state.eval_loss
        if eval_loss <= 0:
            return 0.0
        
        # Convert loss to approximate ROUGE score
        rouge_score = max(0, 1.0 - eval_loss * 0.5)
        return rouge_score


class TrainingTimeObjective(OptimizationObjective):
    """Training time objective (minimize)."""
    
    def __init__(self):
        super().__init__(direction="minimize")
    
    def calculate(self, training_result: Dict[str, Any], training_state: Any) -> float:
        """Calculate training time in minutes."""
        training_time = training_result.get("total_training_time", 0.0)
        return training_time / 60.0  # Convert to minutes


class MemoryUsageObjective(OptimizationObjective):
    """Memory usage objective (minimize)."""
    
    def __init__(self):
        super().__init__(direction="minimize")
    
    def calculate(self, training_result: Dict[str, Any], training_state: Any) -> float:
        """Calculate peak memory usage in GB."""
        peak_memory_mb = training_state.peak_memory_mb
        return peak_memory_mb / 1024.0  # Convert to GB


class MultiObjective(OptimizationObjective):
    """Multi-objective optimization combining multiple objectives."""
    
    def __init__(self, objectives: List[OptimizationObjective], weights: List[float] = None):
        # Multi-objective can be either minimize or maximize depending on combination
        super().__init__(direction="minimize")  # Use minimize as default
        
        self.objectives = objectives
        if weights is None:
            weights = [1.0] * len(objectives)
        
        if len(weights) != len(objectives):
            raise ValueError("Number of weights must match number of objectives")
        
        self.weights = weights
    
    def calculate(self, training_result: Dict[str, Any], training_state: Any) -> float:
        """Calculate weighted combination of objectives."""
        total_score = 0.0
        
        for objective, weight in zip(self.objectives, self.weights):
            obj_value = objective.calculate(training_result, training_state)
            
            # Normalize and weight the objective
            if objective.direction == "maximize":
                # Convert maximize objectives to minimize (invert)
                obj_value = -obj_value
            
            total_score += weight * obj_value
        
        return total_score
    
    def is_better(self, value1: float, value2: float) -> bool:
        """Multi-objective always minimizes the weighted sum."""
        return value1 < value2


class ParetoDominanceObjective(OptimizationObjective):
    """Pareto dominance for true multi-objective optimization."""
    
    def __init__(self, objectives: List[OptimizationObjective]):
        super().__init__(direction="minimize")  # Use minimize as default
        self.objectives = objectives
        self.pareto_front: List[Dict[str, Any]] = []
    
    def calculate(self, training_result: Dict[str, Any], training_state: Any) -> float:
        """Calculate Pareto dominance score."""
        # Calculate all objective values
        obj_values = []
        for objective in self.objectives:
            value = objective.calculate(training_result, training_state)
            obj_values.append(value)
        
        # For now, return weighted sum (simplified)
        # In full implementation, would maintain Pareto front
        return sum(obj_values) / len(obj_values)
    
    def dominates(self, solution1: List[float], solution2: List[float]) -> bool:
        """Check if solution1 dominates solution2."""
        better_in_any = False
        
        for i, (obj, val1, val2) in enumerate(zip(self.objectives, solution1, solution2)):
            if obj.direction == "minimize":
                if val1 > val2:
                    return False  # solution1 is worse in this objective
                elif val1 < val2:
                    better_in_any = True
            else:  # maximize
                if val1 < val2:
                    return False  # solution1 is worse in this objective
                elif val1 > val2:
                    better_in_any = True
        
        return better_in_any


def create_objective(objective_name: str, **kwargs) -> OptimizationObjective:
    """Factory function to create objectives by name."""
    objectives = {
        "perplexity": PerplexityObjective,
        "bleu": BLEUObjective,
        "rouge": ROUGEObjective,
        "training_time": TrainingTimeObjective,
        "memory_usage": MemoryUsageObjective,
    }
    
    if objective_name not in objectives:
        raise ValueError(f"Unknown objective: {objective_name}. Available: {list(objectives.keys())}")
    
    return objectives[objective_name](**kwargs)