"""
MLX-optimized optimizers and schedulers for LoRA training.

Provides Apple Silicon optimized optimization components with custom
learning rate schedulers and gradient processing.
"""

import mlx.core as mx
import mlx.optimizers as optim
from typing import Any, Dict, Iterator, Optional, Union
import math


def create_optimizer(
    parameters: Iterator[mx.array],
    optimizer_name: str = "adamw",
    learning_rate: float = 2e-4,
    weight_decay: float = 0.01,
    **kwargs
) -> optim.Optimizer:
    """
    Create MLX optimizer for LoRA training.
    
    Args:
        parameters: Iterator of model parameters
        optimizer_name: Name of optimizer ("adamw", "sgd", "adam")
        learning_rate: Learning rate
        weight_decay: Weight decay coefficient
        **kwargs: Additional optimizer arguments
    
    Returns:
        MLX optimizer instance
    """
    # Convert parameters to list for MLX optimizers
    param_list = list(parameters)
    
    if optimizer_name.lower() == "adamw":
        return optim.AdamW(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            betas=kwargs.get("betas", (0.9, 0.999)),
            eps=kwargs.get("eps", 1e-8),
        )
    
    elif optimizer_name.lower() == "adam":
        return optim.Adam(
            learning_rate=learning_rate,
            betas=kwargs.get("betas", (0.9, 0.999)),
            eps=kwargs.get("eps", 1e-8),
        )
    
    elif optimizer_name.lower() == "sgd":
        return optim.SGD(
            learning_rate=learning_rate,
            momentum=kwargs.get("momentum", 0.9),
            weight_decay=weight_decay,
        )
    
    elif optimizer_name.lower() == "rmsprop":
        return optim.RMSprop(
            learning_rate=learning_rate,
            alpha=kwargs.get("alpha", 0.99),
            eps=kwargs.get("eps", 1e-8),
            weight_decay=weight_decay,
        )
    
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")


class LinearScheduler:
    """Linear learning rate scheduler with warmup."""
    
    def __init__(
        self,
        optimizer: optim.Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        last_epoch: int = -1,
    ):
        self.optimizer = optimizer
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.last_epoch = last_epoch
        self.base_lr = optimizer.learning_rate
        
    def get_lr(self) -> float:
        """Get current learning rate."""
        if self.last_epoch < self.num_warmup_steps:
            # Warmup phase: linear increase
            return self.base_lr * (self.last_epoch + 1) / self.num_warmup_steps
        else:
            # Decay phase: linear decrease
            progress = (self.last_epoch - self.num_warmup_steps) / (
                self.num_training_steps - self.num_warmup_steps
            )
            return self.base_lr * (1.0 - progress)
    
    def step(self) -> None:
        """Update learning rate."""
        self.last_epoch += 1
        new_lr = self.get_lr()
        self.optimizer.learning_rate = new_lr
    
    def get_last_lr(self) -> list[float]:
        """Get last learning rate (compatibility with PyTorch)."""
        return [self.get_lr()]


class CosineAnnealingScheduler:
    """Cosine annealing learning rate scheduler with warmup."""
    
    def __init__(
        self,
        optimizer: optim.Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ):
        self.optimizer = optimizer
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.eta_min = eta_min
        self.last_epoch = last_epoch
        self.base_lr = optimizer.learning_rate
    
    def get_lr(self) -> float:
        """Get current learning rate."""
        if self.last_epoch < self.num_warmup_steps:
            # Warmup phase: linear increase
            return self.base_lr * (self.last_epoch + 1) / self.num_warmup_steps
        else:
            # Cosine annealing phase
            progress = (self.last_epoch - self.num_warmup_steps) / (
                self.num_training_steps - self.num_warmup_steps
            )
            return self.eta_min + (self.base_lr - self.eta_min) * (
                1 + math.cos(math.pi * progress)
            ) / 2
    
    def step(self) -> None:
        """Update learning rate."""
        self.last_epoch += 1
        new_lr = self.get_lr()
        self.optimizer.learning_rate = new_lr
    
    def get_last_lr(self) -> list[float]:
        """Get last learning rate (compatibility with PyTorch)."""
        return [self.get_lr()]


class PolynomialScheduler:
    """Polynomial learning rate scheduler with warmup."""
    
    def __init__(
        self,
        optimizer: optim.Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        power: float = 1.0,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ):
        self.optimizer = optimizer
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.power = power
        self.eta_min = eta_min
        self.last_epoch = last_epoch
        self.base_lr = optimizer.learning_rate
    
    def get_lr(self) -> float:
        """Get current learning rate."""
        if self.last_epoch < self.num_warmup_steps:
            # Warmup phase: linear increase
            return self.base_lr * (self.last_epoch + 1) / self.num_warmup_steps
        else:
            # Polynomial decay phase
            progress = (self.last_epoch - self.num_warmup_steps) / (
                self.num_training_steps - self.num_warmup_steps
            )
            return self.eta_min + (self.base_lr - self.eta_min) * (
                1.0 - progress
            ) ** self.power
    
    def step(self) -> None:
        """Update learning rate."""
        self.last_epoch += 1
        new_lr = self.get_lr()
        self.optimizer.learning_rate = new_lr
    
    def get_last_lr(self) -> list[float]:
        """Get last learning rate (compatibility with PyTorch)."""
        return [self.get_lr()]


class ExponentialScheduler:
    """Exponential learning rate scheduler with warmup."""
    
    def __init__(
        self,
        optimizer: optim.Optimizer,
        num_warmup_steps: int,
        gamma: float = 0.95,
        last_epoch: int = -1,
    ):
        self.optimizer = optimizer
        self.num_warmup_steps = num_warmup_steps
        self.gamma = gamma
        self.last_epoch = last_epoch
        self.base_lr = optimizer.learning_rate
    
    def get_lr(self) -> float:
        """Get current learning rate."""
        if self.last_epoch < self.num_warmup_steps:
            # Warmup phase: linear increase
            return self.base_lr * (self.last_epoch + 1) / self.num_warmup_steps
        else:
            # Exponential decay phase
            decay_steps = self.last_epoch - self.num_warmup_steps
            return self.base_lr * (self.gamma ** decay_steps)
    
    def step(self) -> None:
        """Update learning rate."""
        self.last_epoch += 1
        new_lr = self.get_lr()
        self.optimizer.learning_rate = new_lr
    
    def get_last_lr(self) -> list[float]:
        """Get last learning rate (compatibility with PyTorch)."""
        return [self.get_lr()]


def create_scheduler(
    optimizer: optim.Optimizer,
    scheduler_name: str = "linear",
    num_warmup_steps: int = 100,
    num_training_steps: int = 1000,
    **kwargs
) -> Any:
    """
    Create learning rate scheduler for LoRA training.
    
    Args:
        optimizer: MLX optimizer
        scheduler_name: Name of scheduler ("linear", "cosine", "polynomial", "exponential")
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
        **kwargs: Additional scheduler arguments
    
    Returns:
        Learning rate scheduler instance
    """
    if scheduler_name.lower() == "linear":
        return LinearScheduler(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
    
    elif scheduler_name.lower() == "cosine":
        return CosineAnnealingScheduler(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            eta_min=kwargs.get("eta_min", 0.0),
        )
    
    elif scheduler_name.lower() == "polynomial":
        return PolynomialScheduler(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            power=kwargs.get("power", 1.0),
            eta_min=kwargs.get("eta_min", 0.0),
        )
    
    elif scheduler_name.lower() == "exponential":
        return ExponentialScheduler(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            gamma=kwargs.get("gamma", 0.95),
        )
    
    elif scheduler_name.lower() == "none":
        return None
    
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")


class GradientClipping:
    """Gradient clipping utilities for MLX."""
    
    @staticmethod
    def clip_by_norm(
        gradients: Dict[str, mx.array], 
        max_norm: float
    ) -> Dict[str, mx.array]:
        """
        Clip gradients by global norm.
        
        Args:
            gradients: Dictionary of gradients
            max_norm: Maximum allowed gradient norm
            
        Returns:
            Clipped gradients
        """
        # Compute global norm
        total_norm = 0.0
        for grad in gradients.values():
            if grad is not None:
                total_norm += mx.sum(grad ** 2)
        
        total_norm = mx.sqrt(total_norm)
        
        # Compute clipping coefficient
        clip_coeff = max_norm / (total_norm + 1e-6)
        clip_coeff = mx.minimum(clip_coeff, 1.0)
        
        # Apply clipping
        clipped_gradients = {}
        for name, grad in gradients.items():
            if grad is not None:
                clipped_gradients[name] = grad * clip_coeff
            else:
                clipped_gradients[name] = grad
        
        return clipped_gradients
    
    @staticmethod
    def clip_by_value(
        gradients: Dict[str, mx.array], 
        clip_value: float
    ) -> Dict[str, mx.array]:
        """
        Clip gradients by absolute value.
        
        Args:
            gradients: Dictionary of gradients
            clip_value: Maximum absolute value for gradients
            
        Returns:
            Clipped gradients
        """
        clipped_gradients = {}
        for name, grad in gradients.items():
            if grad is not None:
                clipped_gradients[name] = mx.clip(grad, -clip_value, clip_value)
            else:
                clipped_gradients[name] = grad
        
        return clipped_gradients


class AdaptiveLossScaling:
    """Adaptive loss scaling for mixed precision training."""
    
    def __init__(
        self, 
        init_scale: float = 2**15,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
    ):
        self.scale = init_scale
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self.growth_interval = growth_interval
        self.growth_tracker = 0
    
    def update_scale(self, found_inf: bool) -> float:
        """Update loss scale based on gradient overflow detection."""
        if found_inf:
            # Reduce scale on overflow
            self.scale *= self.backoff_factor
            self.growth_tracker = 0
        else:
            # Increase scale periodically if no overflow
            self.growth_tracker += 1
            if self.growth_tracker >= self.growth_interval:
                self.scale *= self.growth_factor
                self.growth_tracker = 0
        
        # Ensure scale stays within reasonable bounds
        self.scale = mx.clip(self.scale, 1.0, 2**24)
        return self.scale
    
    def scale_loss(self, loss: mx.array) -> mx.array:
        """Scale loss for backward pass."""
        return loss * self.scale
    
    def unscale_gradients(self, gradients: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """Unscale gradients after backward pass."""
        unscaled_gradients = {}
        for name, grad in gradients.items():
            if grad is not None:
                unscaled_gradients[name] = grad / self.scale
            else:
                unscaled_gradients[name] = grad
        
        return unscaled_gradients
    
    def check_overflow(self, gradients: Dict[str, mx.array]) -> bool:
        """Check if gradients contain infinite or NaN values."""
        for grad in gradients.values():
            if grad is not None:
                if mx.any(mx.isinf(grad)) or mx.any(mx.isnan(grad)):
                    return True
        return False