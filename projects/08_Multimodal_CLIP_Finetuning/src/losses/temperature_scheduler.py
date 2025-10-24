#!/usr/bin/env python3
"""
Adaptive temperature scheduling for contrastive learning.

This module implements temperature scheduling strategies to dynamically adjust
the temperature parameter during training for optimal convergence.
"""

from __future__ import annotations

import math


class TemperatureScheduler:
    """Adaptive temperature scheduling for contrastive learning.

    This scheduler adjusts the temperature parameter during training using
    various strategies:
    - Warmup: Gradually decrease temperature from high to target value
    - Constant: Maintain constant temperature
    - Cosine: Cosine annealing schedule
    - Exponential: Exponential decay
    - Adaptive: Adjust based on loss trajectory

    Temperature scheduling helps:
    - Stabilize early training (higher temp = softer targets)
    - Refine features in later stages (lower temp = sharper targets)
    - Adapt to training dynamics

    Attributes:
        initial_temp: Initial temperature value
        min_temp: Minimum temperature bound
        max_temp: Maximum temperature bound
        warmup_steps: Number of warmup steps
        total_steps: Total training steps (for schedule types)
        schedule_type: Type of scheduling strategy
    """

    VALID_SCHEDULES = {"constant", "warmup", "cosine", "exponential", "adaptive"}

    def __init__(
        self,
        initial_temp: float = 0.07,
        min_temp: float = 0.01,
        max_temp: float = 0.1,
        warmup_steps: int = 1000,
        total_steps: int | None = None,
        schedule_type: str = "warmup",
        decay_rate: float = 0.95,
    ) -> None:
        """Initialize temperature scheduler.

        Args:
            initial_temp: Initial temperature value
            min_temp: Minimum temperature bound
            max_temp: Maximum temperature bound
            warmup_steps: Number of warmup steps
            total_steps: Total training steps (required for cosine/exponential)
            schedule_type: Scheduling strategy (constant, warmup, cosine, exponential, adaptive)
            decay_rate: Decay rate for exponential schedule (0 < rate < 1)

        Raises:
            ValueError: If parameters are invalid
        """
        if initial_temp <= 0:
            raise ValueError(f"Initial temperature must be positive, got {initial_temp}")

        if min_temp <= 0:
            raise ValueError(f"Minimum temperature must be positive, got {min_temp}")

        if max_temp <= 0:
            raise ValueError(f"Maximum temperature must be positive, got {max_temp}")

        if min_temp > max_temp:
            raise ValueError(
                f"Minimum temperature ({min_temp}) cannot be greater than "
                f"maximum temperature ({max_temp})"
            )

        if not min_temp <= initial_temp <= max_temp:
            raise ValueError(
                f"Initial temperature ({initial_temp}) must be between "
                f"min ({min_temp}) and max ({max_temp})"
            )

        if warmup_steps < 0:
            raise ValueError(f"Warmup steps must be non-negative, got {warmup_steps}")

        if schedule_type not in self.VALID_SCHEDULES:
            raise ValueError(
                f"Schedule type must be one of {self.VALID_SCHEDULES}, "
                f"got {schedule_type}"
            )

        if schedule_type in {"cosine", "exponential"} and total_steps is None:
            raise ValueError(
                f"total_steps must be specified for {schedule_type} schedule"
            )

        if not 0 < decay_rate < 1:
            raise ValueError(f"Decay rate must be in (0, 1), got {decay_rate}")

        self.initial_temp = initial_temp
        self.min_temp = min_temp
        self.max_temp = max_temp
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.schedule_type = schedule_type
        self.decay_rate = decay_rate

        # Current state
        self.current_temp = initial_temp
        self.current_step = 0

        # Adaptive scheduling state
        self._loss_history: list[float] = []
        self._temp_history: list[float] = [initial_temp]
        self._adaptive_window = 100  # Window size for adaptive adjustment

    def step(self, step: int | None = None, loss: float | None = None) -> float:
        """Update temperature based on training progress.

        Args:
            step: Current training step (if None, uses internal counter)
            loss: Current loss value (used for adaptive scheduling)

        Returns:
            Updated temperature value

        Raises:
            ValueError: If required arguments are missing for schedule type
        """
        if step is None:
            step = self.current_step + 1

        self.current_step = step

        # Store loss for adaptive scheduling
        if loss is not None:
            self._loss_history.append(loss)

        # Compute temperature based on schedule type
        if self.schedule_type == "constant":
            temp = self._constant_schedule()
        elif self.schedule_type == "warmup":
            temp = self._warmup_schedule(step)
        elif self.schedule_type == "cosine":
            temp = self._cosine_schedule(step)
        elif self.schedule_type == "exponential":
            temp = self._exponential_schedule(step)
        elif self.schedule_type == "adaptive":
            temp = self._adaptive_schedule(step, loss)
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")

        # Clip to bounds
        temp = max(self.min_temp, min(self.max_temp, temp))

        self.current_temp = temp
        self._temp_history.append(temp)

        return temp

    def _constant_schedule(self) -> float:
        """Constant temperature (no scheduling)."""
        return self.initial_temp

    def _warmup_schedule(self, step: int) -> float:
        """Linear warmup from max_temp to initial_temp."""
        if step < self.warmup_steps:
            # Linear interpolation from max_temp to initial_temp
            alpha = step / self.warmup_steps
            temp = self.max_temp + alpha * (self.initial_temp - self.max_temp)
        else:
            # After warmup, use initial_temp
            temp = self.initial_temp

        return temp

    def _cosine_schedule(self, step: int) -> float:
        """Cosine annealing from initial_temp to min_temp."""
        if step < self.warmup_steps:
            # Warmup phase
            return self._warmup_schedule(step)

        # Cosine annealing after warmup
        progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        progress = min(1.0, progress)  # Clip to [0, 1]

        # Cosine annealing formula
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        temp = self.min_temp + (self.initial_temp - self.min_temp) * cosine_decay

        return temp

    def _exponential_schedule(self, step: int) -> float:
        """Exponential decay from initial_temp towards min_temp."""
        if step < self.warmup_steps:
            # Warmup phase
            return self._warmup_schedule(step)

        # Exponential decay after warmup
        steps_after_warmup = step - self.warmup_steps
        decay_factor = self.decay_rate ** steps_after_warmup
        temp = self.min_temp + (self.initial_temp - self.min_temp) * decay_factor

        return temp

    def _adaptive_schedule(self, step: int, loss: float | None) -> float:
        """Adaptive scheduling based on loss trajectory.

        Increases temperature if loss is increasing (to soften targets)
        Decreases temperature if loss is decreasing (to sharpen targets)
        """
        if step < self.warmup_steps:
            # Use warmup schedule initially (no loss required during warmup)
            return self._warmup_schedule(step)

        # After warmup, loss is required
        if loss is None:
            raise ValueError("Loss must be provided for adaptive scheduling after warmup")

        # Need sufficient history for adaptation
        if len(self._loss_history) < self._adaptive_window:
            return self.current_temp

        # Compute recent loss trend
        recent_losses = self._loss_history[-self._adaptive_window :]
        mid_point = len(recent_losses) // 2
        early_avg = sum(recent_losses[:mid_point]) / mid_point
        late_avg = sum(recent_losses[mid_point:]) / (len(recent_losses) - mid_point)

        # Determine adjustment
        if late_avg > early_avg:
            # Loss increasing -> increase temperature (soften targets)
            adjustment = 1.02
        else:
            # Loss decreasing -> decrease temperature (sharpen targets)
            adjustment = 0.98

        temp = self.current_temp * adjustment

        return temp

    def get_temperature(self) -> float:
        """Get current temperature value."""
        return self.current_temp

    def get_state(self) -> dict[str, object]:
        """Get scheduler state for checkpointing.

        Returns:
            Dictionary containing scheduler state
        """
        return {
            "current_temp": self.current_temp,
            "current_step": self.current_step,
            "loss_history": self._loss_history.copy(),
            "temp_history": self._temp_history.copy(),
        }

    def load_state(self, state: dict[str, object]) -> None:
        """Load scheduler state from checkpoint.

        Args:
            state: Dictionary containing scheduler state
        """
        self.current_temp = state["current_temp"]
        self.current_step = state["current_step"]
        self._loss_history = state["loss_history"].copy()
        self._temp_history = state["temp_history"].copy()

    def reset(self) -> None:
        """Reset scheduler to initial state."""
        self.current_temp = self.initial_temp
        self.current_step = 0
        self._loss_history = []
        self._temp_history = [self.initial_temp]

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"TemperatureScheduler("
            f"type={self.schedule_type}, "
            f"current_temp={self.current_temp:.4f}, "
            f"step={self.current_step}, "
            f"range=[{self.min_temp:.4f}, {self.max_temp:.4f}]"
            f")"
        )
