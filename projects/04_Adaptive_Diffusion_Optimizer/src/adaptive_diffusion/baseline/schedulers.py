"""
Standard Diffusion Schedulers

Implements DDPM, DDIM, and DPM-Solver schedulers with MLX optimization.
Based on:
- DDPM: Ho et al. (2020) - Denoising Diffusion Probabilistic Models
- DDIM: Song et al. (2021) - Denoising Diffusion Implicit Models
- DPM-Solver: Lu et al. (2022) - Fast ODE Solver for Diffusion Models
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import mlx.core as mx
import numpy as np


class BaseScheduler(ABC):
    """Base class for diffusion schedulers."""

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
    ):
        """
        Initialize base scheduler.

        Args:
            num_train_timesteps: Number of diffusion steps for training
            beta_start: Starting value of beta schedule
            beta_end: Ending value of beta schedule
            beta_schedule: Type of beta schedule ('linear', 'scaled_linear', 'squaredcos_cap_v2')
        """
        self.num_train_timesteps = num_train_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_schedule = beta_schedule

        # Initialize beta schedule
        self.betas = self._get_beta_schedule()
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = mx.cumprod(self.alphas)

        # Precompute useful values
        self.sqrt_alphas_cumprod = mx.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = mx.sqrt(1.0 - self.alphas_cumprod)

    def _get_beta_schedule(self) -> mx.array:
        """Get beta schedule based on configuration."""
        if self.beta_schedule == "linear":
            return mx.linspace(
                self.beta_start, self.beta_end, self.num_train_timesteps
            )
        elif self.beta_schedule == "scaled_linear":
            # Used in Stable Diffusion
            return mx.linspace(
                self.beta_start**0.5, self.beta_end**0.5, self.num_train_timesteps
            ) ** 2
        elif self.beta_schedule == "squaredcos_cap_v2":
            # Improved cosine schedule
            return self._betas_for_alpha_bar(
                lambda t: np.cos((t + 0.008) / 1.008 * np.pi / 2) ** 2
            )
        else:
            raise ValueError(f"Unknown beta schedule: {self.beta_schedule}")

    def _betas_for_alpha_bar(self, alpha_bar_fn) -> mx.array:
        """Helper for cosine schedule."""
        betas = []
        for i in range(self.num_train_timesteps):
            t1 = i / self.num_train_timesteps
            t2 = (i + 1) / self.num_train_timesteps
            betas.append(min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), 0.999))
        return mx.array(betas)

    @abstractmethod
    def add_noise(
        self, original_samples: mx.array, noise: mx.array, timesteps: mx.array
    ) -> mx.array:
        """Add noise to samples at given timesteps."""
        pass

    @abstractmethod
    def step(
        self, model_output: mx.array, timestep: int, sample: mx.array
    ) -> mx.array:
        """Perform one denoising step."""
        pass


class DDPMScheduler(BaseScheduler):
    """
    Denoising Diffusion Probabilistic Models (DDPM) scheduler.

    Based on: Ho et al. (2020) - https://arxiv.org/abs/2006.11239
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        variance_type: str = "fixed_small",
    ):
        """
        Initialize DDPM scheduler.

        Args:
            num_train_timesteps: Number of diffusion steps
            beta_start: Starting beta value
            beta_end: Ending beta value
            beta_schedule: Beta schedule type
            variance_type: Variance calculation type ('fixed_small', 'fixed_large', 'learned')
        """
        super().__init__(num_train_timesteps, beta_start, beta_end, beta_schedule)
        self.variance_type = variance_type

        # Precompute variance
        self.variance = self._get_variance()

    def _get_variance(self) -> mx.array:
        """Calculate variance for each timestep."""
        # Variance is beta_t for fixed_small
        if self.variance_type == "fixed_small":
            variance = self.betas
        elif self.variance_type == "fixed_large":
            variance = mx.clip(self.betas, 1e-20, None)
        else:
            raise ValueError(f"Unknown variance type: {self.variance_type}")

        # Ensure variance is positive
        variance = mx.clip(variance, 1e-20, None)
        return variance

    def add_noise(
        self, original_samples: mx.array, noise: mx.array, timesteps: mx.array
    ) -> mx.array:
        """
        Add noise to samples using forward diffusion process.

        q(x_t | x_0) = N(sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)
        """
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps]
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps]

        # Reshape for broadcasting
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = mx.expand_dims(sqrt_alpha_prod, axis=-1)
            sqrt_one_minus_alpha_prod = mx.expand_dims(
                sqrt_one_minus_alpha_prod, axis=-1
            )

        noisy_samples = (
            sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        )
        return noisy_samples

    def step(
        self, model_output: mx.array, timestep: int, sample: mx.array
    ) -> mx.array:
        """
        Perform one reverse diffusion step.

        Args:
            model_output: Predicted noise from model
            timestep: Current timestep
            sample: Current noisy sample

        Returns:
            Denoised sample at t-1
        """
        t = timestep
        prev_t = t - 1 if t > 0 else 0

        # Get precomputed values
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if t > 0 else mx.array(1.0)
        beta_prod_t = 1 - alpha_prod_t

        # Predict x_0 from noise
        pred_original_sample = (
            sample - mx.sqrt(beta_prod_t) * model_output
        ) / mx.sqrt(alpha_prod_t)

        # Compute coefficients
        pred_sample_direction = mx.sqrt(1 - alpha_prod_t_prev) * model_output

        # Compute previous sample mean
        prev_sample = (
            mx.sqrt(alpha_prod_t_prev) * pred_original_sample + pred_sample_direction
        )

        # Add noise (except for final step)
        if t > 0:
            variance = mx.sqrt(self.variance[t])
            noise = mx.random.normal(sample.shape)
            prev_sample = prev_sample + variance * noise

        return prev_sample


class DDIMScheduler(BaseScheduler):
    """
    Denoising Diffusion Implicit Models (DDIM) scheduler.

    Based on: Song et al. (2021) - https://arxiv.org/abs/2010.02502
    Supports fast sampling with fewer steps (10-50 vs 1000).
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        eta: float = 0.0,
        num_inference_steps: int = 50,
    ):
        """
        Initialize DDIM scheduler.

        Args:
            num_train_timesteps: Number of training timesteps
            beta_start: Starting beta value
            beta_end: Ending beta value
            beta_schedule: Beta schedule type
            eta: Stochasticity parameter (0.0 = deterministic, 1.0 = DDPM)
            num_inference_steps: Number of steps for inference
        """
        super().__init__(num_train_timesteps, beta_start, beta_end, beta_schedule)
        self.eta = eta
        self.num_inference_steps = num_inference_steps

        # Set inference timesteps
        self.set_timesteps(num_inference_steps)

    def set_timesteps(self, num_inference_steps: int):
        """Set timesteps for inference (subset of training timesteps)."""
        self.num_inference_steps = num_inference_steps

        # Create evenly spaced timesteps
        step_ratio = self.num_train_timesteps // num_inference_steps
        timesteps = (
            np.arange(0, num_inference_steps) * step_ratio
        ).round()[::-1].copy().astype(np.int64)
        self.timesteps = mx.array(timesteps)

    def add_noise(
        self, original_samples: mx.array, noise: mx.array, timesteps: mx.array
    ) -> mx.array:
        """Add noise using forward process (same as DDPM)."""
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps]
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps]

        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = mx.expand_dims(sqrt_alpha_prod, axis=-1)
            sqrt_one_minus_alpha_prod = mx.expand_dims(
                sqrt_one_minus_alpha_prod, axis=-1
            )

        noisy_samples = (
            sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        )
        return noisy_samples

    def step(
        self, model_output: mx.array, timestep: int, sample: mx.array
    ) -> mx.array:
        """
        Perform one DDIM reverse diffusion step.

        Args:
            model_output: Predicted noise from model
            timestep: Current timestep index
            sample: Current noisy sample

        Returns:
            Denoised sample at previous timestep
        """
        # Get current and previous alpha values
        t = self.timesteps[timestep]
        prev_timestep = (
            self.timesteps[timestep + 1] if timestep < len(self.timesteps) - 1 else 0
        )

        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = (
            self.alphas_cumprod[prev_timestep]
            if prev_timestep >= 0
            else mx.array(1.0)
        )

        beta_prod_t = 1 - alpha_prod_t

        # Predict x_0
        pred_original_sample = (
            sample - mx.sqrt(beta_prod_t) * model_output
        ) / mx.sqrt(alpha_prod_t)

        # Compute variance
        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * (
            1 - alpha_prod_t / alpha_prod_t_prev
        )
        std_dev_t = self.eta * mx.sqrt(variance)

        # Compute predicted sample direction
        pred_sample_direction = mx.sqrt(1 - alpha_prod_t_prev - std_dev_t**2) * model_output

        # Compute previous sample
        prev_sample = (
            mx.sqrt(alpha_prod_t_prev) * pred_original_sample + pred_sample_direction
        )

        # Add noise if eta > 0
        if self.eta > 0 and timestep < len(self.timesteps) - 1:
            noise = mx.random.normal(sample.shape)
            prev_sample = prev_sample + std_dev_t * noise

        return prev_sample


class DPMSolverScheduler(BaseScheduler):
    """
    DPM-Solver scheduler for fast diffusion sampling.

    Based on: Lu et al. (2022) - https://arxiv.org/abs/2206.00927
    Achieves 10-20 step sampling with quality comparable to 200-step DDIM.
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        solver_order: int = 2,
        num_inference_steps: int = 20,
    ):
        """
        Initialize DPM-Solver scheduler.

        Args:
            num_train_timesteps: Number of training timesteps
            beta_start: Starting beta value
            beta_end: Ending beta value
            beta_schedule: Beta schedule type
            solver_order: Order of DPM-Solver (1, 2, or 3)
            num_inference_steps: Number of inference steps
        """
        super().__init__(num_train_timesteps, beta_start, beta_end, beta_schedule)
        self.solver_order = solver_order
        self.num_inference_steps = num_inference_steps

        # Initialize lambda schedule for DPM-Solver
        self.lambdas = 0.5 * mx.log(self.alphas_cumprod / (1 - self.alphas_cumprod))

        # Set inference timesteps
        self.set_timesteps(num_inference_steps)

        # Store previous model outputs for higher-order solvers
        self.model_outputs = []

    def set_timesteps(self, num_inference_steps: int):
        """Set timesteps for inference."""
        self.num_inference_steps = num_inference_steps

        # Create timesteps using exponential spacing (better for DPM-Solver)
        timesteps = (
            np.linspace(0, self.num_train_timesteps - 1, num_inference_steps)
            .round()[::-1]
            .copy()
            .astype(np.int64)
        )
        self.timesteps = mx.array(timesteps)
        self.model_outputs = []

    def add_noise(
        self, original_samples: mx.array, noise: mx.array, timesteps: mx.array
    ) -> mx.array:
        """Add noise using forward process."""
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps]
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps]

        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = mx.expand_dims(sqrt_alpha_prod, axis=-1)
            sqrt_one_minus_alpha_prod = mx.expand_dims(
                sqrt_one_minus_alpha_prod, axis=-1
            )

        noisy_samples = (
            sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        )
        return noisy_samples

    def step(
        self, model_output: mx.array, timestep: int, sample: mx.array
    ) -> mx.array:
        """
        Perform one DPM-Solver step.

        Args:
            model_output: Predicted noise from model
            timestep: Current timestep index
            sample: Current noisy sample

        Returns:
            Denoised sample at previous timestep
        """
        t = self.timesteps[timestep]
        prev_timestep = (
            self.timesteps[timestep + 1] if timestep < len(self.timesteps) - 1 else 0
        )

        # Store model output for higher-order solvers
        self.model_outputs.append(model_output)
        if len(self.model_outputs) > self.solver_order:
            self.model_outputs.pop(0)

        # Get lambda values
        lambda_t = self.lambdas[t]
        lambda_s = self.lambdas[prev_timestep]

        # Get alpha values
        alpha_t = self.alphas_cumprod[t]
        alpha_s = self.alphas_cumprod[prev_timestep]
        sigma_t = mx.sqrt(1 - alpha_t)
        sigma_s = mx.sqrt(1 - alpha_s)

        # Compute h (timestep difference in log-SNR space)
        h = lambda_s - lambda_t

        # Convert noise prediction to data prediction
        x0_pred = (sample - sigma_t * model_output) / mx.sqrt(alpha_t)

        if self.solver_order == 1 or len(self.model_outputs) < 2:
            # First-order solver (linear)
            prev_sample = (
                mx.sqrt(alpha_s / alpha_t) * sample
                - sigma_s * mx.expm1(h) * model_output
            )
        elif self.solver_order == 2:
            # Second-order solver (improved)
            prev_model_output = self.model_outputs[-2]
            prev_sample = (
                mx.sqrt(alpha_s / alpha_t) * sample
                - sigma_s * mx.expm1(h) * model_output
                - sigma_s
                * (mx.expm1(h) / h - 1.0)
                * (model_output - prev_model_output)
            )
        else:
            # Third-order solver
            if len(self.model_outputs) >= 3:
                prev_model_output = self.model_outputs[-2]
                prev_prev_model_output = self.model_outputs[-3]

                prev_sample = (
                    mx.sqrt(alpha_s / alpha_t) * sample
                    - sigma_s * mx.expm1(h) * model_output
                    - sigma_s
                    * (mx.expm1(h) / h - 1.0)
                    * (model_output - prev_model_output)
                    - sigma_s
                    * (mx.expm1(h) / h - 1.0 - 0.5 / h)
                    * (
                        model_output
                        - 2 * prev_model_output
                        + prev_prev_model_output
                    )
                )
            else:
                # Fallback to second-order
                prev_model_output = self.model_outputs[-2]
                prev_sample = (
                    mx.sqrt(alpha_s / alpha_t) * sample
                    - sigma_s * mx.expm1(h) * model_output
                    - sigma_s
                    * (mx.expm1(h) / h - 1.0)
                    * (model_output - prev_model_output)
                )

        return prev_sample


def get_scheduler(scheduler_name: str, **kwargs) -> BaseScheduler:
    """
    Factory function to create scheduler by name.

    Args:
        scheduler_name: Name of scheduler ('ddpm', 'ddim', 'dpm-solver')
        **kwargs: Additional arguments for scheduler

    Returns:
        Initialized scheduler instance
    """
    schedulers = {
        "ddpm": DDPMScheduler,
        "ddim": DDIMScheduler,
        "dpm-solver": DPMSolverScheduler,
    }

    if scheduler_name.lower() not in schedulers:
        raise ValueError(
            f"Unknown scheduler: {scheduler_name}. "
            f"Available: {list(schedulers.keys())}"
        )

    return schedulers[scheduler_name.lower()](**kwargs)
