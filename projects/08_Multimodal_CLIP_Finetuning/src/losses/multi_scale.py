#!/usr/bin/env python3
"""
Multi-scale contrastive learning for hierarchical feature matching.

This module implements multi-scale contrastive loss that learns representations
at multiple temperature scales, enabling the model to capture both coarse and
fine-grained similarities.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleLoss(nn.Module):
    """Multi-scale contrastive learning loss.

    This loss computes contrastive learning at multiple temperature scales,
    allowing the model to learn both:
    - Coarse-grained features (higher temperature, softer distribution)
    - Fine-grained features (lower temperature, sharper distribution)

    The multi-scale approach helps:
    - Capture hierarchical relationships
    - Improve robustness to variations
    - Better generalization across domains

    Attributes:
        scales: List of temperature scales to use
        base_temperature: Base temperature value
        scale_weights: Weights for aggregating losses at different scales
    """

    def __init__(
        self,
        scales: list[float] | None = None,
        base_temperature: float = 0.07,
        scale_weights: list[float] | None = None,
        normalize_weights: bool = True,
    ) -> None:
        """Initialize multi-scale loss.

        Args:
            scales: List of temperature scale multipliers (e.g., [1.0, 0.75, 0.5])
                Lower scales = sharper distributions (fine-grained)
                Higher scales = softer distributions (coarse-grained)
            base_temperature: Base temperature before scaling
            scale_weights: Weights for each scale (must match len(scales))
                If None, uses equal weighting
            normalize_weights: Whether to normalize weights to sum to 1

        Raises:
            ValueError: If parameters are invalid
        """
        super().__init__()

        if scales is None:
            scales = [1.0, 0.75, 0.5]

        if not scales:
            raise ValueError("Must provide at least one scale")

        if any(s <= 0 for s in scales):
            raise ValueError(f"All scales must be positive, got {scales}")

        if base_temperature <= 0:
            raise ValueError(f"Base temperature must be positive, got {base_temperature}")

        self.scales = scales
        self.num_scales = len(scales)
        self.register_buffer(
            "log_base_temperature", torch.log(torch.tensor(base_temperature))
        )

        # Initialize scale weights
        if scale_weights is None:
            scale_weights = [1.0] * self.num_scales
        else:
            if len(scale_weights) != self.num_scales:
                raise ValueError(
                    f"Number of weights ({len(scale_weights)}) must match "
                    f"number of scales ({self.num_scales})"
                )
            if any(w < 0 for w in scale_weights):
                raise ValueError(f"All weights must be non-negative, got {scale_weights}")

        # Normalize weights if requested
        if normalize_weights:
            total = sum(scale_weights)
            if total == 0:
                raise ValueError("Sum of weights cannot be zero")
            scale_weights = [w / total for w in scale_weights]

        self.register_buffer(
            "scale_weights", torch.tensor(scale_weights, dtype=torch.float32)
        )

    @property
    def base_temperature(self) -> torch.Tensor:
        """Get base temperature value."""
        return torch.exp(self.log_base_temperature)

    def _compute_scale_loss(
        self,
        image_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
        temperature: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute contrastive loss at a single scale.

        Args:
            image_embeds: Normalized image embeddings [batch_size, embed_dim]
            text_embeds: Normalized text embeddings [batch_size, embed_dim]
            temperature: Temperature for this scale

        Returns:
            Tuple of (image_to_text_loss, text_to_image_loss)
        """
        batch_size = image_embeds.size(0)

        # Compute similarity and scale by temperature
        logits_i2t = (image_embeds @ text_embeds.T) / temperature
        logits_t2i = (text_embeds @ image_embeds.T) / temperature

        # Create labels
        labels = torch.arange(batch_size, device=image_embeds.device)

        # Compute losses
        i2t_loss = F.cross_entropy(logits_i2t, labels)
        t2i_loss = F.cross_entropy(logits_t2i, labels)

        return i2t_loss, t2i_loss

    def forward(
        self,
        image_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute multi-scale contrastive loss.

        Args:
            image_embeds: Image embeddings [batch_size, embed_dim]
            text_embeds: Text embeddings [batch_size, embed_dim]

        Returns:
            Dictionary containing:
                - loss: Weighted average of losses across all scales
                - scale_losses: Individual losses at each scale [num_scales]
                - scale_i2t_losses: Image-to-text losses at each scale [num_scales]
                - scale_t2i_losses: Text-to-image losses at each scale [num_scales]
                - base_temperature: Base temperature value
                - temperatures: Temperatures used at each scale [num_scales]
                - scale_weights: Weights for each scale [num_scales]

        Raises:
            ValueError: If embeddings have invalid shapes
        """
        # Validate inputs
        if image_embeds.dim() != 2 or text_embeds.dim() != 2:
            raise ValueError(
                f"Expected 2D embeddings, got image: {image_embeds.shape}, "
                f"text: {text_embeds.shape}"
            )

        if image_embeds.size(0) != text_embeds.size(0):
            raise ValueError(
                f"Batch size mismatch: image {image_embeds.size(0)} vs "
                f"text {text_embeds.size(0)}"
            )

        if image_embeds.size(1) != text_embeds.size(1):
            raise ValueError(
                f"Embedding dimension mismatch: image {image_embeds.size(1)} vs "
                f"text {text_embeds.size(1)}"
            )

        # Normalize embeddings once
        image_embeds = F.normalize(image_embeds, p=2, dim=1)
        text_embeds = F.normalize(text_embeds, p=2, dim=1)

        # Compute losses at each scale
        scale_losses = []
        scale_i2t_losses = []
        scale_t2i_losses = []
        temperatures = []

        base_temp = self.base_temperature.item()

        for scale in self.scales:
            temperature = base_temp * scale
            temperatures.append(temperature)

            i2t_loss, t2i_loss = self._compute_scale_loss(
                image_embeds, text_embeds, temperature
            )

            scale_loss = (i2t_loss + t2i_loss) / 2.0
            scale_losses.append(scale_loss)
            scale_i2t_losses.append(i2t_loss)
            scale_t2i_losses.append(t2i_loss)

        # Stack losses for easier processing
        scale_losses_tensor = torch.stack(scale_losses)
        scale_i2t_tensor = torch.stack(scale_i2t_losses)
        scale_t2i_tensor = torch.stack(scale_t2i_losses)

        # Compute weighted average loss
        total_loss = (scale_losses_tensor * self.scale_weights).sum()

        return {
            "loss": total_loss,
            "scale_losses": scale_losses_tensor.detach(),
            "scale_i2t_losses": scale_i2t_tensor.detach(),
            "scale_t2i_losses": scale_t2i_tensor.detach(),
            "base_temperature": self.base_temperature.detach(),
            "temperatures": torch.tensor(temperatures, device=image_embeds.device),
            "scale_weights": self.scale_weights,
        }

    def extra_repr(self) -> str:
        """String representation for debugging."""
        scales_str = ", ".join(f"{s:.2f}" for s in self.scales)
        weights_str = ", ".join(f"{w:.2f}" for w in self.scale_weights.tolist())
        return (
            f"base_temperature={self.base_temperature.item():.4f}, "
            f"scales=[{scales_str}], "
            f"weights=[{weights_str}]"
        )
