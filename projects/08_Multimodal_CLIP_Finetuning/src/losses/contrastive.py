#!/usr/bin/env python3
"""
Standard CLIP contrastive loss with temperature scaling.

This module implements the standard contrastive loss used in CLIP training,
which learns to align image and text embeddings by maximizing similarity
for matching pairs and minimizing similarity for non-matching pairs.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CLIPContrastiveLoss(nn.Module):
    """Standard CLIP contrastive loss with temperature scaling.

    This loss function implements the symmetric contrastive learning objective
    used in CLIP, computing both image-to-text and text-to-image losses.

    The loss encourages:
    - High similarity between matching image-text pairs
    - Low similarity between non-matching pairs
    - Temperature controls the sharpness of the softmax distribution

    Attributes:
        temperature: Temperature parameter for scaling logits (lower = sharper)
        log_temperature: Learnable log temperature parameter
    """

    def __init__(self, temperature: float = 0.07, learnable_temp: bool = False) -> None:
        """Initialize CLIP contrastive loss.

        Args:
            temperature: Initial temperature for scaling logits.
                Lower values make the distribution sharper (more confident).
                Typical range: 0.01 - 0.1
            learnable_temp: Whether to make temperature a learnable parameter
        """
        super().__init__()

        if temperature <= 0:
            raise ValueError(f"Temperature must be positive, got {temperature}")

        if learnable_temp:
            # Store as log for numerical stability during optimization
            self.log_temperature = nn.Parameter(torch.log(torch.tensor(temperature)))
        else:
            self.register_buffer("log_temperature", torch.log(torch.tensor(temperature)))

    @property
    def temperature(self) -> torch.Tensor:
        """Get current temperature value."""
        return torch.exp(self.log_temperature)

    def forward(
        self,
        image_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute contrastive loss.

        Args:
            image_embeds: Image embeddings [batch_size, embed_dim]
            text_embeds: Text embeddings [batch_size, embed_dim]

        Returns:
            Dictionary containing:
                - loss: Total contrastive loss
                - image_to_text_loss: Image-to-text contrastive loss
                - text_to_image_loss: Text-to-image contrastive loss
                - temperature: Current temperature value
                - logits_per_image: Image-to-text similarity logits
                - logits_per_text: Text-to-image similarity logits

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

        batch_size = image_embeds.size(0)

        # Normalize embeddings to unit vectors
        # This makes the dot product equivalent to cosine similarity
        image_embeds = F.normalize(image_embeds, p=2, dim=1)
        text_embeds = F.normalize(text_embeds, p=2, dim=1)

        # Compute similarity matrix: [batch_size, batch_size]
        # Each row represents one image's similarity to all texts
        # Each column represents one text's similarity to all images
        logits_per_image = image_embeds @ text_embeds.T / self.temperature
        logits_per_text = text_embeds @ image_embeds.T / self.temperature

        # Create labels: diagonal entries are positive pairs
        # For batch_size=4: labels = [0, 1, 2, 3]
        labels = torch.arange(batch_size, device=image_embeds.device)

        # Compute cross-entropy loss in both directions
        # Image-to-text: for each image, predict which text matches
        # Text-to-image: for each text, predict which image matches
        image_to_text_loss = F.cross_entropy(logits_per_image, labels)
        text_to_image_loss = F.cross_entropy(logits_per_text, labels)

        # Average the two directional losses
        total_loss = (image_to_text_loss + text_to_image_loss) / 2.0

        return {
            "loss": total_loss,
            "image_to_text_loss": image_to_text_loss,
            "text_to_image_loss": text_to_image_loss,
            "temperature": self.temperature.detach(),
            "logits_per_image": logits_per_image.detach(),
            "logits_per_text": logits_per_text.detach(),
        }

    def extra_repr(self) -> str:
        """String representation for debugging."""
        return f"temperature={self.temperature.item():.4f}"
