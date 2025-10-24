#!/usr/bin/env python3
"""
Contrastive loss with hard negative mining.

This module implements contrastive loss with hard negative mining to focus
learning on challenging examples that are similar but not matching pairs.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HardNegativeMiningLoss(nn.Module):
    """Contrastive loss with hard negative mining.

    This loss identifies hard negatives (high similarity but wrong pairs)
    and weights them more heavily during training to improve the model's
    ability to distinguish between subtle differences.

    Hard negative mining strategies:
    - Semi-hard: Negatives with similarity > margin but < positive
    - Hard: Top-k most similar negatives
    - Weighted: Weight negatives proportional to their difficulty

    Attributes:
        temperature: Temperature parameter for scaling logits
        hard_negative_ratio: Ratio of negatives to treat as hard (0-1)
        mining_strategy: Strategy for identifying hard negatives
    """

    def __init__(
        self,
        temperature: float = 0.07,
        hard_negative_ratio: float = 0.5,
        mining_strategy: str = "semi-hard",
        hard_negative_weight: float = 2.0,
    ) -> None:
        """Initialize hard negative mining loss.

        Args:
            temperature: Temperature for scaling logits
            hard_negative_ratio: Ratio of negatives to treat as hard (0-1)
            mining_strategy: Strategy for mining ("semi-hard", "hard", "weighted")
            hard_negative_weight: Weight multiplier for hard negatives

        Raises:
            ValueError: If parameters are invalid
        """
        super().__init__()

        if temperature <= 0:
            raise ValueError(f"Temperature must be positive, got {temperature}")

        if not 0 <= hard_negative_ratio <= 1:
            raise ValueError(
                f"Hard negative ratio must be in [0, 1], got {hard_negative_ratio}"
            )

        if mining_strategy not in {"semi-hard", "hard", "weighted"}:
            raise ValueError(
                f"Mining strategy must be 'semi-hard', 'hard', or 'weighted', "
                f"got {mining_strategy}"
            )

        if hard_negative_weight < 1.0:
            raise ValueError(
                f"Hard negative weight must be >= 1.0, got {hard_negative_weight}"
            )

        self.register_buffer("log_temperature", torch.log(torch.tensor(temperature)))
        self.hard_negative_ratio = hard_negative_ratio
        self.mining_strategy = mining_strategy
        self.hard_negative_weight = hard_negative_weight

    @property
    def temperature(self) -> torch.Tensor:
        """Get current temperature value."""
        return torch.exp(self.log_temperature)

    def _identify_hard_negatives(
        self,
        similarity_matrix: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Identify hard negative samples.

        Args:
            similarity_matrix: Similarity scores [batch_size, batch_size]
            labels: Ground truth labels [batch_size]

        Returns:
            Boolean mask indicating hard negatives [batch_size, batch_size]
        """
        batch_size = similarity_matrix.size(0)
        device = similarity_matrix.device

        # Create mask for negative pairs (off-diagonal)
        negative_mask = ~torch.eye(batch_size, dtype=torch.bool, device=device)

        if self.mining_strategy == "semi-hard":
            # Semi-hard: negatives harder than a margin but easier than positives
            # Get positive similarities (diagonal)
            positive_sim = similarity_matrix.diagonal()

            # Expand for broadcasting
            positive_sim = positive_sim.unsqueeze(1)  # [batch_size, 1]

            # Semi-hard negatives: similarity > 0 but < positive
            hard_mask = (similarity_matrix > 0) & (similarity_matrix < positive_sim)
            hard_mask = hard_mask & negative_mask

        elif self.mining_strategy == "hard":
            # Hard: top-k most similar negatives
            num_hard = max(1, int(self.hard_negative_ratio * (batch_size - 1)))

            # For each row, get top-k negative similarities
            hard_mask = torch.zeros_like(negative_mask)
            for i in range(batch_size):
                # Get similarities for this sample (excluding diagonal)
                neg_sims = similarity_matrix[i].clone()
                neg_sims[i] = float("-inf")  # Mask diagonal

                # Get top-k indices
                _, top_k_indices = torch.topk(neg_sims, k=num_hard)
                hard_mask[i, top_k_indices] = True

        else:  # weighted
            # Weighted: weight proportional to similarity
            # Use all negatives but weight them
            hard_mask = negative_mask

        return hard_mask

    def forward(
        self,
        image_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute contrastive loss with hard negative mining.

        Args:
            image_embeds: Image embeddings [batch_size, embed_dim]
            text_embeds: Text embeddings [batch_size, embed_dim]

        Returns:
            Dictionary containing:
                - loss: Total contrastive loss with hard negative mining
                - image_to_text_loss: Image-to-text loss
                - text_to_image_loss: Text-to-image loss
                - hard_negative_count: Number of hard negatives identified
                - hard_negative_ratio_actual: Actual ratio of hard negatives
                - temperature: Current temperature value

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

        batch_size = image_embeds.size(0)

        # Normalize embeddings
        image_embeds = F.normalize(image_embeds, p=2, dim=1)
        text_embeds = F.normalize(text_embeds, p=2, dim=1)

        # Compute similarity matrices (before temperature scaling)
        similarity_i2t = image_embeds @ text_embeds.T
        similarity_t2i = text_embeds @ image_embeds.T

        # Identify hard negatives
        labels = torch.arange(batch_size, device=image_embeds.device)
        hard_mask_i2t = self._identify_hard_negatives(similarity_i2t, labels)
        hard_mask_t2i = self._identify_hard_negatives(similarity_t2i, labels)

        # Scale similarities by temperature
        logits_i2t = similarity_i2t / self.temperature
        logits_t2i = similarity_t2i / self.temperature

        if self.mining_strategy == "weighted":
            # Apply weighted hard negative mining
            # Weight negatives proportional to their similarity
            weights_i2t = torch.ones_like(logits_i2t)
            weights_t2i = torch.ones_like(logits_t2i)

            # Increase weight for hard negatives
            neg_weights_i2t = (
                torch.sigmoid(similarity_i2t) * self.hard_negative_weight
            )
            neg_weights_t2i = (
                torch.sigmoid(similarity_t2i) * self.hard_negative_weight
            )

            weights_i2t = torch.where(hard_mask_i2t, neg_weights_i2t, weights_i2t)
            weights_t2i = torch.where(hard_mask_t2i, neg_weights_t2i, weights_t2i)

            # Compute weighted cross-entropy
            log_probs_i2t = F.log_softmax(logits_i2t, dim=1)
            log_probs_t2i = F.log_softmax(logits_t2i, dim=1)

            # Get positive log probabilities
            positive_log_probs_i2t = log_probs_i2t[
                torch.arange(batch_size), labels
            ]
            positive_log_probs_t2i = log_probs_t2i[
                torch.arange(batch_size), labels
            ]

            # Compute weighted loss
            i2t_loss = -(weights_i2t * log_probs_i2t).sum(dim=1).mean()
            t2i_loss = -(weights_t2i * log_probs_t2i).sum(dim=1).mean()

        else:
            # Standard cross-entropy with hard negative reweighting
            # Increase logits for hard negatives to make them harder
            logits_i2t = torch.where(
                hard_mask_i2t,
                logits_i2t * self.hard_negative_weight,
                logits_i2t,
            )
            logits_t2i = torch.where(
                hard_mask_t2i,
                logits_t2i * self.hard_negative_weight,
                logits_t2i,
            )

            i2t_loss = F.cross_entropy(logits_i2t, labels)
            t2i_loss = F.cross_entropy(logits_t2i, labels)

        total_loss = (i2t_loss + t2i_loss) / 2.0

        # Compute statistics
        hard_count_i2t = hard_mask_i2t.sum().item()
        hard_count_t2i = hard_mask_t2i.sum().item()
        total_negatives = batch_size * (batch_size - 1)
        hard_count = (hard_count_i2t + hard_count_t2i) / 2
        actual_ratio = hard_count / (total_negatives / 2) if total_negatives > 0 else 0

        return {
            "loss": total_loss,
            "image_to_text_loss": i2t_loss,
            "text_to_image_loss": t2i_loss,
            "hard_negative_count": hard_count,
            "hard_negative_ratio_actual": actual_ratio,
            "temperature": self.temperature.detach(),
        }

    def extra_repr(self) -> str:
        """String representation for debugging."""
        return (
            f"temperature={self.temperature.item():.4f}, "
            f"hard_negative_ratio={self.hard_negative_ratio:.2f}, "
            f"mining_strategy={self.mining_strategy}, "
            f"hard_negative_weight={self.hard_negative_weight:.2f}"
        )
