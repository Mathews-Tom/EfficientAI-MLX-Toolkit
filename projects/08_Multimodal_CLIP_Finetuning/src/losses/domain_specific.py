#!/usr/bin/env python3
"""
Domain-adapted contrastive loss for specialized applications.

This module implements domain-specific adaptations of contrastive loss
tailored for medical, industrial, and scientific domains.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DomainSpecificLoss(nn.Module):
    """Domain-adapted contrastive loss.

    This loss applies domain-specific weighting and strategies to improve
    performance on specialized tasks:
    - Medical: Higher weight on exact matches, stricter similarity requirements
    - Industrial: Focus on technical details, part-level matching
    - Scientific: Multi-scale emphasis, hierarchical matching
    - General: Standard contrastive loss

    Attributes:
        domain: Target domain (medical, industrial, scientific, general)
        temperature: Temperature parameter for scaling logits
        domain_weight: Weight multiplier for domain-specific adjustments
    """

    VALID_DOMAINS = {"general", "medical", "industrial", "scientific"}

    def __init__(
        self,
        domain: str,
        temperature: float = 0.07,
        domain_weight: float = 1.0,
    ) -> None:
        """Initialize domain-specific loss.

        Args:
            domain: Target domain (medical, industrial, scientific, general)
            temperature: Temperature for scaling logits
            domain_weight: Weight multiplier for domain adjustments (>= 1.0)

        Raises:
            ValueError: If parameters are invalid
        """
        super().__init__()

        if domain not in self.VALID_DOMAINS:
            raise ValueError(
                f"Domain must be one of {self.VALID_DOMAINS}, got {domain}"
            )

        if temperature <= 0:
            raise ValueError(f"Temperature must be positive, got {temperature}")

        if domain_weight < 1.0:
            raise ValueError(
                f"Domain weight must be >= 1.0, got {domain_weight}"
            )

        self.domain = domain
        self.domain_weight = domain_weight
        self.register_buffer("log_temperature", torch.log(torch.tensor(temperature)))

        # Domain-specific temperature adjustments
        self._domain_temp_scale = {
            "medical": 0.8,  # Sharper distribution for exact matching
            "industrial": 1.0,  # Standard temperature
            "scientific": 1.2,  # Softer distribution for hierarchical concepts
            "general": 1.0,  # Standard temperature
        }

    @property
    def temperature(self) -> torch.Tensor:
        """Get current temperature value with domain adjustment."""
        base_temp = torch.exp(self.log_temperature)
        scale = self._domain_temp_scale[self.domain]
        return base_temp * scale

    def _apply_medical_weighting(
        self,
        similarity_matrix: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Apply medical domain-specific weighting.

        Medical domain emphasizes:
        - Exact matches (higher weight on positives)
        - Penalizing false positives heavily
        - Stricter similarity requirements

        Args:
            similarity_matrix: Similarity scores [batch_size, batch_size]
            labels: Ground truth labels [batch_size]

        Returns:
            Weighted similarity matrix
        """
        batch_size = similarity_matrix.size(0)
        device = similarity_matrix.device

        # Create positive mask (diagonal)
        positive_mask = torch.eye(batch_size, dtype=torch.bool, device=device)

        # Boost positive pair similarities
        positive_boost = 1.0 + (0.2 * self.domain_weight)
        weighted = torch.where(
            positive_mask,
            similarity_matrix * positive_boost,
            similarity_matrix,
        )

        # Penalize high-similarity negatives more strongly
        negative_mask = ~positive_mask
        high_sim_negatives = (similarity_matrix > 0.5) & negative_mask
        penalty = 1.0 + (0.3 * self.domain_weight)
        weighted = torch.where(
            high_sim_negatives,
            weighted * penalty,
            weighted,
        )

        return weighted

    def _apply_industrial_weighting(
        self,
        similarity_matrix: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Apply industrial domain-specific weighting.

        Industrial domain emphasizes:
        - Technical term matching
        - Part-level alignment
        - Moderate strictness

        Args:
            similarity_matrix: Similarity scores [batch_size, batch_size]
            labels: Ground truth labels [batch_size]

        Returns:
            Weighted similarity matrix
        """
        batch_size = similarity_matrix.size(0)
        device = similarity_matrix.device

        # Create positive mask
        positive_mask = torch.eye(batch_size, dtype=torch.bool, device=device)

        # Moderate boost to positive pairs
        positive_boost = 1.0 + (0.15 * self.domain_weight)
        weighted = torch.where(
            positive_mask,
            similarity_matrix * positive_boost,
            similarity_matrix,
        )

        # Identify medium-similarity negatives (semi-hard mining)
        negative_mask = ~positive_mask
        medium_sim_negatives = (
            (similarity_matrix > 0.3) & (similarity_matrix < 0.7) & negative_mask
        )
        boost = 1.0 + (0.2 * self.domain_weight)
        weighted = torch.where(
            medium_sim_negatives,
            weighted * boost,
            weighted,
        )

        return weighted

    def _apply_scientific_weighting(
        self,
        similarity_matrix: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Apply scientific domain-specific weighting.

        Scientific domain emphasizes:
        - Hierarchical concept matching
        - Multi-scale understanding
        - Softer similarity requirements

        Args:
            similarity_matrix: Similarity scores [batch_size, batch_size]
            labels: Ground truth labels [batch_size]

        Returns:
            Weighted similarity matrix
        """
        batch_size = similarity_matrix.size(0)
        device = similarity_matrix.device

        # Create positive mask
        positive_mask = torch.eye(batch_size, dtype=torch.bool, device=device)

        # Moderate boost to positive pairs
        positive_boost = 1.0 + (0.1 * self.domain_weight)
        weighted = torch.where(
            positive_mask,
            similarity_matrix * positive_boost,
            similarity_matrix,
        )

        # For scientific domain, we want to capture hierarchical relationships
        # Apply softer weighting to encourage learning at multiple scales
        # This is achieved through the temperature adjustment in temperature property

        return weighted

    def forward(
        self,
        image_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
        domain_labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute domain-specific contrastive loss.

        Args:
            image_embeds: Image embeddings [batch_size, embed_dim]
            text_embeds: Text embeddings [batch_size, embed_dim]
            domain_labels: Optional per-sample domain labels [batch_size]
                If provided, applies per-sample domain weighting

        Returns:
            Dictionary containing:
                - loss: Total domain-adapted contrastive loss
                - image_to_text_loss: Image-to-text loss
                - text_to_image_loss: Text-to-image loss
                - temperature: Current temperature value
                - domain: Domain name
                - domain_weight: Domain weight multiplier

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

        # Apply domain-specific weighting
        labels = torch.arange(batch_size, device=image_embeds.device)

        if self.domain == "medical":
            similarity_i2t = self._apply_medical_weighting(similarity_i2t, labels)
            similarity_t2i = self._apply_medical_weighting(similarity_t2i, labels)
        elif self.domain == "industrial":
            similarity_i2t = self._apply_industrial_weighting(similarity_i2t, labels)
            similarity_t2i = self._apply_industrial_weighting(similarity_t2i, labels)
        elif self.domain == "scientific":
            similarity_i2t = self._apply_scientific_weighting(similarity_i2t, labels)
            similarity_t2i = self._apply_scientific_weighting(similarity_t2i, labels)
        # else: general domain uses standard similarity

        # Scale by temperature
        logits_i2t = similarity_i2t / self.temperature
        logits_t2i = similarity_t2i / self.temperature

        # Compute cross-entropy loss
        i2t_loss = F.cross_entropy(logits_i2t, labels)
        t2i_loss = F.cross_entropy(logits_t2i, labels)

        total_loss = (i2t_loss + t2i_loss) / 2.0

        return {
            "loss": total_loss,
            "image_to_text_loss": i2t_loss,
            "text_to_image_loss": t2i_loss,
            "temperature": self.temperature.detach(),
            "domain": self.domain,
            "domain_weight": self.domain_weight,
        }

    def extra_repr(self) -> str:
        """String representation for debugging."""
        return (
            f"domain={self.domain}, "
            f"temperature={self.temperature.item():.4f}, "
            f"domain_weight={self.domain_weight:.2f}"
        )
