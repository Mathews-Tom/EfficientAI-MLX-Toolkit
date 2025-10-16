"""
Domain Adapter for Learned Optimization Strategies

Adapts hyperparameter optimization strategies based on domain-specific
characteristics (image type, style, complexity, etc.).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

import mlx.core as mx
import numpy as np

logger = logging.getLogger(__name__)


class DomainType(Enum):
    """Supported domain types for adaptation."""

    PHOTOREALISTIC = "photorealistic"
    ARTISTIC = "artistic"
    SYNTHETIC = "synthetic"
    SCIENTIFIC = "scientific"
    ARCHITECTURAL = "architectural"
    PORTRAIT = "portrait"
    LANDSCAPE = "landscape"
    ABSTRACT = "abstract"
    GENERAL = "general"


@dataclass
class DomainConfig:
    """
    Configuration for domain-specific optimization.

    Attributes:
        domain_type: Type of domain
        num_steps: Optimized number of sampling steps
        adaptive_threshold: Optimized adaptive threshold
        progress_power: Optimized progress power
        quality_weight: Weight for quality in reward
        speed_weight: Weight for speed in reward
        confidence: Confidence in these settings (0-1)
    """

    domain_type: DomainType
    num_steps: int = 50
    adaptive_threshold: float = 0.5
    progress_power: float = 2.0
    quality_weight: float = 0.7
    speed_weight: float = 0.3
    confidence: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "domain_type": self.domain_type.value,
            "num_steps": self.num_steps,
            "adaptive_threshold": self.adaptive_threshold,
            "progress_power": self.progress_power,
            "quality_weight": self.quality_weight,
            "speed_weight": self.speed_weight,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DomainConfig:
        """Create from dictionary."""
        domain_type = DomainType(data["domain_type"])
        return cls(
            domain_type=domain_type,
            num_steps=data.get("num_steps", 50),
            adaptive_threshold=data.get("adaptive_threshold", 0.5),
            progress_power=data.get("progress_power", 2.0),
            quality_weight=data.get("quality_weight", 0.7),
            speed_weight=data.get("speed_weight", 0.3),
            confidence=data.get("confidence", 0.5),
        )


class DomainAdapter:
    """
    Adapter for domain-specific optimization strategies.

    Learns and applies domain-specific hyperparameter configurations
    to optimize for different content types.
    """

    def __init__(self):
        """Initialize domain adapter."""
        self.domain_configs: dict[DomainType, DomainConfig] = {}
        self._initialize_default_configs()

    def _initialize_default_configs(self):
        """Initialize default configurations for each domain."""
        # Photorealistic: High quality, more steps
        self.domain_configs[DomainType.PHOTOREALISTIC] = DomainConfig(
            domain_type=DomainType.PHOTOREALISTIC,
            num_steps=60,
            adaptive_threshold=0.6,
            progress_power=2.5,
            quality_weight=0.8,
            speed_weight=0.2,
            confidence=0.7,
        )

        # Artistic: Balanced quality-speed
        self.domain_configs[DomainType.ARTISTIC] = DomainConfig(
            domain_type=DomainType.ARTISTIC,
            num_steps=50,
            adaptive_threshold=0.5,
            progress_power=2.0,
            quality_weight=0.7,
            speed_weight=0.3,
            confidence=0.7,
        )

        # Synthetic: Lower quality OK, faster
        self.domain_configs[DomainType.SYNTHETIC] = DomainConfig(
            domain_type=DomainType.SYNTHETIC,
            num_steps=35,
            adaptive_threshold=0.4,
            progress_power=1.5,
            quality_weight=0.6,
            speed_weight=0.4,
            confidence=0.6,
        )

        # Scientific: High precision
        self.domain_configs[DomainType.SCIENTIFIC] = DomainConfig(
            domain_type=DomainType.SCIENTIFIC,
            num_steps=70,
            adaptive_threshold=0.7,
            progress_power=3.0,
            quality_weight=0.9,
            speed_weight=0.1,
            confidence=0.7,
        )

        # Portrait: High quality for faces
        self.domain_configs[DomainType.PORTRAIT] = DomainConfig(
            domain_type=DomainType.PORTRAIT,
            num_steps=65,
            adaptive_threshold=0.65,
            progress_power=2.8,
            quality_weight=0.85,
            speed_weight=0.15,
            confidence=0.75,
        )

        # Landscape: Balanced
        self.domain_configs[DomainType.LANDSCAPE] = DomainConfig(
            domain_type=DomainType.LANDSCAPE,
            num_steps=55,
            adaptive_threshold=0.55,
            progress_power=2.2,
            quality_weight=0.75,
            speed_weight=0.25,
            confidence=0.7,
        )

        # Abstract: Lower precision OK
        self.domain_configs[DomainType.ABSTRACT] = DomainConfig(
            domain_type=DomainType.ABSTRACT,
            num_steps=40,
            adaptive_threshold=0.45,
            progress_power=1.8,
            quality_weight=0.65,
            speed_weight=0.35,
            confidence=0.6,
        )

        # General: Balanced defaults
        self.domain_configs[DomainType.GENERAL] = DomainConfig(
            domain_type=DomainType.GENERAL,
            num_steps=50,
            adaptive_threshold=0.5,
            progress_power=2.0,
            quality_weight=0.7,
            speed_weight=0.3,
            confidence=0.5,
        )

    def detect_domain(
        self, sample: mx.array | None = None, prompt: str | None = None
    ) -> DomainType:
        """
        Detect domain type from sample or prompt.

        Args:
            sample: Optional sample to analyze
            prompt: Optional text prompt to analyze

        Returns:
            Detected domain type
        """
        # If prompt provided, use keyword matching
        if prompt is not None:
            prompt_lower = prompt.lower()

            # Check keywords for domain detection (order matters - check specific before general)
            if any(
                word in prompt_lower
                for word in ["diagram", "chart", "scientific", "technical", "medical"]
            ):
                return DomainType.SCIENTIFIC

            if any(
                word in prompt_lower
                for word in ["portrait", "face", "person", "headshot", "selfie"]
            ):
                return DomainType.PORTRAIT

            if any(
                word in prompt_lower
                for word in ["landscape", "scenery", "nature", "mountain", "vista"]
            ):
                return DomainType.LANDSCAPE

            if any(
                word in prompt_lower
                for word in ["photo", "photograph", "realistic", "real", "camera"]
            ):
                return DomainType.PHOTOREALISTIC

            if any(
                word in prompt_lower
                for word in ["render", "3d", "cgi", "synthetic", "digital"]
            ):
                return DomainType.SYNTHETIC

            if any(
                word in prompt_lower
                for word in ["art", "painting", "artistic", "style", "canvas"]
            ):
                return DomainType.ARTISTIC

            if any(
                word in prompt_lower
                for word in ["abstract", "pattern", "geometric", "shapes"]
            ):
                return DomainType.ABSTRACT

        # If sample provided, analyze characteristics
        if sample is not None:
            complexity = self._estimate_complexity(sample)
            structure = self._estimate_structure(sample)

            # High complexity + high structure = photorealistic
            if complexity > 0.7 and structure > 0.7:
                return DomainType.PHOTOREALISTIC

            # Low structure = abstract
            if structure < 0.3:
                return DomainType.ABSTRACT

            # Medium complexity = artistic
            if 0.4 <= complexity <= 0.7:
                return DomainType.ARTISTIC

        # Default to general
        return DomainType.GENERAL

    def _estimate_complexity(self, sample: mx.array) -> float:
        """Estimate sample complexity (0-1)."""
        if len(sample.shape) < 3:
            return 0.5

        # Use variance as complexity proxy
        variance = float(mx.var(sample))
        complexity = min(variance / 0.5, 1.0)  # Normalize

        return complexity

    def _estimate_structure(self, sample: mx.array) -> float:
        """Estimate structural content (0-1)."""
        if len(sample.shape) != 4:
            return 0.5

        # Compute gradients as structure proxy
        dx = sample[:, 1:, :, :] - sample[:, :-1, :, :]
        dy = sample[:, :, 1:, :] - sample[:, :, :-1, :]

        # Align shapes
        min_h = min(dx.shape[1], dy.shape[1])
        min_w = min(dx.shape[2], dy.shape[2])

        dx_aligned = dx[:, :min_h, :min_w, :]
        dy_aligned = dy[:, :min_h, :min_w, :]

        # High gradient consistency = high structure
        gradient_std = float(
            mx.std(mx.sqrt(dx_aligned**2 + dy_aligned**2))
        )
        structure = 1.0 - min(gradient_std, 1.0)

        return structure

    def get_config(
        self,
        domain_type: DomainType | None = None,
        sample: mx.array | None = None,
        prompt: str | None = None,
    ) -> DomainConfig:
        """
        Get domain-specific configuration.

        Args:
            domain_type: Explicit domain type (auto-detected if None)
            sample: Optional sample for auto-detection
            prompt: Optional prompt for auto-detection

        Returns:
            Domain configuration
        """
        if domain_type is None:
            domain_type = self.detect_domain(sample=sample, prompt=prompt)

        config = self.domain_configs.get(domain_type, self.domain_configs[DomainType.GENERAL])

        logger.info(f"Using domain config for {domain_type.value} (confidence: {config.confidence:.2f})")

        return config

    def update_config(
        self,
        domain_type: DomainType,
        config: DomainConfig | None = None,
        **kwargs,
    ):
        """
        Update configuration for a domain.

        Args:
            domain_type: Domain to update
            config: New configuration (or use kwargs)
            **kwargs: Individual config parameters to update
        """
        if config is not None:
            self.domain_configs[domain_type] = config
        elif kwargs:
            current = self.domain_configs.get(
                domain_type, self.domain_configs[DomainType.GENERAL]
            )

            # Update specified parameters
            updated = DomainConfig(
                domain_type=domain_type,
                num_steps=kwargs.get("num_steps", current.num_steps),
                adaptive_threshold=kwargs.get(
                    "adaptive_threshold", current.adaptive_threshold
                ),
                progress_power=kwargs.get("progress_power", current.progress_power),
                quality_weight=kwargs.get("quality_weight", current.quality_weight),
                speed_weight=kwargs.get("speed_weight", current.speed_weight),
                confidence=kwargs.get("confidence", current.confidence),
            )

            self.domain_configs[domain_type] = updated

        logger.info(f"Updated config for {domain_type.value}")

    def learn_from_results(
        self,
        domain_type: DomainType,
        quality: float,
        speed: float,
        hyperparameters: dict[str, Any],
        learning_rate: float = 0.1,
    ):
        """
        Learn from optimization results to improve domain configs.

        Args:
            domain_type: Domain that was optimized
            quality: Achieved quality
            speed: Achieved speed
            hyperparameters: Hyperparameters that achieved this result
            learning_rate: Learning rate for updates
        """
        current = self.domain_configs.get(
            domain_type, self.domain_configs[DomainType.GENERAL]
        )

        # If results are better than current confidence, update config
        overall_score = quality * current.quality_weight + speed * current.speed_weight

        if overall_score > current.confidence:
            # Update hyperparameters with exponential moving average
            updated_num_steps = int(
                current.num_steps * (1 - learning_rate)
                + hyperparameters.get("num_steps", current.num_steps) * learning_rate
            )

            updated_threshold = (
                current.adaptive_threshold * (1 - learning_rate)
                + hyperparameters.get("adaptive_threshold", current.adaptive_threshold)
                * learning_rate
            )

            updated_power = (
                current.progress_power * (1 - learning_rate)
                + hyperparameters.get("progress_power", current.progress_power)
                * learning_rate
            )

            # Increase confidence
            updated_confidence = min(current.confidence + 0.05, 0.95)

            self.update_config(
                domain_type,
                num_steps=updated_num_steps,
                adaptive_threshold=updated_threshold,
                progress_power=updated_power,
                confidence=updated_confidence,
            )

            logger.info(
                f"Learned from results for {domain_type.value}: "
                f"quality={quality:.3f}, speed={speed:.3f}"
            )

    def get_all_configs(self) -> dict[DomainType, DomainConfig]:
        """
        Get all domain configurations.

        Returns:
            Dictionary mapping domain types to configs
        """
        return self.domain_configs.copy()


def create_domain_adapter() -> DomainAdapter:
    """
    Factory function to create domain adapter.

    Returns:
        Initialized DomainAdapter
    """
    return DomainAdapter()
