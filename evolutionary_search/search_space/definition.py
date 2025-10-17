"""
Search space definition for diffusion model architectures.

This module defines the search space for evolutionary architecture search,
including architecture components, genome representation, and constraints
for Apple Silicon hardware.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

__all__ = [
    "ArchitectureComponent",
    "ArchitectureGenome",
    "SearchSpaceConfig",
    "LayerConfig",
]


class ArchitectureComponent(Enum):
    """Architecture component types for diffusion models."""

    ATTENTION_BLOCK = "attention"
    CONV_BLOCK = "conv"
    RESIDUAL_BLOCK = "residual"
    NORMALIZATION = "norm"
    ACTIVATION = "activation"
    TIMESTEP_EMBEDDING = "timestep_emb"
    CROSS_ATTENTION = "cross_attention"


@dataclass
class LayerConfig:
    """Configuration for a single layer in the architecture."""

    component_type: ArchitectureComponent
    parameters: dict[str, Any]
    layer_index: int
    in_channels: int | None = None
    out_channels: int | None = None

    def validate(self) -> bool:
        """Validate layer configuration."""
        if self.component_type == ArchitectureComponent.CONV_BLOCK:
            required_params = {"kernel_size", "stride", "padding"}
            return required_params.issubset(self.parameters.keys())
        elif self.component_type == ArchitectureComponent.ATTENTION_BLOCK:
            required_params = {"num_heads", "embed_dim"}
            return required_params.issubset(self.parameters.keys())
        return True


@dataclass
class ArchitectureGenome:
    """
    Genome representation for diffusion model architectures.

    Attributes:
        layers: List of layer configurations
        connections: List of (from_layer, to_layer) tuples
        parameters: Global architecture parameters
        fitness_score: Fitness score from evaluation
        generation: Generation number
        metadata: Additional metadata (memory usage, FLOPs, etc.)
    """

    layers: list[LayerConfig]
    connections: list[tuple[int, int]] = field(default_factory=list)
    parameters: dict[str, Any] = field(default_factory=dict)
    fitness_score: float = 0.0
    generation: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate genome after initialization."""
        if not self.connections:
            # Create sequential connections if none provided
            self.connections = [
                (i, i + 1) for i in range(len(self.layers) - 1)
            ]

    def validate(self) -> bool:
        """Validate genome structure and constraints."""
        # Check layer count
        if len(self.layers) < 2:
            return False

        # Validate all layers
        for layer in self.layers:
            if not layer.validate():
                return False

        # Validate connections
        max_idx = len(self.layers) - 1
        for from_idx, to_idx in self.connections:
            if not (0 <= from_idx <= max_idx and 0 <= to_idx <= max_idx):
                return False
            if from_idx >= to_idx:  # No backward connections
                return False

        return True

    def count_parameters(self) -> int:
        """Estimate total parameter count."""
        total_params = 0
        for layer in self.layers:
            if layer.component_type == ArchitectureComponent.CONV_BLOCK:
                kernel_size = layer.parameters.get("kernel_size", 3)
                in_ch = layer.in_channels or 64
                out_ch = layer.out_channels or 64
                total_params += kernel_size * kernel_size * in_ch * out_ch
            elif layer.component_type == ArchitectureComponent.ATTENTION_BLOCK:
                embed_dim = layer.parameters.get("embed_dim", 512)
                num_heads = layer.parameters.get("num_heads", 8)
                # Q, K, V projections + output projection
                total_params += 4 * embed_dim * embed_dim
        return total_params


@dataclass
class SearchSpaceConfig:
    """
    Configuration for the search space.

    Defines constraints and ranges for architecture components
    to ensure hardware compatibility (Apple Silicon).
    """

    # Layer constraints
    min_layers: int = 4
    max_layers: int = 32
    allowed_components: list[ArchitectureComponent] = field(
        default_factory=lambda: list(ArchitectureComponent)
    )

    # Parameter ranges
    channel_sizes: list[int] = field(default_factory=lambda: [64, 128, 256, 512])
    attention_heads: list[int] = field(default_factory=lambda: [4, 8, 16])
    kernel_sizes: list[int] = field(default_factory=lambda: [3, 5, 7])

    # Hardware constraints (Apple Silicon)
    max_memory_mb: int = 16384  # 16GB for M1/M2
    max_parameters: int = 1_000_000_000  # 1B parameters
    target_inference_ms: float = 100.0  # Target inference time

    def validate_genome(self, genome: ArchitectureGenome) -> bool:
        """Validate genome against search space constraints."""
        # Check layer count
        if not (self.min_layers <= len(genome.layers) <= self.max_layers):
            return False

        # Check component types
        for layer in genome.layers:
            if layer.component_type not in self.allowed_components:
                return False

        # Check parameter count
        if genome.count_parameters() > self.max_parameters:
            return False

        return genome.validate()

    @classmethod
    def default(cls) -> SearchSpaceConfig:
        """Create default search space configuration."""
        return cls()
