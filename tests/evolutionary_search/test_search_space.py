"""
Tests for search space definition and architecture genome.
"""

from __future__ import annotations

import pytest

from evolutionary_search.search_space import (
    ArchitectureComponent,
    ArchitectureGenome,
    LayerConfig,
    SearchSpaceConfig,
)


class TestLayerConfig:
    """Tests for LayerConfig."""

    def test_layer_config_creation(self) -> None:
        """Test creating a layer configuration."""
        layer = LayerConfig(
            component_type=ArchitectureComponent.CONV_BLOCK,
            parameters={"kernel_size": 3, "stride": 1, "padding": 1},
            layer_index=0,
            in_channels=3,
            out_channels=64,
        )

        assert layer.component_type == ArchitectureComponent.CONV_BLOCK
        assert layer.parameters["kernel_size"] == 3
        assert layer.in_channels == 3
        assert layer.out_channels == 64

    def test_conv_layer_validation(self) -> None:
        """Test validation of convolutional layer."""
        # Valid conv layer
        layer = LayerConfig(
            component_type=ArchitectureComponent.CONV_BLOCK,
            parameters={"kernel_size": 3, "stride": 1, "padding": 1},
            layer_index=0,
        )
        assert layer.validate()

        # Invalid conv layer (missing required parameters)
        layer_invalid = LayerConfig(
            component_type=ArchitectureComponent.CONV_BLOCK,
            parameters={"stride": 1},
            layer_index=0,
        )
        assert not layer_invalid.validate()

    def test_attention_layer_validation(self) -> None:
        """Test validation of attention layer."""
        # Valid attention layer
        layer = LayerConfig(
            component_type=ArchitectureComponent.ATTENTION_BLOCK,
            parameters={"num_heads": 8, "embed_dim": 512},
            layer_index=0,
        )
        assert layer.validate()

        # Invalid attention layer
        layer_invalid = LayerConfig(
            component_type=ArchitectureComponent.ATTENTION_BLOCK,
            parameters={"num_heads": 8},
            layer_index=0,
        )
        assert not layer_invalid.validate()


class TestArchitectureGenome:
    """Tests for ArchitectureGenome."""

    def test_genome_creation(self) -> None:
        """Test creating an architecture genome."""
        layers = [
            LayerConfig(
                component_type=ArchitectureComponent.CONV_BLOCK,
                parameters={"kernel_size": 3, "stride": 1, "padding": 1},
                layer_index=0,
                in_channels=3,
                out_channels=64,
            ),
            LayerConfig(
                component_type=ArchitectureComponent.ATTENTION_BLOCK,
                parameters={"num_heads": 4, "embed_dim": 64},
                layer_index=1,
                in_channels=64,
                out_channels=64,
            ),
        ]

        genome = ArchitectureGenome(layers=layers)

        assert len(genome.layers) == 2
        assert len(genome.connections) == 1  # Auto-generated sequential
        assert genome.connections[0] == (0, 1)

    def test_genome_validation(self) -> None:
        """Test genome validation."""
        # Valid genome
        layers = [
            LayerConfig(
                component_type=ArchitectureComponent.CONV_BLOCK,
                parameters={"kernel_size": 3, "stride": 1, "padding": 1},
                layer_index=i,
                in_channels=64,
                out_channels=64,
            )
            for i in range(4)
        ]
        genome = ArchitectureGenome(layers=layers)
        assert genome.validate()

        # Invalid genome (too few layers)
        genome_invalid = ArchitectureGenome(layers=layers[:1])
        assert not genome_invalid.validate()

    def test_genome_with_skip_connections(self) -> None:
        """Test genome with skip connections."""
        layers = [
            LayerConfig(
                component_type=ArchitectureComponent.CONV_BLOCK,
                parameters={"kernel_size": 3, "stride": 1, "padding": 1},
                layer_index=i,
                in_channels=64,
                out_channels=64,
            )
            for i in range(4)
        ]

        connections = [(0, 1), (1, 2), (2, 3), (0, 3)]  # Skip from 0 to 3

        genome = ArchitectureGenome(layers=layers, connections=connections)

        assert len(genome.connections) == 4
        assert (0, 3) in genome.connections
        assert genome.validate()

    def test_invalid_connections(self) -> None:
        """Test validation of invalid connections."""
        layers = [
            LayerConfig(
                component_type=ArchitectureComponent.CONV_BLOCK,
                parameters={"kernel_size": 3, "stride": 1, "padding": 1},
                layer_index=i,
                in_channels=64,
                out_channels=64,
            )
            for i in range(4)
        ]

        # Backward connection (invalid)
        connections = [(0, 1), (2, 1)]
        genome = ArchitectureGenome(layers=layers, connections=connections)
        assert not genome.validate()

        # Out of bounds connection
        connections = [(0, 10)]
        genome = ArchitectureGenome(layers=layers, connections=connections)
        assert not genome.validate()

    def test_parameter_count(self) -> None:
        """Test parameter count estimation."""
        layers = [
            LayerConfig(
                component_type=ArchitectureComponent.CONV_BLOCK,
                parameters={"kernel_size": 3, "stride": 1, "padding": 1},
                layer_index=0,
                in_channels=64,
                out_channels=64,
            ),
            LayerConfig(
                component_type=ArchitectureComponent.ATTENTION_BLOCK,
                parameters={"num_heads": 8, "embed_dim": 512},
                layer_index=1,
                in_channels=64,
                out_channels=64,
            ),
        ]

        genome = ArchitectureGenome(layers=layers)
        param_count = genome.count_parameters()

        assert param_count > 0
        assert isinstance(param_count, int)


class TestSearchSpaceConfig:
    """Tests for SearchSpaceConfig."""

    def test_default_config(self) -> None:
        """Test default search space configuration."""
        config = SearchSpaceConfig.default()

        assert config.min_layers == 4
        assert config.max_layers == 32
        assert len(config.allowed_components) > 0

    def test_custom_config(self) -> None:
        """Test custom search space configuration."""
        config = SearchSpaceConfig(
            min_layers=8,
            max_layers=16,
            channel_sizes=[128, 256],
            max_parameters=500_000_000,
        )

        assert config.min_layers == 8
        assert config.max_layers == 16
        assert 128 in config.channel_sizes

    def test_genome_validation_layer_count(self) -> None:
        """Test genome validation against layer count constraints."""
        config = SearchSpaceConfig(min_layers=4, max_layers=8)

        # Valid genome
        layers = [
            LayerConfig(
                component_type=ArchitectureComponent.CONV_BLOCK,
                parameters={"kernel_size": 3, "stride": 1, "padding": 1},
                layer_index=i,
                in_channels=64,
                out_channels=64,
            )
            for i in range(6)
        ]
        genome = ArchitectureGenome(layers=layers)
        assert config.validate_genome(genome)

        # Too few layers
        genome_small = ArchitectureGenome(layers=layers[:2])
        assert not config.validate_genome(genome_small)

        # Too many layers
        layers_large = [
            LayerConfig(
                component_type=ArchitectureComponent.CONV_BLOCK,
                parameters={"kernel_size": 3, "stride": 1, "padding": 1},
                layer_index=i,
                in_channels=64,
                out_channels=64,
            )
            for i in range(20)
        ]
        genome_large = ArchitectureGenome(layers=layers_large)
        assert not config.validate_genome(genome_large)

    def test_genome_validation_component_types(self) -> None:
        """Test genome validation against allowed component types."""
        config = SearchSpaceConfig(
            allowed_components=[
                ArchitectureComponent.CONV_BLOCK,
                ArchitectureComponent.RESIDUAL_BLOCK,
            ]
        )

        # Valid genome (only allowed components)
        layers = [
            LayerConfig(
                component_type=ArchitectureComponent.CONV_BLOCK,
                parameters={"kernel_size": 3, "stride": 1, "padding": 1},
                layer_index=i,
                in_channels=64,
                out_channels=64,
            )
            for i in range(4)
        ]
        genome = ArchitectureGenome(layers=layers)
        assert config.validate_genome(genome)

        # Invalid genome (contains disallowed component)
        layers_invalid = layers + [
            LayerConfig(
                component_type=ArchitectureComponent.ATTENTION_BLOCK,
                parameters={"num_heads": 8, "embed_dim": 512},
                layer_index=4,
                in_channels=64,
                out_channels=64,
            )
        ]
        genome_invalid = ArchitectureGenome(layers=layers_invalid)
        assert not config.validate_genome(genome_invalid)

    def test_parameter_count_constraint(self) -> None:
        """Test parameter count constraint validation."""
        config = SearchSpaceConfig(max_parameters=100_000)

        # Small genome (should pass)
        layers_small = [
            LayerConfig(
                component_type=ArchitectureComponent.CONV_BLOCK,
                parameters={"kernel_size": 3, "stride": 1, "padding": 1},
                layer_index=i,
                in_channels=8,
                out_channels=8,
            )
            for i in range(4)
        ]
        genome_small = ArchitectureGenome(layers=layers_small)
        assert config.validate_genome(genome_small)

        # Large genome (should fail)
        layers_large = [
            LayerConfig(
                component_type=ArchitectureComponent.CONV_BLOCK,
                parameters={"kernel_size": 7, "stride": 1, "padding": 3},
                layer_index=i,
                in_channels=512,
                out_channels=512,
            )
            for i in range(20)
        ]
        genome_large = ArchitectureGenome(layers=layers_large)
        assert not config.validate_genome(genome_large)


class TestArchitectureComponents:
    """Tests for ArchitectureComponent enum."""

    def test_component_enum(self) -> None:
        """Test architecture component enumeration."""
        assert ArchitectureComponent.ATTENTION_BLOCK.value == "attention"
        assert ArchitectureComponent.CONV_BLOCK.value == "conv"
        assert ArchitectureComponent.RESIDUAL_BLOCK.value == "residual"

    def test_component_iteration(self) -> None:
        """Test iterating over all components."""
        components = list(ArchitectureComponent)
        assert len(components) >= 5
        assert ArchitectureComponent.ATTENTION_BLOCK in components
