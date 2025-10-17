"""
Population generation for evolutionary architecture search.

This module implements population initialization with diverse
architecture sampling and population management utilities.
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np

from evolutionary_search.search_space import (
    ArchitectureComponent,
    ArchitectureGenome,
    LayerConfig,
    SearchSpaceConfig,
)

__all__ = [
    "PopulationGenerator",
    "DiversityMetrics",
]


class PopulationGenerator:
    """
    Generates initial population for evolutionary search.

    Implements diverse sampling strategies to ensure good coverage
    of the search space.
    """

    def __init__(self, search_space: SearchSpaceConfig, seed: int | None = None):
        """
        Initialize population generator.

        Args:
            search_space: Search space configuration
            seed: Random seed for reproducibility
        """
        self.search_space = search_space
        self.rng = np.random.RandomState(seed)
        random.seed(seed)

    def generate_random_genome(self) -> ArchitectureGenome:
        """
        Generate a random architecture genome.

        Returns:
            Random but valid ArchitectureGenome
        """
        # Random number of layers
        num_layers = self.rng.randint(
            self.search_space.min_layers, self.search_space.max_layers + 1
        )

        # Generate layers
        layers: list[LayerConfig] = []
        for i in range(num_layers):
            component_type = random.choice(self.search_space.allowed_components)
            parameters = self._generate_layer_parameters(component_type)

            # Set channels
            if i == 0:
                in_channels = 3  # RGB input
            else:
                in_channels = layers[-1].out_channels

            out_channels = random.choice(self.search_space.channel_sizes)

            layer = LayerConfig(
                component_type=component_type,
                parameters=parameters,
                layer_index=i,
                in_channels=in_channels,
                out_channels=out_channels,
            )
            layers.append(layer)

        # Generate connections (sequential by default, can add skip connections)
        connections = [(i, i + 1) for i in range(num_layers - 1)]

        # Add some skip connections randomly
        if num_layers > 4 and self.rng.random() > 0.5:
            num_skips = self.rng.randint(1, max(2, num_layers // 4))
            for _ in range(num_skips):
                from_idx = self.rng.randint(0, num_layers - 2)
                to_idx = self.rng.randint(from_idx + 2, num_layers)
                connections.append((from_idx, to_idx))

        genome = ArchitectureGenome(
            layers=layers,
            connections=connections,
            parameters={"architecture_type": "random_sampled"},
            generation=0,
        )

        return genome

    def generate_population(self, size: int) -> list[ArchitectureGenome]:
        """
        Generate initial population.

        Args:
            size: Population size

        Returns:
            List of diverse architecture genomes
        """
        population: list[ArchitectureGenome] = []

        # Strategy 1: Random sampling (60%)
        num_random = int(size * 0.6)
        for _ in range(num_random):
            genome = self.generate_random_genome()
            if self.search_space.validate_genome(genome):
                population.append(genome)

        # Strategy 2: Small architectures (20%)
        num_small = int(size * 0.2)
        for _ in range(num_small):
            genome = self._generate_small_architecture()
            if self.search_space.validate_genome(genome):
                population.append(genome)

        # Strategy 3: Large architectures (20%)
        num_large = int(size * 0.2)
        for _ in range(num_large):
            genome = self._generate_large_architecture()
            if self.search_space.validate_genome(genome):
                population.append(genome)

        # Fill up to target size if needed
        while len(population) < size:
            genome = self.generate_random_genome()
            if self.search_space.validate_genome(genome):
                population.append(genome)

        return population[:size]

    def _generate_layer_parameters(
        self, component_type: ArchitectureComponent
    ) -> dict[str, Any]:
        """Generate parameters for a specific component type."""
        params: dict[str, Any] = {}

        if component_type == ArchitectureComponent.CONV_BLOCK:
            params = {
                "kernel_size": random.choice(self.search_space.kernel_sizes),
                "stride": random.choice([1, 2]),
                "padding": random.choice([0, 1, 2]),
                "groups": 1,
            }
        elif component_type == ArchitectureComponent.ATTENTION_BLOCK:
            params = {
                "num_heads": random.choice(self.search_space.attention_heads),
                "embed_dim": random.choice(self.search_space.channel_sizes),
                "dropout": self.rng.uniform(0.0, 0.3),
            }
        elif component_type == ArchitectureComponent.RESIDUAL_BLOCK:
            params = {
                "kernel_size": 3,
                "num_layers": random.choice([2, 3]),
            }
        elif component_type == ArchitectureComponent.NORMALIZATION:
            params = {
                "norm_type": random.choice(["batch", "layer", "group"]),
                "eps": 1e-5,
            }
        elif component_type == ArchitectureComponent.ACTIVATION:
            params = {
                "activation_type": random.choice(["relu", "gelu", "silu"]),
            }
        elif component_type == ArchitectureComponent.TIMESTEP_EMBEDDING:
            params = {
                "embed_dim": random.choice(self.search_space.channel_sizes),
            }
        elif component_type == ArchitectureComponent.CROSS_ATTENTION:
            params = {
                "num_heads": random.choice(self.search_space.attention_heads),
                "embed_dim": random.choice(self.search_space.channel_sizes),
            }

        return params

    def _generate_small_architecture(self) -> ArchitectureGenome:
        """Generate a small, efficient architecture."""
        num_layers = self.search_space.min_layers + 2
        layers: list[LayerConfig] = []

        for i in range(num_layers):
            # Prefer conv blocks for small architectures
            component_type = random.choice(
                [
                    ArchitectureComponent.CONV_BLOCK,
                    ArchitectureComponent.RESIDUAL_BLOCK,
                ]
            )
            parameters = self._generate_layer_parameters(component_type)

            in_channels = 3 if i == 0 else layers[-1].out_channels
            out_channels = min(self.search_space.channel_sizes)  # Use small channels

            layer = LayerConfig(
                component_type=component_type,
                parameters=parameters,
                layer_index=i,
                in_channels=in_channels,
                out_channels=out_channels,
            )
            layers.append(layer)

        connections = [(i, i + 1) for i in range(num_layers - 1)]

        return ArchitectureGenome(
            layers=layers,
            connections=connections,
            parameters={"architecture_type": "small"},
            generation=0,
        )

    def _generate_large_architecture(self) -> ArchitectureGenome:
        """Generate a large, high-capacity architecture."""
        num_layers = self.search_space.max_layers - 2
        layers: list[LayerConfig] = []

        for i in range(num_layers):
            # Include more attention blocks for large architectures
            if i % 4 == 0:
                component_type = ArchitectureComponent.ATTENTION_BLOCK
            else:
                component_type = random.choice(
                    [
                        ArchitectureComponent.CONV_BLOCK,
                        ArchitectureComponent.RESIDUAL_BLOCK,
                    ]
                )

            parameters = self._generate_layer_parameters(component_type)

            in_channels = 3 if i == 0 else layers[-1].out_channels
            out_channels = max(self.search_space.channel_sizes)  # Use large channels

            layer = LayerConfig(
                component_type=component_type,
                parameters=parameters,
                layer_index=i,
                in_channels=in_channels,
                out_channels=out_channels,
            )
            layers.append(layer)

        connections = [(i, i + 1) for i in range(num_layers - 1)]

        # Add more skip connections
        num_skips = num_layers // 3
        for _ in range(num_skips):
            from_idx = self.rng.randint(0, num_layers - 2)
            to_idx = self.rng.randint(from_idx + 2, num_layers)
            connections.append((from_idx, to_idx))

        return ArchitectureGenome(
            layers=layers,
            connections=connections,
            parameters={"architecture_type": "large"},
            generation=0,
        )


class DiversityMetrics:
    """Compute diversity metrics for a population."""

    @staticmethod
    def compute_diversity(population: list[ArchitectureGenome]) -> float:
        """
        Compute diversity score for population.

        Args:
            population: List of genomes

        Returns:
            Diversity score (0-1, higher is more diverse)
        """
        if len(population) < 2:
            return 0.0

        # Compute pairwise distances
        distances: list[float] = []
        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                dist = DiversityMetrics._genome_distance(
                    population[i], population[j]
                )
                distances.append(dist)

        # Average distance as diversity metric
        return float(np.mean(distances)) if distances else 0.0

    @staticmethod
    def _genome_distance(genome1: ArchitectureGenome, genome2: ArchitectureGenome) -> float:
        """
        Compute distance between two genomes.

        Returns:
            Distance score (0-1)
        """
        # Layer count difference
        layer_diff = abs(len(genome1.layers) - len(genome2.layers)) / 32.0

        # Connection difference
        conn_diff = abs(len(genome1.connections) - len(genome2.connections)) / 32.0

        # Parameter count difference
        param_diff = (
            abs(genome1.count_parameters() - genome2.count_parameters()) / 1e9
        )

        # Component type distribution difference
        types1 = [layer.component_type for layer in genome1.layers]
        types2 = [layer.component_type for layer in genome2.layers]

        type_diff = 0.0
        for comp_type in ArchitectureComponent:
            count1 = types1.count(comp_type) / max(len(types1), 1)
            count2 = types2.count(comp_type) / max(len(types2), 1)
            type_diff += abs(count1 - count2)

        type_diff /= len(ArchitectureComponent)

        # Weighted combination
        distance = 0.3 * layer_diff + 0.2 * conn_diff + 0.3 * param_diff + 0.2 * type_diff

        return min(1.0, distance)

    @staticmethod
    def compute_novelty(
        genome: ArchitectureGenome, population: list[ArchitectureGenome], k: int = 5
    ) -> float:
        """
        Compute novelty score for a genome relative to population.

        Args:
            genome: Genome to evaluate
            population: Reference population
            k: Number of nearest neighbors to consider

        Returns:
            Novelty score (higher means more novel)
        """
        if len(population) < k:
            return 1.0

        # Compute distances to all population members
        distances = [
            DiversityMetrics._genome_distance(genome, other) for other in population
        ]

        # Sort and take k nearest
        distances_sorted = sorted(distances)
        k_nearest = distances_sorted[:k]

        # Average distance to k nearest neighbors
        return float(np.mean(k_nearest))
