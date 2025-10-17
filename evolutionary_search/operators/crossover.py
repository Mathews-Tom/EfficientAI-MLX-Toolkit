"""
Crossover operators for evolutionary architecture search.

This module implements various crossover strategies for combining
parent genomes to produce offspring architectures.
"""

from __future__ import annotations

import copy
import random
from abc import ABC, abstractmethod

import numpy as np

from evolutionary_search.search_space import (
    ArchitectureGenome,
    LayerConfig,
    SearchSpaceConfig,
)

__all__ = [
    "CrossoverOperator",
    "UniformCrossover",
    "LayerCrossover",
    "SinglePointCrossover",
]


class CrossoverOperator(ABC):
    """Base class for crossover operators."""

    def __init__(self, search_space: SearchSpaceConfig, seed: int | None = None):
        """
        Initialize crossover operator.

        Args:
            search_space: Search space configuration for validation
            seed: Random seed for reproducibility
        """
        self.search_space = search_space
        self.rng = np.random.RandomState(seed)
        random.seed(seed)

    @abstractmethod
    def crossover(
        self, parent1: ArchitectureGenome, parent2: ArchitectureGenome
    ) -> tuple[ArchitectureGenome, ArchitectureGenome]:
        """
        Perform crossover between two parent genomes.

        Args:
            parent1: First parent genome
            parent2: Second parent genome

        Returns:
            Tuple of (offspring1, offspring2)
        """
        pass

    def _repair_genome(self, genome: ArchitectureGenome) -> ArchitectureGenome:
        """
        Repair genome to satisfy constraints.

        Args:
            genome: Genome to repair

        Returns:
            Repaired genome
        """
        # Ensure minimum layers
        while len(genome.layers) < self.search_space.min_layers:
            # Add a random layer
            from evolutionary_search.population import PopulationGenerator

            gen = PopulationGenerator(self.search_space, seed=None)
            component_type = random.choice(self.search_space.allowed_components)
            parameters = gen._generate_layer_parameters(component_type)

            new_layer = LayerConfig(
                component_type=component_type,
                parameters=parameters,
                layer_index=len(genome.layers),
                in_channels=genome.layers[-1].out_channels if genome.layers else 3,
                out_channels=random.choice(self.search_space.channel_sizes),
            )
            genome.layers.append(new_layer)

        # Ensure maximum layers
        if len(genome.layers) > self.search_space.max_layers:
            genome.layers = genome.layers[: self.search_space.max_layers]

        # Update layer indices
        for i, layer in enumerate(genome.layers):
            layer.layer_index = i

        # Repair connections
        valid_connections = []
        max_idx = len(genome.layers) - 1
        for from_idx, to_idx in genome.connections:
            if 0 <= from_idx <= max_idx and 0 <= to_idx <= max_idx and from_idx < to_idx:
                valid_connections.append((from_idx, to_idx))

        # Ensure sequential connectivity
        if not valid_connections:
            valid_connections = [(i, i + 1) for i in range(len(genome.layers) - 1)]
        genome.connections = valid_connections

        return genome


class UniformCrossover(CrossoverOperator):
    """
    Uniform crossover operator.

    Each layer is randomly selected from either parent with equal probability.
    """

    def __init__(
        self,
        search_space: SearchSpaceConfig,
        swap_probability: float = 0.5,
        seed: int | None = None,
    ):
        """
        Initialize uniform crossover.

        Args:
            search_space: Search space configuration
            swap_probability: Probability of selecting from parent2
            seed: Random seed
        """
        super().__init__(search_space, seed)
        self.swap_probability = swap_probability

    def crossover(
        self, parent1: ArchitectureGenome, parent2: ArchitectureGenome
    ) -> tuple[ArchitectureGenome, ArchitectureGenome]:
        """Perform uniform crossover."""
        # Create offspring as copies
        offspring1 = copy.deepcopy(parent1)
        offspring2 = copy.deepcopy(parent2)

        # Determine crossover length
        min_len = min(len(parent1.layers), len(parent2.layers))
        max_len = max(len(parent1.layers), len(parent2.layers))

        # Uniform layer swap
        offspring1_layers = []
        offspring2_layers = []

        for i in range(max_len):
            if i < min_len:
                if self.rng.random() < self.swap_probability:
                    offspring1_layers.append(copy.deepcopy(parent2.layers[i]))
                    offspring2_layers.append(copy.deepcopy(parent1.layers[i]))
                else:
                    offspring1_layers.append(copy.deepcopy(parent1.layers[i]))
                    offspring2_layers.append(copy.deepcopy(parent2.layers[i]))
            else:
                # Handle different lengths
                if len(parent1.layers) > len(parent2.layers):
                    offspring1_layers.append(copy.deepcopy(parent1.layers[i]))
                else:
                    offspring2_layers.append(copy.deepcopy(parent2.layers[i]))

        offspring1.layers = offspring1_layers
        offspring2.layers = offspring2_layers

        # Inherit connections randomly
        if self.rng.random() < 0.5:
            offspring1.connections = copy.deepcopy(parent1.connections)
            offspring2.connections = copy.deepcopy(parent2.connections)
        else:
            offspring1.connections = copy.deepcopy(parent2.connections)
            offspring2.connections = copy.deepcopy(parent1.connections)

        # Repair and validate
        offspring1 = self._repair_genome(offspring1)
        offspring2 = self._repair_genome(offspring2)

        return offspring1, offspring2


class SinglePointCrossover(CrossoverOperator):
    """
    Single-point crossover operator.

    Splits genomes at a random point and swaps tails.
    """

    def crossover(
        self, parent1: ArchitectureGenome, parent2: ArchitectureGenome
    ) -> tuple[ArchitectureGenome, ArchitectureGenome]:
        """Perform single-point crossover."""
        # Create offspring as copies
        offspring1 = copy.deepcopy(parent1)
        offspring2 = copy.deepcopy(parent2)

        # Determine crossover point
        min_len = min(len(parent1.layers), len(parent2.layers))
        if min_len < 2:
            return offspring1, offspring2

        crossover_point = self.rng.randint(1, min_len)

        # Swap tails
        offspring1.layers = (
            parent1.layers[:crossover_point] + parent2.layers[crossover_point:]
        )
        offspring2.layers = (
            parent2.layers[:crossover_point] + parent1.layers[crossover_point:]
        )

        # Repair and validate
        offspring1 = self._repair_genome(offspring1)
        offspring2 = self._repair_genome(offspring2)

        return offspring1, offspring2


class LayerCrossover(CrossoverOperator):
    """
    Layer-wise crossover that preserves functional blocks.

    Identifies functional blocks (e.g., residual blocks, attention modules)
    and swaps them between parents.
    """

    def __init__(
        self,
        search_space: SearchSpaceConfig,
        block_size: int = 3,
        seed: int | None = None,
    ):
        """
        Initialize layer crossover.

        Args:
            search_space: Search space configuration
            block_size: Size of functional blocks to preserve
            seed: Random seed
        """
        super().__init__(search_space, seed)
        self.block_size = block_size

    def crossover(
        self, parent1: ArchitectureGenome, parent2: ArchitectureGenome
    ) -> tuple[ArchitectureGenome, ArchitectureGenome]:
        """Perform layer-wise block crossover."""
        # Create offspring as copies
        offspring1 = copy.deepcopy(parent1)
        offspring2 = copy.deepcopy(parent2)

        # Identify blocks in both parents
        blocks1 = self._identify_blocks(parent1.layers)
        blocks2 = self._identify_blocks(parent2.layers)

        # Randomly swap some blocks
        num_swaps = min(len(blocks1), len(blocks2))
        if num_swaps > 0:
            swap_indices = self.rng.choice(
                num_swaps, size=max(1, num_swaps // 2), replace=False
            )

            for idx in swap_indices:
                # Swap blocks
                blocks1[idx], blocks2[idx] = blocks2[idx], blocks1[idx]

        # Reconstruct layers
        offspring1.layers = [layer for block in blocks1 for layer in block]
        offspring2.layers = [layer for block in blocks2 for layer in block]

        # Repair and validate
        offspring1 = self._repair_genome(offspring1)
        offspring2 = self._repair_genome(offspring2)

        return offspring1, offspring2

    def _identify_blocks(self, layers: list[LayerConfig]) -> list[list[LayerConfig]]:
        """
        Identify functional blocks in layer sequence.

        Args:
            layers: List of layer configurations

        Returns:
            List of blocks (each block is a list of layers)
        """
        blocks: list[list[LayerConfig]] = []
        current_block: list[LayerConfig] = []

        for layer in layers:
            current_block.append(layer)

            # End block at certain conditions
            if (
                len(current_block) >= self.block_size
                or "attention" in layer.component_type.value
            ):
                blocks.append(copy.deepcopy(current_block))
                current_block = []

        # Add remaining layers as final block
        if current_block:
            blocks.append(current_block)

        return blocks
