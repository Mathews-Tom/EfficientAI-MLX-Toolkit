"""
Mutation operators for evolutionary architecture search.

This module implements various mutation strategies for introducing
diversity and exploring the architecture search space.
"""

from __future__ import annotations

import copy
import random
from abc import ABC, abstractmethod

import numpy as np

from evolutionary_search.search_space import (
    ArchitectureComponent,
    ArchitectureGenome,
    LayerConfig,
    SearchSpaceConfig,
)

__all__ = [
    "MutationOperator",
    "LayerMutation",
    "ParameterMutation",
    "StructuralMutation",
    "CompositeMutation",
]


class MutationOperator(ABC):
    """Base class for mutation operators."""

    def __init__(
        self,
        search_space: SearchSpaceConfig,
        mutation_rate: float = 0.1,
        seed: int | None = None,
    ):
        """
        Initialize mutation operator.

        Args:
            search_space: Search space configuration
            mutation_rate: Probability of mutation per genome
            seed: Random seed for reproducibility
        """
        self.search_space = search_space
        self.mutation_rate = mutation_rate
        self.rng = np.random.RandomState(seed)
        random.seed(seed)

    @abstractmethod
    def mutate(self, genome: ArchitectureGenome) -> ArchitectureGenome:
        """
        Mutate a genome.

        Args:
            genome: Genome to mutate

        Returns:
            Mutated genome
        """
        pass

    def _should_mutate(self) -> bool:
        """Determine if mutation should occur."""
        return self.rng.random() < self.mutation_rate


class LayerMutation(MutationOperator):
    """
    Layer-level mutation operator.

    Adds, removes, or replaces layers in the architecture.
    """

    def __init__(
        self,
        search_space: SearchSpaceConfig,
        mutation_rate: float = 0.2,
        add_probability: float = 0.33,
        remove_probability: float = 0.33,
        seed: int | None = None,
    ):
        """
        Initialize layer mutation.

        Args:
            search_space: Search space configuration
            mutation_rate: Overall mutation probability
            add_probability: Probability of adding a layer
            remove_probability: Probability of removing a layer
            seed: Random seed
        """
        super().__init__(search_space, mutation_rate, seed)
        self.add_probability = add_probability
        self.remove_probability = remove_probability
        self.replace_probability = 1.0 - add_probability - remove_probability

    def mutate(self, genome: ArchitectureGenome) -> ArchitectureGenome:
        """Mutate layers in genome."""
        if not self._should_mutate():
            return genome

        mutated = copy.deepcopy(genome)

        # Normalize probabilities
        total = self.add_probability + self.remove_probability + self.replace_probability
        if total <= 0:
            total = 1.0

        norm_probs = [
            self.add_probability / total,
            self.remove_probability / total,
            self.replace_probability / total,
        ]

        # Choose mutation type
        mutation_type = self.rng.choice(
            ["add", "remove", "replace"],
            p=norm_probs,
        )

        if mutation_type == "add":
            mutated = self._add_layer(mutated)
        elif mutation_type == "remove":
            mutated = self._remove_layer(mutated)
        else:
            mutated = self._replace_layer(mutated)

        return mutated

    def _add_layer(self, genome: ArchitectureGenome) -> ArchitectureGenome:
        """Add a random layer to genome."""
        if len(genome.layers) >= self.search_space.max_layers:
            return genome

        # Choose insertion point
        insert_idx = self.rng.randint(0, len(genome.layers))

        # Generate new layer
        from evolutionary_search.population import PopulationGenerator

        gen = PopulationGenerator(self.search_space, seed=None)
        component_type = random.choice(self.search_space.allowed_components)
        parameters = gen._generate_layer_parameters(component_type)

        # Determine channels
        if insert_idx == 0:
            in_channels = 3
            out_channels = genome.layers[0].in_channels or 64
        elif insert_idx == len(genome.layers):
            in_channels = genome.layers[-1].out_channels or 64
            out_channels = random.choice(self.search_space.channel_sizes)
        else:
            in_channels = genome.layers[insert_idx - 1].out_channels or 64
            out_channels = genome.layers[insert_idx].in_channels or 64

        new_layer = LayerConfig(
            component_type=component_type,
            parameters=parameters,
            layer_index=insert_idx,
            in_channels=in_channels,
            out_channels=out_channels,
        )

        # Insert layer
        genome.layers.insert(insert_idx, new_layer)

        # Update indices
        for i, layer in enumerate(genome.layers):
            layer.layer_index = i

        # Update connections
        new_connections = []
        for from_idx, to_idx in genome.connections:
            if from_idx >= insert_idx:
                from_idx += 1
            if to_idx >= insert_idx:
                to_idx += 1
            new_connections.append((from_idx, to_idx))

        # Add connections for new layer
        if insert_idx > 0:
            new_connections.append((insert_idx - 1, insert_idx))
        if insert_idx < len(genome.layers) - 1:
            new_connections.append((insert_idx, insert_idx + 1))

        genome.connections = new_connections

        return genome

    def _remove_layer(self, genome: ArchitectureGenome) -> ArchitectureGenome:
        """Remove a random layer from genome."""
        if len(genome.layers) <= self.search_space.min_layers:
            return genome

        # Choose removal index
        remove_idx = self.rng.randint(0, len(genome.layers))

        # Remove layer
        genome.layers.pop(remove_idx)

        # Update indices
        for i, layer in enumerate(genome.layers):
            layer.layer_index = i

        # Update connections
        new_connections = []
        for from_idx, to_idx in genome.connections:
            if from_idx == remove_idx or to_idx == remove_idx:
                continue

            if from_idx > remove_idx:
                from_idx -= 1
            if to_idx > remove_idx:
                to_idx -= 1

            if from_idx < to_idx:
                new_connections.append((from_idx, to_idx))

        # Ensure sequential connectivity
        if not new_connections:
            new_connections = [(i, i + 1) for i in range(len(genome.layers) - 1)]

        genome.connections = new_connections

        return genome

    def _replace_layer(self, genome: ArchitectureGenome) -> ArchitectureGenome:
        """Replace a random layer in genome."""
        if not genome.layers:
            return genome

        # Choose replacement index
        replace_idx = self.rng.randint(0, len(genome.layers))

        # Generate replacement layer
        from evolutionary_search.population import PopulationGenerator

        gen = PopulationGenerator(self.search_space, seed=None)
        component_type = random.choice(self.search_space.allowed_components)
        parameters = gen._generate_layer_parameters(component_type)

        # Preserve channels
        old_layer = genome.layers[replace_idx]
        new_layer = LayerConfig(
            component_type=component_type,
            parameters=parameters,
            layer_index=replace_idx,
            in_channels=old_layer.in_channels,
            out_channels=old_layer.out_channels,
        )

        genome.layers[replace_idx] = new_layer

        return genome


class ParameterMutation(MutationOperator):
    """
    Parameter-level mutation operator.

    Modifies layer parameters (kernel sizes, attention heads, etc.)
    without changing structure.
    """

    def __init__(
        self,
        search_space: SearchSpaceConfig,
        mutation_rate: float = 0.3,
        parameter_mutation_strength: float = 0.2,
        seed: int | None = None,
    ):
        """
        Initialize parameter mutation.

        Args:
            search_space: Search space configuration
            mutation_rate: Overall mutation probability
            parameter_mutation_strength: Proportion of parameters to mutate
            seed: Random seed
        """
        super().__init__(search_space, mutation_rate, seed)
        self.parameter_mutation_strength = parameter_mutation_strength

    def mutate(self, genome: ArchitectureGenome) -> ArchitectureGenome:
        """Mutate layer parameters."""
        if not self._should_mutate():
            return genome

        mutated = copy.deepcopy(genome)

        # Determine number of layers to mutate
        num_mutations = max(1, int(len(mutated.layers) * self.parameter_mutation_strength))
        mutation_indices = self.rng.choice(
            len(mutated.layers), size=num_mutations, replace=False
        )

        for idx in mutation_indices:
            layer = mutated.layers[idx]
            layer.parameters = self._mutate_parameters(layer)

            # Potentially mutate channels
            if self.rng.random() < 0.3:
                layer.out_channels = random.choice(self.search_space.channel_sizes)

        return mutated

    def _mutate_parameters(self, layer: LayerConfig) -> dict:
        """Mutate parameters for a specific layer."""
        from evolutionary_search.population import PopulationGenerator

        gen = PopulationGenerator(self.search_space, seed=None)

        if self.rng.random() < 0.5:
            # Generate completely new parameters
            return gen._generate_layer_parameters(layer.component_type)
        else:
            # Modify existing parameters
            mutated_params = copy.deepcopy(layer.parameters)

            if layer.component_type == ArchitectureComponent.CONV_BLOCK:
                if "kernel_size" in mutated_params and self.rng.random() < 0.5:
                    mutated_params["kernel_size"] = random.choice(
                        self.search_space.kernel_sizes
                    )
                if "stride" in mutated_params and self.rng.random() < 0.3:
                    mutated_params["stride"] = random.choice([1, 2])

            elif layer.component_type == ArchitectureComponent.ATTENTION_BLOCK:
                if "num_heads" in mutated_params and self.rng.random() < 0.5:
                    mutated_params["num_heads"] = random.choice(
                        self.search_space.attention_heads
                    )
                if "embed_dim" in mutated_params and self.rng.random() < 0.3:
                    mutated_params["embed_dim"] = random.choice(
                        self.search_space.channel_sizes
                    )

            return mutated_params


class StructuralMutation(MutationOperator):
    """
    Structural mutation operator.

    Modifies connections and skip connections in the architecture.
    """

    def __init__(
        self,
        search_space: SearchSpaceConfig,
        mutation_rate: float = 0.15,
        seed: int | None = None,
    ):
        """
        Initialize structural mutation.

        Args:
            search_space: Search space configuration
            mutation_rate: Mutation probability
            seed: Random seed
        """
        super().__init__(search_space, mutation_rate, seed)

    def mutate(self, genome: ArchitectureGenome) -> ArchitectureGenome:
        """Mutate connections in genome."""
        if not self._should_mutate():
            return genome

        mutated = copy.deepcopy(genome)

        # Choose mutation type
        mutation_type = self.rng.choice(["add_skip", "remove_skip", "modify_connection"])

        if mutation_type == "add_skip":
            mutated = self._add_skip_connection(mutated)
        elif mutation_type == "remove_skip":
            mutated = self._remove_skip_connection(mutated)
        else:
            mutated = self._modify_connection(mutated)

        return mutated

    def _add_skip_connection(self, genome: ArchitectureGenome) -> ArchitectureGenome:
        """Add a skip connection."""
        if len(genome.layers) < 3:
            return genome

        # Find valid skip connection
        max_attempts = 10
        for _ in range(max_attempts):
            from_idx = self.rng.randint(0, len(genome.layers) - 2)
            to_idx = self.rng.randint(from_idx + 2, len(genome.layers))

            connection = (from_idx, to_idx)
            if connection not in genome.connections:
                genome.connections.append(connection)
                break

        return genome

    def _remove_skip_connection(self, genome: ArchitectureGenome) -> ArchitectureGenome:
        """Remove a skip connection."""
        # Find skip connections (non-sequential)
        skip_connections = [
            (f, t) for f, t in genome.connections if t != f + 1
        ]

        if skip_connections:
            remove_conn = random.choice(skip_connections)
            genome.connections.remove(remove_conn)

        return genome

    def _modify_connection(self, genome: ArchitectureGenome) -> ArchitectureGenome:
        """Modify an existing connection."""
        if not genome.connections:
            return genome

        # Choose random connection
        modify_idx = self.rng.randint(0, len(genome.connections))
        from_idx, to_idx = genome.connections[modify_idx]

        # Modify target
        new_to_idx = self.rng.randint(from_idx + 1, len(genome.layers))
        genome.connections[modify_idx] = (from_idx, new_to_idx)

        return genome


class CompositeMutation(MutationOperator):
    """
    Composite mutation operator.

    Combines multiple mutation operators with weighted probabilities.
    """

    def __init__(
        self,
        search_space: SearchSpaceConfig,
        mutation_rate: float = 0.3,
        seed: int | None = None,
    ):
        """
        Initialize composite mutation.

        Args:
            search_space: Search space configuration
            mutation_rate: Overall mutation probability
            seed: Random seed
        """
        super().__init__(search_space, mutation_rate, seed)

        # Initialize component operators
        self.layer_mutation = LayerMutation(search_space, mutation_rate=0.2, seed=seed)
        self.parameter_mutation = ParameterMutation(
            search_space, mutation_rate=0.5, seed=seed
        )
        self.structural_mutation = StructuralMutation(
            search_space, mutation_rate=0.3, seed=seed
        )

        self.operators = [
            (self.layer_mutation, 0.3),
            (self.parameter_mutation, 0.5),
            (self.structural_mutation, 0.2),
        ]

    def mutate(self, genome: ArchitectureGenome) -> ArchitectureGenome:
        """Apply composite mutation."""
        if not self._should_mutate():
            return genome

        mutated = copy.deepcopy(genome)

        # Apply multiple mutations
        num_mutations = self.rng.choice([1, 2, 3], p=[0.6, 0.3, 0.1])

        for _ in range(num_mutations):
            # Select mutation operator by weight
            operators, weights = zip(*self.operators)
            selected_op = self.rng.choice(operators, p=weights)
            mutated = selected_op.mutate(mutated)

        return mutated
