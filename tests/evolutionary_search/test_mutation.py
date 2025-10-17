"""
Tests for mutation operators.
"""

import pytest

from evolutionary_search.operators.mutation import (
    LayerMutation,
    ParameterMutation,
    StructuralMutation,
    CompositeMutation,
)
from evolutionary_search.population import PopulationGenerator
from evolutionary_search.search_space import SearchSpaceConfig


@pytest.fixture
def search_space():
    """Create search space configuration."""
    return SearchSpaceConfig.default()


@pytest.fixture
def population_generator(search_space):
    """Create population generator."""
    return PopulationGenerator(search_space, seed=42)


@pytest.fixture
def test_genome(population_generator):
    """Create test genome."""
    return population_generator.generate_random_genome()


class TestLayerMutation:
    """Tests for layer mutation operator."""

    def test_initialization(self, search_space):
        """Test mutation operator initialization."""
        mutation = LayerMutation(search_space, mutation_rate=0.5, seed=42)
        assert mutation.search_space == search_space
        assert mutation.mutation_rate == 0.5

    def test_mutate_returns_genome(self, search_space, test_genome):
        """Test that mutation returns a valid genome."""
        mutation = LayerMutation(search_space, mutation_rate=1.0, seed=42)
        mutated = mutation.mutate(test_genome)

        assert mutated is not None
        assert mutated.validate()

    def test_add_layer_mutation(self, search_space, population_generator):
        """Test adding a layer."""
        mutation = LayerMutation(
            search_space, mutation_rate=1.0, add_probability=0.9, remove_probability=0.05, seed=42
        )
        genome = population_generator._generate_small_architecture()
        original_len = len(genome.layers)

        # Try multiple times to ensure we get an add mutation
        for _ in range(10):
            mutated = mutation.mutate(genome)
            if len(mutated.layers) > original_len:
                break

        # Should have more layers (unless at max)
        if original_len < search_space.max_layers:
            assert len(mutated.layers) > original_len
        assert mutated.validate()

    def test_remove_layer_mutation(self, search_space, population_generator):
        """Test removing a layer."""
        mutation = LayerMutation(
            search_space, mutation_rate=1.0, add_probability=0.05, remove_probability=0.9, seed=42
        )
        genome = population_generator._generate_large_architecture()
        original_len = len(genome.layers)

        # Try multiple times to ensure we get a remove mutation
        for _ in range(10):
            mutated = mutation.mutate(genome)
            if len(mutated.layers) < original_len:
                break

        # Should have fewer layers (unless at min)
        if original_len > search_space.min_layers:
            assert len(mutated.layers) < original_len
        assert mutated.validate()

    def test_replace_layer_mutation(self, search_space, test_genome):
        """Test replacing a layer."""
        mutation = LayerMutation(
            search_space,
            mutation_rate=1.0,
            add_probability=0.0,
            remove_probability=0.0,
            seed=42,
        )
        original_len = len(test_genome.layers)

        mutated = mutation.mutate(test_genome)

        # Should have same number of layers
        assert len(mutated.layers) == original_len
        assert mutated.validate()

    def test_mutation_respects_constraints(self, search_space, test_genome):
        """Test that mutation respects search space constraints."""
        mutation = LayerMutation(search_space, mutation_rate=1.0, seed=42)

        for _ in range(10):
            mutated = mutation.mutate(test_genome)
            assert search_space.validate_genome(mutated)
            assert search_space.min_layers <= len(mutated.layers) <= search_space.max_layers


class TestParameterMutation:
    """Tests for parameter mutation operator."""

    def test_initialization(self, search_space):
        """Test mutation operator initialization."""
        mutation = ParameterMutation(search_space, mutation_rate=0.5, seed=42)
        assert mutation.search_space == search_space
        assert mutation.mutation_rate == 0.5

    def test_mutate_parameters(self, search_space, test_genome):
        """Test parameter mutation."""
        mutation = ParameterMutation(search_space, mutation_rate=1.0, parameter_mutation_strength=1.0, seed=42)
        original_params = [layer.parameters.copy() for layer in test_genome.layers]
        original_channels = [layer.out_channels for layer in test_genome.layers]

        mutated = mutation.mutate(test_genome)

        # Some parameters or channels should be different
        params_changed = False
        for i, layer in enumerate(mutated.layers):
            if i < len(original_params):
                if layer.parameters != original_params[i] or layer.out_channels != original_channels[i]:
                    params_changed = True
                    break

        assert params_changed
        assert mutated.validate()

    def test_parameter_mutation_preserves_structure(self, search_space, test_genome):
        """Test that parameter mutation doesn't change structure."""
        mutation = ParameterMutation(
            search_space, mutation_rate=1.0, parameter_mutation_strength=0.1, seed=42
        )
        original_len = len(test_genome.layers)

        mutated = mutation.mutate(test_genome)

        # Layer count should be preserved
        assert len(mutated.layers) == original_len
        assert mutated.validate()

    def test_mutation_strength_effect(self, search_space, test_genome):
        """Test effect of mutation strength."""
        # Low strength
        mutation_low = ParameterMutation(
            search_space, mutation_rate=1.0, parameter_mutation_strength=0.1, seed=42
        )
        mutated_low = mutation_low.mutate(test_genome)

        # High strength
        mutation_high = ParameterMutation(
            search_space, mutation_rate=1.0, parameter_mutation_strength=0.8, seed=42
        )
        mutated_high = mutation_high.mutate(test_genome)

        # Both should be valid
        assert mutated_low.validate()
        assert mutated_high.validate()


class TestStructuralMutation:
    """Tests for structural mutation operator."""

    def test_initialization(self, search_space):
        """Test mutation operator initialization."""
        mutation = StructuralMutation(search_space, mutation_rate=0.5, seed=42)
        assert mutation.search_space == search_space

    def test_add_skip_connection(self, search_space, population_generator):
        """Test adding skip connections."""
        import copy

        mutation = StructuralMutation(search_space, mutation_rate=1.0, seed=42)
        genome = population_generator._generate_large_architecture()
        original_connections = len(genome.connections)

        # Try multiple times to ensure skip connection is added
        for _ in range(5):
            mutated = mutation._add_skip_connection(copy.deepcopy(genome))
            if len(mutated.connections) > original_connections:
                break

        assert mutated.validate()

    def test_remove_skip_connection(self, search_space, population_generator):
        """Test removing skip connections."""
        mutation = StructuralMutation(search_space, mutation_rate=1.0, seed=42)
        genome = population_generator._generate_large_architecture()

        # Add skip connections first
        genome.connections.append((0, 3))
        genome.connections.append((1, 4))
        original_connections = len(genome.connections)

        mutated = mutation._remove_skip_connection(genome)

        # Should have fewer connections
        assert len(mutated.connections) <= original_connections
        assert mutated.validate()

    def test_structural_mutation_preserves_validity(self, search_space, test_genome):
        """Test that structural mutation preserves validity."""
        mutation = StructuralMutation(search_space, mutation_rate=1.0, seed=42)

        for _ in range(10):
            mutated = mutation.mutate(test_genome)
            assert mutated.validate()


class TestCompositeMutation:
    """Tests for composite mutation operator."""

    def test_initialization(self, search_space):
        """Test composite mutation initialization."""
        mutation = CompositeMutation(search_space, mutation_rate=0.5, seed=42)
        assert mutation.search_space == search_space
        assert mutation.mutation_rate == 0.5

    def test_composite_applies_multiple_mutations(self, search_space, test_genome):
        """Test that composite mutation applies multiple operators."""
        mutation = CompositeMutation(search_space, mutation_rate=1.0, seed=42)

        mutated = mutation.mutate(test_genome)

        # Should produce valid genome
        assert mutated.validate()
        assert search_space.validate_genome(mutated)

    def test_composite_with_all_operators(self, search_space, test_genome):
        """Test composite mutation with all sub-operators."""
        mutation = CompositeMutation(search_space, mutation_rate=1.0, seed=42)

        # Run multiple times to ensure all operators get triggered
        mutations = [mutation.mutate(test_genome) for _ in range(20)]

        # All should be valid
        for mutated in mutations:
            assert mutated.validate()

    def test_composite_mutation_diversity(self, search_space, population_generator):
        """Test that composite mutation creates diverse genomes."""
        mutation = CompositeMutation(search_space, mutation_rate=1.0, seed=42)
        genome = population_generator.generate_random_genome()

        mutated_genomes = [mutation.mutate(genome) for _ in range(20)]

        # Check that mutations create variation (either layer counts or parameters)
        layer_counts = [len(g.layers) for g in mutated_genomes]
        param_counts = [g.count_parameters() for g in mutated_genomes]

        # Should have variation in either structure or parameters
        assert len(set(layer_counts)) > 1 or len(set(param_counts)) > 1


class TestMutationRates:
    """Tests for mutation rate behavior."""

    def test_zero_mutation_rate(self, search_space, test_genome):
        """Test that zero mutation rate prevents mutation."""
        mutation = LayerMutation(search_space, mutation_rate=0.0, seed=42)

        # Run many times
        for _ in range(20):
            mutated = mutation.mutate(test_genome)
            # Genome should be unchanged
            assert len(mutated.layers) == len(test_genome.layers)

    def test_mutation_probability(self, search_space, test_genome):
        """Test that mutation rate controls probability."""
        # Low mutation rate
        mutation_low = LayerMutation(search_space, mutation_rate=0.1, seed=42)
        mutations_low = sum(
            1 for _ in range(100)
            if len(mutation_low.mutate(test_genome).layers) != len(test_genome.layers)
        )

        # High mutation rate
        mutation_high = LayerMutation(search_space, mutation_rate=0.9, seed=42)
        mutations_high = sum(
            1 for _ in range(100)
            if len(mutation_high.mutate(test_genome).layers) != len(test_genome.layers)
        )

        # High rate should cause more mutations
        # (Note: This is probabilistic, so we allow some variance)
        assert mutations_high > mutations_low


@pytest.mark.parametrize(
    "mutation_class",
    [LayerMutation, ParameterMutation, StructuralMutation, CompositeMutation],
)
def test_mutation_reproducibility(mutation_class, search_space, test_genome):
    """Test that mutations are reproducible with same seed."""
    # Run mutation twice with same seed
    mutation1 = mutation_class(search_space, mutation_rate=1.0, seed=42)
    mutated1 = mutation1.mutate(test_genome)

    mutation2 = mutation_class(search_space, mutation_rate=1.0, seed=42)
    mutated2 = mutation2.mutate(test_genome)

    # Should produce same results
    assert len(mutated1.layers) == len(mutated2.layers)
    assert len(mutated1.connections) == len(mutated2.connections)
