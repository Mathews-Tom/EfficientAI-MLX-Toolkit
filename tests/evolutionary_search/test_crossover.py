"""
Tests for crossover operators.
"""

import pytest

from evolutionary_search.operators.crossover import (
    UniformCrossover,
    SinglePointCrossover,
    LayerCrossover,
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
def parent_genomes(population_generator):
    """Create parent genomes for testing."""
    return population_generator.generate_population(2)


class TestUniformCrossover:
    """Tests for uniform crossover operator."""

    def test_initialization(self, search_space):
        """Test crossover operator initialization."""
        crossover = UniformCrossover(search_space, seed=42)
        assert crossover.search_space == search_space
        assert crossover.swap_probability == 0.5

    def test_crossover_produces_two_offspring(self, search_space, parent_genomes):
        """Test that crossover produces two offspring."""
        crossover = UniformCrossover(search_space, seed=42)
        parent1, parent2 = parent_genomes

        offspring1, offspring2 = crossover.crossover(parent1, parent2)

        assert offspring1 is not None
        assert offspring2 is not None
        assert offspring1 != parent1
        assert offspring2 != parent2

    def test_offspring_validity(self, search_space, parent_genomes):
        """Test that offspring are valid genomes."""
        crossover = UniformCrossover(search_space, seed=42)
        parent1, parent2 = parent_genomes

        offspring1, offspring2 = crossover.crossover(parent1, parent2)

        assert offspring1.validate()
        assert offspring2.validate()
        assert search_space.validate_genome(offspring1)
        assert search_space.validate_genome(offspring2)

    def test_offspring_have_mixed_layers(self, search_space, population_generator):
        """Test that offspring contain layers from both parents."""
        # Create distinct parents
        gen = PopulationGenerator(search_space, seed=42)
        parent1 = gen._generate_small_architecture()
        parent2 = gen._generate_large_architecture()

        crossover = UniformCrossover(search_space, seed=42)
        offspring1, offspring2 = crossover.crossover(parent1, parent2)

        # Offspring should have intermediate sizes
        assert len(offspring1.layers) >= search_space.min_layers
        assert len(offspring2.layers) >= search_space.min_layers

    def test_swap_probability_effect(self, search_space, parent_genomes):
        """Test effect of different swap probabilities."""
        parent1, parent2 = parent_genomes

        # Low swap probability
        crossover_low = UniformCrossover(search_space, swap_probability=0.1, seed=42)
        offspring1_low, _ = crossover_low.crossover(parent1, parent2)

        # High swap probability
        crossover_high = UniformCrossover(search_space, swap_probability=0.9, seed=42)
        offspring1_high, _ = crossover_high.crossover(parent1, parent2)

        # Both should be valid
        assert offspring1_low.validate()
        assert offspring1_high.validate()


class TestSinglePointCrossover:
    """Tests for single-point crossover operator."""

    def test_initialization(self, search_space):
        """Test crossover operator initialization."""
        crossover = SinglePointCrossover(search_space, seed=42)
        assert crossover.search_space == search_space

    def test_crossover_produces_offspring(self, search_space, parent_genomes):
        """Test crossover produces valid offspring."""
        crossover = SinglePointCrossover(search_space, seed=42)
        parent1, parent2 = parent_genomes

        offspring1, offspring2 = crossover.crossover(parent1, parent2)

        assert offspring1.validate()
        assert offspring2.validate()

    def test_crossover_with_same_length_parents(self, search_space, population_generator):
        """Test crossover with parents of same length."""
        # Generate parents with same number of layers
        parent1 = population_generator.generate_random_genome()
        parent2 = population_generator.generate_random_genome()

        # Ensure same length
        target_len = min(len(parent1.layers), len(parent2.layers))
        parent1.layers = parent1.layers[:target_len]
        parent2.layers = parent2.layers[:target_len]

        crossover = SinglePointCrossover(search_space, seed=42)
        offspring1, offspring2 = crossover.crossover(parent1, parent2)

        assert offspring1.validate()
        assert offspring2.validate()


class TestLayerCrossover:
    """Tests for layer-wise crossover operator."""

    def test_initialization(self, search_space):
        """Test crossover operator initialization."""
        crossover = LayerCrossover(search_space, block_size=3, seed=42)
        assert crossover.search_space == search_space
        assert crossover.block_size == 3

    def test_crossover_preserves_blocks(self, search_space, parent_genomes):
        """Test that crossover preserves functional blocks."""
        crossover = LayerCrossover(search_space, block_size=3, seed=42)
        parent1, parent2 = parent_genomes

        offspring1, offspring2 = crossover.crossover(parent1, parent2)

        assert offspring1.validate()
        assert offspring2.validate()

    def test_block_identification(self, search_space, population_generator):
        """Test functional block identification."""
        crossover = LayerCrossover(search_space, block_size=3, seed=42)
        genome = population_generator.generate_random_genome()

        blocks = crossover._identify_blocks(genome.layers)

        # Should have at least one block
        assert len(blocks) > 0

        # All layers should be accounted for
        total_layers = sum(len(block) for block in blocks)
        assert total_layers == len(genome.layers)


class TestCrossoverRepair:
    """Tests for genome repair mechanism."""

    def test_repair_ensures_minimum_layers(self, search_space):
        """Test that repair ensures minimum layer count."""
        crossover = UniformCrossover(search_space, seed=42)
        gen = PopulationGenerator(search_space, seed=42)
        genome = gen.generate_random_genome()

        # Remove layers below minimum
        genome.layers = genome.layers[:2]
        assert len(genome.layers) < search_space.min_layers

        repaired = crossover._repair_genome(genome)
        assert len(repaired.layers) >= search_space.min_layers

    def test_repair_enforces_maximum_layers(self, search_space, population_generator):
        """Test that repair enforces maximum layer count."""
        crossover = UniformCrossover(search_space, seed=42)
        genome = population_generator._generate_large_architecture()

        # Add extra layers
        extra_layers = genome.layers[:5]
        genome.layers.extend(extra_layers)
        genome.layers.extend(extra_layers)

        repaired = crossover._repair_genome(genome)
        assert len(repaired.layers) <= search_space.max_layers

    def test_repair_fixes_connections(self, search_space, population_generator):
        """Test that repair fixes invalid connections."""
        crossover = UniformCrossover(search_space, seed=42)
        genome = population_generator.generate_random_genome()

        # Add invalid connections
        genome.connections.append((-1, 0))
        genome.connections.append((10, 100))
        genome.connections.append((5, 3))  # Backward connection

        repaired = crossover._repair_genome(genome)
        assert repaired.validate()


@pytest.mark.parametrize(
    "crossover_class",
    [UniformCrossover, SinglePointCrossover, LayerCrossover],
)
def test_crossover_reproducibility(crossover_class, search_space, parent_genomes):
    """Test that crossover is reproducible with same seed."""
    parent1, parent2 = parent_genomes

    # Run crossover twice with same seed
    crossover1 = crossover_class(search_space, seed=42)
    offspring1_a, offspring1_b = crossover1.crossover(parent1, parent2)

    crossover2 = crossover_class(search_space, seed=42)
    offspring2_a, offspring2_b = crossover2.crossover(parent1, parent2)

    # Should produce same results
    assert len(offspring1_a.layers) == len(offspring2_a.layers)
    assert len(offspring1_b.layers) == len(offspring2_b.layers)
