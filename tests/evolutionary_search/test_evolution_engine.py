"""
Tests for evolution engine.
"""

import pytest

from evolutionary_search.engine import EvolutionEngine, EvolutionConfig, EvolutionResult
from evolutionary_search.fitness import FitnessEvaluator
from evolutionary_search.operators import (
    UniformCrossover,
    CompositeMutation,
    TournamentSelection,
)
from evolutionary_search.search_space import SearchSpaceConfig


@pytest.fixture
def search_space():
    """Create search space configuration."""
    return SearchSpaceConfig.default()


@pytest.fixture
def evolution_config():
    """Create evolution configuration for testing."""
    return EvolutionConfig(
        population_size=10,
        num_generations=5,
        elite_size=2,
        crossover_rate=0.8,
        mutation_rate=0.3,
        convergence_threshold=0.001,
        convergence_generations=3,
        diversity_threshold=0.1,
        seed=42,
    )


@pytest.fixture
def evolution_engine(search_space, evolution_config):
    """Create evolution engine."""
    return EvolutionEngine(search_space, evolution_config)


class TestEvolutionConfig:
    """Tests for evolution configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = EvolutionConfig()

        assert config.population_size == 50
        assert config.num_generations == 100
        assert config.elite_size == 5
        assert config.crossover_rate == 0.8
        assert config.mutation_rate == 0.3

    def test_custom_config(self):
        """Test custom configuration."""
        config = EvolutionConfig(
            population_size=20,
            num_generations=10,
            elite_size=3,
            seed=123,
        )

        assert config.population_size == 20
        assert config.num_generations == 10
        assert config.elite_size == 3
        assert config.seed == 123


class TestEvolutionEngine:
    """Tests for evolution engine."""

    def test_initialization(self, search_space, evolution_config):
        """Test engine initialization."""
        engine = EvolutionEngine(search_space, evolution_config)

        assert engine.search_space == search_space
        assert engine.config == evolution_config
        assert engine.generation == 0
        assert len(engine.population) == 0

    def test_initialization_with_custom_operators(self, search_space, evolution_config):
        """Test engine initialization with custom operators."""
        fitness_eval = FitnessEvaluator()
        crossover = UniformCrossover(search_space, seed=42)
        mutation = CompositeMutation(search_space, seed=42)
        selection = TournamentSelection(seed=42)

        engine = EvolutionEngine(
            search_space,
            evolution_config,
            fitness_evaluator=fitness_eval,
            crossover_operator=crossover,
            mutation_operator=mutation,
            selection_operator=selection,
        )

        assert engine.fitness_evaluator == fitness_eval
        assert engine.crossover_operator == crossover
        assert engine.mutation_operator == mutation
        assert engine.selection_operator == selection

    def test_initialize_population(self, evolution_engine):
        """Test population initialization."""
        evolution_engine._initialize_population()

        assert len(evolution_engine.population) == evolution_engine.config.population_size
        for genome in evolution_engine.population:
            assert genome.validate()

    def test_evaluate_population(self, evolution_engine):
        """Test population fitness evaluation."""
        evolution_engine._initialize_population()
        evolution_engine._evaluate_population()

        # All genomes should have fitness scores
        for genome in evolution_engine.population:
            assert genome.fitness_score > 0.0

        # Best genome should be set
        assert evolution_engine.best_genome is not None
        assert evolution_engine.best_fitness > 0.0

    def test_record_generation_stats(self, evolution_engine):
        """Test generation statistics recording."""
        evolution_engine._initialize_population()
        evolution_engine._evaluate_population()
        evolution_engine._record_generation_stats()

        assert len(evolution_engine.generation_history) == 1

        stats = evolution_engine.generation_history[0]
        assert "generation" in stats
        assert "best_fitness" in stats
        assert "mean_fitness" in stats
        assert "std_fitness" in stats
        assert "diversity" in stats

    def test_create_next_generation(self, evolution_engine):
        """Test next generation creation."""
        evolution_engine._initialize_population()
        evolution_engine._evaluate_population()

        original_population = evolution_engine.population.copy()
        evolution_engine._create_next_generation()

        # Population size should be maintained
        assert len(evolution_engine.population) == evolution_engine.config.population_size

        # Should have new individuals (not all identical to originals)
        new_genomes = [g for g in evolution_engine.population if g not in original_population]
        assert len(new_genomes) > 0

    def test_convergence_detection(self, evolution_engine):
        """Test convergence detection."""
        evolution_engine._initialize_population()
        evolution_engine._evaluate_population()

        # Initially should not converge
        assert not evolution_engine._check_convergence()

        # Simulate no improvement
        for _ in range(evolution_engine.config.convergence_generations):
            converged = evolution_engine._check_convergence()

        # Should converge after N generations without improvement
        assert converged

    def test_diversity_maintenance(self, evolution_engine):
        """Test diversity maintenance mechanism."""
        evolution_engine._initialize_population()
        evolution_engine._evaluate_population()

        # Get initial population
        initial_pop = evolution_engine.population.copy()

        # Artificially reduce diversity by duplicating individuals
        evolution_engine.population = [initial_pop[0]] * evolution_engine.config.population_size

        # Maintain diversity
        evolution_engine._maintain_diversity()

        # Population should be more diverse now
        from evolutionary_search.population import DiversityMetrics

        diversity = DiversityMetrics.compute_diversity(evolution_engine.population)
        assert diversity > 0.0

    def test_evolve_returns_result(self, evolution_engine):
        """Test that evolve returns valid result."""
        result = evolution_engine.evolve()

        assert isinstance(result, EvolutionResult)
        assert result.best_genome is not None
        assert result.best_fitness is not None
        assert len(result.generation_history) > 0
        assert len(result.final_population) == evolution_engine.config.population_size
        assert result.total_evaluations > 0

    def test_evolve_improves_fitness(self, evolution_engine):
        """Test that evolution improves fitness over generations."""
        result = evolution_engine.evolve()

        # Get fitness progression
        if len(result.generation_history) > 1:
            first_gen_fitness = result.generation_history[0]["best_fitness"]
            last_gen_fitness = result.generation_history[-1]["best_fitness"]

            # Fitness should improve or stay same
            assert last_gen_fitness >= first_gen_fitness

    def test_evolve_respects_generation_limit(self):
        """Test that evolution respects generation limit."""
        search_space = SearchSpaceConfig.default()
        config = EvolutionConfig(
            population_size=10,
            num_generations=5,
            convergence_generations=100,  # High to avoid early convergence
            seed=42,
        )
        engine = EvolutionEngine(search_space, config)

        result = engine.evolve()

        # Should not exceed generation limit
        assert len(result.generation_history) <= config.num_generations

    def test_elitism_preserves_best(self, evolution_engine):
        """Test that elitism preserves best individuals."""
        result = evolution_engine.evolve()

        # Best fitness should never decrease across generations
        best_fitness_progression = [
            gen["best_fitness"] for gen in result.generation_history
        ]

        for i in range(1, len(best_fitness_progression)):
            assert best_fitness_progression[i] >= best_fitness_progression[i - 1]

    def test_evolution_with_small_population(self):
        """Test evolution with very small population."""
        search_space = SearchSpaceConfig.default()
        config = EvolutionConfig(
            population_size=4,
            num_generations=3,
            elite_size=1,
            seed=42,
        )
        engine = EvolutionEngine(search_space, config)

        result = engine.evolve()

        assert result.best_genome is not None
        assert len(result.final_population) == 4

    def test_evolution_reproducibility(self, search_space):
        """Test that evolution is reproducible with same seed."""
        config1 = EvolutionConfig(
            population_size=10,
            num_generations=3,
            seed=42,
        )
        engine1 = EvolutionEngine(search_space, config1)
        result1 = engine1.evolve()

        config2 = EvolutionConfig(
            population_size=10,
            num_generations=3,
            seed=42,
        )
        engine2 = EvolutionEngine(search_space, config2)
        result2 = engine2.evolve()

        # Should produce same best fitness
        assert abs(result1.best_fitness.combined_score - result2.best_fitness.combined_score) < 0.01


class TestEvolutionResult:
    """Tests for evolution result."""

    def test_result_attributes(self, evolution_engine):
        """Test that result contains all expected attributes."""
        result = evolution_engine.evolve()

        assert hasattr(result, "best_genome")
        assert hasattr(result, "best_fitness")
        assert hasattr(result, "generation_history")
        assert hasattr(result, "final_population")
        assert hasattr(result, "convergence_generation")
        assert hasattr(result, "total_evaluations")

    def test_result_best_genome_validity(self, evolution_engine):
        """Test that best genome in result is valid."""
        result = evolution_engine.evolve()

        assert result.best_genome.validate()
        assert evolution_engine.search_space.validate_genome(result.best_genome)

    def test_generation_history_completeness(self, evolution_engine):
        """Test that generation history is complete."""
        result = evolution_engine.evolve()

        for i, gen_stats in enumerate(result.generation_history):
            assert gen_stats["generation"] == i
            assert "best_fitness" in gen_stats
            assert "mean_fitness" in gen_stats
            assert "diversity" in gen_stats


class TestEvolutionIntegration:
    """Integration tests for complete evolution process."""

    def test_complete_evolution_run(self):
        """Test a complete evolution run from start to finish."""
        search_space = SearchSpaceConfig.default()
        config = EvolutionConfig(
            population_size=20,
            num_generations=10,
            elite_size=3,
            crossover_rate=0.8,
            mutation_rate=0.3,
            seed=42,
        )

        engine = EvolutionEngine(search_space, config)
        result = engine.evolve()

        # Verify result quality
        assert result.best_genome is not None
        assert result.best_fitness.combined_score > 0.0
        assert len(result.generation_history) <= config.num_generations
        assert result.total_evaluations >= config.population_size

    def test_evolution_with_different_configurations(self):
        """Test evolution with various configurations."""
        search_space = SearchSpaceConfig.default()

        configs = [
            EvolutionConfig(population_size=10, num_generations=5, seed=42),
            EvolutionConfig(population_size=20, num_generations=5, seed=42),
            EvolutionConfig(population_size=10, num_generations=10, seed=42),
        ]

        for config in configs:
            engine = EvolutionEngine(search_space, config)
            result = engine.evolve()

            assert result.best_genome is not None
            assert len(result.final_population) == config.population_size
