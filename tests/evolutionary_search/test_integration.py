"""
Integration tests for evolutionary architecture search.
"""

import pytest

from evolutionary_search.engine import EvolutionConfig, EvolutionEngine
from evolutionary_search.optimization import NSGAII, NSGAConfig, ParetoArchive
from evolutionary_search.search_space import SearchSpaceConfig


class TestEvolutionIntegration:
    """End-to-end integration tests."""

    def test_complete_evolution_pipeline(self):
        """Test complete evolution from initialization to result."""
        search_space = SearchSpaceConfig.default()
        config = EvolutionConfig(
            population_size=10,
            num_generations=5,
            elite_size=2,
            seed=42,
        )

        engine = EvolutionEngine(search_space, config)
        result = engine.evolve()

        # Verify result completeness
        assert result.best_genome is not None
        assert result.best_genome.validate()
        assert result.best_fitness.combined_score > 0
        assert len(result.generation_history) > 0
        assert len(result.final_population) == config.population_size
        assert result.total_evaluations >= config.population_size

    def test_multiobjective_evolution_pipeline(self):
        """Test multi-objective evolution pipeline."""
        search_space = SearchSpaceConfig.default()
        config = NSGAConfig(
            population_size=10,
            num_generations=3,
            archive_size=20,
            seed=42,
        )

        nsga = NSGAII(search_space, config)
        archive = nsga.evolve()

        # Verify archive
        assert isinstance(archive, ParetoArchive)
        assert len(archive.all_solutions) > 0
        assert len(archive.fronts) > 0

        # Verify best front
        best_front = archive.get_best_front()
        assert best_front is not None
        assert len(best_front.solutions) > 0

    def test_evolution_with_constraints(self):
        """Test evolution respects hardware constraints."""
        search_space = SearchSpaceConfig(
            min_layers=4,
            max_layers=16,
            max_parameters=500_000_000,  # 500M limit
            max_memory_mb=8192,  # 8GB limit
        )

        config = EvolutionConfig(
            population_size=15,
            num_generations=5,
            seed=42,
        )

        engine = EvolutionEngine(search_space, config)
        result = engine.evolve()

        # Verify all solutions respect constraints
        for genome in result.final_population:
            assert search_space.validate_genome(genome)
            assert genome.count_parameters() <= search_space.max_parameters

    def test_evolution_fitness_improvement(self):
        """Test that evolution improves fitness over generations."""
        search_space = SearchSpaceConfig.default()
        config = EvolutionConfig(
            population_size=20,
            num_generations=10,
            seed=42,
        )

        engine = EvolutionEngine(search_space, config)
        result = engine.evolve()

        # Check fitness progression
        if len(result.generation_history) > 1:
            first_gen = result.generation_history[0]
            last_gen = result.generation_history[-1]

            # Best fitness should not decrease
            assert last_gen["best_fitness"] >= first_gen["best_fitness"]

            # Mean fitness should generally improve
            # (allowing some variance for stochastic process)
            assert last_gen["mean_fitness"] >= first_gen["mean_fitness"] * 0.95

    def test_evolution_convergence(self):
        """Test evolution convergence detection."""
        search_space = SearchSpaceConfig.default()
        config = EvolutionConfig(
            population_size=10,
            num_generations=50,
            convergence_threshold=0.001,
            convergence_generations=5,
            seed=42,
        )

        engine = EvolutionEngine(search_space, config)
        result = engine.evolve()

        # Should converge before max generations in most cases
        assert result.convergence_generation < config.num_generations


class TestCrossComponentIntegration:
    """Tests for integration across components."""

    def test_operators_with_search_space(self):
        """Test operators respect search space constraints."""
        from evolutionary_search.operators import UniformCrossover, CompositeMutation
        from evolutionary_search.population import PopulationGenerator

        search_space = SearchSpaceConfig(min_layers=6, max_layers=12)
        gen = PopulationGenerator(search_space, seed=42)

        # Generate parents
        parent1, parent2 = gen.generate_population(2)

        # Test crossover
        crossover = UniformCrossover(search_space, seed=42)
        child1, child2 = crossover.crossover(parent1, parent2)

        assert search_space.validate_genome(child1)
        assert search_space.validate_genome(child2)

        # Test mutation
        mutation = CompositeMutation(search_space, mutation_rate=1.0, seed=42)
        mutated = mutation.mutate(child1)

        assert search_space.validate_genome(mutated)

    def test_fitness_evaluation_consistency(self):
        """Test fitness evaluation consistency."""
        from evolutionary_search.fitness import FitnessEvaluator
        from evolutionary_search.population import PopulationGenerator

        search_space = SearchSpaceConfig.default()
        gen = PopulationGenerator(search_space, seed=42)
        evaluator = FitnessEvaluator()

        genome = gen.generate_random_genome()

        # Evaluate multiple times
        scores = [evaluator.evaluate(genome) for _ in range(3)]

        # Should be consistent (deterministic for same genome)
        for i in range(1, len(scores)):
            assert abs(scores[i].combined_score - scores[0].combined_score) < 0.01


class TestPerformanceIntegration:
    """Performance integration tests."""

    def test_evolution_completes_in_reasonable_time(self):
        """Test that evolution completes efficiently."""
        import time

        search_space = SearchSpaceConfig.default()
        config = EvolutionConfig(
            population_size=20,
            num_generations=5,
            seed=42,
        )

        start_time = time.time()
        engine = EvolutionEngine(search_space, config)
        result = engine.evolve()
        elapsed = time.time() - start_time

        # Should complete in reasonable time (< 10 seconds for small test)
        assert elapsed < 10.0
        assert result is not None
