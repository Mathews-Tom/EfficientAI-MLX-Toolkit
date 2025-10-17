"""
Tests for multi-objective optimization.
"""

import pytest

from evolutionary_search.fitness import MultiObjectiveScore
from evolutionary_search.optimization import (
    NSGAII,
    NSGAIII,
    NSGAConfig,
    ParetoArchive,
    ParetoFront,
    dominates,
    GaussianProcessSurrogate,
    RandomForestSurrogate,
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


class TestParetoFront:
    """Tests for Pareto front."""

    def test_dominates_function(self):
        """Test Pareto dominance check."""
        obj1 = {"quality": 0.8, "speed": 0.7, "memory": 0.6}
        obj2 = {"quality": 0.7, "speed": 0.6, "memory": 0.5}
        obj3 = {"quality": 0.9, "speed": 0.5, "memory": 0.7}

        assert dominates(obj1, obj2)
        assert not dominates(obj2, obj1)
        assert not dominates(obj1, obj3)
        assert not dominates(obj3, obj1)

    def test_pareto_front_creation(self, population_generator):
        """Test creating a Pareto front."""
        genomes = population_generator.generate_population(5)
        objectives = [
            {"quality": 0.8, "speed": 0.7},
            {"quality": 0.7, "speed": 0.8},
            {"quality": 0.6, "speed": 0.6},
            {"quality": 0.9, "speed": 0.6},
            {"quality": 0.5, "speed": 0.9},
        ]

        front = ParetoFront(solutions=genomes, objectives=objectives, rank=0)

        assert len(front.solutions) == 5
        assert len(front.objectives) == 5
        assert front.rank == 0

    def test_crowding_distance_computation(self, population_generator):
        """Test crowding distance calculation."""
        genomes = population_generator.generate_population(5)
        objectives = [
            {"quality": 0.1, "speed": 0.9},
            {"quality": 0.3, "speed": 0.7},
            {"quality": 0.5, "speed": 0.5},
            {"quality": 0.7, "speed": 0.3},
            {"quality": 0.9, "speed": 0.1},
        ]

        front = ParetoFront(solutions=genomes, objectives=objectives)
        distances = front.compute_crowding_distances()

        assert len(distances) == 5
        # Boundary points should have infinite distance
        assert distances[0] == float("inf") or distances[-1] == float("inf")


class TestParetoArchive:
    """Tests for Pareto archive."""

    def test_archive_initialization(self):
        """Test archive initialization."""
        archive = ParetoArchive(max_size=10)
        assert archive.max_size == 10
        assert len(archive.all_solutions) == 0

    def test_add_solution_to_archive(self, population_generator):
        """Test adding solutions to archive."""
        archive = ParetoArchive(max_size=10)
        genome = population_generator.generate_random_genome()
        objectives = {"quality": 0.8, "speed": 0.7, "memory": 0.6}

        added = archive.add_solution(genome, objectives)

        assert added
        assert len(archive.all_solutions) == 1

    def test_archive_rejects_dominated_solutions(self, population_generator):
        """Test that archive rejects dominated solutions."""
        archive = ParetoArchive(max_size=10)

        genome1 = population_generator.generate_random_genome()
        obj1 = {"quality": 0.8, "speed": 0.7, "memory": 0.6}
        archive.add_solution(genome1, obj1)

        genome2 = population_generator.generate_random_genome()
        obj2 = {"quality": 0.7, "speed": 0.6, "memory": 0.5}  # Dominated
        added = archive.add_solution(genome2, obj2)

        assert not added
        assert len(archive.all_solutions) == 1

    def test_archive_removes_dominated_by_new(self, population_generator):
        """Test that archive removes solutions dominated by new one."""
        archive = ParetoArchive(max_size=10)

        genome1 = population_generator.generate_random_genome()
        obj1 = {"quality": 0.7, "speed": 0.6, "memory": 0.5}
        archive.add_solution(genome1, obj1)

        genome2 = population_generator.generate_random_genome()
        obj2 = {"quality": 0.8, "speed": 0.7, "memory": 0.6}  # Dominates obj1
        added = archive.add_solution(genome2, obj2)

        assert added
        assert len(archive.all_solutions) == 1  # Only obj2 remains

    def test_archive_maintains_max_size(self, population_generator):
        """Test that archive trims to max size."""
        archive = ParetoArchive(max_size=5)

        # Add many non-dominated solutions
        for i in range(10):
            genome = population_generator.generate_random_genome()
            obj = {"quality": float(i) / 10, "speed": 1.0 - float(i) / 10}
            archive.add_solution(genome, obj)

        assert len(archive.all_solutions) <= 5


class TestNSGAII:
    """Tests for NSGA-II algorithm."""

    def test_nsga2_initialization(self, search_space):
        """Test NSGA-II initialization."""
        config = NSGAConfig(population_size=10, num_generations=2, seed=42)
        nsga = NSGAII(search_space, config)

        assert nsga.search_space == search_space
        assert nsga.config == config
        assert nsga.generation == 0

    def test_nsga2_evolve(self, search_space):
        """Test NSGA-II evolution."""
        config = NSGAConfig(population_size=10, num_generations=3, seed=42)
        nsga = NSGAII(search_space, config)

        archive = nsga.evolve()

        assert isinstance(archive, ParetoArchive)
        assert len(archive.all_solutions) > 0

    def test_nsga2_fast_non_dominated_sort(self, search_space, population_generator):
        """Test fast non-dominated sorting."""
        config = NSGAConfig(population_size=10, num_generations=1, seed=42)
        nsga = NSGAII(search_space, config)

        # Create population with objectives
        population = population_generator.generate_population(10)
        for i, genome in enumerate(population):
            score = MultiObjectiveScore(
                objectives={
                    "quality": float(i) / 10,
                    "speed": 1.0 - float(i) / 10,
                },
                constraints_satisfied=True,
            )
            genome.metadata["multi_objective_score"] = score

        fronts = nsga._fast_non_dominated_sort(population)

        assert len(fronts) > 0
        assert len(fronts[0]) > 0  # First front should exist


class TestNSGAIII:
    """Tests for NSGA-III algorithm."""

    def test_nsga3_initialization(self, search_space):
        """Test NSGA-III initialization."""
        config = NSGAConfig(population_size=10, num_generations=2, seed=42)
        nsga = NSGAIII(search_space, config, num_reference_points=10)

        assert nsga.num_reference_points == 10
        assert nsga.reference_points is not None

    def test_nsga3_evolve(self, search_space):
        """Test NSGA-III evolution."""
        config = NSGAConfig(population_size=10, num_generations=2, seed=42)
        nsga = NSGAIII(search_space, config, num_reference_points=10)

        archive = nsga.evolve()

        assert isinstance(archive, ParetoArchive)
        assert len(archive.all_solutions) > 0


class TestSurrogateModels:
    """Tests for surrogate models."""

    def test_gp_surrogate_initialization(self):
        """Test GP surrogate initialization."""
        surrogate = GaussianProcessSurrogate(noise_level=0.1)
        assert not surrogate.is_trained

    def test_gp_surrogate_training(self, population_generator):
        """Test GP surrogate training."""
        from evolutionary_search.fitness import FitnessEvaluator, FitnessMetrics

        surrogate = GaussianProcessSurrogate()
        evaluator = FitnessEvaluator()

        # Generate training data
        genomes = population_generator.generate_population(10)
        fitness_scores = [evaluator.evaluate(g) for g in genomes]

        surrogate.train(genomes, fitness_scores)

        assert surrogate.is_trained
        assert surrogate.X_train is not None

    def test_gp_surrogate_prediction(self, population_generator):
        """Test GP surrogate prediction."""
        from evolutionary_search.fitness import FitnessEvaluator

        surrogate = GaussianProcessSurrogate()
        evaluator = FitnessEvaluator()

        # Train surrogate
        genomes = population_generator.generate_population(10)
        fitness_scores = [evaluator.evaluate(g) for g in genomes]
        surrogate.train(genomes, fitness_scores)

        # Make prediction
        test_genome = population_generator.generate_random_genome()
        prediction = surrogate.predict(test_genome)

        assert prediction is not None
        assert 0.0 <= prediction.combined_score <= 1.0

    def test_rf_surrogate_training(self, population_generator):
        """Test Random Forest surrogate training."""
        from evolutionary_search.fitness import FitnessEvaluator

        surrogate = RandomForestSurrogate(num_trees=5)
        evaluator = FitnessEvaluator()

        # Generate training data
        genomes = population_generator.generate_population(10)
        fitness_scores = [evaluator.evaluate(g) for g in genomes]

        surrogate.train(genomes, fitness_scores)

        assert surrogate.is_trained

    def test_rf_surrogate_prediction(self, population_generator):
        """Test Random Forest surrogate prediction."""
        from evolutionary_search.fitness import FitnessEvaluator

        surrogate = RandomForestSurrogate(num_trees=5)
        evaluator = FitnessEvaluator()

        # Train surrogate
        genomes = population_generator.generate_population(10)
        fitness_scores = [evaluator.evaluate(g) for g in genomes]
        surrogate.train(genomes, fitness_scores)

        # Make prediction
        test_genome = population_generator.generate_random_genome()
        prediction = surrogate.predict(test_genome)

        assert prediction is not None
        assert 0.0 <= prediction.combined_score <= 1.0
        assert "uncertainty" in prediction.raw_metrics
