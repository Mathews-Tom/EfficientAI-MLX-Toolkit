"""
Tests for selection operators.
"""

import pytest

from evolutionary_search.operators.selection import (
    TournamentSelection,
    RouletteSelection,
    ElitistSelection,
    RankSelection,
)
from evolutionary_search.population import PopulationGenerator
from evolutionary_search.search_space import SearchSpaceConfig


@pytest.fixture
def search_space():
    """Create search space configuration."""
    return SearchSpaceConfig.default()


@pytest.fixture
def population(search_space):
    """Create test population with varied fitness."""
    gen = PopulationGenerator(search_space, seed=42)
    pop = gen.generate_population(20)

    # Assign varied fitness scores
    for i, genome in enumerate(pop):
        genome.fitness_score = float(i) / len(pop)

    return pop


class TestTournamentSelection:
    """Tests for tournament selection operator."""

    def test_initialization(self):
        """Test selection operator initialization."""
        selection = TournamentSelection(tournament_size=3, seed=42)
        assert selection.tournament_size == 3

    def test_select_returns_correct_count(self, population):
        """Test that selection returns requested number."""
        selection = TournamentSelection(tournament_size=3, seed=42)
        num_selected = 10

        selected = selection.select(population, num_selected)

        assert len(selected) == num_selected

    def test_tournament_selects_better_individuals(self, population):
        """Test that tournament selection favors higher fitness."""
        selection = TournamentSelection(tournament_size=5, seed=42)
        num_selected = 100

        selected = selection.select(population, num_selected)

        # Average fitness of selected should be higher than population
        pop_avg = sum(g.fitness_score for g in population) / len(population)
        selected_avg = sum(g.fitness_score for g in selected) / len(selected)

        assert selected_avg >= pop_avg

    def test_tournament_size_effect(self, population):
        """Test effect of different tournament sizes."""
        # Small tournament
        selection_small = TournamentSelection(tournament_size=2, seed=42)
        selected_small = selection_small.select(population, 50)

        # Large tournament
        selection_large = TournamentSelection(tournament_size=5, seed=42)
        selected_large = selection_large.select(population, 50)

        # Larger tournament should select higher fitness on average
        avg_small = sum(g.fitness_score for g in selected_small) / len(selected_small)
        avg_large = sum(g.fitness_score for g in selected_large) / len(selected_large)

        assert avg_large >= avg_small

    def test_tournament_with_small_population(self):
        """Test tournament selection with small population."""
        selection = TournamentSelection(tournament_size=5, seed=42)

        # Create small population
        search_space = SearchSpaceConfig.default()
        gen = PopulationGenerator(search_space, seed=42)
        small_pop = gen.generate_population(3)
        for i, g in enumerate(small_pop):
            g.fitness_score = float(i)

        selected = selection.select(small_pop, 2)
        assert len(selected) == 2


class TestRouletteSelection:
    """Tests for roulette wheel selection operator."""

    def test_initialization(self):
        """Test selection operator initialization."""
        selection = RouletteSelection(seed=42)
        assert selection is not None

    def test_select_returns_correct_count(self, population):
        """Test that selection returns requested number."""
        selection = RouletteSelection(seed=42)
        num_selected = 10

        selected = selection.select(population, num_selected)

        assert len(selected) == num_selected

    def test_roulette_favors_higher_fitness(self, population):
        """Test that roulette selection favors higher fitness."""
        selection = RouletteSelection(seed=42)
        num_selected = 100

        selected = selection.select(population, num_selected)

        # Average fitness of selected should be higher than population
        pop_avg = sum(g.fitness_score for g in population) / len(population)
        selected_avg = sum(g.fitness_score for g in selected) / len(selected)

        assert selected_avg >= pop_avg * 0.9  # Allow some variance

    def test_roulette_with_negative_fitness(self):
        """Test roulette selection with negative fitness values."""
        selection = RouletteSelection(seed=42)

        # Create population with negative fitness
        search_space = SearchSpaceConfig.default()
        gen = PopulationGenerator(search_space, seed=42)
        pop = gen.generate_population(10)
        for i, g in enumerate(pop):
            g.fitness_score = float(i - 5)  # Range: -5 to 4

        selected = selection.select(pop, 5)
        assert len(selected) == 5

    def test_roulette_with_zero_fitness(self):
        """Test roulette selection when all fitness is zero."""
        selection = RouletteSelection(seed=42)

        # Create population with zero fitness
        search_space = SearchSpaceConfig.default()
        gen = PopulationGenerator(search_space, seed=42)
        pop = gen.generate_population(10)
        for g in pop:
            g.fitness_score = 0.0

        selected = selection.select(pop, 5)
        assert len(selected) == 5


class TestElitistSelection:
    """Tests for elitist selection operator."""

    def test_initialization(self):
        """Test selection operator initialization."""
        selection = ElitistSelection(elite_fraction=0.2, seed=42)
        assert selection.elite_fraction == 0.2

    def test_select_returns_top_individuals(self, population):
        """Test that elitist selection returns top individuals."""
        selection = ElitistSelection(seed=42)
        num_selected = 5

        selected = selection.select(population, num_selected)

        # All selected should be among the top
        selected_fitness = [g.fitness_score for g in selected]
        all_fitness = [g.fitness_score for g in population]
        all_fitness_sorted = sorted(all_fitness, reverse=True)

        for fitness in selected_fitness:
            assert fitness in all_fitness_sorted[:num_selected]

    def test_elitist_maintains_order(self, population):
        """Test that elitist selection maintains fitness order."""
        selection = ElitistSelection(seed=42)
        num_selected = 10

        selected = selection.select(population, num_selected)
        selected_fitness = [g.fitness_score for g in selected]

        # Should be in descending order
        assert selected_fitness == sorted(selected_fitness, reverse=True)

    def test_elitist_with_large_selection(self, population):
        """Test elitist selection when selecting entire population."""
        selection = ElitistSelection(seed=42)

        selected = selection.select(population, len(population))

        assert len(selected) == len(population)
        # First should be highest fitness
        assert selected[0].fitness_score == max(g.fitness_score for g in population)


class TestRankSelection:
    """Tests for rank-based selection operator."""

    def test_initialization(self):
        """Test selection operator initialization."""
        selection = RankSelection(selection_pressure=1.5, seed=42)
        assert selection.selection_pressure == 1.5

    def test_select_returns_correct_count(self, population):
        """Test that selection returns requested number."""
        selection = RankSelection(seed=42)
        num_selected = 10

        selected = selection.select(population, num_selected)

        assert len(selected) == num_selected

    def test_rank_selection_bias(self, population):
        """Test that rank selection has appropriate selection bias."""
        selection = RankSelection(selection_pressure=2.0, seed=42)
        num_selected = 100

        selected = selection.select(population, num_selected)

        # Higher ranks should be more common
        pop_avg = sum(g.fitness_score for g in population) / len(population)
        selected_avg = sum(g.fitness_score for g in selected) / len(selected)

        assert selected_avg >= pop_avg

    def test_selection_pressure_effect(self, population):
        """Test effect of different selection pressures."""
        # Low pressure
        selection_low = RankSelection(selection_pressure=1.1, seed=42)
        selected_low = selection_low.select(population, 50)

        # High pressure
        selection_high = RankSelection(selection_pressure=2.0, seed=42)
        selected_high = selection_high.select(population, 50)

        # High pressure should favor better individuals more
        avg_low = sum(g.fitness_score for g in selected_low) / len(selected_low)
        avg_high = sum(g.fitness_score for g in selected_high) / len(selected_high)

        assert avg_high >= avg_low

    def test_selection_pressure_clamping(self):
        """Test that selection pressure is clamped to valid range."""
        # Below minimum
        selection_low = RankSelection(selection_pressure=0.5, seed=42)
        assert selection_low.selection_pressure == 1.0

        # Above maximum
        selection_high = RankSelection(selection_pressure=3.0, seed=42)
        assert selection_high.selection_pressure == 2.0


@pytest.mark.parametrize(
    "selection_class",
    [TournamentSelection, RouletteSelection, ElitistSelection, RankSelection],
)
def test_selection_reproducibility(selection_class, population):
    """Test that selection is reproducible with same seed."""
    # Run selection twice with same seed
    selection1 = selection_class(seed=42)
    selected1 = selection1.select(population, 10)

    selection2 = selection_class(seed=42)
    selected2 = selection2.select(population, 10)

    # Should produce same results
    fitness1 = [g.fitness_score for g in selected1]
    fitness2 = [g.fitness_score for g in selected2]

    assert fitness1 == fitness2


@pytest.mark.parametrize(
    "selection_class",
    [TournamentSelection, RouletteSelection, ElitistSelection, RankSelection],
)
def test_selection_with_varied_population_sizes(selection_class):
    """Test selection with different population sizes."""
    search_space = SearchSpaceConfig.default()
    gen = PopulationGenerator(search_space, seed=42)

    for pop_size in [5, 10, 50]:
        pop = gen.generate_population(pop_size)
        for i, g in enumerate(pop):
            g.fitness_score = float(i)

        selection = selection_class(seed=42)
        selected = selection.select(pop, min(5, pop_size))

        assert len(selected) <= pop_size
