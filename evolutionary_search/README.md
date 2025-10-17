# Evolutionary Diffusion Architecture Search

Comprehensive evolutionary search framework for optimizing diffusion model architectures with multi-objective optimization and Apple Silicon hardware awareness.

## Overview

This module provides a complete Neural Architecture Search (NAS) framework specifically designed for diffusion models, with focus on:

- **Multi-objective optimization**: Balance quality, speed, and memory efficiency
- **Hardware awareness**: Optimized for Apple Silicon (M1/M2/M3)
- **Flexible operators**: Crossover, mutation, and selection strategies
- **Efficient evaluation**: Surrogate models for fast fitness approximation

## Architecture

```
evolutionary_search/
├── search_space/       # Architecture genome representation
├── fitness/            # Multi-objective fitness evaluation
├── population/         # Population initialization and diversity
├── operators/          # Genetic operators (crossover, mutation, selection)
├── engine/             # Evolution engine with convergence
├── optimization/       # NSGA-II/III, Pareto front management
└── README.md           # This file
```

## Quick Start

### Basic Evolution

```python
from evolutionary_search.search_space import SearchSpaceConfig
from evolutionary_search.engine import EvolutionConfig, EvolutionEngine

# Configure search space
search_space = SearchSpaceConfig(
    min_layers=4,
    max_layers=32,
    max_memory_mb=16384,  # 16GB for Apple Silicon
    max_parameters=1_000_000_000  # 1B parameter limit
)

# Configure evolution
config = EvolutionConfig(
    population_size=50,
    num_generations=100,
    elite_size=5,
    crossover_rate=0.8,
    mutation_rate=0.3,
    seed=42
)

# Run evolution
engine = EvolutionEngine(search_space, config)
result = engine.evolve()

# Get best architecture
best_genome = result.best_genome
print(f"Best fitness: {result.best_fitness.combined_score:.4f}")
print(f"Layers: {len(best_genome.layers)}")
print(f"Parameters: {best_genome.count_parameters():,}")
```

### Multi-Objective Optimization

```python
from evolutionary_search.optimization import NSGAII, NSGAConfig

# Configure NSGA-II
config = NSGAConfig(
    population_size=50,
    num_generations=100,
    archive_size=100,
    seed=42
)

# Run multi-objective evolution
nsga = NSGAII(search_space, config)
archive = nsga.evolve()

# Get Pareto front
best_front = archive.get_best_front()
print(f"Pareto front size: {len(best_front.solutions)}")

# Analyze trade-offs
for solution, objectives in zip(best_front.solutions, best_front.objectives):
    print(f"Quality: {objectives['quality']:.3f}, "
          f"Speed: {objectives['speed']:.3f}, "
          f"Memory: {objectives['memory']:.3f}")
```

## Components

### 1. Search Space Definition

The search space defines the possible architectures:

```python
from evolutionary_search.search_space import (
    ArchitectureGenome,
    LayerConfig,
    ArchitectureComponent,
)

# Create custom architecture
layers = [
    LayerConfig(
        component_type=ArchitectureComponent.CONV_BLOCK,
        parameters={"kernel_size": 3, "stride": 1, "padding": 1},
        layer_index=0,
        in_channels=3,
        out_channels=64
    ),
    LayerConfig(
        component_type=ArchitectureComponent.ATTENTION_BLOCK,
        parameters={"num_heads": 8, "embed_dim": 512},
        layer_index=1,
        in_channels=64,
        out_channels=64
    ),
]

genome = ArchitectureGenome(layers=layers)
assert genome.validate()
```

### 2. Fitness Evaluation

Multi-objective fitness evaluation:

```python
from evolutionary_search.fitness import FitnessEvaluator

evaluator = FitnessEvaluator(
    quality_weight=0.4,
    speed_weight=0.3,
    memory_weight=0.3,
    target_inference_ms=100.0,
    target_memory_mb=2048.0
)

# Evaluate architecture
metrics = evaluator.evaluate(genome)
print(f"Quality: {metrics.quality_score:.3f}")
print(f"Speed: {metrics.speed_score:.3f}")
print(f"Memory: {metrics.memory_score:.3f}")
print(f"Combined: {metrics.combined_score:.3f}")
```

### 3. Genetic Operators

#### Crossover

```python
from evolutionary_search.operators import UniformCrossover, LayerCrossover

# Uniform crossover
crossover = UniformCrossover(search_space, swap_probability=0.5)
offspring1, offspring2 = crossover.crossover(parent1, parent2)

# Layer-wise crossover (preserves functional blocks)
layer_crossover = LayerCrossover(search_space, block_size=3)
offspring1, offspring2 = layer_crossover.crossover(parent1, parent2)
```

#### Mutation

```python
from evolutionary_search.operators import (
    LayerMutation,
    ParameterMutation,
    StructuralMutation,
    CompositeMutation
)

# Layer mutation (add/remove/replace layers)
layer_mut = LayerMutation(search_space, mutation_rate=0.2)
mutated = layer_mut.mutate(genome)

# Parameter mutation (modify layer parameters)
param_mut = ParameterMutation(search_space, mutation_rate=0.3)
mutated = param_mut.mutate(genome)

# Structural mutation (modify connections)
struct_mut = StructuralMutation(search_space, mutation_rate=0.15)
mutated = struct_mut.mutate(genome)

# Composite mutation (combines all)
composite_mut = CompositeMutation(search_space, mutation_rate=0.3)
mutated = composite_mut.mutate(genome)
```

#### Selection

```python
from evolutionary_search.operators import (
    TournamentSelection,
    RouletteSelection,
    ElitistSelection,
    RankSelection
)

# Tournament selection
tournament = TournamentSelection(tournament_size=3)
selected = tournament.select(population, num_selected=10)

# Elitist selection
elitist = ElitistSelection(elite_fraction=0.1)
selected = elitist.select(population, num_selected=5)
```

### 4. Pareto Front Analysis

```python
from evolutionary_search.optimization import ParetoArchive

# Create archive
archive = ParetoArchive(max_size=100)

# Add solutions
for genome in population:
    objectives = {
        "quality": 0.8,
        "speed": 0.7,
        "memory": 0.6
    }
    archive.add_solution(genome, objectives)

# Analyze Pareto fronts
for rank, front in enumerate(archive.fronts):
    print(f"Rank {rank}: {len(front.solutions)} solutions")

    # Compute metrics
    distances = front.compute_crowding_distances()
    hypervolume = front.get_hypervolume(reference_point={"quality": 0, "speed": 0, "memory": 0})
    print(f"  Hypervolume: {hypervolume:.4f}")
```

### 5. Surrogate Models

Fast evaluation using surrogate models:

```python
from evolutionary_search.optimization import GaussianProcessSurrogate

# Train surrogate
surrogate = GaussianProcessSurrogate(noise_level=0.1)
surrogate.train(evaluated_genomes, fitness_scores)

# Fast prediction
new_genome = generate_candidate()
predicted_fitness = surrogate.predict(new_genome)
print(f"Predicted fitness: {predicted_fitness.combined_score:.4f}")
```

## Advanced Usage

### Custom Evolution Engine

```python
from evolutionary_search.engine import EvolutionEngine
from evolutionary_search.operators import UniformCrossover, CompositeMutation, TournamentSelection

# Create custom operators
crossover = UniformCrossover(search_space, swap_probability=0.7)
mutation = CompositeMutation(search_space, mutation_rate=0.4)
selection = TournamentSelection(tournament_size=5)

# Create engine with custom operators
engine = EvolutionEngine(
    search_space,
    config,
    crossover_operator=crossover,
    mutation_operator=mutation,
    selection_operator=selection
)

result = engine.evolve()
```

### Tracking Evolution Progress

```python
result = engine.evolve()

# Analyze generation history
for gen in result.generation_history:
    print(f"Generation {gen['generation']}: "
          f"Best={gen['best_fitness']:.4f}, "
          f"Mean={gen['mean_fitness']:.4f}, "
          f"Diversity={gen['diversity']:.4f}")
```

### Hardware Constraints

```python
# Configure for Apple Silicon M1 (8GB)
search_space = SearchSpaceConfig(
    max_memory_mb=8192,
    max_parameters=500_000_000,
    target_inference_ms=50.0
)

# Configure for Apple Silicon M2 Ultra (192GB)
search_space_large = SearchSpaceConfig(
    max_memory_mb=196608,
    max_parameters=10_000_000_000,
    target_inference_ms=100.0
)
```

## Performance Optimization

### Using Surrogate Models

```python
from evolutionary_search.optimization import RandomForestSurrogate

# Train surrogate after initial population
initial_pop = generator.generate_population(50)
initial_fitness = [evaluator.evaluate(g) for g in initial_pop]

surrogate = RandomForestSurrogate(num_trees=10)
surrogate.train(initial_pop, initial_fitness)

# Use surrogate for fast pre-screening
candidates = [generate_candidate() for _ in range(1000)]
predictions = [surrogate.predict(c) for c in candidates]

# Fully evaluate only top candidates
top_candidates = sorted(zip(candidates, predictions),
                       key=lambda x: x[1].combined_score,
                       reverse=True)[:50]
```

### Parallel Evaluation

```python
from concurrent.futures import ProcessPoolExecutor

def evaluate_genome(genome):
    evaluator = FitnessEvaluator()
    return evaluator.evaluate(genome)

# Parallel fitness evaluation
with ProcessPoolExecutor(max_workers=8) as executor:
    fitness_scores = list(executor.map(evaluate_genome, population))
```

## Testing

The framework includes comprehensive tests:

```bash
# Run all evolutionary search tests
uv run pytest tests/evolutionary_search/ -v

# Run specific test modules
uv run pytest tests/evolutionary_search/test_evolution_engine.py -v
uv run pytest tests/evolutionary_search/test_optimization.py -v
uv run pytest tests/evolutionary_search/test_integration.py -v

# Run with coverage
uv run pytest tests/evolutionary_search/ --cov=evolutionary_search --cov-report=term-missing
```

## References

1. **NSGA-II**: Deb et al., "A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II" (2002)
2. **NSGA-III**: Deb & Jain, "An Evolutionary Many-Objective Optimization Algorithm Using Reference-Point-Based Nondominated Sorting Approach" (2014)
3. **NAS**: Elsken et al., "Neural Architecture Search: A Survey" (2019)
4. **Apple Silicon Optimization**: Apple MLX documentation

## Contributing

When adding new components:

1. Follow existing code structure
2. Add comprehensive tests (>90% coverage)
3. Update documentation
4. Ensure Apple Silicon compatibility
5. Run full test suite before committing

## License

Part of EfficientAI-MLX-Toolkit
