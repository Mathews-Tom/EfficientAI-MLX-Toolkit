"""
Tests for baseline benchmarking of standard diffusion architectures.
"""

from __future__ import annotations

import pytest

from evolutionary_search.fitness import FitnessEvaluator
from evolutionary_search.population import PopulationGenerator
from evolutionary_search.search_space import (
    ArchitectureComponent,
    ArchitectureGenome,
    LayerConfig,
    SearchSpaceConfig,
)


class TestBaselineBenchmark:
    """Tests for baseline architecture benchmarks."""

    @pytest.fixture
    def search_space(self) -> SearchSpaceConfig:
        """Create search space for testing."""
        return SearchSpaceConfig(
            min_layers=4,
            max_layers=16,
            allowed_components=list(ArchitectureComponent),
        )

    @pytest.fixture
    def evaluator(self) -> FitnessEvaluator:
        """Create fitness evaluator for testing."""
        return FitnessEvaluator()

    def test_baseline_unet_architecture(self) -> None:
        """Test baseline UNet-like architecture."""
        # Create a simple UNet-like architecture
        layers = [
            # Encoder
            LayerConfig(
                component_type=ArchitectureComponent.CONV_BLOCK,
                parameters={"kernel_size": 3, "stride": 2, "padding": 1},
                layer_index=0,
                in_channels=3,
                out_channels=64,
            ),
            LayerConfig(
                component_type=ArchitectureComponent.RESIDUAL_BLOCK,
                parameters={"kernel_size": 3, "num_layers": 2},
                layer_index=1,
                in_channels=64,
                out_channels=128,
            ),
            # Bottleneck with attention
            LayerConfig(
                component_type=ArchitectureComponent.ATTENTION_BLOCK,
                parameters={"num_heads": 8, "embed_dim": 128},
                layer_index=2,
                in_channels=128,
                out_channels=128,
            ),
            # Decoder
            LayerConfig(
                component_type=ArchitectureComponent.RESIDUAL_BLOCK,
                parameters={"kernel_size": 3, "num_layers": 2},
                layer_index=3,
                in_channels=128,
                out_channels=64,
            ),
            LayerConfig(
                component_type=ArchitectureComponent.CONV_BLOCK,
                parameters={"kernel_size": 3, "stride": 1, "padding": 1},
                layer_index=4,
                in_channels=64,
                out_channels=3,
            ),
        ]

        genome = ArchitectureGenome(
            layers=layers,
            parameters={"architecture_type": "unet_baseline"},
        )

        assert genome.validate()
        assert genome.count_parameters() > 0

    def test_baseline_simple_cnn(self) -> None:
        """Test baseline simple CNN architecture."""
        layers = [
            LayerConfig(
                component_type=ArchitectureComponent.CONV_BLOCK,
                parameters={"kernel_size": 3, "stride": 1, "padding": 1},
                layer_index=i,
                in_channels=3 if i == 0 else 64,
                out_channels=64,
            )
            for i in range(4)
        ]

        genome = ArchitectureGenome(
            layers=layers, parameters={"architecture_type": "simple_cnn"}
        )

        assert genome.validate()
        assert len(genome.layers) == 4

    def test_baseline_resnet_like(self) -> None:
        """Test baseline ResNet-like architecture with skip connections."""
        layers = [
            LayerConfig(
                component_type=ArchitectureComponent.RESIDUAL_BLOCK,
                parameters={"kernel_size": 3, "num_layers": 2},
                layer_index=i,
                in_channels=64,
                out_channels=64,
            )
            for i in range(6)
        ]

        # Add skip connections
        connections = [(i, i + 1) for i in range(5)]
        connections.extend([(0, 3), (1, 4), (2, 5)])  # Long skip connections

        genome = ArchitectureGenome(
            layers=layers,
            connections=connections,
            parameters={"architecture_type": "resnet_baseline"},
        )

        assert genome.validate()
        assert len([c for c in genome.connections if c[1] - c[0] > 1]) == 3

    def test_evaluate_baseline_architectures(self, evaluator: FitnessEvaluator) -> None:
        """Test evaluating multiple baseline architectures."""
        # Create different baseline architectures
        architectures = []

        # Small efficient architecture
        small_layers = [
            LayerConfig(
                component_type=ArchitectureComponent.CONV_BLOCK,
                parameters={"kernel_size": 3, "stride": 1, "padding": 1},
                layer_index=i,
                in_channels=3 if i == 0 else 32,
                out_channels=32,
            )
            for i in range(4)
        ]
        architectures.append(
            ArchitectureGenome(
                layers=small_layers, parameters={"architecture_type": "small"}
            )
        )

        # Large capacity architecture
        large_layers = [
            LayerConfig(
                component_type=ArchitectureComponent.CONV_BLOCK,
                parameters={"kernel_size": 3, "stride": 1, "padding": 1},
                layer_index=i,
                in_channels=3 if i == 0 else 256,
                out_channels=256,
            )
            for i in range(8)
        ]
        architectures.append(
            ArchitectureGenome(
                layers=large_layers, parameters={"architecture_type": "large"}
            )
        )

        # Evaluate all
        results = []
        for genome in architectures:
            metrics = evaluator.evaluate(genome)
            results.append(
                {
                    "type": genome.parameters["architecture_type"],
                    "params": genome.count_parameters(),
                    "fitness": metrics.combined_score,
                    "quality": metrics.quality_score,
                    "speed": metrics.speed_score,
                    "memory": metrics.memory_score,
                }
            )

        # Check that we got results for both
        assert len(results) == 2
        assert all(r["fitness"] > 0 for r in results)

        # Small architecture should be faster and more memory efficient
        small_result = next(r for r in results if r["type"] == "small")
        large_result = next(r for r in results if r["type"] == "large")

        assert small_result["params"] < large_result["params"]
        assert small_result["speed"] >= large_result["speed"]
        assert small_result["memory"] >= large_result["memory"]

    def test_benchmark_population_diversity(
        self, search_space: SearchSpaceConfig, evaluator: FitnessEvaluator
    ) -> None:
        """Test benchmarking a diverse population."""
        generator = PopulationGenerator(search_space, seed=42)
        population = generator.generate_population(20)

        # Evaluate all genomes
        metrics_list = [evaluator.evaluate(genome) for genome in population]

        # Check diversity in fitness scores
        fitness_scores = [m.combined_score for m in metrics_list]

        assert len(fitness_scores) == 20
        assert min(fitness_scores) >= 0.0
        assert max(fitness_scores) <= 1.0

        # Should have some variation
        assert len(set(fitness_scores)) > 5

    def test_benchmark_parameter_count_correlation(
        self, search_space: SearchSpaceConfig, evaluator: FitnessEvaluator
    ) -> None:
        """Test correlation between parameter count and fitness."""
        generator = PopulationGenerator(search_space, seed=42)
        population = generator.generate_population(30)

        results = []
        for genome in population:
            metrics = evaluator.evaluate(genome)
            results.append(
                {
                    "params": genome.count_parameters(),
                    "quality": metrics.quality_score,
                    "speed": metrics.speed_score,
                }
            )

        # More parameters should generally correlate with:
        # - Higher quality (more capacity)
        # - Lower speed (more computation)

        small_archs = sorted(results, key=lambda x: x["params"])[:10]
        large_archs = sorted(results, key=lambda x: x["params"], reverse=True)[:10]

        avg_small_quality = sum(a["quality"] for a in small_archs) / len(small_archs)
        avg_large_quality = sum(a["quality"] for a in large_archs) / len(large_archs)

        avg_small_speed = sum(a["speed"] for a in small_archs) / len(small_archs)
        avg_large_speed = sum(a["speed"] for a in large_archs) / len(large_archs)

        # Larger models should have higher estimated quality
        assert avg_large_quality >= avg_small_quality

        # Smaller models should be faster
        assert avg_small_speed >= avg_large_speed

    def test_benchmark_attention_impact(self, evaluator: FitnessEvaluator) -> None:
        """Test impact of attention blocks on fitness."""
        # Architecture without attention
        layers_no_attention = [
            LayerConfig(
                component_type=ArchitectureComponent.CONV_BLOCK,
                parameters={"kernel_size": 3, "stride": 1, "padding": 1},
                layer_index=i,
                in_channels=64,
                out_channels=64,
            )
            for i in range(6)
        ]
        genome_no_attention = ArchitectureGenome(layers=layers_no_attention)

        # Architecture with attention
        layers_with_attention = [
            LayerConfig(
                component_type=(
                    ArchitectureComponent.ATTENTION_BLOCK
                    if i % 2 == 0
                    else ArchitectureComponent.CONV_BLOCK
                ),
                parameters=(
                    {"num_heads": 8, "embed_dim": 64}
                    if i % 2 == 0
                    else {"kernel_size": 3, "stride": 1, "padding": 1}
                ),
                layer_index=i,
                in_channels=64,
                out_channels=64,
            )
            for i in range(6)
        ]
        genome_with_attention = ArchitectureGenome(layers=layers_with_attention)

        metrics_no_attention = evaluator.evaluate(genome_no_attention)
        metrics_with_attention = evaluator.evaluate(genome_with_attention)

        # Attention should improve quality estimate
        assert metrics_with_attention.quality_score >= metrics_no_attention.quality_score

    def test_baseline_architecture_validation(self, search_space: SearchSpaceConfig) -> None:
        """Test that baseline architectures satisfy search space constraints."""
        # Generate various baseline architectures
        generator = PopulationGenerator(search_space, seed=42)

        small_genome = generator._generate_small_architecture()
        large_genome = generator._generate_large_architecture()
        random_genome = generator.generate_random_genome()

        # All should validate against search space
        assert search_space.validate_genome(small_genome)
        assert search_space.validate_genome(large_genome)
        assert search_space.validate_genome(random_genome)

    def test_benchmark_reproducibility(
        self, search_space: SearchSpaceConfig, evaluator: FitnessEvaluator
    ) -> None:
        """Test that benchmarks are reproducible with same seed."""
        generator1 = PopulationGenerator(search_space, seed=123)
        generator2 = PopulationGenerator(search_space, seed=123)

        pop1 = generator1.generate_population(10)
        pop2 = generator2.generate_population(10)

        # Verify same architectures generated
        for g1, g2 in zip(pop1, pop2):
            assert len(g1.layers) == len(g2.layers)
            assert len(g1.connections) == len(g2.connections)

        # Verify same fitness evaluation
        metrics1 = [evaluator.evaluate(g) for g in pop1]
        metrics2 = [evaluator.evaluate(g) for g in pop2]

        # Should get same fitness scores with same architectures
        for m1, m2 in zip(metrics1, metrics2):
            assert abs(m1.combined_score - m2.combined_score) < 0.001
            assert abs(m1.quality_score - m2.quality_score) < 0.001
            assert abs(m1.speed_score - m2.speed_score) < 0.001
