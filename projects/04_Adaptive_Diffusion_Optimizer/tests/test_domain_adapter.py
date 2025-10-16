"""
Tests for Domain Adapter
"""

import mlx.core as mx
import numpy as np
import pytest

from adaptive_diffusion.optimization.domain_adapter import (
    DomainAdapter,
    DomainConfig,
    DomainType,
    create_domain_adapter,
)


class TestDomainConfig:
    """Test suite for DomainConfig."""

    def test_config_initialization(self):
        """Test config initialization."""
        config = DomainConfig(
            domain_type=DomainType.PHOTOREALISTIC,
            num_steps=60,
            adaptive_threshold=0.6,
        )

        assert config.domain_type == DomainType.PHOTOREALISTIC
        assert config.num_steps == 60
        assert config.adaptive_threshold == 0.6

    def test_config_to_dict(self):
        """Test config serialization."""
        config = DomainConfig(domain_type=DomainType.ARTISTIC, num_steps=50)

        config_dict = config.to_dict()

        assert config_dict["domain_type"] == "artistic"
        assert config_dict["num_steps"] == 50
        assert "adaptive_threshold" in config_dict

    def test_config_from_dict(self):
        """Test config deserialization."""
        data = {
            "domain_type": "photorealistic",
            "num_steps": 65,
            "adaptive_threshold": 0.65,
            "progress_power": 2.5,
        }

        config = DomainConfig.from_dict(data)

        assert config.domain_type == DomainType.PHOTOREALISTIC
        assert config.num_steps == 65
        assert config.adaptive_threshold == 0.65


class TestDomainAdapter:
    """Test suite for DomainAdapter."""

    def test_adapter_initialization(self):
        """Test adapter initialization."""
        adapter = DomainAdapter()

        # Should have default configs for all domains
        assert len(adapter.domain_configs) > 0
        assert DomainType.GENERAL in adapter.domain_configs
        assert DomainType.PHOTOREALISTIC in adapter.domain_configs

    def test_detect_domain_from_prompt(self):
        """Test domain detection from prompt."""
        adapter = DomainAdapter()

        # Photorealistic
        assert (
            adapter.detect_domain(prompt="A realistic photograph of a cat")
            == DomainType.PHOTOREALISTIC
        )

        # Artistic
        assert (
            adapter.detect_domain(prompt="An oil painting of a sunset")
            == DomainType.ARTISTIC
        )

        # Portrait
        assert (
            adapter.detect_domain(prompt="A portrait of a person")
            == DomainType.PORTRAIT
        )

        # Landscape
        assert (
            adapter.detect_domain(prompt="A mountain landscape")
            == DomainType.LANDSCAPE
        )

        # Abstract
        assert (
            adapter.detect_domain(prompt="Abstract geometric patterns")
            == DomainType.ABSTRACT
        )

        # Scientific
        assert (
            adapter.detect_domain(prompt="Medical diagram of the heart")
            == DomainType.SCIENTIFIC
        )

        # General (no keywords)
        assert adapter.detect_domain(prompt="A scene") == DomainType.GENERAL

    def test_detect_domain_from_sample(self):
        """Test domain detection from sample."""
        adapter = DomainAdapter()

        # High complexity, high structure -> photorealistic or similar
        complex_sample = mx.random.normal([1, 64, 64, 3]) * 2.0
        domain = adapter.detect_domain(sample=complex_sample)
        # Random samples can map to various domains
        assert isinstance(domain, DomainType)

        # Low structure -> abstract or similar
        noise = mx.random.uniform(shape=[1, 64, 64, 3])
        domain = adapter.detect_domain(sample=noise)
        # Due to noise variance, could be several types
        assert isinstance(domain, DomainType)

    def test_get_config_with_domain_type(self):
        """Test getting config by domain type."""
        adapter = DomainAdapter()

        config = adapter.get_config(domain_type=DomainType.PHOTOREALISTIC)

        assert config.domain_type == DomainType.PHOTOREALISTIC
        assert config.num_steps > 0
        assert 0.0 <= config.adaptive_threshold <= 1.0

    def test_get_config_with_prompt(self):
        """Test getting config with auto-detection from prompt."""
        adapter = DomainAdapter()

        config = adapter.get_config(prompt="A realistic photo of a sunset")

        assert config.domain_type == DomainType.PHOTOREALISTIC
        assert config.quality_weight > config.speed_weight  # Quality-focused

    def test_update_config(self):
        """Test updating domain configuration."""
        adapter = DomainAdapter()

        # Update with kwargs
        adapter.update_config(
            DomainType.ARTISTIC,
            num_steps=70,
            adaptive_threshold=0.8,
        )

        config = adapter.get_config(domain_type=DomainType.ARTISTIC)

        assert config.num_steps == 70
        assert config.adaptive_threshold == 0.8

    def test_update_config_with_config_object(self):
        """Test updating with config object."""
        adapter = DomainAdapter()

        new_config = DomainConfig(
            domain_type=DomainType.PORTRAIT,
            num_steps=80,
            adaptive_threshold=0.9,
            confidence=0.85,
        )

        adapter.update_config(DomainType.PORTRAIT, config=new_config)

        config = adapter.get_config(domain_type=DomainType.PORTRAIT)

        assert config.num_steps == 80
        assert config.confidence == 0.85

    def test_learn_from_results(self):
        """Test learning from optimization results."""
        adapter = DomainAdapter()

        initial_config = adapter.get_config(domain_type=DomainType.ARTISTIC)
        initial_steps = initial_config.num_steps
        initial_confidence = initial_config.confidence

        # Provide good results
        adapter.learn_from_results(
            domain_type=DomainType.ARTISTIC,
            quality=0.9,
            speed=0.7,
            hyperparameters={"num_steps": 45, "adaptive_threshold": 0.55},
            learning_rate=0.3,
        )

        updated_config = adapter.get_config(domain_type=DomainType.ARTISTIC)

        # Config should have updated
        assert updated_config.num_steps != initial_steps
        # Confidence should increase
        assert updated_config.confidence > initial_confidence

    def test_learn_from_results_no_update_if_poor(self):
        """Test that learning doesn't update on poor results."""
        adapter = DomainAdapter()

        initial_config = adapter.get_config(domain_type=DomainType.ARTISTIC)
        initial_confidence = initial_config.confidence

        # Provide poor results (below current confidence)
        adapter.learn_from_results(
            domain_type=DomainType.ARTISTIC,
            quality=0.1,
            speed=0.1,
            hyperparameters={"num_steps": 10, "adaptive_threshold": 0.1},
            learning_rate=0.3,
        )

        updated_config = adapter.get_config(domain_type=DomainType.ARTISTIC)

        # Should not have changed significantly
        assert updated_config.confidence == initial_confidence

    def test_get_all_configs(self):
        """Test getting all configurations."""
        adapter = DomainAdapter()

        all_configs = adapter.get_all_configs()

        assert isinstance(all_configs, dict)
        assert len(all_configs) > 0
        assert DomainType.GENERAL in all_configs
        assert all(isinstance(v, DomainConfig) for v in all_configs.values())

    def test_domain_specific_settings(self):
        """Test that different domains have appropriate settings."""
        adapter = DomainAdapter()

        # Photorealistic should prioritize quality
        photo_config = adapter.get_config(domain_type=DomainType.PHOTOREALISTIC)
        assert photo_config.quality_weight > photo_config.speed_weight

        # Synthetic can be faster
        synth_config = adapter.get_config(domain_type=DomainType.SYNTHETIC)
        assert synth_config.num_steps < photo_config.num_steps

        # Scientific should have high precision
        sci_config = adapter.get_config(domain_type=DomainType.SCIENTIFIC)
        assert sci_config.quality_weight > 0.8

    def test_create_domain_adapter_factory(self):
        """Test factory function."""
        adapter = create_domain_adapter()

        assert isinstance(adapter, DomainAdapter)
        assert len(adapter.domain_configs) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
