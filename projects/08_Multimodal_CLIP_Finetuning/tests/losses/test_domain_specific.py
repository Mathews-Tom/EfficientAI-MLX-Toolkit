#!/usr/bin/env python3
"""Tests for domain-specific contrastive loss."""

from __future__ import annotations

import pytest
import torch

from src.losses.domain_specific import DomainSpecificLoss


class TestDomainSpecificLoss:
    """Tests for DomainSpecificLoss."""

    def test_initialization(self):
        """Test loss initialization."""
        loss_fn = DomainSpecificLoss(
            domain="medical",
            temperature=0.07,
            domain_weight=1.5,
        )
        assert loss_fn.domain == "medical"
        assert loss_fn.domain_weight == 1.5

    def test_initialization_all_domains(self):
        """Test initialization with all valid domains."""
        domains = ["general", "medical", "industrial", "scientific"]

        for domain in domains:
            loss_fn = DomainSpecificLoss(domain=domain)
            assert loss_fn.domain == domain

    def test_initialization_invalid_domain(self):
        """Test initialization with invalid domain."""
        with pytest.raises(ValueError, match="Domain must be one of"):
            DomainSpecificLoss(domain="invalid")

    def test_initialization_invalid_params(self):
        """Test initialization with invalid parameters."""
        # Invalid temperature
        with pytest.raises(ValueError, match="Temperature must be positive"):
            DomainSpecificLoss(domain="medical", temperature=-0.01)

        # Invalid domain weight
        with pytest.raises(ValueError, match="Domain weight must be"):
            DomainSpecificLoss(domain="medical", domain_weight=0.5)

    def test_forward_general_domain(self):
        """Test forward pass with general domain."""
        batch_size = 8
        embed_dim = 128

        loss_fn = DomainSpecificLoss(domain="general", temperature=0.07)

        image_embeds = torch.randn(batch_size, embed_dim)
        text_embeds = torch.randn(batch_size, embed_dim)

        output = loss_fn(image_embeds, text_embeds)

        # Check outputs
        assert "loss" in output
        assert "image_to_text_loss" in output
        assert "text_to_image_loss" in output
        assert "temperature" in output
        assert "domain" in output
        assert "domain_weight" in output

        assert output["loss"] > 0
        assert output["domain"] == "general"

    def test_forward_medical_domain(self):
        """Test forward pass with medical domain."""
        batch_size = 8
        embed_dim = 128

        loss_fn = DomainSpecificLoss(domain="medical", temperature=0.07, domain_weight=1.5)

        image_embeds = torch.randn(batch_size, embed_dim)
        text_embeds = torch.randn(batch_size, embed_dim)

        output = loss_fn(image_embeds, text_embeds)

        assert output["loss"] > 0
        assert output["domain"] == "medical"
        assert output["domain_weight"] == 1.5

    def test_forward_industrial_domain(self):
        """Test forward pass with industrial domain."""
        batch_size = 8
        embed_dim = 128

        loss_fn = DomainSpecificLoss(domain="industrial", temperature=0.07)

        image_embeds = torch.randn(batch_size, embed_dim)
        text_embeds = torch.randn(batch_size, embed_dim)

        output = loss_fn(image_embeds, text_embeds)

        assert output["loss"] > 0
        assert output["domain"] == "industrial"

    def test_forward_scientific_domain(self):
        """Test forward pass with scientific domain."""
        batch_size = 8
        embed_dim = 128

        loss_fn = DomainSpecificLoss(domain="scientific", temperature=0.07)

        image_embeds = torch.randn(batch_size, embed_dim)
        text_embeds = torch.randn(batch_size, embed_dim)

        output = loss_fn(image_embeds, text_embeds)

        assert output["loss"] > 0
        assert output["domain"] == "scientific"

    def test_domain_temperature_scaling(self):
        """Test that different domains use different temperature scales."""
        batch_size = 8
        embed_dim = 128

        image_embeds = torch.randn(batch_size, embed_dim)
        text_embeds = torch.randn(batch_size, embed_dim)

        # Medical domain should have lower effective temperature (sharper)
        loss_medical = DomainSpecificLoss(domain="medical", temperature=0.07)
        output_medical = loss_medical(image_embeds, text_embeds)

        # Scientific domain should have higher effective temperature (softer)
        loss_scientific = DomainSpecificLoss(domain="scientific", temperature=0.07)
        output_scientific = loss_scientific(image_embeds, text_embeds)

        # Temperature values should be different
        temp_medical = output_medical["temperature"]
        temp_scientific = output_scientific["temperature"]

        # Medical should be sharper (lower temp)
        assert temp_medical < temp_scientific

    def test_domain_weighting_effect(self):
        """Test that domain weighting affects loss values."""
        batch_size = 8
        embed_dim = 128

        image_embeds = torch.randn(batch_size, embed_dim)
        text_embeds = torch.randn(batch_size, embed_dim)

        # No domain weighting
        loss_no_weight = DomainSpecificLoss(domain="medical", domain_weight=1.0)
        output_no_weight = loss_no_weight(image_embeds, text_embeds)

        # With domain weighting
        loss_weighted = DomainSpecificLoss(domain="medical", domain_weight=2.0)
        output_weighted = loss_weighted(image_embeds, text_embeds)

        # Losses should be different due to weighting
        assert not torch.isclose(
            output_no_weight["loss"],
            output_weighted["loss"],
            rtol=0.01,
        )

    def test_forward_with_domain_labels(self):
        """Test forward pass with per-sample domain labels."""
        batch_size = 8
        embed_dim = 128

        loss_fn = DomainSpecificLoss(domain="medical", temperature=0.07)

        image_embeds = torch.randn(batch_size, embed_dim)
        text_embeds = torch.randn(batch_size, embed_dim)
        domain_labels = torch.zeros(batch_size, dtype=torch.long)

        # Should accept domain_labels parameter (even if not used yet)
        output = loss_fn(image_embeds, text_embeds, domain_labels=domain_labels)

        assert output["loss"] > 0

    def test_batch_size_1(self):
        """Test with batch size 1."""
        batch_size = 1
        embed_dim = 128

        loss_fn = DomainSpecificLoss(domain="medical", temperature=0.07)

        image_embeds = torch.randn(batch_size, embed_dim)
        text_embeds = torch.randn(batch_size, embed_dim)

        output = loss_fn(image_embeds, text_embeds)

        assert output["loss"] >= 0

    def test_large_batch(self):
        """Test with large batch size."""
        batch_size = 64
        embed_dim = 256

        loss_fn = DomainSpecificLoss(domain="industrial", temperature=0.07)

        image_embeds = torch.randn(batch_size, embed_dim)
        text_embeds = torch.randn(batch_size, embed_dim)

        output = loss_fn(image_embeds, text_embeds)

        assert output["loss"] > 0

    def test_gradient_flow(self):
        """Test that gradients flow properly."""
        batch_size = 4
        embed_dim = 128

        loss_fn = DomainSpecificLoss(domain="medical", temperature=0.07)

        image_embeds = torch.randn(batch_size, embed_dim, requires_grad=True)
        text_embeds = torch.randn(batch_size, embed_dim, requires_grad=True)

        output = loss_fn(image_embeds, text_embeds)
        loss = output["loss"]

        loss.backward()

        # Check gradients exist and are non-zero
        assert image_embeds.grad is not None
        assert text_embeds.grad is not None
        assert image_embeds.grad.abs().sum() > 0
        assert text_embeds.grad.abs().sum() > 0

    def test_invalid_shapes(self):
        """Test with invalid input shapes."""
        loss_fn = DomainSpecificLoss(domain="medical", temperature=0.07)

        # Wrong dimensions
        with pytest.raises(ValueError, match="Expected 2D embeddings"):
            loss_fn(torch.randn(4, 8, 16), torch.randn(4, 8, 16))

        # Mismatched batch sizes
        with pytest.raises(ValueError, match="Batch size mismatch"):
            loss_fn(torch.randn(4, 128), torch.randn(8, 128))

    def test_deterministic(self):
        """Test deterministic behavior."""
        batch_size = 4
        embed_dim = 128

        loss_fn = DomainSpecificLoss(domain="medical", temperature=0.07)

        image_embeds = torch.randn(batch_size, embed_dim)
        text_embeds = torch.randn(batch_size, embed_dim)

        output1 = loss_fn(image_embeds, text_embeds)
        output2 = loss_fn(image_embeds, text_embeds)

        # Should produce identical results
        assert torch.equal(output1["loss"], output2["loss"])

    def test_compare_domains(self):
        """Test that different domains produce different losses."""
        batch_size = 8
        embed_dim = 128

        image_embeds = torch.randn(batch_size, embed_dim)
        text_embeds = torch.randn(batch_size, embed_dim)

        domains = ["general", "medical", "industrial", "scientific"]
        outputs = []

        for domain in domains:
            loss_fn = DomainSpecificLoss(domain=domain, temperature=0.07)
            outputs.append(loss_fn(image_embeds, text_embeds))

        # All domains should produce valid losses
        for output in outputs:
            assert output["loss"] > 0

        # At least some losses should be different
        losses = [output["loss"] for output in outputs]
        assert len(set(l.item() for l in losses)) > 1

    def test_extra_repr(self):
        """Test string representation."""
        loss_fn = DomainSpecificLoss(
            domain="medical",
            temperature=0.07,
            domain_weight=1.5,
        )
        repr_str = loss_fn.extra_repr()

        assert "domain=medical" in repr_str
        assert "temperature=" in repr_str
        assert "domain_weight=" in repr_str


@pytest.mark.apple_silicon
class TestDomainSpecificLossMPS:
    """Tests for domain-specific loss on MPS device."""

    def test_forward_mps(self):
        """Test forward pass on MPS device."""
        if not torch.backends.mps.is_available():
            pytest.skip("MPS not available")

        batch_size = 8
        embed_dim = 128

        loss_fn = DomainSpecificLoss(domain="medical", temperature=0.07).to("mps")

        image_embeds = torch.randn(batch_size, embed_dim, device="mps")
        text_embeds = torch.randn(batch_size, embed_dim, device="mps")

        output = loss_fn(image_embeds, text_embeds)

        assert output["loss"].device.type == "mps"
        assert output["loss"] > 0
