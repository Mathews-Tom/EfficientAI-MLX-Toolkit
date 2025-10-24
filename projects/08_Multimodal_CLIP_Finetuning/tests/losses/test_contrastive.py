#!/usr/bin/env python3
"""Tests for standard CLIP contrastive loss."""

from __future__ import annotations

import pytest
import torch

from src.losses.contrastive import CLIPContrastiveLoss


class TestCLIPContrastiveLoss:
    """Tests for CLIPContrastiveLoss."""

    def test_initialization(self):
        """Test loss initialization."""
        loss_fn = CLIPContrastiveLoss(temperature=0.07)
        assert torch.isclose(loss_fn.temperature, torch.tensor(0.07), atol=1e-6)

    def test_initialization_learnable_temp(self):
        """Test initialization with learnable temperature."""
        loss_fn = CLIPContrastiveLoss(temperature=0.07, learnable_temp=True)
        assert isinstance(loss_fn.log_temperature, torch.nn.Parameter)
        assert torch.isclose(loss_fn.temperature, torch.tensor(0.07), atol=1e-6)

    def test_initialization_invalid_temperature(self):
        """Test initialization with invalid temperature."""
        with pytest.raises(ValueError, match="Temperature must be positive"):
            CLIPContrastiveLoss(temperature=-0.01)

        with pytest.raises(ValueError, match="Temperature must be positive"):
            CLIPContrastiveLoss(temperature=0.0)

    def test_forward_basic(self):
        """Test basic forward pass."""
        batch_size = 4
        embed_dim = 512

        loss_fn = CLIPContrastiveLoss(temperature=0.07)

        # Create random embeddings
        image_embeds = torch.randn(batch_size, embed_dim)
        text_embeds = torch.randn(batch_size, embed_dim)

        # Compute loss
        output = loss_fn(image_embeds, text_embeds)

        # Check outputs
        assert "loss" in output
        assert "image_to_text_loss" in output
        assert "text_to_image_loss" in output
        assert "temperature" in output
        assert "logits_per_image" in output
        assert "logits_per_text" in output

        # Check shapes and values
        assert output["loss"].shape == torch.Size([])
        assert output["loss"] > 0
        assert output["logits_per_image"].shape == (batch_size, batch_size)
        assert output["logits_per_text"].shape == (batch_size, batch_size)

    def test_forward_symmetric(self):
        """Test that loss is symmetric (i2t ≈ t2i)."""
        batch_size = 8
        embed_dim = 256

        loss_fn = CLIPContrastiveLoss(temperature=0.07)

        # Create random embeddings
        image_embeds = torch.randn(batch_size, embed_dim)
        text_embeds = torch.randn(batch_size, embed_dim)

        output = loss_fn(image_embeds, text_embeds)

        # Losses should be similar (but not exactly equal due to numerical differences)
        i2t_loss = output["image_to_text_loss"]
        t2i_loss = output["text_to_image_loss"]

        # They should be within reasonable range
        assert torch.isclose(i2t_loss, t2i_loss, rtol=0.1)

        # Total loss should be average
        expected_total = (i2t_loss + t2i_loss) / 2
        assert torch.isclose(output["loss"], expected_total)

    def test_forward_perfect_matches(self):
        """Test loss with perfect matching embeddings."""
        batch_size = 4
        embed_dim = 128

        loss_fn = CLIPContrastiveLoss(temperature=0.07)

        # Create identical embeddings (perfect matches)
        embeds = torch.randn(batch_size, embed_dim)

        output = loss_fn(embeds, embeds)

        # Loss should be very low (close to 0) for perfect matches
        # Not exactly 0 due to numerical precision
        assert output["loss"] < 0.1

    def test_forward_different_temperatures(self):
        """Test that different temperatures produce different losses."""
        batch_size = 4
        embed_dim = 128

        image_embeds = torch.randn(batch_size, embed_dim)
        text_embeds = torch.randn(batch_size, embed_dim)

        # Test with different temperatures
        loss_low = CLIPContrastiveLoss(temperature=0.01)(image_embeds, text_embeds)
        loss_mid = CLIPContrastiveLoss(temperature=0.07)(image_embeds, text_embeds)
        loss_high = CLIPContrastiveLoss(temperature=0.5)(image_embeds, text_embeds)

        # Losses should be different
        assert not torch.isclose(loss_low["loss"], loss_mid["loss"])
        assert not torch.isclose(loss_mid["loss"], loss_high["loss"])

    def test_forward_batch_size_1(self):
        """Test with batch size 1."""
        batch_size = 1
        embed_dim = 256

        loss_fn = CLIPContrastiveLoss(temperature=0.07)

        image_embeds = torch.randn(batch_size, embed_dim)
        text_embeds = torch.randn(batch_size, embed_dim)

        output = loss_fn(image_embeds, text_embeds)

        # With batch size 1, loss can be 0 if the single pair matches perfectly
        assert output["loss"] >= 0
        assert output["logits_per_image"].shape == (1, 1)

    def test_forward_large_batch(self):
        """Test with large batch size."""
        batch_size = 128
        embed_dim = 512

        loss_fn = CLIPContrastiveLoss(temperature=0.07)

        image_embeds = torch.randn(batch_size, embed_dim)
        text_embeds = torch.randn(batch_size, embed_dim)

        output = loss_fn(image_embeds, text_embeds)

        assert output["loss"] > 0
        assert output["logits_per_image"].shape == (batch_size, batch_size)

    def test_forward_invalid_shapes(self):
        """Test with invalid input shapes."""
        loss_fn = CLIPContrastiveLoss(temperature=0.07)

        # Wrong number of dimensions
        with pytest.raises(ValueError, match="Expected 2D embeddings"):
            loss_fn(torch.randn(4, 8, 16), torch.randn(4, 8, 16))

        # Mismatched batch sizes
        with pytest.raises(ValueError, match="Batch size mismatch"):
            loss_fn(torch.randn(4, 128), torch.randn(8, 128))

        # Mismatched embedding dimensions
        with pytest.raises(ValueError, match="Embedding dimension mismatch"):
            loss_fn(torch.randn(4, 128), torch.randn(4, 256))

    def test_gradient_flow(self):
        """Test that gradients flow properly."""
        batch_size = 4
        embed_dim = 128

        loss_fn = CLIPContrastiveLoss(temperature=0.07, learnable_temp=True)

        image_embeds = torch.randn(batch_size, embed_dim, requires_grad=True)
        text_embeds = torch.randn(batch_size, embed_dim, requires_grad=True)

        output = loss_fn(image_embeds, text_embeds)
        loss = output["loss"]

        # Backward pass
        loss.backward()

        # Check gradients exist
        assert image_embeds.grad is not None
        assert text_embeds.grad is not None
        assert loss_fn.log_temperature.grad is not None

        # Gradients should be non-zero
        assert image_embeds.grad.abs().sum() > 0
        assert text_embeds.grad.abs().sum() > 0

    def test_normalization(self):
        """Test that embeddings are properly normalized."""
        batch_size = 4
        embed_dim = 128

        loss_fn = CLIPContrastiveLoss(temperature=0.07)

        # Create unnormalized embeddings
        image_embeds = torch.randn(batch_size, embed_dim) * 10  # Large magnitude
        text_embeds = torch.randn(batch_size, embed_dim) * 5

        output = loss_fn(image_embeds, text_embeds)

        # Loss should still be reasonable (normalization handles large magnitudes)
        assert output["loss"] > 0
        assert torch.isfinite(output["loss"])

    def test_logits_range(self):
        """Test that logits are in reasonable range after temperature scaling."""
        batch_size = 4
        embed_dim = 128

        loss_fn = CLIPContrastiveLoss(temperature=0.07)

        image_embeds = torch.randn(batch_size, embed_dim)
        text_embeds = torch.randn(batch_size, embed_dim)

        output = loss_fn(image_embeds, text_embeds)

        # After normalization and temperature scaling, logits should be in reasonable range
        # Normalized dot products are in [-1, 1], divided by 0.07 gives roughly [-14, 14]
        assert output["logits_per_image"].abs().max() < 50

    def test_deterministic(self):
        """Test that same inputs produce same outputs."""
        batch_size = 4
        embed_dim = 128

        loss_fn = CLIPContrastiveLoss(temperature=0.07)

        image_embeds = torch.randn(batch_size, embed_dim)
        text_embeds = torch.randn(batch_size, embed_dim)

        # Run twice
        output1 = loss_fn(image_embeds, text_embeds)
        output2 = loss_fn(image_embeds, text_embeds)

        # Should be identical
        assert torch.equal(output1["loss"], output2["loss"])
        assert torch.equal(output1["logits_per_image"], output2["logits_per_image"])

    def test_extra_repr(self):
        """Test string representation."""
        loss_fn = CLIPContrastiveLoss(temperature=0.123)
        repr_str = loss_fn.extra_repr()

        assert "temperature=" in repr_str
        assert "0.123" in repr_str


@pytest.mark.apple_silicon
class TestCLIPContrastiveLossMPS:
    """Tests for CLIP contrastive loss on MPS device."""

    def test_forward_mps(self):
        """Test forward pass on MPS device."""
        if not torch.backends.mps.is_available():
            pytest.skip("MPS not available")

        batch_size = 4
        embed_dim = 128

        loss_fn = CLIPContrastiveLoss(temperature=0.07).to("mps")

        image_embeds = torch.randn(batch_size, embed_dim, device="mps")
        text_embeds = torch.randn(batch_size, embed_dim, device="mps")

        output = loss_fn(image_embeds, text_embeds)

        assert output["loss"].device.type == "mps"
        assert output["loss"] > 0
