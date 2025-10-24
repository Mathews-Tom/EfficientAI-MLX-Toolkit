#!/usr/bin/env python3
"""Tests for multi-scale contrastive loss."""

from __future__ import annotations

import pytest
import torch

from src.losses.multi_scale import MultiScaleLoss


class TestMultiScaleLoss:
    """Tests for MultiScaleLoss."""

    def test_initialization_default(self):
        """Test loss initialization with defaults."""
        loss_fn = MultiScaleLoss()
        assert loss_fn.scales == [1.0, 0.75, 0.5]
        assert loss_fn.num_scales == 3
        assert len(loss_fn.scale_weights) == 3

    def test_initialization_custom_scales(self):
        """Test initialization with custom scales."""
        scales = [1.0, 0.8, 0.6, 0.4]
        loss_fn = MultiScaleLoss(scales=scales, base_temperature=0.07)

        assert loss_fn.scales == scales
        assert loss_fn.num_scales == 4
        assert torch.isclose(loss_fn.base_temperature, torch.tensor(0.07), atol=1e-6)

    def test_initialization_custom_weights(self):
        """Test initialization with custom weights."""
        scales = [1.0, 0.75, 0.5]
        weights = [2.0, 1.5, 1.0]

        loss_fn = MultiScaleLoss(
            scales=scales,
            scale_weights=weights,
            normalize_weights=False,
        )

        assert torch.allclose(loss_fn.scale_weights, torch.tensor(weights))

    def test_initialization_normalized_weights(self):
        """Test that weights are normalized."""
        scales = [1.0, 0.75, 0.5]
        weights = [2.0, 2.0, 2.0]  # Sum = 6.0

        loss_fn = MultiScaleLoss(
            scales=scales,
            scale_weights=weights,
            normalize_weights=True,
        )

        # Should be normalized to sum to 1.0
        expected = torch.tensor([1 / 3, 1 / 3, 1 / 3])
        assert torch.allclose(loss_fn.scale_weights, expected, atol=1e-6)

    def test_initialization_invalid_params(self):
        """Test initialization with invalid parameters."""
        # Empty scales
        with pytest.raises(ValueError, match="Must provide at least one scale"):
            MultiScaleLoss(scales=[])

        # Negative scale
        with pytest.raises(ValueError, match="All scales must be positive"):
            MultiScaleLoss(scales=[1.0, -0.5])

        # Invalid temperature
        with pytest.raises(ValueError, match="Base temperature must be positive"):
            MultiScaleLoss(base_temperature=-0.01)

        # Mismatched weights
        with pytest.raises(ValueError, match="Number of weights"):
            MultiScaleLoss(scales=[1.0, 0.75], scale_weights=[1.0, 0.75, 0.5])

        # Negative weights
        with pytest.raises(ValueError, match="All weights must be non-negative"):
            MultiScaleLoss(scales=[1.0, 0.75], scale_weights=[1.0, -0.5])

        # Zero sum weights
        with pytest.raises(ValueError, match="Sum of weights cannot be zero"):
            MultiScaleLoss(
                scales=[1.0, 0.75],
                scale_weights=[0.0, 0.0],
                normalize_weights=True,
            )

    def test_forward_basic(self):
        """Test basic forward pass."""
        batch_size = 8
        embed_dim = 128

        loss_fn = MultiScaleLoss(scales=[1.0, 0.75, 0.5])

        image_embeds = torch.randn(batch_size, embed_dim)
        text_embeds = torch.randn(batch_size, embed_dim)

        output = loss_fn(image_embeds, text_embeds)

        # Check outputs
        assert "loss" in output
        assert "scale_losses" in output
        assert "scale_i2t_losses" in output
        assert "scale_t2i_losses" in output
        assert "base_temperature" in output
        assert "temperatures" in output
        assert "scale_weights" in output

        assert output["loss"] > 0
        assert output["scale_losses"].shape == (3,)
        assert output["scale_i2t_losses"].shape == (3,)
        assert output["scale_t2i_losses"].shape == (3,)

    def test_forward_single_scale(self):
        """Test forward with single scale (equivalent to standard loss)."""
        batch_size = 8
        embed_dim = 128

        loss_fn = MultiScaleLoss(scales=[1.0])

        image_embeds = torch.randn(batch_size, embed_dim)
        text_embeds = torch.randn(batch_size, embed_dim)

        output = loss_fn(image_embeds, text_embeds)

        assert output["loss"] > 0
        assert output["scale_losses"].shape == (1,)

    def test_forward_many_scales(self):
        """Test forward with many scales."""
        batch_size = 8
        embed_dim = 128

        scales = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
        loss_fn = MultiScaleLoss(scales=scales)

        image_embeds = torch.randn(batch_size, embed_dim)
        text_embeds = torch.randn(batch_size, embed_dim)

        output = loss_fn(image_embeds, text_embeds)

        assert output["loss"] > 0
        assert output["scale_losses"].shape == (len(scales),)

    def test_scale_losses_different(self):
        """Test that different scales produce different losses."""
        batch_size = 8
        embed_dim = 128

        loss_fn = MultiScaleLoss(scales=[1.0, 0.5, 0.25])

        image_embeds = torch.randn(batch_size, embed_dim)
        text_embeds = torch.randn(batch_size, embed_dim)

        output = loss_fn(image_embeds, text_embeds)

        # Losses at different scales should be different
        scale_losses = output["scale_losses"]
        assert not torch.allclose(scale_losses[0], scale_losses[1])
        assert not torch.allclose(scale_losses[1], scale_losses[2])

    def test_weighted_aggregation(self):
        """Test that loss is correctly weighted aggregation."""
        batch_size = 8
        embed_dim = 128

        scales = [1.0, 0.75, 0.5]
        weights = [2.0, 1.0, 0.5]

        loss_fn = MultiScaleLoss(
            scales=scales,
            scale_weights=weights,
            normalize_weights=True,
        )

        image_embeds = torch.randn(batch_size, embed_dim)
        text_embeds = torch.randn(batch_size, embed_dim)

        output = loss_fn(image_embeds, text_embeds)

        # Verify weighted sum
        scale_losses = output["scale_losses"]
        weights_tensor = output["scale_weights"]

        expected_loss = (scale_losses * weights_tensor).sum()
        assert torch.isclose(output["loss"], expected_loss)

    def test_temperatures_computed_correctly(self):
        """Test that temperatures are correctly computed."""
        base_temp = 0.07
        scales = [1.0, 0.75, 0.5]

        loss_fn = MultiScaleLoss(scales=scales, base_temperature=base_temp)

        batch_size = 4
        embed_dim = 64

        image_embeds = torch.randn(batch_size, embed_dim)
        text_embeds = torch.randn(batch_size, embed_dim)

        output = loss_fn(image_embeds, text_embeds)

        temperatures = output["temperatures"]

        # Check temperatures match expected values
        expected = torch.tensor([base_temp * s for s in scales])
        assert torch.allclose(temperatures, expected, atol=1e-6)

    def test_batch_size_1(self):
        """Test with batch size 1."""
        batch_size = 1
        embed_dim = 128

        loss_fn = MultiScaleLoss(scales=[1.0, 0.75, 0.5])

        image_embeds = torch.randn(batch_size, embed_dim)
        text_embeds = torch.randn(batch_size, embed_dim)

        output = loss_fn(image_embeds, text_embeds)

        assert output["loss"] >= 0

    def test_large_batch(self):
        """Test with large batch size."""
        batch_size = 64
        embed_dim = 256

        loss_fn = MultiScaleLoss(scales=[1.0, 0.75, 0.5])

        image_embeds = torch.randn(batch_size, embed_dim)
        text_embeds = torch.randn(batch_size, embed_dim)

        output = loss_fn(image_embeds, text_embeds)

        assert output["loss"] > 0

    def test_gradient_flow(self):
        """Test that gradients flow properly."""
        batch_size = 4
        embed_dim = 128

        loss_fn = MultiScaleLoss(scales=[1.0, 0.75, 0.5])

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
        loss_fn = MultiScaleLoss(scales=[1.0, 0.75])

        # Wrong dimensions
        with pytest.raises(ValueError, match="Expected 2D embeddings"):
            loss_fn(torch.randn(4, 8, 16), torch.randn(4, 8, 16))

        # Mismatched batch sizes
        with pytest.raises(ValueError, match="Batch size mismatch"):
            loss_fn(torch.randn(4, 128), torch.randn(8, 128))

        # Mismatched embedding dimensions
        with pytest.raises(ValueError, match="Embedding dimension mismatch"):
            loss_fn(torch.randn(4, 128), torch.randn(4, 256))

    def test_deterministic(self):
        """Test deterministic behavior."""
        batch_size = 4
        embed_dim = 128

        loss_fn = MultiScaleLoss(scales=[1.0, 0.75, 0.5])

        image_embeds = torch.randn(batch_size, embed_dim)
        text_embeds = torch.randn(batch_size, embed_dim)

        output1 = loss_fn(image_embeds, text_embeds)
        output2 = loss_fn(image_embeds, text_embeds)

        # Should produce identical results
        assert torch.equal(output1["loss"], output2["loss"])
        assert torch.equal(output1["scale_losses"], output2["scale_losses"])

    def test_scale_effect_on_loss(self):
        """Test that scale affects loss magnitude."""
        batch_size = 8
        embed_dim = 128

        image_embeds = torch.randn(batch_size, embed_dim)
        text_embeds = torch.randn(batch_size, embed_dim)

        # Fine-grained (low temperature)
        loss_fine = MultiScaleLoss(scales=[0.5])(image_embeds, text_embeds)

        # Coarse-grained (high temperature)
        loss_coarse = MultiScaleLoss(scales=[2.0])(image_embeds, text_embeds)

        # Losses should be different
        assert not torch.isclose(loss_fine["loss"], loss_coarse["loss"])

    def test_extra_repr(self):
        """Test string representation."""
        loss_fn = MultiScaleLoss(
            scales=[1.0, 0.75, 0.5],
            base_temperature=0.07,
            scale_weights=[1.0, 0.75, 0.5],
            normalize_weights=True,
        )
        repr_str = loss_fn.extra_repr()

        assert "base_temperature=" in repr_str
        assert "scales=" in repr_str
        assert "weights=" in repr_str


@pytest.mark.apple_silicon
class TestMultiScaleLossMPS:
    """Tests for multi-scale loss on MPS device."""

    def test_forward_mps(self):
        """Test forward pass on MPS device."""
        if not torch.backends.mps.is_available():
            pytest.skip("MPS not available")

        batch_size = 8
        embed_dim = 128

        loss_fn = MultiScaleLoss(scales=[1.0, 0.75, 0.5]).to("mps")

        image_embeds = torch.randn(batch_size, embed_dim, device="mps")
        text_embeds = torch.randn(batch_size, embed_dim, device="mps")

        output = loss_fn(image_embeds, text_embeds)

        assert output["loss"].device.type == "mps"
        assert output["loss"] > 0
