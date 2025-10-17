"""Gradient compression for efficient communication."""

from __future__ import annotations

import logging

import mlx.core as mx

logger = logging.getLogger(__name__)


class GradientCompressor:
    """Compresses gradients for efficient communication."""

    def __init__(
        self,
        compression_ratio: float = 0.1,
        use_quantization: bool = True,
        use_sparsification: bool = False,
    ):
        """Initialize gradient compressor.

        Args:
            compression_ratio: Target compression ratio
            use_quantization: Use quantization compression
            use_sparsification: Use sparsification (top-k)
        """
        self.compression_ratio = compression_ratio
        self.use_quantization = use_quantization
        self.use_sparsification = use_sparsification

    def compress(
        self, gradients: dict[str, mx.array]
    ) -> tuple[dict[str, mx.array], dict]:
        """Compress gradients.

        Args:
            gradients: Gradient dictionary

        Returns:
            Tuple of (compressed gradients, metadata for decompression)
        """
        compressed = {}
        metadata = {}

        for name, grad in gradients.items():
            if self.use_quantization:
                compressed[name], metadata[name] = self._quantize(grad)
            elif self.use_sparsification:
                compressed[name], metadata[name] = self._sparsify(grad)
            else:
                compressed[name] = grad
                metadata[name] = {}

        logger.debug(f"Compressed {len(gradients)} gradients")
        return compressed, metadata

    def decompress(
        self, compressed: dict[str, mx.array], metadata: dict
    ) -> dict[str, mx.array]:
        """Decompress gradients.

        Args:
            compressed: Compressed gradients
            metadata: Decompression metadata

        Returns:
            Decompressed gradients
        """
        decompressed = {}

        for name, comp_grad in compressed.items():
            meta = metadata.get(name, {})

            if "scale" in meta:  # Quantized
                decompressed[name] = comp_grad * meta["scale"]
            elif "indices" in meta:  # Sparsified
                decompressed[name] = self._desparsify(comp_grad, meta)
            else:
                decompressed[name] = comp_grad

        return decompressed

    def _quantize(
        self, gradient: mx.array, num_bits: int = 8
    ) -> tuple[mx.array, dict]:
        """Quantize gradient to reduce precision.

        Args:
            gradient: Gradient to quantize
            num_bits: Number of bits for quantization

        Returns:
            Tuple of (quantized gradient, metadata)
        """
        # Simple linear quantization
        min_val = float(mx.min(gradient))
        max_val = float(mx.max(gradient))

        if min_val == max_val:
            return mx.zeros_like(gradient), {"scale": 1.0, "min": min_val}

        scale = (max_val - min_val) / (2 ** num_bits - 1)

        quantized = mx.round((gradient - min_val) / scale)

        metadata = {
            "scale": scale,
            "min": min_val,
            "num_bits": num_bits,
        }

        return quantized, metadata

    def _sparsify(
        self, gradient: mx.array, k_ratio: float = 0.1
    ) -> tuple[mx.array, dict]:
        """Sparsify gradient by keeping top-k elements.

        Args:
            gradient: Gradient to sparsify
            k_ratio: Ratio of elements to keep

        Returns:
            Tuple of (sparse gradient, metadata)
        """
        k = max(1, int(gradient.size * k_ratio))

        # Flatten gradient
        flat_grad = mx.reshape(gradient, (-1,))

        # Get top-k indices
        abs_grad = mx.abs(flat_grad)
        top_k_indices = mx.argsort(abs_grad)[-k:]

        # Create sparse gradient
        sparse = mx.zeros_like(flat_grad)
        sparse[top_k_indices] = flat_grad[top_k_indices]

        # Reshape back
        sparse = mx.reshape(sparse, gradient.shape)

        metadata = {
            "indices": top_k_indices,
            "shape": gradient.shape,
            "k": k,
        }

        return sparse, metadata

    def _desparsify(self, sparse: mx.array, metadata: dict) -> mx.array:
        """Reconstruct from sparse representation.

        Args:
            sparse: Sparse gradient
            metadata: Sparsification metadata

        Returns:
            Reconstructed gradient
        """
        # Already in dense form from compression
        return sparse

    def estimate_compression_rate(
        self, gradients: dict[str, mx.array]
    ) -> float:
        """Estimate achieved compression rate.

        Args:
            gradients: Original gradients

        Returns:
            Compression rate
        """
        if self.use_quantization:
            return self.compression_ratio
        elif self.use_sparsification:
            return self.compression_ratio
        else:
            return 1.0
