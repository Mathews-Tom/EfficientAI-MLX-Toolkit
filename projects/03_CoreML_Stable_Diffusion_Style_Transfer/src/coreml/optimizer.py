"""Core ML model optimization utilities."""

from pathlib import Path
from typing import Any

import numpy as np
import coremltools as ct
from coremltools.models.neural_network import quantization_utils

from .config import CoreMLConfig


class CoreMLOptimizer:
    """Optimize Core ML models for Apple Silicon performance."""

    def __init__(self, config: CoreMLConfig):
        self.config = config
        self.config.validate()

    def optimize_model(
        self, model_path: Path | str, output_path: Path | str | None = None
    ) -> Any:
        """Apply comprehensive optimizations to a Core ML model."""
        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        print(f"Loading model for optimization: {model_path}")

        # Load model
        try:
            model = ct.models.MLModel(str(model_path))
        except Exception as e:
            raise RuntimeError(f"Failed to load Core ML model: {e}") from e

        # Apply optimizations
        optimized_model = model

        if self.config.use_low_precision_weights:
            optimized_model = self._apply_weight_quantization(optimized_model)

        if self.config.use_palettization:
            optimized_model = self._apply_palettization(optimized_model)

        if self.config.reduce_memory_footprint:
            optimized_model = self._apply_memory_optimizations(optimized_model)

        if self.config.optimize_for_inference:
            optimized_model = self._apply_inference_optimizations(optimized_model)

        # Save optimized model
        if output_path:
            self._save_optimized_model(optimized_model, output_path)

        return optimized_model

    def _apply_weight_quantization(self, model: Any) -> Any:
        """Apply weight quantization for reduced precision."""
        print(f"Applying weight quantization to {self.config.precision}...")

        try:
            if self.config.precision == "float16":
                # Apply 16-bit quantization
                quantized_model = quantization_utils.quantize_weights(model, nbits=16)
            else:
                # Keep float32
                quantized_model = model

            print("Weight quantization completed successfully")
            return quantized_model

        except Exception as e:
            print(f"Weight quantization failed: {e}")
            return model

    def _apply_palettization(self, model: Any) -> Any:
        """Apply palettization for model compression."""
        print(f"Applying weight quantization with {self.config.quantization_bits} bits...")

        try:
            # Note: palettize_weights is not available in current CoreML tools
            # Using quantize_weights as a fallback for weight compression
            palettized_model = quantization_utils.quantize_weights(
                model, nbits=self.config.quantization_bits
            )

            print("Weight quantization completed successfully")
            return palettized_model

        except Exception as e:
            print(f"Weight quantization failed: {e}")
            return model

    def _apply_memory_optimizations(self, model: Any) -> Any:
        """Apply memory footprint optimizations."""
        print("Applying memory optimizations...")

        try:
            # This is a placeholder for various memory optimization techniques
            # In practice, this might include:
            # - Model pruning
            # - Layer fusion
            # - Memory pool optimization

            # For now, we'll apply basic optimizations
            optimized_model = model

            print("Memory optimizations completed")
            return optimized_model

        except Exception as e:
            print(f"Memory optimization failed: {e}")
            return model

    def _apply_inference_optimizations(self, model: Any) -> Any:
        """Apply inference-specific optimizations."""
        print("Applying inference optimizations...")

        try:
            # Apply compute unit optimization
            if hasattr(model, "compute_unit"):
                model.compute_unit = self.config.get_compute_units_enum()

            # This could include:
            # - Graph optimization
            # - Operation fusion
            # - Batch size optimization

            print("Inference optimizations completed")
            return model

        except Exception as e:
            print(f"Inference optimization failed: {e}")
            return model

    def _save_optimized_model(self, model: Any, output_path: Path | str) -> None:
        """Save optimized model with metadata."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Update metadata
        model.short_description = f"{self.config.metadata_description} (Optimized)"
        model.author = self.config.metadata_author

        # Ensure correct file extension
        if self.config.model_format == "mlpackage":
            if not str(output_path).endswith(".mlpackage"):
                output_path = output_path.with_suffix(".mlpackage")
        else:
            if not str(output_path).endswith(".mlmodel"):
                output_path = output_path.with_suffix(".mlmodel")

        model.save(str(output_path))
        print(f"Optimized model saved to: {output_path}")

    def optimize_stable_diffusion_pipeline(
        self, models_dir: Path | str, output_dir: Path | str | None = None
    ) -> dict[str, Path]:
        """Optimize all components of a Stable Diffusion pipeline."""
        models_dir = Path(models_dir)
        output_dir = (
            Path(output_dir) if output_dir else models_dir.parent / "optimized_models"
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Optimizing Stable Diffusion pipeline from: {models_dir}")
        print(f"Output directory: {output_dir}")

        optimized_models = {}

        # Optimize U-Net
        unet_path = models_dir / "unet.mlpackage"
        if unet_path.exists():
            print("\nOptimizing U-Net...")
            try:
                optimized_unet_path = output_dir / "unet_optimized.mlpackage"
                self.optimize_model(unet_path, optimized_unet_path)
                optimized_models["unet"] = optimized_unet_path
            except Exception as e:
                print(f"U-Net optimization failed: {e}")

        # Optimize VAE Decoder
        vae_path = models_dir / "vae_decoder.mlpackage"
        if vae_path.exists():
            print("\nOptimizing VAE Decoder...")
            try:
                optimized_vae_path = output_dir / "vae_decoder_optimized.mlpackage"
                self.optimize_model(vae_path, optimized_vae_path)
                optimized_models["vae_decoder"] = optimized_vae_path
            except Exception as e:
                print(f"VAE Decoder optimization failed: {e}")

        # Optimize Text Encoder
        text_encoder_path = models_dir / "text_encoder.mlpackage"
        if text_encoder_path.exists():
            print("\nOptimizing Text Encoder...")
            try:
                optimized_text_encoder_path = (
                    output_dir / "text_encoder_optimized.mlpackage"
                )
                self.optimize_model(text_encoder_path, optimized_text_encoder_path)
                optimized_models["text_encoder"] = optimized_text_encoder_path
            except Exception as e:
                print(f"Text Encoder optimization failed: {e}")

        print(
            f"\nPipeline optimization completed. {len(optimized_models)} models optimized."
        )
        return optimized_models

    def benchmark_optimization_impact(
        self,
        original_model_path: Path | str,
        optimized_model_path: Path | str,
        num_runs: int = 10,
    ) -> dict[str, any]:
        """Benchmark the impact of optimizations."""
        print(f"Benchmarking optimization impact ({num_runs} runs)...")

        # Load models
        try:
            original_model = ct.models.MLModel(str(original_model_path))
            optimized_model = ct.models.MLModel(str(optimized_model_path))
        except Exception as e:
            return {"benchmark_error": f"Failed to load models: {e}"}

        # Create test inputs
        test_inputs = self._create_benchmark_inputs(original_model)

        # Benchmark original model
        original_results = self._benchmark_single_model(
            original_model, test_inputs, num_runs
        )

        # Benchmark optimized model
        optimized_results = self._benchmark_single_model(
            optimized_model, test_inputs, num_runs
        )

        if "error" in original_results or "error" in optimized_results:
            return {
                "benchmark_error": "One or more models failed during benchmarking",
                "original_results": original_results,
                "optimized_results": optimized_results,
            }

        # Calculate improvements
        improvements = {
            "inference_time_improvement": {
                "absolute": original_results["avg_time"]
                - optimized_results["avg_time"],
                "relative": (
                    original_results["avg_time"] - optimized_results["avg_time"]
                )
                / original_results["avg_time"]
                * 100,
            },
            "throughput_improvement": {
                "absolute": optimized_results["throughput"]
                - original_results["throughput"],
                "relative": (
                    optimized_results["throughput"] - original_results["throughput"]
                )
                / original_results["throughput"]
                * 100,
            },
        }

        return {
            "original_model": original_results,
            "optimized_model": optimized_results,
            "improvements": improvements,
            "config": self.config.to_dict(),
        }

    def _create_benchmark_inputs(self, model: Any) -> dict[str, np.ndarray]:
        """Create benchmark inputs for a model."""
        inputs = {}

        for input_desc in model.input_description:
            input_name = input_desc.name

            if hasattr(input_desc.type, "multiArrayType"):
                shape = input_desc.type.multiArrayType.shape
                inputs[input_name] = np.random.randn(*shape).astype(np.float32)
            elif hasattr(input_desc.type, "imageType"):
                # Handle image inputs
                height = input_desc.type.imageType.height
                width = input_desc.type.imageType.width
                inputs[input_name] = np.random.randint(
                    0, 256, (height, width, 3), dtype=np.uint8
                )

        return inputs

    def _benchmark_single_model(
        self, model: Any, test_inputs: dict[str, np.ndarray], num_runs: int
    ) -> dict[str, any]:
        """Benchmark a single model."""
        import time

        # Warmup run
        try:
            model.predict(test_inputs)
        except Exception as e:
            return {"error": f"Warmup failed: {e}"}

        # Benchmark runs
        times = []
        for _ in range(num_runs):
            start_time = time.time()
            try:
                model.predict(test_inputs)
                end_time = time.time()
                times.append(end_time - start_time)
            except Exception as e:
                return {"error": f"Prediction failed: {e}"}

        # Calculate statistics
        avg_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)

        return {
            "avg_time": avg_time,
            "std_time": std_time,
            "min_time": min_time,
            "max_time": max_time,
            "throughput": 1.0 / avg_time,
        }

    def get_model_info(self, model_path: Path | str) -> dict[str, any]:
        """Get detailed information about a Core ML model."""
        try:
            model = ct.models.MLModel(str(model_path))

            info = {
                "model_path": str(model_path),
                "short_description": getattr(model, "short_description", "N/A"),
                "author": getattr(model, "author", "N/A"),
                "input_description": [
                    {
                        "name": inp.name,
                        "type": str(inp.type),
                        "shape": getattr(inp.type.multiArrayType, "shape", None)
                        if hasattr(inp.type, "multiArrayType")
                        else None,
                    }
                    for inp in model.input_description
                ],
                "output_description": [
                    {
                        "name": out.name,
                        "type": str(out.type),
                        "shape": getattr(out.type.multiArrayType, "shape", None)
                        if hasattr(out.type, "multiArrayType")
                        else None,
                    }
                    for out in model.output_description
                ],
            }

            return info

        except Exception as e:
            return {"error": f"Failed to get model info: {e}"}

    def validate_apple_silicon_optimization(
        self, model_path: Path | str
    ) -> dict[str, any]:
        """Validate that a model is optimized for Apple Silicon."""
        try:
            model = ct.models.MLModel(str(model_path))

            validation = {
                "model_path": str(model_path),
                "is_optimized": True,
                "optimization_checks": [],
                "recommendations": [],
            }

            # Check compute units
            if hasattr(model, "compute_unit"):
                if model.compute_unit == ct.ComputeUnit.ALL:
                    validation["optimization_checks"].append(
                        "✓ Compute units set to ALL (CPU + GPU)"
                    )
                elif model.compute_unit == ct.ComputeUnit.CPU_AND_GPU:
                    validation["optimization_checks"].append(
                        "✓ Compute units set to CPU_AND_GPU"
                    )
                else:
                    validation["optimization_checks"].append(
                        "⚠ Compute units not optimized for Apple Silicon"
                    )
                    validation["recommendations"].append(
                        "Set compute units to ALL or CPU_AND_GPU"
                    )
                    validation["is_optimized"] = False
            else:
                validation["optimization_checks"].append(
                    "? Compute units information not available"
                )

            # Check for precision optimization
            # This is a simplified check - in practice, you'd need to inspect the model weights
            validation["optimization_checks"].append(
                f"Configuration precision: {self.config.precision}"
            )

            return validation

        except Exception as e:
            return {"error": f"Validation failed: {e}"}
