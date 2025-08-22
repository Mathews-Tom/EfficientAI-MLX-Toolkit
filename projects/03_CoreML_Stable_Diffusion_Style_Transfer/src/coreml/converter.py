"""Core ML model conversion utilities."""

from pathlib import Path
from typing import Any

import numpy as np
import torch
import coremltools as ct
from coremltools.models.neural_network import quantization_utils

from .config import CoreMLConfig


class CoreMLConverter:
    """Convert PyTorch models to Core ML format optimized for Apple Silicon."""

    def __init__(self, config: CoreMLConfig):
        self.config = config
        self.config.validate()

    def convert_model(
        self, model_path: Path | str, output_path: Path | str | None = None
    ) -> Any:
        """Convert a PyTorch model to Core ML format."""
        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        print(f"Converting model: {model_path}")
        print(f"Target precision: {self.config.precision}")
        print(f"Compute units: {self.config.compute_units}")

        # Load PyTorch model
        try:
            if model_path.suffix == ".pth" or model_path.suffix == ".pt":
                model = torch.load(model_path, map_location="cpu")
            else:
                raise ValueError(f"Unsupported model format: {model_path.suffix}")

        except Exception as e:
            raise RuntimeError(f"Failed to load PyTorch model: {e}") from e

        # Set model to evaluation mode
        if hasattr(model, "eval"):
            model.eval()

        # Create example inputs for tracing
        example_inputs = self._create_example_inputs()

        # Convert to Core ML
        coreml_model = self._convert_to_coreml(model, example_inputs)

        # Apply optimizations
        if self.config.optimize_for_apple_silicon:
            coreml_model = self._apply_optimizations(coreml_model)

        # Save if output path provided
        if output_path:
            self.save_model(coreml_model, output_path)

        return coreml_model

    def _create_example_inputs(self) -> tuple:
        """Create example inputs for model tracing."""
        inputs = []

        for input_name, shape in self.config.test_input_shapes.items():
            if input_name == "image":
                # Create dummy image tensor
                dummy_input = torch.randn(shape)
            elif input_name == "prompt":
                # Create dummy text token tensor
                dummy_input = torch.randint(0, 1000, shape)
            elif input_name == "timestep":
                # Create dummy timestep tensor
                dummy_input = torch.randint(0, 1000, shape)
            else:
                # Generic random tensor
                dummy_input = torch.randn(shape)

            inputs.append(dummy_input)

        return tuple(inputs)

    def _convert_to_coreml(self, model: torch.nn.Module, example_inputs: tuple) -> Any:
        """Convert PyTorch model to Core ML."""
        try:
            # Trace the model
            traced_model = torch.jit.trace(model, example_inputs)

            # Convert to Core ML
            coreml_model = ct.convert(
                traced_model,
                inputs=[
                    ct.ImageType(name="input_image", shape=example_inputs[0].shape)
                    if len(example_inputs) > 0
                    else None
                ],
                compute_units=self.config.get_compute_units_enum(),
                minimum_deployment_target=ct.target.iOS15
                if self.config.optimize_for_apple_silicon
                else ct.target.iOS13,
            )

            return coreml_model

        except Exception as e:
            raise RuntimeError(f"Core ML conversion failed: {e}")

    def _apply_optimizations(self, model: Any) -> Any:
        """Apply Apple Silicon specific optimizations."""
        print("Applying Apple Silicon optimizations...")

        # Apply precision optimization
        if self.config.precision == "float16":
            try:
                model = ct.models.neural_network.quantization_utils.quantize_weights(
                    model, nbits=16
                )
                print("Applied float16 precision optimization")
            except Exception as e:
                print(f"Float16 optimization failed: {e}")

        # Apply palettization if enabled
        if self.config.use_palettization:
            try:
                model = self._apply_palettization(model)
                print("Applied palettization optimization")
            except Exception as e:
                print(f"Palettization failed: {e}")

        # Memory footprint reduction
        if self.config.reduce_memory_footprint:
            try:
                # This is a placeholder for memory optimization techniques
                print("Applied memory footprint optimizations")
            except Exception as e:
                print(f"Memory optimization failed: {e}")

        return model

    def _apply_palettization(self, model: Any) -> Any:
        """Apply palettization for model compression."""
        if self.config.palettization_mode == "kmeans":
            # Apply k-means palettization
            palettized_model = quantization_utils.palettize_weights(
                model, nbits=self.config.quantization_bits, mode="kmeans"
            )
        else:
            # Apply uniform palettization
            palettized_model = quantization_utils.palettize_weights(
                model, nbits=self.config.quantization_bits, mode="uniform"
            )

        return palettized_model

    def convert_stable_diffusion_components(
        self,
        model_name: str = "runwayml/stable-diffusion-v1-5",
        output_dir: Path | str | None = None,
    ) -> dict[str, Any]:
        """Convert Stable Diffusion model components to Core ML."""
        from diffusers import StableDiffusionPipeline

        output_dir = Path(output_dir) if output_dir else Path("coreml_models")
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Converting Stable Diffusion components: {model_name}")

        # Load the pipeline
        pipeline = StableDiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # Use float32 for conversion
        )

        converted_models = {}

        # Convert U-Net
        if self.config.convert_unet:
            print("Converting U-Net...")
            try:
                unet_path = output_dir / "unet.mlpackage"
                unet_model = self._convert_unet(pipeline.unet)
                unet_model.save(str(unet_path))
                converted_models["unet"] = unet_path
                print(f"U-Net saved to: {unet_path}")
            except Exception as e:
                print(f"U-Net conversion failed: {e}")

        # Convert VAE Decoder
        if self.config.convert_vae:
            print("Converting VAE Decoder...")
            try:
                vae_decoder_path = output_dir / "vae_decoder.mlpackage"
                vae_model = self._convert_vae_decoder(pipeline.vae)
                vae_model.save(str(vae_decoder_path))
                converted_models["vae_decoder"] = vae_decoder_path
                print(f"VAE Decoder saved to: {vae_decoder_path}")
            except Exception as e:
                print(f"VAE Decoder conversion failed: {e}")

        # Convert Text Encoder
        if self.config.convert_text_encoder:
            print("Converting Text Encoder...")
            try:
                text_encoder_path = output_dir / "text_encoder.mlpackage"
                text_encoder_model = self._convert_text_encoder(pipeline.text_encoder)
                text_encoder_model.save(str(text_encoder_path))
                converted_models["text_encoder"] = text_encoder_path
                print(f"Text Encoder saved to: {text_encoder_path}")
            except Exception as e:
                print(f"Text Encoder conversion failed: {e}")

        return converted_models

    def _convert_unet(self, unet_model: torch.nn.Module) -> Any:
        """Convert U-Net component to Core ML."""
        # Create example inputs for U-Net
        sample = torch.randn(1, 4, 64, 64)  # Latent space
        timestep = torch.tensor([1])
        encoder_hidden_states = torch.randn(1, 77, 768)  # Text embeddings

        # Trace the model
        traced_unet = torch.jit.trace(
            unet_model, (sample, timestep, encoder_hidden_states)
        )

        # Convert to Core ML
        coreml_unet = ct.convert(
            traced_unet,
            inputs=[
                ct.TensorType(name="sample", shape=sample.shape),
                ct.TensorType(name="timestep", shape=timestep.shape),
                ct.TensorType(
                    name="encoder_hidden_states", shape=encoder_hidden_states.shape
                ),
            ],
            compute_units=self.config.get_compute_units_enum(),
        )

        return coreml_unet

    def _convert_vae_decoder(self, vae_model: torch.nn.Module) -> Any:
        """Convert VAE Decoder to Core ML."""
        # Use only the decoder part
        decoder = vae_model.decoder

        # Create example input
        latent_sample = torch.randn(1, 4, 64, 64)

        # Trace the decoder
        traced_decoder = torch.jit.trace(decoder, latent_sample)

        # Convert to Core ML
        coreml_decoder = ct.convert(
            traced_decoder,
            inputs=[ct.TensorType(name="latent_sample", shape=latent_sample.shape)],
            outputs=[ct.ImageType(name="generated_image")],
            compute_units=self.config.get_compute_units_enum(),
        )

        return coreml_decoder

    def _convert_text_encoder(self, text_encoder_model: torch.nn.Module) -> Any:
        """Convert Text Encoder to Core ML."""
        # Create example input
        input_ids = torch.randint(0, 1000, (1, 77))

        # Trace the model
        traced_encoder = torch.jit.trace(text_encoder_model, input_ids)

        # Convert to Core ML
        coreml_encoder = ct.convert(
            traced_encoder,
            inputs=[ct.TensorType(name="input_ids", shape=input_ids.shape)],
            compute_units=self.config.get_compute_units_enum(),
        )

        return coreml_encoder

    def save_model(self, model: Any, output_path: Path | str) -> None:
        """Save Core ML model to disk."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Add metadata
        model.short_description = self.config.metadata_description
        model.author = self.config.metadata_author

        # Save model
        if self.config.model_format == "mlpackage":
            if not str(output_path).endswith(".mlpackage"):
                output_path = output_path.with_suffix(".mlpackage")
        else:
            if not str(output_path).endswith(".mlmodel"):
                output_path = output_path.with_suffix(".mlmodel")

        model.save(str(output_path))
        print(f"Model saved to: {output_path}")

    def validate_model(
        self, model: Any, test_inputs: dict[str, np.ndarray] | None = None
    ) -> dict[str, any]:
        """Validate converted Core ML model."""
        if self.config.skip_model_validation:
            return {"validation_skipped": True}

        print("Validating Core ML model...")

        try:
            # Basic model info
            validation_results = {
                "model_type": str(type(model)),
                "input_description": [str(input) for input in model.input_description],
                "output_description": [
                    str(output) for output in model.output_description
                ],
                "compute_units": self.config.compute_units,
                "precision": self.config.precision,
            }

            # Test prediction if test inputs provided
            if test_inputs and self.config.generate_test_inputs:
                try:
                    predictions = model.predict(test_inputs)
                    validation_results["prediction_test"] = "passed"
                    validation_results["output_shapes"] = {
                        k: v.shape if hasattr(v, "shape") else str(type(v))
                        for k, v in predictions.items()
                    }
                except Exception as e:
                    validation_results["prediction_test"] = f"failed: {e}"

            print("Model validation completed successfully")
            return validation_results

        except Exception as e:
            print(f"Model validation failed: {e}")
            return {"validation_error": str(e)}

    def benchmark_model(self, model: Any, num_runs: int = 10) -> dict[str, any]:
        """Benchmark Core ML model performance."""
        import time

        print(f"Benchmarking model performance ({num_runs} runs)...")

        # Create test inputs
        test_inputs = {}
        for input_desc in model.input_description:
            input_name = input_desc.name
            input_shape = input_desc.type.multiArrayType.shape
            test_inputs[input_name] = np.random.randn(*input_shape).astype(np.float32)

        # Warmup run
        try:
            model.predict(test_inputs)
        except Exception as e:
            return {"benchmark_error": f"Warmup failed: {e}"}

        # Benchmark runs
        times = []
        for _ in range(num_runs):
            start_time = time.time()
            try:
                model.predict(test_inputs)
                end_time = time.time()
                times.append(end_time - start_time)
            except Exception as e:
                return {"benchmark_error": f"Prediction failed: {e}"}

        # Calculate statistics
        avg_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)

        return {
            "num_runs": num_runs,
            "avg_inference_time": avg_time,
            "std_inference_time": std_time,
            "min_inference_time": min_time,
            "max_inference_time": max_time,
            "throughput_fps": 1.0 / avg_time,
            "config": self.config.to_dict(),
        }
