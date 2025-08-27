#!/usr/bin/env python3
"""
Command-line interface for Core ML Stable Diffusion Style Transfer.

Provides commands for style transfer, model training, Core ML conversion,
and performance benchmarking on Apple Silicon.
"""

import sys
import os
import typer
from pathlib import Path
import yaml
from PIL import Image

# Add the src directory to Python path for standalone execution
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import shared utilities from the parent package (available via editable install)
# Make this conditional to support both unified and standalone execution
try:
    from utils.logging_utils import get_logger, setup_logging
    SHARED_UTILS_AVAILABLE = True
except ImportError:
    import logging
    SHARED_UTILS_AVAILABLE = False

    def get_logger(name: str) -> logging.Logger:
        """Fallback logger for standalone execution."""
        return logging.getLogger(name)

    def setup_logging(log_level: str = "INFO") -> logging.Logger:
        """Fallback logging setup for standalone execution."""
        logging.basicConfig(level=getattr(logging, log_level.upper()))
        return logging.getLogger()

# Import style transfer modules
from diffusion import DiffusionConfig, StableDiffusionMLX
from style_transfer import StyleTransferConfig, StyleTransferPipeline
from coreml import CoreMLConverter, CoreMLConfig
from inference import InferenceConfig, InferenceEngine

app = typer.Typer(
    name="stable-diffusion-style-transfer",
    help="Core ML Stable Diffusion Style Transfer Framework",
    add_completion=False,
)

# Set up project-specific logging
logger = get_logger(__name__)

# Log which execution mode we're in
if SHARED_UTILS_AVAILABLE:
    logger.debug("Running with shared utilities from parent package (unified execution)")
else:
    logger.debug("Running with fallback utilities (standalone execution)")


@app.command()
def info(
    config: Path = typer.Option("configs/default.yaml", help="Configuration file path"),
):
    """Show information about the Style Transfer framework configuration."""

    # Resolve config path relative to project directory
    project_dir = Path(__file__).parent.parent  # Go up from src/ to project root
    if not config.is_absolute():
        config = project_dir / config

    typer.echo(f"🎨 Core ML Stable Diffusion Style Transfer Information")
    typer.echo("=" * 60)

    if config.exists():
        try:
            with open(config) as f:
                config_data = yaml.safe_load(f)

            # Display diffusion model information
            diffusion = config_data.get("diffusion", {})
            typer.echo(f"🖼️  Diffusion Model Configuration:")
            typer.echo(f"   Model: {diffusion.get('model_name', 'N/A')}")
            typer.echo(f"   Steps: {diffusion.get('num_inference_steps', 'N/A')}")
            typer.echo(f"   Guidance Scale: {diffusion.get('guidance_scale', 'N/A')}")

            # Display style transfer information
            style = config_data.get("style_transfer", {})
            typer.echo(f"\n🎭 Style Transfer Configuration:")
            typer.echo(f"   Style Strength: {style.get('style_strength', 'N/A')}")
            typer.echo(f"   Content Strength: {style.get('content_strength', 'N/A')}")
            typer.echo(f"   Output Resolution: {style.get('output_resolution', 'N/A')}")

            # Display Core ML information
            coreml = config_data.get("coreml", {})
            typer.echo(f"\n🍎 Core ML Configuration:")
            typer.echo(f"   Optimization: {coreml.get('optimize_for_apple_silicon', 'N/A')}")
            typer.echo(f"   Compute Units: {coreml.get('compute_units', 'N/A')}")
            typer.echo(f"   Precision: {coreml.get('precision', 'N/A')}")

            # Display hardware information
            hardware = config_data.get("hardware", {})
            typer.echo(f"\n💻 Hardware Configuration:")
            typer.echo(f"   Prefer MLX: {hardware.get('prefer_mlx', 'N/A')}")
            typer.echo(f"   Use MPS: {hardware.get('use_mps', 'N/A')}")
            typer.echo(f"   Memory Optimization: {hardware.get('memory_optimization', 'N/A')}")

        except Exception as e:
            typer.echo(f"❌ Failed to read config: {e}", err=True)
    else:
        typer.echo(f"❌ Configuration file not found: {config}")

    typer.echo("=" * 60)


@app.command()
def validate(
    config: Path = typer.Option("configs/default.yaml", help="Configuration file path"),
):
    """Validate configuration file."""

    # Resolve config path relative to project directory
    project_dir = Path(__file__).parent.parent
    if not config.is_absolute():
        config = project_dir / config

    typer.echo(f"✅ Validating configuration: {config}")

    if not config.exists():
        typer.echo(f"❌ Configuration file not found: {config}", err=True)
        raise typer.Exit(1)

    try:
        # Load and validate configuration
        with open(config) as f:
            config_dict = yaml.safe_load(f)

        # Validate each component
        success_count = 0

        # Validate diffusion config
        try:
            diffusion_config = DiffusionConfig.from_dict(config_dict.get("diffusion", {}))
            typer.echo("✅ Diffusion configuration: Valid")
            success_count += 1
        except Exception as e:
            typer.echo(f"❌ Diffusion configuration failed: {e}", err=True)

        # Validate style transfer config
        try:
            style_config = StyleTransferConfig.from_dict(config_dict.get("style_transfer", {}))
            typer.echo("✅ Style transfer configuration: Valid")
            success_count += 1
        except Exception as e:
            typer.echo(f"❌ Style transfer configuration failed: {e}", err=True)

        # Validate Core ML config
        try:
            coreml_config = CoreMLConfig.from_dict(config_dict.get("coreml", {}))
            typer.echo("✅ Core ML configuration: Valid")
            success_count += 1
        except Exception as e:
            typer.echo(f"❌ Core ML configuration failed: {e}", err=True)

        if success_count == 3:
            typer.echo("🎉 All configurations are valid!")
        else:
            typer.echo(f"⚠️  {success_count}/3 configurations are valid")
            raise typer.Exit(1)

    except Exception as e:
        typer.echo(f"❌ Configuration validation failed: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def transfer(
    content_image: Path = typer.Option(..., help="Path to content image"),
    style_image: Path = typer.Option(..., help="Path to style image"),
    output: Path = typer.Option("output.png", help="Output image path"),
    config: Path = typer.Option("configs/default.yaml", help="Configuration file path"),
    style_strength: float | None = typer.Option(None, help="Style strength (0.0-1.0)"),
    content_strength: float | None = typer.Option(None, help="Content strength (0.0-1.0)"),
    steps: int | None = typer.Option(None, help="Number of inference steps"),
    guidance_scale: float | None = typer.Option(None, help="Guidance scale"),
):
    """Perform style transfer on an image."""

    # Resolve config path relative to project directory
    project_dir = Path(__file__).parent.parent
    if not config.is_absolute():
        config = project_dir / config

    typer.echo(f"🎨 Starting Style Transfer")
    typer.echo(f"🖼️  Content: {content_image}")
    typer.echo(f"🎭 Style: {style_image}")
    typer.echo(f"💾 Output: {output}")

    # Validate input files
    if not content_image.exists():
        typer.echo(f"❌ Content image not found: {content_image}", err=True)
        raise typer.Exit(1)

    if not style_image.exists():
        typer.echo(f"❌ Style image not found: {style_image}", err=True)
        raise typer.Exit(1)

    try:
        # Load configuration
        with open(config) as f:
            config_data = yaml.safe_load(f)

        # Create style transfer config with overrides
        style_config = StyleTransferConfig.from_dict(config_data.get("style_transfer", {}))

        if style_strength is not None:
            style_config.style_strength = style_strength
        if content_strength is not None:
            style_config.content_strength = content_strength
        if steps is not None:
            style_config.num_inference_steps = steps
        if guidance_scale is not None:
            style_config.guidance_scale = guidance_scale

        typer.echo(f"⚙️  Settings: Style={style_config.style_strength:.2f}, "
                  f"Content={style_config.content_strength:.2f}, "
                  f"Steps={style_config.num_inference_steps}")

        # Create style transfer pipeline
        pipeline = StyleTransferPipeline(style_config)

        # Load images
        content_img = Image.open(content_image).convert("RGB")
        style_img = Image.open(style_image).convert("RGB")

        typer.echo("🚀 Running style transfer...")

        # Perform style transfer
        result_image = pipeline.transfer_style(
            content_image=content_img,
            style_image=style_img
        )

        # Save result
        result_image.save(output)
        typer.echo(f"✅ Style transfer completed: {output}")

    except Exception as e:
        typer.echo(f"❌ Style transfer failed: {e}", err=True)
        raise typer.Exit(1)


# Training functionality is under development
# @app.command()
# def train(...):
#     """Train a custom style transfer model."""
#     typer.echo("🚧 Training functionality is under development")
#     raise typer.Exit(1)


@app.command()
def convert(
    model_path: Path = typer.Option(..., help="Path to trained model"),
    output_path: Path = typer.Option(..., help="Output Core ML model path"),
    config: Path = typer.Option("configs/default.yaml", help="Configuration file path"),
    optimize: bool = typer.Option(True, help="Optimize for Apple Silicon"),
    compute_units: str | None = typer.Option(None, help="Core ML compute units (all, cpu_only, cpu_and_gpu)"),
):
    """Convert trained model to Core ML format."""

    project_dir = Path(__file__).parent.parent
    if not config.is_absolute():
        config = project_dir / config

    typer.echo(f"🔄 Converting Model to Core ML")
    typer.echo(f"📥 Input: {model_path}")
    typer.echo(f"📤 Output: {output_path}")

    if not model_path.exists():
        typer.echo(f"❌ Model not found: {model_path}", err=True)
        raise typer.Exit(1)

    try:
        # Load configuration
        with open(config) as f:
            config_data = yaml.safe_load(f)

        # Create Core ML config with overrides
        coreml_config = CoreMLConfig.from_dict(config_data.get("coreml", {}))

        if not optimize:
            coreml_config.optimize_for_apple_silicon = False
        if compute_units is not None:
            coreml_config.compute_units = compute_units

        typer.echo(f"⚙️  Optimization: {coreml_config.optimize_for_apple_silicon}")
        typer.echo(f"💻 Compute Units: {coreml_config.compute_units}")

        # Create converter
        converter = CoreMLConverter(coreml_config)

        # Convert model
        typer.echo("🚀 Converting model...")
        coreml_model = converter.convert_model(model_path)

        # Save converted model
        converter.save_model(coreml_model, output_path)

        typer.echo(f"✅ Conversion completed: {output_path}")

    except Exception as e:
        typer.echo(f"❌ Conversion failed: {e}", err=True)
        raise typer.Exit(1)


# Serving functionality is under development
# @app.command()
# def serve(...):
#     """Start inference server for style transfer."""
#     typer.echo("🚧 Serving functionality is under development")
#     raise typer.Exit(1)


@app.command()
def benchmark(
    model_path: Path = typer.Option(..., help="Path to model to benchmark"),
    test_images: Path = typer.Option(..., help="Directory with test images"),
    config: Path = typer.Option("configs/default.yaml", help="Configuration file path"),
    output: Path | None = typer.Option(None, help="Output directory for results"),
    iterations: int = typer.Option(10, help="Number of benchmark iterations"),
):
    """Benchmark style transfer performance."""

    project_dir = Path(__file__).parent.parent
    if not config.is_absolute():
        config = project_dir / config

    typer.echo(f"📊 Benchmarking Style Transfer Performance")
    typer.echo(f"🤖 Model: {model_path}")
    typer.echo(f"🖼️  Test Images: {test_images}")

    if not model_path.exists():
        typer.echo(f"❌ Model not found: {model_path}", err=True)
        raise typer.Exit(1)

    if not test_images.exists():
        typer.echo(f"❌ Test images directory not found: {test_images}", err=True)
        raise typer.Exit(1)

    try:
        # Load configuration
        with open(config) as f:
            config_data = yaml.safe_load(f)

        # Create inference engine
        inference_config = InferenceConfig.from_dict(config_data.get("inference", {}))
        inference_config.model_path = model_path

        engine = InferenceEngine(inference_config)

        # Run benchmark
        typer.echo(f"🚀 Running {iterations} benchmark iterations...")
        results = engine.benchmark(
            test_images_dir=test_images,
            iterations=iterations,
            output_dir=output
        )

        # Display results
        typer.echo("📊 Benchmark Results:")
        typer.echo(f"   Average Time: {results.get('avg_time', 'N/A'):.3f}s")
        typer.echo(f"   Throughput: {results.get('throughput', 'N/A'):.2f} images/s")
        typer.echo(f"   Memory Usage: {results.get('memory_usage', 'N/A'):.1f} MB")

        if output:
            typer.echo(f"💾 Detailed results saved to: {output}")

        typer.echo("✅ Benchmarking completed")

    except Exception as e:
        typer.echo(f"❌ Benchmarking failed: {e}", err=True)
        raise typer.Exit(1)


if __name__ == "__main__":
    app()