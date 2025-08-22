#!/usr/bin/env python3
"""
Command-line interface for Model Compression Framework.

Provides easy-to-use commands for quantization, pruning, distillation,
and comprehensive benchmarking of compressed models.
"""

import sys
import os
import typer
from pathlib import Path
from typing import Optional, List
import yaml

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

# Import compression modules - MLX is mandatory
from compression import ModelCompressor, CompressionConfig
from quantization import MLXQuantizer, QuantizationConfig
from pruning import MLXPruner, PruningConfig

# MLX is mandatory for this project
import mlx.core as mx
from mlx_lm.utils import load as load_model_and_tokenizer

app = typer.Typer(
    name="model-compression-framework",
    help="MLX-Native Model Compression Framework",
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
    """Show information about the Model Compression framework configuration."""

    # Resolve config path relative to project directory
    project_dir = Path(__file__).parent.parent  # Go up from src/ to project root
    if not config.is_absolute():
        config = project_dir / config

    typer.echo(f"ℹ️  Model Compression Framework Information")
    typer.echo("=" * 50)

    if config.exists():
        try:
            with open(config) as f:
                config_data = yaml.safe_load(f)

            # Display quantization information
            quantization = config_data.get("quantization", {})
            typer.echo(f"📊 Quantization Configuration:")
            typer.echo(f"   Target Bits: {quantization.get('target_bits', 'N/A')}")
            typer.echo(f"   Method: {quantization.get('method', 'N/A')}")
            typer.echo(f"   MLX Enabled: {quantization.get('use_mlx_quantization', 'N/A')}")

            # Display pruning information
            pruning = config_data.get("pruning", {})
            typer.echo(f"\n✂️  Pruning Configuration:")
            typer.echo(f"   Target Sparsity: {pruning.get('target_sparsity', 'N/A')}")
            typer.echo(f"   Method: {pruning.get('method', 'N/A')}")
            typer.echo(f"   Structured: {pruning.get('structured', 'N/A')}")

            # Display distillation information
            distillation = config_data.get("distillation", {})
            typer.echo(f"\n🎓 Distillation Configuration:")
            typer.echo(f"   Temperature: {distillation.get('temperature', 'N/A')}")
            typer.echo(f"   Alpha: {distillation.get('alpha', 'N/A')}")
            typer.echo(f"   MLX Enabled: {distillation.get('use_mlx_distillation', 'N/A')}")

            # Display model information
            model = config_data.get("model", {})
            typer.echo(f"\n🤖 Model Configuration:")
            typer.echo(f"   Model: {model.get('model_name', 'N/A')}")
            typer.echo(f"   Output Dir: {model.get('output_dir', 'N/A')}")
            typer.echo(f"   MLX Enabled: {model.get('use_mlx', 'N/A')}")

        except Exception as e:
            typer.echo(f"❌ Failed to read config: {e}", err=True)
    else:
        typer.echo(f"❌ Configuration file not found: {config}")

    typer.echo("=" * 50)


@app.command()
def validate(
    config: Path = typer.Option("configs/default.yaml", help="Configuration file path"),
):
    """Validate configuration file."""

    # Resolve config path relative to project directory
    project_dir = Path(__file__).parent.parent  # Go up from src/ to project root
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

        # Validate quantization config
        try:
            from quantization.config import QuantizationConfig
            quant_config = QuantizationConfig.from_dict(config_dict.get("quantization", {}))
            typer.echo("✅ Quantization configuration: Valid")
            success_count += 1
        except Exception as e:
            typer.echo(f"❌ Quantization configuration failed: {e}", err=True)

        # Validate pruning config
        try:
            from pruning.config import PruningConfig
            prune_config = PruningConfig.from_dict(config_dict.get("pruning", {}))
            typer.echo("✅ Pruning configuration: Valid")
            success_count += 1
        except Exception as e:
            typer.echo(f"❌ Pruning configuration failed: {e}", err=True)

        # Validate compression config
        try:
            from compression.config import CompressionConfig
            comp_config = CompressionConfig.from_dict(config_dict)
            typer.echo("✅ Compression configuration: Valid")
            success_count += 1
        except Exception as e:
            typer.echo(f"❌ Compression configuration failed: {e}", err=True)

        if success_count == 3:
            typer.echo("🎉 All configurations are valid!")
        else:
            typer.echo(f"⚠️  {success_count}/3 configurations are valid")
            raise typer.Exit(1)

    except Exception as e:
        typer.echo(f"❌ Configuration validation failed: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def quantize(
    model_path: str = typer.Option(..., help="Model name or path to quantize"),
    config: Path = typer.Option("configs/default.yaml", help="Configuration file path"),
    output: Optional[Path] = typer.Option(None, help="Override output directory"),
    bits: Optional[int] = typer.Option(None, help="Override target bits"),
    method: Optional[str] = typer.Option(None, help="Override quantization method"),
    calibration_data: Optional[Path] = typer.Option(None, help="Path to calibration data"),
):
    """Quantize a model using the specified configuration."""

    # Resolve config path relative to project directory
    project_dir = Path(__file__).parent.parent
    if not config.is_absolute():
        config = project_dir / config

    typer.echo(f"🚀 Starting Model Quantization")
    typer.echo(f"🤖 Model: {model_path}")
    typer.echo(f"📁 Config: {config}")

    # Load configuration
    if not config.exists():
        typer.echo(f"❌ Configuration file not found: {config}", err=True)
        raise typer.Exit(1)

    try:
        with open(config) as f:
            config_data = yaml.safe_load(f)

        # Create quantization config
        from quantization.config import QuantizationConfig
        quant_config = QuantizationConfig.from_dict(config_data.get("quantization", {}))

        # Apply command-line overrides
        if bits:
            quant_config.target_bits = bits
        if method:
            from quantization.config import QuantizationMethod
            quant_config.method = QuantizationMethod(method)
        if output:
            quant_config.quantized_model_path = output
        if calibration_data:
            quant_config.calibration_dataset_path = calibration_data

        typer.echo(f"🎯 Quantization: {quant_config.target_bits}-bit, method={quant_config.method.value}")

        # Create quantizer and run quantization
        quantizer = MLXQuantizer(quant_config)
        quantized_model = quantizer.quantize(model_path=model_path)

        # Save quantized model
        if output:
            quantizer.save_quantized_model(output)
            typer.echo(f"💾 Quantized model saved to: {output}")

        # Display results
        info = quantizer.get_quantization_info()
        stats = info.get("stats", {})

        if "actual_compression_ratio" in stats:
            typer.echo(f"📊 Compression ratio: {stats['actual_compression_ratio']:.2f}x")
        if "size_reduction_mb" in stats:
            typer.echo(f"💾 Size reduction: {stats['size_reduction_mb']:.1f} MB")

        typer.echo("✅ Quantization completed successfully")

    except Exception as e:
        typer.echo(f"❌ Quantization failed: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def prune(
    model_path: str = typer.Option(..., help="Model name or path to prune"),
    config: Path = typer.Option("configs/default.yaml", help="Configuration file path"),
    output: Optional[Path] = typer.Option(None, help="Override output directory"),
    sparsity: Optional[float] = typer.Option(None, help="Override target sparsity"),
    method: Optional[str] = typer.Option(None, help="Override pruning method"),
    structured: Optional[bool] = typer.Option(None, help="Override structured pruning"),
):
    """Prune a model using the specified configuration."""

    # Resolve config path relative to project directory
    project_dir = Path(__file__).parent.parent
    if not config.is_absolute():
        config = project_dir / config

    typer.echo(f"✂️  Starting Model Pruning")
    typer.echo(f"🤖 Model: {model_path}")
    typer.echo(f"📁 Config: {config}")

    # Load configuration
    if not config.exists():
        typer.echo(f"❌ Configuration file not found: {config}", err=True)
        raise typer.Exit(1)

    try:
        with open(config) as f:
            config_data = yaml.safe_load(f)

        # Create pruning config
        from pruning.config import PruningConfig
        prune_config = PruningConfig.from_dict(config_data.get("pruning", {}))

        # Apply command-line overrides
        if sparsity:
            prune_config.target_sparsity = sparsity
        if method:
            from pruning.config import PruningMethod
            prune_config.method = PruningMethod(method)
        if structured is not None:
            prune_config.structured = structured
        if output:
            prune_config.pruned_model_path = output

        typer.echo(f"🎯 Pruning: {prune_config.target_sparsity:.1%} sparsity, method={prune_config.method.value}")
        typer.echo(f"🏗️  Type: {'Structured' if prune_config.structured else 'Unstructured'}")

        # Load model
        model, tokenizer = load_model_and_tokenizer(model_path)
        typer.echo("✅ Model loaded successfully")

        # Create pruner and run pruning
        pruner = MLXPruner(prune_config)
        pruned_model = pruner.prune(model)

        # Save pruned model
        if output:
            pruner.save_pruned_model(output)
            typer.echo(f"💾 Pruned model saved to: {output}")

        # Display results
        info = pruner.get_pruning_info()
        stats = info.get("stats", {})

        if "parameters_removed_percent" in stats:
            typer.echo(f"📊 Parameters removed: {stats['parameters_removed_percent']:.1f}%")
        if "actual_sparsity" in stats:
            typer.echo(f"🎯 Achieved sparsity: {stats['actual_sparsity']:.1%}")

        typer.echo("✅ Pruning completed successfully")

    except Exception as e:
        typer.echo(f"❌ Pruning failed: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def compress(
    model_path: str = typer.Option(..., help="Model name or path to compress"),
    config: Path = typer.Option("configs/default.yaml", help="Configuration file path"),
    output: Optional[Path] = typer.Option(None, help="Override output directory"),
    methods: Optional[List[str]] = typer.Option(None, help="Compression methods to apply"),
):
    """Apply comprehensive model compression with multiple techniques."""

    # Resolve config path relative to project directory
    project_dir = Path(__file__).parent.parent
    if not config.is_absolute():
        config = project_dir / config

    typer.echo(f"🚀 Starting Comprehensive Model Compression")
    typer.echo(f"🤖 Model: {model_path}")
    typer.echo(f"📁 Config: {config}")

    # Load configuration
    if not config.exists():
        typer.echo(f"❌ Configuration file not found: {config}", err=True)
        raise typer.Exit(1)

    try:
        with open(config) as f:
            config_data = yaml.safe_load(f)

        # Create compression config
        from compression.config import CompressionConfig
        comp_config = CompressionConfig.from_dict(config_data)

        # Apply command-line overrides
        if output:
            comp_config.output_dir = output
        if methods:
            comp_config.enabled_methods = methods

        typer.echo(f"🎯 Methods: {', '.join(comp_config.enabled_methods)}")

        # Create compressor and run compression
        compressor = ModelCompressor(comp_config)
        compressed_model = compressor.compress(model_path)

        # Display results
        results = compressor.get_compression_results()

        typer.echo("📊 Compression Results:")
        for method, result in results.items():
            typer.echo(f"   {method}: {result.get('compression_ratio', 'N/A')}x compression")

        typer.echo("✅ Compression completed successfully")

    except Exception as e:
        typer.echo(f"❌ Compression failed: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def benchmark(
    model_path: str = typer.Option(..., help="Model name or path to benchmark"),
    config: Path = typer.Option("configs/default.yaml", help="Configuration file path"),
    methods: Optional[List[str]] = typer.Option(None, help="Compression methods to benchmark"),
    output: Optional[Path] = typer.Option(None, help="Output directory for results"),
):
    """Benchmark different compression methods on a model."""

    # Resolve config path relative to project directory
    project_dir = Path(__file__).parent.parent
    if not config.is_absolute():
        config = project_dir / config

    typer.echo(f"📊 Starting Compression Benchmarking")
    typer.echo(f"🤖 Model: {model_path}")

    # Load configuration
    if not config.exists():
        typer.echo(f"❌ Configuration file not found: {config}", err=True)
        raise typer.Exit(1)

    try:
        from benchmarking import CompressionBenchmark

        # Create and run benchmark
        benchmark = CompressionBenchmark(config)
        results = benchmark.run_comprehensive_benchmark(
            model_path=model_path,
            methods=methods or ["quantization", "pruning"],
            output_dir=output
        )

        # Display results
        typer.echo("📊 Benchmark Results:")
        for method, metrics in results.items():
            typer.echo(f"\n{method.upper()}:")
            for metric, value in metrics.items():
                typer.echo(f"   {metric}: {value}")

        if output:
            typer.echo(f"\n💾 Detailed results saved to: {output}")

        typer.echo("✅ Benchmarking completed successfully")

    except Exception as e:
        typer.echo(f"❌ Benchmarking failed: {e}", err=True)
        raise typer.Exit(1)


if __name__ == "__main__":
    app()