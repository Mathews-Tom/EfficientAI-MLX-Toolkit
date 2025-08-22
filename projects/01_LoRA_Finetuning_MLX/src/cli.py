#!/usr/bin/env python3
"""
Command-line interface for LoRA Fine-tuning Framework.

Provides easy-to-use commands for training, inference, optimization,
and serving LoRA fine-tuned models.
"""

import typer
from pathlib import Path
from typing import Optional, List
import yaml

from lora import LoRAConfig, TrainingConfig, InferenceConfig, load_config

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
from training import LoRATrainer
from inference import LoRAInferenceEngine, create_fastapi_app
from optimization import run_optimization

import mlx_lm
from mlx_lm.utils import load as load_model_and_tokenizer

app = typer.Typer(
    name="lora-framework",
    help="MLX-Native LoRA Fine-Tuning Framework",
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
def train(
    config: Path = typer.Option("configs/default.yaml", help="Configuration file path"),
    model: Optional[str] = typer.Option(None, help="Override model name"),
    data: Optional[Path] = typer.Option(None, help="Override dataset path"),
    output: Optional[Path] = typer.Option(None, help="Override output directory"),
    epochs: Optional[int] = typer.Option(None, help="Override number of epochs"),
    batch_size: Optional[int] = typer.Option(None, help="Override batch size"),
    learning_rate: Optional[float] = typer.Option(None, help="Override learning rate"),
    rank: Optional[int] = typer.Option(None, help="Override LoRA rank"),
    alpha: Optional[float] = typer.Option(None, help="Override LoRA alpha"),
):
    """Train a LoRA model with the specified configuration."""

    # Resolve config path relative to project directory
    project_dir = Path(__file__).parent.parent  # Go up from src/ to project root
    if not config.is_absolute():
        config = project_dir / config

    typer.echo(f"🚀 Starting LoRA Fine-tuning")
    typer.echo(f"📁 Config: {config}")

    # Load configuration
    if not config.exists():
        typer.echo(f"❌ Configuration file not found: {config}", err=True)
        raise typer.Exit(1)

    try:
        configs = load_config(config)
        lora_config = configs["lora"]
        training_config = configs["training"]

        # Apply command-line overrides
        if model:
            training_config.model_name = model
        if data:
            training_config.dataset_path = Path(data)
        if output:
            training_config.output_dir = Path(output)
        if epochs:
            training_config.num_epochs = epochs
        if batch_size:
            training_config.batch_size = batch_size
        if learning_rate:
            training_config.learning_rate = learning_rate
        if rank:
            lora_config.rank = rank
        if alpha:
            lora_config.alpha = alpha

        typer.echo(f"📊 LoRA Config: rank={lora_config.rank}, alpha={lora_config.alpha}")
        typer.echo(f"🎯 Training: epochs={training_config.num_epochs}, lr={training_config.learning_rate}")


        # Load model and tokenizer
        model_to_load = training_config.model_name
        typer.echo(f"📥 Loading model: {model_to_load}")
        try:
            model_obj, tokenizer = load_model_and_tokenizer(model_to_load)
            typer.echo("✅ Model loaded successfully")
        except Exception as e:
            typer.echo(f"❌ Failed to load model: {e}", err=True)
            raise typer.Exit(1)

        # Create trainer
        trainer = LoRATrainer(
            model=model_obj,
            lora_config=lora_config,
            training_config=training_config
        )

        # Set default data path if not provided
        if not data:
            data = Path(__file__).parent.parent / "data" / "samples" / "sample_conversations.jsonl"
            if not data.exists():
                typer.echo("❌ No training data found. Please specify --data parameter", err=True)
                raise typer.Exit(1)
            typer.echo(f"📄 Using default training data: {data}")

        # Run training
        typer.echo("🚀 Starting training...")
        try:
            results = trainer.train(tokenizer=tokenizer, train_data_path=str(data))
            typer.echo("✅ Training completed successfully")

            # Print results summary
            if results:
                train_loss = results.get('train_loss', 'N/A')
                train_time = results.get('total_train_time', 'N/A')

                # Format train_loss safely
                if isinstance(train_loss, (int, float)):
                    typer.echo(f"📊 Final training loss: {train_loss:.4f}")
                else:
                    typer.echo(f"📊 Final training loss: {train_loss}")

                # Format train_time safely
                if isinstance(train_time, (int, float)):
                    typer.echo(f"⏱️  Total training time: {train_time:.2f}s")
                else:
                    typer.echo(f"⏱️  Total training time: {train_time}")

        except Exception as e:
            typer.echo(f"❌ Training failed: {e}", err=True)
            raise typer.Exit(1)

    except Exception as e:
        typer.echo(f"❌ Training failed: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def optimize(
    model: str = typer.Option(..., help="Model name or path"),
    data: Path = typer.Option(..., help="Dataset path"),
    output: Path = typer.Option("optimization_results/", help="Output directory"),
    trials: int = typer.Option(20, help="Number of optimization trials"),
    max_epochs: int = typer.Option(3, help="Maximum epochs per trial"),
):
    """Run hyperparameter optimization for LoRA fine-tuning."""

    typer.echo(f"🔍 Starting Hyperparameter Optimization")
    typer.echo(f"🤖 Model: {model}")
    typer.echo(f"📊 Dataset: {data}")
    typer.echo(f"🎯 Trials: {trials}")

    try:
        # Run optimization
        best_result = run_optimization(
            model_name=model,
            dataset_path=data,
            output_dir=output,
            n_trials=trials,
            max_epochs_per_trial=max_epochs,
        )

        typer.echo(f"✅ Optimization completed!")
        typer.echo(f"🏆 Best result: {best_result.objective_value:.4f}")
        typer.echo(f"📁 Results saved to: {output}")

    except Exception as e:
        typer.echo(f"❌ Optimization failed: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def serve(
    model_path: Path = typer.Option(..., help="Path to trained model"),
    adapter_path: Optional[Path] = typer.Option(None, help="Path to LoRA adapters"),
    host: str = typer.Option("0.0.0.0", help="Server host"),
    port: int = typer.Option(8000, help="Server port"),
    workers: int = typer.Option(1, help="Number of workers"),
):
    """Start inference server for trained LoRA model."""

    typer.echo(f"🌐 Starting LoRA Inference Server")
    typer.echo(f"📁 Model: {model_path}")
    typer.echo(f"🔧 Adapters: {adapter_path or 'None'}")
    typer.echo(f"🌍 Server: http://{host}:{port}")

    try:
        # Create FastAPI app
        app_instance = create_fastapi_app(
            model_path=model_path,
            adapter_path=adapter_path,
        )

        typer.echo("⚠️  Server creation not implemented in this demo")
        typer.echo("✅ Server configuration validated successfully")
        typer.echo(f"📖 API docs would be available at: http://{host}:{port}/docs")

    except Exception as e:
        typer.echo(f"❌ Server startup failed: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def generate(
    prompt: str = typer.Option(..., help="Input prompt"),
    model_path: Optional[Path] = typer.Option(None, help="Path to trained model (defaults to latest checkpoint)"),
    adapter_path: Optional[Path] = typer.Option(None, help="Path to LoRA adapters"),
    max_length: int = typer.Option(100, help="Maximum tokens to generate"),
    temperature: float = typer.Option(0.7, help="Sampling temperature"),
    top_p: float = typer.Option(0.9, help="Nucleus sampling parameter"),
    top_k: int = typer.Option(50, help="Top-k sampling parameter"),
):
    """Generate text using a trained LoRA model."""

    # Resolve model path - use default if not provided
    if model_path is None:
        project_dir = Path(__file__).parent.parent  # Go up from src/ to project root
        toolkit_root = project_dir.parent.parent  # Go up to toolkit root

        # Check both project-local outputs and toolkit-root outputs
        default_paths = [
            # Project-local outputs (preferred for standalone execution)
            project_dir / "outputs" / "checkpoints" / "best_model",
            project_dir / "outputs" / "checkpoints" / "checkpoint_epoch_2",
            project_dir / "outputs" / "checkpoints" / "checkpoint_epoch_1",
            project_dir / "outputs" / "checkpoints" / "checkpoint_epoch_0",
            # Toolkit-root outputs (used by unified CLI)
            toolkit_root / "outputs" / "checkpoints" / "best_model",
            toolkit_root / "outputs" / "checkpoints" / "checkpoint_epoch_2",
            toolkit_root / "outputs" / "checkpoints" / "checkpoint_epoch_1",
            toolkit_root / "outputs" / "checkpoints" / "checkpoint_epoch_0",
        ]

        # Find the first existing checkpoint
        for path in default_paths:
            if path.exists():
                model_path = path
                break

        if model_path is None:
            typer.echo("❌ No trained model found. Available options:", err=True)
            typer.echo("  1. Train a model first: uv run efficientai-toolkit lora-finetuning-mlx:train", err=True)
            typer.echo("  2. Specify --model-path with base model name and --adapter-path with checkpoint", err=True)
            typer.echo("     Example: --model-path mlx-community/Llama-3.2-1B-Instruct-4bit --adapter-path /path/to/checkpoint", err=True)
            typer.echo("  3. Check if outputs/checkpoints/ directory exists", err=True)
            raise typer.Exit(1)
        else:
            # Check if model_path is a checkpoint directory (contains adapter files)
            checkpoint_indicators = [
                model_path / "adapter_weights.json",
                model_path / "adapter_config.yaml",
                model_path / "adapter_metadata.json"
            ]

            if any(indicator.exists() for indicator in checkpoint_indicators):
                # This is a LoRA checkpoint, not a base model
                typer.echo(f"📁 Found LoRA checkpoint: {model_path}")
                typer.echo("⚠️  For LoRA inference, you need to specify:")
                typer.echo("   --model-path <base-model-name>  (e.g., mlx-community/Llama-3.2-1B-Instruct-4bit)")
                typer.echo(f"   --adapter-path {model_path}")
                typer.echo("\n💡 Example:")
                typer.echo(f"   uv run efficientai-toolkit lora-finetuning-mlx:generate \\")
                typer.echo(f"     --prompt \"{prompt}\" \\")
                typer.echo(f"     --model-path mlx-community/Llama-3.2-1B-Instruct-4bit \\")
                typer.echo(f"     --adapter-path {model_path}")
                raise typer.Exit(1)
            else:
                typer.echo(f"📁 Using base model: {model_path}")

    if adapter_path:
        typer.echo(f"🔧 Using LoRA adapters: {adapter_path}")

    typer.echo(f"✨ Generating text with LoRA model")
    typer.echo(f"💭 Prompt: {prompt}")

    try:

        typer.echo(f"🎯 Configuration: max_length={max_length}, temp={temperature}")

        # Create inference configuration
        from lora import InferenceConfig
        inference_config = InferenceConfig(
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k
        )

        # Load model and create inference engine
        typer.echo("📥 Loading model and creating inference engine...")
        engine = LoRAInferenceEngine.from_pretrained(
            model_path=model_path,
            adapter_path=adapter_path,
            config=inference_config
        )

        # Generate text
        typer.echo("🚀 Generating text...")
        result = engine.generate(
            prompt=prompt,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k
        )

        # Display results
        typer.echo("✅ Generation completed successfully")
        typer.echo("\n📝 Generated Text:")
        typer.echo("─" * 50)
        typer.echo(result.generated_text)
        typer.echo("─" * 50)

        # Display metrics
        typer.echo(f"\n📊 Generation Metrics:")
        typer.echo(f"   Tokens generated: {result.tokens_generated}")
        typer.echo(f"   Inference time: {result.inference_time:.2f}s")
        typer.echo(f"   Speed: {result.tokens_per_second:.1f} tokens/s")

    except Exception as e:
        typer.echo(f"❌ Generation failed: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def info(
    config: Path = typer.Option("configs/default.yaml", help="Configuration file path"),
):
    """Show information about the LoRA framework configuration."""

    # Resolve config path relative to project directory
    project_dir = Path(__file__).parent.parent  # Go up from src/ to project root
    if not config.is_absolute():
        config = project_dir / config

    typer.echo(f"ℹ️  LoRA Framework Information")
    typer.echo("=" * 50)

    if config.exists():
        try:
            with open(config) as f:
                config_data = yaml.safe_load(f)

            # Display key information
            lora = config_data.get("lora", {})
            training = config_data.get("training", {})

            typer.echo(f"📊 LoRA Configuration:")
            typer.echo(f"   Rank: {lora.get('rank', 'N/A')}")
            typer.echo(f"   Alpha: {lora.get('alpha', 'N/A')}")
            typer.echo(f"   Dropout: {lora.get('dropout', 'N/A')}")
            typer.echo(f"   Target Modules: {len(lora.get('target_modules', []))}")

            typer.echo(f"\n🎯 Training Configuration:")
            typer.echo(f"   Model: {training.get('model_name', 'N/A')}")
            typer.echo(f"   Epochs: {training.get('num_epochs', 'N/A')}")
            typer.echo(f"   Batch Size: {training.get('batch_size', 'N/A')}")
            typer.echo(f"   Learning Rate: {training.get('learning_rate', 'N/A')}")
            typer.echo(f"   Optimizer: {training.get('optimizer', 'N/A')}")

            typer.echo(f"\n🍎 Apple Silicon Settings:")
            typer.echo(f"   MLX Enabled: {training.get('use_mlx', 'N/A')}")
            typer.echo(f"   MLX Precision: {lora.get('mlx_precision', 'N/A')}")

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
        # Load and validate configuration step by step
        import yaml
        with open(config) as f:
            config_dict = yaml.safe_load(f)

        # Try each config separately to identify which one fails
        try:
            from lora import LoRAConfig
            lora_config = LoRAConfig(**config_dict.get("lora", {}))
            typer.echo("✅ LoRA configuration: Valid")
        except Exception as e:
            typer.echo(f"❌ LoRA configuration failed: {e}", err=True)
            raise

        try:
            from lora import TrainingConfig
            training_config = TrainingConfig(**config_dict.get("training", {}))
            typer.echo("✅ Training configuration: Valid")
        except Exception as e:
            typer.echo(f"❌ Training configuration failed: {e}", err=True)
            raise

        try:
            from lora import InferenceConfig
            inference_config = InferenceConfig(**config_dict.get("inference", {}))
            typer.echo("✅ Inference configuration: Valid")
        except Exception as e:
            typer.echo(f"❌ Inference configuration failed: {e}", err=True)
            raise

        try:
            from lora.config import OptimizationConfig
            optimization_config = OptimizationConfig(**config_dict.get("optimization", {}))
            typer.echo("✅ Optimization configuration: Valid")
        except Exception as e:
            typer.echo(f"❌ Optimization configuration failed: {e}", err=True)
            raise

        typer.echo("🎉 All configurations are valid!")

    except Exception as e:
        typer.echo(f"❌ Configuration validation failed: {e}", err=True)
        raise typer.Exit(1)


if __name__ == "__main__":
    app()