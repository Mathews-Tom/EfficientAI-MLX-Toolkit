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

from .lora import LoRAConfig, TrainingConfig, InferenceConfig, load_config
from .training import LoRATrainer
from .inference import LoRAInferenceEngine, create_fastapi_app
from .optimization import run_optimization

app = typer.Typer(
    name="lora-framework",
    help="MLX-Native LoRA Fine-Tuning Framework",
    add_completion=False,
)


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
        
        # This would create and run trainer in a real implementation
        typer.echo("⚠️  Model loading and training not implemented in this demo")
        typer.echo("✅ Training configuration validated successfully")
        
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
    model_path: Path = typer.Option(..., help="Path to trained model"),
    prompt: str = typer.Option(..., help="Input prompt"),
    adapter_path: Optional[Path] = typer.Option(None, help="Path to LoRA adapters"),
    max_length: int = typer.Option(100, help="Maximum tokens to generate"),
    temperature: float = typer.Option(0.7, help="Sampling temperature"),
    top_p: float = typer.Option(0.9, help="Nucleus sampling parameter"),
    top_k: int = typer.Option(50, help="Top-k sampling parameter"),
):
    """Generate text using a trained LoRA model."""
    
    typer.echo(f"✨ Generating text with LoRA model")
    typer.echo(f"📁 Model: {model_path}")
    typer.echo(f"💭 Prompt: {prompt}")
    
    try:
        # This would load model and generate text in a real implementation
        typer.echo("⚠️  Text generation not implemented in this demo")
        typer.echo(f"🎯 Configuration: max_length={max_length}, temp={temperature}")
        typer.echo("✅ Generation parameters validated successfully")
        
        # Mock response
        typer.echo("\n📝 Generated Text (Mock):")
        typer.echo("─" * 50)
        typer.echo(f"{prompt} This would be the generated continuation from the LoRA model...")
        typer.echo("─" * 50)
        
    except Exception as e:
        typer.echo(f"❌ Generation failed: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def info(
    config: Path = typer.Option("configs/default.yaml", help="Configuration file path"),
):
    """Show information about the LoRA framework configuration."""
    
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
    
    typer.echo(f"✅ Validating configuration: {config}")
    
    if not config.exists():
        typer.echo(f"❌ Configuration file not found: {config}", err=True)
        raise typer.Exit(1)
    
    try:
        # Load and validate configuration
        configs = load_config(config)
        
        lora_config = configs["lora"]
        training_config = configs["training"]
        inference_config = configs["inference"]
        
        typer.echo("✅ LoRA configuration: Valid")
        typer.echo("✅ Training configuration: Valid") 
        typer.echo("✅ Inference configuration: Valid")
        typer.echo("🎉 All configurations are valid!")
        
    except Exception as e:
        typer.echo(f"❌ Configuration validation failed: {e}", err=True)
        raise typer.Exit(1)


if __name__ == "__main__":
    app()