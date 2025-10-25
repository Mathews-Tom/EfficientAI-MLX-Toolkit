"""
Complete Training Workflow Example

Demonstrates end-to-end training pipeline with MLOps integration:
1. Data versioning with DVC
2. Experiment tracking with MLFlow
3. Apple Silicon optimization
4. Model artifact storage
5. Performance monitoring

Usage:
    uv run python training_workflow.py
    uv run python training_workflow.py --config custom_config.yaml
    uv run python training_workflow.py --dry-run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from mlops.client.mlops_client import MLOpsClient


def load_config(config_path: str | None = None) -> dict:
    """Load training configuration."""
    default_config = {
        "project_namespace": "training-example",
        "model_name": "example_model",
        "learning_rate": 0.001,
        "batch_size": 32,
        "num_epochs": 10,
        "dataset_path": "data/train.csv",
        "model_output_path": "outputs/model",
        "enable_monitoring": True,
        "enable_versioning": True,
    }

    if config_path and Path(config_path).exists():
        import yaml

        with open(config_path) as f:
            custom_config = yaml.safe_load(f)
        default_config.update(custom_config)

    return default_config


def prepare_dataset(dataset_path: str) -> pd.DataFrame:
    """Load and prepare dataset."""
    # Example: Create synthetic dataset
    if not Path(dataset_path).exists():
        print(f"Creating synthetic dataset at {dataset_path}")
        Path(dataset_path).parent.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame(
            {
                "feature1": range(1000),
                "feature2": range(1000, 2000),
                "feature3": [i * 0.5 for i in range(1000)],
                "target": [i % 2 for i in range(1000)],
            }
        )
        df.to_csv(dataset_path, index=False)
    else:
        df = pd.read_csv(dataset_path)

    print(f"Loaded dataset: {len(df)} samples")
    return df


def train_model(config: dict, data: pd.DataFrame, mlops_client: MLOpsClient):
    """Train model with MLOps tracking."""
    print("\n=== Starting Training ===")

    # Log training configuration
    mlops_client.log_params(
        {
            "learning_rate": config["learning_rate"],
            "batch_size": config["batch_size"],
            "num_epochs": config["num_epochs"],
            "dataset_size": len(data),
        }
    )

    # Collect Apple Silicon metrics at start
    silicon_metrics = mlops_client.collect_apple_silicon_metrics()
    print(f"\nApple Silicon: {silicon_metrics.get('chip_type', 'Unknown')}")
    print(f"Unified Memory: {silicon_metrics.get('unified_memory_gb', 0):.1f} GB")

    # Training loop
    for epoch in range(config["num_epochs"]):
        # Simulate training
        train_loss = 1.0 - (epoch * 0.08)  # Simulated loss decrease
        train_acc = 0.5 + (epoch * 0.04)  # Simulated accuracy increase

        # Log metrics
        mlops_client.log_metrics(
            {"train_loss": train_loss, "train_accuracy": train_acc}, step=epoch
        )

        print(
            f"Epoch {epoch + 1}/{config['num_epochs']}: "
            f"Loss={train_loss:.4f}, Acc={train_acc:.4f}"
        )

    print("\n=== Training Complete ===")

    # Final metrics
    final_metrics = {"final_loss": train_loss, "final_accuracy": train_acc}

    mlops_client.log_metrics(final_metrics)

    return final_metrics


def save_model(model_output_path: str, metrics: dict, mlops_client: MLOpsClient):
    """Save model artifacts."""
    print(f"\n=== Saving Model to {model_output_path} ===")

    output_dir = Path(model_output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save model (simulated)
    model_file = output_dir / "model.bin"
    model_file.write_text("model_weights_placeholder")

    # Save config
    config_file = output_dir / "config.json"
    with open(config_file, "w") as f:
        json.dump({"metrics": metrics}, f, indent=2)

    # Log artifacts
    mlops_client.log_artifact(str(model_file))
    mlops_client.log_artifact(str(config_file))

    print(f"Model saved: {model_file}")
    print(f"Config saved: {config_file}")


def main():
    """Run complete training workflow."""
    parser = argparse.ArgumentParser(description="MLOps Training Workflow Example")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument(
        "--dry-run", action="store_true", help="Run without actual training"
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    print("=" * 60)
    print("MLOps Training Workflow Example")
    print("=" * 60)
    print(f"\nProject: {config['project_namespace']}")
    print(f"Model: {config['model_name']}")

    if args.dry_run:
        print("\n[DRY RUN MODE]")
        print(json.dumps(config, indent=2))
        return

    # Initialize MLOps client
    mlops_client = MLOpsClient(project_namespace=config["project_namespace"])

    # Create experiment
    experiment_id = mlops_client.create_experiment(
        experiment_name=f"{config['model_name']}_training",
        tags={"model": config["model_name"], "workflow": "training"},
    )
    print(f"\nExperiment ID: {experiment_id}")

    # Prepare dataset
    data = prepare_dataset(config["dataset_path"])

    # Version dataset with DVC
    if config["enable_versioning"]:
        print(f"\n=== Versioning Dataset ===")
        try:
            mlops_client.version_dataset(config["dataset_path"], push_to_remote=False)
            print(f"Dataset versioned: {config['dataset_path']}")
        except Exception as e:
            print(f"Warning: Dataset versioning failed: {e}")

    # Start training run
    with mlops_client.start_run(
        run_name=f"{config['model_name']}_run", experiment_id=experiment_id
    ):
        # Train model
        metrics = train_model(config, data, mlops_client)

        # Save model
        save_model(config["model_output_path"], metrics, mlops_client)

        # Set up monitoring reference data
        if config["enable_monitoring"]:
            print("\n=== Setting Up Monitoring ===")
            try:
                mlops_client.set_reference_data(data)
                print("Reference data configured for monitoring")
            except Exception as e:
                print(f"Warning: Monitoring setup failed: {e}")

    print("\n" + "=" * 60)
    print("Training Workflow Complete!")
    print("=" * 60)
    print(f"\nArtifacts saved to: {config['model_output_path']}")
    print(f"MLFlow UI: http://localhost:5000")
    print(f"Experiment: {experiment_id}")


if __name__ == "__main__":
    main()
