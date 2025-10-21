"""CLI for Meta-Learning PEFT project."""

from pathlib import Path

import typer
from rich.console import Console

# Conditional import pattern for toolkit integration
try:
    from utils.logging_utils import get_logger, setup_logging

    SHARED_UTILS_AVAILABLE = True
except ImportError:
    import logging

    SHARED_UTILS_AVAILABLE = False

    def setup_logging(level: str = "INFO") -> None:
        logging.basicConfig(level=level)

    def get_logger(name: str) -> logging.Logger:
        return logging.getLogger(name)


from utils.config import load_config

app = typer.Typer(
    name="meta-learning-peft",
    help="Meta-Learning for Parameter-Efficient Fine-Tuning with MLX",
)
console = Console()
logger = get_logger(__name__)


@app.command()
def info() -> None:
    """Display project information."""
    console.print("\n[bold blue]Meta-Learning PEFT with MLX[/bold blue]")
    console.print("\n[bold]Phase:[/bold] META-002 Research and Prototyping")
    console.print("[bold]Status:[/bold] Active Development")
    console.print("\n[bold]Components:[/bold]")
    console.print("  - Reptile meta-learning algorithm")
    console.print("  - Task distribution (synthetic tasks)")
    console.print("  - Baseline benchmarking")
    console.print("\n[bold]Documentation:[/bold]")
    console.print("  - Literature Review: research/literature_review.md")
    console.print("  - Task Distribution: research/task_distribution_design.md")
    console.print("\n[bold]Configuration:[/bold]")
    console.print("  - Default config: configs/default.yaml")
    console.print()


@app.command()
def validate() -> None:
    """Validate project setup and dependencies."""
    console.print("\n[bold]Validating Meta-Learning PEFT Setup...[/bold]\n")

    try:
        import mlx.core as mx

        console.print("[green]✓[/green] MLX framework available")
    except ImportError:
        console.print("[red]✗[/red] MLX framework not found")
        return

    try:
        import mlx.nn as nn

        console.print("[green]✓[/green] MLX neural network module available")
    except ImportError:
        console.print("[red]✗[/red] MLX neural network module not found")
        return

    # Check configuration
    try:
        config = load_config()
        console.print("[green]✓[/green] Configuration file loaded")
        console.print(
            f"  - Algorithm: {config.get('meta_learning', {}).get('algorithm', 'N/A')}"
        )
        console.print(
            f"  - Meta batch size: {config.get('meta_learning', {}).get('meta_batch_size', 'N/A')}"
        )
    except Exception as e:
        console.print(f"[red]✗[/red] Configuration error: {e}")
        return

    # Check research documentation
    project_root = Path(__file__).parent.parent
    lit_review = project_root / "research" / "literature_review.md"
    task_design = project_root / "research" / "task_distribution_design.md"

    if lit_review.exists():
        console.print("[green]✓[/green] Literature review documentation found")
    else:
        console.print("[yellow]⚠[/yellow] Literature review not found")

    if task_design.exists():
        console.print("[green]✓[/green] Task distribution design found")
    else:
        console.print("[yellow]⚠[/yellow] Task distribution design not found")

    console.print("\n[bold green]Validation complete![/bold green]\n")


@app.command()
def train(
    config_path: str = typer.Option(
        None, "--config", "-c", help="Path to configuration file"
    ),
    output_dir: str = typer.Option(
        "outputs/checkpoints", "--output", "-o", help="Output directory"
    ),
    log_level: str = typer.Option("INFO", "--log-level", help="Logging level"),
) -> None:
    """Train meta-learning model (placeholder for future implementation)."""
    setup_logging(level=log_level)
    logger.info("Meta-learning training is planned for Phase 2 (META-003)")

    console.print("\n[bold yellow]Training Not Yet Implemented[/bold yellow]")
    console.print("\n[bold]Current Phase:[/bold] META-002 (Research and Prototyping)")
    console.print(
        "[bold]Training Implementation:[/bold] Planned for META-003 (Meta-Learning Framework)"
    )
    console.print("\n[bold]Current Capabilities:[/bold]")
    console.print("  - Reptile algorithm implementation (src/meta_learning/reptile.py)")
    console.print("  - Task distribution (src/task_embedding/task_distribution.py)")
    console.print("  - Baseline evaluation (src/utils/baseline.py)")
    console.print("\n[bold]Next Steps:[/bold]")
    console.print("  1. Complete test suite (tests/)")
    console.print("  2. Run baseline benchmarks")
    console.print("  3. Validate research infrastructure")
    console.print("  4. Proceed to META-003 for full training pipeline")
    console.print()


@app.command()
def benchmark(
    config_path: str = typer.Option(
        None, "--config", "-c", help="Path to configuration file"
    ),
    output_dir: str = typer.Option(
        "outputs/results", "--output", "-o", help="Output directory for results"
    ),
) -> None:
    """Run baseline benchmarks (placeholder for future implementation)."""
    console.print("\n[bold yellow]Benchmark Not Yet Implemented[/bold yellow]")
    console.print(
        "\n[bold]Baseline Utilities Available:[/bold] src/utils/baseline.py"
    )
    console.print(
        "[bold]Full Benchmark Suite:[/bold] Planned for META-002 completion"
    )
    console.print()


if __name__ == "__main__":
    app()
