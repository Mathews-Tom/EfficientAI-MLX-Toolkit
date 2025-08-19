"""
Command-line interface for the EfficientAI-MLX-Toolkit.

This module provides the main CLI entry point for the toolkit,
with commands for setup, benchmarking, and project management.
"""

import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from environment import EnvironmentSetup, detect_apple_silicon
from utils import BenchmarkRunner, get_logger, setup_logging

app = typer.Typer(
    name="efficientai-toolkit", help="Apple Silicon optimized AI toolkit", add_completion=False
)

console = Console()
logger = get_logger(__name__)


@app.command()
def setup(
    project_dir: Path = typer.Option(
        Path.cwd(), "--project-dir", "-p", help="Project directory path"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
) -> None:
    """Set up the development environment."""
    # Setup logging
    log_level = "DEBUG" if verbose else "INFO"
    setup_logging(log_level=log_level)

    console.print("[bold blue]EfficientAI-MLX-Toolkit Environment Setup[/bold blue]")

    try:
        env_setup = EnvironmentSetup(project_dir)
        results = env_setup.run_full_setup()

        # Display results table
        table = Table(title="Setup Results")
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="green")

        for component, success in results.items():
            status = "✅ Success" if success else "❌ Failed"
            table.add_row(component.replace("_", " ").title(), status)

        console.print(table)

        if all(results.values()):
            console.print("[bold green]Environment setup completed successfully![/bold green]")
        else:
            console.print("[bold red]Some setup steps failed. Check logs for details.[/bold red]")
            raise typer.Exit(1)

    except Exception as e:
        console.print(f"[bold red]Setup failed: {e}[/bold red]")
        logger.error("Environment setup failed: %s", e)
        raise typer.Exit(1)


@app.command()
def info() -> None:
    """Display system and hardware information."""
    console.print("[bold blue]System Information[/bold blue]")

    hardware_info = detect_apple_silicon()

    table = Table(title="Hardware Detection")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")

    for key, value in hardware_info.items():
        display_key = key.replace("_", " ").title()
        display_value = str(value)

        if isinstance(value, bool):
            display_value = "✅ Yes" if value else "❌ No"

        table.add_row(display_key, display_value)

    console.print(table)


@app.command()
def benchmark(
    name: str = typer.Argument(..., help="Benchmark name"),
    iterations: int = typer.Option(3, "--iterations", "-i", help="Number of iterations"),
    output_dir: Path = typer.Option(
        Path("benchmark_results"), "--output-dir", "-o", help="Output directory for results"
    ),
) -> None:
    """Run a simple benchmark test."""
    console.print(f"[bold blue]Running benchmark: {name}[/bold blue]")

    try:
        runner = BenchmarkRunner(output_dir)

        def sample_benchmark() -> dict[str, float]:
            """Sample benchmark function."""
            import random
            import time

            # Simulate some work
            time.sleep(random.uniform(0.1, 0.5))

            return {"accuracy": random.uniform(0.8, 0.95), "f1_score": random.uniform(0.75, 0.90)}

        result = runner.run_benchmark(name, sample_benchmark, iterations=iterations)

        # Display results
        table = Table(title=f"Benchmark Results: {name}")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Execution Time", f"{result.execution_time:.4f}s")
        table.add_row("Success", "✅ Yes" if result.success else "❌ No")

        for metric, value in result.performance_metrics.items():
            table.add_row(metric.replace("_", " ").title(), f"{value:.4f}")

        console.print(table)

        # Save results
        output_file = runner.save_results()
        console.print(f"[green]Results saved to: {output_file}[/green]")

    except Exception as e:
        console.print(f"[bold red]Benchmark failed: {e}[/bold red]")
        logger.error("Benchmark failed: %s", e)
        raise typer.Exit(1)


def main() -> None:
    """Main CLI entry point."""
    app()


if __name__ == "__main__":
    main()
