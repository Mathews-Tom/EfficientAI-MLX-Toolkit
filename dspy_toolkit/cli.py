"""
Command-line interface for DSPy Integration Framework.
"""

# Standard library imports
import asyncio
import logging
from pathlib import Path
from typing import Optional

# Optional third-party imports
try:
    import typer
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.table import Table

    TYPER_AVAILABLE = True
except ImportError:
    TYPER_AVAILABLE = False
    typer = None
    Console = None

# Local imports
from .deployment import create_dspy_app
from .framework import DSPyFramework
from .types import DSPyConfig

if TYPER_AVAILABLE:
    app = typer.Typer(name="dspy-toolkit", help="DSPy Integration Framework CLI")
    console = Console()
else:
    app = None
    console = None

logger = logging.getLogger(__name__)


def main():
    """Main CLI entry point."""
    if not TYPER_AVAILABLE:
        print("CLI dependencies not available. Please install typer and rich.")
        return

    app()


@app.command()
def init(
    project_name: str = typer.Argument(..., help="Project name"),
    model_provider: str = typer.Option(
        "mlx", help="LLM provider (mlx, openai, anthropic)"
    ),
    model_name: str = typer.Option("mlx/mlx-7b", help="Model name"),
    cache_dir: str | None = typer.Option(None, help="Cache directory"),
):
    """Initialize a new DSPy integration project."""
    try:
        cache_path = (
            Path(cache_dir) if cache_dir else Path(f".dspy_cache/{project_name}")
        )

        config = DSPyConfig(
            model_provider=model_provider, model_name=model_name, cache_dir=cache_path
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Initializing DSPy framework...", total=None)

            framework = DSPyFramework(config)

            progress.update(task, description="Framework initialized successfully!")

        console.print(
            Panel(
                f"[green]DSPy Integration Framework initialized successfully![/green]\n\n"
                f"Project: {project_name}\n"
                f"Provider: {model_provider}\n"
                f"Model: {model_name}\n"
                f"Cache: {cache_path}",
                title="Initialization Complete",
            )
        )

    except Exception as e:
        console.print(f"[red]Initialization failed: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def serve(
    config_file: str | None = typer.Option(None, help="Configuration file path"),
    host: str = typer.Option("0.0.0.0", help="Host to bind to"),
    port: int = typer.Option(8000, help="Port to bind to"),
    reload: bool = typer.Option(False, help="Enable auto-reload"),
):
    """Start the DSPy FastAPI server."""
    try:
        # Load configuration
        if config_file:
            # Would implement config file loading
            console.print(f"Loading config from {config_file}")
            config = DSPyConfig()  # Placeholder
        else:
            config = DSPyConfig()

        # Create FastAPI app
        dspy_app = create_dspy_app(config)
        fastapi_app = dspy_app.get_app()

        console.print(
            Panel(
                f"[green]Starting DSPy FastAPI server...[/green]\n\n"
                f"Host: {host}\n"
                f"Port: {port}\n"
                f"Reload: {reload}",
                title="Server Starting",
            )
        )

        # Start server
        import uvicorn

        uvicorn.run(fastapi_app, host=host, port=port, reload=reload)

    except Exception as e:
        console.print(f"[red]Server startup failed: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def status(
    config_file: str | None = typer.Option(None, help="Configuration file path"),
):
    """Show DSPy framework status and health."""
    try:
        # Load configuration
        config = DSPyConfig()  # Would load from config file if provided

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Checking framework status...", total=None)

            framework = DSPyFramework(config)
            health = framework.health_check()
            stats = framework.get_framework_stats()

            progress.update(task, description="Status check complete!")

        # Display health status
        status_color = "green" if health["overall_status"] == "healthy" else "red"
        console.print(
            Panel(
                f"[{status_color}]Overall Status: {health['overall_status'].upper()}[/{status_color}]",
                title="Health Status",
            )
        )

        # Display component status
        table = Table(title="Component Status")
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="magenta")

        for component, status in health["components"].items():
            status_style = "green" if status == "healthy" else "red"
            table.add_row(component, f"[{status_style}]{status}[/{status_style}]")

        console.print(table)

        # Display issues if any
        if health["issues"]:
            console.print("\n[red]Issues:[/red]")
            for issue in health["issues"]:
                console.print(f"  • {issue}")

        # Display statistics
        console.print("\n[bold]Framework Statistics:[/bold]")
        if "signatures" in stats:
            console.print(
                f"  Signatures: {stats['signatures'].get('total_signatures', 0)}"
            )
        if "modules" in stats:
            console.print(f"  Modules: {stats['modules'].get('total_modules', 0)}")
        if "optimizer" in stats:
            console.print(
                f"  Optimizations: {stats['optimizer'].get('total_optimizations', 0)}"
            )

    except Exception as e:
        console.print(f"[red]Status check failed: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def list_modules(
    project: str | None = typer.Option(None, help="Filter by project name"),
):
    """List available DSPy modules."""
    try:
        config = DSPyConfig()
        framework = DSPyFramework(config)

        modules = framework.module_manager.list_modules()

        if not modules:
            console.print("[yellow]No modules found.[/yellow]")
            return

        table = Table(title="Available DSPy Modules")
        table.add_column("Module Name", style="cyan")
        table.add_column("Type", style="magenta")
        table.add_column("Registered", style="green")

        for module_name in modules:
            metadata = framework.module_manager.get_module_metadata(module_name)
            if metadata:
                module_type = metadata.get("module_type", "Unknown")
                registered_at = metadata.get("registered_at", "Unknown")

                # Filter by project if specified
                if project and not module_name.startswith(f"{project}_"):
                    continue

                table.add_row(module_name, module_type, registered_at)

        console.print(table)

    except Exception as e:
        console.print(f"[red]Failed to list modules: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def list_signatures(
    project: str | None = typer.Option(None, help="Project name"),
):
    """List available DSPy signatures."""
    try:
        config = DSPyConfig()
        framework = DSPyFramework(config)

        if project:
            signatures = framework.signature_registry.get_project_signatures(project)
            if not signatures:
                console.print(
                    f"[yellow]No signatures found for project {project}.[/yellow]"
                )
                return

            table = Table(title=f"Signatures for Project: {project}")
            table.add_column("Signature Name", style="cyan")
            table.add_column("Type", style="magenta")

            for sig_name, sig_class in signatures.items():
                table.add_row(sig_name, sig_class.__name__)

            console.print(table)
        else:
            all_signatures = framework.signature_registry.get_all_signatures()
            if not all_signatures:
                console.print("[yellow]No signatures found.[/yellow]")
                return

            for project_name, signatures in all_signatures.items():
                console.print(f"\n[bold]{project_name}:[/bold]")
                for sig_name, sig_class in signatures.items():
                    console.print(f"  • {sig_name} ({sig_class.__name__})")

    except Exception as e:
        console.print(f"[red]Failed to list signatures: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def benchmark(
    config_file: str | None = typer.Option(None, help="Configuration file path"),
):
    """Run framework benchmarks."""
    try:
        config = DSPyConfig()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Running benchmarks...", total=None)

            framework = DSPyFramework(config)
            results = framework.benchmark_framework()

            progress.update(task, description="Benchmarks complete!")

        console.print(
            Panel("[green]Benchmark Results[/green]", title="Performance Benchmarks")
        )

        for component, metrics in results.items():
            if isinstance(metrics, dict):
                console.print(f"\n[bold]{component}:[/bold]")
                for metric, value in metrics.items():
                    if isinstance(value, float):
                        console.print(f"  {metric}: {value:.4f}")
                    else:
                        console.print(f"  {metric}: {value}")

    except Exception as e:
        console.print(f"[red]Benchmark failed: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def export(
    output_dir: str = typer.Argument(..., help="Output directory for export"),
    config_file: str | None = typer.Option(None, help="Configuration file path"),
):
    """Export framework state and data."""
    try:
        config = DSPyConfig()
        export_path = Path(output_dir)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Exporting framework state...", total=None)

            framework = DSPyFramework(config)
            framework.export_framework_state(export_path)

            progress.update(task, description="Export complete!")

        console.print(
            Panel(
                f"[green]Framework state exported successfully![/green]\n\n"
                f"Export location: {export_path}",
                title="Export Complete",
            )
        )

    except Exception as e:
        console.print(f"[red]Export failed: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def clear_cache(
    confirm: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
):
    """Clear all framework caches."""
    try:
        if not confirm:
            confirm = typer.confirm("Are you sure you want to clear all caches?")
            if not confirm:
                console.print("Cache clear cancelled.")
                return

        config = DSPyConfig()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Clearing caches...", total=None)

            framework = DSPyFramework(config)
            framework.clear_all_caches()

            progress.update(task, description="Caches cleared!")

        console.print("[green]All caches cleared successfully![/green]")

    except Exception as e:
        console.print(f"[red]Cache clear failed: {e}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    main()
