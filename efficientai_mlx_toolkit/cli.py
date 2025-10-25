"""
Command-line interface for the EfficientAI-MLX-Toolkit.

This module provides the main CLI entry point for the toolkit,
with commands for setup, benchmarking, and namespace-based project management.

Supports both unified CLI access (namespace:command) and standalone project execution.
"""

import logging
import subprocess
import sys
from importlib import import_module
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from environment import EnvironmentSetup, detect_apple_silicon
from utils import BenchmarkRunner, get_logger, setup_logging

app = typer.Typer(
    name="efficientai-toolkit",
    help="Apple Silicon optimized AI toolkit. Use 'namespace:command' syntax for project commands.",
    add_completion=False,
    invoke_without_command=True,
    no_args_is_help=True,
)

console = Console()
logger = get_logger(__name__)


def discover_projects() -> dict[str, Path]:
    """Discover all valid projects in the projects directory."""
    projects_dir = Path(__file__).parent.parent / "projects"
    discovered_projects = {}

    if not projects_dir.exists():
        return discovered_projects

    for project_path in projects_dir.iterdir():
        if not project_path.is_dir() or project_path.name.startswith("."):
            continue

        # Check if it has a CLI module
        cli_path = project_path / "src" / "cli.py"
        if cli_path.exists():
            # Convert project name to CLI-friendly format
            project_name = project_path.name.lower().replace("_", "-").replace(" ", "-")
            # Remove leading numbers and separators
            if project_name.startswith(
                ("01-", "02-", "03-", "04-", "05-", "06-", "07-", "08-", "09-", "10-")
            ):
                project_name = project_name[3:]

            discovered_projects[project_name] = project_path

    return discovered_projects


def run_namespaced_command(project_name: str, command: str, args: list[str]):
    """Execute a command from a specific project namespace."""
    projects = discover_projects()
    project_path = projects.get(project_name)

    if not project_path:
        console.print(
            f"[bold red]Error:[/bold red] Project namespace '{project_name}' not found."
        )
        console.print("\nAvailable namespaces:")
        for available_project in sorted(projects.keys()):
            console.print(f"  • {available_project}")
        console.print(
            "\nUse [cyan]uv run efficientai-toolkit projects[/cyan] to see all available projects."
        )
        raise typer.Exit(1)

    # Check if project has a CLI module
    cli_path = project_path / "src" / "cli.py"
    if not cli_path.exists():
        console.print(
            f"[bold red]Error:[/bold red] Project '{project_name}' does not have a CLI module."
        )
        console.print(f"Expected: {cli_path}")
        raise typer.Exit(1)

    try:
        # Add the project's src directory to Python path temporarily
        src_path = str(project_path / "src")
        original_path = sys.path.copy()

        if src_path not in sys.path:
            sys.path.insert(0, src_path)

        # Import the CLI module
        cli_module = import_module("cli")

        if not hasattr(cli_module, "app"):
            console.print(
                f"[bold red]Error:[/bold red] Project '{project_name}' CLI module does not have an 'app' attribute."
            )
            raise typer.Exit(1)

        # Invoke the project's CLI app directly with the command and arguments
        try:
            # Save the original sys.argv and replace it temporarily
            original_argv = sys.argv.copy()
            sys.argv = ["cli.py", command] + args

            # Invoke the CLI app
            cli_module.app(standalone_mode=False)

        except typer.Exit as e:
            # Re-raise typer exits as-is
            raise e
        except SystemExit as e:
            # Convert SystemExit to typer.Exit
            raise typer.Exit(e.code)
        except Exception as e:
            console.print(
                f"[bold red]Error running command '{command}' in project '{project_name}':[/bold red]"
            )
            console.print(f"  {str(e)}")
            raise typer.Exit(1)
        finally:
            # Restore the original sys.argv
            sys.argv[:] = original_argv

    except ImportError as e:
        console.print(
            f"[bold red]Error:[/bold red] Failed to import CLI module for project '{project_name}'."
        )
        console.print(f"  {str(e)}")
        console.print("\nMake sure the project dependencies are installed:")
        console.print(f"  cd {project_path}")
        console.print(f"  uv sync")
        raise typer.Exit(1)
    except Exception as e:
        console.print(
            f"[bold red]Error:[/bold red] Failed to run command '{command}' for project '{project_name}'."
        )
        console.print(f"  {str(e)}")
        raise typer.Exit(1)
    finally:
        # Restore original Python path
        sys.path[:] = original_path


@app.command()
def setup(
    project_dir: Path = typer.Option(
        Path.cwd(), "--project-dir", "-p", help="Project directory path"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging"
    ),
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
            console.print(
                "[bold green]Environment setup completed successfully![/bold green]"
            )
        else:
            console.print(
                "[bold red]Some setup steps failed. Check logs for details.[/bold red]"
            )
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
    iterations: int = typer.Option(
        3, "--iterations", "-i", help="Number of iterations"
    ),
    output_dir: Path = typer.Option(
        Path("benchmark_results"),
        "--output-dir",
        "-o",
        help="Output directory for results",
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

            return {
                "accuracy": random.uniform(0.8, 0.95),
                "f1_score": random.uniform(0.75, 0.90),
            }

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


@app.command()
def test(
    project: str | None = typer.Argument(
        None, help="Project namespace to test (e.g., 'lora-finetuning-mlx')"
    ),
    all_projects: bool = typer.Option(
        False, "--all", help="Run tests for all projects"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose test output"
    ),
    coverage: bool = typer.Option(
        False, "--coverage", "-c", help="Run tests with coverage"
    ),
    markers: str | None = typer.Option(
        None, "--markers", "-m", help="Pytest markers to filter tests"
    ),
    parallel: bool = typer.Option(
        False, "--parallel", "-p", help="Run tests in parallel"
    ),
) -> None:
    """Run tests for projects with unified command."""

    if not project and not all_projects:
        console.print(
            "[bold red]Error:[/bold red] Must specify either a project namespace or --all"
        )
        console.print("\nAvailable namespaces:")
        projects = discover_projects()
        if projects:
            for proj_name in sorted(projects.keys()):
                console.print(f"  • {proj_name}")
        else:
            console.print("  No project namespaces found")
        console.print(
            "\nUse [cyan]uv run efficientai-toolkit projects[/cyan] to see all available projects."
        )
        raise typer.Exit(1)

    projects_to_test = []
    projects = discover_projects()

    if all_projects:
        projects_to_test = list(projects.items())
        console.print(
            f"[bold blue]Running tests for all {len(projects_to_test)} project namespaces[/bold blue]"
        )
    else:
        projects_to_test = [(project, projects[project])]
        console.print(
            f"[bold blue]Running tests for project namespace: {project}[/bold blue]"
        )

    # Build test command
    base_cmd = ["uv", "run", "pytest"]

    if verbose:
        base_cmd.append("-v")

    if coverage:
        base_cmd.extend(["--cov", "--cov-report=term-missing"])

    if markers:
        base_cmd.extend(["-m", markers])

    if parallel:
        base_cmd.extend(["-n", "auto"])

    failed_projects = []

    # Validate project selection
    if project and project not in projects:
        console.print(
            f"[bold red]Error:[/bold red] Project namespace '{project}' not found"
        )
        console.print("\nAvailable namespaces:")
        for proj_name in sorted(projects.keys()):
            console.print(f"  • {proj_name}")
        console.print(
            "\nUse [cyan]uv run efficientai-toolkit projects[/cyan] to see all available projects."
        )
        raise typer.Exit(1)

    for project_name, project_path in projects_to_test:
        console.print(f"\n[cyan]Testing {project_name}...[/cyan]")

        # Check if project has tests
        test_dir = project_path / "tests"
        if not test_dir.exists():
            console.print(
                f"[yellow]  No tests directory found for {project_name}[/yellow]"
            )
            continue

        # Run tests in project directory
        test_cmd = base_cmd + [str(test_dir)]

        try:
            result = subprocess.run(
                test_cmd, cwd=project_path, capture_output=not verbose, text=True
            )

            if result.returncode == 0:
                console.print(f"[green]  ✅ {project_name} tests passed[/green]")
            else:
                console.print(f"[red]  ❌ {project_name} tests failed[/red]")
                failed_projects.append(project_name)
                if not verbose and result.stdout:
                    console.print(f"[dim]  {result.stdout}[/dim]")
                if not verbose and result.stderr:
                    console.print(f"[red]  {result.stderr}[/red]")

        except subprocess.CalledProcessError as e:
            console.print(
                f"[red]  ❌ Failed to run tests for {project_name}: {e}[/red]"
            )
            failed_projects.append(project_name)
        except Exception as e:
            console.print(f"[red]  ❌ Unexpected error for {project_name}: {e}[/red]")
            failed_projects.append(project_name)

    # Summary
    console.print(f"\n[bold]Test Summary:[/bold]")
    total_projects = len(projects_to_test)
    passed_projects = total_projects - len(failed_projects)

    console.print(f"  • Total: {total_projects}")
    console.print(f"  • Passed: [green]{passed_projects}[/green]")
    console.print(f"  • Failed: [red]{len(failed_projects)}[/red]")

    if failed_projects:
        console.print(f"\n[red]Failed projects:[/red]")
        for failed in failed_projects:
            console.print(f"  • {failed}")
        raise typer.Exit(1)
    else:
        console.print("\n[green]🎉 All tests passed![/green]")


@app.command()
def projects() -> None:
    """List all available projects and their namespaces."""
    console.print("[bold blue]Available Projects:[/bold blue]")

    discovered_projects = discover_projects()
    if not discovered_projects:
        console.print("  No projects found in the projects directory")
        return

    table = Table(title="Projects and Namespaces")
    table.add_column("Project", style="cyan")
    table.add_column("Namespace", style="yellow")
    table.add_column("Path", style="green")
    table.add_column("CLI Available", style="magenta")
    table.add_column("Usage Example", style="dim")

    # Sort by directory name (which includes ordering numbers like 01_, 02_, etc.)
    for project_name, project_path in sorted(discovered_projects.items(), key=lambda x: x[1].name):
        cli_available = "✅" if (project_path / "src" / "cli.py").exists() else "❌"
        # Provide more appropriate usage examples based on project type
        if cli_available == "✅":
            if "lora" in project_name:
                usage_example = f"uv run efficientai-toolkit {project_name}:train"
            elif "compression" in project_name:
                usage_example = f"uv run efficientai-toolkit {project_name}:quantize"
            elif "coreml" in project_name:
                usage_example = f"uv run efficientai-toolkit {project_name}:transfer"
            else:
                usage_example = f"uv run efficientai-toolkit {project_name}:info"
        else:
            usage_example = "N/A"
        project_display_name = project_path.name
        table.add_row(
            project_display_name,
            project_name,
            str(project_path.relative_to(Path.cwd())),
            cli_available,
            usage_example,
        )

    console.print(table)

    if any(
        (project_path / "src" / "cli.py").exists()
        for project_path in discovered_projects.values()
    ):
        console.print(
            "\n[dim]Use the format:[/dim] [cyan]uv run efficientai-toolkit namespace:command[/cyan] [dim]to run project commands[/dim]"
        )


@app.command()
def dashboard(
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Server host address"),
    port: int = typer.Option(8000, "--port", "-p", help="Server port"),
    reload: bool = typer.Option(
        False, "--reload", "-r", help="Enable auto-reload for development"
    ),
    repo_root: Path | None = typer.Option(
        None, "--repo-root", help="Repository root directory"
    ),
) -> None:
    """Start the unified MLOps dashboard server."""
    console.print("[bold blue]EfficientAI MLOps Dashboard[/bold blue]")
    console.print(f"Starting server at http://{host}:{port}\n")

    try:
        from mlops.dashboard import DashboardServer

        # Use current directory if repo_root not specified
        if repo_root is None:
            repo_root = Path.cwd()

        server = DashboardServer(repo_root=repo_root, host=host, port=port)

        console.print("[green]Dashboard is running![/green]")
        console.print(f"Access at: [cyan]http://{host}:{port}[/cyan]")
        console.print("\nPress Ctrl+C to stop the server.\n")

        server.run(reload=reload)

    except ImportError as e:
        console.print(
            "[bold red]Error:[/bold red] Dashboard dependencies not installed."
        )
        console.print(f"  {str(e)}")
        console.print("\nInstall dashboard dependencies:")
        console.print("  [cyan]uv add fastapi uvicorn jinja2 python-multipart[/cyan]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[bold red]Failed to start dashboard:[/bold red] {e}")
        logger.error("Dashboard startup failed: %s", e)
        raise typer.Exit(1)


def main() -> None:
    """Main CLI entry point."""
    # Check for namespace:command pattern before typer processes arguments
    if len(sys.argv) > 1:
        potential_command = sys.argv[1]
        known_commands = [
            "setup",
            "info",
            "benchmark",
            "test",
            "projects",
            "dashboard",
            "--help",
            "-h",
        ]

        # Check for namespace:command pattern (but not if it's a known subcommand or help)
        if (
            ":" in potential_command
            and potential_command not in known_commands
            and not potential_command.startswith("-")
        ):
            try:
                namespace, command = potential_command.split(":", 1)
                remaining_args = sys.argv[2:]

                # Validate namespace and command
                if not namespace or not command:
                    console.print(
                        f"[bold red]Error:[/bold red] Invalid namespace:command format: '{potential_command}'"
                    )
                    console.print("Expected format: [cyan]namespace:command[/cyan]")
                    raise typer.Exit(1)

                # Run the namespaced command
                try:
                    run_namespaced_command(namespace, command, remaining_args)
                except typer.Exit as e:
                    sys.exit(e.exit_code)
                return  # Exit after successful execution

            except ValueError:
                console.print(
                    f"[bold red]Error:[/bold red] Invalid namespace:command format: '{potential_command}'"
                )
                console.print("Expected format: [cyan]namespace:command[/cyan]")
                raise typer.Exit(1)

    # If no namespace:command pattern, proceed with normal typer processing
    app()


if __name__ == "__main__":
    main()
