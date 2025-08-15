"""
Knowledge Base CLI using Typer and Rich

A modern, beautiful command-line interface for managing the knowledge base.
"""

import sys
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text

# Add the .meta directory to Python path for imports
kb_meta_path = Path(__file__).parent.parent / ".knowledge-base" / ".meta"
sys.path.insert(0, str(kb_meta_path))

# Import knowledge base modules
from contributor import KnowledgeBaseContributor
from freshness_tracker import ContentFreshnessTracker
from indexer import KnowledgeBaseIndexer
from models import KnowledgeBaseEntry, KnowledgeBaseIndex
from quality_assurance import KnowledgeBaseQualityAssurance
from search import KnowledgeBaseSearcher

# Initialize Rich console
console = Console()

# Create the main Typer app
app = typer.Typer(
    name="kb",
    help="🧠 Knowledge Base Management CLI for EfficientAI-MLX-Toolkit",
    add_completion=False,
    rich_markup_mode="rich",
)


class KnowledgeBaseCLI:
    """Knowledge Base CLI with Rich UI components."""

    def __init__(self, kb_path: Path):
        self.kb_path = Path(kb_path)

        # Initialize components
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("Initializing knowledge base...", total=None)

            self.indexer = KnowledgeBaseIndexer(self.kb_path)
            self.contributor = KnowledgeBaseContributor(self.kb_path)
            self.search = KnowledgeBaseSearcher(self.kb_path)
            self.qa = KnowledgeBaseQualityAssurance(self.kb_path)
            self.freshness_tracker = ContentFreshnessTracker(self.kb_path)

            # Load or build index
            self.index = self.indexer.load_index_from_file()
            if not self.index:
                progress.update(task, description="Building knowledge base index...")
                self.index = self.indexer.build_index()

    def display_entry_table(
        self, entries: List[KnowledgeBaseEntry], title: str = "Entries"
    ):
        """Display entries in a beautiful table."""
        table = Table(title=title, show_header=True, header_style="bold magenta")
        table.add_column("Title", style="cyan", no_wrap=False)
        table.add_column("Category", style="green")
        table.add_column("Tags", style="yellow")
        table.add_column("Updated", style="blue")

        for entry in entries:
            tags_text = ", ".join(entry.tags[:3])  # Show first 3 tags
            if len(entry.tags) > 3:
                tags_text += f" +{len(entry.tags) - 3}"

            table.add_row(
                entry.title,
                entry.category,
                tags_text,
                entry.last_updated.strftime("%Y-%m-%d"),
            )

        console.print(table)

    def display_search_results(self, search_results, query: str):
        """Display search results with relevance scores."""
        if not search_results or not search_results.results:
            console.print(f"[yellow]No results found for: {query}[/yellow]")
            return

        console.print(
            f"\n[bold green]Found {len(search_results.results)} results for: {query}[/bold green]\n"
        )

        for i, result in enumerate(search_results.results, 1):
            entry = result.entry
            score = result.score

            # Create a panel for each result
            content = f"""[bold cyan]{entry.title}[/bold cyan]
[dim]Category:[/dim] {entry.category}
[dim]Tags:[/dim] {', '.join(entry.tags)}
[dim]Relevance:[/dim] {score:.2f}
[dim]Path:[/dim] {entry.content_path}"""

            panel = Panel(content, title=f"Result {i}", border_style="blue")
            console.print(panel)


# Global CLI instance
cli_instance = None


def get_cli(
    kb_path: Path = typer.Option(
        Path(".knowledge-base"), help="Path to knowledge base directory"
    )
) -> KnowledgeBaseCLI:
    """Get or create CLI instance."""
    global cli_instance
    if cli_instance is None or cli_instance.kb_path != kb_path:
        cli_instance = KnowledgeBaseCLI(kb_path)
    return cli_instance


@app.command()
def create(
    title: str = typer.Argument(..., help="Entry title"),
    category: Optional[str] = typer.Option(
        None, "--category", "-c", help="Entry category"
    ),
    tags: Optional[str] = typer.Option(
        None, "--tags", "-t", help="Comma-separated tags"
    ),
    difficulty: Optional[str] = typer.Option(
        "intermediate", "--difficulty", "-d", help="Difficulty level"
    ),
    contributor: Optional[str] = typer.Option(
        None, "--contributor", help="Contributor name"
    ),
    template: Optional[str] = typer.Option(
        "general", "--template", help="Template type"
    ),
    interactive: bool = typer.Option(
        False, "--interactive", "-i", help="Interactive mode"
    ),
    kb_path: Path = typer.Option(
        Path(".knowledge-base"), help="Path to knowledge base directory"
    ),
):
    """✨ Create a new knowledge base entry."""
    cli = get_cli(kb_path)

    try:
        if interactive:
            # Interactive mode
            console.print(Panel("🔧 Interactive Entry Creation", style="bold blue"))

            title = typer.prompt("Entry title")

            # Show available categories
            categories = list(cli.index.categories)
            if categories:
                console.print(f"Available categories: {', '.join(categories)}")
            category = typer.prompt("Category", default="general")

            # Show popular tags
            popular_tags = list(cli.index.tags)[:10]
            if popular_tags:
                console.print(f"Popular tags: {', '.join(popular_tags)}")
            tags_input = typer.prompt("Tags (comma-separated)", default="")
            tags = [tag.strip() for tag in tags_input.split(",") if tag.strip()]

            difficulty = typer.prompt("Difficulty", default="intermediate")
            contributor = typer.prompt("Contributor name", default="Unknown")
        else:
            # Use provided arguments
            tags = tags.split(",") if tags else []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Creating entry...", total=None)

            entry_path = cli.contributor.create_entry_from_template(
                title=title,
                category=category or "general",
                tags=tags,
                difficulty=difficulty,
                contributor=contributor or "Unknown",
                entry_type=template or "standard",
            )

            progress.update(task, description="Rebuilding index...")
            cli.index = cli.indexer.build_index()

        console.print(f"[bold green]✅ Entry created successfully![/bold green]")
        console.print(f"[dim]Path: {entry_path}[/dim]")

    except Exception as e:
        console.print(f"[bold red]❌ Failed to create entry: {e}[/bold red]")
        raise typer.Exit(1)


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    mode: Optional[str] = typer.Option(
        "fuzzy", "--mode", "-m", help="Search mode (fuzzy, exact, semantic)"
    ),
    category: Optional[str] = typer.Option(
        None, "--category", "-c", help="Filter by category"
    ),
    tags: Optional[str] = typer.Option(
        None, "--tags", "-t", help="Filter by tags (comma-separated)"
    ),
    limit: Optional[int] = typer.Option(10, "--limit", "-l", help="Maximum results"),
    kb_path: Path = typer.Option(
        Path(".knowledge-base"), help="Path to knowledge base directory"
    ),
):
    """🔍 Search knowledge base entries."""
    cli = get_cli(kb_path)

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task(f"Searching for: {query}", total=None)

            # Create search filter
            from search import SearchFilter, SearchMode

            search_filter = None
            if category or tags:
                search_filter = SearchFilter(
                    categories=[category] if category else None,
                    tags=tags.split(",") if tags else None,
                )

            # Map mode string to SearchMode enum
            search_mode = SearchMode.FUZZY  # default
            if mode == "exact":
                search_mode = SearchMode.EXACT
            elif mode == "semantic":
                search_mode = SearchMode.SEMANTIC

            results = cli.search.search(
                query=query, filters=search_filter, mode=search_mode, limit=limit
            )

        cli.display_search_results(results, query)

    except Exception as e:
        console.print(f"[bold red]❌ Search failed: {e}[/bold red]")
        raise typer.Exit(1)


@app.command()
def list(
    category: Optional[str] = typer.Option(
        None, "--category", "-c", help="Filter by category"
    ),
    tags: Optional[str] = typer.Option(
        None, "--tags", "-t", help="Filter by tags (comma-separated)"
    ),
    sort: str = typer.Option(
        "title", "--sort", "-s", help="Sort by (title, date, category)"
    ),
    limit: Optional[int] = typer.Option(
        None, "--limit", "-l", help="Maximum entries to show"
    ),
    kb_path: Path = typer.Option(
        Path(".knowledge-base"), help="Path to knowledge base directory"
    ),
):
    """📋 List knowledge base entries."""
    cli = get_cli(kb_path)

    try:
        entries = cli.index.entries

        # Apply filters
        if category:
            entries = [e for e in entries if e.category == category]

        if tags:
            filter_tags = set(tags.split(","))
            entries = [e for e in entries if filter_tags.intersection(set(e.tags))]

        # Sort entries
        if sort == "title":
            entries.sort(key=lambda e: e.title)
        elif sort == "date":
            entries.sort(key=lambda e: e.last_updated, reverse=True)
        elif sort == "category":
            entries.sort(key=lambda e: (e.category, e.title))

        # Limit results
        if limit:
            entries = entries[:limit]

        cli.display_entry_table(
            entries, f"Knowledge Base Entries ({len(entries)} total)"
        )

    except Exception as e:
        console.print(f"[bold red]❌ Failed to list entries: {e}[/bold red]")
        raise typer.Exit(1)


@app.command("rebuild-index")
def rebuild_index(
    kb_path: Path = typer.Option(
        Path(".knowledge-base"), help="Path to knowledge base directory"
    )
):
    """🔄 Rebuild the knowledge base index."""
    cli = get_cli(kb_path)

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Rebuilding index...", total=None)
            cli.index = cli.indexer.build_index(force_rebuild=True)

        # Display stats
        stats_table = Table(title="Index Statistics", show_header=False)
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="green")

        stats_table.add_row("Total entries", str(len(cli.index.entries)))
        stats_table.add_row("Categories", str(len(cli.index.categories)))
        stats_table.add_row("Tags", str(len(cli.index.tags)))

        console.print(stats_table)
        console.print("[bold green]✅ Index rebuilt successfully![/bold green]")

    except Exception as e:
        console.print(f"[bold red]❌ Index rebuild failed: {e}[/bold red]")
        raise typer.Exit(1)


@app.command("quality-check")
def quality_check(
    check_external_links: bool = typer.Option(
        False, "--check-external-links", help="Check external links"
    ),
    auto_fix: bool = typer.Option(False, "--auto-fix", help="Auto-fix issues"),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Output report file"
    ),
    kb_path: Path = typer.Option(
        Path(".knowledge-base"), help="Path to knowledge base directory"
    ),
):
    """🔍 Run quality assurance checks."""
    cli = get_cli(kb_path)

    try:
        # Configure QA
        cli.qa.check_external_links = check_external_links
        cli.qa.auto_fix_enabled = auto_fix

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Running quality checks...", total=None)
            report = cli.qa.run_comprehensive_quality_check()

        # Display quality report
        console.print(Panel("📊 Quality Report", style="bold blue"))

        # Create quality stats table
        stats_table = Table(show_header=False)
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="green")

        stats_table.add_row("Entries checked", str(report.total_entries_checked))
        stats_table.add_row("Quality score", f"{report.quality_score}/100")
        stats_table.add_row("Total issues", str(len(report.issues_found)))

        console.print(stats_table)

        # Show issues by severity
        if report.issues_by_severity:
            console.print("\n[bold]Issues by severity:[/bold]")
            for severity, issues in report.issues_by_severity.items():
                color = {
                    "low": "yellow",
                    "medium": "orange",
                    "high": "red",
                    "critical": "bold red",
                }.get(severity, "white")
                console.print(
                    f"  [{color}]{severity.capitalize()}: {len(issues)}[/{color}]"
                )

        # Show recommendations
        if report.recommendations:
            console.print("\n[bold]💡 Recommendations:[/bold]")
            for rec in report.recommendations:
                console.print(f"  • {rec}")

        # Save report if requested
        if output:
            output_path = Path(output)
            cli.qa.generate_quality_report_json(output_path)
            console.print(f"\n[dim]📄 Report saved to: {output_path}[/dim]")

    except Exception as e:
        console.print(f"[bold red]❌ Quality check failed: {e}[/bold red]")
        raise typer.Exit(1)


@app.command()
def freshness(
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Output report file"
    ),
    kb_path: Path = typer.Option(
        Path(".knowledge-base"), help="Path to knowledge base directory"
    ),
):
    """🕒 Check content freshness and identify stale entries."""
    cli = get_cli(kb_path)

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Analyzing content freshness...", total=None)
            report = cli.freshness_tracker.analyze_content_freshness()

        # Display freshness report
        console.print(Panel("🕒 Freshness Report", style="bold blue"))

        # Create freshness stats table
        stats_table = Table(show_header=False)
        stats_table.add_column("Status", style="cyan")
        stats_table.add_column("Count", style="green")

        for status, count in report.freshness_breakdown.items():
            color = {
                "fresh": "green",
                "aging": "yellow",
                "stale": "orange",
                "critical": "red",
            }.get(status, "white")
            stats_table.add_row(f"[{color}]{status.capitalize()}[/{color}]", str(count))

        console.print(stats_table)

        # Show stale entries
        if report.stale_entries:
            console.print(
                f"\n[bold red]⚠️  Stale entries ({len(report.stale_entries)}):[/bold red]"
            )
            for entry in report.stale_entries[:10]:  # Show top 10
                console.print(
                    f"  • {entry.entry_title} ([red]{entry.days_since_update} days old[/red])"
                )

        # Show recommendations
        if report.recommendations:
            console.print("\n[bold]💡 Recommendations:[/bold]")
            for rec in report.recommendations:
                console.print(f"  • {rec}")

        # Save report if requested
        if output:
            output_path = Path(output)
            cli.freshness_tracker.generate_freshness_report_json(output_path)
            console.print(f"\n[dim]📄 Report saved to: {output_path}[/dim]")

    except Exception as e:
        console.print(f"[bold red]❌ Freshness check failed: {e}[/bold red]")
        raise typer.Exit(1)


@app.command()
def stats(
    kb_path: Path = typer.Option(
        Path(".knowledge-base"), help="Path to knowledge base directory"
    )
):
    """📊 Show knowledge base statistics."""
    cli = get_cli(kb_path)

    try:
        console.print(Panel("📊 Knowledge Base Statistics", style="bold blue"))

        # Basic stats
        basic_stats = Table(show_header=False)
        basic_stats.add_column("Metric", style="cyan")
        basic_stats.add_column("Value", style="green")

        basic_stats.add_row("Total entries", str(len(cli.index.entries)))
        basic_stats.add_row("Categories", str(len(cli.index.categories)))
        basic_stats.add_row("Tags", str(len(cli.index.tags)))
        # Get all unique contributors from all entries
        all_contributors = set()
        for entry in cli.index.entries:
            all_contributors.update(entry.contributors)
        basic_stats.add_row("Contributors", str(len(all_contributors)))

        console.print(basic_stats)

        # Category breakdown
        category_counts = {}
        for entry in cli.index.entries:
            category_counts[entry.category] = category_counts.get(entry.category, 0) + 1

        if category_counts:
            console.print("\n[bold]Entries by category:[/bold]")
            category_table = Table(show_header=False)
            category_table.add_column("Category", style="cyan")
            category_table.add_column("Count", style="green")

            for category, count in sorted(category_counts.items()):
                category_table.add_row(category, str(count))

            console.print(category_table)

        # Recent activity
        recent_entries = sorted(
            cli.index.entries, key=lambda e: e.last_updated, reverse=True
        )[:5]
        if recent_entries:
            console.print("\n[bold]Recent updates:[/bold]")
            for entry in recent_entries:
                date_str = entry.last_updated.strftime("%Y-%m-%d")
                console.print(f"  • {entry.title} ([dim]{date_str}[/dim])")

    except Exception as e:
        console.print(f"[bold red]❌ Failed to show stats: {e}[/bold red]")
        raise typer.Exit(1)


def main():
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
