"""
Common plotting and visualization utilities for benchmarking results.

This module provides standardized visualization functions for the EfficientAI-MLX-Toolkit,
with support for performance metrics, benchmark comparisons, and result export.

Note: This module requires the 'visualization' optional dependencies.
Install with: uv sync --extra visualization
"""

import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path

# Required visualization dependencies (declared in pyproject.toml)
import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)


class PlottingError(Exception):
    """Raised when plotting operations fail."""

    def __init__(
        self,
        message: str,
        plot_type: str | None = None,
        details: Mapping[str, str | int | float] | None = None,
    ) -> None:
        super().__init__(message)
        self.plot_type = plot_type
        self.details = dict(details or {})


def create_performance_plot(
    benchmark_results: Sequence[Mapping[str, str | float]],
    output_path: Path,
    title: str = "Performance Benchmark Results",
    metric_name: str = "execution_time",
) -> Path:
    """
    Create a performance comparison plot from benchmark results.

    Args:
        benchmark_results: List of benchmark result dictionaries
        output_path: Path to save the plot
        title: Plot title
        metric_name: Name of metric to plot

    Returns:
        Path to saved plot file

    Raises:
        PlottingError: If plotting fails

    Example:
        >>> results = [
        ...     {"name": "baseline", "execution_time": 1.5},
        ...     {"name": "optimized", "execution_time": 0.8}
        ... ]
        >>> create_performance_plot(results, Path("performance.png"))
    """
    try:
        logger.info("Creating performance plot with %d results", len(benchmark_results))

        # Extract data
        names = [str(result.get("name", "Unknown")) for result in benchmark_results]
        values = [float(result.get(metric_name, 0.0)) for result in benchmark_results]

        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))

        bars = ax.bar(names, values, color="skyblue", alpha=0.7)

        # Customize plot
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_ylabel(metric_name.replace("_", " ").title(), fontsize=12)
        ax.set_xlabel("Benchmark", fontsize=12)

        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{value:.3f}",
                ha="center",
                va="bottom",
            )

        # Rotate x-axis labels if needed
        if len(names) > 5:
            plt.xticks(rotation=45, ha="right")

        plt.tight_layout()

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save plot
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info("Performance plot saved to %s", output_path)
        return output_path

    except Exception as e:
        raise PlottingError(
            f"Failed to create performance plot: {e}",
            plot_type="performance",
            details={"metric": metric_name, "num_results": len(benchmark_results)},
        ) from e


def create_comparison_plot(
    baseline_results: Mapping[str, float],
    comparison_results: Sequence[Mapping[str, str | float]],
    output_path: Path,
    title: str = "Benchmark Comparison",
) -> Path:
    """
    Create a comparison plot showing improvements over baseline.

    Args:
        baseline_results: Baseline benchmark results
        comparison_results: List of comparison benchmark results
        output_path: Path to save the plot
        title: Plot title

    Returns:
        Path to saved plot file

    Raises:
        PlottingError: If plotting fails
    """
    try:
        logger.info(
            "Creating comparison plot with baseline and %d comparisons", len(comparison_results)
        )

        # Extract metric names from baseline
        metrics = [
            key for key in baseline_results.keys() if isinstance(baseline_results[key], int | float)
        ]

        if not metrics:
            raise PlottingError("No numeric metrics found in baseline results")

        # Prepare data for plotting
        comparison_names = [str(result.get("name", "Unknown")) for result in comparison_results]

        fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 4 * len(metrics)))
        if len(metrics) == 1:
            axes = [axes]

        for i, metric in enumerate(metrics):
            ax = axes[i]

            baseline_value = float(baseline_results[metric])
            comparison_values = [float(result.get(metric, 0.0)) for result in comparison_results]

            # Calculate improvement percentages
            improvements = [
                ((comp_val - baseline_value) / baseline_value * 100) if baseline_value != 0 else 0
                for comp_val in comparison_values
            ]

            # Create bar plot
            colors = ["green" if imp > 0 else "red" for imp in improvements]
            bars = ax.bar(comparison_names, improvements, color=colors, alpha=0.7)

            # Customize subplot
            ax.set_title(f'{metric.replace("_", " ").title()} Improvement (%)', fontsize=12)
            ax.set_ylabel("Improvement (%)", fontsize=10)
            ax.axhline(y=0, color="black", linestyle="-", alpha=0.3)

            # Add value labels
            for bar, improvement in zip(bars, improvements):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{improvement:.1f}%",
                    ha="center",
                    va="bottom" if height >= 0 else "top",
                )

            # Rotate x-axis labels if needed
            if len(comparison_names) > 3:
                ax.tick_params(axis="x", rotation=45)

        plt.suptitle(title, fontsize=14, fontweight="bold")
        plt.tight_layout()

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save plot
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info("Comparison plot saved to %s", output_path)
        return output_path

    except Exception as e:
        raise PlottingError(
            f"Failed to create comparison plot: {e}",
            plot_type="comparison",
            details={"num_metrics": len(metrics) if "metrics" in locals() else 0},
        ) from e


def create_memory_usage_plot(
    memory_data: Sequence[Mapping[str, str | float]],
    output_path: Path,
    title: str = "Memory Usage Analysis",
) -> Path:
    """
    Create a memory usage visualization plot.

    Args:
        memory_data: List of memory usage data points
        output_path: Path to save the plot
        title: Plot title

    Returns:
        Path to saved plot file

    Raises:
        PlottingError: If plotting fails
    """
    try:
        logger.info("Creating memory usage plot with %d data points", len(memory_data))

        if not memory_data:
            raise PlottingError("No memory data provided")

        # Extract memory metrics
        memory_metrics = set()
        for data_point in memory_data:
            for key in data_point.keys():
                if isinstance(data_point[key], int | float) and "memory" in key.lower():
                    memory_metrics.add(key)

        if not memory_metrics:
            raise PlottingError("No memory metrics found in data")

        # Create subplots for different memory types
        fig, ax = plt.subplots(figsize=(12, 6))

        names = [str(data.get("name", f"Point {i}")) for i, data in enumerate(memory_data)]

        # Plot each memory metric
        x_pos = range(len(names))
        bar_width = 0.8 / len(memory_metrics)

        for i, metric in enumerate(sorted(memory_metrics)):
            values = [float(data.get(metric, 0.0)) for data in memory_data]
            offset = (i - len(memory_metrics) / 2 + 0.5) * bar_width

            ax.bar(
                [x + offset for x in x_pos],
                values,
                bar_width,
                label=metric.replace("_", " ").title(),
                alpha=0.7,
            )

        # Customize plot
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_ylabel("Memory Usage (MB)", fontsize=12)
        ax.set_xlabel("Benchmark", fontsize=12)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.legend()

        plt.tight_layout()

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save plot
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info("Memory usage plot saved to %s", output_path)
        return output_path

    except Exception as e:
        raise PlottingError(
            f"Failed to create memory usage plot: {e}",
            plot_type="memory",
            details={"num_data_points": len(memory_data)},
        ) from e


def save_benchmark_results(
    results: Sequence[Mapping[str, str | float | dict]],
    output_dir: Path,
    formats: Sequence[str] | None = None,
) -> list[Path]:
    """
    Save benchmark results in multiple formats.

    Args:
        results: Benchmark results to save
        output_dir: Directory to save results
        formats: List of formats to save ('json', 'csv', 'html')

    Returns:
        List of paths to saved files

    Raises:
        PlottingError: If saving fails
    """
    if formats is None:
        formats = ["json", "csv"]

    output_dir.mkdir(parents=True, exist_ok=True)
    saved_files: list[Path] = []

    try:
        logger.info("Saving benchmark results in %d formats", len(formats))

        # JSON format
        if "json" in formats:
            json_path = output_dir / "benchmark_results.json"
            json_path.write_text(json.dumps(list(results), indent=2, default=str), encoding="utf-8")
            saved_files.append(json_path)
            logger.debug("Saved JSON results to %s", json_path)

        # CSV format
        if "csv" in formats:
            csv_path = output_dir / "benchmark_results.csv"

            # Flatten nested dictionaries for CSV
            flattened_results = []
            for result in results:
                flat_result = {}
                for key, value in result.items():
                    if isinstance(value, dict):
                        for sub_key, sub_value in value.items():
                            flat_result[f"{key}_{sub_key}"] = sub_value
                    else:
                        flat_result[key] = value
                flattened_results.append(flat_result)

            df = pd.DataFrame(flattened_results)
            df.to_csv(csv_path, index=False)
            saved_files.append(csv_path)
            logger.debug("Saved CSV results to %s", csv_path)

        # HTML format
        if "html" in formats:
            html_path = output_dir / "benchmark_results.html"

            # Create HTML table
            df = pd.DataFrame(results)
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Benchmark Results</title>
                <style>
                    table {{ border-collapse: collapse; width: 100%; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                </style>
            </head>
            <body>
                <h1>Benchmark Results</h1>
                {df.to_html(index=False, escape=False)}
            </body>
            </html>
            """

            html_path.write_text(html_content, encoding="utf-8")
            saved_files.append(html_path)
            logger.debug("Saved HTML results to %s", html_path)

        logger.info("Successfully saved results to %d files", len(saved_files))
        return saved_files

    except Exception as e:
        raise PlottingError(
            f"Failed to save benchmark results: {e}",
            plot_type="export",
            details={"formats": list(formats), "num_results": len(results)},
        ) from e
