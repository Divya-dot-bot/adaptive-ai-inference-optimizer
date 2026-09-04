"""
Visualization utilities for comparing local simulation experiments.

All plots created by this module compare results generated from synthetic
workloads and locally simulated AI inference infrastructure.

They must not be presented as measurements from real production or cloud
infrastructure.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _validate_comparison_dataframe(
    comparison: pd.DataFrame,
    required_columns: set[str],
) -> None:
    """
    Validate an experiment comparison DataFrame.
    """
    if not isinstance(comparison, pd.DataFrame):
        raise TypeError(
            "comparison must be a pandas DataFrame."
        )

    if comparison.empty:
        raise ValueError(
            "comparison cannot be empty."
        )

    missing_columns = required_columns.difference(
        comparison.columns
    )

    if missing_columns:
        missing = ", ".join(
            sorted(missing_columns)
        )

        raise ValueError(
            f"comparison is missing required column(s): {missing}."
        )


def _save_figure(
    output_path: str | Path | None,
) -> None:
    """
    Save the current figure when an output path is provided.
    """
    if output_path is None:
        return

    path = Path(output_path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.savefig(
        path,
        bbox_inches="tight",
        dpi=150,
    )


def plot_strategy_metric(
    comparison: pd.DataFrame,
    metric: str,
    title: str | None = None,
    ylabel: str | None = None,
    output_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """
    Create a bar chart comparing one metric across strategies.

    Args:
        comparison:
            DataFrame containing one row per autoscaling strategy.

        metric:
            Numeric metric column to compare.

        title:
            Optional chart title.

        ylabel:
            Optional y-axis label.

        output_path:
            Optional figure save path.

        show:
            Whether to display the figure.
    """
    _validate_comparison_dataframe(
        comparison,
        {"strategy", metric},
    )

    plot_title = (
        title
        if title is not None
        else f"Autoscaling Strategy Comparison: {metric}"
    )

    y_axis_label = (
        ylabel
        if ylabel is not None
        else metric.replace("_", " ").title()
    )

    plt.figure(figsize=(10, 5))

    plt.bar(
        comparison["strategy"],
        comparison[metric],
    )

    plt.title(plot_title)
    plt.xlabel("Autoscaling Strategy")
    plt.ylabel(y_axis_label)

    plt.grid(
        True,
        axis="y",
        alpha=0.3,
    )

    plt.tight_layout()

    _save_figure(output_path)

    if show:
        plt.show()
    else:
        plt.close()


def plot_core_metrics(
    comparison: pd.DataFrame,
    output_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """
    Create a grouped bar chart for selected core experiment metrics.

    Metrics are normalized relative to the maximum value of each metric so
    that differently scaled measurements can be displayed together.

    This plot is intended for high-level visual comparison only. Exact metric
    values should be inspected in the experiment results table.

    Args:
        comparison:
            DataFrame containing one row per autoscaling strategy.

        output_path:
            Optional figure save path.

        show:
            Whether to display the figure.
    """
    metrics = [
        "total_cost",
        "average_latency_ms",
        "p95_latency_ms",
        "total_sla_violations",
    ]

    _validate_comparison_dataframe(
        comparison,
        {"strategy", *metrics},
    )

    normalized = comparison.copy()

    for metric in metrics:
        maximum = normalized[metric].max()

        if maximum > 0:
            normalized[metric] = (
                normalized[metric] / maximum
            )
        else:
            normalized[metric] = 0.0

    strategies = normalized["strategy"].tolist()
    metric_count = len(metrics)
    strategy_count = len(strategies)

    x_positions = list(range(strategy_count))
    bar_width = 0.8 / metric_count

    plt.figure(figsize=(12, 6))

    for index, metric in enumerate(metrics):
        positions = [
            position
            - 0.4
            + (index * bar_width)
            + (bar_width / 2)
            for position in x_positions
        ]

        plt.bar(
            positions,
            normalized[metric],
            width=bar_width,
            label=metric.replace("_", " ").title(),
        )

    plt.xticks(
        x_positions,
        strategies,
    )

    plt.title(
        "Normalized Core Metrics Across Autoscaling Strategies"
    )
    plt.xlabel("Autoscaling Strategy")
    plt.ylabel("Normalized Metric Value")
    plt.legend()

    plt.grid(
        True,
        axis="y",
        alpha=0.3,
    )

    plt.tight_layout()

    _save_figure(output_path)

    if show:
        plt.show()
    else:
        plt.close()


def plot_cost_vs_latency(
    comparison: pd.DataFrame,
    cost_column: str = "total_cost",
    latency_column: str = "average_latency_ms",
    output_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """
    Plot the simulated cost-latency trade-off across strategies.

    Each point represents one completed local simulation experiment.

    Lower cost and lower latency are generally preferable, but the plot does
    not determine a universal winner because SLA and throughput must also be
    considered.
    """
    _validate_comparison_dataframe(
        comparison,
        {
            "strategy",
            cost_column,
            latency_column,
        },
    )

    plt.figure(figsize=(8, 6))

    plt.scatter(
        comparison[cost_column],
        comparison[latency_column],
    )

    for _, row in comparison.iterrows():
        plt.annotate(
            str(row["strategy"]),
            (
                row[cost_column],
                row[latency_column],
            ),
        )

    plt.title(
        "Simulated Cost vs Estimated Average Latency"
    )
    plt.xlabel("Total Simulated Infrastructure Cost")
    plt.ylabel("Estimated Average Latency (ms)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    _save_figure(output_path)

    if show:
        plt.show()
    else:
        plt.close()


def plot_sla_vs_cost(
    comparison: pd.DataFrame,
    cost_column: str = "total_cost",
    violation_column: str = "total_sla_violations",
    output_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """
    Plot the trade-off between simulated cost and estimated SLA violations.

    Each point represents one locally simulated experiment.
    """
    _validate_comparison_dataframe(
        comparison,
        {
            "strategy",
            cost_column,
            violation_column,
        },
    )

    plt.figure(figsize=(8, 6))

    plt.scatter(
        comparison[cost_column],
        comparison[violation_column],
    )

    for _, row in comparison.iterrows():
        plt.annotate(
            str(row["strategy"]),
            (
                row[cost_column],
                row[violation_column],
            ),
        )

    plt.title(
        "Simulated Cost vs Estimated SLA Violations"
    )
    plt.xlabel("Total Simulated Infrastructure Cost")
    plt.ylabel("Estimated SLA Violations")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    _save_figure(output_path)

    if show:
        plt.show()
    else:
        plt.close()