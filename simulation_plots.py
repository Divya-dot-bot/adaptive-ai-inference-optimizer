"""
Visualization utilities for local AI inference simulation results.

All plots created by this module represent results from synthetic workloads
and locally simulated infrastructure. They are not real production or cloud
deployment measurements.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _validate_results_dataframe(
    results: pd.DataFrame,
    required_columns: set[str],
) -> None:
    """
    Validate a simulation results DataFrame.

    Args:
        results:
            DataFrame containing locally simulated experiment results.

        required_columns:
            Columns required for a visualization.

    Raises:
        TypeError:
            If results is not a pandas DataFrame.

        ValueError:
            If results is empty or required columns are missing.
    """
    if not isinstance(results, pd.DataFrame):
        raise TypeError(
            "results must be a pandas DataFrame."
        )

    if results.empty:
        raise ValueError(
            "results cannot be empty."
        )

    missing_columns = required_columns.difference(
        results.columns
    )

    if missing_columns:
        missing = ", ".join(
            sorted(missing_columns)
        )

        raise ValueError(
            f"results is missing required column(s): {missing}."
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


def plot_active_instances(
    results: pd.DataFrame,
    time_column: str = "timestamp",
    instance_column: str = "active_instances",
    title: str = "Active Simulated Inference Instances",
    output_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """
    Plot the number of active simulated inference instances over time.
    """
    _validate_results_dataframe(
        results,
        {
            time_column,
            instance_column,
        },
    )

    plt.figure(figsize=(12, 5))

    plt.step(
        results[time_column],
        results[instance_column],
        where="post",
    )

    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Active Simulated Instances")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    _save_figure(output_path)

    if show:
        plt.show()
    else:
        plt.close()


def plot_queue_length(
    results: pd.DataFrame,
    time_column: str = "timestamp",
    queue_column: str = "queue_length",
    title: str = "Simulated Request Queue Length",
    output_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """
    Plot simulated request queue length over time.
    """
    _validate_results_dataframe(
        results,
        {
            time_column,
            queue_column,
        },
    )

    plt.figure(figsize=(12, 5))

    plt.plot(
        results[time_column],
        results[queue_column],
    )

    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Queued Requests")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    _save_figure(output_path)

    if show:
        plt.show()
    else:
        plt.close()


def plot_latency(
    results: pd.DataFrame,
    time_column: str = "timestamp",
    latency_column: str = "estimated_latency_ms",
    p95_column: str = "estimated_p95_latency_ms",
    sla_threshold_ms: float | None = None,
    title: str = "Estimated Simulation Latency",
    output_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """
    Plot estimated average and p95 simulation latency over time.

    Optionally adds a horizontal simulated SLA latency threshold.
    """
    required_columns = {
        time_column,
        latency_column,
    }

    if p95_column in results.columns:
        required_columns.add(p95_column)

    _validate_results_dataframe(
        results,
        required_columns,
    )

    if sla_threshold_ms is not None and sla_threshold_ms <= 0:
        raise ValueError(
            "sla_threshold_ms must be greater than 0."
        )

    plt.figure(figsize=(12, 5))

    plt.plot(
        results[time_column],
        results[latency_column],
        label="Estimated Average Latency",
    )

    if p95_column in results.columns:
        plt.plot(
            results[time_column],
            results[p95_column],
            label="Estimated p95 Latency",
        )

    if sla_threshold_ms is not None:
        plt.axhline(
            y=sla_threshold_ms,
            linestyle="--",
            label="Simulated SLA Threshold",
        )

    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Estimated Latency (ms)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    _save_figure(output_path)

    if show:
        plt.show()
    else:
        plt.close()


def plot_infrastructure_cost(
    results: pd.DataFrame,
    time_column: str = "timestamp",
    cost_column: str = "infrastructure_cost",
    title: str = "Simulated Infrastructure Cost per Step",
    output_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """
    Plot simulated infrastructure cost over time.
    """
    _validate_results_dataframe(
        results,
        {
            time_column,
            cost_column,
        },
    )

    plt.figure(figsize=(12, 5))

    plt.plot(
        results[time_column],
        results[cost_column],
    )

    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Simulated Cost")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    _save_figure(output_path)

    if show:
        plt.show()
    else:
        plt.close()


def plot_cumulative_cost(
    results: pd.DataFrame,
    time_column: str = "timestamp",
    cost_column: str = "infrastructure_cost",
    title: str = "Cumulative Simulated Infrastructure Cost",
    output_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """
    Plot cumulative simulated infrastructure cost over time.
    """
    _validate_results_dataframe(
        results,
        {
            time_column,
            cost_column,
        },
    )

    cumulative_cost = results[cost_column].cumsum()

    plt.figure(figsize=(12, 5))

    plt.plot(
        results[time_column],
        cumulative_cost,
    )

    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Cumulative Simulated Cost")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    _save_figure(output_path)

    if show:
        plt.show()
    else:
        plt.close()


def plot_throughput(
    results: pd.DataFrame,
    time_column: str = "timestamp",
    throughput_column: str = "throughput",
    incoming_column: str = "incoming_requests",
    title: str = "Simulated Incoming Requests and Throughput",
    output_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """
    Compare incoming synthetic workload with simulated throughput.
    """
    _validate_results_dataframe(
        results,
        {
            time_column,
            throughput_column,
            incoming_column,
        },
    )

    plt.figure(figsize=(12, 5))

    plt.plot(
        results[time_column],
        results[incoming_column],
        label="Synthetic Incoming Requests",
    )

    plt.plot(
        results[time_column],
        results[throughput_column],
        label="Simulated Throughput",
    )

    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Requests per Simulation Step")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    _save_figure(output_path)

    if show:
        plt.show()
    else:
        plt.close()


def plot_sla_violations(
    results: pd.DataFrame,
    time_column: str = "timestamp",
    violation_column: str = "sla_violations",
    title: str = "Estimated SLA Violations",
    output_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """
    Plot estimated SLA violations over time.
    """
    _validate_results_dataframe(
        results,
        {
            time_column,
            violation_column,
        },
    )

    plt.figure(figsize=(12, 5))

    plt.plot(
        results[time_column],
        results[violation_column],
    )

    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Estimated SLA Violations")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    _save_figure(output_path)

    if show:
        plt.show()
    else:
        plt.close()