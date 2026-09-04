"""
Visualization utilities for synthetically generated AI inference workloads.

All plots created by this module represent synthetic workload data generated
by the project. They must not be interpreted as real production traffic.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _validate_workload_dataframe(
    workload: pd.DataFrame,
    required_columns: set[str],
) -> None:
    """
    Validate a synthetic workload DataFrame.

    Args:
        workload:
            DataFrame containing synthetically generated workload data.

        required_columns:
            Columns required by the visualization.

    Raises:
        TypeError:
            If workload is not a pandas DataFrame.

        ValueError:
            If workload is empty or required columns are missing.
    """
    if not isinstance(workload, pd.DataFrame):
        raise TypeError(
            "workload must be a pandas DataFrame."
        )

    if workload.empty:
        raise ValueError(
            "workload cannot be empty."
        )

    missing_columns = required_columns.difference(
        workload.columns
    )

    if missing_columns:
        missing = ", ".join(
            sorted(missing_columns)
        )

        raise ValueError(
            f"workload is missing required column(s): {missing}."
        )


def _save_figure(
    output_path: str | Path | None,
) -> None:
    """
    Save the current Matplotlib figure when an output path is provided.

    Parent directories are created automatically.

    Args:
        output_path:
            Optional destination path for the figure.
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


def plot_workload(
    workload: pd.DataFrame,
    time_column: str = "timestamp",
    request_column: str = "requests",
    title: str = "Synthetic AI Inference Workload",
    output_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """
    Plot synthetic incoming request volume over time.
    """
    _validate_workload_dataframe(
        workload,
        {
            time_column,
            request_column,
        },
    )

    plt.figure(figsize=(12, 5))

    plt.plot(
        workload[time_column],
        workload[request_column],
    )

    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Synthetic Incoming Requests")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    _save_figure(output_path)

    if show:
        plt.show()
    else:
        plt.close()


def plot_workload_components(
    workload: pd.DataFrame,
    time_column: str = "timestamp",
    component_columns: list[str] | None = None,
    title: str = "Synthetic Workload Components",
    output_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """
    Plot individual components used to construct synthetic workload traffic.
    """
    _validate_workload_dataframe(
        workload,
        {time_column},
    )

    if component_columns is None:
        possible_columns = [
            "baseline",
            "daily_pattern",
            "weekly_pattern",
            "peak_pattern",
            "noise",
            "spike",
            "surge",
            "intensity_multiplier",
        ]

        selected_columns = [
            column
            for column in possible_columns
            if column in workload.columns
        ]
    else:
        selected_columns = component_columns

    if not selected_columns:
        raise ValueError(
            "No workload component columns were selected."
        )

    missing_columns = set(
        selected_columns
    ).difference(workload.columns)

    if missing_columns:
        missing = ", ".join(
            sorted(missing_columns)
        )

        raise ValueError(
            f"workload is missing component column(s): {missing}."
        )

    plt.figure(figsize=(12, 6))

    for column in selected_columns:
        plt.plot(
            workload[time_column],
            workload[column],
            label=column.replace("_", " ").title(),
        )

    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Synthetic Component Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    _save_figure(output_path)

    if show:
        plt.show()
    else:
        plt.close()


def plot_workload_distribution(
    workload: pd.DataFrame,
    request_column: str = "requests",
    bins: int = 30,
    title: str = "Distribution of Synthetic Request Volume",
    output_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """
    Plot the distribution of synthetic incoming request counts.
    """
    if bins < 1:
        raise ValueError(
            "bins must be at least 1."
        )

    _validate_workload_dataframe(
        workload,
        {request_column},
    )

    plt.figure(figsize=(8, 5))

    plt.hist(
        workload[request_column],
        bins=bins,
    )

    plt.title(title)
    plt.xlabel("Synthetic Incoming Requests")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    _save_figure(output_path)

    if show:
        plt.show()
    else:
        plt.close()


def plot_workload_by_hour(
    workload: pd.DataFrame,
    time_column: str = "timestamp",
    request_column: str = "requests",
    title: str = "Average Synthetic Workload by Hour",
    output_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """
    Plot average synthetic request volume for each hour of the day.
    """
    _validate_workload_dataframe(
        workload,
        {
            time_column,
            request_column,
        },
    )

    timestamps = pd.to_datetime(
        workload[time_column]
    )

    hourly_data = pd.DataFrame(
        {
            "hour": timestamps.dt.hour,
            "requests": workload[request_column],
        }
    )

    hourly_average = (
        hourly_data.groupby("hour")["requests"]
        .mean()
        .reindex(range(24), fill_value=0.0)
    )

    plt.figure(figsize=(10, 5))

    plt.plot(
        hourly_average.index,
        hourly_average.values,
        marker="o",
    )

    plt.title(title)
    plt.xlabel("Hour of Day")
    plt.ylabel("Average Synthetic Requests")
    plt.xticks(range(24))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    _save_figure(output_path)

    if show:
        plt.show()
    else:
        plt.close()