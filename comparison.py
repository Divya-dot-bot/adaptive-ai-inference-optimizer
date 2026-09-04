"""
Experiment comparison utilities.

This module combines summaries from multiple locally simulated experiments
into comparison tables and calculates relative changes between strategies.

IMPORTANT:
All comparisons are based on synthetic workloads and local simulation
results. They must not be presented as measurements from real production
infrastructure.
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from .summary import ExperimentSummary, summary_to_dataframe

def compare_experiments(
    summaries: Iterable[ExperimentSummary],
) -> pd.DataFrame:
    """
    Combine multiple experiment summaries into one comparison table.
    """
    summary_list = list(summaries)

    if not summary_list:
        raise ValueError(
            "At least one experiment summary is required."
        )

    strategy_names = [
        summary.strategy
        for summary in summary_list
    ]

    if len(strategy_names) != len(set(strategy_names)):
        raise ValueError(
            "Each experiment summary must have a unique strategy name."
        )

    frames = [
        summary_to_dataframe(summary)
        for summary in summary_list
    ]

    comparison = pd.concat(
        frames,
        ignore_index=True,
    )

    return comparison.sort_values(
        by="strategy"
    ).reset_index(drop=True)


def compare_against_baseline(
    comparison: pd.DataFrame,
    baseline_strategy: str,
    metrics: list[str] | None = None,
) -> pd.DataFrame:
    """
    Calculate percentage change for each strategy relative to a baseline.
    """
    if not isinstance(comparison, pd.DataFrame):
        raise TypeError(
            "comparison must be a pandas DataFrame."
        )

    if comparison.empty:
        raise ValueError(
            "comparison cannot be empty."
        )

    if "strategy" not in comparison.columns:
        raise ValueError(
            "comparison must contain a strategy column."
        )

    if baseline_strategy not in set(comparison["strategy"]):
        raise ValueError(
            f"Baseline strategy '{baseline_strategy}' "
            "was not found in comparison."
        )

    default_metrics = [
        "total_cost",
        "average_latency_ms",
        "p95_latency_ms",
        "average_throughput",
        "total_sla_violations",
        "sla_violation_rate",
        "average_queue_length",
        "unfinished_request_rate",
        "cost_per_processed_request",
    ]

    selected_metrics = (
        metrics
        if metrics is not None
        else default_metrics
    )

    missing_metrics = [
        metric
        for metric in selected_metrics
        if metric not in comparison.columns
    ]

    if missing_metrics:
        missing = ", ".join(missing_metrics)

        raise ValueError(
            "comparison is missing requested metric(s): "
            f"{missing}."
        )

    baseline_row = comparison.loc[
        comparison["strategy"] == baseline_strategy
    ].iloc[0]

    result = comparison.copy()

    for metric in selected_metrics:
        baseline_value = baseline_row[metric]

        column_name = (
            f"{metric}_change_vs_baseline_pct"
        )

        if pd.isna(baseline_value) or baseline_value == 0:
            result[column_name] = float("nan")
        else:
            result[column_name] = (
                (result[metric] - baseline_value)
                / baseline_value
                * 100
            )

    return result


def rank_strategies(
    comparison: pd.DataFrame,
    metric: str,
    ascending: bool = True,
) -> pd.DataFrame:
    """
    Rank autoscaling strategies by one metric.
    """
    if not isinstance(comparison, pd.DataFrame):
        raise TypeError(
            "comparison must be a pandas DataFrame."
        )

    if comparison.empty:
        raise ValueError(
            "comparison cannot be empty."
        )

    if metric not in comparison.columns:
        raise ValueError(
            f"Metric '{metric}' was not found in comparison."
        )

    result = comparison.copy()

    result["rank"] = result[metric].rank(
        method="min",
        ascending=ascending,
    ).astype(int)

    return result.sort_values(
        by=[metric, "strategy"],
        ascending=[ascending, True],
    ).reset_index(drop=True)