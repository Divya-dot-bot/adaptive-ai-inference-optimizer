"""
Experiment summary utilities.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from .metrics import (
    calculate_average_active_instances,
    calculate_average_latency,
    calculate_average_queue_length,
    calculate_average_throughput,
    calculate_cost_per_processed_request,
    calculate_max_active_instances,
    calculate_max_queue_length,
    calculate_p95_latency,
    calculate_request_drop_rate,
    calculate_sla_violation_rate,
    calculate_total_cost,
    calculate_total_sla_violations,
    calculate_total_throughput,
    calculate_utilization,
)


@dataclass(frozen=True)
class ExperimentSummary:
    """Summary of one autoscaling simulation experiment."""

    strategy: str
    total_steps: int
    total_cost: float
    average_latency_ms: float
    p95_latency_ms: float
    average_throughput: float
    total_throughput: float
    total_sla_violations: int
    sla_violation_rate: float
    average_queue_length: float
    max_queue_length: int
    average_active_instances: float
    max_active_instances: int
    utilization: float
    unfinished_request_rate: float
    cost_per_processed_request: float

    def to_dict(self) -> dict[str, object]:
        """Convert the experiment summary to a dictionary."""
        return asdict(self)


def create_experiment_summary(
    results: pd.DataFrame,
    strategy_name: str | None = None,
) -> ExperimentSummary:
    """Create a complete summary from one simulation experiment."""
    if not isinstance(results, pd.DataFrame):
        raise TypeError(
            "results must be a pandas DataFrame."
        )

    if results.empty:
        raise ValueError(
            "results cannot be empty."
        )

    if strategy_name is None:
        if "strategy" not in results.columns:
            raise ValueError(
                "strategy_name was not provided and the results "
                "DataFrame does not contain a strategy column."
            )

        strategies = results["strategy"].dropna().unique()

        if len(strategies) == 0:
            raise ValueError(
                "The results DataFrame does not contain a valid strategy."
            )

        if len(strategies) > 1:
            raise ValueError(
                "Multiple strategies found. Provide strategy_name "
                "explicitly when summarizing mixed results."
            )

        resolved_strategy_name = str(strategies[0])

    else:
        if not strategy_name.strip():
            raise ValueError(
                "strategy_name cannot be empty."
            )

        resolved_strategy_name = strategy_name

    return ExperimentSummary(
        strategy=resolved_strategy_name,
        total_steps=len(results),
        total_cost=calculate_total_cost(results),
        average_latency_ms=calculate_average_latency(results),
        p95_latency_ms=calculate_p95_latency(results),
        average_throughput=calculate_average_throughput(results),
        total_throughput=calculate_total_throughput(results),
        total_sla_violations=calculate_total_sla_violations(results),
        sla_violation_rate=calculate_sla_violation_rate(results),
        average_queue_length=calculate_average_queue_length(results),
        max_queue_length=calculate_max_queue_length(results),
        average_active_instances=calculate_average_active_instances(
            results
        ),
        max_active_instances=calculate_max_active_instances(
            results
        ),
        utilization=calculate_utilization(results),
        unfinished_request_rate=calculate_request_drop_rate(
            results
        ),
        cost_per_processed_request=calculate_cost_per_processed_request(
            results
        ),
    )


def summary_to_dataframe(
    summary: ExperimentSummary,
) -> pd.DataFrame:
    """Convert one ExperimentSummary into a one-row DataFrame."""
    return pd.DataFrame(
        [summary.to_dict()]
    )