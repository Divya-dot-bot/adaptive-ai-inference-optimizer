"""
Tests for experiment evaluation and comparison utilities.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.evaluation import (
    ExperimentSummary,
    calculate_average_active_instances,
    calculate_average_latency,
    calculate_average_queue_length,
    calculate_average_throughput,
    calculate_cost_per_processed_request,
    calculate_forecasting_errors,
    calculate_max_active_instances,
    calculate_max_queue_length,
    calculate_p95_latency,
    calculate_request_drop_rate,
    calculate_sla_violation_rate,
    calculate_total_cost,
    calculate_total_sla_violations,
    calculate_total_throughput,
    calculate_utilization,
    compare_against_baseline,
    compare_experiments,
    create_experiment_summary,
    rank_strategies,
    summary_to_dataframe,
)


@pytest.fixture
def simulation_results() -> pd.DataFrame:
    """Create reproducible synthetic simulation results for testing."""
    return pd.DataFrame(
        {
            "strategy": ["static"] * 4,
            "incoming_requests": [100, 120, 80, 100],
            "throughput": [90, 110, 80, 100],
            "queue_length": [10, 20, 0, 0],
            "active_instances": [2, 2, 2, 2],
            "estimated_latency_ms": [100.0, 120.0, 80.0, 90.0],
            "estimated_p95_latency_ms": [130.0, 150.0, 100.0, 120.0],
            "infrastructure_cost": [2.0, 2.0, 2.0, 2.0],
            "sla_violations": [5, 10, 0, 0],
        }
    )


def test_calculate_total_cost(
    simulation_results: pd.DataFrame,
) -> None:
    """Total cost should equal the sum of simulation-step costs."""
    assert calculate_total_cost(
        simulation_results
    ) == 8.0


def test_calculate_average_latency(
    simulation_results: pd.DataFrame,
) -> None:
    """Average latency should equal the mean latency."""
    result = calculate_average_latency(
        simulation_results
    )

    assert result == 97.5


def test_calculate_p95_latency(
    simulation_results: pd.DataFrame,
) -> None:
    """p95 latency should be calculated from latency values."""
    result = calculate_p95_latency(
        simulation_results
    )

    assert result > 0


def test_calculate_average_throughput(
    simulation_results: pd.DataFrame,
) -> None:
    """Average throughput should equal the mean throughput."""
    result = calculate_average_throughput(
        simulation_results
    )

    assert result == 95.0


def test_calculate_total_throughput(
    simulation_results: pd.DataFrame,
) -> None:
    """Total throughput should equal the sum of processed requests."""
    assert calculate_total_throughput(
        simulation_results
    ) == 380.0


def test_calculate_total_sla_violations(
    simulation_results: pd.DataFrame,
) -> None:
    """Total SLA violations should equal the column sum."""
    assert calculate_total_sla_violations(
        simulation_results
    ) == 15


def test_calculate_sla_violation_rate(
    simulation_results: pd.DataFrame,
) -> None:
    """SLA violation rate should be between zero and one."""
    result = calculate_sla_violation_rate(
        simulation_results
    )

    assert 0.0 <= result <= 1.0


def test_calculate_average_queue_length(
    simulation_results: pd.DataFrame,
) -> None:
    """Average queue length should equal the mean queue size."""
    assert calculate_average_queue_length(
        simulation_results
    ) == 7.5


def test_calculate_max_queue_length(
    simulation_results: pd.DataFrame,
) -> None:
    """Maximum queue length should equal the largest queue value."""
    assert calculate_max_queue_length(
        simulation_results
    ) == 20


def test_calculate_average_active_instances(
    simulation_results: pd.DataFrame,
) -> None:
    """Average active instances should equal the mean instance count."""
    assert calculate_average_active_instances(
        simulation_results
    ) == 2.0


def test_calculate_max_active_instances(
    simulation_results: pd.DataFrame,
) -> None:
    """Maximum active instances should equal the largest instance count."""
    assert calculate_max_active_instances(
        simulation_results
    ) == 2


def test_calculate_utilization(
    simulation_results: pd.DataFrame,
) -> None:
    """Utilization should be a non-negative ratio."""
    result = calculate_utilization(
        simulation_results
    )

    assert result >= 0.0


def test_calculate_request_drop_rate(
    simulation_results: pd.DataFrame,
) -> None:
    """Unfinished request rate should be a non-negative ratio."""
    result = calculate_request_drop_rate(
        simulation_results
    )

    assert result >= 0.0


def test_calculate_cost_per_processed_request(
    simulation_results: pd.DataFrame,
) -> None:
    """Cost per processed request should be positive."""
    result = calculate_cost_per_processed_request(
        simulation_results
    )

    assert result > 0.0


def test_calculate_forecasting_errors() -> None:
    """Forecasting error calculation should return MAE, RMSE, and MAPE."""
    actual = [100.0, 200.0, 300.0]
    predicted = [110.0, 190.0, 310.0]

    errors = calculate_forecasting_errors(
        actual,
        predicted,
    )

    assert "mae" in errors
    assert "rmse" in errors
    assert "mape" in errors

    assert errors["mae"] >= 0.0
    assert errors["rmse"] >= 0.0


def test_create_experiment_summary(
    simulation_results: pd.DataFrame,
) -> None:
    """A complete experiment summary should be created."""
    summary = create_experiment_summary(
        simulation_results
    )

    assert isinstance(
        summary,
        ExperimentSummary,
    )
    assert summary.strategy == "static"
    assert summary.total_steps == 4
    assert summary.total_cost == 8.0
    assert summary.total_throughput == 380.0


def test_summary_to_dataframe() -> None:
    """An experiment summary should convert to a one-row DataFrame."""
    summary = ExperimentSummary(
        strategy="static",
        total_steps=10,
        total_cost=20.0,
        average_latency_ms=100.0,
        p95_latency_ms=150.0,
        average_throughput=90.0,
        total_throughput=900.0,
        total_sla_violations=5,
        sla_violation_rate=0.01,
        average_queue_length=10.0,
        max_queue_length=30,
        average_active_instances=2.0,
        max_active_instances=3,
        utilization=0.75,
        unfinished_request_rate=0.02,
        cost_per_processed_request=0.02,
    )

    dataframe = summary_to_dataframe(
        summary
    )

    assert len(dataframe) == 1
    assert dataframe.loc[0, "strategy"] == "static"


def test_compare_experiments() -> None:
    """Multiple experiment summaries should form one comparison table."""
    static = ExperimentSummary(
        strategy="static",
        total_steps=10,
        total_cost=20.0,
        average_latency_ms=100.0,
        p95_latency_ms=150.0,
        average_throughput=90.0,
        total_throughput=900.0,
        total_sla_violations=10,
        sla_violation_rate=0.02,
        average_queue_length=15.0,
        max_queue_length=40,
        average_active_instances=2.0,
        max_active_instances=2,
        utilization=0.75,
        unfinished_request_rate=0.01,
        cost_per_processed_request=0.022,
    )

    reactive = ExperimentSummary(
        strategy="reactive",
        total_steps=10,
        total_cost=18.0,
        average_latency_ms=95.0,
        p95_latency_ms=140.0,
        average_throughput=92.0,
        total_throughput=920.0,
        total_sla_violations=7,
        sla_violation_rate=0.015,
        average_queue_length=10.0,
        max_queue_length=30,
        average_active_instances=1.8,
        max_active_instances=3,
        utilization=0.80,
        unfinished_request_rate=0.01,
        cost_per_processed_request=0.019,
    )

    comparison = compare_experiments(
        [static, reactive]
    )

    assert len(comparison) == 2
    assert "strategy" in comparison.columns
    assert "total_cost" in comparison.columns


def test_compare_against_baseline() -> None:
    """Strategies should receive percentage changes versus the baseline."""
    comparison = pd.DataFrame(
        {
            "strategy": ["static", "reactive"],
            "total_cost": [100.0, 80.0],
            "average_latency_ms": [100.0, 90.0],
        }
    )

    result = compare_against_baseline(
        comparison,
        baseline_strategy="static",
        metrics=[
            "total_cost",
            "average_latency_ms",
        ],
    )

    assert (
        "total_cost_change_vs_baseline_pct"
        in result.columns
    )

    reactive_change = result.loc[
        result["strategy"] == "reactive",
        "total_cost_change_vs_baseline_pct",
    ].iloc[0]

    assert reactive_change == -20.0


def test_rank_strategies() -> None:
    """Strategies should be ranked according to the selected metric."""
    comparison = pd.DataFrame(
        {
            "strategy": [
                "static",
                "reactive",
                "forecast",
            ],
            "total_cost": [
                100.0,
                80.0,
                70.0,
            ],
        }
    )

    ranked = rank_strategies(
        comparison,
        metric="total_cost",
        ascending=True,
    )

    assert ranked.iloc[0]["strategy"] == "forecast"
    assert ranked.iloc[0]["rank"] == 1
    assert ranked.iloc[-1]["strategy"] == "static"


def test_compare_experiments_rejects_empty_list() -> None:
    """Comparison should require at least one summary."""
    with pytest.raises(ValueError):
        compare_experiments([])