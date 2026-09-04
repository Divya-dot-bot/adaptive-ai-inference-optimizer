"""
Run a fair comparison of autoscaling strategies.

Strategies evaluated:
1. Static provisioning
2. Reactive threshold-based autoscaling
3. Forecast-based autoscaling
4. Optimization-based autoscaling

All strategies are evaluated on the same synthetic workload and using
the same simulation configuration.

IMPORTANT:
All results are generated from synthetic workloads and a local simulator.
They must not be presented as measurements from a real production system.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.autoscaling.base import AutoscalingContext
from src.autoscaling.forecast_based import ForecastAutoscaler
from src.autoscaling.optimization_based import (
    OptimizationBasedAutoscaler,
)
from src.autoscaling.reactive import ReactiveAutoscaler
from src.autoscaling.static import StaticAutoscaler
from src.simulation.config import SimulationConfig
from src.simulation.simulator import InferenceSimulator


PROJECT_ROOT = Path(__file__).resolve().parent

WORKLOAD_PATH = (
    PROJECT_ROOT
    / "data"
    / "generated"
    / "synthetic_workload.csv"
)

RESULTS_DIR = PROJECT_ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
METRICS_DIR = RESULTS_DIR / "metrics"


def load_workload() -> pd.DataFrame:
    """Load the synthetic workload."""

    if not WORKLOAD_PATH.exists():
        raise FileNotFoundError(
            "Synthetic workload file was not found.\n"
            "Run this first:\n"
            "python run_workload_experiment.py"
        )

    workload = pd.read_csv(WORKLOAD_PATH)

    if "requests" not in workload.columns:
        raise ValueError(
            "The workload file must contain a 'requests' column."
        )

    return workload


def create_simulation_config() -> SimulationConfig:
    """Create the common simulation configuration."""

    return SimulationConfig(
        time_step_seconds=60,
        instance_capacity=100,
        initial_instances=2,
        min_instances=1,
        max_instances=10,
        base_latency_ms=50.0,
        queue_latency_factor_ms=2.0,
        sla_latency_threshold_ms=500.0,
        on_demand_cost_per_hour=1.0,
        spot_cost_per_hour=0.4,
        default_instance_type="on_demand",
    )


def create_strategies(
    config: SimulationConfig,
) -> dict[str, Any]:
    """Create all strategies for comparison."""

    return {
        "Static": StaticAutoscaler(
            instance_count=3,
        ),

        "Reactive": ReactiveAutoscaler(
            min_instances=config.min_instances,
            max_instances=config.max_instances,
            scale_up_threshold=0.80,
            scale_down_threshold=0.30,
            scale_up_step=1,
            scale_down_step=1,
        ),

        "Forecast": ForecastAutoscaler(
            target_utilization=0.80,
        ),

        "Optimization": OptimizationBasedAutoscaler(
            min_instances=config.min_instances,
            max_instances=config.max_instances,
            instance_capacity=config.instance_capacity,
            target_utilization=0.80,
            cost_weight=1.0,
            latency_weight=5.0,
            sla_weight=10.0,
            queue_weight=5.0,
        ),
    }


def get_forecast(
    workload: list[int],
    step_index: int,
) -> float:
    """
    Return a simple one-step-ahead forecast.

    For this experiment, the next workload value is used as the
    forecast when available. At the final step, the current workload
    value is used.

    This provides a deterministic oracle-style forecast for comparing
    proactive autoscaling behavior under identical conditions.
    """

    if step_index + 1 < len(workload):
        return float(workload[step_index + 1])

    return float(workload[step_index])


def decide_instances(
    strategy_name: str,
    autoscaler: Any,
    simulator: InferenceSimulator,
    incoming_requests: int,
    forecast_requests: float | None,
    config: SimulationConfig,
) -> int:
    """
    Get the desired instance count for a strategy.

    This function handles the different public interfaces currently
    used by the autoscaling implementations.
    """

    current_instances = simulator.state.active_instances
    queue_length = simulator.state.queue_length

    if strategy_name == "Static":
        target_instances = autoscaler.decide(
            current_instances=current_instances,
            incoming_requests=incoming_requests,
            queue_length=queue_length,
        )

    elif strategy_name == "Reactive":
        target_instances = autoscaler.decide(
            current_instances=current_instances,
            incoming_requests=incoming_requests,
            queue_length=queue_length,
            instance_capacity=config.instance_capacity,
        )

    elif strategy_name == "Forecast":
        context = AutoscalingContext(
            current_instances=current_instances,
            incoming_requests=incoming_requests,
            queue_length=queue_length,
            instance_capacity=config.instance_capacity,
            min_instances=config.min_instances,
            max_instances=config.max_instances,
            forecast_requests=forecast_requests,
        )

        decision = autoscaler.decide(context)

        target_instances = decision.target_instances

    elif strategy_name == "Optimization":
        target_instances = autoscaler.decide(
            current_instances=current_instances,
            incoming_requests=incoming_requests,
            queue_length=queue_length,
        )

    else:
        raise ValueError(
            f"Unsupported strategy: {strategy_name}"
        )

    return max(
        config.min_instances,
        min(int(target_instances), config.max_instances),
    )


def run_single_strategy(
    strategy_name: str,
    autoscaler: Any,
    workload: list[int],
    config: SimulationConfig,
) -> pd.DataFrame:
    """Run one strategy across the complete workload."""

    simulator = InferenceSimulator(config=config)

    records: list[dict[str, Any]] = []

    for step_index, incoming_requests in enumerate(workload):

        forecast_requests: float | None = None

        if strategy_name == "Forecast":
            forecast_requests = get_forecast(
                workload=workload,
                step_index=step_index,
            )

        target_instances = decide_instances(
            strategy_name=strategy_name,
            autoscaler=autoscaler,
            simulator=simulator,
            incoming_requests=int(incoming_requests),
            forecast_requests=forecast_requests,
            config=config,
        )

        # IMPORTANT:
        # Apply the autoscaler's decision BEFORE running this step.
        simulator.set_active_instances(
            target_instances
        )

        step_metrics = simulator.step(
            int(incoming_requests)
        )

        record = {
            "step": step_index,
            "strategy": strategy_name,
            "forecast_requests": forecast_requests,
            "target_instances": target_instances,
            "incoming_requests": step_metrics.incoming_requests,
            "processed_requests": step_metrics.processed_requests,
            "throughput": step_metrics.throughput,
            "queue_length": step_metrics.queue_length,
            "active_instances": step_metrics.active_instances,
            "total_capacity": step_metrics.total_capacity,
            "estimated_latency_ms": (
                step_metrics.estimated_latency_ms
            ),
            "estimated_p95_latency_ms": (
                step_metrics.estimated_p95_latency_ms
            ),
            "infrastructure_cost": (
                step_metrics.infrastructure_cost
            ),
            "sla_violations": step_metrics.sla_violations,
        }

        records.append(record)

    return pd.DataFrame(records)


def summarize_strategy(
    strategy_name: str,
    results: pd.DataFrame,
) -> dict[str, Any]:
    """Calculate exact summary metrics for one strategy."""

    return {
        "strategy": strategy_name,

        "total_cost": float(
            results["infrastructure_cost"].sum()
        ),

        "average_latency_ms": float(
            results["estimated_latency_ms"].mean()
        ),

        "p95_latency_ms": float(
            results["estimated_latency_ms"].quantile(0.95)
        ),

        "average_estimated_p95_latency_ms": float(
            results["estimated_p95_latency_ms"].mean()
        ),

        "total_sla_violations": int(
            results["sla_violations"].sum()
        ),

        "average_throughput": float(
            results["throughput"].mean()
        ),

        "total_processed_requests": int(
            results["processed_requests"].sum()
        ),

        "average_active_instances": float(
            results["active_instances"].mean()
        ),

        "maximum_active_instances": int(
            results["active_instances"].max()
        ),

        "minimum_active_instances": int(
            results["active_instances"].min()
        ),

        "maximum_queue_length": int(
            results["queue_length"].max()
        ),

        "average_queue_length": float(
            results["queue_length"].mean()
        ),
    }


def run_experiment() -> pd.DataFrame:
    """Run all strategies and return the comparison table."""

    print("=" * 70)
    print("ADAPTIVE AI INFERENCE OPTIMIZER")
    print("AUTOSCALING STRATEGY COMPARISON")
    print("=" * 70)

    print("\nLoading synthetic workload...")
    workload_dataframe = load_workload()

    workload = (
        workload_dataframe["requests"]
        .astype(int)
        .tolist()
    )

    print(
        f"Loaded {len(workload)} workload time steps."
    )

    print("\nCreating simulation configuration...")
    config = create_simulation_config()

    print("\nCreating autoscaling strategies...")
    strategies = create_strategies(config)

    comparison_rows: list[dict[str, Any]] = []

    for strategy_name, autoscaler in strategies.items():

        print(
            f"\nRunning {strategy_name} strategy..."
        )

        results = run_single_strategy(
            strategy_name=strategy_name,
            autoscaler=autoscaler,
            workload=workload,
            config=config,
        )

        strategy_results_path = (
            TABLES_DIR
            / f"{strategy_name.lower()}_results.csv"
        )

        strategy_results_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        results.to_csv(
            strategy_results_path,
            index=False,
        )

        summary = summarize_strategy(
            strategy_name=strategy_name,
            results=results,
        )

        comparison_rows.append(summary)

        print(
            f"{strategy_name} completed successfully."
        )

    return pd.DataFrame(comparison_rows)


def save_results(
    comparison: pd.DataFrame,
) -> None:
    """Save the final comparison tables."""

    TABLES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison_path = (
        TABLES_DIR
        / "strategy_comparison.csv"
    )

    metrics_path = (
        METRICS_DIR
        / "strategy_metrics.csv"
    )

    comparison.to_csv(
        comparison_path,
        index=False,
    )

    comparison.to_csv(
        metrics_path,
        index=False,
    )

    print("\nResults saved successfully:")
    print(comparison_path)
    print(metrics_path)


def print_summary(
    comparison: pd.DataFrame,
) -> None:
    """Print the final exact experiment results."""

    print("\n" + "=" * 70)
    print("EXPERIMENT RESULTS")
    print("=" * 70)

    print(
        comparison.to_string(
            index=False
        )
    )

    print("\nIMPORTANT:")
    print(
        "These results were generated using a synthetic workload "
        "and locally simulated AI inference infrastructure."
    )
    print(
        "They are simulation results and must not be represented "
        "as measurements from real production infrastructure."
    )


def main() -> None:
    """Execute the complete experiment."""

    comparison = run_experiment()

    save_results(comparison)

    print_summary(comparison)

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()