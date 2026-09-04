"""
Tests for synthetic workload generation.

All workloads tested in this module are synthetic and are used only for
simulation and research experiments.
"""

import pandas as pd
import pytest

from src.workload import (
    SyntheticWorkloadGenerator,
    WorkloadConfig,
    get_scenario,
)


def test_workload_has_expected_number_of_rows() -> None:
    """Generated workload should contain the configured number of rows."""
    config = WorkloadConfig(
        periods=100,
        random_seed=42,
    )

    generator = SyntheticWorkloadGenerator(config)
    workload = generator.generate()

    assert len(workload) == 100


def test_workload_has_required_columns() -> None:
    """Generated workload should contain all expected columns."""
    config = WorkloadConfig(
        periods=10,
        random_seed=42,
    )

    workload = SyntheticWorkloadGenerator(config).generate()

    expected_columns = {
        "timestamp",
        "requests",
        "baseline",
        "daily_pattern",
        "weekly_pattern",
        "peak_pattern",
        "noise",
        "spikes",
        "surges",
        "is_spike",
        "is_surge",
    }

    assert expected_columns.issubset(workload.columns)


def test_requests_are_non_negative() -> None:
    """Synthetic request counts must never be negative."""
    config = WorkloadConfig(
        periods=500,
        noise_std=1000.0,
        random_seed=42,
    )

    workload = SyntheticWorkloadGenerator(config).generate()

    assert (workload["requests"] >= 0).all()


def test_requests_are_integers() -> None:
    """Final request counts should be discrete integer values."""
    config = WorkloadConfig(
        periods=100,
        random_seed=42,
    )

    workload = SyntheticWorkloadGenerator(config).generate()

    assert pd.api.types.is_integer_dtype(workload["requests"])


def test_same_seed_produces_same_workload() -> None:
    """The same configuration and seed should reproduce the same workload."""
    config = WorkloadConfig(
        periods=200,
        random_seed=42,
    )

    workload_one = SyntheticWorkloadGenerator(config).generate()
    workload_two = SyntheticWorkloadGenerator(config).generate()

    pd.testing.assert_frame_equal(
        workload_one,
        workload_two,
    )


def test_different_seeds_produce_different_workloads() -> None:
    """Different random seeds should normally produce different workloads."""
    config_one = WorkloadConfig(
        periods=200,
        random_seed=42,
    )

    config_two = WorkloadConfig(
        periods=200,
        random_seed=123,
    )

    workload_one = SyntheticWorkloadGenerator(config_one).generate()
    workload_two = SyntheticWorkloadGenerator(config_two).generate()

    assert not workload_one.equals(workload_two)


def test_metadata_marks_data_as_synthetic() -> None:
    """Generated workload metadata must explicitly identify the data as synthetic."""
    config = WorkloadConfig(
        periods=10,
        random_seed=42,
    )

    generator = SyntheticWorkloadGenerator(config)
    metadata = generator.get_metadata()

    assert metadata["data_type"] == "synthetic"
    assert metadata["not_real_production_data"] is True


@pytest.mark.parametrize(
    "scenario_name",
    [
        "normal",
        "predictable_peak",
        "random_spikes",
        "sudden_surge",
        "noisy",
    ],
)
def test_predefined_scenarios_return_valid_config(
    scenario_name: str,
) -> None:
    """Every predefined scenario should return a valid workload configuration."""
    config = get_scenario(
        scenario_name,
        random_seed=42,
    )

    assert isinstance(config, WorkloadConfig)
    assert config.random_seed == 42


def test_unknown_scenario_raises_value_error() -> None:
    """An unknown scenario name should raise a clear error."""
    with pytest.raises(ValueError):
        get_scenario(
            "unknown_scenario",
            random_seed=42,
        )