"""
Tests for the local inference infrastructure simulator.
"""

from __future__ import annotations

import pytest

from src.simulation import (
InstancePool,
InstanceType,
SimulationConfig,
SimulationState,
)

def test_simulation_config_has_valid_defaults() -> None:
 """Default simulation configuration should be valid."""
config = SimulationConfig()

assert config.initial_instances >= config.min_instances
assert config.max_instances >= config.initial_instances
assert config.instance_capacity > 0
assert config.step_duration_seconds > 0


def test_instance_type_creation() -> None:
 """An instance type should store capacity and cost information."""
instance = InstanceType(
name="test_instance",
capacity=100,
cost_per_step=0.5,
)


assert instance.name == "test_instance"
assert instance.capacity == 100
assert instance.cost_per_step == 0.5


def test_instance_type_rejects_invalid_capacity() -> None:
 """Instance capacity must be positive."""
with pytest.raises(ValueError):
 InstanceType(
name="invalid",
capacity=0,
cost_per_step=0.5,
)

def test_instance_type_rejects_negative_cost() -> None:
 """Instance cost must not be negative."""
with pytest.raises(ValueError):
 InstanceType(
name="invalid",
capacity=100,
cost_per_step=-1.0,
)

def test_instance_pool_capacity() -> None:
 """Instance pool capacity should scale with active instances."""
instance = InstanceType(
name="test_instance",
capacity=50,
cost_per_step=1.0,
)


pool = InstancePool(
    instance_type=instance,
    active_instances=4,
)

assert pool.total_capacity == 200


def test_instance_pool_cost() -> None:
 """Instance pool cost should scale with active instances."""
instance = InstanceType(
name="test_instance",
capacity=50,
cost_per_step=2.0,
)


pool = InstancePool(
    instance_type=instance,
    active_instances=3,
)

assert pool.calculate_cost() == 6.0


def test_instance_pool_rejects_negative_instances() -> None:
 """Active instance count must not be negative."""
instance = InstanceType(
name="test_instance",
capacity=50,
cost_per_step=1.0,
)


with pytest.raises(ValueError):
    InstancePool(
        instance_type=instance,
        active_instances=-1,
    )


def test_simulation_state_initial_values() -> None:
 """A new simulation state should have expected initial values."""
state = SimulationState(
active_instances=2
)

assert state.active_instances == 2
assert state.queue_length == 0
assert state.total_processed_requests == 0
assert state.total_sla_violations == 0
assert state.total_cost == 0.0


def test_simulation_state_rejects_negative_instances() -> None:
 """Simulation state must reject negative active instances."""
with pytest.raises(ValueError):
 SimulationState(
active_instances=-1
)
