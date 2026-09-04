"""
Tests for the optimization module.
"""

from __future__ import annotations

import pytest

from src.optimization import (
InstanceOptimizer,
ObjectiveWeights,
calculate_objective,
)
from src.simulation import SimulationConfig

def make_config(
min_instances: int = 1,
max_instances: int = 10,
instance_capacity: int = 100,
) -> SimulationConfig:
 """Create a small simulation configuration for optimization tests."""
 return SimulationConfig(
min_instances=min_instances,
max_instances=max_instances,
instance_capacity=instance_capacity,
initial_instances=min_instances,
)

def test_calculate_objective_returns_non_negative_score() -> None:
 """The optimization objective should return a non-negative total score."""
result = calculate_objective(
infrastructure_cost=3.0,
sla_violations=0,
estimated_latency_ms=100.0,
sla_latency_threshold_ms=500.0,
)

assert result.total_score >= 0.0

def test_higher_infrastructure_cost_increases_objective() -> None:
 """Higher cost should produce a higher objective when other factors match."""
low_cost = calculate_objective(
infrastructure_cost=2.0,
sla_violations=0,
estimated_latency_ms=100.0,
sla_latency_threshold_ms=500.0,
)

high_cost = calculate_objective(
    infrastructure_cost=4.0,
    sla_violations=0,
    estimated_latency_ms=100.0,
    sla_latency_threshold_ms=500.0,
)

assert high_cost.total_score > low_cost.total_score

def test_sla_violations_increase_objective() -> None:
 """SLA violations should increase the optimization score."""
no_violations = calculate_objective(
infrastructure_cost=2.0,
sla_violations=0,
estimated_latency_ms=100.0,
sla_latency_threshold_ms=500.0,
)

violations = calculate_objective(
    infrastructure_cost=2.0,
    sla_violations=10,
    estimated_latency_ms=100.0,
    sla_latency_threshold_ms=500.0,
)

assert violations.total_score > no_violations.total_score

def test_excessive_latency_increases_objective() -> None:
 """Latency above the SLA threshold should increase the objective."""
acceptable_latency = calculate_objective(
infrastructure_cost=2.0,
sla_violations=0,
estimated_latency_ms=100.0,
sla_latency_threshold_ms=500.0,
)

excessive_latency = calculate_objective(
    infrastructure_cost=2.0,
    sla_violations=0,
    estimated_latency_ms=1000.0,
    sla_latency_threshold_ms=500.0,
)

assert excessive_latency.total_score > acceptable_latency.total_score

def test_optimizer_returns_candidate_within_allowed_range() -> None:
 """The optimizer should select a count within configured limits."""
config = make_config(
min_instances=1,
max_instances=10,
)

optimizer = InstanceOptimizer(config=config)

result = optimizer.optimize(
    expected_requests=250,
)

assert (
    config.min_instances
    <= result.best_candidate.instance_count
    <= config.max_instances
)

def test_optimizer_evaluates_every_allowed_instance_count() -> None:
 """The optimizer should evaluate every integer in the configured range."""
config = make_config(
min_instances=2,
max_instances=5,
)

optimizer = InstanceOptimizer(config=config)

result = optimizer.optimize(
    expected_requests=250,
)

assert len(result.candidates) == 4

counts = [
    candidate.instance_count
    for candidate in result.candidates
]

assert counts == [2, 3, 4, 5]

def test_optimizer_selects_minimum_for_zero_demand() -> None:
 """With zero demand, the cheapest valid candidate should be preferred."""
config = make_config(
min_instances=2,
max_instances=10,
)

optimizer = InstanceOptimizer(config=config)

result = optimizer.optimize(
    expected_requests=0,
)

assert result.best_candidate.instance_count == 2

def test_optimizer_respects_maximum_limit() -> None:
 """Even extreme demand should not produce a candidate above the maximum."""
config = make_config(
min_instances=1,
max_instances=5,
)

optimizer = InstanceOptimizer(config=config)

result = optimizer.optimize(
    expected_requests=10_000,
)

assert result.best_candidate.instance_count <= 5
assert max(
    candidate.instance_count
    for candidate in result.candidates
) == 5

def test_optimizer_handles_existing_queue() -> None:
 """Requests already queued should be included in optimization."""
config = make_config(
min_instances=1,
max_instances=10,
)

optimizer = InstanceOptimizer(config=config)

result = optimizer.optimize(
    expected_requests=100,
    current_queue_length=200,
)

best = result.best_candidate

assert 1 <= best.instance_count <= 10
assert best.estimated_processed_requests >= 0
assert best.estimated_queue_length >= 0

def test_optimizer_rejects_negative_expected_requests() -> None:
 """Negative expected workload should be rejected."""
config = make_config()

optimizer = InstanceOptimizer(config=config)

with pytest.raises(ValueError):
    optimizer.optimize(
        expected_requests=-1,
    )

def test_optimizer_rejects_negative_queue_length() -> None:
 """Negative queue length should be rejected."""
config = make_config()

optimizer = InstanceOptimizer(config=config)

with pytest.raises(ValueError):
    optimizer.optimize(
        expected_requests=100,
        current_queue_length=-1,
    )

def test_custom_weights_can_be_used() -> None:
 """The optimizer should accept custom objective weights."""
config = make_config()


weights = ObjectiveWeights()

optimizer = InstanceOptimizer(
    config=config,
    weights=weights,
)

result = optimizer.optimize(
    expected_requests=250,
)

assert result.best_candidate is not None
assert len(result.candidates) > 0
