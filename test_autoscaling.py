"""
Tests for autoscaling strategies.
"""

from __future__ import annotations

import pytest

from src.autoscaling import (
    AutoscalingContext,
    ForecastAutoscaler,
    ReactiveAutoscaler,
    ScalingDecision,
    StaticAutoscaler,
)
from src.optimization.autoscaler import OptimizationAutoscaler
from src.simulation import SimulationConfig


def make_context(
    *,
    current_instances: int = 2,
    incoming_requests: int = 100,
    queue_length: int = 0,
    instance_capacity: int = 100,
    min_instances: int = 1,
    max_instances: int = 10,
    forecast_requests: float | None = None,
) -> AutoscalingContext:
    """Create a reusable autoscaling context for tests."""

    return AutoscalingContext(
        current_instances=current_instances,
        incoming_requests=incoming_requests,
        queue_length=queue_length,
        instance_capacity=instance_capacity,
        min_instances=min_instances,
        max_instances=max_instances,
        forecast_requests=forecast_requests,
    )


# ============================================================
# Static Autoscaler
# ============================================================


def test_static_autoscaler_returns_fixed_instances() -> None:
    """Static autoscaler should always return its configured value."""

    autoscaler = StaticAutoscaler(
        instance_count=3,
    )

    result = autoscaler.decide(
        current_instances=2,
        incoming_requests=500,
        queue_length=100,
    )

    assert result == 3


def test_static_autoscaler_rejects_non_positive_instance_count() -> None:
    """Static autoscaler should reject zero or negative counts."""

    with pytest.raises(ValueError):
        StaticAutoscaler(
            instance_count=0,
        )

    with pytest.raises(ValueError):
        StaticAutoscaler(
            instance_count=-1,
        )


def test_static_autoscaler_rejects_negative_current_instances() -> None:
    """Static autoscaler should validate current instances."""

    autoscaler = StaticAutoscaler(
        instance_count=3,
    )

    with pytest.raises(ValueError):
        autoscaler.decide(
            current_instances=-1,
            incoming_requests=100,
            queue_length=0,
        )


# ============================================================
# Reactive Autoscaler
# ============================================================


def test_reactive_autoscaler_scales_up() -> None:
    """Reactive autoscaler should scale up under high utilization."""

    autoscaler = ReactiveAutoscaler(
        min_instances=1,
        max_instances=10,
        scale_up_threshold=0.8,
        scale_down_threshold=0.3,
        scale_up_step=1,
        scale_down_step=1,
    )

    result = autoscaler.decide(
        current_instances=2,
        incoming_requests=180,
        queue_length=0,
        instance_capacity=100,
    )

    assert result == 3


def test_reactive_autoscaler_scales_down() -> None:
    """Reactive autoscaler should scale down under low utilization."""

    autoscaler = ReactiveAutoscaler(
        min_instances=1,
        max_instances=10,
        scale_up_threshold=0.8,
        scale_down_threshold=0.3,
        scale_up_step=1,
        scale_down_step=1,
    )

    result = autoscaler.decide(
        current_instances=5,
        incoming_requests=50,
        queue_length=0,
        instance_capacity=100,
    )

    assert result == 4


def test_reactive_autoscaler_respects_min_instances() -> None:
    """Reactive autoscaler should never scale below its minimum."""

    autoscaler = ReactiveAutoscaler(
        min_instances=2,
        max_instances=10,
        scale_up_threshold=0.8,
        scale_down_threshold=0.3,
    )

    result = autoscaler.decide(
        current_instances=2,
        incoming_requests=0,
        queue_length=0,
        instance_capacity=100,
    )

    assert result == 2


def test_reactive_autoscaler_respects_max_instances() -> None:
    """Reactive autoscaler should never exceed its maximum."""

    autoscaler = ReactiveAutoscaler(
        min_instances=1,
        max_instances=5,
        scale_up_threshold=0.8,
        scale_down_threshold=0.3,
        scale_up_step=2,
        scale_down_step=1,
    )

    result = autoscaler.decide(
        current_instances=5,
        incoming_requests=1000,
        queue_length=500,
        instance_capacity=100,
    )

    assert result == 5


# ============================================================
# Forecast Autoscaler
# ============================================================


def test_forecast_autoscaler_scales_for_forecast() -> None:
    """Forecast autoscaler should provision enough capacity."""

    autoscaler = ForecastAutoscaler(
        target_utilization=0.8,
    )

    context = make_context(
        current_instances=1,
        incoming_requests=100,
        queue_length=0,
        instance_capacity=100,
        min_instances=1,
        max_instances=10,
        forecast_requests=240,
    )

    result = autoscaler.decide(context)

    assert isinstance(result, ScalingDecision)
    assert result.strategy_name == "forecast"
    assert result.target_instances == 3


def test_forecast_autoscaler_respects_max_instances() -> None:
    """Forecast autoscaler should respect the configured maximum."""

    autoscaler = ForecastAutoscaler(
        target_utilization=0.8,
    )

    context = make_context(
        min_instances=1,
        max_instances=3,
        instance_capacity=100,
        forecast_requests=1000,
    )

    result = autoscaler.decide(context)

    assert result.target_instances == 3


def test_forecast_autoscaler_uses_observed_workload_without_forecast() -> None:
    """Forecast autoscaler should fall back to observed workload."""

    autoscaler = ForecastAutoscaler(
        target_utilization=0.8,
    )

    context = make_context(
        incoming_requests=160,
        queue_length=0,
        instance_capacity=100,
        min_instances=1,
        max_instances=10,
        forecast_requests=None,
    )

    result = autoscaler.decide(context)

    assert result.target_instances == 2
    assert "fallback" in result.reason.lower()


def test_forecast_autoscaler_rejects_invalid_utilization() -> None:
    """Forecast autoscaler should reject invalid utilization."""

    with pytest.raises(ValueError):
        ForecastAutoscaler(
            target_utilization=0,
        )

    with pytest.raises(ValueError):
        ForecastAutoscaler(
            target_utilization=1.5,
        )


# ============================================================
# Optimization Autoscaler
# ============================================================


def test_optimization_autoscaler_returns_valid_decision() -> None:
    """Optimization autoscaler should return a decision within limits."""

    config = SimulationConfig(
        min_instances=1,
        max_instances=10,
    )

    autoscaler = OptimizationAutoscaler(
        config=config,
    )

    context = make_context(
        current_instances=2,
        incoming_requests=300,
        queue_length=50,
        instance_capacity=100,
        min_instances=1,
        max_instances=10,
        forecast_requests=350,
    )

    result = autoscaler.decide(context)

    assert isinstance(result, ScalingDecision)
    assert result.strategy_name == "optimization"
    assert 1 <= result.target_instances <= 10


def test_optimization_autoscaler_uses_observed_workload_fallback() -> None:
    """Optimization autoscaler should work without a forecast."""

    config = SimulationConfig(
        min_instances=1,
        max_instances=10,
    )

    autoscaler = OptimizationAutoscaler(
        config=config,
    )

    context = make_context(
        incoming_requests=200,
        queue_length=20,
        forecast_requests=None,
    )

    result = autoscaler.decide(context)

    assert isinstance(result, ScalingDecision)
    assert 1 <= result.target_instances <= 10
    assert "fallback" in result.reason.lower()