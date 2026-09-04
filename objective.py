"""
Objective functions for the Adaptive AI Inference Optimizer.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ObjectiveWeights:
    """Weights used to balance the optimization objective."""

    cost_weight: float = 1.0
    sla_violation_weight: float = 10.0
    latency_penalty_weight: float = 0.05

    def __post_init__(self) -> None:
        """Validate objective weights."""
        if self.cost_weight < 0:
            raise ValueError(
                "cost_weight cannot be negative."
            )

        if self.sla_violation_weight < 0:
            raise ValueError(
                "sla_violation_weight cannot be negative."
            )

        if self.latency_penalty_weight < 0:
            raise ValueError(
                "latency_penalty_weight cannot be negative."
            )


@dataclass(frozen=True)
class ObjectiveResult:
    """Store the result of one objective-function evaluation."""

    total_score: float
    cost_component: float
    sla_component: float
    latency_component: float
    excessive_latency_ms: float


def calculate_objective(
    infrastructure_cost: float,
    sla_violations: int,
    estimated_latency_ms: float,
    sla_latency_threshold_ms: float,
    weights: ObjectiveWeights | None = None,
) -> ObjectiveResult:
    """Calculate the optimization objective."""
    if infrastructure_cost < 0:
        raise ValueError(
            "infrastructure_cost cannot be negative."
        )

    if sla_violations < 0:
        raise ValueError(
            "sla_violations cannot be negative."
        )

    if estimated_latency_ms < 0:
        raise ValueError(
            "estimated_latency_ms cannot be negative."
        )

    if sla_latency_threshold_ms <= 0:
        raise ValueError(
            "sla_latency_threshold_ms must be greater than 0."
        )

    selected_weights = weights or ObjectiveWeights()

    excessive_latency_ms = max(
        0.0,
        estimated_latency_ms - sla_latency_threshold_ms,
    )

    cost_component = (
        selected_weights.cost_weight
        * infrastructure_cost
    )

    sla_component = (
        selected_weights.sla_violation_weight
        * sla_violations
    )

    latency_component = (
        selected_weights.latency_penalty_weight
        * excessive_latency_ms
    )

    total_score = (
        cost_component
        + sla_component
        + latency_component
    )

    return ObjectiveResult(
        total_score=total_score,
        cost_component=cost_component,
        sla_component=sla_component,
        latency_component=latency_component,
        excessive_latency_ms=excessive_latency_ms,
    )