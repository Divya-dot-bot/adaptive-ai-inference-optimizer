"""
Optimization-based autoscaling strategy.

This strategy evaluates possible instance counts and selects the count
with the lowest estimated objective score based on cost, latency,
queue pressure, and SLA violations.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OptimizationDecision:
    """Stores the result of an optimization decision."""

    instance_count: int
    objective_score: float


class OptimizationBasedAutoscaler:
    """
    Select the number of instances that minimizes an estimated objective.

    The objective combines simulated infrastructure cost, workload
    pressure, estimated latency, and SLA violations.
    """

    def __init__(
        self,
        min_instances: int = 1,
        max_instances: int = 10,
        instance_capacity: int = 100,
        target_utilization: float = 0.8,
        cost_weight: float = 1.0,
        latency_weight: float = 1.0,
        sla_weight: float = 2.0,
        queue_weight: float = 1.0,
    ) -> None:
        """Initialize the optimization-based autoscaler."""

        if min_instances < 0:
            raise ValueError(
                "min_instances cannot be negative."
            )

        if max_instances < min_instances:
            raise ValueError(
                "max_instances must be greater than or equal to "
                "min_instances."
            )

        if instance_capacity <= 0:
            raise ValueError(
                "instance_capacity must be greater than 0."
            )

        if not 0 < target_utilization <= 1:
            raise ValueError(
                "target_utilization must be between 0 and 1."
            )

        if cost_weight < 0:
            raise ValueError(
                "cost_weight cannot be negative."
            )

        if latency_weight < 0:
            raise ValueError(
                "latency_weight cannot be negative."
            )

        if sla_weight < 0:
            raise ValueError(
                "sla_weight cannot be negative."
            )

        if queue_weight < 0:
            raise ValueError(
                "queue_weight cannot be negative."
            )

        self.min_instances = min_instances
        self.max_instances = max_instances
        self.instance_capacity = instance_capacity
        self.target_utilization = target_utilization
        self.cost_weight = cost_weight
        self.latency_weight = latency_weight
        self.sla_weight = sla_weight
        self.queue_weight = queue_weight

    def decide(
        self,
        current_instances: int,
        incoming_requests: int,
        queue_length: int,
    ) -> int:
        """
        Return the instance count with the lowest estimated objective score.
        """

        if current_instances < 0:
            raise ValueError(
                "current_instances cannot be negative."
            )

        if incoming_requests < 0:
            raise ValueError(
                "incoming_requests cannot be negative."
            )

        if queue_length < 0:
            raise ValueError(
                "queue_length cannot be negative."
            )

        best_decision: OptimizationDecision | None = None

        total_demand = incoming_requests + queue_length

        for instance_count in range(
            self.min_instances,
            self.max_instances + 1,
        ):
            score = self._calculate_objective(
                instance_count=instance_count,
                total_demand=total_demand,
            )

            decision = OptimizationDecision(
                instance_count=instance_count,
                objective_score=score,
            )

            if (
                best_decision is None
                or decision.objective_score
                < best_decision.objective_score
            ):
                best_decision = decision

        if best_decision is None:
            return self.min_instances

        return best_decision.instance_count

    def _calculate_objective(
        self,
        instance_count: int,
        total_demand: int,
    ) -> float:
        """
        Calculate the estimated objective score for an instance count.
        """

        capacity = instance_count * self.instance_capacity

        utilization = total_demand / max(capacity, 1)

        infrastructure_cost = float(instance_count)

        utilization_error = abs(
            utilization - self.target_utilization
        )

        overloaded_requests = max(
            0,
            total_demand - capacity,
        )

        estimated_queue_pressure = (
            overloaded_requests / self.instance_capacity
        )

        estimated_latency_penalty = (
            utilization_error
            if utilization <= 1
            else utilization ** 2
        )

        estimated_sla_violations = (
            overloaded_requests / max(total_demand, 1)
        )

        objective = (
            self.cost_weight * infrastructure_cost
            + self.latency_weight * estimated_latency_penalty
            + self.sla_weight * estimated_sla_violations
            + self.queue_weight * estimated_queue_pressure
        )

        return float(objective)