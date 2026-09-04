"""
Instance-count optimizer for the Adaptive AI Inference Optimizer.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.simulation import (
    SimulationConfig,
    calculate_estimated_latency,
    calculate_estimated_p95_latency,
    calculate_sla_violations,
)

from .objective import (
    ObjectiveResult,
    ObjectiveWeights,
    calculate_objective,
)


@dataclass(frozen=True)
class OptimizationCandidate:
    """Evaluation of one possible instance count."""

    instance_count: int
    estimated_processed_requests: int
    estimated_queue_length: int
    estimated_latency_ms: float
    estimated_p95_latency_ms: float
    estimated_sla_violations: int
    estimated_infrastructure_cost: float
    objective: ObjectiveResult


@dataclass(frozen=True)
class OptimizationResult:
    """Final optimization result."""

    best_candidate: OptimizationCandidate
    candidates: list[OptimizationCandidate]


class InstanceOptimizer:
    """Select the best simulated instance count."""

    def __init__(
        self,
        config: SimulationConfig,
        weights: ObjectiveWeights | None = None,
    ) -> None:
        self.config = config
        self.weights = weights or ObjectiveWeights()

    def _estimate_candidate(
        self,
        instance_count: int,
        expected_requests: int,
        current_queue_length: int,
    ) -> OptimizationCandidate:

        total_work = (
            expected_requests
            + current_queue_length
        )

        total_capacity = (
            instance_count
            * self.config.instance_capacity
        )

        estimated_processed_requests = min(
            total_work,
            total_capacity,
        )

        estimated_queue_length = max(
            0,
            total_work - estimated_processed_requests,
        )

        estimated_latency_ms = calculate_estimated_latency(
            queue_length=estimated_queue_length,
            active_instances=instance_count,
            config=self.config,
        )

        estimated_p95_latency_ms = (
            calculate_estimated_p95_latency(
                estimated_latency_ms=estimated_latency_ms,
                queue_length=estimated_queue_length,
                active_instances=instance_count,
                config=self.config,
            )
        )

        estimated_sla_violations = calculate_sla_violations(
            processed_requests=estimated_processed_requests,
            estimated_latency_ms=estimated_latency_ms,
            estimated_p95_latency_ms=estimated_p95_latency_ms,
            config=self.config,
        )

        estimated_infrastructure_cost = (
            instance_count
            * self.config.get_instance_cost_per_step(
                instance_type=(
                    self.config.default_instance_type
                )
            )
        )

        objective = calculate_objective(
            infrastructure_cost=estimated_infrastructure_cost,
            sla_violations=estimated_sla_violations,
            estimated_latency_ms=estimated_latency_ms,
            sla_latency_threshold_ms=(
                self.config.sla_latency_threshold_ms
            ),
            weights=self.weights,
        )

        return OptimizationCandidate(
            instance_count=instance_count,
            estimated_processed_requests=(
                estimated_processed_requests
            ),
            estimated_queue_length=estimated_queue_length,
            estimated_latency_ms=estimated_latency_ms,
            estimated_p95_latency_ms=(
                estimated_p95_latency_ms
            ),
            estimated_sla_violations=estimated_sla_violations,
            estimated_infrastructure_cost=(
                estimated_infrastructure_cost
            ),
            objective=objective,
        )

    def optimize(
        self,
        expected_requests: int,
        current_queue_length: int = 0,
    ) -> OptimizationResult:

        if expected_requests < 0:
            raise ValueError(
                "expected_requests cannot be negative."
            )

        if current_queue_length < 0:
            raise ValueError(
                "current_queue_length cannot be negative."
            )

        candidates: list[
            OptimizationCandidate
        ] = []

        for instance_count in range(
            self.config.min_instances,
            self.config.max_instances + 1,
        ):
            candidate = self._estimate_candidate(
                instance_count=instance_count,
                expected_requests=expected_requests,
                current_queue_length=current_queue_length,
            )

            candidates.append(candidate)

        best_candidate = min(
            candidates,
            key=lambda candidate: (
                candidate.objective.total_score
            ),
        )

        return OptimizationResult(
            best_candidate=best_candidate,
            candidates=candidates,
        )