"""
Optimization-based autoscaling strategy.
"""

from __future__ import annotations

from src.autoscaling.base import (
    AutoscalingContext,
    BaseAutoscaler,
    ScalingDecision,
)
from src.simulation import SimulationConfig

from .objective import ObjectiveWeights
from .optimizer import (
    InstanceOptimizer,
    OptimizationResult,
)


class OptimizationAutoscaler(BaseAutoscaler):
    """Autoscale by selecting the instance count with the best objective."""

    def __init__(
        self,
        config: SimulationConfig,
        weights: ObjectiveWeights | None = None,
    ) -> None:
        super().__init__(
            strategy_name="optimization"
        )

        self.config = config
        self.weights = weights or ObjectiveWeights()

        self.optimizer = InstanceOptimizer(
            config=self.config,
            weights=self.weights,
        )

        self.last_result: OptimizationResult | None = None

    def decide(
        self,
        context: AutoscalingContext,
    ) -> ScalingDecision:

        if context.forecast_requests is not None:
            expected_requests = max(
                0,
                int(round(context.forecast_requests)),
            )
            workload_source = "provided workload forecast"
        else:
            expected_requests = context.incoming_requests
            workload_source = "latest observed workload fallback"

        result = self.optimizer.optimize(
            expected_requests=expected_requests,
            current_queue_length=context.queue_length,
        )

        self.last_result = result

        best = result.best_candidate

        target_instances = self.clamp_target(
            target_instances=best.instance_count,
            min_instances=context.min_instances,
            max_instances=context.max_instances,
        )

        reason = (
            f"Optimization evaluated {len(result.candidates)} "
            f"candidate instance counts using the "
            f"{workload_source}. "
            f"Expected requests: {expected_requests}. "
            f"Current queue: {context.queue_length}. "
            f"Selected {target_instances} instance(s). "
            f"Objective score: "
            f"{best.objective.total_score:.4f}."
        )

        return ScalingDecision(
            target_instances=target_instances,
            strategy_name=self.strategy_name,
            reason=reason,
        )