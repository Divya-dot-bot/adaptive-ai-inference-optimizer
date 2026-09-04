"""
Forecast-based autoscaling strategy.
"""

from __future__ import annotations

import math

from .base import (
    AutoscalingContext,
    BaseAutoscaler,
    ScalingDecision,
)


class ForecastAutoscaler(BaseAutoscaler):
    """Proactively scale instances using forecasted workload."""

    def __init__(
        self,
        target_utilization: float = 0.8,
    ) -> None:
        if not 0 < target_utilization <= 1:
            raise ValueError(
                "target_utilization must be greater than 0 "
                "and less than or equal to 1."
            )

        super().__init__(strategy_name="forecast")

        self.target_utilization = target_utilization

    def decide(
        self,
        context: AutoscalingContext,
    ) -> ScalingDecision:

        if context.forecast_requests is None:
            predicted_requests = float(
                context.incoming_requests
            )
            forecast_source = (
                "No forecast was provided; observed workload "
                "was used as a fallback."
            )
        else:
            predicted_requests = float(
                context.forecast_requests
            )
            forecast_source = (
                "The provided workload forecast was used."
            )

        total_expected_work = (
            predicted_requests
            + context.queue_length
        )

        effective_capacity_per_instance = (
            context.instance_capacity
            * self.target_utilization
        )

        required_instances = math.ceil(
            total_expected_work
            / effective_capacity_per_instance
        )

        target_instances = self.clamp_target(
            target_instances=required_instances,
            min_instances=context.min_instances,
            max_instances=context.max_instances,
        )

        reason = (
            f"{forecast_source} "
            f"Predicted requests: {predicted_requests:.2f}. "
            f"Queue length: {context.queue_length}. "
            f"Calculated target: {required_instances}. "
            f"Applied target: {target_instances}."
        )

        return ScalingDecision(
            target_instances=target_instances,
            strategy_name=self.strategy_name,
            reason=reason,
        )