"""
Reactive threshold-based autoscaling strategy.

This strategy adjusts the number of simulated inference instances according
to current workload utilization.
"""

from __future__ import annotations


class ReactiveAutoscaler:
    """
    Scale simulated instances up or down based on current utilization.
    """

    def __init__(
        self,
        min_instances: int,
        max_instances: int,
        scale_up_threshold: float,
        scale_down_threshold: float,
        scale_up_step: int = 1,
        scale_down_step: int = 1,
    ) -> None:

        if min_instances < 0:
            raise ValueError(
                "min_instances cannot be negative."
            )

        if max_instances < min_instances:
            raise ValueError(
                "max_instances must be greater than or equal to "
                "min_instances."
            )

        if not 0 <= scale_down_threshold <= 1:
            raise ValueError(
                "scale_down_threshold must be between 0 and 1."
            )

        if not 0 <= scale_up_threshold <= 1:
            raise ValueError(
                "scale_up_threshold must be between 0 and 1."
            )

        if scale_down_threshold >= scale_up_threshold:
            raise ValueError(
                "scale_down_threshold must be less than "
                "scale_up_threshold."
            )

        if scale_up_step <= 0:
            raise ValueError(
                "scale_up_step must be greater than 0."
            )

        if scale_down_step <= 0:
            raise ValueError(
                "scale_down_step must be greater than 0."
            )

        self.min_instances = min_instances
        self.max_instances = max_instances
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.scale_up_step = scale_up_step
        self.scale_down_step = scale_down_step

    def decide(
        self,
        current_instances: int,
        incoming_requests: int,
        queue_length: int,
        instance_capacity: int,
    ) -> int:
        """
        Return the desired number of active instances.
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

        if instance_capacity <= 0:
            raise ValueError(
                "instance_capacity must be greater than 0."
            )

        current_capacity = (
            current_instances * instance_capacity
        )

        total_demand = (
            incoming_requests + queue_length
        )

        if current_capacity == 0:
            if total_demand > 0:
                return min(
                    self.min_instances + self.scale_up_step,
                    self.max_instances,
                )
            return self.min_instances

        utilization = (
            total_demand / current_capacity
        )

        if utilization >= self.scale_up_threshold:
            return min(
                current_instances + self.scale_up_step,
                self.max_instances,
            )

        if utilization <= self.scale_down_threshold:
            return max(
                current_instances - self.scale_down_step,
                self.min_instances,
            )

        return max(
            self.min_instances,
            min(current_instances, self.max_instances),
        )