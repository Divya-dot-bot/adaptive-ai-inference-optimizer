"""
Autoscaling controller.

This module provides a common controller interface between the different
autoscaling strategies and the local InferenceSimulator.

The project currently contains strategies with different public interfaces:

- StaticAutoscaler.decide(current_instances, incoming_requests, queue_length)
- ReactiveAutoscaler.decide(
      current_instances,
      incoming_requests,
      queue_length,
      instance_capacity
  )
- ForecastAutoscaler.decide(AutoscalingContext)
- OptimizationBasedAutoscaler.decide(
      current_instances,
      incoming_requests,
      queue_length
  )

AutoscalingController normalizes these interfaces and returns a single
integer representing the target number of active instances.
"""

from __future__ import annotations

from typing import Any

from .base import AutoscalingContext, BaseAutoscaler, ScalingDecision


class AutoscalingController:
    """
    Coordinate autoscaling decisions for the InferenceSimulator.

    The controller receives the current simulation state, calls the
    configured autoscaler using its supported interface, and returns
    a validated target instance count.
    """

    def __init__(
        self,
        autoscaler: Any,
        min_instances: int = 1,
        max_instances: int = 10,
        instance_capacity: int = 100,
    ) -> None:
        """
        Initialize the autoscaling controller.

        Args:
            autoscaler:
                An autoscaling strategy object with a decide() method.

            min_instances:
                Minimum allowed number of instances.

            max_instances:
                Maximum allowed number of instances.

            instance_capacity:
                Number of requests one simulated instance can process
                during a simulation step.
        """
        if autoscaler is None:
            raise ValueError(
                "autoscaler cannot be None."
            )

        if not hasattr(autoscaler, "decide"):
            raise TypeError(
                "autoscaler must provide a decide() method."
            )

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

        self.autoscaler = autoscaler
        self.min_instances = min_instances
        self.max_instances = max_instances
        self.instance_capacity = instance_capacity

    def decide(
        self,
        current_instances: int,
        incoming_requests: int,
        queue_length: int,
        forecast_requests: float | None = None,
    ) -> int:
        """
        Calculate the target number of active instances.

        Args:
            current_instances:
                Number of currently active simulated instances.

            incoming_requests:
                Number of requests arriving during the current step.

            queue_length:
                Number of requests already waiting in the queue.

            forecast_requests:
                Optional forecast for future incoming requests.

        Returns:
            Validated target number of active instances.
        """
        self._validate_inputs(
            current_instances=current_instances,
            incoming_requests=incoming_requests,
            queue_length=queue_length,
            forecast_requests=forecast_requests,
        )

        decision = self._call_autoscaler(
            current_instances=current_instances,
            incoming_requests=incoming_requests,
            queue_length=queue_length,
            forecast_requests=forecast_requests,
        )

        target_instances = self._extract_target_instances(
            decision
        )

        return self._clamp_target(
            target_instances
        )

    def _validate_inputs(
        self,
        current_instances: int,
        incoming_requests: int,
        queue_length: int,
        forecast_requests: float | None,
    ) -> None:
        """Validate controller inputs."""
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

        if (
            forecast_requests is not None
            and forecast_requests < 0
        ):
            raise ValueError(
                "forecast_requests cannot be negative."
            )

    def _call_autoscaler(
        self,
        current_instances: int,
        incoming_requests: int,
        queue_length: int,
        forecast_requests: float | None,
    ) -> Any:
        """
        Call the configured autoscaler.

        BaseAutoscaler implementations use AutoscalingContext and return
        ScalingDecision objects. Legacy/simple strategies return an integer.
        """
        if isinstance(
            self.autoscaler,
            BaseAutoscaler,
        ):
            context = AutoscalingContext(
                current_instances=current_instances,
                incoming_requests=incoming_requests,
                queue_length=queue_length,
                instance_capacity=self.instance_capacity,
                min_instances=self.min_instances,
                max_instances=self.max_instances,
                forecast_requests=forecast_requests,
            )

            return self.autoscaler.decide(
                context
            )

        autoscaler_name = (
            self.autoscaler.__class__.__name__
        )

        if autoscaler_name == "ReactiveAutoscaler":
            return self.autoscaler.decide(
                current_instances=current_instances,
                incoming_requests=incoming_requests,
                queue_length=queue_length,
                instance_capacity=self.instance_capacity,
            )

        return self.autoscaler.decide(
            current_instances=current_instances,
            incoming_requests=incoming_requests,
            queue_length=queue_length,
        )

    @staticmethod
    def _extract_target_instances(
        decision: Any,
    ) -> int:
        """
        Extract an integer target instance count from a strategy decision.
        """
        if isinstance(
            decision,
            ScalingDecision,
        ):
            target_instances = (
                decision.target_instances
            )
        elif isinstance(
            decision,
            int,
        ):
            target_instances = decision
        else:
            raise TypeError(
                "Autoscaler decide() must return either an int "
                "or a ScalingDecision."
            )

        if isinstance(
            target_instances,
            bool,
        ):
            raise TypeError(
                "target_instances must be an integer, not a boolean."
            )

        if not isinstance(
            target_instances,
            int,
        ):
            raise TypeError(
                "target_instances must be an integer."
            )

        return target_instances

    def _clamp_target(
        self,
        target_instances: int,
    ) -> int:
        """
        Restrict the target to the configured instance range.
        """
        return max(
            self.min_instances,
            min(
                target_instances,
                self.max_instances,
            ),
        )