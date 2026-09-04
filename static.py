"""
Static provisioning autoscaling strategy.
"""

from __future__ import annotations


class StaticAutoscaler:
    """
    Keep a fixed number of simulated instances active.
    """

    def __init__(self, instance_count: int) -> None:
        """
        Initialize the static autoscaler.

        Args:
            instance_count:
                Fixed number of instances to return.
        """
        if instance_count <= 0:
            raise ValueError(
                "instance_count must be greater than 0."
            )

        self.instance_count = instance_count

    def decide(
        self,
        current_instances: int,
        incoming_requests: int,
        queue_length: int,
    ) -> int:
        """
        Always return the configured fixed instance count.

        The input parameters are accepted so this autoscaler has a
        consistent public interface with the tests and other strategies.
        Static provisioning intentionally ignores workload changes.
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

        return self.instance_count