"""
Synthetic AI inference workload generator.

This module generates reproducible synthetic workloads for simulation-based
experiments. The generated data does not represent real production traffic.

The workload combines:

- Baseline traffic
- Daily traffic patterns
- Weekly traffic patterns
- Peak-hour traffic
- Random Gaussian noise
- Random traffic spikes
- Sustained sudden workload surges
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import WorkloadConfig
from .patterns import (
    create_daily_pattern,
    create_noise,
    create_peak_hour_pattern,
    create_random_spikes,
    create_sudden_surges,
    create_weekly_pattern,
)


class SyntheticWorkloadGenerator:
    """
    Generate reproducible synthetic AI inference workloads.

    A single workload is created by combining several configurable demand
    components. All generated workloads are synthetic and intended only for
    local simulation and research experiments.

    Args:
        config: Configuration controlling workload generation.
    """

    def __init__(self, config: WorkloadConfig) -> None:
        self.config = config
        self.rng = np.random.default_rng(config.random_seed)

    def _create_timestamps(self) -> pd.DatetimeIndex:
        """
        Create timestamps for the synthetic workload.

        Returns:
            A DatetimeIndex containing one timestamp per simulation time step.
        """
        return pd.date_range(
            start=self.config.start_time,
            periods=self.config.periods,
            freq=self.config.frequency,
        )

    def generate(self) -> pd.DataFrame:
        """
        Generate a complete synthetic workload.

        The final request count at each time step is calculated as:

            workload =
                baseline
                + daily_pattern
                + weekly_pattern
                + peak_hour_pattern
                + random_noise
                + random_spikes
                + sudden_surges

        Negative request values are clipped to zero. Final request counts are
        rounded to whole numbers because requests are discrete events.

        Returns:
            A DataFrame containing timestamps, synthetic request counts, and
            individual workload components.
        """
        timestamps = self._create_timestamps()
        size = len(timestamps)

        baseline = np.full(
            shape=size,
            fill_value=self.config.baseline_requests,
            dtype=float,
        )

        daily_pattern = create_daily_pattern(
            timestamps=timestamps,
            amplitude=self.config.daily_amplitude,
        )

        weekly_pattern = create_weekly_pattern(
            timestamps=timestamps,
            amplitude=self.config.weekly_amplitude,
        )

        peak_pattern = create_peak_hour_pattern(
            timestamps=timestamps,
            baseline_requests=self.config.baseline_requests,
            peak_start_hour=self.config.peak_start_hour,
            peak_end_hour=self.config.peak_end_hour,
            peak_multiplier=self.config.peak_multiplier,
        )

        noise = create_noise(
            size=size,
            noise_std=self.config.noise_std,
            rng=self.rng,
        )

        spikes = create_random_spikes(
            size=size,
            baseline_requests=self.config.baseline_requests,
            spike_probability=self.config.spike_probability,
            spike_multiplier_min=self.config.spike_multiplier_min,
            spike_multiplier_max=self.config.spike_multiplier_max,
            rng=self.rng,
        )

        surges = create_sudden_surges(
            size=size,
            baseline_requests=self.config.baseline_requests,
            surge_probability=self.config.surge_probability,
            surge_duration_steps=self.config.surge_duration_steps,
            surge_multiplier=self.config.surge_multiplier,
            rng=self.rng,
        )

        raw_requests = (
            baseline
            + daily_pattern
            + weekly_pattern
            + peak_pattern
            + noise
            + spikes
            + surges
        )

        requests = np.maximum(raw_requests, 0.0)
        requests = np.rint(requests).astype(int)

        workload = pd.DataFrame(
            {
                "timestamp": timestamps,
                "requests": requests,
                "baseline": baseline,
                "daily_pattern": daily_pattern,
                "weekly_pattern": weekly_pattern,
                "peak_pattern": peak_pattern,
                "noise": noise,
                "spikes": spikes,
                "surges": surges,
            }
        )

        workload["is_spike"] = workload["spikes"] > 0
        workload["is_surge"] = workload["surges"] > 0

        return workload

    def get_metadata(self) -> dict[str, object]:
        """
        Return metadata describing the workload generation configuration.

        The metadata can be saved alongside generated datasets to support
        reproducibility and experiment tracking.

        Returns:
            A dictionary containing workload generation information.
        """
        return {
            "data_type": "synthetic",
            "description": (
                "Synthetic AI inference workload generated for local "
                "simulation and research experiments."
            ),
            "not_real_production_data": True,
            "random_seed": self.config.random_seed,
            "start_time": self.config.start_time,
            "periods": self.config.periods,
            "frequency": self.config.frequency,
            "baseline_requests": self.config.baseline_requests,
            "daily_amplitude": self.config.daily_amplitude,
            "weekly_amplitude": self.config.weekly_amplitude,
            "noise_std": self.config.noise_std,
            "spike_probability": self.config.spike_probability,
            "spike_multiplier_min": self.config.spike_multiplier_min,
            "spike_multiplier_max": self.config.spike_multiplier_max,
            "surge_probability": self.config.surge_probability,
            "surge_duration_steps": self.config.surge_duration_steps,
            "surge_multiplier": self.config.surge_multiplier,
            "peak_start_hour": self.config.peak_start_hour,
            "peak_end_hour": self.config.peak_end_hour,
            "peak_multiplier": self.config.peak_multiplier,
        }