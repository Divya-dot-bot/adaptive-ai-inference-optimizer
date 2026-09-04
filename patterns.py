"""
Synthetic workload pattern functions.

These functions create components of a synthetic AI inference workload.
The generated patterns are simulation assumptions and do not represent
real production traffic.
"""

import numpy as np
import pandas as pd


def create_daily_pattern(
    timestamps: pd.DatetimeIndex,
    amplitude: float,
) -> np.ndarray:
    """
    Create a smooth daily traffic pattern.

    A sine wave is used to simulate predictable changes in demand across
    a 24-hour period.

    Args:
        timestamps: Timestamps for the workload time series.
        amplitude: Maximum strength of the daily variation.

    Returns:
        A NumPy array containing the daily workload component.
    """
    hours = (
        timestamps.hour
        + timestamps.minute / 60.0
        + timestamps.second / 3600.0
    )

    daily_pattern = amplitude * np.sin(
        2 * np.pi * hours / 24.0
    )

    return daily_pattern


def create_weekly_pattern(
    timestamps: pd.DatetimeIndex,
    amplitude: float,
) -> np.ndarray:
    """
    Create a weekly traffic pattern.

    A sine wave is used to simulate systematic workload differences across
    days of the week.

    Args:
        timestamps: Timestamps for the workload time series.
        amplitude: Maximum strength of the weekly variation.

    Returns:
        A NumPy array containing the weekly workload component.
    """
    day_of_week = (
        timestamps.dayofweek
        + timestamps.hour / 24.0
    )

    weekly_pattern = amplitude * np.sin(
        2 * np.pi * day_of_week / 7.0
    )

    return weekly_pattern


def create_peak_hour_pattern(
    timestamps: pd.DatetimeIndex,
    baseline_requests: float,
    peak_start_hour: int,
    peak_end_hour: int,
    peak_multiplier: float,
) -> np.ndarray:
    """
    Create additional workload during configured daily peak hours.

    The peak pattern adds extra requests above the baseline rather than
    multiplying the entire final workload. This keeps each workload component
    explicit and easier to interpret.

    Args:
        timestamps: Timestamps for the workload time series.
        baseline_requests: Baseline requests per time step.
        peak_start_hour: Inclusive start hour of the peak period.
        peak_end_hour: Exclusive end hour of the peak period.
        peak_multiplier: Multiplier describing traffic during peak hours.

    Returns:
        A NumPy array containing the additional peak-hour workload.
    """
    hours = timestamps.hour

    if peak_start_hour < peak_end_hour:
        is_peak = (
            (hours >= peak_start_hour)
            & (hours < peak_end_hour)
        )
    else:
        # Supports peak periods that cross midnight.
        is_peak = (
            (hours >= peak_start_hour)
            | (hours < peak_end_hour)
        )

    additional_requests = baseline_requests * (peak_multiplier - 1.0)

    peak_pattern = np.where(
        is_peak,
        additional_requests,
        0.0,
    )

    return peak_pattern.astype(float)


def create_noise(
    size: int,
    noise_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Create Gaussian random noise.

    Args:
        size: Number of time steps.
        noise_std: Standard deviation of the noise.
        rng: NumPy random number generator.

    Returns:
        A NumPy array containing random noise.
    """
    return rng.normal(
        loc=0.0,
        scale=noise_std,
        size=size,
    )


def create_random_spikes(
    size: int,
    baseline_requests: float,
    spike_probability: float,
    spike_multiplier_min: float,
    spike_multiplier_max: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Create random short-lived traffic spikes.

    Each time step independently has a configurable probability of containing
    a spike. Spike magnitude is randomly sampled from the configured range.

    Args:
        size: Number of time steps.
        baseline_requests: Baseline requests per time step.
        spike_probability: Probability of a spike at each time step.
        spike_multiplier_min: Minimum spike multiplier.
        spike_multiplier_max: Maximum spike multiplier.
        rng: NumPy random number generator.

    Returns:
        A NumPy array containing additional requests caused by spikes.
    """
    spike_occurs = rng.random(size) < spike_probability

    spike_multipliers = rng.uniform(
        low=spike_multiplier_min,
        high=spike_multiplier_max,
        size=size,
    )

    additional_requests = (
        baseline_requests
        * (spike_multipliers - 1.0)
    )

    spikes = np.where(
        spike_occurs,
        additional_requests,
        0.0,
    )

    return spikes


def create_sudden_surges(
    size: int,
    baseline_requests: float,
    surge_probability: float,
    surge_duration_steps: int,
    surge_multiplier: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Create sustained sudden workload surges.

    At each time step, a surge can begin with the configured probability.
    Once started, the surge adds extra workload for a fixed number of
    consecutive time steps.

    Multiple surges may overlap, in which case their additional workload
    is accumulated.

    Args:
        size: Number of time steps.
        baseline_requests: Baseline requests per time step.
        surge_probability: Probability of starting a surge at each step.
        surge_duration_steps: Number of consecutive time steps for a surge.
        surge_multiplier: Traffic multiplier during a surge.
        rng: NumPy random number generator.

    Returns:
        A NumPy array containing additional workload caused by surges.
    """
    surges = np.zeros(size, dtype=float)

    surge_starts = rng.random(size) < surge_probability

    additional_requests = (
        baseline_requests * (surge_multiplier - 1.0)
    )

    for start_index in np.where(surge_starts)[0]:
        end_index = min(
            start_index + surge_duration_steps,
            size,
        )

        surges[start_index:end_index] += additional_requests

    return surges