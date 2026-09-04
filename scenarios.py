"""
Predefined synthetic workload scenarios.

Each scenario represents a simulated experimental condition. The workloads
generated from these configurations are synthetic and do not represent real
production AI inference traffic.
"""

from dataclasses import replace

from .config import WorkloadConfig


def get_normal_scenario(
    random_seed: int = 42,
) -> WorkloadConfig:
    """
    Create a configuration for relatively stable synthetic traffic.

    The workload contains normal baseline traffic, daily and weekly patterns,
    moderate noise, and a low probability of unexpected spikes or surges.

    Args:
        random_seed: Seed used for reproducible workload generation.

    Returns:
        A WorkloadConfig for the normal traffic scenario.
    """
    return WorkloadConfig(
        random_seed=random_seed,
        baseline_requests=100.0,
        daily_amplitude=30.0,
        weekly_amplitude=15.0,
        noise_std=5.0,
        spike_probability=0.002,
        spike_multiplier_min=1.5,
        spike_multiplier_max=2.0,
        surge_probability=0.0005,
        surge_duration_steps=6,
        surge_multiplier=2.0,
        peak_start_hour=18,
        peak_end_hour=22,
        peak_multiplier=1.3,
    )


def get_predictable_peak_scenario(
    random_seed: int = 42,
) -> WorkloadConfig:
    """
    Create a configuration with strong predictable peak-hour traffic.

    This scenario is useful for testing whether forecast-based autoscaling
    can anticipate recurring demand changes.

    Args:
        random_seed: Seed used for reproducible workload generation.

    Returns:
        A WorkloadConfig for predictable peak traffic.
    """
    return WorkloadConfig(
        random_seed=random_seed,
        baseline_requests=100.0,
        daily_amplitude=50.0,
        weekly_amplitude=20.0,
        noise_std=5.0,
        spike_probability=0.001,
        spike_multiplier_min=1.5,
        spike_multiplier_max=2.0,
        surge_probability=0.0002,
        surge_duration_steps=6,
        surge_multiplier=2.0,
        peak_start_hour=17,
        peak_end_hour=22,
        peak_multiplier=2.0,
    )


def get_random_spikes_scenario(
    random_seed: int = 42,
) -> WorkloadConfig:
    """
    Create a configuration containing frequent unpredictable traffic spikes.

    This scenario is useful for evaluating how reactive and forecast-based
    autoscaling strategies behave when demand contains abrupt short-lived
    changes.

    Args:
        random_seed: Seed used for reproducible workload generation.

    Returns:
        A WorkloadConfig for random spike traffic.
    """
    return WorkloadConfig(
        random_seed=random_seed,
        baseline_requests=100.0,
        daily_amplitude=30.0,
        weekly_amplitude=15.0,
        noise_std=10.0,
        spike_probability=0.03,
        spike_multiplier_min=1.5,
        spike_multiplier_max=3.0,
        surge_probability=0.0005,
        surge_duration_steps=6,
        surge_multiplier=2.0,
        peak_start_hour=18,
        peak_end_hour=22,
        peak_multiplier=1.3,
    )


def get_sudden_surge_scenario(
    random_seed: int = 42,
) -> WorkloadConfig:
    """
    Create a configuration containing sustained sudden workload surges.

    This scenario is useful for testing infrastructure behavior when demand
    changes rapidly and remains elevated for multiple time steps.

    Args:
        random_seed: Seed used for reproducible workload generation.

    Returns:
        A WorkloadConfig for sudden workload surges.
    """
    return WorkloadConfig(
        random_seed=random_seed,
        baseline_requests=100.0,
        daily_amplitude=30.0,
        weekly_amplitude=15.0,
        noise_std=8.0,
        spike_probability=0.005,
        spike_multiplier_min=1.5,
        spike_multiplier_max=2.5,
        surge_probability=0.01,
        surge_duration_steps=12,
        surge_multiplier=3.0,
        peak_start_hour=18,
        peak_end_hour=22,
        peak_multiplier=1.3,
    )


def get_noisy_traffic_scenario(
    random_seed: int = 42,
) -> WorkloadConfig:
    """
    Create a configuration with high random workload variability.

    The scenario preserves predictable seasonal behavior but adds stronger
    random noise.

    Args:
        random_seed: Seed used for reproducible workload generation.

    Returns:
        A WorkloadConfig for noisy traffic.
    """
    return WorkloadConfig(
        random_seed=random_seed,
        baseline_requests=100.0,
        daily_amplitude=35.0,
        weekly_amplitude=15.0,
        noise_std=35.0,
        spike_probability=0.005,
        spike_multiplier_min=1.5,
        spike_multiplier_max=2.5,
        surge_probability=0.001,
        surge_duration_steps=6,
        surge_multiplier=2.0,
        peak_start_hour=18,
        peak_end_hour=22,
        peak_multiplier=1.4,
    )


def get_scenario(
    scenario_name: str,
    random_seed: int = 42,
) -> WorkloadConfig:
    """
    Return a predefined synthetic workload scenario by name.

    Available scenarios:
        - normal
        - predictable_peak
        - random_spikes
        - sudden_surge
        - noisy

    Args:
        scenario_name: Name of the requested scenario.
        random_seed: Seed used for reproducible workload generation.

    Returns:
        A configured WorkloadConfig.

    Raises:
        ValueError: If the scenario name is unknown.
    """
    scenarios = {
        "normal": get_normal_scenario,
        "predictable_peak": get_predictable_peak_scenario,
        "random_spikes": get_random_spikes_scenario,
        "sudden_surge": get_sudden_surge_scenario,
        "noisy": get_noisy_traffic_scenario,
    }

    normalized_name = scenario_name.lower().strip()

    if normalized_name not in scenarios:
        available_scenarios = ", ".join(scenarios.keys())

        raise ValueError(
            f"Unknown scenario '{scenario_name}'. "
            f"Available scenarios: {available_scenarios}."
        )

    config = scenarios[normalized_name](
        random_seed=random_seed,
    )

    return replace(config)