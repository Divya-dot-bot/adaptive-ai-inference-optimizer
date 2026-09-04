"""
Feature engineering utilities for workload forecasting.

This module converts historical synthetic workload data into supervised
machine learning features using lag, rolling-window, and time-based features.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def validate_feature_columns(
    dataframe: pd.DataFrame,
    timestamp_column: str,
    target_column: str,
) -> None:
    """Validate that required columns exist in the input DataFrame."""
    required_columns = {
        timestamp_column,
        target_column,
    }

    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(
            f"Missing required column(s): {missing}"
        )


def create_lag_features(
    dataframe: pd.DataFrame,
    target_column: str,
    lag_steps: Sequence[int],
) -> pd.DataFrame:
    """Create lag features from historical target values."""
    result = dataframe.copy()

    for lag in lag_steps:
        if lag < 1:
            raise ValueError(
                "All lag steps must be greater than or equal to 1."
            )

        result[f"lag_{lag}"] = result[target_column].shift(lag)

    return result


def create_rolling_features(
    dataframe: pd.DataFrame,
    target_column: str,
    windows: Sequence[int],
) -> pd.DataFrame:
    """Create rolling mean and standard deviation features."""
    result = dataframe.copy()
    shifted_target = result[target_column].shift(1)

    for window in windows:
        if window < 1:
            raise ValueError(
                "All rolling windows must be greater than or equal to 1."
            )

        rolling_values = shifted_target.rolling(
            window=window,
            min_periods=window,
        )

        result[f"rolling_mean_{window}"] = rolling_values.mean()
        result[f"rolling_std_{window}"] = rolling_values.std()

    return result


def create_time_features(
    dataframe: pd.DataFrame,
    timestamp_column: str = "timestamp",
    use_hour_feature: bool = True,
    use_day_of_week_feature: bool = True,
    use_is_weekend_feature: bool = True,
) -> pd.DataFrame:
    """Create calendar-based features from timestamps."""

    if timestamp_column not in dataframe.columns:
        raise ValueError(
            f"Timestamp column '{timestamp_column}' does not exist."
        )

    result = dataframe.copy()

    timestamps = pd.to_datetime(
        result[timestamp_column]
    )

    if use_hour_feature:
        result["hour"] = timestamps.dt.hour

        result["hour_sin"] = np.sin(
            2 * np.pi * result["hour"] / 24
        )

        result["hour_cos"] = np.cos(
            2 * np.pi * result["hour"] / 24
        )

    if use_day_of_week_feature:
        result["day_of_week"] = timestamps.dt.dayofweek

    if use_is_weekend_feature:
        result["is_weekend"] = (
            timestamps.dt.dayofweek >= 5
        ).astype(int)

    return result

def create_forecasting_features(
    dataframe: pd.DataFrame,
    timestamp_column: str = "timestamp",
    target_column: str = "requests",
    lag_steps: Sequence[int] = (1, 2, 3),
    rolling_windows: Sequence[int] = (3, 6),
    use_hour_feature: bool = True,
    use_day_of_week_feature: bool = True,
    use_is_weekend_feature: bool = True,
    drop_missing: bool = True,
) -> pd.DataFrame:
    """Create a complete feature set for workload forecasting."""
    validate_feature_columns(
        dataframe=dataframe,
        timestamp_column=timestamp_column,
        target_column=target_column,
    )

    result = dataframe.copy()

    result = create_lag_features(
        dataframe=result,
        target_column=target_column,
        lag_steps=lag_steps,
    )

    result = create_rolling_features(
        dataframe=result,
        target_column=target_column,
        windows=rolling_windows,
    )

    result = create_time_features(
        dataframe=result,
        timestamp_column=timestamp_column,
        use_hour_feature=use_hour_feature,
        use_day_of_week_feature=use_day_of_week_feature,
        use_is_weekend_feature=use_is_weekend_feature,
    )

    if drop_missing:
        result = result.dropna().reset_index(drop=True)

    return result


def get_feature_columns(
    dataframe: pd.DataFrame,
    timestamp_column: str = "timestamp",
    target_column: str = "requests",
    excluded_columns: Sequence[str] = (),
) -> list[str]:
    """Return columns suitable for use as machine learning features."""
    excluded = {
        timestamp_column,
        target_column,
        *excluded_columns,
    }

    return [
        column
        for column in dataframe.columns
        if column not in excluded
    ]


def split_features_and_target(
    dataframe: pd.DataFrame,
    target_column: str = "requests",
    feature_columns: Sequence[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Split a feature DataFrame into X and y arrays."""
    if target_column not in dataframe.columns:
        raise ValueError(
            f"Target column '{target_column}' does not exist."
        )

    if feature_columns is None:
        feature_columns = [
            column
            for column in dataframe.select_dtypes(
                include=[np.number]
            ).columns
            if column != target_column
        ]

    X = dataframe[list(feature_columns)].to_numpy(
        dtype=float
    )

    y = dataframe[target_column].to_numpy(
        dtype=float
    )

    return X, y