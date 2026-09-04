"""
Evaluation metrics for workload forecasting.

These metrics are used to evaluate forecasts generated from synthetic
AI inference workload data. Results are experimental measurements from
local simulations and must not be presented as real production results.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def _validate_inputs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Validate and convert forecasting metric inputs.
    """
    true_values = np.asarray(y_true, dtype=float)
    predicted_values = np.asarray(y_pred, dtype=float)

    if true_values.ndim != 1:
        raise ValueError(
            "y_true must be a one-dimensional array."
        )

    if predicted_values.ndim != 1:
        raise ValueError(
            "y_pred must be a one-dimensional array."
        )

    if len(true_values) == 0:
        raise ValueError(
            "Metric inputs cannot be empty."
        )

    if len(true_values) != len(predicted_values):
        raise ValueError(
            "y_true and y_pred must have the same number of observations."
        )

    return true_values, predicted_values


def calculate_mae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    Calculate Mean Absolute Error.
    """
    true_values, predicted_values = _validate_inputs(
        y_true,
        y_pred,
    )

    return float(
        mean_absolute_error(
            true_values,
            predicted_values,
        )
    )


def calculate_rmse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    Calculate Root Mean Squared Error.
    """
    true_values, predicted_values = _validate_inputs(
        y_true,
        y_pred,
    )

    mse = mean_squared_error(
        true_values,
        predicted_values,
    )

    return float(np.sqrt(mse))


def calculate_mape(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    epsilon: float = 1e-8,
) -> float:
    """
    Calculate Mean Absolute Percentage Error.
    """
    if epsilon <= 0:
        raise ValueError(
            "epsilon must be greater than 0."
        )

    true_values, predicted_values = _validate_inputs(
        y_true,
        y_pred,
    )

    non_zero_mask = np.abs(true_values) > epsilon

    if not np.any(non_zero_mask):
        raise ValueError(
            "MAPE cannot be calculated because all actual values are zero."
        )

    percentage_errors = np.abs(
        (
            true_values[non_zero_mask]
            - predicted_values[non_zero_mask]
        )
        / true_values[non_zero_mask]
    )

    return float(
        np.mean(percentage_errors) * 100.0
    )


def evaluate_forecast(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """
    Calculate MAE, RMSE, and MAPE.
    """
    return {
        "mae": calculate_mae(y_true, y_pred),
        "rmse": calculate_rmse(y_true, y_pred),
        "mape": calculate_mape(y_true, y_pred),
    }


def format_forecast_metrics(
    metrics: dict[str, Any],
    decimal_places: int = 4,
) -> dict[str, float]:
    """
    Round forecasting metrics for readable reporting.
    """
    if decimal_places < 0:
        raise ValueError(
            "decimal_places must be greater than or equal to 0."
        )

    return {
        name: round(float(value), decimal_places)
        for name, value in metrics.items()
    }