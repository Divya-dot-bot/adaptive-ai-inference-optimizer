"""
Chronological data splitting utilities for workload forecasting.

Forecasting data must preserve temporal order. These utilities split
synthetic workload data into training, validation, and test sets without
random shuffling.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TimeSeriesSplit:
    """
    Container for chronological train, validation, and test splits.

    Attributes:
        train: Training portion containing the earliest observations.
        validation: Validation portion following the training data.
        test: Test portion containing the latest observations.
    """

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def validate_split_fractions(
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
) -> None:
    """Validate chronological split fractions."""

    fractions = (
        train_fraction,
        validation_fraction,
        test_fraction,
    )

    if any(fraction <= 0 for fraction in fractions):
        raise ValueError(
            "All split fractions must be greater than 0."
        )

    total = sum(fractions)

    if not np.isclose(total, 1.0):
        raise ValueError(
            "Split fractions must sum to 1.0. "
            f"Received total: {total}"
        )


def chronological_split(
    dataframe: pd.DataFrame,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> TimeSeriesSplit:
    """
    Split a DataFrame into chronological train, validation, and test sets.
    """

    if dataframe.empty:
        raise ValueError(
            "Cannot split an empty DataFrame."
        )

    validate_split_fractions(
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
    )

    total_rows = len(dataframe)

    train_end = int(total_rows * train_fraction)
    validation_end = train_end + int(
        total_rows * validation_fraction
    )

    if train_end == 0:
        raise ValueError(
            "Training split contains no observations."
        )

    if validation_end == train_end:
        raise ValueError(
            "Validation split contains no observations."
        )

    if validation_end >= total_rows:
        raise ValueError(
            "Test split contains no observations."
        )

    train = dataframe.iloc[:train_end].copy()
    validation = dataframe.iloc[
        train_end:validation_end
    ].copy()
    test = dataframe.iloc[validation_end:].copy()

    return TimeSeriesSplit(
        train=train,
        validation=validation,
        test=test,
    )


def chronological_split_arrays(
    X: np.ndarray,
    y: np.ndarray,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Split feature and target arrays chronologically."""

    X = np.asarray(X)
    y = np.asarray(y)

    if X.ndim != 2:
        raise ValueError(
            "X must be a two-dimensional array."
        )

    if y.ndim != 1:
        raise ValueError(
            "y must be a one-dimensional array."
        )

    if len(X) != len(y):
        raise ValueError(
            "X and y must contain the same number of observations."
        )

    if len(X) == 0:
        raise ValueError(
            "Cannot split empty arrays."
        )

    validate_split_fractions(
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
    )

    total_rows = len(X)

    train_end = int(total_rows * train_fraction)
    validation_end = train_end + int(
        total_rows * validation_fraction
    )

    if train_end == 0:
        raise ValueError(
            "Training split contains no observations."
        )

    if validation_end == train_end:
        raise ValueError(
            "Validation split contains no observations."
        )

    if validation_end >= total_rows:
        raise ValueError(
            "Test split contains no observations."
        )

    X_train = X[:train_end]
    X_validation = X[train_end:validation_end]
    X_test = X[validation_end:]

    y_train = y[:train_end]
    y_validation = y[train_end:validation_end]
    y_test = y[validation_end:]

    return (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
    )