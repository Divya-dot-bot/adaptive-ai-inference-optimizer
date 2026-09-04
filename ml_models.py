"""
Machine learning models for synthetic workload forecasting.

These models predict future synthetic AI inference workload using engineered
historical features such as lag values, rolling statistics, and time features.

All data used in this project is synthetically generated unless explicitly
documented otherwise.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor

class BaseMLForecaster:
 """
Common interface for machine learning workload forecasting models.

```
Unlike simple baseline forecasters, ML models are trained using a
two-dimensional feature matrix X and a one-dimensional target vector y.
"""

def __init__(self, model: Any) -> None:
    """
    Initialize the forecaster with a scikit-learn regression model.

    Args:
        model: A scikit-learn-compatible regression model.
    """
    self.model = model
    self.is_fitted = False

def fit(
    self,
    X: np.ndarray,
    y: np.ndarray,
) -> "BaseMLForecaster":
    """
    Fit the machine learning forecasting model.

    Args:
        X: Two-dimensional feature matrix.
        y: One-dimensional target array.

    Returns:
        The fitted forecaster.

    Raises:
        ValueError: If X or y has an invalid shape or lengths differ.
    """
    X_array = np.asarray(X, dtype=float)
    y_array = np.asarray(y, dtype=float)

    if X_array.ndim != 2:
        raise ValueError(
            "X must be a two-dimensional feature array."
        )

    if y_array.ndim != 1:
        raise ValueError(
            "y must be a one-dimensional target array."
        )

    if len(X_array) != len(y_array):
        raise ValueError(
            "X and y must contain the same number of observations."
        )

    if len(X_array) == 0:
        raise ValueError(
            "Cannot fit a model with empty training data."
        )

    self.model.fit(X_array, y_array)
    self.is_fitted = True

    return self

def predict(
    self,
    X: np.ndarray,
) -> np.ndarray:
    """
    Predict workload values from feature data.

    Args:
        X: Two-dimensional feature matrix.

    Returns:
        One-dimensional array of predicted workload values.

    Raises:
        RuntimeError: If the model has not been fitted.
        ValueError: If X is not two-dimensional.
    """
    if not self.is_fitted:
        raise RuntimeError(
            "The forecasting model must be fitted before prediction."
        )

    X_array = np.asarray(X, dtype=float)

    if X_array.ndim != 2:
        raise ValueError(
            "X must be a two-dimensional feature array."
        )

    predictions = self.model.predict(X_array)

    # Synthetic request counts cannot be negative.
    return np.maximum(
        np.asarray(predictions, dtype=float),
        0.0,
    )

class RandomForestForecaster(BaseMLForecaster):
 """
Random Forest regressor for workload forecasting.
"""


def __init__(
    self,
    n_estimators: int = 200,
    max_depth: int | None = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    random_seed: int = 42,
) -> None:
    """
    Initialize the Random Forest workload forecaster.

    Args:
        n_estimators: Number of decision trees.
        max_depth: Maximum tree depth, or None for no fixed limit.
        min_samples_split: Minimum samples required to split a node.
        min_samples_leaf: Minimum samples required at a leaf.
        random_seed: Fixed seed for reproducibility.
    """
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_seed,
        n_jobs=-1,
    )

    super().__init__(model=model)


class HistGradientBoostingForecaster(BaseMLForecaster):
 """
Histogram-based Gradient Boosting regressor for workload forecasting.
"""


def __init__(
    self,
    max_iter: int = 200,
    learning_rate: float = 0.05,
    max_leaf_nodes: int = 31,
    random_seed: int = 42,
) -> None:
    """
    Initialize the HistGradientBoosting workload forecaster.

    Args:
        max_iter: Maximum number of boosting iterations.
        learning_rate: Learning rate used by boosting.
        max_leaf_nodes: Maximum number of leaves per tree.
        random_seed: Fixed seed for reproducibility.
    """
    model = HistGradientBoostingRegressor(
        max_iter=max_iter,
        learning_rate=learning_rate,
        max_leaf_nodes=max_leaf_nodes,
        random_state=random_seed,
    )

    super().__init__(model=model)

