"""
Base classes and interfaces for workload forecasting models.

All forecasting models in this project are evaluated on synthetic workload
data generated locally unless explicitly documented otherwise.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

class BaseForecaster(ABC):
 """
Abstract base class for workload forecasting models.

```
Every forecasting model should implement the fit and predict methods.
This provides a consistent interface for baseline and machine learning
forecasting approaches.
"""

@abstractmethod
def fit(self, y: np.ndarray) -> "BaseForecaster":
    """
    Fit the forecasting model using historical workload values.

    Args:
        y: One-dimensional array of historical workload observations.

    Returns:
        The fitted forecasting model.
    """

@abstractmethod
def predict(self, steps: int) -> np.ndarray:
    """
    Predict future workload values.

    Args:
        steps: Number of future time steps to forecast.

    Returns:
        A one-dimensional NumPy array containing predicted workload
        values.

    Raises:
        ValueError: If steps is less than 1.
    """
