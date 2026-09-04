"""
Baseline workload forecasting models.

These models provide simple reference points for evaluating more complex
machine learning forecasting approaches.
"""

from __future__ import annotations

import numpy as np

from .base import BaseForecaster


class NaiveForecaster(BaseForecaster):
    """Naive persistence forecasting model."""

    def __init__(self) -> None:
        self.last_value: float | None = None

    def fit(self, y: np.ndarray) -> "NaiveForecaster":
        values = np.asarray(y, dtype=float)

        if values.ndim != 1:
            raise ValueError(
                "NaiveForecaster expects a one-dimensional array."
            )

        if len(values) == 0:
            raise ValueError(
                "Cannot fit NaiveForecaster with an empty array."
            )

        self.last_value = float(values[-1])

        return self

    def predict(self, steps: int) -> np.ndarray:
        if self.last_value is None:
            raise RuntimeError(
                "NaiveForecaster must be fitted before prediction."
            )

        if steps < 1:
            raise ValueError(
                "steps must be at least 1."
            )

        return np.full(
            steps,
            self.last_value,
            dtype=float,
        )


class MovingAverageForecaster(BaseForecaster):
    """Recursive moving-average workload forecasting model."""

    def __init__(
        self,
        window_size: int = 3,
        *,
        window: int | None = None,
    ) -> None:
        """
        Initialize the moving average forecaster.

        `window_size` is the primary argument. `window` is accepted as
        a compatibility alias.
        """
        if window is not None:
            if window_size != 3 and window_size != window:
                raise ValueError(
                    "Specify only one window value."
                )
            window_size = window

        if window_size < 1:
            raise ValueError(
                "window_size must be at least 1."
            )

        self.window_size = window_size
        self.history: np.ndarray | None = None

    def fit(
        self,
        y: np.ndarray,
    ) -> "MovingAverageForecaster":
        values = np.asarray(y, dtype=float)

        if values.ndim != 1:
            raise ValueError(
                "MovingAverageForecaster expects a "
                "one-dimensional array."
            )

        if len(values) == 0:
            raise ValueError(
                "Cannot fit MovingAverageForecaster "
                "with an empty array."
            )

        if len(values) < self.window_size:
            raise ValueError(
                "The number of observations must be greater "
                "than or equal to window_size."
            )

        self.history = values.copy()

        return self

    def predict(self, steps: int) -> np.ndarray:
        if self.history is None:
            raise RuntimeError(
                "MovingAverageForecaster must be fitted "
                "before prediction."
            )

        if steps < 1:
            raise ValueError(
                "steps must be at least 1."
            )

        history = list(self.history.astype(float))
        predictions: list[float] = []

        for _ in range(steps):
            prediction = float(
                np.mean(
                    history[-self.window_size:]
                )
            )

            predictions.append(prediction)

            # Recursive forecasting:
            # use the new prediction in future windows.
            history.append(prediction)

        return np.asarray(
            predictions,
            dtype=float,
        )