"""
Tests for workload forecasting components.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.forecasting import (
MovingAverageForecaster,
NaiveForecaster,
create_time_features,
evaluate_forecast,
)

def test_naive_forecaster_predicts_last_value() -> None:
 """Naive forecasting should repeat the most recent value."""
values = np.array(
[10.0, 20.0, 30.0, 40.0]
)


model = NaiveForecaster()
model.fit(values)

predictions = model.predict(3)

expected = np.array(
    [40.0, 40.0, 40.0]
)

np.testing.assert_array_equal(
    predictions,
    expected,
)


def test_naive_forecaster_requires_fit() -> None:
 """Naive forecasting should fail before fitting."""
model = NaiveForecaster()


with pytest.raises(RuntimeError):
    model.predict(1)

def test_naive_forecaster_rejects_empty_values() -> None:
 """Naive forecasting should reject empty training data."""
model = NaiveForecaster()


with pytest.raises(ValueError):
    model.fit([])

def test_moving_average_forecaster_predicts_average() -> None:
 """Moving average forecasting should use the configured window."""
values = np.array(
[10.0, 20.0, 30.0, 40.0]
)


model = MovingAverageForecaster(
    window_size=2
)

model.fit(values)

predictions = model.predict(3)

expected = np.array(
    [35.0, 37.5, 36.25]
)

np.testing.assert_allclose(
    predictions,
    expected,
)

def test_moving_average_forecaster_rejects_invalid_window() -> None:
 """Moving average forecaster should reject non-positive windows."""
with pytest.raises(ValueError):
 MovingAverageForecaster(
window_size=0
)

def test_moving_average_forecaster_requires_enough_data() -> None:
 """Moving average forecaster should require enough training values."""
model = MovingAverageForecaster(
window_size=5
)


with pytest.raises(ValueError):
    model.fit(
        [10.0, 20.0, 30.0]
    )


def test_create_time_features() -> None:
 """Time feature creation should add useful calendar features."""
timestamps = pd.date_range(
start="2026-01-01",
periods=3,
freq="h",
)


data = pd.DataFrame(
    {
        "timestamp": timestamps,
        "requests": [10, 20, 30],
    }
)

features = create_time_features(
    data
)

assert len(features) == 3
assert "hour" in features.columns
assert "day_of_week" in features.columns
assert "hour_sin" in features.columns
assert "hour_cos" in features.columns

def test_evaluate_forecast_returns_metrics() -> None:
 """Forecast evaluation should return MAE, RMSE, and MAPE."""
actual = np.array(
[10.0, 20.0, 30.0]
)


predicted = np.array(
    [12.0, 18.0, 33.0]
)

metrics = evaluate_forecast(
    actual,
    predicted,
)

assert "mae" in metrics
assert "rmse" in metrics
assert "mape" in metrics

assert metrics["mae"] >= 0
assert metrics["rmse"] >= 0
assert metrics["mape"] >= 0
