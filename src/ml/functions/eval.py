from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error
)
import numpy as np


def evaluate_model(model, val):
    forecast = model.predict(n_periods=len(val))

    mae = mean_absolute_error(val, forecast)
    rmse = np.sqrt(mean_squared_error(val, forecast))
    mape = mean_absolute_percentage_error(val, forecast)

    return {
        "forecast": forecast,
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape
    }


def compute_metrics(actual, forecast):
    """Generic metric computation — works with any model (XGBoost, Prophet, etc.)"""
    actual = np.array(actual)
    forecast = np.array(forecast)

    mae  = mean_absolute_error(actual, forecast)
    rmse = np.sqrt(mean_squared_error(actual, forecast))
    mape = mean_absolute_percentage_error(actual, forecast)

    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}