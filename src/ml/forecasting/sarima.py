import pandas as pd
import numpy as np
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error
from ml.modeling import _split
from ml.config import SEASONAL_PERIOD


def fit_sarima(s: pd.Series) -> dict:
    train, val, test = _split(s)

    model = auto_arima(
        train,
        m=SEASONAL_PERIOD,
        seasonal=True,
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore",
        information_criterion="aic",
    )

    val_pred = model.predict(n_periods=len(val))
    val_wape = round(np.sum(np.abs(val - val_pred)) / np.sum(np.abs(val)) * 100, 2)

    model.update(val)
    test_pred = model.predict(n_periods=len(test))

    final_mae  = round(mean_absolute_error(test, test_pred), 2)
    final_rmse = round(np.sqrt(np.mean((test - test_pred) ** 2)), 2)
    final_wape = round(np.sum(np.abs(test - test_pred)) / np.sum(np.abs(test)) * 100, 2)

    best_params = {
        "order": model.order,
        "seasonal_order": model.seasonal_order,
    }

    metrics = {
        "val_wape": val_wape,
        "final_mae": final_mae,
        "final_rmse": final_rmse,
        "final_wape": final_wape,
    }

    from_date = s.index.min().strftime("%Y-%m-%d")
    to_date = s.index.max().strftime("%Y-%m-%d")

    return {
        "model": model,
        "best_params": best_params,
        "metrics": metrics,
        "final_features": None,
        "from": from_date,
        "to": to_date,
    }
