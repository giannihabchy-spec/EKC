import calendar
import pandas as pd
import numpy as np
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error
from ml.modeling import _split
from ml.config import SEASONAL_PERIOD


def fit_sarima(s: pd.Series, freq: str = "D") -> dict:
    train, val, test = _split(s)

    model = auto_arima(
        train,
        m=SEASONAL_PERIOD[freq],
        seasonal=True,
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore",
        information_criterion="aic",
    )

    val_pred = model.predict(n_periods=len(val))
    val_wape = round(np.sum(np.abs(val - val_pred)) / np.sum(np.abs(val)) * 100, 2)

    model.update(val)
    final_pred = model.predict(n_periods=len(test))

    final_mae  = round(mean_absolute_error(test, final_pred), 2)
    final_rmse = round(np.sqrt(np.mean((test - final_pred) ** 2)), 2)
    final_wape = round(np.sum(np.abs(test - final_pred)) / np.sum(np.abs(test)) * 100, 2)

    test_pred = {
        "dates": [d.strftime("%Y-%m-%d") for d in test.index],
        "values": [round(float(v), 2) for v in final_pred],
    }

    model.update(test)

    months = 8
    step = pd.Timedelta(days=1) if freq == "D" else pd.Timedelta(weeks=1)
    max_date = s.index.max()
    end_month = max_date.month + months
    end_year = max_date.year + (end_month - 1) // 12
    end_month = (end_month - 1) % 12 + 1
    last_day = calendar.monthrange(end_year, end_month)[1]
    horizon_end = pd.Timestamp(year=end_year, month=end_month, day=last_day)
    future_dates = pd.date_range(start=max_date + step, end=horizon_end, freq=freq)
    forecast_values = np.clip(model.predict(n_periods=len(future_dates)), 0, None).round()

    forecast = {
        "dates": [d.strftime("%Y-%m-%d") for d in future_dates],
        "values": [round(float(v), 2) for v in forecast_values],
    }

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
        "test_pred": test_pred,
        "forecast": forecast,
        "from": from_date,
        "to": to_date,
    }
