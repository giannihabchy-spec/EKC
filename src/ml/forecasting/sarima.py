import pandas as pd
import numpy as np
from pmdarima import auto_arima
from ml.functions.eval import evaluate_model
from ml.modeling import _split
from ml.config import SEASONAL_PERIOD


def fit_sarima(s: pd.Series) -> dict:
    """
    Fit auto-ARIMA on train (70%), evaluate on val (15%), keep test (15%) untouched.

    Returns:
        {
            "model":          fitted AutoARIMA,
            "train":          pd.Series,
            "val":            pd.Series,
            "test":           pd.Series,   # untouched — for final evaluation only
            "val_metrics":    {"MAE", "RMSE", "MAPE"},
            "val_forecast":   pd.Series,
            "order":          (p, d, q),
            "seasonal_order": (P, D, Q, m),
            "status":         "ok" | "error",
            "msg":            str,
        }
    """
    try:
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

        val_eval = evaluate_model(model, val)

        return {
            "model":          model,
            "train":          train,
            "val":            val,
            "test":           test,
            "val_metrics":    {k: v for k, v in val_eval.items() if k != "forecast"},
            "val_forecast":   pd.Series(val_eval["forecast"], index=val.index),
            "order":          model.order,
            "seasonal_order": model.seasonal_order,
            "status":         "ok",
            "msg":            f"SARIMA{model.order}x{model.seasonal_order}",
        }

    except Exception as e:
        return {
            "model":  None,
            "status": "error",
            "msg":    str(e),
        }


def fit_all(series: dict[str, pd.Series]) -> dict[str, dict]:
    """
    Fit SARIMA on every series.
    Works for any number of groups — iterates over whatever get_series returns.

    Returns:
        { group_name: result_dict }
    """
    results = {}
    for name, s in series.items():
        s = s.copy()
        s.name = name
        results[name] = fit_sarima(s)
    return results


def forecast_all(results: dict[str, dict], n_periods: int = 30) -> dict[str, pd.Series]:
    """
    Produce n_periods-ahead forecasts starting after the test set.

    Returns:
        { group_name: pd.Series(forecast, index=future_dates) }
    """
    forecasts = {}
    for name, res in results.items():
        if res["status"] != "ok":
            continue

        model = res["model"]
        last_date = res["test"].index[-1]
        future_index = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=n_periods,
            freq="D",
        )
        preds = np.clip(model.predict(n_periods=n_periods), 0, None)
        forecasts[name] = pd.Series(preds, index=future_index, name=name)

    return forecasts


def summary(results: dict[str, dict]) -> pd.DataFrame:
    """Val-set metrics for all groups."""
    rows = []
    for name, res in results.items():
        row = {"group": name, "status": res["status"], "model": res.get("msg", "")}
        if res["status"] == "ok":
            row.update(res["val_metrics"])
        rows.append(row)
    return pd.DataFrame(rows)
