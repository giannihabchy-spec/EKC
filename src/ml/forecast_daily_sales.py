import json
import calendar
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from ml.loaders import load_daily_sales
from ml.modeling import get_series, _split, _make_features
from supa.db import get_last_table


def _rebuild_model(model_name: str, best_params: dict):
    if model_name == "rf":
        return RandomForestRegressor(**best_params, random_state=42, n_jobs=-1)
    if model_name == "xgb":
        return XGBRegressor(
            **best_params,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
            tree_method="hist",
        )
    raise ValueError(f"Unknown model: {model_name}")


def _horizon_end(max_date: pd.Timestamp) -> pd.Timestamp:
    """Last day of the month that is 2 months ahead of max_date."""
    month = max_date.month + 2
    year = max_date.year + (month - 1) // 12
    month = (month - 1) % 12 + 1
    last_day = calendar.monthrange(year, month)[1]
    return pd.Timestamp(year=year, month=month, day=last_day)


def forecast_daily_sales(branch_id: int) -> dict:
    """
    For each category, re-fits the best model and returns a dict:

        {
            category: {
                "model":       str,
                "train_val":   pd.Series,   # actuals, train+val period
                "test_actual": pd.Series,   # actuals, test period
                "test_pred":   pd.Series,   # model predictions on test period
                "forecast":    pd.Series,   # recursive forecast, max_date+1 to end of month+2
                "metrics":     dict,        # final_mae, final_rmse, final_wape
            }
        }
    """
    data = load_daily_sales(branch_id)
    results_df = get_last_table(branch_id, "forecast_daily_sales_results")

    if results_df.empty:
        raise ValueError(f"No forecast results found for branch_id={branch_id}")

    best = results_df[results_df["is_best"]].copy()
    best["result"] = best["result"].apply(
        lambda x: json.loads(x) if isinstance(x, str) else x
    )

    series = get_series(data, "category")
    output = {}

    for _, row in best.iterrows():
        category = row["category"]
        model_name = row["model"]
        result_json = row["result"]
        best_params = result_json["best_params"]
        final_features = result_json["final_features"]

        if category not in series:
            continue

        s = series[category]
        train, val, test = _split(s)
        full_features = _make_features(s)

        train_val_idx = train.index.union(val.index)
        test_idx = test.index

        x_train_val = full_features.loc[full_features.index.isin(train_val_idx), final_features]
        y_train_val = full_features.loc[full_features.index.isin(train_val_idx), "sales"]
        x_test = full_features.loc[full_features.index.isin(test_idx), final_features]
        y_test = full_features.loc[full_features.index.isin(test_idx), "sales"]

        model = _rebuild_model(model_name, best_params)
        model.fit(x_train_val, y_train_val)
        test_pred = model.predict(x_test)

        # Recursive forecast
        max_date = s.index.max()
        future_dates = pd.date_range(
            start=max_date + pd.Timedelta(days=1),
            end=_horizon_end(max_date),
            freq="D",
        )

        history = s.copy()
        forecast_values = []

        for d in future_dates:
            feat_df = _make_features(history)
            last_row = feat_df.iloc[[-1]][final_features]
            pred = float(model.predict(last_row)[0])
            pred = max(0.0, round(pred))
            history[d] = pred
            forecast_values.append(pred)

        output[category] = {
            "model": model_name,
            "train_val": s.loc[train_val_idx],
            "test_actual": y_test,
            "test_pred": pd.Series(test_pred, index=test_idx, name="test_pred"),
            "forecast": pd.Series(forecast_values, index=future_dates, name="forecast"),
            "metrics": result_json["metrics"],
        }

    return output
