import json
import calendar
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from pmdarima import auto_arima
from typing import Literal

from ml.loaders import load_daily_sales
from ml.modeling import get_series, _split, _make_features
from ml.config import SEASONAL_PERIOD
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
    if model_name == "sarima":
        return None
    raise ValueError(f"Unknown model: {model_name}")


def _horizon_end(max_date: pd.Timestamp, months: int) -> pd.Timestamp:
    month = max_date.month + months
    year = max_date.year + (month - 1) // 12
    month = (month - 1) % 12 + 1
    last_day = calendar.monthrange(year, month)[1]
    return pd.Timestamp(year=year, month=month, day=last_day)


def forecast_daily_sales(branch_id: int, months: int = 2, freq: str = "D") -> dict:

    data = load_daily_sales(branch_id)
    results_df = get_last_table(branch_id, "forecast_daily_sales_results")

    if results_df.empty:
        raise ValueError(f"No forecast results found for branch_id={branch_id}")

    best = results_df[(results_df["is_best"]) & (results_df['freq'] == freq)].copy()
    best["result"] = best["result"].apply(
        lambda x: json.loads(x) if isinstance(x, str) else x
    )

    series = get_series(data, "category", freq)
    output = {}

    step = pd.Timedelta(days=1) if freq == "D" else pd.Timedelta(weeks=1)

    for _, row in best.iterrows():
        category = row["category"]
        model_name = row["model"]
        result_json = row["result"]
        best_params = result_json["best_params"]
        final_features = result_json.get("final_features")

        if category not in series:
            continue

        s = series[category]
        train, val, test = _split(s)
        train_val_idx = train.index.union(val.index)
        test_idx = test.index

        max_date = s.index.max()
        future_dates = pd.date_range(
            start=max_date + step,
            end=_horizon_end(max_date, months),
            freq=freq,
        )

        if model_name == "sarima":
            sarima_model = auto_arima(
                s.loc[train_val_idx],
                m=SEASONAL_PERIOD[freq],
                seasonal=True,
                stepwise=True,
                suppress_warnings=True,
                error_action="ignore",
                order=tuple(best_params["order"]),
                seasonal_order=tuple(best_params["seasonal_order"]),
            )
            test_pred = sarima_model.predict(n_periods=len(test))
            sarima_model.update(test)
            forecast_values = np.clip(sarima_model.predict(n_periods=len(future_dates)), 0, None).round()
        else:
            full_features = _make_features(s, freq)

            x_train_val = full_features.loc[full_features.index.isin(train_val_idx), final_features]
            y_train_val = full_features.loc[full_features.index.isin(train_val_idx), "sales"]
            x_test = full_features.loc[full_features.index.isin(test_idx), final_features]
            y_test = full_features.loc[full_features.index.isin(test_idx), "sales"]

            model = _rebuild_model(model_name, best_params)
            model.fit(x_train_val, y_train_val)
            test_pred = model.predict(x_test)

            history = s.copy()
            forecast_values = []
            for d in future_dates:
                feat_df = _make_features(history, freq)
                last_row = feat_df.iloc[[-1]][final_features]
                pred = max(0.0, round(float(model.predict(last_row)[0])))
                history[d] = pred
                forecast_values.append(pred)

        y_test = s.loc[test_idx]

        output[category] = {
            "model": model_name,
            "train_val": s.loc[train_val_idx],
            "test_actual": y_test,
            "test_pred": pd.Series(np.array(test_pred), index=test_idx, name="test_pred"),
            "forecast": pd.Series(np.array(forecast_values), index=future_dates, name="forecast"),
            "metrics": result_json["metrics"],
        }

    return output



def forecast_daily_sales_by_model(branch_id: int, m: Literal['rf','xgb','sarima'], months: int = 2, freq: str = "D") -> dict:

    data = load_daily_sales(branch_id)
    results_df = get_last_table(branch_id, "forecast_daily_sales_results")

    if results_df.empty:
        raise ValueError(f"No forecast results found for branch_id={branch_id}")

    best = results_df[(results_df["model"] == m) & (results_df['freq'] == freq)].copy()
    best["result"] = best["result"].apply(
        lambda x: json.loads(x) if isinstance(x, str) else x
    )

    series = get_series(data, "category", freq)
    output = {}

    step = pd.Timedelta(days=1) if freq == "D" else pd.Timedelta(weeks=1)

    for _, row in best.iterrows():
        category = row["category"]
        model_name = row["model"]
        result_json = row["result"]
        best_params = result_json["best_params"]
        final_features = result_json.get("final_features")

        if category not in series:
            continue

        s = series[category]
        train, val, test = _split(s)
        train_val_idx = train.index.union(val.index)
        test_idx = test.index

        max_date = s.index.max()
        future_dates = pd.date_range(
            start=max_date + step,
            end=_horizon_end(max_date, months),
            freq=freq,
        )

        if model_name == "sarima":
            sarima_model = auto_arima(
                s.loc[train_val_idx],
                m=SEASONAL_PERIOD[freq],
                seasonal=True,
                stepwise=True,
                suppress_warnings=True,
                error_action="ignore",
                order=tuple(best_params["order"]),
                seasonal_order=tuple(best_params["seasonal_order"]),
            )
            test_pred = sarima_model.predict(n_periods=len(test))
            sarima_model.update(test)
            forecast_values = np.clip(sarima_model.predict(n_periods=len(future_dates)), 0, None).round()
        else:
            full_features = _make_features(s, freq)

            x_train_val = full_features.loc[full_features.index.isin(train_val_idx), final_features]
            y_train_val = full_features.loc[full_features.index.isin(train_val_idx), "sales"]
            x_test = full_features.loc[full_features.index.isin(test_idx), final_features]
            y_test = full_features.loc[full_features.index.isin(test_idx), "sales"]

            model = _rebuild_model(model_name, best_params)
            model.fit(x_train_val, y_train_val)
            test_pred = model.predict(x_test)

            history = s.copy()
            forecast_values = []
            for d in future_dates:
                feat_df = _make_features(history, freq)
                last_row = feat_df.iloc[[-1]][final_features]
                pred = max(0.0, round(float(model.predict(last_row)[0])))
                history[d] = pred
                forecast_values.append(pred)

        y_test = s.loc[test_idx]

        output[category] = {
            "model": model_name,
            "train_val": s.loc[train_val_idx],
            "test_actual": y_test,
            "test_pred": pd.Series(np.array(test_pred), index=test_idx, name="test_pred"),
            "forecast": pd.Series(np.array(forecast_values), index=future_dates, name="forecast"),
            "metrics": result_json["metrics"],
        }

    return output