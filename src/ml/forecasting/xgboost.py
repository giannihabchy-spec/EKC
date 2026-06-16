import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from ml.functions.eval import compute_metrics
from ml.forecasting.sarima import _split


LAGS     = [1, 2, 3, 7, 14, 21, 28]
ROLLS    = [7, 14, 28]


def _make_features(s: pd.Series) -> pd.DataFrame:
    """
    Build a feature DataFrame from a daily sales series.
    Calendar features + lag features + rolling means.
    """
    df = s.rename("sales").to_frame()

    df["day_of_week"] = df.index.dayofweek
    df["day_of_month"] = df.index.day
    df["month"] = df.index.month
    df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)

    for lag in LAGS:
        df[f"lag_{lag}"] = df["sales"].shift(lag)

    for window in ROLLS:
        df[f"roll_mean_{window}"] = df["sales"].shift(1).rolling(window).mean()
        df[f"roll_std_{window}"]  = df["sales"].shift(1).rolling(window).std()

    df = df.dropna()
    return df


def fit_xgboost(s: pd.Series) -> dict:
    """
    Fit XGBoost on train (70%), evaluate on val (15%), keep test (15%) untouched.

    Returns the same result-dict shape as fit_sarima so they're interchangeable:
        {
            "model":        XGBRegressor,
            "train":        pd.Series,
            "val":          pd.Series,
            "test":         pd.Series,
            "val_metrics":  {"MAE", "RMSE", "MAPE"},
            "val_forecast": pd.Series,
            "status":       "ok" | "error",
            "msg":          str,
        }
    """
    try:
        train, val, test = _split(s)

        full_features = _make_features(s)

        train_feat = full_features.loc[full_features.index.isin(train.index)]
        val_feat   = full_features.loc[full_features.index.isin(val.index)]

        feature_cols = [c for c in full_features.columns if c != "sales"]

        X_train, y_train = train_feat[feature_cols], train_feat["sales"]
        X_val,   y_val   = val_feat[feature_cols],   val_feat["sales"]

        model = XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            early_stopping_rounds=30,
            eval_metric="rmse",
            verbosity=0,
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        val_preds = np.clip(model.predict(X_val), 0, None)
        metrics   = compute_metrics(y_val, val_preds)

        return {
            "model":        model,
            "train":        train,
            "val":          val,
            "test":         test,
            "feature_cols": feature_cols,
            "val_metrics":  metrics,
            "val_forecast": pd.Series(val_preds, index=y_val.index, name=s.name),
            "status":       "ok",
            "msg":          "XGBoost",
        }

    except Exception as e:
        return {
            "model":  None,
            "status": "error",
            "msg":    str(e),
        }


def fit_all(series: dict[str, pd.Series]) -> dict[str, dict]:
    """Fit XGBoost on every series. Works for any number of groups."""
    results = {}
    for name, s in series.items():
        s = s.copy()
        s.name = name
        results[name] = fit_xgboost(s)
    return results


def _recursive_forecast(s_history: pd.Series, model, feature_cols: list, n_periods: int) -> np.ndarray:
    """
    Predict n_periods ahead one step at a time, feeding each prediction
    back as a lag feature for the next step.
    """
    history = list(s_history.values)

    preds = []
    for _ in range(n_periods):
        tmp = pd.Series(history, index=pd.date_range(end="2000-01-01", periods=len(history), freq="D"))
        feat_df = _make_features(tmp)
        if feat_df.empty:
            preds.append(np.nan)
            history.append(np.nan)
            continue
        row = feat_df.iloc[[-1]][feature_cols]
        pred = float(np.clip(model.predict(row), 0, None))
        preds.append(pred)
        history.append(pred)

    return np.array(preds)


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

        s_history = pd.concat([res["train"], res["val"], res["test"]])
        last_date = res["test"].index[-1]
        future_index = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=n_periods,
            freq="D",
        )

        preds = _recursive_forecast(
            s_history, res["model"], res["feature_cols"], n_periods
        )
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
