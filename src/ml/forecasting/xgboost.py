import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error
from xgboost import XGBRegressor

from ml.functions.eval import compute_metrics
from ml.modeling import (
    _split,
    _make_features,
)





def _base_model(early_stopping: bool = False) -> XGBRegressor:
    params = dict(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="rmse",
        verbosity=0,
    )
    if early_stopping:
        params["early_stopping_rounds"] = 30
    return XGBRegressor(**params)




def fit_xgboost(s: pd.Series) -> dict:

    try:
        train, val, test = _split(s)

        full_features = _make_features(s)

        train_feat = full_features.loc[full_features.index.isin(train.index)]
        val_feat   = full_features.loc[full_features.index.isin(val.index)]

        feature_cols = [c for c in full_features.columns if c != "sales"]

        X_train, y_train = train_feat[feature_cols], train_feat["sales"]
        X_val,   y_val   = val_feat[feature_cols],   val_feat["sales"]

        model = _base_model(early_stopping=True)
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


def walk_forward_cv(s: pd.Series, model: XGBRegressor = None, n_rep: int = 5) -> pd.DataFrame:
    """
    Walk-forward CV where the last fold is exactly the real train/val split.

    Args:
        s:     full series (train + val + test)
        model: unfitted XGBRegressor used as template (cloned each fold)
        n_rep: number of folds

    Returns:
        DataFrame: fold, train_from, train_to, val_from, val_to, MAPE
    """
    from sklearn.base import clone

    if model is None:
        model = _base_model()

    # anchor on the exact same split fit_xgboost uses
    train, val, _ = _split(s)
    features      = _make_features(s)
    feature_cols  = [c for c in features.columns if c != "sales"]

    # find positions in features (after dropna) that correspond to real boundaries
    train_end_pos = features.index.searchsorted(train.index[-1], side="right")
    val_end_pos   = features.index.searchsorted(val.index[-1],   side="right")
    val_size      = val_end_pos - train_end_pos   # constant across all folds

    rows = []
    for i in range(n_rep):
        # last fold (i = n_rep-1) → exactly train → val
        steps_back    = (n_rep - 1 - i)
        fold_val_end  = val_end_pos  - steps_back * val_size
        fold_val_start= fold_val_end - val_size

        if fold_val_start <= 0:
            continue

        X_train = features.iloc[:fold_val_start][feature_cols]
        y_train = features.iloc[:fold_val_start]["sales"]
        X_val   = features.iloc[fold_val_start:fold_val_end][feature_cols]
        y_val   = features.iloc[fold_val_start:fold_val_end]["sales"]

        m = clone(model)
        m.fit(X_train, y_train, verbose=False)

        preds = np.clip(m.predict(X_val), 0, None)
        mape  = mean_absolute_percentage_error(y_val, preds)

        rows.append({
            "fold":       i + 1,
            "train_from": X_train.index[0].date(),
            "train_to":   X_train.index[-1].date(),
            "val_from":   X_val.index[0].date(),
            "val_to":     X_val.index[-1].date(),
            "MAPE":       round(mape, 4),
        })

    return pd.DataFrame(rows)