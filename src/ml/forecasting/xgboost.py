import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.inspection import permutation_importance
from ml.modeling import (
    _split,
    _make_features,
)
from ml.config import xgb_tuning
from ml.forecast_daily_sales import recursive_forecast


n_ests = xgb_tuning['n_est']
learning_rates = xgb_tuning['learning_rate']
max_depths = xgb_tuning['max_depth']
subsamples = xgb_tuning['subsample']
colsample_bytrees = xgb_tuning['colsample_bytree']



def fit_xgb(s: pd.Series, freq: str = "D") -> dict:
    train, val, test = _split(s)

    full_features = _make_features(s, freq)
    train_data = full_features.loc[full_features.index.isin(train.index)]
    val_data   = full_features.loc[full_features.index.isin(val.index)]
    test_data  = full_features.loc[full_features.index.isin(test.index)]

    feature_cols = [c for c in full_features.columns if c != "sales"]

    x_train, y_train = train_data[feature_cols], train_data["sales"]
    x_val, y_val     = val_data[feature_cols], val_data["sales"]
    x_test, y_test   = test_data[feature_cols], test_data["sales"]

    best_wape = float("inf")
    best_params = None

    for n_est in n_ests:

        for learning_rate in learning_rates:

            for max_depth in max_depths:

                for subsample in subsamples:

                    for colsample_bytree in colsample_bytrees:

                        xgb = XGBRegressor(
                            n_estimators=n_est,
                            learning_rate=learning_rate,
                            max_depth=max_depth,
                            subsample=subsample,
                            colsample_bytree=colsample_bytree,
                            objective="reg:squarederror",
                            random_state=42,
                            n_jobs=-1,
                            tree_method="hist"
                        )

                        xgb.fit(x_train, y_train)

                        pred = xgb.predict(x_val)
                        wape = np.sum(np.abs(y_val - pred)) / np.sum(np.abs(y_val))

                        if wape < best_wape:
                            best_wape = wape
                            best_params = {
                                "n_estimators": n_est,
                                "learning_rate": learning_rate,
                                "max_depth": max_depth,
                                "subsample": subsample,
                                "colsample_bytree": colsample_bytree
                            }

    xgb = XGBRegressor(
        **best_params,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
        tree_method="hist"
    )

    xgb.fit(x_train, y_train)
    pred = xgb.predict(x_val)

    result = permutation_importance(
        xgb,
        x_val,
        y_val,
        scoring="neg_mean_absolute_error",
        n_repeats=10,
        random_state=42,
        n_jobs=-1
    )

    imp = pd.Series(
        result.importances_mean,
        index=x_val.columns
    ).sort_values(ascending=False)

    selected_features = imp[imp > 0].index.to_list()

    if len(selected_features) == 0:
        selected_features = feature_cols

    new_x_train = x_train[selected_features]
    new_x_val = x_val[selected_features]

    xgb.fit(new_x_train, y_train)
    new_pred = xgb.predict(new_x_val)

    # mae = round(mean_absolute_error(y_val, pred), 2)
    # new_mae = round(mean_absolute_error(y_val, new_pred), 2)

    wape = np.sum(np.abs(y_val - pred)) / np.sum(np.abs(y_val))
    new_wape = np.sum(np.abs(y_val - new_pred)) / np.sum(np.abs(y_val))

    if new_wape <= wape:
        final_features = selected_features
        val_wape = round(new_wape * 100, 2)
    else:
        final_features = feature_cols
        val_wape = round(wape * 100, 2)

    x_train_val = pd.concat([x_train, x_val])[final_features]
    y_train_val = pd.concat([y_train, y_val])
    x_test = x_test[final_features]

    xgb.fit(x_train_val, y_train_val)
    final_pred = xgb.predict(x_test)

    final_mae  = round(mean_absolute_error(y_test, final_pred), 2)
    final_rmse = round(np.sqrt(np.mean((y_test - final_pred) ** 2)), 2)
    final_wape = round(np.sum(np.abs(y_test - final_pred)) / np.sum(np.abs(y_test)) * 100, 2)

    test_pred = {
        "dates": [d.strftime("%Y-%m-%d") for d in y_test.index],
        "values": [round(float(v), 2) for v in final_pred],
    }

    x_full = full_features[final_features]
    y_full = full_features["sales"]
    xgb.fit(x_full, y_full)

    forecast = recursive_forecast(xgb, s, final_features, freq)

    metrics = {
        "val_wape": val_wape,
        "final_mae": final_mae,
        "final_rmse": final_rmse,
        "final_wape": final_wape,
    }

    from_date = full_features.index.min().strftime("%Y-%m-%d")
    to_date = full_features.index.max().strftime("%Y-%m-%d")

    return {
        "model": xgb,
        "best_params": best_params,
        "metrics": metrics,
        "final_features": final_features,
        "test_pred": test_pred,
        "forecast": forecast,
        'from': from_date,
        'to': to_date,
    }