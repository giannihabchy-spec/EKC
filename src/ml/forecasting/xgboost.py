import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.inspection import permutation_importance
from ml.modeling import (
    get_series,
    _split,
    _make_features
)
from ml.loaders import load_daily_sales


def fit_xgb(s: pd.Series) -> dict:
    train, val, test = _split(s)

    full_features = _make_features(s)
    train_feat = full_features.loc[full_features.index.isin(train.index)]
    val_feat   = full_features.loc[full_features.index.isin(val.index)]
    test_feat  = full_features.loc[full_features.index.isin(test.index)]

    feature_cols = [c for c in full_features.columns if c != "sales"]

    x_train, y_train = train_feat[feature_cols], train_feat["sales"]
    x_val, y_val     = val_feat[feature_cols], val_feat["sales"]
    x_test, y_test   = test_feat[feature_cols], test_feat["sales"]

    best_mae = float("inf")
    best_params = None

    for n_est in [100, 300, 500]:
        for learning_rate in [0.01, 0.05, 0.1]:
            for max_depth in [3, 5, 7]:
                for subsample in [0.8, 1.0]:
                    for colsample_bytree in [0.8, 1.0]:

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
                        mae = mean_absolute_error(y_val, pred)

                        if mae < best_mae:
                            best_mae = mae
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

    # fallback in case permutation importance selects nothing
    if len(selected_features) == 0:
        selected_features = feature_cols

    new_x_train = x_train[selected_features]
    new_x_val = x_val[selected_features]

    xgb.fit(new_x_train, y_train)
    new_pred = xgb.predict(new_x_val)

    mae = round(mean_absolute_error(y_val, pred), 2)
    new_mae = round(mean_absolute_error(y_val, new_pred), 2)

    if new_mae <= mae:
        final_features = selected_features
    else:
        final_features = feature_cols

    x_train_val = pd.concat([x_train, x_val])[final_features]
    y_train_val = pd.concat([y_train, y_val])
    x_test = x_test[final_features]

    xgb.fit(x_train_val, y_train_val)
    final_pred = xgb.predict(x_test)

    final_mae  = round(mean_absolute_error(y_test, final_pred), 2)
    final_rmse = round(np.sqrt(np.mean((y_test - final_pred) ** 2)), 2)
    final_wape = round(np.sum(np.abs(y_test - final_pred)) / np.sum(np.abs(y_test)) * 100, 2)

    return {
        "model": xgb,
        "best_params": best_params,
        "final_mae": final_mae,
        "final_rmse": final_rmse,
        "final_wape": final_wape,
        "final_features": final_features,
    }