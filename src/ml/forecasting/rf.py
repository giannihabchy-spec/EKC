import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.inspection import permutation_importance
from ml.modeling import (
    _split,
    _make_features
)
from ml.config import rf_tuning

max_depths = rf_tuning['max_depth']
min_samples_leafs = rf_tuning['min_samples_leaf']
max_featuress = rf_tuning['max_features']
n_ests = rf_tuning['n_est']



def fit_rf(s: pd.Series, freq: str = "D") -> dict:
    train, val, test = _split(s)

    full_features = _make_features(s, freq)
    train_data = full_features.loc[full_features.index.isin(train.index)]
    val_data   = full_features.loc[full_features.index.isin(val.index)]
    test_data = full_features.loc[full_features.index.isin(test.index)]

    feature_cols = [c for c in full_features.columns if c != "sales"]
    x_train, y_train = train_data[feature_cols], train_data["sales"]
    x_val,   y_val   = val_data[feature_cols],   val_data["sales"]
    x_test,   y_test   = test_data[feature_cols],   test_data["sales"]

    best_wape = float("inf")
    best_params = None

    for max_depth in max_depths:

        for min_samples_leaf in min_samples_leafs:

            for max_features in max_featuress:

                for n_est in n_ests:

                    rf = RandomForestRegressor(
                        n_estimators=n_est,
                        max_depth=max_depth,
                        min_samples_leaf=min_samples_leaf,
                        max_features=max_features,
                        random_state=42,
                        n_jobs=-1
                    )

                    rf.fit(x_train, y_train)

                    pred = rf.predict(x_val)

                    wape = np.sum(np.abs(y_val - pred)) / np.sum(np.abs(y_val))

                    if wape < best_wape:
                        best_wape = wape
                        best_params = {
                            'n_estimators': n_est,
                            "max_depth": max_depth,
                            "min_samples_leaf": min_samples_leaf,
                            "max_features": max_features
                        }
    n_est = best_params['n_estimators']
    max_depth = best_params['max_depth']
    min_samples_leaf = best_params['min_samples_leaf']
    max_features = best_params['max_features']

    rf = RandomForestRegressor(
        n_estimators=n_est,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(x_train, y_train)
    pred = rf.predict(x_val)

    result = permutation_importance(
        rf,
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
    selected_features = imp[imp>0].index.to_list()
    if not selected_features:
        selected_features = feature_cols

    new_x_train = x_train[selected_features]
    new_x_val = x_val[selected_features]
    rf.fit(new_x_train,y_train)
    new_pred = rf.predict(new_x_val)

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

    x_train_val = pd.concat([x_train,x_val])[final_features]
    y_train_val = pd.concat([y_train,y_val])
    x_test = x_test[final_features]

    rf.fit(x_train_val,y_train_val)
    final_pred = rf.predict(x_test)

    final_mae  = round(mean_absolute_error(y_test, final_pred), 2)
    final_rmse = round(np.sqrt(np.mean((y_test - final_pred) ** 2)), 2)
    final_wape = round(np.sum(np.abs(y_test - final_pred)) / np.sum(np.abs(y_test)) * 100, 2)

    metrics = {
        "val_wape": val_wape,
        "final_mae": final_mae,
        "final_rmse": final_rmse,
        "final_wape": final_wape,
    }

    from_date = full_features.index.min().strftime("%Y-%m-%d")
    to_date = full_features.index.max().strftime("%Y-%m-%d")

    return {
        'model': rf,
        "best_params": best_params,
        "metrics": metrics,
        'final_features': final_features,
        'from': from_date,
        'to': to_date,
    }
