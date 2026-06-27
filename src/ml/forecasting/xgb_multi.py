import calendar
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.inspection import permutation_importance
from ml.modeling import _split, _add_holiday_features
from ml.config import xgb_tuning, D_LAGS, W_LAGS, D_ROLLS, W_ROLLS


def _make_multi_features(sales_df: pd.DataFrame, freq: str) -> pd.DataFrame:
    df = sales_df.copy()

    if freq == "D":
        df["day_of_week"] = df.index.dayofweek
        df["day_of_month"] = df.index.day
        df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)

    df["month"] = df.index.month
    df["year"] = df.index.year
    df["week_of_month"] = ((df.index.day - 1) // 7 + 1)
    df["weekofyear"] = df.index.isocalendar().week.astype(int)

    lags = D_LAGS if freq == "D" else W_LAGS
    rolls = D_ROLLS if freq == "D" else W_ROLLS

    for cat in sales_df.columns:
        for lag in lags:
            df[f"{cat}_lag_{lag}"] = sales_df[cat].shift(lag)
        for window in rolls:
            df[f"{cat}_roll_mean_{window}"] = sales_df[cat].shift(1).rolling(window).mean()
            df[f"{cat}_roll_std_{window}"] = sales_df[cat].shift(1).rolling(window).std()

    df = _add_holiday_features(df, freq)

    feature_cols = [c for c in df.columns if c not in sales_df.columns]
    df = df.dropna(subset=feature_cols)
    return df


def _recursive_forecast_multi(models, features_map, sales_df, categories, freq, months=8):
    step = pd.Timedelta(days=1) if freq == "D" else pd.Timedelta(weeks=1)
    max_date = sales_df.index.max()
    end_month = max_date.month + months
    end_year = max_date.year + (end_month - 1) // 12
    end_month = (end_month - 1) % 12 + 1
    last_day = calendar.monthrange(end_year, end_month)[1]
    horizon_end = pd.Timestamp(year=end_year, month=end_month, day=last_day)
    future_dates = pd.date_range(start=max_date + step, end=horizon_end, freq=freq)

    history = sales_df.copy()
    forecast_values = {cat: [] for cat in categories}

    for d in future_dates:
        feat_df = _make_multi_features(history, freq)
        last_row = feat_df.iloc[[-1]]

        new_row = {}
        for cat in categories:
            cat_features = features_map[cat]
            pred = max(0.0, round(float(models[cat].predict(last_row[cat_features])[0])))
            new_row[cat] = pred
            forecast_values[cat].append(pred)

        history.loc[d] = new_row

    return {
        cat: {
            "dates": [d.strftime("%Y-%m-%d") for d in future_dates],
            "values": forecast_values[cat],
        }
        for cat in categories
    }


def fit_single_xgb(series: dict, freq: str = "D") -> dict:
    categories = sorted(series.keys())

    ref = series[categories[0]]
    index = ref.index

    sales_df = pd.DataFrame({cat: series[cat] for cat in categories}, index=index).fillna(0)

    n = len(index)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)
    n_train_val = n_train + n_val

    train_end = index[n_train - 1]
    val_end = index[n_train_val - 1]

    full_features = _make_multi_features(sales_df, freq)
    feature_cols = [c for c in full_features.columns if c not in categories]

    train_data = full_features.loc[full_features.index <= train_end]
    val_data = full_features.loc[(full_features.index > train_end) & (full_features.index <= val_end)]
    test_data = full_features.loc[full_features.index > val_end]

    x_train = train_data[feature_cols]
    x_val = val_data[feature_cols]
    x_test = test_data[feature_cols]

    models = {}
    features_map = {}
    output = {}

    for cat in categories:
        y_train = train_data[cat]
        y_val = val_data[cat]
        y_test = test_data[cat]

        best_wape = float("inf")
        best_params = None

        for n_est in xgb_tuning["n_est"]:
            for lr in xgb_tuning["learning_rate"]:
                for max_depth in xgb_tuning["max_depth"]:
                    for subsample in xgb_tuning["subsample"]:
                        for colsample in xgb_tuning["colsample_bytree"]:

                            model = XGBRegressor(
                                n_estimators=n_est,
                                learning_rate=lr,
                                max_depth=max_depth,
                                subsample=subsample,
                                colsample_bytree=colsample,
                                objective="reg:squarederror",
                                random_state=42,
                                n_jobs=-1,
                                tree_method="hist",
                            )

                            model.fit(x_train, y_train)
                            pred = model.predict(x_val)

                            denom = np.sum(np.abs(y_val))
                            if denom == 0:
                                continue

                            wape = np.sum(np.abs(y_val - pred)) / denom

                            if wape < best_wape:
                                best_wape = wape
                                best_params = {
                                    "n_estimators": n_est,
                                    "learning_rate": lr,
                                    "max_depth": max_depth,
                                    "subsample": subsample,
                                    "colsample_bytree": colsample,
                                }

        if best_params is None:
            raise ValueError(f"No valid configuration found for {cat}")

        model = XGBRegressor(
            **best_params,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
            tree_method="hist",
        )

        model.fit(x_train, y_train)
        pred = model.predict(x_val)

        result = permutation_importance(
            model, x_val, y_val,
            scoring="neg_mean_absolute_error",
            n_repeats=10,
            random_state=42,
            n_jobs=-1,
        )

        imp = pd.Series(result.importances_mean, index=x_val.columns).sort_values(ascending=False)
        selected_features = imp[imp > 0].index.to_list()
        if len(selected_features) == 0:
            selected_features = feature_cols

        model.fit(x_train[selected_features], y_train)
        new_pred = model.predict(x_val[selected_features])

        wape = np.sum(np.abs(y_val - pred)) / np.sum(np.abs(y_val))
        new_wape = np.sum(np.abs(y_val - new_pred)) / np.sum(np.abs(y_val))

        if new_wape <= wape:
            final_features = selected_features
            val_wape = round(new_wape * 100, 2)
        else:
            final_features = feature_cols
            val_wape = round(wape * 100, 2)

        x_tv = pd.concat([x_train, x_val])[final_features]
        y_tv = pd.concat([y_train, y_val])

        model.fit(x_tv, y_tv)
        final_pred = model.predict(x_test[final_features])

        final_mae = round(float(mean_absolute_error(y_test, final_pred)), 2)
        final_rmse = round(float(np.sqrt(np.mean((y_test.values - final_pred) ** 2))), 2)
        denom = np.sum(np.abs(y_test))
        final_wape = round(float(np.sum(np.abs(y_test - final_pred)) / denom * 100), 2) if denom > 0 else 0.0

        test_pred = {
            "dates": [d.strftime("%Y-%m-%d") for d in y_test.index],
            "values": [round(float(v), 2) for v in final_pred],
        }

        x_full = full_features[final_features]
        y_full = full_features[cat]
        model.fit(x_full, y_full)

        models[cat] = model
        features_map[cat] = final_features

        output[cat] = {
            "best_params": best_params,
            "metrics": {
                "val_wape": val_wape,
                "final_mae": final_mae,
                "final_rmse": final_rmse,
                "final_wape": final_wape,
            },
            "final_features": final_features,
            "test_pred": test_pred,
            "from": index.min().strftime("%Y-%m-%d"),
            "to": index.max().strftime("%Y-%m-%d"),
        }

    # Forecast with all models together (cross-category recursive)
    forecasts = _recursive_forecast_multi(models, features_map, sales_df, categories, freq)

    for cat in categories:
        output[cat]["forecast"] = forecasts[cat]

    return output
