import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ml.modeling import _split, _make_features


def _build_future_features(s: pd.Series, n_periods: int) -> pd.DataFrame:
    """
    Build a feature matrix for n_periods future dates using actual history for lags.
    Skips dropna — NaN lags (where future day index < lag size) are forward-filled
    from the last known value so tree models can consume them.
    """
    from ml.modeling import LAGS, ROLLS

    future_index = pd.date_range(
        start=s.index[-1] + pd.Timedelta(days=1),
        periods=n_periods,
        freq="D",
    )
    future_placeholder = pd.Series(np.nan, index=future_index)
    extended = pd.concat([s, future_placeholder])

    df = extended.rename("sales").to_frame()
    df["day_of_week"]  = df.index.dayofweek
    df["day_of_month"] = df.index.day
    df["month"]        = df.index.month
    df["year"]         = df.index.year
    df["weekofyear"]   = df.index.isocalendar().week.astype(int)
    df["is_weekend"]   = (df.index.dayofweek >= 5).astype(int)

    for lag in LAGS:
        df[f"lag_{lag}"] = df["sales"].shift(lag)

    for window in ROLLS:
        df[f"roll_mean_{window}"] = df["sales"].shift(1).rolling(window).mean()
        df[f"roll_std_{window}"]  = df["sales"].shift(1).rolling(window).std()

    future_feat = df.loc[future_index].drop(columns="sales")
    future_feat = future_feat.ffill().fillna(0)
    return future_feat


def _direct_forecast(s: pd.Series, model, feature_cols: list, n_periods: int = 60) -> pd.Series:
    future_index = pd.date_range(
        start=s.index[-1] + pd.Timedelta(days=1),
        periods=n_periods,
        freq="D",
    )
    x_future = _build_future_features(s, n_periods)[feature_cols]
    preds    = np.clip(model.predict(x_future), 0, None)
    return pd.Series(preds, index=future_index)


def _plot_one(ax, s, result, name, n_forecast_days):
    model          = result["model"]
    final_features = result["final_features"]

    train, val, test = _split(s)

    full_features = _make_features(s)
    test_feat     = full_features.loc[full_features.index.isin(test.index)]
    x_test        = test_feat[final_features]
    test_pred     = pd.Series(
        np.clip(model.predict(x_test), 0, None),
        index=x_test.index,
    )

    forecast = _direct_forecast(s, model, final_features, n_forecast_days)

    actuals = pd.concat([train, val])
    ax.plot(actuals.index, actuals.values, color="steelblue", alpha=0.5, label="train + val")
    ax.plot(test.index, test.values, color="green", label="test (actual)")
    ax.plot(test_pred.index, test_pred.values, color="orange", linestyle="--", label="test (predicted)")
    ax.plot(forecast.index, forecast.values, color="red", linestyle="--", label="forecast")

    ax.axvline(test.index[0], color="gray", linestyle=":", linewidth=1)
    ax.axvline(forecast.index[0], color="gray", linestyle=":", linewidth=1)

    mae  = result.get("final_mae", "")
    rmse = result.get("final_rmse", "")
    wape = result.get("final_wape", "")
    ax.set_title(f"{name}   |   MAE: {mae}   RMSE: {rmse}   WAPE: {wape}%")
    ax.legend(fontsize=8)


def plot_result(s: pd.Series, result: dict, name: str = "", n_forecast_days: int = 60):
    fig, ax = plt.subplots(figsize=(14, 4))
    _plot_one(ax, s, result, name, n_forecast_days)
    plt.tight_layout()
    plt.show()


def plot_all(series: dict, results: dict, n_forecast_days: int = 60):
    ok = [name for name, res in results.items() if res.get("status") == "ok"]

    if not ok:
        print("No fitted categories to plot.")
        return

    fig, axes = plt.subplots(len(ok), 1, figsize=(14, 4 * len(ok)))
    if len(ok) == 1:
        axes = [axes]

    for ax, name in zip(axes, ok):
        _plot_one(ax, series[name], results[name], name, n_forecast_days)

    plt.tight_layout()
    plt.show()
