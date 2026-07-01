import os
import json
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

from ml.modeling import get_series, _split
from ml.loaders import (
    load_daily_sales,
    _load_all_results,
    _load_results,
    _pick_best
)


def _plot(series: dict, results: dict, figsize: tuple = (16, 5)) -> None:
    for category, result in results.items():
        if category not in series:
            continue

        s = series[category]
        train, val, test = _split(s)
        train_val = s.loc[train.index.union(val.index)]

        model_name = result.get("model", "?")
        metrics = result.get("metrics", {})

        tp = result.get("test_pred", {})
        test_pred_idx = pd.to_datetime(tp.get("dates", []))
        test_pred = pd.Series(tp.get("values", []), index=test_pred_idx)

        fc = result.get("forecast", {})
        forecast_idx = pd.to_datetime(fc.get("dates", []))
        forecast = pd.Series(fc.get("values", []), index=forecast_idx)

        fig, ax = plt.subplots(figsize=figsize)

        ax.plot(train_val.index, train_val.values,
                color="#4C72B0", linewidth=1.2, label="Train (actual)")

        ax.plot(test.index, test.values,
                color="#55A868", linewidth=1.4, label="Test (actual)")

        ax.plot(test_pred.index, test_pred.values,
                color="#DD8452", linewidth=1.4, linestyle="--", label="Test (predicted)")

        if len(forecast) > 0:
            last_actual_date = test.index[-1]
            last_actual_value = test.iloc[-1]
            ax.plot(
                [last_actual_date, forecast.index[0]],
                [last_actual_value, forecast.iloc[0]],
                color="#C44E52", linewidth=1.2, linestyle="--",
            )

            ax.plot(forecast.index, forecast.values,
                    color="#C44E52", linewidth=1.4, linestyle="--", label="Forecast")

            ax.axvspan(forecast.index[0], forecast.index[-1],
                       alpha=0.06, color="#C44E52")
            ax.axvline(x=forecast.index[0], color="gray", linewidth=0.8, linestyle=":")

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

        mae = metrics.get("final_mae", "—")
        rmse = metrics.get("final_rmse", "—")
        wape = metrics.get("final_wape", "—")

        ax.set_title(
            f"{category}  |  {model_name.upper()}  —  "
            f"MAE: {mae}  |  RMSE: {rmse}  |  WAPE: {wape}%",
            fontsize=12, fontweight="bold",
        )
        ax.set_ylabel("Sales (USD)")
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.show()


def plot_by_model(branch_id: int, model: str, freq: str, figsize: tuple = (16, 5)) -> None:
    data = load_daily_sales(branch_id)
    series = get_series(data, "category", freq)
    results = _load_results(model, branch_id, freq)
    for entry in results.values():
        entry.setdefault("model", model)
    _plot(series, results, figsize)


def plot_best(branch_id: int, freq: str, figsize: tuple = (16, 5)) -> None:
    data = load_daily_sales(branch_id)
    series = get_series(data, "category", freq)
    all_results = _load_all_results(branch_id, freq)
    best = _pick_best(all_results)
    _plot(series, best, figsize)


def plot_by_category(branch_id: int, category: str, freq: str, figsize: tuple = (16, 5)) -> None:
    """
    Plot all available models for a single category of a branch.
    One plot per model, stacked vertically.
    """
    data    = load_daily_sales(branch_id)
    series  = get_series(data, "category", freq)

    if category not in series:
        print(f"Category '{category}' not found for branch {branch_id}.")
        return

    all_results = _load_all_results(branch_id, freq)

    if category not in all_results:
        print(f"No results found for category '{category}'.")
        return

    for model_name, result in all_results[category]:
        result.setdefault("model", model_name)
        _plot({category: series[category]}, {category: result}, figsize)
