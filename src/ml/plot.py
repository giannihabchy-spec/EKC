import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd


def plot_forecasts(forecast_output: dict, figsize: tuple = (16, 5)) -> None:
    """
    Plot one chart per category showing:
      - Train+val actuals (blue)
      - Test actuals (green)
      - Test predictions (orange dashed)
      - Future forecast (red dashed)

    Args:
        forecast_output: dict returned by forecast_daily_sales()
        figsize: (width, height) per subplot
    """
    for category, data in forecast_output.items():
        fig, ax = plt.subplots(figsize=figsize)

        train_val   = data["train_val"]
        test_actual = data["test_actual"]
        test_pred   = data["test_pred"]
        forecast    = data["forecast"]
        metrics     = data["metrics"]
        model_name  = data["model"]

        ax.plot(train_val.index, train_val.values,
                color="#4C72B0", linewidth=1.2, label="Train (actual)")

        ax.plot(test_actual.index, test_actual.values,
                color="#55A868", linewidth=1.4, label="Test (actual)")

        ax.plot(test_pred.index, test_pred.values,
                color="#DD8452", linewidth=1.4, linestyle="--", label="Test (predicted)")

        # connector line so the forecast doesn't float
        last_actual_date  = test_actual.index[-1]
        last_actual_value = test_actual.iloc[-1]
        ax.plot(
            [last_actual_date, forecast.index[0]],
            [last_actual_value, forecast.iloc[0]],
            color="#C44E52", linewidth=1.2, linestyle="--",
        )

        ax.plot(forecast.index, forecast.values,
                color="#C44E52", linewidth=1.4, linestyle="--", label="Forecast")

        # Shade the forecast region
        ax.axvspan(forecast.index[0], forecast.index[-1],
                   alpha=0.06, color="#C44E52")

        # Vertical separator between test and forecast
        ax.axvline(x=forecast.index[0], color="gray", linewidth=0.8, linestyle=":")

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

        mae  = metrics.get("final_mae", "—")
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
