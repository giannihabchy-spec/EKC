import calendar
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from ml.modeling import _make_features
from ml.config import SEASONAL_PERIOD



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


def _pick_best(all_results: dict) -> dict:
    best = {}
    for category, entries in all_results.items():
        winner = min(
            entries,
            key=lambda e: e[1].get("metrics", {}).get("final_wape", float("inf")),
        )
        best[category] = winner
    return best


def recursive_forecast(model, s: pd.Series, final_features: list, freq: str, months: int = 8) -> dict:
    step = pd.Timedelta(days=1) if freq == "D" else pd.Timedelta(weeks=1)
    max_date = s.index.max()
    future_dates = pd.date_range(
        start=max_date + step,
        end=_horizon_end(max_date, months),
        freq=freq,
    )

    history = s.copy()
    values = []
    for d in future_dates:
        feat_df = _make_features(history, freq)
        last_row = feat_df.iloc[[-1]][final_features]
        pred = max(0.0, round(float(model.predict(last_row)[0])))
        history[d] = pred
        values.append(pred)

    return {
        "dates": [d.strftime("%Y-%m-%d") for d in future_dates],
        "values": values,
    }

