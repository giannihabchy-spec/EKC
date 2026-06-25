from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error
)
import numpy as np
import streamlit as st


def evaluate_model(model, val):
    forecast = model.predict(n_periods=len(val))

    mae = mean_absolute_error(val, forecast)
    rmse = np.sqrt(mean_squared_error(val, forecast))
    mape = mean_absolute_percentage_error(val, forecast)

    return {
        "forecast": forecast,
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape
    }


def compute_metrics(actual, forecast):
    """Generic metric computation — works with any model (XGBoost, Prophet, etc.)"""
    actual = np.array(actual)
    forecast = np.array(forecast)

    mae  = mean_absolute_error(actual, forecast)
    rmse = np.sqrt(mean_squared_error(actual, forecast))
    mape = mean_absolute_percentage_error(actual, forecast)

    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


def display_results(data):
    display_cols = ['category', 'model', 'final_wape', 'val_wape']
    cols = [c for c in display_cols if c in data.columns]
    view = data[cols].copy()

    has_errors = 'final_wape' in data.columns and data['final_wape'].isna().any()
    has_errors = has_errors or 'final_wape' not in data.columns

    for category, group in view.groupby('category'):
        with st.container(border=True):
            st.write(f"**{category}**")
            show = group.drop(columns='category').reset_index(drop=True)
            if 'final_wape' in show.columns and show['final_wape'].isna().all():
                st.error("All models failed")
            else:
                st.dataframe(show, use_container_width=True)