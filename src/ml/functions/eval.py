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
    data = data.loc[data['is_best'] == True, ['category', 'from', 'to', 'model', 'final_wape', 'val_wape']].copy()
    for idx, row in data.iterrows():
    #     st.markdown(f"Category: {row['category']}     From: {row['from']}     to: {row['to']}     best model: {row['model']}     final wape: {row['final_wape']}")

        with st.container(border=True):
            st.write(f"Category: {row['category']}")
            st.write(f"From: {row['from']}")
            st.write(f"To: {row['to']}")
            st.write(f"Best model: {row['model']}")
            st.write(f"Val WAPE: {row['val_wape']}")
            st.write(f"Final WAPE: {row['final_wape']}")