import json
import numpy as np
import pandas as pd
from ml.regestry import MODEL_REGISTRY, MULTI_REGISTRY
from ml.loaders import load_daily_sales
from ml.modeling import (
    get_series,
    result_to_json
)


def fit_all(branch_id: int, threshold: float = 0.1, freq: str = "D") -> pd.DataFrame:

    data   = load_daily_sales(branch_id)
    report_date = data['report_date'].max()

    series = get_series(data, "category", freq)

    filtered_series = {}
    for category, s in series.items():
        if len(s) == 0:
            continue
        near_zero_ratio = len(s[np.abs(s) < 10]) / len(s)
        if near_zero_ratio <= threshold:
            filtered_series[category] = s

    rows = []

    # Per-category models
    for category, s in filtered_series.items():
        for model_name, fit_fn in MODEL_REGISTRY.items():
            try:
                result = fit_fn(s, freq)
                rows.append({
                    "branch_id":      branch_id,
                    "category":       category,
                    "freq":           freq,
                    "model":          model_name,
                    "from":           result["from"],
                    "to":             result["to"],
                    "val_wape":      result["metrics"]["val_wape"],
                    "final_mae":      result["metrics"]["final_mae"],
                    "final_rmse":     result["metrics"]["final_rmse"],
                    "final_wape":     result["metrics"]["final_wape"],
                    "best_params":    result["best_params"],
                    "final_features": result.get("final_features"),
                    "test_pred":      result["test_pred"],
                    "forecast":       result["forecast"],
                    "model_obj":      result.get("model"),
                    "result":    result_to_json(model_name, result),
                })
            except Exception as e:
                rows.append({
                    "branch_id":  branch_id,
                    "category":   category,
                    "freq":       freq,
                    "model":      model_name,
                    "result":     json.dumps({"error": str(e)}),
                })

    # Multi-category models
    for model_name, fit_fn in MULTI_REGISTRY.items():
        try:
            multi_result = fit_fn(filtered_series, freq)
            for category, result in multi_result.items():
                rows.append({
                    "branch_id":      branch_id,
                    "category":       category,
                    "freq":           freq,
                    "model":          model_name,
                    "from":           result["from"],
                    "to":             result["to"],
                    "val_wape":      result["metrics"]["val_wape"],
                    "final_mae":      result["metrics"]["final_mae"],
                    "final_rmse":     result["metrics"]["final_rmse"],
                    "final_wape":     result["metrics"]["final_wape"],
                    "best_params":    result["best_params"],
                    "final_features": result.get("final_features"),
                    "test_pred":      result["test_pred"],
                    "forecast":       result["forecast"],
                    "model_obj":      None,
                    "result":    json.dumps({
                        "model": model_name,
                        "best_params": result["best_params"],
                        "metrics": result["metrics"],
                    }),
                })
        except Exception as e:
            import streamlit as st
            import traceback
            st.error(f"**{model_name} failed:** {e}")
            st.code(traceback.format_exc())
            for category in filtered_series:
                rows.append({
                    "branch_id":  branch_id,
                    "category":   category,
                    "freq":       freq,
                    "model":      model_name,
                    "result":     json.dumps({"error": str(e)}),
                })

    df = pd.DataFrame(rows)

    if df.empty or "final_wape" not in df.columns:
        return df

    df['report_date'] = report_date

    return df
