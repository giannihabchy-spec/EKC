import numpy as np
from ml.modeling import get_series
from ml.loaders import load_daily_sales


def fit_all(branch_id, fit_func, threshold: float = 0.1) -> dict:
    data = load_daily_sales(branch_id)
    series = get_series(data,'category')

    results = {}
    for name, s in series.items():
        near_zero_ratio = len(s[np.abs(s) < 10]) / len(s)
        if near_zero_ratio > threshold:
            results[name] = {
                "status": "skipped",
                "reason": f"{near_zero_ratio:.1%} of values are near-zero (threshold: {threshold:.1%})",
            }
            continue

        try:
            result = fit_func(s)
            result["status"] = "ok"
            results[name] = result
        except Exception as e:
            results[name] = {"status": "error", "reason": str(e)}

    return results