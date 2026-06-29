import os
import json
import re
import pandas as pd

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def _result_path(model, branch_id, freq):
    return os.path.join(RESULTS_DIR, model, f"branch_{branch_id}_{freq}.json")


def delete_existing_results(branch_id, data, freq):
    results_df = data["results"]
    categories = results_df["category"].unique()
    deleted = []

    for model in results_df["model"].unique():
        path = _result_path(model, branch_id, freq)
        if not os.path.exists(path):
            continue

        with open(path, "r") as f:
            existing = json.load(f)

        removed = [cat for cat in categories if cat in existing]
        for cat in removed:
            del existing[cat]

        if existing:
            with open(path, "w") as f:
                json.dump(existing, f, indent=2)
        else:
            os.remove(path)

        if removed:
            deleted.append(f"{model}: {', '.join(removed)}")

    if deleted:
        return {"status": "ok", "msg": "Deleted: " + " | ".join(deleted)}

    return {"status": "ok", "msg": "No existing data found"}


def save_results(data):
    results_df = data["results"]
    saved = 0

    for model, group in results_df.groupby("model"):
        branch_id = group["branch_id"].iloc[0]
        freq = group["freq"].iloc[0]
        path = _result_path(model, branch_id, freq)

        os.makedirs(os.path.dirname(path), exist_ok=True)

        if os.path.exists(path):
            with open(path, "r") as f:
                existing = json.load(f)
        else:
            existing = {}

        for _, row in group.iterrows():
            metric_keys = ["val_wape", "final_mae", "final_rmse", "final_wape"]
            has_metrics = any(
                k in row and row.get(k) is not None and not (isinstance(row.get(k), float) and row.get(k) != row.get(k))
                for k in metric_keys
            )
            if not has_metrics:
                continue

            entry = {}
            if "best_params" in row and row["best_params"] is not None:
                entry["best_params"] = row["best_params"]
            entry["metrics"] = {
                k: row[k] for k in metric_keys
                if k in row and row.get(k) is not None
            }
            if "final_features" in row and row["final_features"] is not None:
                entry["final_features"] = row["final_features"]
            if "test_pred" in row and row["test_pred"] is not None:
                entry["test_pred"] = row["test_pred"]
            if "forecast" in row and row["forecast"] is not None:
                entry["forecast"] = row["forecast"]
            if "from" in row:
                entry["from"] = row["from"]
            if "to" in row:
                entry["to"] = row["to"]

            existing[row["category"]] = entry
            saved += 1

        with open(path, "w") as f:
            json.dump(existing, f, indent=2)

    return {"status": "ok", "message": f"{saved} result(s) saved"}


def get_fitted(results):
    if "final_wape" not in results.columns:
        return {"categories": [], "models": [], "count": 0}
    fitted = results.dropna(subset=["final_wape"])
    categories = fitted["category"].unique().tolist()
    models = fitted["model"].unique().tolist()
    return {"categories": categories, "models": models, "count": len(fitted)}


def results_to_df_dict(
        selected_branch_ids: list[int] | None = None,
        selected_categories: list[str] | None = None,
        selected_freqs: list[str] | None = None,
        selected_models: list[str] | None = None,
):
    forecast_rows = []
    test_pred_rows = []

    if not os.path.exists(RESULTS_DIR):
        return {
            "forecasts": pd.DataFrame(columns=["branch_id", "category", "freq", "model", "date", "sales", "final_wape"]),
            "test_pred": pd.DataFrame(columns=["branch_id", "category", "freq", "model", "date", "sales", "final_wape"]),
        }

    for model_name in os.listdir(RESULTS_DIR):
        model_dir = os.path.join(RESULTS_DIR, model_name)
        if not os.path.isdir(model_dir):
            continue

        for filename in os.listdir(model_dir):
            match = re.match(r"branch_(\d+)_([DW])\.json$", filename)
            if not match:
                continue

            branch_id = int(match.group(1))
            freq = match.group(2)

            with open(os.path.join(model_dir, filename), "r") as f:
                data = json.load(f)

            for category, entry in data.items():
                metrics = entry.get("metrics", {})
                final_wape = metrics.get("final_wape")

                base = {
                    "branch_id": branch_id,
                    "category": category,
                    "freq": freq,
                    "model": model_name,
                    "final_wape": final_wape,
                }

                forecast = entry.get("forecast")
                if forecast and "dates" in forecast and "values" in forecast:
                    for date, sales in zip(forecast["dates"], forecast["values"]):
                        forecast_rows.append({**base, "date": date, "sales": sales})

                test_pred = entry.get("test_pred")
                if test_pred and "dates" in test_pred and "values" in test_pred:
                    for date, sales in zip(test_pred["dates"], test_pred["values"]):
                        test_pred_rows.append({**base, "date": date, "sales": sales})

    cols = ["branch_id", "category", "freq", "model", "date", "sales", "final_wape"]
    forecasts = pd.DataFrame(forecast_rows, columns=cols)
    test_pred = pd.DataFrame(test_pred_rows, columns=cols)


    if selected_branch_ids:
        forecasts = forecasts[forecasts['branch_id'].isin(selected_branch_ids)].copy()
        test_pred = test_pred[test_pred['branch_id'].isin(selected_branch_ids)].copy()

    if selected_categories:
        forecasts = forecasts[forecasts['category'].isin(selected_categories)].copy()
        test_pred = test_pred[test_pred['category'].isin(selected_categories)].copy()

    if selected_freqs:
        forecasts = forecasts[forecasts['freq'].isin(selected_freqs)].copy()
        test_pred = test_pred[test_pred['freq'].isin(selected_freqs)].copy()

    if selected_models:
        forecasts = forecasts[forecasts['model'].isin(selected_models)].copy()
        test_pred = test_pred[test_pred['model'].isin(selected_models)].copy()


    return {
        "forecasts": forecasts,
        "test_pred": test_pred,
    }
