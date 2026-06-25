import os
import json

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
