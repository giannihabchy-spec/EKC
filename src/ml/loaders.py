import pandas as pd
from supa.db import get_pg_connection
import os
import json

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def load_daily_sales(branch_id: int) -> pd.DataFrame:
    """
    Pull ac_daily_sales for a branch and return a clean DataFrame:
        date (datetime), category, item_group, sales (float)
    Sorted by category / item_group / date, no nulls in key columns.
    """
    conn = get_pg_connection()
    try:
        query = """
            SELECT date, category, item_group, sales, currency, client_rate, report_date
            FROM public.ac_daily_sales
            WHERE branch_id = %s
            ORDER BY category, item_group, date
        """
        data = pd.read_sql(query, conn, params=(branch_id,))
    finally:
        conn.close()

    data["date"] = pd.to_datetime(data["date"])
    data["sales"] = pd.to_numeric(data["sales"], errors="coerce")
    data = data.dropna(subset=["date", "category", "item_group", "sales"])

    if (data["currency"] == "Unknown").any():
        raise ValueError("Unknown currency found")

    mask = data['currency'] != 'Usd'

    data['original_sales'] = data['sales']
    data.loc[mask, 'sales'] = data.loc[mask, 'original_sales'] / data.loc[mask, 'client_rate']

    return data


def _load_all_results(branch_id: int, freq: str) -> dict:
    filename = f"branch_{branch_id}_{freq}.json"
    all_results = {}

    if not os.path.isdir(RESULTS_DIR):
        return all_results

    for model_name in os.listdir(RESULTS_DIR):
        path = os.path.join(RESULTS_DIR, model_name, filename)
        if not os.path.isfile(path):
            continue
        with open(path, "r") as f:
            data = json.load(f)
        for category, entry in data.items():
            all_results.setdefault(category, []).append((model_name, entry))

    return all_results


def _load_results(model: str, branch_id: int, freq: str) -> dict:
    path = os.path.join(RESULTS_DIR, model, f"branch_{branch_id}_{freq}.json")
    if not os.path.isfile(path):
        raise ValueError(f"No results found at {path}")
    with open(path, "r") as f:
        return json.load(f)
    

def _pick_best(all_results: dict) -> dict:
    best = {}
    for category, entries in all_results.items():
        winner = min(
            entries,
            key=lambda e: e[1].get("metrics", {}).get("final_wape", float("inf")),
        )
        best[category] = {"model": winner[0], **winner[1]}
    return best