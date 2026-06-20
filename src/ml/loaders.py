import pandas as pd
from supa.db import get_pg_connection


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



