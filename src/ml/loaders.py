import pandas as pd
from supa.db import get_pg_connection


def load_daily_sales(branch_id: int) -> pd.DataFrame:
    """
    Pull ac_daily_sales for a branch and return a clean DataFrame:
        date (datetime), category, item_group, sales (float)
    Sorted by category / item_group / date, no nulls in key columns.
    """
    conn = get_pg_connection()
    query = f"""
        SELECT date, category, item_group, sales
        FROM public.ac_daily_sales
        WHERE branch_id = {branch_id}
        ORDER BY category, item_group, date
    """
    df = pd.read_sql(query, conn, params=(branch_id,))
    conn.close()

    df["date"] = pd.to_datetime(df["date"])
    df["sales"] = pd.to_numeric(df["sales"], errors="coerce")
    df = df.dropna(subset=["date", "category", "item_group", "sales"])

    return df



