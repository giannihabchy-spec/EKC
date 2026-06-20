from datetime import datetime
import streamlit as st
import pandas as pd
from supa.db import get_pg_connection


def get_client_list(supabase):
    try:
        response = supabase.table("branches").select("outlet").execute()
        return [row["outlet"] for row in response.data]
    except Exception as e:
        st.error(f"Error fetching clients: {e}")
        return []

def get_period_options():
    return pd.period_range(
        start=datetime.now() - pd.DateOffset(years=2), 
        end=datetime.now(), 
        freq='M'
    ).strftime('%Y-%m').tolist()[::-1]

def get_client_list_for_daily_sales():
    try:
        with get_pg_connection() as conn:
            outlets = pd.read_sql("""
                SELECT DISTINCT b.outlet
                FROM branches b
                JOIN ac_daily_sales s
                  ON s.branch_id = b.id
                ORDER BY b.outlet;
            """, conn)

        return outlets["outlet"].tolist()

    except Exception as e:
        st.error(f"Error fetching clients: {e}")
        return []