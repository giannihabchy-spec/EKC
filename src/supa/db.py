import pandas as pd
import os
from supabase import create_client as supabase_init
import psycopg2
import streamlit as st
from pathlib import Path
import tomllib


ROOT = Path(__file__).parents[2]
SECRETS_PATH = ROOT / ".streamlit" / "secrets.toml"
with SECRETS_PATH.open("rb") as f:
    secrets = tomllib.load(f)
for key in ["url", "key", "host", "port", "name", "user", "password"]:
    if key in secrets:
        os.environ[key] = str(secrets[key])

def init_supabase():
    url: str = os.getenv("url")
    key: str = os.getenv("key")
    return supabase_init(url, key)

supabase = init_supabase()


def get_pg_connection():
    return psycopg2.connect(
        host=os.getenv("host"),
        dbname=os.getenv("dbname", "postgres"),
        user=os.getenv("user", "postgres"),
        password=os.getenv("password"),
        port=os.getenv("port", "5432"),
        sslmode="require"
    )


def get_branch_id(branch_name):

    try:
        response = (
            supabase
            .table("branches")
            .select("id, outlet")
            .eq("outlet", branch_name)
            .execute()
        )
    except Exception as e:
        msg = f"⚠️ Failed to fetch client '{branch_name}' from clients table: {e}"
        return {
            "status": "error",
            "message": msg,
            "branch_id": None
        }

    rows = response.data if response and hasattr(response, "data") else []

    if not rows:
        msg = f"⚠️ Client '{branch_name}' was not found in the clients table"
        return {
            "status": "error",
            "message": msg,
            "branch_id": None
        }

    if len(rows) > 1:
        msg = f"⚠️ Multiple clients found for '{branch_name}' in the clients table"
        return {
            "status": "error",
            "message": msg,
            "branch_id": None
        }

    return {
            "status": "ok",
            "branch_id": rows[0]["id"]
        }


def get_branch_omega_name(branch_id):
    try:
        response = (
            supabase
            .table("branches")
            .select("id, omega_name")
            .eq("id", branch_id)
            .execute()
        )
    except Exception as e:
        msg = f"⚠️ Failed to fetch client's Omega name: {e}"
        return {
            "status": "error",
            "message": msg,
            "branch_id": None
        }

    rows = response.data if response and hasattr(response, "data") else []

    if not rows:
        msg = f"⚠️ Client was not found in the branches table"
        return {
            "status": "error",
            "message": msg,
            "branch_id": None
        }

    if len(rows) > 1:
        msg = f"⚠️ Multiple Omega names found for the client"
        return {
            "status": "error",
            "message": msg,
            "branch_id": None
        }

    return {
            "status": "ok",
            "omega_name": rows[0]["omega_name"]
        }


def _ensure_supa_env_from_secrets():
    mapping = {
        "SUPABASE_URL": "url",
        "SUPABASE_KEY": "key",
        "host":         "host",
        "name":         "dbname",   # secrets uses "name", psycopg2 expects "dbname"
        "user":         "user",
        "password":     "password",
        "port":         "port",
    }
    for secret_key, env_key in mapping.items():
        if os.getenv(env_key):
            continue
        val = st.secrets.get(secret_key)
        if val:
            os.environ[env_key] = str(val)


def get_omega_currency(branch_id):

    try:
        response = (
            supabase
            .table("branches")
            .select("id, omega_currency")
            .eq("id", branch_id)
            .execute()
        )
    except Exception as e:
        msg = f"⚠️ Failed to fetch client's omega_currency"
        return {
            "status": "error",
            "message": msg,
            "omega_currency": None
        }

    rows = response.data if response and hasattr(response, "data") else []

    if not rows:
        msg = f"⚠️ Client was not found in the clients table"
        return {
            "status": "error",
            "message": msg,
            "omega_currency": None
        }

    if len(rows) > 1:
        msg = f"⚠️ Multiple currencies found for the client"
        return {
            "status": "error",
            "message": msg,
            "omega_currency": None
        }

    return {
            "status": "ok",
            "omega_currency": rows[0]["omega_currency"]
        }


def get_monthly_rates():

    conn = get_pg_connection()

    try:
        data = pd.read_sql("select * from monthly_rate;", conn)

    finally:
        conn.close()

    return data


def get_last_table(branch_id, table_name):
    conn = get_pg_connection()

    try:
        query = f"""
            SELECT *
            FROM public.{table_name}
            WHERE branch_id = %s
            AND report_date = (
                SELECT MAX(report_date)
                FROM public.{table_name}
                WHERE branch_id = %s
            );
        """

        data = pd.read_sql(query, conn, params=(branch_id, branch_id))

    finally:   
        conn.close()

    return data


def get_branch_id_from_omega_name(omega_client):

    try:
        response = (
            supabase
            .table("branches")
            .select("id, omega_name")
            .contains("omega_name", [omega_client])
            .execute()
        )
    except Exception as e:
        msg = f"⚠️ Failed to fetch client '{omega_client}' from clients table: {e}"
        return {
            "status": "error",
            "message": msg,
            "branch_id": None
        }

    rows = response.data if response and hasattr(response, "data") else []

    if not rows:
        msg = f"⚠️ Client '{omega_client}' was not found in the clients table"
        return {
            "status": "error",
            "message": msg,
            "branch_id": None
        }

    if len(rows) > 1:
        msg = f"⚠️ Multiple clients found for '{omega_client}' in the clients table"
        return {
            "status": "error",
            "message": msg,
            "branch_id": None
        }

    return {
            "status": "ok",
            "branch_id": rows[0]["id"]
        }


def check_data_coverage(branch_id, selected_client, selected_period):
    conn = get_pg_connection()

    try:
        date = pd.to_datetime(selected_period)

        # Waste
        first_day = date.to_period("M").start_time
        last_day = (date.to_period("M") + 1).start_time
        waste_query = """
        SELECT count(*)
        FROM waste_logs
        WHERE outlet = %s
        AND date >= %s
        AND date < %s
        """
        waste = pd.read_sql(
            waste_query, 
            conn, 
            params = (selected_client, first_day, last_day))


        # Inventory
        first_day = date.to_period("M").end_time - pd.Timedelta(days=3)
        last_day = (date.to_period("M") + 1).start_time + pd.Timedelta(days=4)
        inv_query = """
        SELECT count(*)
        FROM inventory_logs
        WHERE outlet = %s
        AND date >= %s
        AND date < %s
        """
        inventory = pd.read_sql(
            inv_query,
            conn,
            params=(selected_client, first_day, last_day)
        )


        # Transfers
        first_day = date.to_period("M").start_time
        last_day = (date.to_period("M") + 1).start_time
        transfers_query = """
        SELECT count(*)
        FROM transfers
        WHERE date::timestamp >= %s
        AND date::timestamp < %s
        AND (
            from_outlet = %s
            OR to_outlet = %s
        )
        """
        transfers = pd.read_sql(
            transfers_query,
            conn,
            params=(first_day, last_day, selected_client, selected_client)
        )


        # Production
        first_day = date.to_period("M").start_time
        last_day = (date.to_period("M") + 1).start_time
        prod_query = """
        SELECT count(*)
        FROM production_log
        WHERE log_date >= %s
        AND log_date < %s
        AND branch_id = %s
        """

        production = pd.read_sql(
            prod_query,
            conn,
            params=(first_day, last_day, branch_id)
        )

        # Purchase    
        first_day = date.to_period("M").start_time
        last_day = (date.to_period("M") + 1).start_time
        query = """
        SELECT count(*)
        FROM purchase_logs
        WHERE outlet = %s
        AND invoice_date >= %s
        AND invoice_date < %s
        """
        purchase = pd.read_sql(
            query,
            conn,
            params=(selected_client, first_day, last_day)
        )

    finally:
        conn.close()

    data = {
        'waste': int(waste.iloc[0,0]),
        'inventory': int(inventory.iloc[0,0]),
        'transfers': int(transfers.iloc[0,0]),
        'production': int(production.iloc[0,0]),
        'purchase': int(purchase.iloc[0,0]),
    }

    return [f'{k}: {v} rows' for k, v in data.items() if v != 0]