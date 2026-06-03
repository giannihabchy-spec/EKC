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


def get_pg_connection():
    return psycopg2.connect(
        host=os.getenv("host"),
        dbname=os.getenv("dbname", "postgres"),
        user=os.getenv("user", "postgres"),
        password=os.getenv("password"),
        port=os.getenv("port", "5432"),
        sslmode="require"
    )


def init_supabase():
    url: str = os.getenv("url")
    key: str = os.getenv("key")
    return supabase_init(url, key)


def get_branch_id(branch_name, supabase):

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


def get_branch_omega_name(branch_id, supabase):
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


def get_omega_currency(branch_id, supabase):

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
    data = pd.read_sql("select * from monthly_rate;", conn)
    conn.close()

    return data