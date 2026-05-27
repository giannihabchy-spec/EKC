import streamlit as st
import sys
from pathlib import Path
import warnings
import pandas as pd
from supa.db import _ensure_supa_env_from_secrets

st.set_page_config(
    page_title="Auto Calc Pipeline",
    layout="wide",
    initial_sidebar_state="collapsed"
)

_ensure_supa_env_from_secrets()


from etl.config import get_jobs, no_nulls
from etl.orchestrator import clean_folder, cleaner_by_code
from etl.merger import merge
from etl.strip_all import strip_all
from etl.special_characters import special_char
from etl.saver import save_cleaned_data
from etl.reset_view import reset_workbook_view
from etl.prev_unit_cost import uc_pre_month
from etl.clearer import clear_all, clear_junk_rows
from etl.clear_sheets import clear_sheets
from etl.writer import write_master
from etl.validators import check_sheets_exist, get_missing_columns
from etl.locate_cols import get_excel_cols
from etl.extract_sheets import extract_sheets
from etl.quick_variance.orchestrator import adjust_cleaners

from supa.config import SHEET_CONFIG
from supa.db import get_pg_connection, init_supabase, get_branch_id
from supa.loaders import extract_sheets_and_client, push_sheets
from supa.streamlit_functions import get_client_list, get_period_options
from supa.modeling import (
    normalize_all_dataframes,
    add_metadata,
    convert_date_columns,
    apply_grouping,
    normalize_string_columns,
    clean_numeric_values,
)
from supa.validators import (
    validate_required_columns,
    validate_client_name,
    validate_report_period,
    find_existing_data,
    delete_existing_data,
    check_duplicates,
    check_rows
)

st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

warnings.filterwarnings(
    "ignore",
    message="Workbook contains no default style*",
    category=UserWarning,
)

st.title("Quick Variance")
st.markdown("---")


if "ptdb_supabase_client" not in st.session_state:
    st.session_state.ptdb_supabase_client = init_supabase()
supabase = st.session_state.ptdb_supabase_client


