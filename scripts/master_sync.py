import streamlit as st
import sys
from pathlib import Path
import warnings
import pandas as pd
from supa.db import _ensure_supa_env_from_secrets

st.set_page_config(
    page_title="Master Sync",
    layout="wide",
    initial_sidebar_state="collapsed"
)

from etl.config import get_jobs, no_nulls
from etl.orchestrator import clean_folder, cleaner_by_code
from etl.merger import merge_disc, merge_ib
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
from etl.extract_sheets import extract_sheets, sheets_to_extract

from supa.config import SHEET_CONFIG
from supa.db import get_pg_connection, init_supabase, get_branch_id, get_locations
from supa.loaders import extract_sheets_and_client, push_sheets
from supa.streamlit_functions import get_client_list, get_period_options
from supa.modeling import (
    normalize_all_dataframes,
    add_metadata,
    convert_date_columns,
    apply_grouping,
    normalize_string_columns,
    clean_numeric_values,
    create_sales_category,
)
from supa.validators import (
    validate_required_columns,
    validate_client_name,
    validate_report_period,
    find_existing_data,
    delete_existing_data,
    check_duplicates,
    check_rows,
    validate_currency_rate
)

if "ptdb_supabase_client" not in st.session_state:
    st.session_state.ptdb_supabase_client = init_supabase()
supabase = st.session_state.ptdb_supabase_client

st.title("Maser Sync")
st.markdown("---")

col1, col2, col3, col4 = st.columns(4)
with col1:
    folder_input = st.text_input("📁 Target Folder Path", placeholder="C:/Path/To/Folder")
with col2:
    client_options = get_client_list(supabase)
    selected_client = st.selectbox("Select Branch", options=client_options, key="ptdb_client")
    branch_id = get_branch_id(selected_client)['branch_id']
with col3:
    source = st.selectbox("🔀 Source", options=["cloud", "local"], index=0)
with col4:
    all_locations = get_locations(branch_id)
    locations = st.multiselect('Select Locations', options = all_locations, default = all_locations, key="ptdb_data")


if st.button("▶ Run", type="primary", use_container_width=True):


    if not folder_input:
        st.error("Please provide a folder location")
        st.stop()
    if not selected_client:
        st.error("Please select a client")
        st.stop()
    if not source:
        st.error("Please select a source")
        st.stop()
    if not locations:
        st.error("Please select the location(s)")
        st.stop()

    base_folder = Path(folder_input).resolve()
    jobs = get_jobs(source)

    if not base_folder.is_dir():
        st.error(f"Error: '{base_folder}' is not a valid directory.")