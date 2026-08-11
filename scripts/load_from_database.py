import pandas as pd
import streamlit as st
from supa.db import _ensure_supa_env_from_secrets

st.set_page_config(
    page_title="Load From Database",
    layout="wide",
    initial_sidebar_state="collapsed"
)

_ensure_supa_env_from_secrets()

from supa.config import SHEET_CONFIG
from supa.db import (
    get_pg_connection,
    init_supabase,
    get_branch_id,
    check_data_coverage
)
from supa.loaders import(
    extract_sheets_and_client,
    push_sheets,
    load_logs,
)
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


st.title("Load From Database")
st.markdown("---")

col1, col2, col3, col4 = st.columns(4)
with col1:
    uploaded_file = st.file_uploader("Upload Excel Report", type=["xlsx"], key="ptdb_upload")
with col2:
    client_options = get_client_list(supabase)
    selected_client = st.selectbox("Select Branch", options=client_options, key="ptdb_client")
    branch_id = get_branch_id(selected_client)['branch_id']
with col3:
    period_options = get_period_options()
    selected_period = st.selectbox("Select Reporting Period", options=period_options, key="ptdb_period")
with col4:
    all_available = check_data_coverage(branch_id, selected_client, selected_period)['result']
    data_choice = st.multiselect('Select Data to Load', options = all_available, default = all_available, key="ptdb_data")


if st.button("▶ Run", type="primary", use_container_width=True):


    if not uploaded_file or not selected_client or not selected_period or not data_choice:
        st.error("Please provide a file, a client, a date and the data to load.")
        st.stop()

    report_date = pd.to_datetime(selected_period)

    with st.status("Loading data...", expanded=True) as load_st:
        st.write(data_choice)
        data = load_logs(branch_id, selected_client, selected_period, data_choice)
        st.write(data)
        load_st.update(label="Processing Data", state="complete", expanded=True)






    st.success(f"Successfully loaded data to database.")