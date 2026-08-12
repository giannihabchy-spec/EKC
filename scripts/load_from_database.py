import pandas as pd
import streamlit as st
from pathlib import Path
from supa.db import _ensure_supa_env_from_secrets

st.set_page_config(
    page_title="Load From Database",
    layout="wide",
    initial_sidebar_state="collapsed"
)

_ensure_supa_env_from_secrets()

from etl.special_characters import special_char
from etl.strip_all import strip_all
from etl.saver import save_cleaned_data

from supa.config import SHEET_CONFIG
from supa.db import (
    get_pg_connection,
    init_supabase,
    get_branch_id,
    check_data_coverage
)
from supa.loaders import(
    extract_info,
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
    convert_sheet_names
)
from supa.validators import (
    validate_required_columns,
    validate_client_name,
    validate_selected_date,
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
with col1:
    folder_input = st.text_input("Location to save the data (Optional)", placeholder="C:/Path/To/Folder")
    destination = Path(folder_input).resolve() if folder_input.strip() else None



if st.button("▶ Run", type="primary", use_container_width=True):


    if not uploaded_file or not selected_client or not selected_period or not data_choice:
        st.error("Please provide a file, a client, a date and the data to load.")
        st.stop()

    report_date = pd.to_datetime(selected_period)


    with st.status("Extracting Info...", expanded=True) as extract_st:
        file_date, file_client_name, currency, rate, info = extract_info(uploaded_file)
        if info['status'] != 'ok':
            st.error(info['msg'])
            extract_st.update(label="Extracting Info", state="error", expanded=True)
            st.stop()
        st.write(info['msg'])

        extract_st.update(label="Extracting Info", state="complete", expanded=True)


    with st.status("Validating Client and Date...", expanded=True) as val_st:
        client_res = validate_client_name(file_client_name, selected_client)
        if client_res["status"] != "ok":
            st.write(client_res["message"])
            val_st.update(label="Validating Client and Date", state="error", expanded=True)
            st.stop()
        st.write(client_res["message"])

        date_res = validate_selected_date(file_date, selected_period)
        if date_res["status"] != "ok":
            st.write(date_res["msg"])
            val_st.update(label="Validating Client and Date", state="error", expanded=True)
            st.stop()
        st.write(date_res["msg"])

        val_st.update(label="Validating Client and Date", state="complete", expanded=True)


    with st.status("Loading Data...", expanded=True) as load_st:
        data = load_logs(branch_id, selected_client, selected_period, data_choice)
        st.write('Data Loaded')
        data = strip_all(data)
        data = special_char(data)

        if destination is not None:
            save_cleaned_data(data, destination, f"{selected_client} logs.xlsx")
            st.write('Data Saved')

        load_st.update(label="Loading Data", state="complete", expanded=True)


    with st.status("Validating Workbook...", expanded=True) as valw_st:

        data = convert_sheet_names(data)
        data = normalize_all_dataframes(data)
        sht_st = validate_required_columns(data, SHEET_CONFIG)
        if sht_st['status'] != 'ok':
            st.write(sht_st['message'])
            extract_st.update(label="Validating Workbook", state="error", expanded=True)
            st.stop()
        st.write(sht_st['message'])

        valw_st.update(label="Validating Client and Date", state="complete", expanded=True)
