import streamlit as st
import sys
from pathlib import Path
import warnings
import pandas as pd
from supa.db import _ensure_supa_env_from_secrets

st.set_page_config(
    page_title="Quick Variance",
    layout="wide",
    initial_sidebar_state="collapsed"
)

_ensure_supa_env_from_secrets()


from etl.config import get_jobs
from etl.orchestrator import clean_folder, cleaner_by_code
from etl.merger import merge
from etl.strip_all import strip_all
from etl.special_characters import special_char
from etl.saver import save_cleaned_data
from etl.quick_variance.orchestrator import adjust_cleaners

from supa.config import SHEET_CONFIG
from supa.db import get_pg_connection, init_supabase, get_branch_id, get_branch_omega_name
from supa.loaders import push_sheets
from supa.streamlit_functions import get_client_list, get_period_options
from supa.modeling import (
    normalize_all_dataframes,
    add_metadata,
    apply_grouping,
    normalize_string_columns,
    clean_numeric_values,
    adjust_configs,
    add_quick_variance,
    convert_sheet_names_in_dict,
)
from supa.validators import (
    find_existing_data,
    delete_existing_data,
    validate_file_dates,
    validate_omega_name,
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


col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    folder_input = st.text_input("📁 Target Folder Path", placeholder="C:/Path/To/Folder")
with col2:
    source = st.selectbox("🔀 Source", options=["cloud", "local"], index=0)
with col3:
    client_options = get_client_list(supabase)
    selected_client = st.selectbox("Select Branch", options=client_options, key="ptdb_client")
with col4:
    period_options = get_period_options()
    selected_period = st.selectbox("Select Reporting Period", options=period_options, key="ptdb_period")
with col5:
    mode = st.selectbox("Select Mode", options=["Do not overwrite", "Overwrite"], index=0, key="ptdb_mode")

if st.button("▶ Run", type="primary", use_container_width=True):

    cleaner_by_code = adjust_cleaners()

    if folder_input.strip() =='':
        with st.status("Name Patterns", expanded=True) as status_pat_0:
            lines = []
            for i, j in cleaner_by_code[source].items():
                lines.append(f"{j[0]} {'-'*((70-len(j[0]))-2)} {i}")
                
            st.code("\n".join(lines), language=None)
            status_pat_0.update(state="complete",expanded=True)
            st.stop()
    
    base_folder = Path(folder_input).resolve()

    if not base_folder.is_dir():
        st.error(f"Error: '{base_folder}' is not a valid directory.")
        st.stop()

    with st.status("Initializing ETL...", expanded=True) as status_init:
        st.write(f"Folder: `{base_folder.name}`")
        status_init.update(label="Initialization", state="complete", expanded=True)

    with st.status("Name Patterns", expanded=True) as status_pat:
        lines = []
        folder_files = [f.name for f in base_folder.iterdir() if f.is_file()]
        for i, j in cleaner_by_code[source].items():
            emoji = "✅" if any(i in f for f in folder_files) else "❌"
            lines.append(f"{j[0]} {emoji} {'-'*((70-len(j[0]))-2)} {i}")

        st.code("\n".join(lines), language=None)
        status_pat.update(expanded=False)


    with st.status("Extracting Info...", expanded=True) as status_ext_info:

        qr_res = get_branch_id(selected_client)
        if qr_res["status"] != "ok":
            st.write(qr_res["message"])
            status_ext_info.update(label="Extracting Info", state="error", expanded=True)
            st.stop()
        branch_id = qr_res["branch_id"]
        omega_name = get_branch_omega_name(branch_id)['omega_name'] #######################################
        report_date = pd.to_datetime(selected_period)
        jobs = get_jobs(source)

        st.write('Done')

        status_ext_info.update(label="Extracting Info", state="complete", expanded=True)


    with st.status("Cleaning...", expanded=True) as status_clean:
        cleaned = clean_folder(base_folder, source=source, log_func=st.write, patterns=cleaner_by_code, quick_variance=True)

        if not cleaned:
            st.error("Make sure the selected source is correct")
            status_clean.update(label='Empty folder',state="error", expanded=True)
            st.stop()

        cleaned = merge(cleaned)
        cleaned = strip_all(cleaned)
        cleaned = special_char(cleaned)
        save_cleaned_data(cleaned, base_folder)
        st.write("Cleaned data is saved.")
        status_clean.update(label="Cleaning", state="complete", expanded=True)


    with st.status("Validating...", expanded=True) as status_validation:
        date_validation = validate_file_dates(cleaned, report_date)
        if date_validation['status'] != 'ok':
            st.error(date_validation['msg'])
            status_validation.update(label="Validating", state="error", expanded=True)
            st.stop()
        st.write(date_validation['msg'])

        name_validation = validate_omega_name(cleaned, omega_name) #############################################
        if name_validation['status'] != 'ok':
            st.error(name_validation['msg'])
            status_validation.update(label="Validating", state="error", expanded=True)
            st.stop()
        st.write(name_validation['msg']) #######################################################################

        status_validation.update(label="Validating", state="complete", expanded=True)


    with st.status("Formatting Data...", expanded=True) as form_st:
        SHEET_CONFIG = adjust_configs(SHEET_CONFIG)
        cleaned = normalize_all_dataframes(cleaned)
        cleaned = convert_sheet_names_in_dict(cleaned, jobs)

        norm_res = normalize_string_columns(cleaned)
        if norm_res["status"] != "ok":
            st.write(norm_res["message"])
            form_st.update(label="Formatting Data", state="error", expanded=True)
            st.stop()
        cleaned = norm_res["data"]
        cleaned = add_quick_variance(cleaned)
        st.write(norm_res["message"])

        grp_res = apply_grouping(cleaned, SHEET_CONFIG)
        if grp_res["status"] != "ok":
            st.write(grp_res["message"])
            form_st.update(label="Formatting Data", state="error", expanded=True)
            st.stop()
        cleaned = grp_res["data"]
        st.write(grp_res["message"])

        meta_res = add_metadata(cleaned, branch_id, selected_period, 'Unknown', 0)
        if meta_res["status"] != "ok":
            st.write(meta_res["message"])
            form_st.update(label="Formatting Data", state="error", expanded=True)
            st.stop()
        cleaned = meta_res["data"]
        st.write(meta_res["message"])

        cleaned = clean_numeric_values(cleaned)
        # save_cleaned_data(cleaned, base_folder, 'very cleaned data.xlsx')
        
        form_st.update(label="Formatting Data", state="complete", expanded=True)

    
    with st.status("Checking existing data...", expanded=True) as exsting_data_st:

        try:
            conn = get_pg_connection()
        except Exception as e:
            st.error(f"❌ Could not connect to the database. Check that `host`, `name`, `user`, `password`, and `port` are set in Streamlit secrets.\n\n`{e}`")
            exsting_data_st.update(label="Checking existing data", state="error", expanded=True)
            st.stop()
        conn.autocommit = False
        chk_res = find_existing_data(conn, SHEET_CONFIG, branch_id, selected_period)

        if chk_res["status"] != "ok":
            st.write(chk_res["msg"])
            if mode != "Overwrite":
                st.write("Process cancelled because data already exists.")
                exsting_data_st.update(label="Checking existing data", state="error", expanded=True)
                conn.close()
                st.stop()

            st.write("Existing data will be replaced.")
            with st.status("Deleting existing data...", expanded=True) as del_st:
                del_res = delete_existing_data(conn, SHEET_CONFIG, branch_id, selected_period, True)
                if del_res["status"] != "ok":
                    st.write(del_res["msg"])
                    del_st.update(label="Deleting existing data", state="error", expanded=True)
                    conn.close()
                    st.stop()
                st.write(del_res["msg"])
                del_st.update(label="Deleting existing data", state="complete", expanded=True)
        else:
            st.write(chk_res["msg"])

        exsting_data_st.update(label="Checking existing data", state="complete", expanded=True)


    with st.status("Writing to Database...", expanded=True) as write_st:
        try:
            load_res = push_sheets(cleaned, SHEET_CONFIG, conn, True)
            if load_res["status"] != "ok":
                st.write(load_res["message"])
                write_st.update(label="Writing to Database", state="error", expanded=True)
                st.stop()

            st.write(load_res["message"])
            write_st.update(label="Writing to Database", state="complete", expanded=True)
        finally:
            conn.close()

    st.success(f"Successfully loaded data to database.")