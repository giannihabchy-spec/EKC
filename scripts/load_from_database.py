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
from etl.locate_cols import get_excel_cols
from etl.validators import (
    get_missing_columns,
    check_sheets_exist
)
from etl.config import JOBS_CLOUD
from etl.clear_sheets import clear_sheets
from etl.writer import write_master
from etl.reset_view import reset_workbook_view


from supa.db import (
    init_supabase,
    get_branch_id,
    check_data_coverage
)
from supa.loaders import(
    extract_info,
    load_logs,
)
from supa.streamlit_functions import get_client_list, get_period_options
from supa.modeling import (
    normalize_all_dataframes,
    convert_sheet_names_to_autocalc,
    convert_sheet_names_to_file_names,
    readable_dates
)
from supa.validators import (
    validate_client_name,
    validate_selected_date,
)


if "ptdb_supabase_client" not in st.session_state:
    st.session_state.ptdb_supabase_client = init_supabase()
supabase = st.session_state.ptdb_supabase_client


st.title("Load From Database")
st.markdown("---")

col1, col2, col3, col4 = st.columns(4)
with col1:
    folder_input = st.text_input("Path to the folder contining 'Auto Calc.xlsx'", placeholder="C:/Path/To/Folder")
    destination = Path(folder_input).resolve() if folder_input.strip() else None
    master_path = destination / "Auto Calc.xlsx" if destination else None
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


    if not folder_input or not selected_client or not selected_period or not data_choice:
        st.error("Please provide a file, a client, a date and the data to load.")
        st.stop()
    if not master_path.is_file():
        st.error("No 'Auto Calc.xlsx' file found in the folder.")
        st.stop()


    with st.status("Extracting Info...", expanded=True) as extract_st:
        file_date, file_client_name, currency, rate, info = extract_info(master_path)
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
        date_str = readable_dates(selected_period)
        save_cleaned_data(data, destination, f"{selected_client} logs {date_str}.xlsx")
        st.write('Data Saved')

        load_st.update(label="Loading Data", state="complete", expanded=True)


    with st.status("Opening Workbook...", expanded=True) as status_ow:
        reset_workbook_view(master_path)
        st.write('Completed')
        status_ow.update(label='Opening Workbook',state="complete", expanded=True)


    with st.status("Validating Workbook...", expanded=True) as valw_st:

        data = convert_sheet_names_to_autocalc(data)
        # data = normalize_all_dataframes(data)
        missing_sheets = check_sheets_exist(master_path, JOBS_CLOUD)
        if missing_sheets['status'] != 'ok':
            st.write(missing_sheets['msg'])
            valw_st.update(label='Validating Workbook',state="error", expanded=True)
            st.stop()
        st.write(missing_sheets['msg'])
        missing_cols = get_missing_columns(master_path, JOBS_CLOUD)
        if missing_cols['status'] != 'ok':
            st.write(missing_cols['msg'])
            valw_st.update(label='Validating Workbook',state="error", expanded=True)
            st.stop()
        st.write(missing_cols['msg'])
        loc_res = get_excel_cols(master_path, JOBS_CLOUD)
        if loc_res['status'] != 'ok':
            st.write(loc_res['msg'])
            valw_st.update(label='Locating columns',state="error", expanded=True)
            st.stop() 
        jobs = loc_res['result']
        data = convert_sheet_names_to_file_names(data)
        st.write(loc_res['msg'])

        valw_st.update(label="Validating Workbook", state="complete", expanded=True)


    with st.status("Clearing...", expanded=True) as status_clear:
        clear_sheets(str(master_path), jobs=jobs, cleaned=data, log_func=st.write)
        status_clear.update(label="Clearing", state="complete", expanded=True)

    with st.status("Writing...", expanded=True) as status_write:
        write_master(str(master_path), data, jobs, suppress_warnings=True, log_func=st.write)
        status_write.update(label="Writing", state="complete", expanded=True)
        st.write("Loaded all available data")

    st.success("✅ Successfully updated 'Auto Calc.xlsx'")