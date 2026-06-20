from ml.forecasting.fit_all import fit_all
from ml.config import sheet_config
from ml.validators import delete_all_for_branch
from ml.functions.eval import display_results
import streamlit as st
from supa.streamlit_functions import get_client_list_for_daily_sales
from supa.db import get_branch_id
from ml.validators import delete_all_for_branch
from supa.db import get_pg_connection
from supa.loaders import push_sheets


st.set_page_config(
    page_title="Push Results",
    layout="wide",
    initial_sidebar_state="collapsed"
)
st.title("Push Results")
st.markdown("---")



col1, col2 = st.columns(2)
with col1:
    client_options = get_client_list_for_daily_sales()
    selected_client = st.selectbox("Select Branch", options=client_options, key="ptdb_client")
with col2:
    threshold = st.number_input(
        "Threshold",
        min_value=0.0,
        value=0.1,
        step=0.01,
        format="%.2f",
        key="threshold"
    )

if st.button("▶ Run", type="primary", use_container_width=True):

    with st.status("Fitting...", expanded=True) as fit_st:
        qr_res = get_branch_id(selected_client)
        branch_id = qr_res["branch_id"]

        results = fit_all(branch_id, threshold)
        data = {'results': results}

        display_results(results)

        fit_st.update(state="complete",expanded=True)


    with st.status("Checking existing data...", expanded=True) as delete_st:
        delete_result = delete_all_for_branch(branch_id, data, sheet_config)
        if delete_result['status'] != 'ok':
            st.write(delete_result['msg'])
            delete_st.update(label="Checking existing data", state="error", expanded=True)
            st.stop()
        st.write(delete_result['msg'])
        delete_st.update(label="Deleting existing data", state="complete", expanded=True)


    with st.status("Writing to Database...", expanded=True) as write_st:
        conn = get_pg_connection()
        try:
            load_res = push_sheets(data, sheet_config, conn)
            if load_res["status"] != "ok":
                st.write(load_res["message"])
                write_st.update(label="Writing to Database", state="error", expanded=True)
                st.stop()

            st.write(load_res["message"])
            write_st.update(label="Writing to Database", state="complete", expanded=True)
        finally:
            conn.close()

    st.success("✅ Done")