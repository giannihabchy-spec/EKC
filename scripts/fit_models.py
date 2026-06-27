from ml.forecasting.fit_all import fit_all
from ml.functions.eval import display_results
from ml.functions.results_io import delete_existing_results, save_results, results_to_df_dict, get_fitted
from ml.regestry import MODEL_REGISTRY, MULTI_REGISTRY
from ml.validators import update_supa_results
from ml.config import sheet_config
import streamlit as st
from supa.streamlit_functions import get_client_list_for_daily_sales
from supa.db import get_branch_id, get_pg_connection
from supa.loaders import push_sheets


st.set_page_config(
    page_title="Fit Models",
    layout="wide",
    initial_sidebar_state="collapsed"
)
st.title("Fit Models")
st.markdown("---")

single_models = list(MODEL_REGISTRY.keys())
multi_models = list(MULTI_REGISTRY.keys())

if single_models or multi_models:
    col_s, col_m = st.columns(2)
    with col_s:
        st.markdown("**Per-category models**")
        if single_models:
            for m in single_models:
                st.write(f"- `{m}`")
        else:
            st.caption("None")
    with col_m:
        st.markdown("**Multi-category models**")
        if multi_models:
            for m in multi_models:
                st.write(f"- `{m}`")
        else:
            st.caption("None")
else:
    st.warning("No models registered.")

st.markdown("---")

col1, col2, col3 = st.columns(3)
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
with col3:
    freq = st.selectbox("Frequency", options=["Daily", "Weekly"], index=0)

if st.button("▶ Run", type="primary", use_container_width=True):

    with st.status("Fitting...", expanded=True) as fit_st:
        qr_res = get_branch_id(selected_client)
        branch_id = qr_res["branch_id"]

        freq = "D" if freq == "Daily" else "W"
        results = fit_all(branch_id, threshold, freq)
        summary = get_fitted(results)
        categories = summary.get('categories')
        models = summary.get('models')

        data = {'results': results}

        display_results(results)

        fit_st.update(state="complete",expanded=True)


    with st.status("Checking existing data...", expanded=True) as delete_st:
        delete_result = delete_existing_results(branch_id, data, freq)
        if delete_result['status'] != 'ok':
            st.write(delete_result['msg'])
            delete_st.update(label="Checking existing data", state="error", expanded=True)
            st.stop()
        st.write(delete_result['msg'])
        delete_st.update(label="Deleting existing data", state="complete", expanded=True)


    with st.status("Saving results...", expanded=True) as write_st:
        load_res = save_results(data)
        if load_res["status"] != "ok":
            st.write(load_res["message"])
            write_st.update(label="Saving results", state="error", expanded=True)
            st.stop()
        st.write(load_res["message"])
        write_st.update(label="Results saved", state="complete", expanded=True)


    with st.status("Updating Database...", expanded=True) as supa_st:
        results_dict = results_to_df_dict(
            [branch_id],
            categories,
            [freq],
            models
        )
        supa_result = update_supa_results(results_dict, sheet_config)
        if supa_result['status'] == 'error':
            st.write(supa_result['msg'])
            supa_st.update(label="Updating Database", state="error", expanded=True)
            st.stop()
        else:
            st.write(supa_result['msg'])
            conn = get_pg_connection()
            try:
                load_res = push_sheets(results_dict, sheet_config, conn, True)
                if load_res["status"] != "ok":
                    st.write(load_res["message"])
                    supa_st.update(label="Writing to Database", state="error", expanded=True)
                    st.stop()

                st.write(load_res["message"])
                supa_st.update(label="Writing to Database", state="complete", expanded=True)
            finally:
                conn.close()


        supa_st.update(label="Updating Database", state="complete", expanded=True)



    st.success("✅ Done")