import importlib                                    
import streamlit as st
from pathlib import Path
from ml.ops.concat import concat_files
from ml.validators import validate_omega_name
from ml.modeling import (
    match_monthly_rate,
    add_metadata,
    adjust_configs,
    convert_sheet_names_in_dict,
    add_old_data
)
from etl.saver import save_cleaned_data
from etl.strip_all import strip_all
from etl.special_characters import special_char
from supa.config import SHEET_CONFIG
from supa.streamlit_functions import get_client_list
from supa.db import (
    init_supabase,
    get_branch_id,
    get_omega_currency,
)
from supa.modeling import (
    normalize_all_dataframes,
    normalize_string_columns,
    clean_numeric_values
)

if "ptdb_supabase_client" not in st.session_state:
    st.session_state.ptdb_supabase_client = init_supabase()
supabase = st.session_state.ptdb_supabase_client

_ROOT = Path(__file__).resolve().parent.parent
_PREPROC_ROOT = _ROOT / "src" / "ml" / "preprocessors"


def _list_preprocessors(source: str) -> list[str]:
    folder = _PREPROC_ROOT / source
    if not folder.is_dir():
        return []
    return sorted(
        p.stem
        for p in folder.glob("*.py")
        if p.stem != "__init__"
    )

st.set_page_config(
    page_title="Concat",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.title("Concat")
st.markdown("---")

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    folder_input = st.text_input("📁 Target Folder Path", placeholder="C:/Path/To/Folder")
with col2:
    client_options = get_client_list(supabase)
    selected_client = st.selectbox("Select Branch", options=client_options, key="ptdb_client")
with col3:
    source = st.selectbox("🔀 Source", options=["cloud", "local"], index=0)
with col4:
    preprocessor_options = _list_preprocessors(source)
    if preprocessor_options:
        preprocessor = st.selectbox(
            "🔧 Preprocessor",
            options=preprocessor_options,
            format_func=lambda name: name.replace("_", " "),
            key=f"concat_preprocessor_{source}",
        )
        preprocess_func = importlib.import_module(
            f"ml.preprocessors.{source}.{preprocessor}"
        ).preprocess
    else:
        st.selectbox(
            "🔧 Preprocessor",
            options=["No preprocessors available"],
            disabled=True,
        )
        preprocessor = None
        preprocess_func = None
with col5:
    push = st.selectbox("💾 push", options=["d'ont push", "push"], index=0)



if st.button("▶ Run", type="primary", use_container_width=True):

    with st.status("Filtering...", expanded=True) as filter_st:
        base_folder = Path(folder_input).resolve()
        result = concat_files(base_folder, preprocessing_func=preprocess_func)
        if result['status'] != 'ok':
            st.write(result['msg'])
            filter_st.update(state="error",expanded=True)
            st.stop()
        
        prep_name = result['prep_name']
        final_name = result['final_name']
        destination = result['destination']
        data = result['data']

        data = strip_all(data)
        data = special_char(data)
        save_cleaned_data(data, destination,final_name)
        st.write('Data is Filtered')

        filter_st.update(state="complete",expanded=True)

    if push == 'push': #######################################################    if push

        with st.status("Validating...", expanded=True) as validating_st:
            qr_res = get_branch_id(selected_client, supabase)
            branch_id = qr_res["branch_id"]

            name_val = validate_omega_name(data, branch_id, supabase)
            if name_val['status'] != 'ok':
                st.write(name_val['msg'])
                validating_st.update(label="Validating", state="error", expanded=True)
                st.stop()
            st.write(name_val['msg'])


        with st.status("Formatitng...", expanded=True) as form_st:
            SHEET_CONFIG = adjust_configs(SHEET_CONFIG) ######################    "Sales" and "Sales. Cat."

            data = match_monthly_rate(data)
            meta_res = add_metadata(data, branch_id, supabase)
            if meta_res['status'] != 'ok':
                st.write(meta_res["message"])
                form_st.update(label="Formatting", state="error", expanded=True)
                st.stop()

            data = normalize_all_dataframes(data)
            data = convert_sheet_names_in_dict(data)

            norm_res = normalize_string_columns(data)
            if norm_res["status"] != "ok":
                st.write(norm_res["message"])
                form_st.update(label="Formatting", state="error", expanded=True)
                st.stop()
            data = norm_res["data"]
            data = add_old_data(data)
            data = clean_numeric_values(data)

            save_cleaned_data(data, 'C:/Users/Gianni Habchi/Desktop', 'concat data la halla2.xlsx')

            form_st.update(label="Formatting Data", state="complete", expanded=True)










    st.success("✅ Done")