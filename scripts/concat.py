import importlib                                    
import streamlit as st
from pathlib import Path
from ml.ops.concat import concat_files
from etl.saver import save_cleaned_data
from etl.strip_all import strip_all
from etl.special_characters import special_char

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

col1, col2, col3 = st.columns(3)
with col1:
    folder_input = st.text_input("📁 Target Folder Path", placeholder="C:/Path/To/Folder")
with col2:
    source = st.selectbox("🔀 Source", options=["cloud", "local"], index=0)
with col3:
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

if st.button("▶ Run", type="primary", use_container_width=True):

    with st.status("Filtering...", expanded=True) as filter_st:
        base_folder = Path(folder_input).resolve()
        result = concat_files(base_folder, preprocessing_func=preprocess_func)
        if result['status'] != 'ok':
            st.write(result['msg'])
            filter_st.update(state="error",expanded=True)
            st.stop()
        
        final_name = result['final_name']
        destination = result['destination']
        data = result['data']

        data = strip_all(data)
        data = special_char(data)
        save_cleaned_data(data, destination,final_name)
        st.write(result['msg'])

        filter_st.update(state="complete",expanded=True)

    st.success("✅ Done")