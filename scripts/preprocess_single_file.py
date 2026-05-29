import importlib                                    
import streamlit as st
from pathlib import Path
from etl.strip_all import strip_all
from etl.special_characters import special_char
from etl.saver import save_cleaned_data

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

st.title("Preprocess Single File")
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
        file_name = preprocessor.replace("_", " ")
        files = [p for p in base_folder.iterdir() if p.is_file()]
        if len(files) > 1:
            st.error("Folder must contain exactly one file.")
            filter_st.update(state="error",expanded=True)
            st.stop()
        file = files[0]
        data = preprocess_func(file)

        data_dict = strip_all({file_name: data})
        data_dict = special_char(data_dict)

        save_cleaned_data(data_dict, base_folder)
        st.write('Done')
        filter_st.update(state="complete",expanded=True)

    st.success("✅ Data Saved")