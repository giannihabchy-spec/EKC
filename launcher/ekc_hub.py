import streamlit as st
from github.check_updates import require_latest_code

st.set_page_config(
    page_title="EKC Tools",
    layout="wide",
    initial_sidebar_state="expanded",
)

require_latest_code()

st.title("EKC Tools")
st.markdown("Use the **sidebar** to switch between apps.")
st.markdown("---")     