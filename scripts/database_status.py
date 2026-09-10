import streamlit as st
from supa.db import _ensure_supa_env_from_secrets

st.set_page_config(
    page_title="Database Status",
    layout="wide",
    initial_sidebar_state="collapsed"
)

_ensure_supa_env_from_secrets()

from supa.db import (
    init_supabase,
    get_pushed_outlets
)
from supa.modeling import readable_dates
from supa.streamlit_functions import get_period_options


if "ptdb_supabase_client" not in st.session_state:
    st.session_state.ptdb_supabase_client = init_supabase()
supabase = st.session_state.ptdb_supabase_client


st.title("Database Status")
st.markdown("---")

col1 = st.columns(1)[0]
with col1:
    period_options = get_period_options()
    selected_period = st.selectbox("Select Period", options=period_options, index = 1, key="ptdb_period")


if st.button("▶ Run", type="primary", use_container_width=True):


#     with st.status("Checking Database", expanded=True) as chk_st:

#         data = get_pushed_outlets(selected_period)
#         pending = data['pending']
#         pushed = data['pushed']

#         if pushed:
#             for outlet in pushed:
#                 st.write(outlet)
#         else:
#             st.write(f"No pushed reports for {readable_dates(selected_period)}")

#         chk_st.update(label="Pushed Outlets", state="complete", expanded=True)

#     with st.status("Pending Outlets", expanded=True) as pen_st:

#         if pending:
#             for outlet in pending:
#                 st.write(outlet)
#         else:
#             st.write(f"All reports are pushed for {readable_dates(selected_period)}")

#         pen_st.update(label="Pending Outlets", state="complete", expanded=True)

#############################################################################################################################

    data = get_pushed_outlets(selected_period)
    pending = data["pending"]
    pushed = data["pushed"]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader(f"✅ Pushed ({len(pushed)})")

        with st.container(border=True):
            if pushed:
                for outlet in pushed:
                    st.write(f"✓ {outlet}")
            else:
                st.caption(
                    f"No pushed reports for {readable_dates(selected_period)}"
                )

    with col2:
        st.subheader(f"⏳ Pending ({len(pending)})")

        with st.container(border=True):
            if pending:
                for outlet in pending:
                    st.write(f"○ {outlet}")
            else:
                st.success(
                    f"All reports are pushed for {readable_dates(selected_period)}"
                )

