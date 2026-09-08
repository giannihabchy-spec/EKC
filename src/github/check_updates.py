import subprocess
import streamlit as st


def require_latest_code():
    try:
        # Get the latest information from GitHub
        subprocess.run(
            ["git", "fetch", "origin"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # Check how many commits local master is behind GitHub
        behind = subprocess.check_output(
            ["git", "rev-list", "--count", "HEAD..origin/master"],
            text=True,
        ).strip()

        if int(behind) > 0:
            st.error(
                f"🚨 Your local version is {behind} commit(s) behind GitHub."
            )

            st.warning(
                "You must pull the latest changes before using EKC Tools."
            )

            st.stop()

    except Exception as e:
        st.error(f"Could not check for updates: {e}")
        st.stop()