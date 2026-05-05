"""Streamlit dashboard entrypoint.

Run with:
    streamlit run dashboard/app.py --server.port 8501

Pages live in `dashboard/pages/` and are auto-discovered by Streamlit
(numeric prefix controls sidebar order).
"""
from __future__ import annotations

import streamlit as st

from app.config import settings
from dashboard.persistence import load_onboarding, load_runtime_settings


def main() -> None:
    st.set_page_config(
        page_title="MCI Home Server",
        page_icon="🪞",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("🪞 MCI Home Server — Dashboard")
    st.caption("Always-on home brain for the contextual AI prototype")

    # Top-level status panel
    onboarding = load_onboarding(settings.onboarding_file)
    runtime = load_runtime_settings(settings.storage_root / "runtime_settings.json")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("User", onboarding.user_name)
    col2.metric("LLM provider", runtime.active_llm_provider)
    col3.metric("FPS target", runtime.target_fps)
    col4.metric("Fresh start", "ON" if runtime.fresh_start else "OFF")

    st.markdown("---")
    st.markdown(
        """
        ### Pages
        - **📷 Faces** — drag-drop photos to register relatives, re-embed
        - **📊 Live** — recent events, open risk windows, current activity
        - **💬 Feedback** — review wrong nudges Shanta has flagged
        - **👤 Onboarding** — persona setup, emergency contacts, age/weight
        - **⚙️ Settings** — LLM provider, FPS, thresholds, fresh start

        Use the sidebar to navigate.
        """
    )

    st.info(
        "💡 The dashboard reads from `storage/events/` and `storage/feedback/` "
        "files written by the FastAPI server. You can run them in separate terminals."
    )


if __name__ == "__main__":
    main()
