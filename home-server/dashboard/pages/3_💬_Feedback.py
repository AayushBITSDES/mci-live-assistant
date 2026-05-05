"""Feedback review: last 20 wrong-nudge entries from the feedback log."""
from __future__ import annotations

import streamlit as st

from app.config import settings
from dashboard.state import read_recent_feedback

st.set_page_config(page_title="Feedback", page_icon="💬", layout="wide")
st.title("💬 Feedback Review")
st.caption("Every time Shanta says a nudge was wrong, we capture full context here.")

settings.ensure_dirs()

if st.button("🔄 Refresh"):
    st.rerun()

entries = read_recent_feedback(settings.feedback_dir, limit=20)

if not entries:
    st.info(
        "No wrong-nudge entries yet. They appear here when the user says "
        "**'wrong'** or **'that's not right'** during a nudge."
    )
else:
    st.subheader(f"Last {len(entries)} flagged nudges (newest first)")
    for i, entry in enumerate(entries):
        ts = entry.get("timestamp", "")
        nudge = entry.get("nudge_sentence", "—")
        reason = entry.get("user_reason") or "(no reason given)"
        scenario = entry.get("scenario", "—")

        with st.expander(f"`{ts}` — {scenario} — {nudge[:60]}"):
            st.markdown(f"**Nudge:** {nudge}")
            st.markdown(f"**User said:** _{reason}_")
            if "context_summary" in entry:
                st.markdown("**Context at the time:**")
                st.code(entry["context_summary"], language="text")
            if "detected_objects" in entry:
                st.markdown(f"**Detections:** `{entry['detected_objects']}`")
